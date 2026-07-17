# Timbrica: PyTorch predictor for FasterLivePortrait's model wrappers.
#
# WHY THIS EXISTS: stock onnxruntime-gpu cannot run LivePortrait's warping graph on
# CUDA — the 5-D (volumetric) GridSample node has no CUDA kernel ("Only 4-D tensor is
# supported", verified live on ORT 1.22 / RTX 4090, 2026-07-17), and CPU fallback is
# ~15 s/frame. FLP's answer is TensorRT + a custom plugin or a self-compiled ORT fork;
# ours is simpler: run the ORIGINAL Kuaishou torch modules (MIT, KlingTeam/LivePortrait
# weights) behind FLP's predictor interface — torch's grid_sample supports 5-D on CUDA
# natively. Same math, no custom kernels, no per-GPU engine builds.
#
# The wrappers (warping_spade_model.py etc.) feed raw numpy and post-process outputs
# themselves (softmax→degree, reshape) — the ONNX graphs are bare module forwards, so
# a bare module forward is a drop-in replacement. I/O contracts mirrored 1:1:
#
#   appearance_feature_extractor: (img[1,3,256,256] f32 0..1)      -> [feature_3d[1,32,16,64,64]]
#   motion_extractor:             (img[1,3,256,256] f32 0..1)      -> [pitch(1,66) yaw roll t exp scale kp]  (RAW logits)
#   warping_spade:                (feature_3d, kp_driving, kp_source) -> [img[1,3,512,512] f32 0..1]
#   stitching / _eye / _lip:      (feat[1,N])                       -> [delta[1,M]]
#
# Install BEFORE importing any flp pipeline module:
#     import torch_predictor; torch_predictor.install()

import os
import sys

import numpy as np
import torch
import yaml

LP_MODULES_DIR = os.environ.get("LP_MODULES_DIR", "/app")          # parent of lp_modules/
LP_WEIGHTS_DIR = os.environ.get("LP_WEIGHTS_DIR", "/app/checkpoints/liveportrait_torch")
LP_MODELS_YAML = os.environ.get("LP_MODELS_YAML", "/app/config/lp_models.yaml")

_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
_HALF = torch.cuda.is_available()  # fp16 autocast on GPU (upstream default too)


def _strip_ddp(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


class TorchPredictor:
    """Drop-in for OnnxRuntimePredictor: .predict(*np) -> list[np]."""

    def __init__(self, **kwargs):
        sys.path.insert(0, LP_MODULES_DIR)
        from lp_modules.appearance_feature_extractor import AppearanceFeatureExtractor
        from lp_modules.motion_extractor import MotionExtractor
        from lp_modules.spade_generator import SPADEDecoder
        from lp_modules.stitching_retargeting_network import StitchingRetargetingNetwork
        from lp_modules.warping_network import WarpingNetwork

        with open(LP_MODELS_YAML) as f:
            params = yaml.safe_load(f)["model_params"]

        self.kind = kwargs["torch_model"]
        w = lambda n: os.path.join(LP_WEIGHTS_DIR, n)

        if self.kind == "appearance_feature_extractor":
            self.net = AppearanceFeatureExtractor(**params["appearance_feature_extractor_params"])
            self.net.load_state_dict(_strip_ddp(torch.load(w("appearance_feature_extractor.pth"), map_location="cpu")))
        elif self.kind == "motion_extractor":
            self.net = MotionExtractor(**params["motion_extractor_params"])
            self.net.load_state_dict(_strip_ddp(torch.load(w("motion_extractor.pth"), map_location="cpu")))
        elif self.kind == "warping_spade":
            self.warp = WarpingNetwork(**params["warping_module_params"])
            self.warp.load_state_dict(_strip_ddp(torch.load(w("warping_module.pth"), map_location="cpu")))
            self.spade = SPADEDecoder(**params["spade_generator_params"])
            self.spade.load_state_dict(_strip_ddp(torch.load(w("spade_generator.pth"), map_location="cpu")))
            self.warp.to(_DEVICE).eval()
            self.spade.to(_DEVICE).eval()
            self.net = None
        elif self.kind in ("stitching", "stitching_eye", "stitching_lip"):
            ckpt = torch.load(w("stitching_retargeting_module.pth"), map_location="cpu")
            sub = {"stitching": ("stitching", "retarget_shoulder"),
                   "stitching_eye": ("eye", "retarget_eye"),
                   "stitching_lip": ("lip", "retarget_mouth")}[self.kind]
            self.net = StitchingRetargetingNetwork(**params["stitching_retargeting_module_params"][sub[0]])
            self.net.load_state_dict(_strip_ddp(ckpt[sub[1]]))
        else:
            raise ValueError(f"unknown torch_model: {self.kind}")

        if self.net is not None:
            self.net.to(_DEVICE).eval()

    # BaseModel probes these; static answers are fine (nothing downstream uses them).
    def input_spec(self):
        return []

    def output_spec(self):
        return []

    @torch.no_grad()
    def predict(self, *data):
        t = [torch.from_numpy(np.ascontiguousarray(d)).to(_DEVICE).float() for d in data]
        with torch.autocast(device_type="cuda", enabled=_HALF):
            if self.kind == "warping_spade":
                feature_3d, kp_driving, kp_source = t
                ret = self.warp(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
                out = self.spade(feature=ret["out"])
                return [out.float().cpu().numpy()]
            if self.kind == "motion_extractor":
                kp_info = self.net(t[0])
                order = ["pitch", "yaw", "roll", "t", "exp", "scale", "kp"]
                return [kp_info[k].float().cpu().numpy() for k in order]
            out = self.net(t[0])
        if isinstance(out, dict):  # defensive; none of ours return dicts here
            out = out["out"]
        return [out.float().cpu().numpy()]

    def __del__(self):
        pass


def install():
    """Route predict_type 'torch' to TorchPredictor.

    base_model.py does `from .predictor import get_predictor` — importing the
    src.models package binds the ORIGINAL symbol into base_model before we can
    patch the predictor module. So patch the name in BOTH modules.
    """
    import src.models.base_model as flp_base
    import src.models.predictor as flp_predictor

    orig = flp_predictor.get_predictor

    def get_predictor(**kwargs):
        if kwargs.get("predict_type") == "torch":
            return TorchPredictor(**kwargs)
        return orig(**kwargs)

    flp_predictor.get_predictor = get_predictor
    flp_base.get_predictor = get_predictor
