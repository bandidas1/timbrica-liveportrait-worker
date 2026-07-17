"""Bake the models into the image at build time (cold start = disk read, not download).

- LivePortrait ONNX (MIT, Kuaishou weights, exported by warmshao/FasterLivePortrait).
  The InsightFace-lineage files (retinaface_det_static / face_2dpose_106_static) are
  deliberately NOT downloaded — face analysis is MediaPipe (Apache-2.0).
- Restore pre-pass: RestoreFormer++ (Apache-2.0, our own fp16 conversion, HF mirror)
  + Real-ESRGAN x4 (BSD-3) from the timbrica RU origin (unmetered, ACAO:*).
"""

import os
import urllib.request

from huggingface_hub import hf_hub_download

import shutil

# LivePortrait torch modules (MIT, Kuaishou weights) — the animation engine.
TORCH_DEST = "/app/checkpoints/liveportrait_torch"
os.makedirs(TORCH_DEST, exist_ok=True)

TORCH_FILES = [
    ("liveportrait/base_models/appearance_feature_extractor.pth", "appearance_feature_extractor.pth"),
    ("liveportrait/base_models/motion_extractor.pth", "motion_extractor.pth"),
    ("liveportrait/base_models/warping_module.pth", "warping_module.pth"),
    ("liveportrait/base_models/spade_generator.pth", "spade_generator.pth"),
    ("liveportrait/retargeting_models/stitching_retargeting_module.pth", "stitching_retargeting_module.pth"),
]

for src, name in TORCH_FILES:
    p = hf_hub_download("KlingTeam/LivePortrait", src)
    dst = os.path.join(TORCH_DEST, name)
    shutil.copy(p, dst)
    print("baked", dst, os.path.getsize(dst))

# LivePortrait's own 203-pt landmark refiner (MIT) — plain 4-D ONNX, runs per request
# (not per frame) on onnxruntime.
ONNX_DEST = "/app/checkpoints/liveportrait_onnx"
os.makedirs(ONNX_DEST, exist_ok=True)
p = hf_hub_download("warmshao/FasterLivePortrait", "liveportrait_onnx/landmark.onnx")
shutil.copy(p, os.path.join(ONNX_DEST, "landmark.onnx"))
print("baked landmark", os.path.getsize(os.path.join(ONNX_DEST, "landmark.onnx")))

RESTORE_DEST = "/app/checkpoints/restore"
os.makedirs(RESTORE_DEST, exist_ok=True)

rf = hf_hub_download("Faridzar/restoreformer-mirror", "restoreformer-pp-fp16.onnx")
shutil.copy(rf, os.path.join(RESTORE_DEST, "restoreformer-pp-fp16.onnx"))
print("baked restoreformer", os.path.getsize(os.path.join(RESTORE_DEST, "restoreformer-pp-fp16.onnx")))

ESRGAN_URL = "https://timbrica.com/vendor/onnx/realesrgan-x4-fp16.onnx"
esr = os.path.join(RESTORE_DEST, "realesrgan-x4-fp16.onnx")
urllib.request.urlretrieve(ESRGAN_URL, esr)
print("baked realesrgan", os.path.getsize(esr))
assert os.path.getsize(esr) > 10_000_000, "realesrgan download truncated"
