"""OFFLINE preset builder — driving videos / synthetic pose curves → presets/*.pkl.

A preset is a de-identified motion template (the exact dict FasterLivePortrait's
run.py saves): {n_frames, output_fps, motion[], c_eyes_lst[], c_lip_lst[], loop}.
`motion[i]` = {pitch,yaw,roll,t,exp,scale,kp,R} float32 — pose angles + expression
coefficients only, NO pixels and NO identity. That is the whole licensing story of
presets: whatever footage they were extracted from, nothing of the person ships.

Run on a GPU box with the worker image's deps (or inside the image):

  extract    — video → template (front-facing head-and-shoulders footage)
  synth      — take frame 0 of a base template, synthesize a smooth pose sweep
               (nod = pitch, look-around = yaw, tilt = roll); fully deterministic
  loopify    — blend the tail back into frame 0 so `tile` looping is seamless
  preview    — render a template against a source image (visual QA)
  info       — print template stats

Examples:
  python build_presets.py extract --video smile.mp4 --name smile --max-frames 125 --loopify
  python build_presets.py synth --base presets/smile.pkl --name nod --pose pitch --amp 8 --cycles 2
  python build_presets.py preview --preset presets/nod.pkl --image photo.jpg --out prev.mp4
"""

import argparse
import copy
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, "/app/flp")
sys.path.insert(0, "/app")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flp"))
sys.path.insert(0, os.path.dirname(__file__))

import torch_predictor

torch_predictor.install()

from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.crop import crop_image_by_bbox, parse_bbox_from_landmark
from src.utils.utils import calc_eye_close_ratio, calc_lip_close_ratio, get_rotation_matrix

CFG_PATH = os.environ.get("PA_CONFIG", "/app/config/infer.yaml")
PRESETS_DIR = os.path.join(os.path.dirname(__file__), "presets")


def _pipe():
    cfg = OmegaConf.load(CFG_PATH)
    return FasterLivePortraitPipeline(cfg=cfg, is_animal=False)


def _save(name, tpl):
    os.makedirs(PRESETS_DIR, exist_ok=True)
    path = os.path.join(PRESETS_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(tpl, f)
    print(f"saved {path}: {tpl['n_frames']} frames @ {tpl['output_fps']} fps, loop={tpl.get('loop')}")


def _f32_motion(m):
    return {k: np.asarray(v, dtype=np.float32) for k, v in m.items()}


def cmd_extract(args):
    """Driving video → motion template. Replicates pipeline.run()'s driving branch
    (crop-driving-video path) without needing a source image."""
    pipe = _pipe()
    fa = pipe.model_dict["face_analysis"]
    lm = pipe.model_dict["landmark"]
    me = pipe.model_dict["motion_extractor"]
    cp = pipe.cfg.crop_params

    vcap = cv2.VideoCapture(args.video)
    fps = int(round(vcap.get(cv2.CAP_PROP_FPS))) or 25
    motion, c_eyes, c_lip = [], [], []
    lmk_pre = None
    n = 0
    while True:
        ret, frame = vcap.read()
        if not ret or (args.max_frames and n >= args.max_frames):
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if lmk_pre is None:
            faces = fa.predict(frame)
            if not faces:
                continue
            lmk = lm.predict(img_rgb, faces[0])
        else:
            lmk = lm.predict(img_rgb, lmk_pre)
        lmk_pre = lmk.copy()

        ret_bbox = parse_bbox_from_landmark(
            lmk, scale=cp.dri_scale, vx_ratio_crop_video=cp.dri_vx_ratio,
            vy_ratio=cp.dri_vy_ratio)["bbox"]
        bbox = [ret_bbox[0, 0], ret_bbox[0, 1], ret_bbox[2, 0], ret_bbox[2, 1]]
        ret_dct = crop_image_by_bbox(img_rgb, bbox, lmk=lmk, dsize=512,
                                     flag_rot=False, borderValue=(0, 0, 0))
        lmk_crop = ret_dct["lmk_crop"]
        img_crop = cv2.resize(ret_dct["img_crop"], (256, 256))

        pitch, yaw, roll, t, exp, scale, kp = me.predict(img_crop)
        R = get_rotation_matrix(pitch, yaw, roll)
        motion.append(_f32_motion(
            {"pitch": pitch, "yaw": yaw, "roll": roll, "t": t, "exp": exp,
             "scale": scale, "kp": kp, "R": R}))
        c_eyes.append(calc_eye_close_ratio(lmk_crop[None]).astype(np.float32))
        c_lip.append(calc_lip_close_ratio(lmk_crop[None]).astype(np.float32))
        n += 1
    vcap.release()
    if n < 10:
        sys.exit(f"only {n} usable frames — bad driving video")

    tpl = {"n_frames": n, "output_fps": args.fps or fps, "motion": motion,
           "c_eyes_lst": c_eyes, "c_lip_lst": c_lip, "loop": "pingpong"}
    if args.trim:
        a, b = [int(v) for v in args.trim.split(":")]
        tpl = _trim(tpl, a, b)
    if args.dampen != 1.0:
        tpl = _dampen(tpl, args.dampen)
    if args.loopify:
        tpl = _loopify(tpl, args.blend)
    _save(args.name, tpl)


def _dampen(tpl, k):
    """Scale every frame's deltas RELATIVE TO FRAME 0 by k (bakes a softer motion
    into the template — no runtime knob needed). k<1 softens, k>1 exaggerates."""
    m0 = tpl["motion"][0]
    for m in tpl["motion"][1:]:
        for key in ("pitch", "yaw", "roll", "t", "exp", "scale"):
            m[key] = (m0[key] + k * (m[key] - m0[key])).astype(np.float32)
        m["R"] = get_rotation_matrix(m["pitch"], m["yaw"], m["roll"]).astype(np.float32)
    return tpl


def _trim(tpl, a, b):
    for key in ("motion", "c_eyes_lst", "c_lip_lst"):
        tpl[key] = tpl[key][a:b]
    tpl["n_frames"] = len(tpl["motion"])
    return tpl


def _loopify(tpl, blend=12):
    """Blend the last `blend` frames back toward frame 0 → seamless `tile` loop."""
    n = tpl["n_frames"]
    blend = min(blend, n // 3)
    m0 = tpl["motion"][0]
    for i in range(blend):
        k = n - blend + i
        a = (i + 1) / blend
        mk = tpl["motion"][k]
        for key in ("pitch", "yaw", "roll", "t", "exp", "scale"):
            mk[key] = ((1 - a) * mk[key] + a * m0[key]).astype(np.float32)
        mk["R"] = get_rotation_matrix(mk["pitch"], mk["yaw"], mk["roll"]).astype(np.float32)
        tpl["c_eyes_lst"][k] = ((1 - a) * tpl["c_eyes_lst"][k] + a * tpl["c_eyes_lst"][0]).astype(np.float32)
        tpl["c_lip_lst"][k] = ((1 - a) * tpl["c_lip_lst"][k] + a * tpl["c_lip_lst"][0]).astype(np.float32)
    tpl["loop"] = "tile"
    return tpl


def cmd_synth(args):
    """Synthetic pose sweep on a neutral anchor — deterministic, no footage.
    Takes frame 0 of --base as the anchor and drives ONE pose angle with a smooth
    sine that starts and ends at 0 (loopable by construction)."""
    with open(args.base, "rb") as f:
        base = pickle.load(f)
    anchor = copy.deepcopy(base["motion"][0])
    e0 = base.get("c_eyes_lst", base.get("c_d_eyes_lst"))[0]
    l0 = base.get("c_lip_lst", base.get("c_d_lip_lst"))[0]

    fps = args.fps
    n = args.frames
    motion, c_eyes, c_lip = [], [], []
    for i in range(n):
        ph = i / (n - 1)
        # sin envelope: 0 → amp → 0, `cycles` swings, eased at both ends
        env = np.sin(np.pi * ph) ** 2
        val = args.amp * env * np.sin(2 * np.pi * args.cycles * ph)
        m = copy.deepcopy(anchor)
        m[args.pose] = (anchor[args.pose] + np.float32(val)).astype(np.float32)
        m["R"] = get_rotation_matrix(m["pitch"], m["yaw"], m["roll"]).astype(np.float32)
        motion.append(m)
        c_eyes.append(e0.copy())
        c_lip.append(l0.copy())

    _save(args.name, {"n_frames": n, "output_fps": fps, "motion": motion,
                      "c_eyes_lst": c_eyes, "c_lip_lst": c_lip, "loop": "tile"})


def cmd_preview(args):
    pipe = _pipe()
    if not pipe.prepare_source(args.image):
        sys.exit("no face in source image")
    with open(args.preset, "rb") as f:
        tpl = pickle.load(f)
    c_eyes = tpl.get("c_eyes_lst") or tpl.get("c_d_eyes_lst")
    c_lip = tpl.get("c_lip_lst") or tpl.get("c_d_lip_lst")
    h, w = pipe.src_imgs[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vout = cv2.VideoWriter(args.out, fourcc, tpl["output_fps"], (w, h))
    for i in range(tpl["n_frames"]):
        info = [tpl["motion"][i], c_eyes[i], c_lip[i]]
        _, out_org = pipe.run_with_pkl(info, pipe.src_imgs[0], pipe.src_infos[0],
                                       first_frame=(i == 0))
        vout.write(cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR))
    vout.release()
    print("preview:", args.out)


def cmd_info(args):
    with open(args.preset, "rb") as f:
        tpl = pickle.load(f)
    m = tpl["motion"]
    print(f"{args.preset}: {tpl['n_frames']} frames @ {tpl.get('output_fps')} fps, loop={tpl.get('loop')}")
    for key in ("pitch", "yaw", "roll"):
        vals = np.array([float(np.ravel(mm[key])[0]) for mm in m])
        print(f"  {key}: min {vals.min():.1f} max {vals.max():.1f} range {vals.max()-vals.min():.1f}")
    exp = np.stack([mm["exp"].ravel() for mm in m])
    print(f"  exp drift (L2 vs frame0): max {np.linalg.norm(exp - exp[0], axis=1).max():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--video", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--fps", type=int, default=0)
    e.add_argument("--max-frames", type=int, default=150)
    e.add_argument("--loopify", action="store_true")
    e.add_argument("--blend", type=int, default=12)
    e.add_argument("--dampen", type=float, default=1.0)
    e.add_argument("--trim", default=None, help="a:b frame range")
    e.set_defaults(fn=cmd_extract)

    s = sub.add_parser("synth")
    s.add_argument("--base", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--pose", choices=["pitch", "yaw", "roll"], required=True)
    s.add_argument("--amp", type=float, required=True)
    s.add_argument("--cycles", type=float, default=1.0)
    s.add_argument("--frames", type=int, default=125)
    s.add_argument("--fps", type=int, default=25)
    s.set_defaults(fn=cmd_synth)

    p = sub.add_parser("preview")
    p.add_argument("--preset", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="preview.mp4")
    p.set_defaults(fn=cmd_preview)

    i = sub.add_parser("info")
    i.add_argument("--preset", required=True)
    i.set_defaults(fn=cmd_info)

    args = ap.parse_args()
    args.fn(args)
