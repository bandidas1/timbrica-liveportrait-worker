"""
timbrica-liveportrait-worker — RunPod serverless photo animation (LivePortrait).

A still photo + a named motion preset → a short MP4 where the chosen face comes
alive (smile / nod / look around …). The paid engine of /photo-animate; the app's
App\\Services\\AiVideo\\RunPodPortraitProvider submits here.

Contract (must stay in lockstep with RunPodPortraitProvider.php):

  INPUT  event["input"] = {
    "preset":     "smile" | "nod" | ...          # a pkl baked into /app/presets
    "duration_s": 5 | 10,
    "image_b64":  "<jpeg/png/webp base64>",      # browser downscales; body ≤ 10 MiB
    "face_box":   {"x","y","w","h"} 0..1 | null, # client hint, worker re-detects
    "restore":    "auto" | "on" | "off",         # face restore pre-pass, default auto
    "multiplier": 0.5..1.4,                      # motion intensity, default 1.0
    "mark":       true                           # AI-provenance metadata (default on)
  }

  OUTPUT {
    "mp4_b64": ..., "width", "height", "fps", "n_frames", "duration_s",
    "faces_detected", "face_used": {x,y,w,h},    # normalized box actually animated
    "restore_applied": "esrgan+rf"|"rf"|"esrgan"|"none",
    "preset_version", "gen_seconds", "timings": {...}
  }
  on failure: {"error": "<reason>"}   # the app maps ANY error to a full token refund

The result rides inline as base64 (1–6 MB — under RunPod's ~20 MB result ceiling).
`result_put_url` (signed PUT) is reserved for a future HD tier; not implemented.
"""

import base64
import io
import os
import pickle
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, "/app/flp")
sys.path.insert(0, "/app")

import torch_predictor

torch_predictor.install()  # MUST precede any flp pipeline import (see its header)

from omegaconf import OmegaConf
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

import restore as restore_mod

CFG_PATH = os.environ.get("PA_CONFIG", "/app/config/infer.yaml")
PRESETS_DIR = os.environ.get("PA_PRESETS_DIR", "/app/presets")
PRESET_VERSION = "v1"

MAX_DIM = 1280          # matches infer.yaml source_max_dim (coords must line up)
SMALL_PHOTO_PX = 0.55e6  # below this → ESRGAN pre-upscale (same as browser photo-restore)
MAX_DURATION_S = 10
DEFAULT_FPS = 25

# Deep-Nostalgia zoom mode: a face smaller than this (px height in the working
# image) animates as mush AND MediaPipe often misses it entirely on group photos —
# so re-crop a portrait region around the face from the FULL-RES original instead.
ZOOM_MIN_FACE_PX = 200
ZOOM_MARGIN = 2.8

# Warm cache — pipeline (and its CUDA modules) load once per worker.
_PIPE = None
_PRESETS: dict = {}


def _pipe() -> FasterLivePortraitPipeline:
    global _PIPE
    if _PIPE is None:
        cfg = OmegaConf.load(CFG_PATH)
        _PIPE = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
    return _PIPE


def _allowed_presets() -> set:
    try:
        return {f[:-4] for f in os.listdir(PRESETS_DIR) if f.endswith(".pkl")}
    except FileNotFoundError:
        return set()


def _load_preset(name: str) -> dict:
    if name not in _PRESETS:
        with open(os.path.join(PRESETS_DIR, f"{name}.pkl"), "rb") as f:
            _PRESETS[name] = pickle.load(f)
    return _PRESETS[name]


def _decode_image(b64: str) -> np.ndarray:
    """base64 → BGR ndarray, EXIF rotation honored (cv2.imdecode ignores EXIF)."""
    try:
        raw = base64.b64decode(b64, validate=True)
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("bad_image") from exc


def _resize_max(img: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Shrink to ≤max_dim and force even dims — done BEFORE detection so the
    landmark coords match what prepare_source sees (its own resize becomes a no-op)."""
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    nw, nh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


def _bbox_of(lmk: np.ndarray) -> tuple:
    x0, y0 = lmk.min(axis=0)
    x1, y1 = lmk.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _iou(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1e-6, area_a + area_b - inter)


def _zoom_crop(orig: np.ndarray, box_norm: tuple, margin: float = ZOOM_MARGIN) -> np.ndarray:
    """Crop a portrait region around a normalized (x0,y0,x1,y1) box from the
    FULL-RES original — the Deep-Nostalgia answer to group photos: animate a
    close-up, not a 60-px face lost in a 1280-px frame."""
    oh, ow = orig.shape[:2]
    x0, y0, x1, y1 = box_norm
    cx, cy = (x0 + x1) / 2 * ow, (y0 + y1) / 2 * oh
    size = max((x1 - x0) * ow, (y1 - y0) * oh) * margin
    size = max(size, 320.0)
    half = size / 2
    ax0 = int(max(0, cx - half))
    ay0 = int(max(0, cy - half))
    ax1 = int(min(ow, cx + half))
    ay1 = int(min(oh, cy + half))
    crop = orig[ay0:ay1, ax0:ax1]
    if min(crop.shape[:2]) < 32:
        return crop
    # tiny crop → ESRGAN before the pipeline sees it (a 250-px crop of a group
    # scan carries no facial detail otherwise)
    if crop.shape[0] * crop.shape[1] < SMALL_PHOTO_PX / 2:
        up = restore_mod.esrgan_x4(crop)
        if up is not None:
            crop = up
    return _resize_max(crop, 1024)


def _pick_face(faces: list, face_box: dict | None, w: int, h: int) -> int:
    if face_box:
        hint = (face_box["x"] * w, face_box["y"] * h,
                (face_box["x"] + face_box["w"]) * w, (face_box["y"] + face_box["h"]) * h)
        scored = [(_iou(_bbox_of(l), hint), i) for i, l in enumerate(faces)]
        best = max(scored)
        if best[0] > 0.1:
            return best[1]
        # hint missed every face (stale coords after restore shift) → fall through
    areas = [(_bbox_of(l)[2] - _bbox_of(l)[0]) * (_bbox_of(l)[3] - _bbox_of(l)[1])
             for l in faces]
    return int(np.argmax(areas))


def _encode_mp4(frames_rgb: list, fps: int, mark: bool) -> bytes:
    """Raw RGB frames → H.264 yuv420p +faststart via one ffmpeg pipe.

    The AI-provenance tag lives HERE: the app's ai-mark.js passes MP4 containers
    through untouched, so the container metadata written by the worker is the only
    machine-readable mark (EU AI Act art. 50 disclosure; visible note is on-page).
    """
    h, w = frames_rgb[0].shape[:2]
    out_path = tempfile.mktemp(suffix=".mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-crf", "18", "-preset", "veryfast", "-movflags", "+faststart",
    ]
    if mark:
        cmd += [
            "-metadata", "comment=AI-generated animation (Timbrica photo-animate)",
            "-metadata", "synopsis=digitalsourcetype=trainedAlgorithmicMedia",
        ]
    cmd += [out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for fr in frames_rgb:
            proc.stdin.write(fr.tobytes())
        proc.stdin.close()
        rc = proc.wait(timeout=120)
        if rc != 0:
            raise RuntimeError(f"encode_failed_{rc}: {proc.stderr.read()[-300:]!r}")
        with open(out_path, "rb") as f:
            data = f.read()
        if len(data) < 10_000 or data[4:8] != b"ftyp":
            raise RuntimeError("encode_bad_output")
        return data
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _frame_indices(n_template: int, n_target: int, loop: str) -> list:
    """Extend a template to n_target frames. pingpong is artifact-free on any
    template; tile only for presets authored loopable (last≈first)."""
    if n_target <= n_template:
        return list(range(n_target))
    idx = []
    if loop == "tile":
        while len(idx) < n_target:
            idx.extend(range(n_template))
        return idx[:n_target]
    forward = list(range(n_template))
    backward = list(range(n_template - 2, 0, -1))
    cycle = forward + backward
    while len(idx) < n_target:
        idx.extend(cycle)
    return idx[:n_target]


def handler(event: dict) -> dict:
    t_start = time.time()
    timings = {}
    try:
        inp = event.get("input", {}) or {}

        preset_name = str(inp.get("preset", ""))
        if preset_name not in _allowed_presets():
            return {"error": "bad_preset"}

        try:
            duration_s = int(inp.get("duration_s", 5))
        except (TypeError, ValueError):
            return {"error": "bad_duration"}
        if duration_s not in (5, 10):
            return {"error": "bad_duration"}

        b64 = inp.get("image_b64") or ""
        if not b64:
            return {"error": "bad_image"}

        face_box = inp.get("face_box") or None
        if face_box is not None:
            try:
                face_box = {k: float(face_box[k]) for k in ("x", "y", "w", "h")}
            except (KeyError, TypeError, ValueError):
                face_box = None

        restore_mode = str(inp.get("restore", "auto"))
        if restore_mode not in ("auto", "on", "off"):
            restore_mode = "auto"

        try:
            multiplier = float(inp.get("multiplier", 1.0))
        except (TypeError, ValueError):
            multiplier = 1.0
        multiplier = min(1.4, max(0.5, multiplier))

        mark = bool(inp.get("mark", True))

        # ── decode + normalize size ────────────────────────────────────────────
        t0 = time.time()
        orig = _decode_image(b64)
        img = _resize_max(orig)
        timings["decode_ms"] = round((time.time() - t0) * 1000)

        pipe = _pipe()
        fa = pipe.model_dict["face_analysis"]

        # ── detect; zoom mode for small/undetected faces (group photos) ───────
        t0 = time.time()
        faces = fa.predict(img)
        restore_applied = []
        zoom = False
        h, w = img.shape[:2]

        if faces:
            pick = _pick_face(faces, face_box, w, h)
            bx0, by0, bx1, by1 = _bbox_of(faces[pick])
            if (by1 - by0) < ZOOM_MIN_FACE_PX:
                crop = _zoom_crop(orig, (bx0 / w, by0 / h, bx1 / w, by1 / h))
                cf = fa.predict(crop)
                if cf:
                    img, faces, zoom = crop, cf, True
                    h, w = img.shape[:2]
                    pick = _pick_face(faces, None, w, h)
        elif face_box:
            # MediaPipe found nothing at 1280 (typical on group scans), but the
            # user pointed at a face — trust the hint, crop full-res, re-detect.
            crop = _zoom_crop(orig, (face_box["x"], face_box["y"],
                                     face_box["x"] + face_box["w"],
                                     face_box["y"] + face_box["h"]),
                              margin=2.2)
            cf = fa.predict(crop)
            if cf:
                img, faces, zoom = crop, cf, True
                h, w = img.shape[:2]
                pick = _pick_face(faces, None, w, h)

        if not faces and restore_mode != "off" and w * h < SMALL_PHOTO_PX:
            # tiny/faded scan rescue: ESRGAN the whole frame, try once more
            up = restore_mod.esrgan_x4(img)
            if up is not None:
                img = _resize_max(up)
                restore_applied.append("esrgan")
                faces = fa.predict(img)
                h, w = img.shape[:2]
                if faces:
                    pick = _pick_face(faces, face_box, w, h)
        if not faces:
            return {"error": "no_face"}
        if zoom:
            restore_applied.append("zoom")
        timings["detect_ms"] = round((time.time() - t0) * 1000)

        # ── face restore pre-pass (fail-soft: never kills the job) ────────────
        if restore_mode != "off":
            t0 = time.time()
            try:
                restored = restore_mod.restoreformer_face(img, faces[pick])
                if restored is not None:
                    img = restored
                    restore_applied.append("rf")
                    # geometry is warp-preserving, but re-detect for honest landmarks
                    faces2 = fa.predict(img)
                    if faces2:
                        prev = _bbox_of(faces[pick])
                        faces = faces2
                        pick = int(np.argmax([_iou(_bbox_of(l), prev) for l in faces]))
            except Exception as exc:  # noqa: BLE001
                print("restore failed (soft):", exc)
            timings["restore_ms"] = round((time.time() - t0) * 1000)

        x0, y0, x1, y1 = _bbox_of(faces[pick])
        face_used = {"x": x0 / w, "y": y0 / h, "w": (x1 - x0) / w, "h": (y1 - y0) / h}

        # ── LivePortrait source prep for the CHOSEN face only ─────────────────
        t0 = time.time()
        chosen = faces[pick]
        src_path = tempfile.mktemp(suffix=".png")
        cv2.imwrite(src_path, img)
        orig_predict = fa.predict
        fa.predict = lambda *a, **k: [chosen]
        try:
            ok = pipe.prepare_source(src_path)
        finally:
            fa.predict = orig_predict
            try:
                os.unlink(src_path)
            except OSError:
                pass
        if not ok:
            return {"error": "no_face"}
        timings["prepare_ms"] = round((time.time() - t0) * 1000)

        # ── animate ───────────────────────────────────────────────────────────
        t0 = time.time()
        tpl = _load_preset(preset_name)
        fps = int(tpl.get("output_fps") or DEFAULT_FPS)
        motion = tpl["motion"]
        c_eyes = tpl.get("c_eyes_lst") or tpl.get("c_d_eyes_lst")
        c_lip = tpl.get("c_lip_lst") or tpl.get("c_d_lip_lst")
        n_target = min(MAX_DURATION_S, duration_s) * fps
        order = _frame_indices(len(motion), n_target, str(tpl.get("loop", "pingpong")))

        pipe.cfg.infer_params.driving_multiplier = multiplier
        frames = []
        for i, k in enumerate(order):
            info = [motion[k], c_eyes[k], c_lip[k]]
            _, out_org = pipe.run_with_pkl(info, pipe.src_imgs[0], pipe.src_infos[0],
                                           first_frame=(i == 0))
            if out_org is None:
                return {"error": "animate_failed"}
            frames.append(np.ascontiguousarray(out_org))
        timings["animate_ms"] = round((time.time() - t0) * 1000)

        # ── encode ────────────────────────────────────────────────────────────
        t0 = time.time()
        mp4 = _encode_mp4(frames, fps, mark)
        timings["encode_ms"] = round((time.time() - t0) * 1000)

        oh, ow = frames[0].shape[:2]
        return {
            "mp4_b64": base64.b64encode(mp4).decode("ascii"),
            "width": ow, "height": oh, "fps": fps,
            "n_frames": len(frames), "duration_s": round(len(frames) / fps, 2),
            "faces_detected": len(faces),
            "face_used": {k: round(v, 4) for k, v in face_used.items()},
            "zoom": zoom,
            "restore_applied": "+".join(restore_applied) if restore_applied else "none",
            "preset_version": PRESET_VERSION,
            "gen_seconds": round(time.time() - t_start, 2),
            "timings": timings,
        }

    except ValueError as exc:
        return {"error": str(exc)[:100]}
    except Exception as exc:  # noqa: BLE001 — the app refunds on ANY error string
        import traceback

        traceback.print_exc()
        return {"error": str(exc)[:200]}


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
