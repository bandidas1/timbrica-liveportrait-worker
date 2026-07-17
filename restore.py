"""Old-photo restore pre-pass: Real-ESRGAN x4 (BSD-3) + RestoreFormer++ (Apache-2.0).

Mirrors the browser tool's engine (public/js/ai-face-restore-engine.js) constant for
constant so a photo restored here looks like one restored on /photo-restore:
  - FFHQ-512 facexlib 5-point template, Umeyama similarity alignment
  - input  (rgb/127.5 - 1), output (y*0.5 + 0.5)*255
  - feathered elliptical paste-back mask
CodeFormer is deliberately NOT here (S-Lab non-commercial — see the ML licensing
audit). ONNX sessions are lazy, CUDA EP with CPU fallback; every public function is
fail-soft (returns None on any problem — the caller animates the un-restored image).
"""

import os

import cv2
import numpy as np

RESTORE_DIR = os.environ.get("PA_RESTORE_DIR", "/app/checkpoints/restore")

# facexlib FFHQ-512 5-point template: [left_eye, right_eye, nose, left_mouth, right_mouth]
FFHQ5 = np.array([
    [192.98138, 239.94708],
    [318.90277, 240.19366],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
], dtype=np.float32)

FACE_SIZE = 512
ESRGAN_TILE = 256
ESRGAN_OVERLAP = 16

# RestoreFormer strength: blend restored↔original inside the paste mask. Full-on
# RF on a heavily pre-upscaled tiny face drifts identity (measured on a 280-px
# Lincoln scan — the output looked like a younger different man); 0.75 keeps the
# sharpening while anchoring identity to the real pixels. Mirrors the browser
# tool's strength slider default behavior.
RF_STRENGTH = float(os.environ.get("PA_RF_STRENGTH", "0.75"))

_SESS: dict = {}


def _session(name: str):
    if name not in _SESS:
        import onnxruntime as ort

        path = os.path.join(RESTORE_DIR, name)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _SESS[name] = ort.InferenceSession(
            path, so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return _SESS[name]


def _mp_5points(lmk478: np.ndarray) -> np.ndarray:
    """MediaPipe 478 landmarks → the 5 alignment points (viewer-order like FFHQ5).

    Iris rings exist because the face model runs with refine_landmarks=True:
    468–472 = subject-RIGHT iris (viewer-left), 473–477 = subject-LEFT iris
    (viewer-right). Nose tip = 1, mouth corners = 61 (viewer-left) / 291.
    """
    le = lmk478[468:473].mean(axis=0)
    re = lmk478[473:478].mean(axis=0)
    nose = lmk478[1]
    lm = lmk478[61]
    rm = lmk478[291]
    return np.array([le, re, nose, lm, rm], dtype=np.float32)


def esrgan_x4(img_bgr: np.ndarray) -> np.ndarray | None:
    """Whole-image ×4 super-resolution, tiled. Returns None on any failure."""
    try:
        sess = _session("realesrgan-x4-fp16.onnx")
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = np.zeros((h * 4, w * 4, 3), dtype=np.float32)
        inp_name = sess.get_inputs()[0].name
        in_dtype = np.float16 if "16" in sess.get_inputs()[0].type else np.float32

        step = ESRGAN_TILE - ESRGAN_OVERLAP * 2
        ys = list(range(0, max(1, h - ESRGAN_OVERLAP), step)) or [0]
        xs = list(range(0, max(1, w - ESRGAN_OVERLAP), step)) or [0]
        for y in ys:
            for x in xs:
                y1, x1 = min(h, y + ESRGAN_TILE), min(w, x + ESRGAN_TILE)
                y0, x0 = max(0, y1 - ESRGAN_TILE), max(0, x1 - ESRGAN_TILE)
                tile = rgb[y0:y1, x0:x1]
                th, tw = tile.shape[:2]
                t = np.transpose(tile, (2, 0, 1))[None].astype(in_dtype)
                y_out = sess.run(None, {inp_name: t})[0][0].astype(np.float32)
                y_img = np.transpose(y_out, (1, 2, 0))
                # inner region (skip overlap edges except at borders)
                iy0 = 0 if y0 == 0 else ESRGAN_OVERLAP
                ix0 = 0 if x0 == 0 else ESRGAN_OVERLAP
                iy1 = th if y1 == h else th - ESRGAN_OVERLAP
                ix1 = tw if x1 == w else tw - ESRGAN_OVERLAP
                out[(y0 + iy0) * 4:(y0 + iy1) * 4, (x0 + ix0) * 4:(x0 + ix1) * 4] = \
                    y_img[iy0 * 4:iy1 * 4, ix0 * 4:ix1 * 4]
        out8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(out8, cv2.COLOR_RGB2BGR)
    except Exception as exc:  # noqa: BLE001
        print("esrgan failed (soft):", exc)
        return None


def restoreformer_face(img_bgr: np.ndarray, lmk478: np.ndarray) -> np.ndarray | None:
    """Restore ONE face in place (aligned 512 crop → model → feathered paste-back).

    Returns the full image with the face restored, or None on failure.
    """
    try:
        sess = _session("restoreformer-pp-fp16.onnx")
        p5 = _mp_5points(lmk478)

        from skimage.transform import SimilarityTransform

        tf = SimilarityTransform()
        if not tf.estimate(p5, FFHQ5):
            return None
        M = tf.params[:2].astype(np.float32)

        aligned = cv2.warpAffine(img_bgr, M, (FACE_SIZE, FACE_SIZE),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        t = (np.transpose(rgb, (2, 0, 1))[None] / 127.5 - 1.0)
        in_meta = sess.get_inputs()[0]
        in_dtype = np.float16 if "16" in in_meta.type else np.float32
        y = sess.run(None, {in_meta.name: t.astype(in_dtype)})[0][0].astype(np.float32)
        restored = np.clip((np.transpose(y, (1, 2, 0)) * 0.5 + 0.5) * 255, 0, 255)
        restored_bgr = cv2.cvtColor(restored.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # feathered elliptical mask (browser parity: alpha falloff, 512²)
        mask = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.float32)
        cv2.ellipse(mask, (FACE_SIZE // 2, FACE_SIZE // 2),
                    (int(FACE_SIZE * 0.42), int(FACE_SIZE * 0.48)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 24)

        h, w = img_bgr.shape[:2]
        inv = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(restored_bgr, inv, (w, h), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(mask, inv, (w, h), flags=cv2.INTER_LINEAR)[..., None]
        alpha = alpha * RF_STRENGTH
        out = (back.astype(np.float32) * alpha
               + img_bgr.astype(np.float32) * (1 - alpha))
        return np.clip(out, 0, 255).astype(np.uint8)
    except Exception as exc:  # noqa: BLE001
        print("restoreformer failed (soft):", exc)
        return None
