# timbrica-liveportrait-worker

RunPod serverless **photo animation** (LivePortrait, ONNX) — the engine of `/photo-animate`
("bring an old photo to life", Deep-Nostalgia class). A still photo + a named motion
preset → a short MP4 where the chosen face smiles / nods / looks around.

The app side is `App\Services\AiVideo\RunPodPortraitProvider`. **Keep the I/O contract
in `handler.py` in lockstep with that provider.**

> Mirrored to `github.com/bandidas1/timbrica-liveportrait-worker` (CI builds the image
> to ghcr, like the demucs / svc / qwen-tts workers). It sits under `services/` so it
> ships with the feature; nothing here imports the Laravel app.

## Licensing (the whole reason this file layout exists)

- **LivePortrait code + Kuaishou weights: MIT.** The ONLY non-commercial piece of stock
  LivePortrait is the InsightFace detector, and Kuaishou's own LICENSE says to replace
  it for commercial use. This worker **never installs InsightFace** — face analysis is
  **MediaPipe FaceMesh (Apache-2.0)**, via FasterLivePortrait's (MIT, pinned SHA in the
  Dockerfile) `onnx_mp_infer` lane.
- Restoration pre-pass: **RestoreFormer++ (Apache-2.0)** + **Real-ESRGAN (BSD-3)** — the
  same ONNX files our browser tools ship (see `docs/ml-licensing-audit-2026-05-31.md`;
  CodeFormer is S-Lab non-commercial and is deliberately NOT used).
- Motion presets are **de-identified parameter curves** (pose angles + expression
  coefficients extracted offline by `build_presets.py`) — no face pixels, no identity.

## Contract (must match RunPodPortraitProvider.php)

```
IN  input = {
  preset:     "smile" | "nod" | "look-around" | "blink" | "laugh" | "gentle",
  duration_s: 5 | 10,
  image_b64:  "<jpeg/png/webp base64>",          # browser downscales; /run body ≤ 10 MiB
  face_box:   {x,y,w,h} normalized 0..1 | null,  # client hint; worker re-detects and
                                                 # picks nearest face (largest if null)
  restore:    "auto" | "on" | "off",             # default auto (see below)
  multiplier: 0.5..1.4,                          # motion intensity, default 1.0
  mark:       true                               # AI-provenance metadata tag (default on)
}
OUT {
  mp4_b64, width, height, fps, n_frames, duration_s,
  faces_detected, face_used: {x,y,w,h},          # normalized, what was animated
  restore_applied: "esrgan+rf" | "rf" | "none",
  preset_version, gen_seconds, timings: {...}
}
err { error: "no_face" | "bad_preset" | "bad_image" | "bad_duration" | "<reason>" }
     # the app maps ANY error to a full token refund
```

The result rides INLINE as base64 (a 5 s clip at ≤1280 px is 1–6 MB — under RunPod's
~20 MB result ceiling). `result_put_url` (signed PUT, SvcTransfer-style) is reserved as
a future input for an HD tier; not implemented.

## Pipeline (per request, warm)

1. decode (PIL, EXIF-rotation honored) → BGR, downscale to ≤1280 px (even dims)
2. MediaPipe FaceMesh detect (max 8 faces) → pick by `face_box` IoU / largest
3. restore pre-pass (`restore.py`, fail-soft — a restore error never kills the job):
   - `auto`: Real-ESRGAN ×4 first when the image is small (<0.55 MP — the same
     threshold the browser photo-restore uses), then RestoreFormer++ on every
     detected face (FFHQ-512 Umeyama alignment, feathered paste-back — mirrors
     `public/js/ai-face-restore-engine.js` constants exactly)
   - `on`: force both; `off`: skip
4. re-detect on the restored image → LivePortrait prepare (appearance features)
   for the CHOSEN face only
5. play the preset's motion template frame-by-frame (relative motion vs frame 0,
   `driving_multiplier` = `multiplier`), paste-back into the full photo
6. ffmpeg rawvideo pipe → H.264 yuv420p `+faststart` + AI-provenance metadata
   (`digitalsourcetype=trainedAlgorithmicMedia`) — MP4 metadata is applied HERE
   because the app's `ai-mark.js` passes MP4 containers through untouched

## Files

| File | Role |
|---|---|
| `handler.py` | RunPod handler: decode → detect/pick face → restore → animate → mp4 |
| `restore.py` | Real-ESRGAN ×4 + RestoreFormer++ pre-pass (ONNX, CUDA→CPU fallback) |
| `download_models.py` | bakes LivePortrait ONNX + restore models into the image at build |
| `build_presets.py` | OFFLINE: driving video / synthetic pose curves → `presets/*.pkl` |
| `presets/*.pkl` | motion templates `{n_frames, output_fps, motion, c_eyes_lst, c_lip_lst}` |
| `patches/mediapipe_face_model.py` | FLP override: multi-face (max 8) + tunable confidence |
| `config/infer.yaml` | pinned FLP onnx_mp config with /app paths |
| `local_smoke.py` | run the handler on a local photo, dump mp4 + preview frames |

## 1. Build presets (offline, once per preset revision)

On a GPU box with this repo + the baked image (or the deps installed):

```bash
# from a driving video (head-and-shoulders, front-facing, the motion you want):
python build_presets.py extract --video smile.mp4 --name smile --fps 25 --max-frames 125 --loopify
# synthetic pose-only presets (no source footage needed, deterministic):
python build_presets.py synth --base presets/smile.pkl --name nod        --pose pitch --amp 8  --cycles 2
python build_presets.py synth --base presets/smile.pkl --name look-around --pose yaw  --amp 11 --cycles 1
python build_presets.py preview --preset presets/nod.pkl --image some-photo.jpg --out preview.mp4
```

Bump `PRESET_VERSION` in `handler.py` whenever a pkl changes — the app stores it per
generation for audit.

## 2. Local smoke (before any endpoint deploy)

```bash
pip install -r requirements.txt
python local_smoke.py old-photo.jpg --preset smile --out ./out
# writes out/result.mp4 + out/frame-*.jpg (visual check) + prints timings
```

## 3. Build & push

CI (`.github/workflows/build.yml` in the mirror repo) builds to
`ghcr.io/bandidas1/timbrica-liveportrait-worker` on push. Pin endpoints by
**digest**, never by tag (see memory `reference_runpod_template_repin`).

## 4. RunPod serverless endpoint

- GPU: RTX 4090 flex (model VRAM ~3 GB; 24 GB is comfortable, A5000/L4 also fine).
- Flex workers, workersMin 0, **idleTimeout ≥ 30 s**, execution timeout ≥ 240 s,
  container disk ~15 GB.
- No env needed inside the container. Note the **endpoint id**.
- App env: `AI_VIDEO_ANIMATE_ENDPOINT=<id>` + shared `RUNPOD_API_KEY`.

## 5. 🔴 LIVE MEASURE before pricing flip

Token price (`AI_VIDEO_ANIMATE_5_TOKENS`, provisionally 500) is not final until:
1. warm `gen_seconds` × RunPod $/s for the GPU + cold-start amortization measured
   on the real endpoint (target COGS ≈ 0.3–1 ₽/clip);
2. every preset eyeballed on real archival photos (sepia, low-res, damaged, group);
3. cold start fits the app's poll budget (≈5 min) with margin.
