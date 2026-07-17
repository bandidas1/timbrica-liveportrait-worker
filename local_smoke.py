"""Run the handler on a local photo — no RunPod. Visual + timing check.

  python local_smoke.py old-photo.jpg --preset smile --duration 5 --out ./out
  python local_smoke.py photo.jpg --preset nod --restore off --face-box 0.1,0.2,0.3,0.4

Writes out/result.mp4 + out/frame-*.jpg (6 spread frames for eyeballing) and prints
the handler's timings. Anything in {"error": ...} exits non-zero.
"""

import argparse
import base64
import json
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--preset", default="smile")
ap.add_argument("--duration", type=int, default=5)
ap.add_argument("--restore", default="auto", choices=["auto", "on", "off"])
ap.add_argument("--multiplier", type=float, default=1.0)
ap.add_argument("--face-box", default=None, help="x,y,w,h normalized")
ap.add_argument("--out", default="./out")
args = ap.parse_args()

with open(args.image, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

face_box = None
if args.face_box:
    x, y, w, h = [float(v) for v in args.face_box.split(",")]
    face_box = {"x": x, "y": y, "w": w, "h": h}

from handler import handler  # noqa: E402 — heavy import after argparse

res = handler({"input": {
    "preset": args.preset,
    "duration_s": args.duration,
    "image_b64": b64,
    "face_box": face_box,
    "restore": args.restore,
    "multiplier": args.multiplier,
    "mark": True,
}})

if "error" in res:
    print("ERROR:", res["error"])
    sys.exit(1)

os.makedirs(args.out, exist_ok=True)
mp4 = base64.b64decode(res.pop("mp4_b64"))
mp4_path = os.path.join(args.out, "result.mp4")
with open(mp4_path, "wb") as f:
    f.write(mp4)

print(json.dumps(res, indent=2))
print(f"mp4: {mp4_path} ({len(mp4)} bytes)")

# 6 spread frames for a quick visual check
subprocess.call([
    "ffmpeg", "-y", "-loglevel", "error", "-i", mp4_path,
    "-vf", f"select='not(mod(n,{max(1, res['n_frames'] // 6)}))'", "-vsync", "vfr",
    os.path.join(args.out, "frame-%02d.jpg"),
])
print("frames:", sorted(f for f in os.listdir(args.out) if f.startswith("frame-")))
