# Timbrica override of FasterLivePortrait's src/models/mediapipe_face_model.py.
#
# Differences from upstream (kept minimal on purpose):
#   - max_num_faces=8 (upstream: 1) — /photo-animate detects every face so the user
#     can pick one on a group photo;
#   - min_detection_confidence=0.35 (upstream 0.5) — archival scans are low-contrast
#     and faded; the app re-validates the pick against the client's face_box hint.
#
# MediaPipe is Apache-2.0 — this file is the InsightFace replacement that Kuaishou's
# own LICENSE prescribes for commercial use of LivePortrait.
import cv2
import mediapipe as mp
import numpy as np


class MediaPipeFaceModel:
    """MediaPipe FaceMesh → list of 478x2 landmark arrays (image pixel coords)."""

    def __init__(self, **kwargs):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=int(kwargs.get("max_num_faces", 8)),
            refine_landmarks=True,
            min_detection_confidence=float(kwargs.get("min_detection_confidence", 0.35)),
        )

    def predict(self, *data):
        img_bgr = data[0]
        h, w = img_bgr.shape[:2]
        results = self.face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return []
        outs = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark]
            outs.append(np.array(landmarks))
        return outs
