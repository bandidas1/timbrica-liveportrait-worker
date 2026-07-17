# LivePortrait photo-animation worker for RunPod serverless.
#
# Engine = ORIGINAL Kuaishou LivePortrait torch modules (MIT) driven through
# FasterLivePortrait's pipeline (MIT) with MediaPipe face analysis (Apache-2.0) —
# InsightFace is NEVER installed (the commercial-safe detector swap Kuaishou's own
# LICENSE prescribes). Torch, not ONNX/TensorRT, because stock onnxruntime-gpu has
# no CUDA kernel for the 5-D GridSample in the warping graph (see
# patches/torch_predictor.py) and TRT needs per-GPU engine builds.
#
# Weights (~700 MB total) are baked in: a cold start is image pull + module load,
# not a checkpoint download.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# torch first (CUDA 12.4 wheel), kept out of requirements.txt so pip resolution
# can't downgrade it behind our back.
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install -r requirements.txt

# FasterLivePortrait (MIT) pinned by SHA — pipeline/crop/mediapipe code only.
ARG FLP_SHA=8aad3602177547aaa5e4beec0c3ef5b7944e7a1f
RUN git clone https://github.com/warmshao/FasterLivePortrait.git flp \
    && cd flp && git checkout ${FLP_SHA} \
    && mkdir -p /app/config && cp assets/mask_template.png /app/config/mask_template.png \
    && rm -rf .git assets tests scripts webui.py api.py run.py \
        src/models/JoyVASA src/models/XPose src/models/kokoro \
        src/pipelines/gradio_live_portrait_pipeline.py \
        src/pipelines/joyvasa_audio_to_motion_pipeline.py \
    # InsightFace lane OUT (non-commercial): the module imports insightface at import
    # time and we never install it — face analysis is MediaPipe only.
    && rm src/models/face_analysis_model.py \
    && sed -i '/FaceAnalysisModel/d' src/models/__init__.py

# Original LivePortrait (MIT) pinned by SHA — torch module definitions + arch yaml.
ARG LP_SHA=9b294b3d0536135442ea73cb01e6cb3ca7029dd3
RUN git clone https://github.com/KlingAIResearch/LivePortrait.git lp-src \
    && cd lp-src && git checkout ${LP_SHA} \
    && cp -r src/modules /app/lp_modules \
    && cp src/config/models.yaml /app/config/lp_models.yaml \
    && cd /app && rm -rf lp-src

# Bake the models (cold start = disk read, not download).
COPY download_models.py .
RUN python download_models.py

# Our overlays: multi-face MediaPipe analysis + torch predictor + pinned config.
COPY patches/mediapipe_face_model.py flp/src/models/mediapipe_face_model.py
COPY patches/torch_predictor.py .
COPY config/infer.yaml config/infer.yaml
COPY presets/ presets/
COPY restore.py handler.py ./

CMD ["python", "-u", "handler.py"]
