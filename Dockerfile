# Pen 6 BRD pipeline: Level 1 (behavior), Level 2 (BRD risk), Level 3 (live alerts).
#
#   git lfs pull                       # the model weights in models/ must be real files, not LFS pointers
#   docker build -t pen6 .             # CPU image
#   docker build -t pen6-gpu --build-arg BASE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime .   # NVIDIA GPU image
#
# Run (see README.md): mount the footage read-only at /videos and keep results in ./outputs on the host.
#   docker run --rm -it --env-file .env -v "/path/to/footage:/videos:ro" -v "$PWD/outputs:/app/outputs" pen6 \
#       python live_watch.py --clip "/videos/Day 2/Cam 01/Event20261108142130006.Avi"
ARG BASE=python:3.12-slim
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 TORCH_HOME=/opt/torch MPLBACKEND=Agg
# ffmpeg decodes the cameras' HEVC clips; libgl/libglib are needed by opencv
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-lock.txt ./
COPY brd_agent/requirements.txt brd_agent/requirements.txt
# CPU torch wheels on the slim image (a GPU base image already has torch with CUDA)
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN python -c "import torch" 2>/dev/null || pip install --index-url ${TORCH_INDEX} torch torchvision; \
    pip install -r requirements.txt -r brd_agent/requirements.txt

# DINOv2 ViT-S/14 (cow appearance for identity) comes from torch.hub: fetch it now so runs work offline
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)"

COPY . .
# fail early if the weights are Git LFS pointers (run `git lfs pull` before building)
RUN python - <<'EOF'
import pathlib, sys
bad = [str(p) for p in pathlib.Path("models").rglob("*.pt") if p.stat().st_size < 10_000]
sys.exit(f"model weights are Git LFS pointers, run `git lfs pull` first: {bad}") if bad else print("weights ok")
EOF

CMD ["python", "check_setup.py", "--help"]
