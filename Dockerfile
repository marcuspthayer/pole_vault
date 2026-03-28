FROM python:3.11-slim-bookworm

# System dependencies for OpenCV, MediaPipe, and YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libopenblas0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin numpy 1.x FIRST — mediapipe 0.10.9 and opencv are incompatible with numpy 2.x
RUN pip install --no-cache-dir "numpy>=1.24,<2"

# Install CPU-only PyTorch (after numpy so it won't upgrade to numpy 2.x)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (API-only, no Streamlit)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY api/ api/
COPY pvapp/ pvapp/
COPY step_detection/ step_detection/

# Bake ML models into the image
COPY yolo11n.pt pole_detect_v3.pt ./

# Create data directory (Railway persistent volume mounts here)
RUN mkdir -p /data/jobs

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/data

# Single worker: each worker loads YOLO + MediaPipe (~1.5GB RAM each)
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 120
