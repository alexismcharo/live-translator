FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/opt/cache

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip ffmpeg ca-certificates curl \
 && ln -s /usr/bin/python3.11 /usr/local/bin/python \
 && python -m pip install --upgrade pip \
 && mkdir -p ${XDG_CACHE_HOME} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Install torch (CUDA) & Python deps first (stable layers)
RUN python -m pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# --- Preload Whisper model into cache (persists in image)
# This downloads model files into /opt/cache/whisper (because of XDG_CACHE_HOME)
RUN python - <<'PY'
import whisper, os
print("Cache dir:", os.getenv("XDG_CACHE_HOME"))
whisper.load_model("large-v3")
print("Preloaded whisper model.")
PY

# Verify (optional)
RUN ls -lh ${XDG_CACHE_HOME}/whisper || true

# --- Copy app code last so edits don’t bust the cached model layer
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
