# --- STAGE 1: Base Image ---
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime


# --- STAGE 2: Build & Runtime ---
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy dependency files
COPY requirements.txt .

# System dependencies for OpenCV, Torch, and image/video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from PyPI (online)
RUN pip install --no-cache-dir -r requirements.txt

# Copy core project files
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY artifacts ./artifacts

# Default environment config
ENV CONFIG_PATH=configs/pipeline_params.yaml

# Expose API ports
EXPOSE 8000
EXPOSE 8001

# Optional: health check
HEALTHCHECK CMD curl --fail http://localhost:8000/docs || exit 1

# API selection
ARG API=batch
ENV API=${API}

# Entry point
CMD if [ "$API" = "single" ]; then \
        uvicorn scripts.single_api:app --host 0.0.0.0 --port 8001; \
    else \
        uvicorn scripts.batch_api:app --host 0.0.0.0 --port 8000; \
    fi