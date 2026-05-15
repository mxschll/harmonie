FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HARMONIE_DATA_DIR=/data \
    HARMONIE_LIBRARIES=/music

# Essentia loads MP3 / M4A / WMA / etc. via FFmpeg.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better Docker layer caching.
COPY pyproject.toml README.md ./
COPY harmonie ./harmonie
RUN pip install --upgrade pip && pip install .

VOLUME ["/data", "/music"]
EXPOSE 8842

# tini: clean SIGTERM forwarding to uvicorn / worker processes.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["harmonie", "serve"]
