# syntax=docker/dockerfile:1

# essentia-tensorflow publishes manylinux wheels for x86_64 only, so this
# image is amd64. Python 3.12 is the newest interpreter it builds wheels for.

FROM python:3.12-slim AS build

# git lets setuptools-scm derive the version from the checkout. Pass
# SETUPTOOLS_SCM_PRETEND_VERSION instead when building without history.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG SETUPTOOLS_SCM_PRETEND_VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}

# --pre is required: essentia-tensorflow only publishes development releases.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /src
COPY . /src
RUN pip install --pre .

# Bake the Essentia models into the image so a container needs no network on
# first run. The paths come from harmonie's own helpers, so they cannot drift
# from what the analyzer looks for at runtime.
ENV XDG_CACHE_HOME=/opt/harmonie/cache \
    TQDM_DISABLE=1
RUN <<'PY' python -
import time

from harmonie.features import (
    ensure_effnet_model,
    ensure_genre_head_model,
    ensure_genre_labels,
)

# essentia.upf.edu drops connections mid-download often enough to break builds.
# The helpers skip files that already exist and download through a .part file,
# so a retry resumes rather than corrupting anything.
for attempt in range(1, 6):
    try:
        print(ensure_effnet_model(), flush=True)
        print(ensure_genre_head_model(), flush=True)
        print(len(ensure_genre_labels()), "labels", flush=True)
        break
    except Exception as exc:
        print(f"model fetch failed (attempt {attempt}): {exc}", flush=True)
        if attempt == 5:
            raise
        time.sleep(5 * attempt)
PY

# Fail the build rather than the first scan if the native stack is broken.
RUN python -c "import essentia.standard, harmonie; print('essentia ok')"


FROM python:3.12-slim

# libgomp is the OpenMP runtime TensorFlow links against; everything else
# essentia needs is bundled in its wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/harmonie/cache /opt/harmonie/cache

# The library always lives at /music inside the container and the database
# always at /data. Keeping those fixed is what makes the database portable:
# the paths stored in it are container paths, identical on every host.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/opt/harmonie/cache \
    HARMONIE_DATA_DIR=/data \
    HARMONIE_LIBRARIES=/music \
    HARMONIE_HOST=0.0.0.0 \
    HARMONIE_PORT=8842

EXPOSE 8842

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8842/health', timeout=4)"]

# `docker run image` serves; `docker run image scan` runs one pass and exits.
ENTRYPOINT ["harmonie"]
CMD ["serve"]
