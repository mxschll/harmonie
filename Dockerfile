# syntax=docker/dockerfile:1

# essentia-tensorflow publishes manylinux wheels for x86_64 only, so this
# image is amd64. Python 3.12 is the newest interpreter it builds wheels for.

FROM python:3.12-slim AS build

# git lets setuptools-scm derive the version from the checkout. Pass
# SETUPTOOLS_SCM_PRETEND_VERSION instead when building without history.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# --pre is required: essentia-tensorflow only publishes development releases.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /src

# Dependencies first, from the project metadata plus an empty package, so
# this layer (essentia's 600 MB wheel included) survives source changes. The
# version is pinned to a constant here: a real one would differ per release
# and invalidate this layer and the model layer below.
COPY pyproject.toml README.md /src/
COPY harmonie/__init__.py /src/harmonie/
RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --pre .

# Bake the models in so a container needs no network on first run. They get
# their own layer, keyed on features.py alone, because the host is a
# university server that is not always reachable; with the layer cached a
# code change no longer depends on it. prefetch_models() is harmonie's own
# fetch path, retries included, so the build cannot drift from what the
# analyzer looks for at runtime.
COPY harmonie/features.py /src/harmonie/
ENV XDG_CACHE_HOME=/opt/harmonie/cache \
    TQDM_DISABLE=1
RUN <<'PY' python -
from harmonie.features import prefetch_models

# False means the style classifier is missing: tracks would be analysed with no
# genre data at all, so fail the build rather than ship that.
if not prefetch_models():
    raise SystemExit("model download failed")
PY

# The lock files are download coordination, not payload.
RUN find /opt/harmonie/cache -name '*.lock' -delete

ARG SETUPTOOLS_SCM_PRETEND_VERSION
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION}
COPY . /src
RUN pip install --pre --no-deps .

# Fail the build rather than the first scan if the native stack is broken.
# Decoding after importing essentia checks that its bundled FFmpeg and PyAV's
# coexist in one process.
RUN <<'PY' python -
import wave
from pathlib import Path

import essentia.standard  # noqa: F401
from harmonie.features import decode_mono

p = Path("/tmp/probe.wav")
with wave.open(str(p), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(44100)
    w.writeframes(bytes(4 * 4410))
audio, rate = decode_mono(p)
assert rate == 44100 and audio.size == 4410, (rate, audio.size)
print("native stack ok")
PY


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
