"""Feature extraction.

For one audio file, :class:`EffnetExtractor` produces:

* a 1280-d embedding (``np.ndarray`` of float32) for similarity search,
* a :class:`Descriptors` block of musical metadata (BPM, key, loudness, …),
* a 400-d Discogs style activation vector from a genre/style classifier
  head running on top of the Effnet embeddings.

Backed by ``essentia-tensorflow`` (the Discogs-Effnet model + classical
descriptor algorithms run on the same decoded audio).

Two extraction modes:

* :meth:`EffnetExtractor.extract` returns embedding + descriptors + styles.
* :meth:`EffnetExtractor.extract_descriptors` returns descriptors only.
  Used to refresh descriptor columns without re-running TensorFlow.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("harmonie.features")


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

# Bump this when the descriptor extraction pipeline changes (algorithm tweak,
# new field, etc.). Existing rows with an older descriptor_version get
# re-processed on the next scan, but the embedding is preserved.
DESCRIPTOR_VERSION = 1


# ---------------------------------------------------------------------------
# Models / constants
# ---------------------------------------------------------------------------

EFFNET_MODEL_URL = (
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/"
    "discogs-effnet-bs64-1.pb"
)
EFFNET_MODEL_FILENAME = "discogs-effnet-bs64-1.pb"
EFFNET_EMBEDDING_DIM = 1280
EFFNET_SAMPLE_RATE = 16000
EFFNET_OUTPUT_NODE = "PartitionedCall:1"

# Sample rate for descriptor algorithms (Essentia defaults).
DESCRIPTOR_SAMPLE_RATE = 44100


# Genre/style classifier head — runs on the 1280-d Effnet embeddings (no
# extra audio decoding) and outputs probabilities for 400 Discogs styles.
# Labels are formatted as ``"Genre---Style"``, e.g. ``"Electronic---House"``.
GENRE_HEAD_MODEL_URL = (
    "https://essentia.upf.edu/models/classification-heads/genre_discogs400/"
    "genre_discogs400-discogs-effnet-1.pb"
)
GENRE_HEAD_MODEL_FILENAME = "genre_discogs400-discogs-effnet-1.pb"
GENRE_HEAD_LABELS_URL = (
    "https://essentia.upf.edu/models/classification-heads/genre_discogs400/"
    "genre_discogs400-discogs-effnet-1.json"
)
GENRE_HEAD_LABELS_FILENAME = "genre_discogs400-discogs-effnet-1.json"
GENRE_NUM_CLASSES = 400
GENRE_HEAD_INPUT_NODE = "serving_default_model_Placeholder"
GENRE_HEAD_OUTPUT_NODE = "PartitionedCall:0"

# Top-K + threshold for the per-track ``track_styles`` rows the DB keeps for
# fast filtering. The full 400-d vector is also stored as a BLOB so consumers
# that want the long tail can still reach it.
STYLE_TOP_K = 10
STYLE_MIN_PROB = 0.05


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Descriptors:
    """Musical metadata for one track. Missing values mean the algorithm
    couldn't be applied or failed."""

    bpm: float | None = None
    bpm_confidence: float | None = None  # ~[0, 5.32]; >1.5 ~ confident
    key: str | None = None  # e.g. "A", "F#"
    scale: str | None = None  # "major" / "minor"
    key_strength: float | None = None  # [0, 1]
    loudness: float | None = None  # ReplayGain, dB
    danceability: float | None = None  # ~[0, 3]
    onset_rate: float | None = None  # onsets per second

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrackFeatures:
    """Complete extraction output for one audio file."""

    embedding: np.ndarray  # float32, shape (D,)
    duration: float  # seconds
    model: str
    descriptors: Descriptors = field(default_factory=Descriptors)
    # Discogs-400 style probabilities, shape (400,) float32, post-sigmoid.
    # ``None`` for backends that don't produce Effnet-compatible embeddings.
    style_activations: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return int(self.embedding.shape[0])


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

# The model host drops connections and times out often enough that a single
# attempt is not good enough: a failed style-classifier fetch silently costs a
# whole scan its genre data.
DOWNLOAD_TIMEOUT_SEC = 60
DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_RETRY_SLEEP_SEC = 3

# flock guards across processes; this guards threads inside one process.
_THREAD_LOCK = threading.Lock()


def _model_cache_dir() -> Path:
    try:
        from platformdirs import user_cache_dir

        base = Path(user_cache_dir("harmonie", "harmonie"))
    except Exception:
        base = Path.home() / ".cache" / "harmonie"
    p = base / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download(url: str, dest: Path) -> None:
    # A unique temp name per caller. Every worker process used to write the
    # same ``.part`` file and rename it into place, so the first rename won
    # and the rest crashed with FileNotFoundError.
    tmp = dest.with_name(f"{dest.name}.part.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    logger.info("downloading: %s", url)
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SEC) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or None
            try:
                from tqdm import tqdm

                # A progress bar is noise in a log file or `docker logs`.
                bar_ctx = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                    disable=not sys.stderr.isatty(),
                )
            except Exception:
                bar_ctx = None
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    if bar_ctx is not None:
                        bar_ctx.update(len(chunk))
            if bar_ctx is not None:
                bar_ctx.close()
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    # Atomic, and tolerant of another process having finished first.
    os.replace(tmp, dest)


@contextlib.contextmanager
def _download_lock(dest: Path):
    """Serialize downloads of ``dest`` across processes and threads.

    Twelve analysis workers starting at once used to mean twelve concurrent
    downloads of the same file, which wasted bandwidth and made a flaky model
    host time out. One downloads; the rest wait and find the file cached.
    """
    with _THREAD_LOCK:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows has no fcntl
            yield
            return

        with open(dest.with_name(dest.name + ".lock"), "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _ensure_cached(url: str, filename: str) -> Path:
    """Return the cached path for ``url``, downloading it once if needed."""
    dest = _model_cache_dir() / filename
    if dest.exists():
        return dest

    with _download_lock(dest):
        # Another process may have finished while we waited for the lock.
        if dest.exists():
            return dest

        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
            try:
                _download(url, dest)
                return dest
            except Exception as exc:  # noqa: BLE001 - retried below, then raised
                last_error = exc
                logger.warning(
                    "download failed (attempt %d/%d): %s",
                    attempt,
                    DOWNLOAD_ATTEMPTS,
                    exc,
                )
                if attempt < DOWNLOAD_ATTEMPTS:
                    time.sleep(DOWNLOAD_RETRY_SLEEP_SEC * attempt)

        assert last_error is not None
        raise last_error


def ensure_effnet_model() -> Path:
    return _ensure_cached(EFFNET_MODEL_URL, EFFNET_MODEL_FILENAME)


def ensure_genre_head_model() -> Path:
    """Download (once) the 400-style classifier head that runs on top of the
    Effnet embeddings."""
    return _ensure_cached(GENRE_HEAD_MODEL_URL, GENRE_HEAD_MODEL_FILENAME)


def prefetch_models() -> bool:
    """Fetch every model before the worker pool starts.

    Workers each construct their own extractor, so without this the download
    happens once per worker. Returns False when the style classifier could not
    be fetched, meaning tracks analysed now will carry no style data.
    """
    ensure_effnet_model()
    try:
        ensure_genre_head_model()
        ensure_genre_labels()
    except Exception as exc:  # noqa: BLE001 - degraded run is still useful
        logger.error(
            "style classifier unavailable (%s). Tracks analysed now get an "
            "embedding but no genres or styles; re-run `harmonie scan --force` "
            "once it can be downloaded.",
            exc,
        )
        return False
    return True


def ensure_genre_labels() -> list[str]:
    """Return the 400 ``"Genre---Style"`` labels in the order produced by the
    classifier head. Cached on disk alongside the model."""
    path = _ensure_cached(GENRE_HEAD_LABELS_URL, GENRE_HEAD_LABELS_FILENAME)
    with open(path) as f:
        meta = json.load(f)
    classes = meta.get("classes")
    if not isinstance(classes, list) or len(classes) != GENRE_NUM_CLASSES:
        raise ValueError(
            f"unexpected genre head metadata: expected {GENRE_NUM_CLASSES} "
            f"classes, got {len(classes) if isinstance(classes, list) else '?'}"
        )
    return [str(c) for c in classes]


def top_styles(
    activations: np.ndarray,
    labels: list[str],
    *,
    top_k: int = STYLE_TOP_K,
    min_prob: float = STYLE_MIN_PROB,
) -> list[tuple[str, float]]:
    """Return the highest-confidence ``(label, probability)`` pairs.

    Up to ``top_k`` entries; entries below ``min_prob`` are dropped.
    Ordered by descending probability.
    """
    if activations.shape != (GENRE_NUM_CLASSES,):
        raise ValueError(
            f"expected activation vector of shape ({GENRE_NUM_CLASSES},), "
            f"got {activations.shape}"
        )
    order = np.argsort(-activations)[:top_k]
    out: list[tuple[str, float]] = []
    for idx in order:
        prob = float(activations[idx])
        if prob < min_prob:
            break
        out.append((labels[int(idx)], prob))
    return out


# ---------------------------------------------------------------------------
# Descriptor extraction
# ---------------------------------------------------------------------------


def _safe(label: str, fn):
    try:
        return fn()
    except Exception as e:  # pragma: no cover
        logger.debug("descriptor %s failed: %s", label, e)
        return None


def compute_descriptors(audio: np.ndarray) -> Descriptors:
    """Compute musical descriptors from a mono float32 signal at 44.1 kHz."""
    from essentia.standard import (
        Danceability,
        KeyExtractor,
        OnsetRate,
        ReplayGain,
        RhythmExtractor2013,
    )

    d = Descriptors()

    rhythm = _safe("rhythm", lambda: RhythmExtractor2013(method="multifeature")(audio))
    if rhythm is not None:
        bpm, _beats, conf, _est, _ints = rhythm
        d.bpm = float(bpm)
        d.bpm_confidence = float(conf)

    key_out = _safe("key", lambda: KeyExtractor()(audio))
    if key_out is not None:
        key, scale, strength = key_out
        d.key = str(key)
        d.scale = str(scale)
        d.key_strength = float(strength)

    rg = _safe("loudness", lambda: ReplayGain()(audio))
    if rg is not None:
        d.loudness = float(rg)

    dance = _safe("danceability", lambda: Danceability()(audio))
    if dance is not None:
        d.danceability = float(dance[0])

    onsets = _safe("onset_rate", lambda: OnsetRate()(audio))
    if onsets is not None:
        d.onset_rate = float(onsets[1])

    return d


# ---------------------------------------------------------------------------
# Backend: Discogs-Effnet
# ---------------------------------------------------------------------------


class EffnetExtractor:
    """1280-d Discogs-Effnet embedding + classical descriptors + 400-style
    activation vector.

    The genre head is optional. If its model file isn't available,
    extraction returns embeddings and descriptors with
    ``TrackFeatures.style_activations = None``.
    """

    name = "discogs-effnet-bs64-1"
    dim = EFFNET_EMBEDDING_DIM

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        genre_head_path: Path | None = None,
        load_genre_head: bool = True,
    ) -> None:
        try:
            from essentia.standard import (
                MonoLoader,
                Resample,
                TensorflowPredictEffnetDiscogs,
            )
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "The 'effnet' backend requires essentia-tensorflow. "
                "Install with: pip install essentia-tensorflow"
            ) from e

        self._MonoLoader = MonoLoader
        self._Resample = Resample
        if model_path is None:
            model_path = ensure_effnet_model()
        self._model = TensorflowPredictEffnetDiscogs(
            graphFilename=str(model_path),
            output=EFFNET_OUTPUT_NODE,
        )

        # Genre head + label table. Loaded lazily so a network blip
        # during model download falls through to embeddings-only output.
        self._genre_head = None
        self._genre_labels: list[str] | None = None
        if load_genre_head:
            try:
                from essentia.standard import TensorflowPredict2D

                if genre_head_path is None:
                    genre_head_path = ensure_genre_head_model()
                self._genre_head = TensorflowPredict2D(
                    graphFilename=str(genre_head_path),
                    input=GENRE_HEAD_INPUT_NODE,
                    output=GENRE_HEAD_OUTPUT_NODE,
                )
                self._genre_labels = ensure_genre_labels()
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "genre head unavailable; tracks will be indexed without "
                    "style activations (%s)",
                    e,
                )
                self._genre_head = None
                self._genre_labels = None

    @property
    def genre_labels(self) -> list[str] | None:
        return self._genre_labels

    def _load_44k(self, path: Path) -> np.ndarray:
        audio = self._MonoLoader(
            filename=str(path),
            sampleRate=DESCRIPTOR_SAMPLE_RATE,
            resampleQuality=4,
        )()
        if audio.size == 0:
            raise ValueError(f"empty audio: {path}")
        return audio

    def extract(self, path: Path) -> TrackFeatures:
        audio_44k = self._load_44k(path)
        duration = float(audio_44k.shape[0]) / DESCRIPTOR_SAMPLE_RATE
        descriptors = compute_descriptors(audio_44k)

        audio_16k = self._Resample(
            inputSampleRate=DESCRIPTOR_SAMPLE_RATE,
            outputSampleRate=EFFNET_SAMPLE_RATE,
            quality=4,
        )(audio_44k)
        emb_frames = self._model(audio_16k)
        if emb_frames.ndim != 2 or emb_frames.shape[1] != EFFNET_EMBEDDING_DIM:
            raise ValueError(f"unexpected embedding shape {emb_frames.shape}")
        emb = emb_frames.mean(axis=0).astype(np.float32, copy=False)

        # Style activations: run the head on each per-frame embedding and
        # average the per-frame probabilities. Note sigmoid(mean(emb)) is
        # not equivalent to mean(sigmoid(head(emb))).
        style_activations: np.ndarray | None = None
        if self._genre_head is not None:
            try:
                act_frames = self._genre_head(emb_frames)
                if act_frames.ndim != 2 or act_frames.shape[1] != GENRE_NUM_CLASSES:
                    raise ValueError(
                        f"unexpected genre head output shape {act_frames.shape}"
                    )
                style_activations = act_frames.mean(axis=0).astype(
                    np.float32, copy=False
                )
            except Exception as e:  # pragma: no cover
                logger.warning(
                    "style classification failed for %s: %s",
                    path,
                    e,
                )

        return TrackFeatures(
            embedding=emb,
            duration=duration,
            model=self.name,
            descriptors=descriptors,
            style_activations=style_activations,
        )

    def extract_descriptors(self, path: Path) -> tuple[Descriptors, float]:
        """Compute only descriptors and duration. Used to top up old rows
        without re-running the model."""
        audio_44k = self._load_44k(path)
        duration = float(audio_44k.shape[0]) / DESCRIPTOR_SAMPLE_RATE
        return compute_descriptors(audio_44k), duration


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# Persisted in tracks.model and scans.model. Bump alongside any change to
# the embedding model (e.g. swap to a newer Discogs-Effnet checkpoint) so
# old rows get re-extracted on the next scan.
MODEL_NAME = EffnetExtractor.name

# Width of the embedding vector. Useful in the main process for sanity
# checks without importing essentia / loading TensorFlow.
EMBEDDING_DIM = EFFNET_EMBEDDING_DIM


# ---------------------------------------------------------------------------
# File signature
# ---------------------------------------------------------------------------


def file_signature(path: Path) -> tuple[int, float]:
    """(size, mtime) — cheap change detection."""
    import os

    st = os.stat(path)
    return st.st_size, st.st_mtime
