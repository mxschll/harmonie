"""Audio feature extraction using Essentia.

Each backend produces, for one audio file:

* a fixed-length embedding (``np.ndarray`` of float32) for similarity search,
* a :class:`Descriptors` block of musical metadata (BPM, key, loudness, …),
* (effnet only) a 400-d Discogs style activation vector from a genre/style
  classifier head running on top of the Effnet embeddings. ``None`` for
  backends that don't produce embeddings in Discogs-Effnet space.

Backends:

* ``effnet`` (default): Discogs-Effnet 1280-d embedding via TensorFlow,
  plus Essentia descriptor algorithms on the same decoded audio.
  Requires ``essentia-tensorflow``.

* ``musicextractor``: Essentia's :class:`MusicExtractor`. Embedding is an
  aggregation of low-level descriptors. Requires only ``essentia``.

Two extraction modes are supported per backend:

* :meth:`Extractor.extract` returns embedding + descriptors + styles.
* :meth:`Extractor.extract_descriptors` returns descriptors only. Used to
  refresh descriptor columns without re-running TensorFlow.
"""

from __future__ import annotations

import json
import logging
import urllib.request
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
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s", url)
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0)) or None
            try:
                from tqdm import tqdm

                bar_ctx = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
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
        if tmp.exists():
            tmp.unlink()
        raise
    tmp.rename(dest)


def ensure_effnet_model() -> Path:
    path = _model_cache_dir() / EFFNET_MODEL_FILENAME
    if not path.exists():
        _download(EFFNET_MODEL_URL, path)
    return path


def ensure_genre_head_model() -> Path:
    """Download (once) the 400-style classifier head that runs on top of the
    Effnet embeddings."""
    path = _model_cache_dir() / GENRE_HEAD_MODEL_FILENAME
    if not path.exists():
        _download(GENRE_HEAD_MODEL_URL, path)
    return path


def ensure_genre_labels() -> list[str]:
    """Return the 400 ``"Genre---Style"`` labels in the order produced by the
    classifier head. Cached on disk alongside the model."""
    path = _model_cache_dir() / GENRE_HEAD_LABELS_FILENAME
    if not path.exists():
        _download(GENRE_HEAD_LABELS_URL, path)
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
# Backend: MusicExtractor
# ---------------------------------------------------------------------------

_MUSIC_EXTRACTOR_KEYS: list[tuple[str, int]] = [
    ("lowlevel.spectral_centroid.mean", 1),
    ("lowlevel.spectral_centroid.stdev", 1),
    ("lowlevel.spectral_rolloff.mean", 1),
    ("lowlevel.spectral_rolloff.stdev", 1),
    ("lowlevel.spectral_flux.mean", 1),
    ("lowlevel.spectral_flux.stdev", 1),
    ("lowlevel.spectral_complexity.mean", 1),
    ("lowlevel.spectral_complexity.stdev", 1),
    ("lowlevel.zerocrossingrate.mean", 1),
    ("lowlevel.zerocrossingrate.stdev", 1),
    ("lowlevel.mfcc.mean", 13),
    ("lowlevel.mfcc.stdev", 13),
    ("tonal.hpcp.mean", 36),
    ("tonal.hpcp.stdev", 36),
    ("tonal.chords_strength.mean", 1),
    ("tonal.key_strength", 1),
    ("rhythm.bpm", 1),
    ("rhythm.danceability", 1),
    ("rhythm.onset_rate", 1),
    ("rhythm.beats_loudness.mean", 1),
    ("rhythm.beats_loudness.stdev", 1),
]


class MusicExtractorBackend:
    """Classical descriptor backend using Essentia's MusicExtractor."""

    name = "musicextractor-v1"

    def __init__(self) -> None:
        try:
            from essentia.standard import MusicExtractor
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "The 'musicextractor' backend requires essentia. "
                "Install with: pip install essentia"
            ) from e
        self._extractor = MusicExtractor(
            lowlevelStats=["mean", "stdev"],
            rhythmStats=["mean", "stdev"],
            tonalStats=["mean", "stdev"],
        )
        self.dim = sum(n for _, n in _MUSIC_EXTRACTOR_KEYS)

    @staticmethod
    def _get(pool, key, default=None):
        try:
            return pool[key]
        except KeyError:
            return default

    def _run(self, path: Path):
        return self._extractor(str(path))

    def _build_embedding(self, pool) -> np.ndarray:
        parts: list[np.ndarray] = []
        for key, expected in _MUSIC_EXTRACTOR_KEYS:
            val = self._get(pool, key, 0.0)
            arr = np.atleast_1d(np.asarray(val, dtype=np.float32))
            if arr.size < expected:
                arr = np.pad(arr, (0, expected - arr.size))
            elif arr.size > expected:
                arr = arr[:expected]
            parts.append(arr)
        emb = np.concatenate(parts).astype(np.float32, copy=False)
        return np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_descriptors(self, pool) -> Descriptors:
        d = Descriptors()
        bpm = self._get(pool, "rhythm.bpm")
        if bpm is not None:
            d.bpm = float(bpm)
        bpm_conf = self._get(pool, "rhythm.bpm_confidence")
        if bpm_conf is not None:
            d.bpm_confidence = float(bpm_conf)
        key = self._get(pool, "tonal.key_edma.key") or self._get(
            pool, "tonal.key_krumhansl.key"
        )
        if key is not None:
            d.key = str(key)
        scale = self._get(pool, "tonal.key_edma.scale") or self._get(
            pool, "tonal.key_krumhansl.scale"
        )
        if scale is not None:
            d.scale = str(scale)
        ks = self._get(pool, "tonal.key_edma.strength") or self._get(
            pool, "tonal.key_krumhansl.strength"
        )
        if ks is not None:
            d.key_strength = float(ks)
        loud = self._get(pool, "lowlevel.average_loudness")
        if loud is not None:
            d.loudness = float(loud)
        dance = self._get(pool, "rhythm.danceability")
        if dance is not None:
            d.danceability = float(dance)
        onset = self._get(pool, "rhythm.onset_rate")
        if onset is not None:
            d.onset_rate = float(onset)
        return d

    def extract(self, path: Path) -> TrackFeatures:
        features, _ = self._run(path)
        emb = self._build_embedding(features)
        descriptors = self._build_descriptors(features)
        duration = float(self._get(features, "metadata.audio_properties.length", 0.0))
        return TrackFeatures(
            embedding=emb, duration=duration, model=self.name, descriptors=descriptors
        )

    def extract_descriptors(self, path: Path) -> tuple[Descriptors, float]:
        features, _ = self._run(path)
        descriptors = self._build_descriptors(features)
        duration = float(self._get(features, "metadata.audio_properties.length", 0.0))
        return descriptors, duration


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

Extractor = EffnetExtractor  # for type hints; both classes have the same API


@dataclass(frozen=True)
class BackendInfo:
    """Lightweight metadata about a backend: just the model name and the
    embedding dimensionality. Reading these doesn't import essentia,
    download model files, or load TensorFlow.
    """

    name: str
    dim: int


def get_backend_info(backend: str = "effnet") -> BackendInfo:
    """Return :class:`BackendInfo` for the given backend without
    instantiating the extractor. The TF graph and model files are loaded
    only inside worker processes on first scan.
    """
    backend = backend.lower()
    if backend in ("effnet", "discogs-effnet", "tf"):
        return BackendInfo(name=EffnetExtractor.name, dim=EFFNET_EMBEDDING_DIM)
    if backend in ("musicextractor", "classic", "music"):
        return BackendInfo(
            name=MusicExtractorBackend.name,
            dim=sum(n for _, n in _MUSIC_EXTRACTOR_KEYS),
        )
    raise ValueError(f"unknown backend: {backend}")


def get_extractor(backend: str = "effnet"):
    backend = backend.lower()
    if backend in ("effnet", "discogs-effnet", "tf"):
        return EffnetExtractor()
    if backend in ("musicextractor", "classic", "music"):
        return MusicExtractorBackend()
    raise ValueError(f"unknown backend: {backend}")


# ---------------------------------------------------------------------------
# File signature
# ---------------------------------------------------------------------------


def file_signature(path: Path) -> tuple[int, float]:
    """(size, mtime) — cheap change detection."""
    import os

    st = os.stat(path)
    return st.st_size, st.st_mtime
