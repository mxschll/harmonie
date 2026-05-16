"""Multiprocessing pool for analysis. One model load per worker process."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .features import (
    DESCRIPTOR_VERSION,
    Descriptors,
    TrackFeatures,
    file_signature,
    get_extractor,
    top_styles,
)
from .tags import Tags, extract_tags

logger = logging.getLogger("harmonie.workers")


# ---------------------------------------------------------------------------
# Job + result types (picklable)
# ---------------------------------------------------------------------------


@dataclass
class FullJob:
    """Compute embedding + descriptors."""
    path: str
    size: int
    mtime: float


@dataclass
class DescriptorJob:
    """Compute descriptors only (top-up an existing row)."""
    path: str


@dataclass
class FullResult:
    path: str
    size: int
    mtime: float
    embedding: np.ndarray
    duration: float
    model: str
    descriptors: Descriptors
    descriptor_version: int
    tags: Tags
    # Optional Discogs-400 style data. Worker computes both the full
    # activation vector (stored as BLOB) and a top-K (label, prob) preview
    # (stored as rows in track_styles). Both ``None`` if the genre head was
    # unavailable or the backend doesn't produce Effnet-compatible embeddings.
    style_activations: Optional[np.ndarray] = None
    top_styles: Optional[list[tuple[str, float]]] = None


@dataclass
class DescriptorResult:
    path: str
    descriptors: Descriptors
    duration: float
    descriptor_version: int
    tags: Tags


@dataclass
class JobError:
    path: str
    error: str


Result = "FullResult | DescriptorResult | JobError"  # for documentation


# ---------------------------------------------------------------------------
# Worker globals (per process)
# ---------------------------------------------------------------------------

_extractor = None
_backend_name = ""


def _worker_init(backend: str) -> None:
    """Run once per worker process. Loads the model into a process global."""
    global _extractor, _backend_name
    # Quiet TF in workers.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    _backend_name = backend
    _extractor = get_extractor(backend)


def _do_full(job: FullJob) -> Result:
    assert _extractor is not None
    try:
        feats: TrackFeatures = _extractor.extract(Path(job.path))
        # Build the top-K preview here so the analyzer just splats it into
        # the DB without needing to know about labels.
        styles: Optional[list[tuple[str, float]]] = None
        labels = getattr(_extractor, "genre_labels", None)
        if feats.style_activations is not None and labels:
            styles = top_styles(feats.style_activations, labels)
        return FullResult(
            path=job.path,
            size=job.size,
            mtime=job.mtime,
            embedding=feats.embedding,
            duration=feats.duration,
            model=feats.model,
            descriptors=feats.descriptors,
            descriptor_version=DESCRIPTOR_VERSION,
            tags=extract_tags(Path(job.path)),
            style_activations=feats.style_activations,
            top_styles=styles,
        )
    except Exception as e:
        return JobError(path=job.path, error=repr(e))


def _do_descriptors(job: DescriptorJob) -> Result:
    assert _extractor is not None
    try:
        descriptors, duration = _extractor.extract_descriptors(Path(job.path))
        return DescriptorResult(
            path=job.path,
            descriptors=descriptors,
            duration=duration,
            descriptor_version=DESCRIPTOR_VERSION,
            tags=extract_tags(Path(job.path)),
        )
    except Exception as e:
        return JobError(path=job.path, error=repr(e))


def _dispatch(job: FullJob | DescriptorJob) -> Result:
    if isinstance(job, FullJob):
        return _do_full(job)
    return _do_descriptors(job)


# ---------------------------------------------------------------------------
# Pool wrapper
# ---------------------------------------------------------------------------


class WorkerPool:
    """Thin wrapper around multiprocessing.Pool that streams results back via
    ``imap_unordered`` so the orchestrator can write to the DB as work
    completes rather than waiting for the whole batch."""

    def __init__(self, *, backend: str, workers: int) -> None:
        self.backend = backend
        self.workers = max(1, workers)
        # 'spawn' avoids fork-after-thread issues with TensorFlow.
        ctx = mp.get_context("spawn")
        self._pool: Optional[mp.pool.Pool] = ctx.Pool(
            processes=self.workers,
            initializer=_worker_init,
            initargs=(backend,),
        )
        logger.info(
            "started worker pool: %d workers, backend=%s", self.workers, backend
        )

    def map(self, jobs: list[FullJob | DescriptorJob], *, chunksize: int = 1):
        if self._pool is None:
            raise RuntimeError("pool is closed")
        # imap_unordered yields results as they complete in any order.
        yield from self._pool.imap_unordered(_dispatch, jobs, chunksize=chunksize)

    def close(self) -> None:
        if self._pool is None:
            return
        self._pool.close()
        self._pool.join()
        self._pool = None

    def terminate(self) -> None:
        if self._pool is None:
            return
        self._pool.terminate()
        self._pool.join()
        self._pool = None

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Job-building helpers
# ---------------------------------------------------------------------------


def build_jobs(
    db, files: list[Path], *, model_name: str, force: bool
) -> tuple[list[FullJob], list[DescriptorJob], int]:
    """Decide which files need a full analysis vs. just a descriptor refresh.

    Returns (full_jobs, descriptor_jobs, skipped_count). Files that don't exist
    or can't be stat'd are silently dropped.
    """
    full_jobs: list[FullJob] = []
    desc_jobs: list[DescriptorJob] = []
    skipped = 0
    for f in files:
        try:
            size, mtime = file_signature(f)
        except FileNotFoundError:
            continue
        path_str = str(f)
        if force or db.needs_embedding(path_str, size, mtime, model_name):
            full_jobs.append(FullJob(path=path_str, size=size, mtime=mtime))
        elif db.needs_descriptor_refresh(path_str, DESCRIPTOR_VERSION):
            desc_jobs.append(DescriptorJob(path=path_str))
        else:
            skipped += 1
    return full_jobs, desc_jobs, skipped
