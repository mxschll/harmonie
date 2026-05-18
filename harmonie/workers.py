"""Multiprocessing pool for analysis. One model load per worker process."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import numpy as np

from .features import (
    DESCRIPTOR_VERSION,
    Descriptors,
    EffnetExtractor,
    TrackFeatures,
    file_signature,
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
    # Full 400-d Discogs activation vector and the top-K (label, prob)
    # preview. Both ``None`` if the genre head was unavailable at
    # extraction time.
    style_activations: np.ndarray | None = None
    top_styles: list[tuple[str, float]] | None = None


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


# Anything a worker can produce for one job.
WorkerResult = Union[FullResult, DescriptorResult, JobError]


# ---------------------------------------------------------------------------
# Worker globals (per process)
# ---------------------------------------------------------------------------

_extractor = None


def _worker_init(log_level: str = "INFO") -> None:
    """Run once per worker process. Loads the extractor, configures logging,
    and (for non-DEBUG levels) silences Essentia's
    ``[WARNING] No network created, or last created network has been
    deleted`` line that fires when the standard-mode TF wrapper destroys
    its internal streaming network.
    """
    global _extractor
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    # Spawn-mode workers don't inherit Python state from the parent, so
    # logging has to be configured here. Format constants are shared
    # with :func:`harmonie.config.configure_logging`. ``force=True``
    # replaces any handler an imported library may have attached so our
    # format wins.
    from .config import LOG_DATEFMT, LOG_FORMAT

    logging.basicConfig(
        level=log_level.upper(),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,
    )

    if log_level.upper() != "DEBUG":
        try:
            import essentia  # type: ignore[import-not-found]

            essentia.log.warningActive = False
        except Exception:  # pragma: no cover - essentia not installed
            pass

    _extractor = EffnetExtractor()


def _do_full(job: FullJob) -> WorkerResult:
    assert _extractor is not None
    logger.info("extracting: %s", job.path)
    try:
        feats: TrackFeatures = _extractor.extract(Path(job.path))
        styles: list[tuple[str, float]] | None = None
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


def _do_descriptors(job: DescriptorJob) -> WorkerResult:
    assert _extractor is not None
    logger.info("refreshing descriptors: %s", job.path)
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


def _dispatch(job: FullJob | DescriptorJob) -> WorkerResult:
    if isinstance(job, FullJob):
        return _do_full(job)
    return _do_descriptors(job)


# ---------------------------------------------------------------------------
# Pool wrapper
# ---------------------------------------------------------------------------


class WorkerPool:
    """Wrapper around :class:`multiprocessing.Pool` that streams results
    back via ``imap_unordered``."""

    def __init__(
        self,
        *,
        workers: int,
        log_level: str = "INFO",
    ) -> None:
        self.workers = max(1, workers)
        # 'spawn' avoids fork-after-thread issues with TensorFlow.
        ctx = mp.get_context("spawn")
        self._pool: mp.pool.Pool | None = ctx.Pool(
            processes=self.workers,
            initializer=_worker_init,
            initargs=(log_level,),
        )
        logger.info("started worker pool: %d workers", self.workers)

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
        """Hard-stop all workers (SIGTERM). Use when the user has asked
        to abort: ``close()`` would otherwise wait for in-flight kernel
        I/O (slow CIFS reads, etc.) to finish."""
        if self._pool is None:
            return
        self._pool.terminate()
        self._pool.join()
        self._pool = None


# ---------------------------------------------------------------------------
# Job-building helpers
# ---------------------------------------------------------------------------


def build_jobs(
    db,
    files: list[Path],
    *,
    model_name: str,
    force: bool,
    on_progress: Callable[[int], None] | None = None,
) -> tuple[list[FullJob], list[DescriptorJob], int]:
    """Decide which files need a full analysis vs. a descriptor refresh.

    Returns ``(full_jobs, descriptor_jobs, skipped_count)``. Files that
    don't exist or can't be stat'd are silently dropped. ``on_progress``
    is invoked with the running count after every file.
    """
    full_jobs: list[FullJob] = []
    desc_jobs: list[DescriptorJob] = []
    skipped = 0
    for i, f in enumerate(files, start=1):
        try:
            size, mtime = file_signature(f)
        except FileNotFoundError:
            if on_progress is not None:
                on_progress(i)
            continue
        path_str = str(f)
        if force or db.needs_embedding(path_str, size, mtime, model_name):
            full_jobs.append(FullJob(path=path_str, size=size, mtime=mtime))
        elif db.needs_descriptor_refresh(path_str, DESCRIPTOR_VERSION):
            desc_jobs.append(DescriptorJob(path=path_str))
        else:
            skipped += 1
        if on_progress is not None:
            on_progress(i)
    return full_jobs, desc_jobs, skipped
