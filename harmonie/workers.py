"""Multiprocessing pool for analysis. One model load per worker process."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

import numpy as np

from .features import (
    DESCRIPTOR_VERSION,
    Descriptors,
    EffnetExtractor,
    TrackFeatures,
    file_fingerprint,
    file_signature,
    prefetch_models,
    top_styles,
)
from .gpu import cuda_device_count
from .tags import Tags, extract_tags

logger = logging.getLogger("harmonie.workers")

# How often the result loop wakes to harvest finished jobs, feed idle workers,
# and notice a cancel or a timed-out job.
RESULT_POLL_SEC = 0.1

# How many jobs may be queued or running at once. Each worker keeps one job
# waiting behind the one it is running, so a worker that finishes between
# polls does not sit idle until the next poll.
INFLIGHT_PER_WORKER = 2

# Long enough to load the model on a slow disk, short enough not to stall a
# scan when a GPU wedges.
GPU_PROBE_TIMEOUT_SEC = 180


def _fingerprint_or_none(path: str, size: int) -> str | None:
    """A file that vanished mid-scan is not worth failing the result for."""
    try:
        return file_fingerprint(Path(path), size)
    except OSError:
        return None


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
    # Computed here rather than by the caller so it describes the same bytes
    # the embedding was extracted from.
    fingerprint: str | None
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

# Shared with the parent; see WorkerPool._spawn_pool. A worker sets slot i to 1
# when it starts the job submitted with ticket i. Written with a plain store so
# that a worker killed mid-job cannot leave a lock behind the way a
# multiprocessing.Queue can.
_started = None


def _worker_init(log_level: str = "INFO", started=None) -> None:
    """Run once per worker process. Loads the extractor, configures logging,
    and (for non-DEBUG levels) silences Essentia's
    ``[WARNING] No network created, or last created network has been
    deleted`` line that fires when the standard-mode TF wrapper destroys
    its internal streaming network.
    """
    global _extractor, _started
    _started = started
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # One worker per core, each its own process with its own TensorFlow: on a
    # GPU host the first would otherwise reserve nearly all of its memory and
    # the rest would fail. Ignored when there is no GPU.
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

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
            fingerprint=_fingerprint_or_none(job.path, job.size),
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


def _dispatch(job: FullJob | DescriptorJob, ticket: int = -1) -> WorkerResult:
    if _started is not None and 0 <= ticket < len(_started):
        _started[ticket] = 1
    if isinstance(job, FullJob):
        return _do_full(job)
    return _do_descriptors(job)


# ---------------------------------------------------------------------------
# Pool wrapper
# ---------------------------------------------------------------------------


def _gpu_usable() -> bool:
    """Try one throwaway inference with the GPU visible.

    TensorFlow treats a visible-but-unusable CUDA device as fatal: it aborts the
    process rather than falling back, which would kill every analysis worker and
    leave a scan making no progress. The probe therefore has to run somewhere
    expendable — its own process.
    """
    code = (
        "import numpy as np;"
        "from harmonie.features import EffnetExtractor;"
        "EffnetExtractor(load_genre_head=False)._model("
        "np.zeros(16000 * 2, dtype='float32'))"
    )
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=GPU_PROBE_TIMEOUT_SEC,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.debug("gpu probe could not run: %s", exc)
        return False
    return proc.returncode == 0


def _configure_gpu() -> None:
    """Use a visible CUDA device for inference, or make sure we do not try."""
    devices = cuda_device_count()
    if not devices:
        return
    if not _gpu_usable():
        logger.warning(
            "%d CUDA device(s) visible but TensorFlow cannot use them; "
            "continuing on CPU",
            devices,
        )
        # Workers inherit this, so they see no GPU and stay on the CPU path.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    logger.info(
        "%d CUDA device(s) visible; running inference on GPU (experimental). "
        "Workers share the card, so keep HARMONIE_WORKERS low",
        devices,
    )


class WorkerPool:
    """Wrapper around :class:`multiprocessing.Pool` that streams results back
    as workers finish and enforces a per-job timeout, so one file that wedges
    the native decoder cannot stall the whole scan."""

    def __init__(
        self,
        *,
        workers: int,
        log_level: str = "INFO",
        should_stop: Callable[[], bool] | None = None,
        job_timeout: float | None = None,
    ) -> None:
        self.workers = max(1, workers)
        self.job_timeout = job_timeout
        self._log_level = log_level
        self._lock = threading.Lock()
        self._stopped = False

        # Fetch the models here, in the parent. Each worker builds its own
        # extractor, so leaving this to them means one download per worker
        # against a host that does not enjoy the attention.
        logger.info("preparing analysis models (first run downloads about 20 MB)")
        prefetch_models(should_stop)

        # 'spawn' avoids fork-after-thread issues with TensorFlow.
        self._ctx = mp.get_context("spawn")
        _configure_gpu()
        logger.info(
            "starting %d worker(s); each loads the models into memory, "
            "which takes a moment",
            self.workers,
        )
        self._started = self._new_started()
        self._pool: mp.pool.Pool | None = self._spawn_pool()
        logger.info("started worker pool: %d workers", self.workers)

    def _new_started(self):
        """One byte per in-flight ticket; see :func:`_dispatch`."""
        return self._ctx.RawArray("b", INFLIGHT_PER_WORKER * self.workers)

    def _spawn_pool(self) -> mp.pool.Pool:
        return self._ctx.Pool(
            processes=self.workers,
            initializer=_worker_init,
            initargs=(self._log_level, self._started),
        )

    def _restart_pool(self) -> bool:
        """Replace the pool after a worker wedged. terminate() takes the whole
        pool down, since a single worker cannot be killed on its own.

        Returns False if the pool was stopped meanwhile, in which case the
        replacement is torn down again rather than left running unowned.
        """
        with self._lock:
            old, self._pool = self._pool, None
        if old is not None:
            with contextlib.suppress(Exception):
                old.terminate()
                old.join()
        self._started = self._new_started()
        fresh = self._spawn_pool()
        with self._lock:
            if not self._stopped:
                self._pool = fresh
                return True
        with contextlib.suppress(Exception):
            fresh.terminate()
            fresh.join()
        return False

    def map(
        self,
        jobs: list[FullJob | DescriptorJob],
        *,
        should_stop: Callable[[], bool] | None = None,
    ):
        """Stream results as workers finish, one job per task.

        At most ``INFLIGHT_PER_WORKER * workers`` jobs are submitted at a
        time. Each gets a ticket, and the worker flags the ticket's slot in
        ``_started`` when it picks the job up, so the timeout clock runs only
        while the job is actually executing: time spent queued behind another
        job, or waiting for a worker to finish loading its models, does not
        count.

        A job that runs past :attr:`job_timeout` is reported as a
        :class:`JobError`, the pool is torn down to kill the worker stuck in
        native code, and every other job that was in flight is requeued onto
        a fresh pool.

        There is deliberately no chunksize option: one job per task is what
        lets a single bad file be pinned to a single result and timed out.
        """
        if self._pool is None:
            raise RuntimeError("pool is closed")
        # Submitting first and checking afterwards would queue the whole
        # library into a pool the scan has already abandoned.
        if should_stop is not None and should_stop():
            return

        pending = deque(jobs)
        inflight: dict[int, tuple[mp.pool.AsyncResult, FullJob | DescriptorJob]] = {}
        started_at: dict[int, float] = {}

        while pending or inflight:
            pool = self._pool
            if pool is None or (should_stop is not None and should_stop()):
                return

            while pending and len(inflight) < len(self._started):
                ticket = next(t for t in range(len(self._started)) if t not in inflight)
                job = pending.popleft()
                self._started[ticket] = 0
                inflight[ticket] = (pool.apply_async(_dispatch, (job, ticket)), job)

            now = time.monotonic()
            for ticket in inflight:
                if ticket not in started_at and self._started[ticket]:
                    started_at[ticket] = now

            finished = [t for t, (ar, _) in inflight.items() if ar.ready()]
            for ticket in finished:
                ar, job = inflight.pop(ticket)
                started_at.pop(ticket, None)
                yield self._collect(ar, job)

            if self.job_timeout is not None:
                overran = [
                    t
                    for t, started in started_at.items()
                    if t in inflight and now - started > self.job_timeout
                ]
                if overran:
                    yield from self._recover(overran, inflight, started_at, pending)
                    if should_stop is not None and should_stop():
                        return
                    if not self._restart_pool():
                        return
                    continue

            if not finished:
                time.sleep(RESULT_POLL_SEC)

    @staticmethod
    def _collect(ar, job) -> WorkerResult:
        try:
            return ar.get()
        except Exception as exc:
            # _do_full/_do_descriptors return JobError on failure, so this is
            # a result the worker could not pickle, or an error outside their
            # try (e.g. the extractor never loaded).
            return JobError(path=job.path, error=repr(exc))

    def _recover(self, overran, inflight, started_at, pending):
        """Fail the overran jobs, harvest whatever else already finished, and
        requeue the rest. The pool is dead after this; the caller restarts it."""
        secs = f"{self.job_timeout:g}"
        for ticket in overran:
            _, job = inflight.pop(ticket)
            started_at.pop(ticket, None)
            logger.warning(
                "extraction exceeded %ss timeout, terminating worker: %s",
                secs,
                job.path,
            )
            yield JobError(
                path=job.path,
                error=f"extraction exceeded {secs}s timeout and was terminated",
            )
        # Yielding above let the consumer run; jobs may have finished since.
        requeue = []
        for ar, job in inflight.values():
            if ar.ready():
                yield self._collect(ar, job)
            else:
                requeue.append(job)
        inflight.clear()
        started_at.clear()
        if requeue:
            logger.info("requeueing %d job(s) interrupted by the restart", len(requeue))
            pending.extendleft(reversed(requeue))

    def close(self) -> None:
        with self._lock:
            self._stopped = True
            pool, self._pool = self._pool, None
        if pool is None:
            return
        pool.close()
        pool.join()

    def terminate(self) -> None:
        """Hard-stop all workers (SIGTERM). Use when the user has asked
        to abort: ``close()`` would otherwise wait for in-flight kernel
        I/O (slow CIFS reads, etc.) to finish."""
        with self._lock:
            self._stopped = True
            pool, self._pool = self._pool, None
        if pool is None:
            return
        pool.terminate()
        pool.join()


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
    relocate: Callable[[str, int, float], bool] | None = None,
) -> tuple[list[FullJob], list[DescriptorJob], int]:
    """Decide which files need a full analysis vs. a descriptor refresh.

    Returns ``(full_jobs, descriptor_jobs, skipped_count)``. Files that
    don't exist or can't be stat'd are silently dropped. ``on_progress``
    is invoked with the running count after every file. ``relocate``, when
    given, is offered each file before the decision, so a library that moved
    is matched to its existing rows instead of analysed from scratch.
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
        if relocate is not None:
            relocate(path_str, size, mtime)
        if force or db.needs_embedding(path_str, size, mtime, model_name):
            full_jobs.append(FullJob(path=path_str, size=size, mtime=mtime))
        elif db.needs_descriptor_refresh(path_str, DESCRIPTOR_VERSION):
            desc_jobs.append(DescriptorJob(path=path_str))
        else:
            skipped += 1
        if on_progress is not None:
            on_progress(i)
    return full_jobs, desc_jobs, skipped
