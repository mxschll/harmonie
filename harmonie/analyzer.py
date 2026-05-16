"""Scan orchestration and scheduler.

The :class:`Analyzer` owns the worker pool and the database for the lifetime
of the service. It runs scans on demand or on a schedule, with a single-run
mutex so concurrent triggers (HTTP /scan + scheduler tick at the same time)
don't double up.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import Settings
from .db import Database
from .features import get_extractor
from .index import EmbeddingIndex
from .scan import iter_audio_files, split_library_path
from .workers import (
    DescriptorResult,
    FullResult,
    JobError,
    WorkerPool,
    build_jobs,
)

logger = logging.getLogger("harmonie.analyzer")


# ---------------------------------------------------------------------------
# Status types
# ---------------------------------------------------------------------------


@dataclass
class ScanStatus:
    state: str = "idle"  # idle | scanning
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    last_duration_sec: Optional[float] = None
    last_error: Optional[str] = None

    discovered: int = 0       # files found by walker
    full: int = 0             # full extractions (model + descriptors)
    descriptors_only: int = 0 # descriptor top-ups
    skipped: int = 0          # already up-to-date
    failed: int = 0
    removed: int = 0          # rows pruned because file vanished

    failures: list[tuple[str, str]] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "last_duration_sec": self.last_duration_sec,
            "last_error": self.last_error,
            "discovered": self.discovered,
            "full": self.full,
            "descriptors_only": self.descriptors_only,
            "skipped": self.skipped,
            "failed": self.failed,
            "removed": self.removed,
            "recent_failures": self.failures[-20:],
        }


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class Analyzer:
    """Owns the DB connection and worker pool for the service lifetime."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.db_path)
        self.index = EmbeddingIndex(self.db)
        # Resolve the model name without holding the model itself in this
        # process — workers will load it independently.
        sample = get_extractor(settings.backend)
        self.model_name: str = sample.name
        self.embedding_dim: int = sample.dim
        del sample
        self.pool: Optional[WorkerPool] = None
        self.status = ScanStatus()
        self._scan_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self.pool is None:
            self.pool = WorkerPool(
                backend=self.settings.backend,
                workers=self.settings.worker_count,
            )

    def stop(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        self.db.close()

    # -- scan ----------------------------------------------------------

    def scan(self, *, force: bool = False) -> ScanStatus:
        """Run one scan synchronously. Safe to call from any thread; the
        internal lock prevents overlap. If a scan is already running, this
        call returns its current status without starting a new one."""
        if not self._scan_lock.acquire(blocking=False):
            logger.info("scan already in progress; skipping new request")
            return self.status

        try:
            self._run_scan(force=force)
        finally:
            self._scan_lock.release()
        return self.status

    def _run_scan(self, *, force: bool) -> None:
        if self.pool is None:
            self.start()
        assert self.pool is not None

        self.status = ScanStatus(state="scanning", started_at=time.time())
        t0 = time.monotonic()

        libs = [Path(p) for p in self.settings.libraries]
        if not libs:
            logger.warning("no libraries configured (HARMONIE_LIBRARIES is empty)")
        # Separate reachable roots from missing ones. We will only prune
        # entries that live under reachable roots — a flaky NAS shouldn't
        # wipe the index just because the mount is down right now.
        reachable: list[Path] = []
        unreachable: list[Path] = []
        for p in libs:
            if Path(p).expanduser().exists():
                reachable.append(p)
            else:
                unreachable.append(p)
        for p in unreachable:
            logger.warning("library root unreachable, skipping: %s", p)

        files = list(iter_audio_files(reachable))
        self.status.discovered = len(files)
        logger.info("discovered %d audio file(s)", len(files))

        full_jobs, desc_jobs, skipped = build_jobs(
            self.db, files, model_name=self.model_name, force=force
        )
        self.status.skipped = skipped
        logger.info(
            "jobs: full=%d, descriptors_only=%d, skipped=%d",
            len(full_jobs), len(desc_jobs), skipped,
        )

        # Send full jobs first (they're slower) so descriptor top-ups can
        # benefit from spare workers later.
        all_jobs: list = list(full_jobs) + list(desc_jobs)
        if all_jobs:
            for result in self.pool.map(all_jobs, chunksize=1):
                self._handle_result(result, reachable_roots=reachable)

        # Drop rows for files that disappeared, scoped to roots we actually
        # walked. Skip pruning entirely if no roots were reachable.
        if reachable:
            present = {str(f) for f in files}
            removed = self.db.prune_missing_under_roots(
                roots=reachable, keep=present
            )
            self.status.removed = removed
            if removed:
                logger.info("pruned %d removed track(s)", removed)
        else:
            logger.warning(
                "no reachable libraries this scan; skipping prune to protect "
                "the index"
            )

        elapsed = time.monotonic() - t0
        # Drop cached embedding matrices so the next query rebuilds with the
        # newly-written rows (or skipped rows that just had descriptors
        # refreshed — those don't change embeddings, but invalidating
        # everything is simpler and the rebuild is cheap).
        self.index.invalidate()
        self.status.state = "idle"
        self.status.finished_at = time.time()
        self.status.last_duration_sec = elapsed
        logger.info(
            "scan complete in %.1fs: full=%d, descriptors_only=%d, skipped=%d, "
            "failed=%d, removed=%d",
            elapsed, self.status.full, self.status.descriptors_only,
            self.status.skipped, self.status.failed, self.status.removed,
        )

    def _handle_result(self, result, *, reachable_roots: list[Path]) -> None:
        if isinstance(result, FullResult):
            try:
                lib_root, rel_path = split_library_path(
                    result.path, reachable_roots
                )
                self.db.upsert_track(
                    path=result.path,
                    size=result.size,
                    mtime=result.mtime,
                    duration=result.duration,
                    embedding=result.embedding,
                    model=result.model,
                    descriptors=result.descriptors,
                    descriptor_version=result.descriptor_version,
                    tags=result.tags,
                    library_root=lib_root,
                    relative_path=rel_path,
                    style_activations=result.style_activations,
                    top_styles=result.top_styles,
                )
                self.status.full += 1
            except Exception as e:  # pragma: no cover
                logger.exception("failed to persist full result for %s", result.path)
                self.status.failed += 1
                self.status.failures.append((result.path, repr(e)))

        elif isinstance(result, DescriptorResult):
            try:
                # Note: library_root/relative_path are not refreshed by the
                # descriptor-only path. They were set when the row was first
                # inserted and only change if the file path itself changes,
                # which would have triggered a full re-extraction.
                self.db.update_descriptors(
                    result.path,
                    descriptors=result.descriptors,
                    descriptor_version=result.descriptor_version,
                    duration=result.duration,
                    tags=result.tags,
                )
                self.status.descriptors_only += 1
            except Exception as e:  # pragma: no cover
                logger.exception(
                    "failed to persist descriptor refresh for %s", result.path
                )
                self.status.failed += 1
                self.status.failures.append((result.path, repr(e)))

        elif isinstance(result, JobError):
            self.status.failed += 1
            self.status.failures.append((result.path, result.error))
            logger.warning("extraction failed for %s: %s", result.path, result.error)

        else:  # pragma: no cover
            logger.error("unknown worker result type: %r", type(result))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


async def scheduler_loop(analyzer: Analyzer, settings: Settings) -> None:
    """Periodically trigger a scan. Cancel-safe."""
    if settings.scan_interval_hours <= 0 and not settings.scan_on_startup:
        logger.info("scheduler disabled (no startup scan, interval=0)")
        return

    if settings.scan_on_startup:
        logger.info("running startup scan")
        await asyncio.to_thread(analyzer.scan)

    if settings.scan_interval_hours <= 0:
        logger.info("scheduled scans disabled (interval=0)")
        return

    interval = settings.scan_interval_hours * 3600
    logger.info(
        "scheduler running; next scan in %.1f hour(s)", settings.scan_interval_hours
    )
    while True:
        try:
            await asyncio.sleep(interval)
            logger.info("triggering scheduled scan")
            await asyncio.to_thread(analyzer.scan)
        except asyncio.CancelledError:
            logger.info("scheduler cancelled")
            raise
        except Exception:  # pragma: no cover
            logger.exception("scheduled scan failed")
