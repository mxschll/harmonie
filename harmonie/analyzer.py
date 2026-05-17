"""Scan orchestration and scheduler."""

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
from .features import get_backend_info
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
    phase: str = "idle"  # idle | enumerating | classifying | extracting | pruning
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    last_duration_sec: Optional[float] = None
    last_error: Optional[str] = None

    discovered: int = 0
    full: int = 0
    descriptors_only: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0

    failures: list[tuple[str, str]] = field(default_factory=list)

    def snapshot(self) -> dict:
        return {
            "state": self.state,
            "phase": self.phase,
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
        # Backend metadata only — workers load the actual model.
        info = get_backend_info(settings.backend)
        self.model_name: str = info.name
        self.embedding_dim: int = info.dim
        self.pool: Optional[WorkerPool] = None
        self.status = ScanStatus()
        self._scan_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self.pool is None:
            self.pool = WorkerPool(
                backend=self.settings.backend,
                workers=self.settings.worker_count,
                log_level=self.settings.log_level,
            )

    def stop(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool = None
        self.db.close()

    # -- scan ----------------------------------------------------------

    def scan(self, *, force: bool = False) -> ScanStatus:
        """Run one scan synchronously. Safe to call from any thread; the
        internal lock prevents overlap. Returns the current status without
        starting a new scan if one is already running."""
        if not self._scan_lock.acquire(blocking=False):
            logger.info("scan already in progress; skipping new request")
            return self.status

        try:
            self._run_scan(force=force)
        finally:
            self._scan_lock.release()
        return self.status

    def _run_scan(self, *, force: bool) -> None:
        self.status = ScanStatus(
            state="scanning", phase="enumerating", started_at=time.time(),
        )
        t0 = time.monotonic()

        libs = [Path(p) for p in self.settings.libraries]
        if not libs:
            logger.warning("no libraries configured (HARMONIE_LIBRARIES is empty)")
        # Only prune entries that live under reachable roots so a flaky
        # NAS doesn't wipe the index when the mount is unavailable.
        reachable: list[Path] = []
        unreachable: list[Path] = []
        for p in libs:
            if Path(p).expanduser().exists():
                reachable.append(p)
            else:
                unreachable.append(p)
        for p in unreachable:
            logger.warning("library root unreachable, skipping: %s", p)

        # Enumeration: update self.status.discovered on each yield, log
        # every 10 seconds.
        if reachable:
            logger.info(
                "scanning libraries: %s",
                ", ".join(str(p) for p in reachable),
            )
        files: list[Path] = []
        last_progress = time.monotonic()
        for f in iter_audio_files(reachable):
            files.append(f)
            self.status.discovered = len(files)
            now = time.monotonic()
            if now - last_progress > 10:
                logger.info(
                    "enumerating: %d audio file(s) found so far...", len(files),
                )
                last_progress = now
        logger.info("discovered %d audio file(s)", len(files))

        # Classification: one stat() per file plus a DB lookup. Periodic
        # progress on the same 10-second cadence as enumeration.
        self.status.phase = "classifying"
        classify_state = {"last": time.monotonic()}

        def _classify_progress(n: int) -> None:
            now = time.monotonic()
            if now - classify_state["last"] > 10:
                logger.info(
                    "classifying: %d / %d file(s) checked...",
                    n, len(files),
                )
                classify_state["last"] = now

        full_jobs, desc_jobs, skipped = build_jobs(
            self.db, files,
            model_name=self.model_name,
            force=force,
            on_progress=_classify_progress,
        )
        self.status.skipped = skipped
        logger.info(
            "jobs: full=%d, descriptors_only=%d, skipped=%d",
            len(full_jobs), len(desc_jobs), skipped,
        )

        # Worker pool starts only when there's work to dispatch.
        all_jobs: list = list(full_jobs) + list(desc_jobs)
        if all_jobs:
            self.status.phase = "extracting"
            if self.pool is None:
                self.start()
            assert self.pool is not None
            for result in self.pool.map(all_jobs, chunksize=1):
                self._handle_result(result, reachable_roots=reachable)

        # Prune rows for files that disappeared, scoped to the roots we
        # actually walked.
        if reachable:
            self.status.phase = "pruning"
            present = {str(f) for f in files}
            removed = self.db.prune_missing_under_roots(
                roots=reachable, keep=present
            )
            self.status.removed = removed
            if removed:
                logger.info("pruned %d removed track(s)", removed)
        else:
            logger.warning(
                "no reachable libraries this scan; skipping prune"
            )

        elapsed = time.monotonic() - t0
        # Drop cached embedding matrices; the next query rebuilds them.
        self.index.invalidate()
        self.status.state = "idle"
        self.status.phase = "idle"
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
                # library_root and relative_path are not refreshed on the
                # descriptor-only path — a path change forces full
                # extraction.
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
