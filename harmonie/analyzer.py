"""Scan orchestration and scheduler."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .config import Settings
from .db import Database
from .features import (
    DESCRIPTOR_VERSION,
    EMBEDDING_DIM,
    MODEL_NAME,
    DownloadCancelled,
    file_fingerprint,
)
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


class _ScanCancelled(Exception):
    """Raised internally by :meth:`Analyzer._run_scan` when the cancel
    flag is set. Caught by ``_run_scan`` itself to record the cancelled
    state in the scans table; never propagates out."""


# String written to scans.last_error when a scan is cancelled by the user.
_CANCELLED_REASON = "cancelled by user"

# How long stop() waits for a cancelled scan to finish recording its outcome.
SHUTDOWN_WAIT_SEC = 30


# ---------------------------------------------------------------------------
# Status types
# ---------------------------------------------------------------------------


@dataclass
class ScanStatus:
    state: str = "idle"  # idle | scanning
    phase: str = "idle"  # idle | enumerating | classifying | extracting | pruning
    started_at: float | None = None
    finished_at: float | None = None
    last_duration_sec: float | None = None
    last_error: str | None = None

    discovered: int = 0
    full: int = 0
    descriptors_only: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0

    failures: list[tuple[str, str]] = field(default_factory=list)
    # Persistent scan-history row id, set when _run_scan starts. None
    # outside of an active scan.
    scan_id: int | None = None

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


def _fingerprint_or_none(path: str, size: int) -> str | None:
    """Fingerprint for storage. A file that vanished between analysis and this
    call is not worth failing the whole result for."""
    try:
        return file_fingerprint(Path(path), size)
    except OSError as exc:
        logger.debug("could not fingerprint %s: %s", path, exc)
        return None


class _LibraryRelocator:
    """Matches files to index rows left behind by a library that moved.

    Analysis is keyed on absolute path, so a library remounted somewhere else
    (a native install switching to a container, say) looks entirely new: every
    track would be analysed again and the old rows would linger, since pruning
    only touches paths under the roots it scanned.

    Two ways to recognise the same file, cheapest first:

    1. Library-relative path, with size and mtime agreeing. Costs nothing
       beyond the stat the scan already did.
    2. Content fingerprint. Reads 128 KiB, and unlike mtime it survives being
       copied between systems, so it catches a library whose timestamps were
       rewritten on the way. Only files under a root that is no longer
       configured are considered, so a rename inside a library that stayed put
       is still a new file.
    """

    def __init__(self, db, roots: list[Path], rows: list[dict]):
        self._db = db
        self._roots = roots
        self._by_relative_path = _unique_by(rows, "relative_path")
        self._by_fingerprint = _unique_by(rows, "fingerprint")
        self.moved = 0
        self.matched_by_fingerprint = 0

    def __bool__(self) -> bool:
        return bool(self._by_relative_path or self._by_fingerprint)

    def __call__(self, path: str, size: int, mtime: float) -> bool:
        library_root, relative_path = split_library_path(path, self._roots)
        if relative_path is None or library_root is None:
            return False

        row = self._by_relative_path.get(relative_path)
        if row is not None and self._same_file(row, size, mtime):
            return self._point_at(row, path, library_root, relative_path)

        if not self._by_fingerprint:
            return False
        try:
            fingerprint = file_fingerprint(Path(path), size)
        except OSError:
            return False
        row = self._by_fingerprint.get(fingerprint)
        if row is None:
            return False
        # The fingerprint covers size and content, so only the timestamp can
        # differ; refresh it or the file goes straight back for re-analysis.
        moved = self._point_at(
            row, path, library_root, relative_path, mtime=mtime, fingerprint=fingerprint
        )
        if moved:
            self.matched_by_fingerprint += 1
        return moved

    @staticmethod
    def _same_file(row: dict, size: int, mtime: float) -> bool:
        return (
            int(row["size"]) == int(size)
            and abs(float(row["mtime"]) - float(mtime)) <= 1.0
        )

    def _point_at(
        self,
        row: dict,
        path: str,
        library_root: str,
        relative_path: str,
        *,
        mtime: float | None = None,
        fingerprint: str | None = None,
    ) -> bool:
        if not self._db.relocate_track(
            row["id"],
            path=path,
            library_root=library_root,
            relative_path=relative_path,
            mtime=mtime,
            fingerprint=fingerprint,
        ):
            return False
        self._by_relative_path.pop(row["relative_path"], None)
        self._by_fingerprint.pop(row["fingerprint"], None)
        self.moved += 1
        return True


def _unique_by(rows: list[dict], key: str) -> dict[str, dict]:
    """Index ``rows`` by ``key``, dropping values that appear more than once.

    An ambiguous key cannot identify a file, and guessing could attach an
    embedding to the wrong track.
    """
    index: dict[str, dict] = {}
    ambiguous: set[str] = set()
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if value in index:
            ambiguous.add(value)
            continue
        index[value] = row
    for value in ambiguous:
        del index[value]
    return index


class Analyzer:
    """Owns the DB connection and worker pool for the service lifetime."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.db = Database(settings.db_path)
        # Mark any scans that were left in 'running' state by a previous
        # process (SIGKILL, OOM, reboot) as 'crashed' before doing
        # anything else.
        crashed = self.db.mark_orphaned_scans_crashed()
        if crashed:
            logger.warning(
                "marked %d previously-running scan(s) as crashed",
                crashed,
            )
        self.index = EmbeddingIndex(self.db)
        # Workers load the actual model; the main process just records
        # which model rows were extracted with.
        self.model_name: str = MODEL_NAME
        self.embedding_dim: int = EMBEDDING_DIM
        self.pool: WorkerPool | None = None
        self.status = ScanStatus()
        self._scan_lock = threading.Lock()
        # Set by request_cancel(). Checked at phase boundaries and inside
        # the result loop so a long scan can be aborted promptly.
        self._cancel_event = threading.Event()

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        if self.pool is not None:
            return
        try:
            pool = WorkerPool(
                workers=self.settings.worker_count,
                log_level=self.settings.log_level,
                should_stop=self._cancel_event.is_set,
            )
        except DownloadCancelled as exc:
            # Cancelled while fetching models on a first run.
            raise _ScanCancelled() from exc
        # Building the pool spawns processes, which takes long enough for a
        # cancel to land in the middle. request_cancel() saw no pool to
        # terminate, so hand back nothing rather than a pool the scan is about
        # to abandon with the whole library queued in it.
        if self._cancel_event.is_set():
            with contextlib.suppress(Exception):
                pool.terminate()
            raise _ScanCancelled()
        self.pool = pool

    def request_cancel(self) -> bool:
        """Ask the running scan to wind down. The result loop breaks out
        on the next iteration and the pool is terminated so workers stuck
        on slow I/O (CIFS, NFS) abort instead of waiting.

        Returns True if a scan was active, False if no-op. Repeated calls are
        safe, and each one tears down whatever pool exists now — a pool built
        after an earlier request still has to go.
        """
        if self.status.state != "scanning":
            return False
        if not self._cancel_event.is_set():
            logger.warning("scan cancellation requested")
        self._cancel_event.set()
        if self.pool is not None:
            with contextlib.suppress(Exception):
                self.pool.terminate()
            self.pool = None
        return True

    def _check_cancel(self) -> None:
        """Raise :class:`_ScanCancelled` if a cancel was requested. Called
        at phase boundaries inside :meth:`_run_scan` so cancellation takes
        effect promptly even during enumeration / classification (which
        don't run their own result-loop check)."""
        if self._cancel_event.is_set():
            raise _ScanCancelled()

    def stop(self) -> None:
        """Shutdown the analyzer: cancel any in-flight scan, wait for it to
        record its outcome, then release the pool and DB.

        The wait is bounded, but that bounds this call only, not process exit: a
        scan blocked in an OS filesystem call cannot be interrupted, and the
        thread running it keeps the interpreter alive. When the wait runs out
        the DB is left open, because the scan still owns it.
        """
        cancelled = self.request_cancel()
        # A cancelled scan still has to record its outcome. Closing the DB
        # underneath it loses the scan-history row and raises "Cannot operate
        # on a closed database", leaving the scan marked running forever.
        acquired = self._scan_lock.acquire(timeout=SHUTDOWN_WAIT_SEC)
        try:
            pool, self.pool = self.pool, None
            if pool is not None:
                # A cancelled or abandoned scan must not drain its queue:
                # close() waits for every job already submitted.
                if cancelled or not acquired:
                    with contextlib.suppress(Exception):
                        pool.terminate()
                else:
                    pool.close()
            if acquired:
                self.db.close()
            else:
                logger.warning(
                    "scan still running after %ss; leaving the database open "
                    "so it can finish writing",
                    SHUTDOWN_WAIT_SEC,
                )
        finally:
            if acquired:
                self._scan_lock.release()

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
            state="scanning",
            phase="enumerating",
            started_at=time.time(),
        )
        t0 = time.monotonic()
        # Reset cancellation state for this scan.
        self._cancel_event.clear()

        # Persist a 'running' scan row up front; we'll fill it in on
        # success or failure via finish_scan in the finally branch.
        self.status.scan_id = self.db.start_scan(
            workers=self.settings.worker_count,
            # The scans.backend column predates removing the
            # MusicExtractor option — it's always "effnet" now but the
            # column stays so historical rows still parse.
            backend="effnet",
            model=self.model_name,
            forced=force,
            harmonie_version=__version__,
            descriptor_version=DESCRIPTOR_VERSION,
        )

        scan_state = "completed"
        scan_last_error: str | None = None
        try:
            libs = [Path(p) for p in self.settings.libraries]
            if not libs:
                logger.warning("no libraries configured (HARMONIE_LIBRARIES is empty)")
            # Only prune entries that live under reachable roots so a
            # flaky NAS doesn't wipe the index when the mount is
            # unavailable.
            reachable: list[Path] = []
            unreachable: list[Path] = []
            for p in libs:
                if Path(p).expanduser().exists():
                    reachable.append(p)
                else:
                    unreachable.append(p)
            for p in unreachable:
                logger.warning("library root unreachable, skipping: %s", p)

            # Enumeration: update self.status.discovered on each yield,
            # log every 10 seconds.
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
                        "enumerating: %d audio file(s) found so far...",
                        len(files),
                    )
                    last_progress = now
            logger.info("discovered %d audio file(s)", len(files))
            self._check_cancel()

            # Classification: one stat() per file plus a DB lookup.
            # Periodic progress on the same 10-second cadence as
            # enumeration.
            self.status.phase = "classifying"
            classify_state = {"last": time.monotonic()}

            def _classify_progress(n: int) -> None:
                now = time.monotonic()
                if now - classify_state["last"] > 10:
                    logger.info(
                        "classifying: %d / %d file(s) checked...",
                        n,
                        len(files),
                    )
                    classify_state["last"] = now

            relocator = self._build_relocator(reachable)
            full_jobs, desc_jobs, skipped = build_jobs(
                self.db,
                files,
                model_name=self.model_name,
                force=force,
                on_progress=_classify_progress,
                relocate=relocator,
            )
            if relocator is not None and relocator.moved:
                logger.info(
                    "re-pointed %d track(s) to their new location "
                    "(%d by content fingerprint); no re-analysis needed",
                    relocator.moved,
                    relocator.matched_by_fingerprint,
                )
            self._record_fingerprints(files)
            self.status.skipped = skipped
            logger.info(
                "jobs: full=%d, descriptors_only=%d, skipped=%d",
                len(full_jobs),
                len(desc_jobs),
                skipped,
            )
            self._check_cancel()

            # Worker pool starts only when there's work to dispatch.
            all_jobs: list = list(full_jobs) + list(desc_jobs)
            if all_jobs:
                self.status.phase = "extracting"
                if self.pool is None:
                    self.start()
                assert self.pool is not None
                # Cancellation terminates the pool. The result loop below
                # polls, so it notices the cancel and stops collecting
                # instead of waiting for results that will never arrive.
                try:
                    for result in self.pool.map(
                        all_jobs,
                        should_stop=self._cancel_event.is_set,
                    ):
                        if self._cancel_event.is_set():
                            break
                        self._handle_result(result, reachable_roots=reachable)
                except Exception:
                    if not self._cancel_event.is_set():
                        raise
                self._check_cancel()

            # Prune rows for files that disappeared, scoped to the roots
            # we actually walked. Skipped on cancel since the file list
            # may be incomplete.
            if not reachable:
                logger.warning("no reachable libraries this scan; skipping prune")
            else:
                self.status.phase = "pruning"
                present = {str(f) for f in files}
                removed = self.db.prune_missing_under_roots(
                    roots=reachable, keep=present
                )
                self.status.removed = removed
                if removed:
                    logger.info("pruned %d removed track(s)", removed)

            # A cancel accepted while pruning ran would otherwise be recorded
            # as a completed scan.
            self._check_cancel()
        except _ScanCancelled:
            scan_state = "cancelled"
            scan_last_error = _CANCELLED_REASON
            self.status.last_error = scan_last_error
            logger.warning("scan cancelled by user")
        except KeyboardInterrupt:
            scan_state = "cancelled"
            scan_last_error = f"{_CANCELLED_REASON} (KeyboardInterrupt)"
            self.status.last_error = scan_last_error
            logger.warning("scan cancelled by user (KeyboardInterrupt)")
            raise
        except Exception as exc:
            scan_state = "crashed"
            scan_last_error = repr(exc)
            self.status.last_error = scan_last_error
            logger.exception("scan crashed")
            raise
        finally:
            elapsed = time.monotonic() - t0
            # Drop cached embedding matrices; the next query rebuilds them.
            self.index.invalidate()
            self.status.state = "idle"
            self.status.phase = "idle"
            self.status.finished_at = time.time()
            self.status.last_duration_sec = elapsed
            # Persist the outcome.
            try:
                self.db.finish_scan(
                    self.status.scan_id,
                    duration_sec=elapsed,
                    discovered=self.status.discovered,
                    full=self.status.full,
                    descriptors_only=self.status.descriptors_only,
                    skipped=self.status.skipped,
                    failed=self.status.failed,
                    removed=self.status.removed,
                    state=scan_state,
                    last_error=scan_last_error,
                )
            except Exception:  # pragma: no cover
                logger.exception("failed to persist scan-history row")
            logger.info(
                "scan complete in %.1fs: full=%d, descriptors_only=%d, "
                "skipped=%d, failed=%d, removed=%d",
                elapsed,
                self.status.full,
                self.status.descriptors_only,
                self.status.skipped,
                self.status.failed,
                self.status.removed,
            )

    def _build_relocator(self, roots: list[Path]) -> _LibraryRelocator | None:
        """Return a relocator only when the index holds rows under a root that
        is no longer configured. An unchanged library pays one indexed DISTINCT
        query per scan and nothing per file."""
        configured = [
            str(Path(p).expanduser().resolve()) for p in self.settings.libraries
        ]
        orphaned = self.db.orphaned_library_roots(configured)
        if not orphaned:
            return None
        rows = self.db.relocation_rows(orphaned)
        relocator = _LibraryRelocator(self.db, roots, rows)
        if not relocator:
            return None
        logger.info(
            "index holds %d track(s) under %s, which is not a configured "
            "library; matching them by relative path and content fingerprint",
            len(rows),
            ", ".join(orphaned),
        )
        return relocator

    def _record_fingerprints(self, files: list[Path]) -> int:
        """Give already-analysed rows a fingerprint if they lack one.

        Costs one 128 KiB read per track, once. Without it a library copied to
        another system has nothing but mtime to match on, and mtime does not
        survive most copies.
        """
        if not self.db.count_missing_fingerprints():
            return 0
        missing = self.db.tracks_missing_fingerprint([str(f) for f in files])
        if not missing:
            return 0
        recorded = 0
        for path, track_id in missing.items():
            if self._cancel_event.is_set():
                break
            try:
                self.db.set_fingerprint(track_id, file_fingerprint(Path(path)))
                recorded += 1
            except OSError as exc:
                logger.debug("could not fingerprint %s: %s", path, exc)
        if recorded:
            logger.info(
                "recorded content fingerprints for %d existing track(s)", recorded
            )
        return recorded

    def _handle_result(self, result, *, reachable_roots: list[Path]) -> None:
        if isinstance(result, FullResult):
            try:
                lib_root, rel_path = split_library_path(result.path, reachable_roots)
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
                    fingerprint=_fingerprint_or_none(result.path, result.size),
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
            if self.status.scan_id is not None:
                try:
                    self.db.record_scan_failure(
                        self.status.scan_id,
                        path=result.path,
                        error=result.error,
                    )
                except Exception:  # pragma: no cover
                    logger.exception(
                        "failed to persist scan_failure row for %s",
                        result.path,
                    )

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
