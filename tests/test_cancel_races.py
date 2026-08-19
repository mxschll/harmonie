"""Cancellation races reported in review of the polling fix."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from harmonie import analyzer as analyzer_mod
from harmonie.analyzer import Analyzer
from harmonie.config import Settings
from harmonie.workers import FullJob


@pytest.fixture
def scan_harness(tmp_path: Path, monkeypatch):
    """An Analyzer whose enumeration and job building are stubbed, so tests can
    drive the cancellation races around pool startup and pruning."""
    lib = tmp_path / "library"
    lib.mkdir()
    analyzer = Analyzer(Settings(libraries=[lib], data_dir=tmp_path))

    def fake_iter(roots) -> Iterator[Path]:
        yield from [Path("/lib/a.flac"), Path("/lib/b.flac")]

    monkeypatch.setattr(analyzer_mod, "iter_audio_files", fake_iter)

    def fake_build_jobs(
        db, files, *, model_name, force, on_progress=None, relocate=None
    ):
        jobs = [FullJob(path=str(f), size=1, mtime=1.0) for f in files]
        return jobs, [], 0

    monkeypatch.setattr(analyzer_mod, "build_jobs", fake_build_jobs)
    yield analyzer


class SlowStartPool:
    """A pool whose construction blocks, like the real one downloading models
    and spawning processes."""

    instances: list[SlowStartPool] = []

    construction_started = threading.Event()
    release_construction = threading.Event()

    def __init__(
        self, *, workers: int, log_level: str = "INFO", should_stop=None
    ) -> None:
        self.workers = workers
        self.submitted: list = []
        self.closed = False
        self.terminated = False
        SlowStartPool.instances.append(self)
        SlowStartPool.construction_started.set()
        SlowStartPool.release_construction.wait(timeout=10)

    def map(self, jobs, *, should_stop=None):
        # The real map() submits to the pool here; record that it happened.
        self.submitted.extend(jobs)
        if should_stop is not None and should_stop():
            return iter([])
        return iter([])

    def close(self) -> None:
        self.closed = True

    def terminate(self) -> None:
        self.terminated = True


@pytest.fixture(autouse=True)
def _reset_slow_pool():
    SlowStartPool.instances = []
    SlowStartPool.construction_started = threading.Event()
    SlowStartPool.release_construction = threading.Event()
    yield


def test_cancel_during_pool_startup_submits_no_jobs(scan_harness, monkeypatch):
    """Cancelling while the pool is still being built must not queue the whole
    library, and must leave a terminated pool rather than one that drains."""
    analyzer = scan_harness
    monkeypatch.setattr(analyzer_mod, "WorkerPool", SlowStartPool)

    scan = threading.Thread(target=analyzer.scan, daemon=True)
    scan.start()

    assert SlowStartPool.construction_started.wait(timeout=5)
    assert analyzer.request_cancel() is True
    SlowStartPool.release_construction.set()
    scan.join(timeout=10)
    assert not scan.is_alive()

    pool = SlowStartPool.instances[0]
    assert pool.submitted == [], "jobs were submitted after cancellation"
    assert pool.terminated is True, "pool created during cancel was not terminated"
    assert pool.closed is False, "close() drains the queue instead of aborting"
    rows, _ = analyzer.db.list_scans(limit=1)
    assert rows[0]["state"] == "cancelled"


def test_map_submits_one_job_per_task():
    """Batching would make imap_unordered return a bare generator with no
    next(timeout=...), which is what the cancellation poll relies on."""
    import inspect

    from harmonie.workers import WorkerPool

    recorded = {}

    class _Exhausted:
        def next(self, timeout=None):
            raise StopIteration

    class RecordingPool:
        def imap_unordered(self, func, jobs, chunksize=1):
            recorded["chunksize"] = chunksize
            return _Exhausted()

    pool = WorkerPool.__new__(WorkerPool)
    pool.workers = 2
    pool._pool = RecordingPool()

    list(pool.map([1, 2, 3]))
    assert recorded["chunksize"] == 1

    # The option is gone rather than silently broken for chunksize > 1.
    assert "chunksize" not in inspect.signature(WorkerPool.map).parameters
    with pytest.raises(TypeError):
        list(pool.map([1, 2, 3], chunksize=2))


def test_map_checks_for_cancellation_before_submitting():
    from harmonie.workers import WorkerPool

    submitted = []

    class _Exhausted:
        def next(self, timeout=None):
            raise StopIteration

    class RecordingPool:
        def imap_unordered(self, func, jobs, chunksize=1):
            submitted.extend(jobs)
            return _Exhausted()

    pool = WorkerPool.__new__(WorkerPool)
    pool.workers = 2
    pool._pool = RecordingPool()

    assert list(pool.map([1, 2, 3], should_stop=lambda: True)) == []
    assert submitted == []


def test_cancel_during_pruning_is_not_recorded_as_completed(scan_harness):
    """request_cancel() returning True promises the scan ends cancelled."""
    analyzer = scan_harness

    class EmptyPool:
        def map(self, jobs, *, should_stop=None):
            return iter([])

        def close(self) -> None:
            pass

        def terminate(self) -> None:
            pass

    analyzer.pool = EmptyPool()

    pruning = threading.Event()
    release = threading.Event()

    def blocking_prune(*, roots, keep) -> int:
        pruning.set()
        release.wait(timeout=10)
        return 0

    analyzer.db.prune_missing_under_roots = blocking_prune

    scan = threading.Thread(target=analyzer.scan, daemon=True)
    scan.start()
    assert pruning.wait(timeout=5)

    accepted = analyzer.request_cancel()
    release.set()
    scan.join(timeout=10)

    rows, _ = analyzer.db.list_scans(limit=1)
    if accepted:
        assert rows[0]["state"] == "cancelled", (
            "cancellation was accepted but the scan was persisted as "
            f"{rows[0]['state']!r}"
        )
    else:
        assert rows[0]["state"] == "completed"
