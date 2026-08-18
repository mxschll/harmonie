"""Cancellation and shutdown behaviour.

Terminating a multiprocessing pool does not wake a consumer blocked in
``imap_unordered``: it keeps waiting for a result no worker will ever send. A
scan that cannot notice its own cancellation leaves the process unable to shut
down, and its scan-history row stuck in 'running'.
"""

from __future__ import annotations

import multiprocessing as mp
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from harmonie.analyzer import Analyzer
from harmonie.config import Settings
from harmonie.workers import WorkerPool


def _slow(x):
    time.sleep(30)
    return x


def test_a_blocking_consumer_hangs_after_terminate():
    """The premise of the fix, with a real pool: this is why map() polls."""
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=2)
    finished = threading.Event()

    def consume():
        for _ in pool.imap_unordered(_slow, range(20)):
            pass
        finished.set()

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(2)
    pool.terminate()

    t.join(timeout=3)
    assert t.is_alive(), "terminate() was expected to leave the consumer stuck"
    assert not finished.is_set()


class _StuckIterator:
    """Stands in for an IMapIterator whose results never arrive."""

    def __init__(self) -> None:
        self.polls = 0

    def next(self, timeout=None):
        self.polls += 1
        raise mp.TimeoutError


class _StubPool:
    def __init__(self, iterator: _StuckIterator) -> None:
        self._iterator = iterator

    def imap_unordered(self, func, jobs, chunksize=1):
        return self._iterator


def _pool_with(iterator: _StuckIterator) -> WorkerPool:
    pool = WorkerPool.__new__(WorkerPool)  # no models, no subprocesses
    pool.workers = 2
    pool._pool = _StubPool(iterator)
    return pool


def test_map_returns_when_the_scan_is_cancelled():
    iterator = _StuckIterator()
    pool = _pool_with(iterator)
    cancelled = threading.Event()
    exited = threading.Event()

    def consume():
        for _ in pool.map([1, 2, 3], should_stop=cancelled.is_set):
            pass
        exited.set()

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(1)
    assert not exited.is_set()
    assert iterator.polls > 0, "map() should be polling, not blocking"

    cancelled.set()
    t.join(timeout=5)
    assert exited.is_set(), "map() should return once the scan is cancelled"


def test_map_returns_when_the_pool_is_terminated_underneath_it():
    iterator = _StuckIterator()
    pool = _pool_with(iterator)
    exited = threading.Event()

    def consume():
        for _ in pool.map([1, 2, 3], should_stop=lambda: False):
            pass
        exited.set()

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(1)
    pool._pool = None  # what terminate() leaves behind

    t.join(timeout=5)
    assert exited.is_set(), "map() should return when the pool is gone"


@pytest.fixture
def analyzer(tmp_path: Path) -> Analyzer:
    lib = tmp_path / "library"
    lib.mkdir()
    return Analyzer(Settings(libraries=[lib], data_dir=tmp_path))


def test_stop_waits_for_an_unwinding_scan_before_closing_the_db(
    analyzer: Analyzer, monkeypatch
):
    """A cancelled scan still records its outcome. Closing the DB underneath it
    raised 'Cannot operate on a closed database' and left the scan running."""
    in_scan = threading.Event()
    release = threading.Event()
    db_usable_at_end = {"ok": None}

    def fake_run_scan(*, force: bool) -> None:
        in_scan.set()
        release.wait(timeout=10)
        # Stand-in for the real finally block's scan-history write.
        try:
            analyzer.db.stats()
            db_usable_at_end["ok"] = True
        except sqlite3.ProgrammingError:
            db_usable_at_end["ok"] = False

    monkeypatch.setattr(analyzer, "_run_scan", fake_run_scan)

    scan_thread = threading.Thread(target=analyzer.scan, daemon=True)
    scan_thread.start()
    assert in_scan.wait(timeout=5)

    stopped = threading.Event()
    stopper = threading.Thread(target=lambda: (analyzer.stop(), stopped.set()))
    stopper.daemon = True
    stopper.start()

    # stop() must wait for the scan, not close the DB out from under it.
    assert not stopped.wait(timeout=1)

    release.set()
    scan_thread.join(timeout=10)
    assert stopped.wait(timeout=10), "stop() should finish once the scan unwinds"
    assert db_usable_at_end["ok"] is True, "DB was closed under the scan thread"


def test_stop_gives_up_waiting_after_the_timeout(analyzer: Analyzer, monkeypatch):
    """A scan that will not unwind must not block shutdown forever."""
    monkeypatch.setattr("harmonie.analyzer.SHUTDOWN_WAIT_SEC", 1)
    in_scan = threading.Event()
    wedged = threading.Event()

    def fake_run_scan(*, force: bool) -> None:
        in_scan.set()
        wedged.wait(timeout=30)

    monkeypatch.setattr(analyzer, "_run_scan", fake_run_scan)
    threading.Thread(target=analyzer.scan, daemon=True).start()
    assert in_scan.wait(timeout=5)

    started = time.monotonic()
    analyzer.stop()
    elapsed = time.monotonic() - started

    assert 1 <= elapsed < 10, f"stop() waited {elapsed:.1f}s"
    wedged.set()


def test_stop_closes_when_no_scan_is_running(analyzer: Analyzer):
    analyzer.stop()
    with pytest.raises(sqlite3.ProgrammingError):
        analyzer.db.stats()
