"""Tests for the worker-process initialization."""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def stub_essentia(monkeypatch):
    """Install a fake ``essentia`` module on ``sys.modules``."""
    fake = types.ModuleType("essentia")
    fake.log = types.SimpleNamespace(warningActive=True)
    monkeypatch.setitem(sys.modules, "essentia", fake)
    return fake


@pytest.fixture
def stub_extractor(monkeypatch):
    """Replace ``EffnetExtractor`` with a sentinel-returning factory."""
    from harmonie import workers as workers_mod

    sentinel = object()
    monkeypatch.setattr(workers_mod, "EffnetExtractor", lambda: sentinel)
    return sentinel


def test_worker_init_silences_essentia_warnings_at_info(stub_essentia, stub_extractor):
    from harmonie.workers import _worker_init

    _worker_init("INFO")
    assert stub_essentia.log.warningActive is False


def test_worker_init_silences_essentia_warnings_at_warning(
    stub_essentia, stub_extractor
):
    from harmonie.workers import _worker_init

    _worker_init("WARNING")
    assert stub_essentia.log.warningActive is False


def test_worker_init_keeps_warnings_at_debug(stub_essentia, stub_extractor):
    from harmonie.workers import _worker_init

    _worker_init("DEBUG")
    assert stub_essentia.log.warningActive is True


def test_worker_init_debug_case_insensitive(stub_essentia, stub_extractor):
    from harmonie.workers import _worker_init

    _worker_init("debug")
    assert stub_essentia.log.warningActive is True


# ---------------------------------------------------------------------------
# Per-file progress logging
# ---------------------------------------------------------------------------


def test_do_full_logs_extracting_line(monkeypatch, caplog):
    from harmonie import workers as workers_mod

    class _FakeExtractor:
        name = "test-model"
        genre_labels = None

        def extract(self, path):
            raise RuntimeError("not real audio")

    monkeypatch.setattr(workers_mod, "_extractor", _FakeExtractor())

    job = workers_mod.FullJob(path="/lib/track-1.flac", size=1, mtime=1.0)
    with caplog.at_level("INFO", logger="harmonie.workers"):
        result = workers_mod._do_full(job)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("extracting" in m and "/lib/track-1.flac" in m for m in msgs)
    assert isinstance(result, workers_mod.JobError)


def test_do_descriptors_logs_refreshing_line(monkeypatch, caplog):
    from harmonie import workers as workers_mod

    class _FakeExtractor:
        name = "test-model"
        genre_labels = None

        def extract_descriptors(self, path):
            raise RuntimeError("nope")

    monkeypatch.setattr(workers_mod, "_extractor", _FakeExtractor())

    job = workers_mod.DescriptorJob(path="/lib/track-2.flac")
    with caplog.at_level("INFO", logger="harmonie.workers"):
        result = workers_mod._do_descriptors(job)

    msgs = [r.getMessage() for r in caplog.records]
    assert any("refreshing descriptors" in m and "/lib/track-2.flac" in m for m in msgs)
    assert isinstance(result, workers_mod.JobError)


# ---------------------------------------------------------------------------
# Per-job timeout against real worker processes
# ---------------------------------------------------------------------------

# These run in spawned children, so they have to be importable by name.

WEDGE_PATH = "/lib/wedged.flac"


class _BusyExtractor:
    """``extract`` never returns for WEDGE_PATH, spinning in a tight Python
    loop the way the native decoder spins in C."""

    name = "test-model"
    genre_labels = None

    def extract(self, path):
        if str(path) == WEDGE_PATH:
            while True:
                pass
        import numpy as np

        from harmonie.features import Descriptors, TrackFeatures

        return TrackFeatures(
            embedding=np.zeros(4, dtype=np.float32),
            duration=1.0,
            model=self.name,
            descriptors=Descriptors(),
            style_activations=None,
        )


def _init_busy_worker(started) -> None:
    from harmonie import workers as workers_mod

    workers_mod._started = started
    workers_mod._extractor = _BusyExtractor()


def _busy_pool(ctx, workers, started):
    return ctx.Pool(
        processes=workers, initializer=_init_busy_worker, initargs=(started,)
    )


def test_timeout_kills_a_worker_stuck_in_a_tight_loop(monkeypatch):
    """End to end with real processes: a worker that never returns is killed,
    the file is failed, and the remaining files still complete."""
    import multiprocessing as mp
    import threading

    from harmonie import workers as workers_mod
    from harmonie.workers import FullJob, FullResult, JobError, WorkerPool

    monkeypatch.setattr(workers_mod, "RESULT_POLL_SEC", 0.05)
    monkeypatch.setattr("harmonie.workers.extract_tags", lambda p: None)
    monkeypatch.setattr("harmonie.workers._fingerprint_or_none", lambda p, s: None)

    ctx = mp.get_context("spawn")
    pool = WorkerPool.__new__(WorkerPool)
    pool.workers = 2
    pool.job_timeout = 1.0
    pool._lock = threading.Lock()
    pool._stopped = False
    pool._ctx = ctx
    pool._new_started = lambda: ctx.RawArray("b", 4)  # type: ignore[method-assign]
    pool._started = pool._new_started()
    pool._spawn_pool = lambda: _busy_pool(ctx, 2, pool._started)  # type: ignore[method-assign]
    pool._pool = pool._spawn_pool()

    jobs = [FullJob(path=f"/lib/{i}.flac", size=1, mtime=1.0) for i in range(6)]
    jobs.insert(2, FullJob(path=WEDGE_PATH, size=1, mtime=1.0))

    try:
        out = list(pool.map(jobs))
    finally:
        pool.terminate()

    errors = [r for r in out if isinstance(r, JobError)]
    done = sorted(r.path for r in out if isinstance(r, FullResult))
    assert [e.path for e in errors] == [WEDGE_PATH]
    assert "timeout" in errors[0].error
    assert done == sorted(j.path for j in jobs if j.path != WEDGE_PATH)
