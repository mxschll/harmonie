"""Cancellation races reported in review of the polling fix."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from harmonie import analyzer as analyzer_mod
from harmonie.analyzer import Analyzer
from harmonie.config import Settings
from harmonie.workers import FullJob, WorkerPool


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
        self,
        *,
        workers: int,
        log_level: str = "INFO",
        should_stop=None,
        job_timeout: float | None = None,
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


# ---------------------------------------------------------------------------
# WorkerPool.map against a fake multiprocessing pool
# ---------------------------------------------------------------------------


class FakeResult:
    """Stands in for an AsyncResult. ``finish()`` makes it ready."""

    def __init__(self, job, ticket):
        self.job = job
        self.ticket = ticket
        self._ready = False

    def finish(self):
        self._ready = True

    def ready(self):
        return self._ready

    def get(self, timeout=None):
        assert self._ready
        return ("done", self.job.path)


class FakePool:
    """Records submissions and lets a test drive which jobs start and finish.
    Every instance registers itself so a test can tell pools apart after a
    restart."""

    instances: list = []

    def __init__(self, started):
        self.started = started
        self.results: list[FakeResult] = []
        self.terminated = False
        # When set, every submitted job completes at once.
        self.auto_finish = False
        FakePool.instances.append(self)

    def apply_async(self, func, args):
        job, ticket = args
        r = FakeResult(job, ticket)
        if self.auto_finish:
            self.started[ticket] = 1
            r.finish()
        self.results.append(r)
        return r

    def start(self, path):
        """Pretend a worker picked ``path`` up."""
        for r in self.results:
            if r.job.path == path:
                self.started[r.ticket] = 1
                return
        raise AssertionError(f"{path} was never submitted to this pool")

    def finish(self, path):
        for r in self.results:
            if r.job.path == path:
                self.started[r.ticket] = 1
                r.finish()
                return
        raise AssertionError(f"{path} was never submitted to this pool")

    def terminate(self):
        self.terminated = True

    def join(self):
        pass

    def close(self):
        pass


def _fake_worker_pool(*, workers: int, job_timeout) -> WorkerPool:
    """A WorkerPool with no models and no subprocesses, backed by FakePool.
    ``_spawn_pool`` is replaced so a restart yields another FakePool."""
    from harmonie import workers as workers_mod

    pool = WorkerPool.__new__(WorkerPool)
    pool.workers = workers
    pool.job_timeout = job_timeout
    pool._lock = threading.Lock()
    pool._stopped = False
    pool._new_started = lambda: bytearray(workers_mod.INFLIGHT_PER_WORKER * workers)  # type: ignore[method-assign]
    pool._started = pool._new_started()
    pool._spawn_pool = lambda: FakePool(pool._started)  # type: ignore[method-assign]
    pool._pool = pool._spawn_pool()
    return pool


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    from harmonie import workers as workers_mod

    monkeypatch.setattr(workers_mod, "RESULT_POLL_SEC", 0.001)
    FakePool.instances = []
    yield


def _jobs(n: int) -> list[FullJob]:
    return [FullJob(path=f"/lib/{i}.flac", size=1, mtime=1.0) for i in range(n)]


def _step(gen, action, fake):
    """Run ``action(fake)`` then take the next value from ``gen``."""
    action(fake)
    return next(gen)


def test_map_submits_one_job_per_task():
    """One job per task — never batched — is what lets a single bad file be
    pinned to one result and timed out."""
    import inspect

    pool = _fake_worker_pool(workers=2, job_timeout=None)
    fake = FakePool.instances[0]
    fake.auto_finish = True
    jobs = _jobs(3)

    out = list(pool.map(jobs))
    assert [r.job for r in fake.results] == jobs
    assert sorted(p for _, p in out) == sorted(j.path for j in jobs)

    # The option is gone rather than silently broken for chunksize > 1.
    assert "chunksize" not in inspect.signature(WorkerPool.map).parameters
    with pytest.raises(TypeError):
        list(pool.map(jobs, chunksize=2))


def test_map_checks_for_cancellation_before_submitting():
    pool = _fake_worker_pool(workers=2, job_timeout=None)
    assert list(pool.map(_jobs(3), should_stop=lambda: True)) == []
    assert FakePool.instances[0].results == []


def test_map_keeps_only_a_bounded_number_of_jobs_in_flight():
    """Queueing the whole library at once would leave the timeout measuring
    queue position rather than run time."""
    from harmonie import workers as workers_mod

    pool = _fake_worker_pool(workers=2, job_timeout=None)
    fake = FakePool.instances[0]
    cap = workers_mod.INFLIGHT_PER_WORKER * 2

    gen = pool.map(_jobs(50))
    consume = threading.Thread(target=lambda: next(gen, None), daemon=True)
    consume.start()
    time.sleep(0.05)
    assert len(fake.results) == cap

    # Finishing one job and resuming the generator admits exactly one more.
    fake.finish("/lib/0.flac")
    consume.join(timeout=2)
    assert not consume.is_alive()
    resume = threading.Thread(target=lambda: next(gen, None), daemon=True)
    resume.start()
    time.sleep(0.05)
    assert len(fake.results) == cap + 1
    assert resume.is_alive(), "map() returned with jobs still outstanding"


def test_map_does_not_time_out_a_job_that_has_not_started():
    """A job buffered behind a slow one, or waiting for a worker to load its
    models, must not be charged for that wait."""
    pool = _fake_worker_pool(workers=1, job_timeout=0.02)
    fake = FakePool.instances[0]
    jobs = _jobs(2)

    gen = pool.map(jobs)
    # Neither job has started. Sit well past the timeout.
    consume = threading.Thread(target=lambda: next(gen, None), daemon=True)
    consume.start()
    time.sleep(0.1)
    assert consume.is_alive(), "map() failed an unstarted job"
    assert len(FakePool.instances) == 1, "pool was restarted for an unstarted job"

    fake.finish("/lib/0.flac")
    fake.finish("/lib/1.flac")
    consume.join(timeout=2)
    assert not fake.terminated


def test_map_times_out_wedged_job_and_requeues_collateral():
    from harmonie.workers import JobError

    pool = _fake_worker_pool(workers=2, job_timeout=0.02)
    first = FakePool.instances[0]
    jobs = _jobs(5)
    wedged = "/lib/2.flac"

    out = []
    gen = pool.map(jobs)

    def take_one():
        out.append(next(gen))

    # Four jobs are in flight on the first pool. 0 finishes.
    t = threading.Thread(target=take_one, daemon=True)
    t.start()
    time.sleep(0.02)
    first.finish("/lib/0.flac")
    t.join(timeout=2)
    assert out == [("done", "/lib/0.flac")]

    # 4 is admitted. 2 starts and never finishes; 1, 3 and 4 never start. The
    # next yield must be 2's JobError once it outlives the timeout.
    t = threading.Thread(target=take_one, daemon=True)
    t.start()
    time.sleep(0.01)
    first.start(wedged)
    t.join(timeout=2)
    assert not t.is_alive()
    assert isinstance(out[-1], JobError)
    assert out[-1].path == wedged
    assert "timeout" in out[-1].error

    # Resuming the generator performs the restart. The old pool is dead, a new
    # one exists, and the three collateral jobs (1, 3, 4) were requeued onto
    # it, none lost or doubled.
    t = threading.Thread(target=take_one, daemon=True)
    t.start()
    time.sleep(0.02)
    assert len(FakePool.instances) == 2
    second = FakePool.instances[1]
    assert first.terminated is True
    assert [r.job.path for r in second.results] == [
        "/lib/1.flac",
        "/lib/3.flac",
        "/lib/4.flac",
    ]
    for path in ("/lib/1.flac", "/lib/3.flac", "/lib/4.flac"):
        second.finish(path)
    t.join(timeout=2)
    out.extend(gen)

    done = sorted(item[1] for item in out if isinstance(item, tuple))
    assert done == ["/lib/0.flac", "/lib/1.flac", "/lib/3.flac", "/lib/4.flac"]
    assert len(FakePool.instances) == 2


def test_map_harvests_results_that_finished_during_recovery():
    """A collateral job that completes while the timed-out one is being
    reported is kept, not thrown away and re-run."""

    pool = _fake_worker_pool(workers=2, job_timeout=0.02)
    first = FakePool.instances[0]
    wedged = "/lib/0.flac"

    gen = pool.map(_jobs(2))
    consume = threading.Thread(target=lambda: next(gen, None), daemon=True)
    consume.start()
    time.sleep(0.02)
    first.start(wedged)
    consume.join(timeout=2)  # returns once the JobError is yielded
    assert not consume.is_alive()

    # While the consumer holds the JobError, job 1 completes on the old pool.
    first.finish("/lib/1.flac")
    rest = list(gen)

    assert rest == [("done", "/lib/1.flac")]
    assert FakePool.instances[1].results == [], "finished job was re-run"


def test_cancel_during_restart_leaves_no_pool_running():
    """request_cancel() can land while a replacement pool is being spawned.
    The replacement must be torn down, not left running with nobody owning it."""
    from harmonie.workers import JobError

    pool = _fake_worker_pool(workers=1, job_timeout=0.02)
    first = FakePool.instances[0]
    spawning = threading.Event()
    release = threading.Event()

    def slow_spawn():
        spawning.set()
        release.wait(timeout=5)
        return FakePool(pool._started)

    pool._spawn_pool = slow_spawn  # type: ignore[method-assign]

    gen = pool.map(_jobs(2))
    yielded = []

    def consume():
        yielded.extend(gen)

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    time.sleep(0.02)
    first.start("/lib/0.flac")
    assert spawning.wait(timeout=2), "no restart happened"

    pool.terminate()  # what request_cancel() does from the API thread
    release.set()
    t.join(timeout=2)
    assert not t.is_alive()

    assert [type(y) for y in yielded] == [JobError]
    assert pool._pool is None
    assert FakePool.instances[1].terminated is True


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
