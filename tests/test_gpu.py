"""CUDA detection and the settings that let workers share one card."""

from __future__ import annotations

import ctypes
import os

import pytest

from harmonie import workers
from harmonie.gpu import cuda_device_count
from harmonie.workers import _worker_init


class _StubContext:
    """Stands in for the spawn context: no subprocesses, no models."""

    def Pool(self, **kwargs):  # noqa: N802 - mirrors multiprocessing's name
        return object()


class _FakeDriver:
    def __init__(self, count: int, init_rc: int = 0, count_rc: int = 0) -> None:
        self._count = count
        self._init_rc = init_rc
        self._count_rc = count_rc

    def cuInit(self, flags):  # noqa: N802 - mirrors the C name
        return self._init_rc

    def cuDeviceGetCount(self, ptr):  # noqa: N802 - mirrors the C name
        ptr._obj.value = self._count
        return self._count_rc


def test_no_driver_means_no_gpu(monkeypatch):
    """The common case: a host or container without the NVIDIA driver."""

    def missing(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", missing)
    assert cuda_device_count() == 0


def test_driver_reports_devices(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _FakeDriver(2))
    assert cuda_device_count() == 2


def test_failed_init_means_no_gpu(monkeypatch):
    """A driver present but unusable — the container was started without
    --gpus, for instance."""
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _FakeDriver(1, init_rc=100))
    assert cuda_device_count() == 0


def test_failed_device_query_means_no_gpu(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _FakeDriver(1, count_rc=3))
    assert cuda_device_count() == 0


def test_missing_symbols_means_no_gpu(monkeypatch):
    class Empty:
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(ctypes, "CDLL", lambda name: Empty())
    assert cuda_device_count() == 0


def test_worker_lets_processes_share_one_card(monkeypatch):
    """Without memory growth the first worker reserves the whole GPU and the
    rest cannot start."""
    monkeypatch.delenv("TF_FORCE_GPU_ALLOW_GROWTH", raising=False)
    monkeypatch.setattr("harmonie.workers.EffnetExtractor", lambda: object())

    _worker_init("WARNING")

    assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"


def test_worker_respects_an_explicit_setting(monkeypatch):
    monkeypatch.setenv("TF_FORCE_GPU_ALLOW_GROWTH", "false")
    monkeypatch.setattr("harmonie.workers.EffnetExtractor", lambda: object())

    _worker_init("WARNING")

    assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "false"


def test_pool_reports_a_visible_card(monkeypatch, caplog):
    """The log line is the only way a user can tell the GPU is in use."""
    monkeypatch.setattr(workers, "prefetch_models", lambda *a, **k: True)
    monkeypatch.setattr(workers, "cuda_device_count", lambda: 2)
    monkeypatch.setattr(workers, "_gpu_usable", lambda: True)
    monkeypatch.setattr(workers.mp, "get_context", lambda name: _StubContext())

    with caplog.at_level("INFO"):
        workers.WorkerPool(workers=2)

    assert "2 CUDA device(s) visible" in caplog.text
    assert "experimental" in caplog.text


def test_pool_falls_back_to_cpu_when_the_gpu_is_unusable(monkeypatch, caplog):
    """TensorFlow aborts on a visible-but-broken device, which would kill every
    worker and leave the scan making no progress."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(workers, "prefetch_models", lambda *a, **k: True)
    monkeypatch.setattr(workers, "cuda_device_count", lambda: 1)
    monkeypatch.setattr(workers, "_gpu_usable", lambda: False)
    monkeypatch.setattr(workers.mp, "get_context", lambda name: _StubContext())

    with caplog.at_level("INFO"):
        workers.WorkerPool(workers=2)

    assert "continuing on CPU" in caplog.text
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_probe_runs_out_of_process(monkeypatch):
    """In-process it would abort the parent, taking the whole scan with it."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["timeout"] = kwargs.get("timeout")

        class Done:
            returncode = 0

        return Done()

    monkeypatch.setattr(workers.subprocess, "run", fake_run)
    assert workers._gpu_usable() is True
    assert seen["argv"][0] == workers.sys.executable
    assert "EffnetExtractor" in seen["argv"][2]
    assert seen["timeout"] == workers.GPU_PROBE_TIMEOUT_SEC


def test_probe_treats_a_crash_as_unusable(monkeypatch):
    class Crashed:
        returncode = -6  # SIGABRT, which is how TensorFlow exits here

    monkeypatch.setattr(workers.subprocess, "run", lambda *a, **k: Crashed())
    assert workers._gpu_usable() is False


def test_probe_timeout_is_unusable(monkeypatch):
    def hangs(*a, **k):
        raise workers.subprocess.TimeoutExpired(cmd="probe", timeout=1)

    monkeypatch.setattr(workers.subprocess, "run", hangs)
    assert workers._gpu_usable() is False


def test_no_probe_without_a_visible_card(monkeypatch):
    """Hosts without a GPU must not pay for a probe."""
    monkeypatch.setattr(workers, "cuda_device_count", lambda: 0)
    monkeypatch.setattr(
        workers, "_gpu_usable", lambda: pytest.fail("probe should not run")
    )
    workers._configure_gpu()


def test_pool_says_nothing_without_a_card(monkeypatch, caplog):
    monkeypatch.setattr(workers, "prefetch_models", lambda *a, **k: True)
    monkeypatch.setattr(workers, "cuda_device_count", lambda: 0)
    monkeypatch.setattr(workers.mp, "get_context", lambda name: _StubContext())

    with caplog.at_level("INFO"):
        workers.WorkerPool(workers=2)

    assert "CUDA" not in caplog.text


@pytest.mark.parametrize("count", [0, 1, 8])
def test_count_is_never_negative(monkeypatch, count):
    monkeypatch.setattr(ctypes, "CDLL", lambda name: _FakeDriver(count))
    assert cuda_device_count() == count
