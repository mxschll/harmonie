"""Model cache download behaviour.

A fresh install starts one worker per core, and every worker used to download
the models itself: they raced on a shared ``.part`` file (workers died with
FileNotFoundError) and hammered a flaky host with N copies of the same request
(timeouts, after which tracks were indexed with no style data at all).
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import pytest

from harmonie import features


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(features, "_model_cache_dir", lambda: models)
    return models


def test_concurrent_callers_download_once(cache_dir: Path, monkeypatch):
    payload = b"model-bytes" * 1000
    calls: list[str] = []
    started = threading.Event()

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        # Hold the connection open so every thread is inside the critical
        # section at the same time if the lock is not doing its job.
        started.set()
        return _FakeResponse(payload)

    monkeypatch.setattr(features.urllib.request, "urlopen", fake_urlopen)

    results: list[Path] = []
    errors: list[BaseException] = []

    def run():
        try:
            results.append(features.ensure_effnet_model())
        except BaseException as exc:  # noqa: BLE001 - recorded for the assert
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 8
    assert len({str(p) for p in results}) == 1
    assert results[0].read_bytes() == payload
    # One download for eight callers, not eight.
    assert len(calls) == 1
    # No temp or lock leftovers next to the model.
    assert sorted(p.name for p in cache_dir.iterdir()) == [
        features.EFFNET_MODEL_FILENAME,
        features.EFFNET_MODEL_FILENAME + ".lock",
    ]


def test_download_retries_a_flaky_host(cache_dir: Path, monkeypatch):
    payload = b"eventually"
    attempts = {"n": 0}

    def fake_urlopen(url, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("Operation timed out")
        return _FakeResponse(payload)

    monkeypatch.setattr(features.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(features, "DOWNLOAD_RETRY_SLEEP_SEC", 0)

    path = features.ensure_effnet_model()

    assert path.read_bytes() == payload
    assert attempts["n"] == 3


def test_download_gives_up_and_raises(cache_dir: Path, monkeypatch):
    def fake_urlopen(url, timeout=None):
        raise TimeoutError("Operation timed out")

    monkeypatch.setattr(features.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(features, "DOWNLOAD_RETRY_SLEEP_SEC", 0)

    with pytest.raises(TimeoutError):
        features.ensure_effnet_model()

    assert not (cache_dir / features.EFFNET_MODEL_FILENAME).exists()
    # A failed attempt must not leave a partial file that later looks cached.
    assert [p.name for p in cache_dir.iterdir()] == [
        features.EFFNET_MODEL_FILENAME + ".lock"
    ]


def test_prefetch_reports_a_missing_style_classifier(cache_dir: Path, monkeypatch):
    """A failed genre-head fetch must be visible, not silently degrade."""
    payload = b"effnet"

    def fake_urlopen(url, timeout=None):
        if url == features.EFFNET_MODEL_URL:
            return _FakeResponse(payload)
        raise TimeoutError("Operation timed out")

    monkeypatch.setattr(features.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(features, "DOWNLOAD_RETRY_SLEEP_SEC", 0)

    assert features.prefetch_models() is False
    assert (cache_dir / features.EFFNET_MODEL_FILENAME).exists()


def test_prefetch_returns_true_when_everything_is_cached(cache_dir: Path, monkeypatch):
    labels = {
        "classes": [f"Genre---Style{i}" for i in range(features.GENRE_NUM_CLASSES)]
    }

    import json

    def fake_urlopen(url, timeout=None):
        if url == features.GENRE_HEAD_LABELS_URL:
            return _FakeResponse(json.dumps(labels).encode())
        return _FakeResponse(b"model")

    monkeypatch.setattr(features.urllib.request, "urlopen", fake_urlopen)

    assert features.prefetch_models() is True
