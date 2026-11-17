"""Smoke tests against a built Docker image.

Skipped unless ``--docker-image`` names an image, e.g.::

    docker build -t harmonie:dev .
    pytest tests/test_docker.py --docker-image harmonie:dev

The image is linux/amd64. On macOS with colima or Docker Desktop, pytest's
temp dir may not be shared with the VM; pass ``--basetemp ~/tmp`` if mounts
come up empty.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from tests.audio_fixtures import tone, write_flac, write_spliced_flac, write_wav

# Model loading in a cold container is the slow part; one tiny file per
# format after that is seconds.
SCAN_TIMEOUT_SEC = 300


@pytest.fixture(scope="module")
def image(request) -> str:
    name = request.config.getoption("--docker-image")
    if not name:
        pytest.skip("pass --docker-image to run against a built image")
    return name


def _docker(*args: str, timeout: float = 60, check: bool = True) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"docker {' '.join(args[:2])} failed ({proc.returncode}):\n{proc.stderr}"
        )
    return proc.stdout


@pytest.fixture(scope="module")
def library(tmp_path_factory) -> Path:
    """Three good files across formats plus the issue #9 splice. Five seconds
    each: the embedding model needs at least one 2.5 s window."""
    lib = tmp_path_factory.mktemp("music")
    a = tone(44100, FIXTURE_SECONDS, 440)
    b = tone(44100, FIXTURE_SECONDS, 660)
    write_wav(lib / "stereo.wav", 44100, np.stack([a, b]))
    write_flac(lib / "stereo.flac", 44100, "stereo", np.stack([a, b]))
    write_flac(
        lib / "mono48k.flac", 48000, "mono", tone(48000, FIXTURE_SECONDS, 220)[None, :]
    )
    write_spliced_flac(lib / "spliced.flac")
    os.chmod(lib, 0o755)
    return lib


FIXTURE_SECONDS = 5.0
GOOD_FILES = {"stereo.wav", "stereo.flac", "mono48k.flac"}


def _mounts(library: Path, data: Path) -> list[str]:
    # The container runs as the host user so the database it writes under
    # /data is removable by pytest's temp-dir cleanup.
    return [
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HARMONIE_WORKERS=1",
        "-v",
        f"{data}:/data",
        "-v",
        f"{library}:/music:ro",
    ]


def test_one_shot_scan_indexes_good_files_and_fails_the_splice(
    image, library, tmp_path
):
    """``docker run IMAGE scan`` indexes every good file, fails the
    parameter-switching FLAC quickly instead of hanging on it, and exits 2
    (the CLI's "some files failed")."""
    data = tmp_path / "data"
    data.mkdir()
    os.chmod(data, 0o777)

    started = time.monotonic()
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "run", "--rm", "--init", *_mounts(library, data), image, "scan"],
        capture_output=True,
        text=True,
        timeout=SCAN_TIMEOUT_SEC,
        check=False,
    )
    elapsed = time.monotonic() - started

    assert proc.returncode == 2, proc.stderr
    summary = next(
        line for line in proc.stdout.splitlines() if line.startswith("Discovered:")
    )
    assert "Full: 3" in summary, summary
    assert "Failed: 1" in summary, summary
    assert "spliced.flac" in proc.stderr
    # A hang would have hit the subprocess timeout; the per-job watchdog
    # defaults to 300 s, so finishing well inside that means no job hung.
    assert elapsed < 200, f"scan took {elapsed:.0f}s"
    assert (data / "harmonie.db").exists()


@pytest.fixture
def served(image, library, tmp_path):
    """A container running ``serve`` with a startup scan, on a random port.
    Yields the base URL; removes the container afterwards. Container logs are
    printed so pytest shows them when a test fails."""
    data = tmp_path / "data"
    data.mkdir()
    os.chmod(data, 0o777)
    container = _docker(
        "run",
        "-d",
        "--init",
        "-p",
        "127.0.0.1:0:8842",
        *_mounts(library, data),
        image,
    ).strip()
    try:
        port = _docker("port", container, "8842").strip().rsplit(":", 1)[-1]
        yield f"http://127.0.0.1:{port}", container
    finally:
        print(_docker("logs", container, check=False))
        _docker("rm", "-f", container, check=False)


def _get(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - local
        return json.load(resp)


def _wait_for(predicate, *, timeout: float, what: str):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
            if last:
                return last
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    raise AssertionError(f"timed out waiting for {what}; last={last!r}")


def test_serve_scans_on_startup_and_answers_queries(served):
    base, container = served

    _wait_for(
        lambda: _get(f"{base}/health").get("status") == "ok", timeout=60, what="health"
    )

    def scan_finished():
        s = _get(f"{base}/api/v1/scan")
        return s if s["state"] == "idle" and s["finished_at"] else None

    status = _wait_for(scan_finished, timeout=SCAN_TIMEOUT_SEC, what="startup scan")
    assert status["full"] == 3, status
    assert status["failed"] == 1, status
    assert [f for f, _ in status["recent_failures"]] == ["/music/spliced.flac"], status

    tracks = _get(f"{base}/api/v1/tracks?limit=10")
    assert {Path(t["path"]).name for t in tracks["items"]} == GOOD_FILES
    for t in tracks["items"]:
        assert t["duration"] == pytest.approx(FIXTURE_SECONDS, abs=0.01)
        assert t["styles"], f"no style classification for {t['path']}"

    track_id = tracks["items"][0]["id"]
    similar = _get(f"{base}/api/v1/tracks/{track_id}/similar?limit=5")
    assert similar["query_id"] == track_id
    assert len(similar["matches"]) == 2
    assert all(0.0 <= m["score"] <= 1.0 for m in similar["matches"])


def test_serve_healthcheck_reports_healthy(served):
    """The image's HEALTHCHECK must pass inside its start period, otherwise
    orchestrators restart a working container."""
    _, container = served

    def healthy():
        state = json.loads(_docker("inspect", container))[0]["State"]
        return state["Health"]["Status"] == "healthy"

    # interval 30s + start-period 30s: the first passing probe lands around
    # 30-60 s after start.
    _wait_for(healthy, timeout=120, what="healthcheck")
