"""Smoke tests for ``harmonie`` CLI commands.

These don't run the actual extractor (no audio files, no TF). Instead each
test populates the SQLite DB directly via the same helpers the unit tests
use, points the CLI's ``get_settings`` at the temp DB, and asserts on the
formatted output.

The point is to catch regressions in CLI plumbing — argument parsing, output
formatting, exit codes — that the API tests don't exercise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harmonie import cli as cli_mod
from harmonie.cli import main
from harmonie.config import Settings
from harmonie.features import DESCRIPTOR_VERSION, Descriptors
from harmonie.tags import Tags


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _populate(
    db, lib_root: Path, name: str, *,
    bpm: float, key: str = "A", scale: str = "minor",
):
    path = str(lib_root / name)
    return db.upsert_track(
        path=path,
        size=100,
        mtime=1.0,
        duration=180.0,
        embedding=np.ones(4, dtype=np.float32),
        model="m1",
        descriptors=Descriptors(
            bpm=bpm, key=key, scale=scale,
            loudness=-12.0, danceability=1.5, onset_rate=4.2,
        ),
        descriptor_version=DESCRIPTOR_VERSION,
        tags=Tags(artist=name, title=name.replace(".flac", "")),
        library_root=str(lib_root),
        relative_path=name,
    )


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch):
    """Build a populated DB at the same path the CLI will open, and patch
    ``cli.get_settings`` to point at it. Returns ``(db, settings, lib)``,
    where ``lib`` is the (hermetic, tmp_path-rooted) library directory the
    fake tracks live under — important because some CLI commands resolve
    user-supplied paths through symlinks (and ``/lib`` is a symlink on
    most Linux distros)."""
    from harmonie.db import Database

    settings = Settings(libraries=[tmp_path], data_dir=tmp_path)
    db = Database(settings.db_path)

    lib = tmp_path / "library"
    lib.mkdir()

    _populate(db, lib, "fast.flac", bpm=140)
    _populate(db, lib, "mid.flac", bpm=120)
    _populate(db, lib, "slow.flac", bpm=80)

    # CLI commands close the DB they open — leave our handle live for the
    # test, then close at teardown.
    monkeypatch.setattr(cli_mod, "get_settings", lambda: settings)
    yield db, settings, lib
    db.close()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_human_output_includes_counts(self, cli_env, capsys):
        rc = main(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Tracks:" in out
        assert "3" in out  # 3 tracks populated
        assert "DB size:" in out
        assert "Total audio:" in out

    def test_json_output_is_valid(self, cli_env, capsys):
        rc = main(["status", "--json"])
        assert rc == 0
        body = json.loads(capsys.readouterr().out)
        assert body["tracks"] == 3
        assert "by_model" in body
        assert "db_path" in body


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_list_all(self, cli_env, capsys):
        rc = main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        # Header + 3 rows + summary line.
        assert "fast.flac" in out
        assert "mid.flac" in out
        assert "slow.flac" in out
        assert "3 of 3" in out

    def test_list_with_bpm_range(self, cli_env, capsys):
        """The CLI uses the same 120..130 syntax as the API."""
        rc = main(["list", "--bpm", "100..130"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "mid.flac" in out
        assert "fast.flac" not in out  # 140 is excluded
        assert "slow.flac" not in out  # 80 is excluded

    def test_list_with_invalid_range_exits_1(self, cli_env, capsys):
        rc = main(["list", "--bpm", "garbage..nonsense"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "invalid range" in err

    def test_list_json_is_parseable(self, cli_env, capsys):
        rc = main(["list", "--json"])
        assert rc == 0
        body = json.loads(capsys.readouterr().out)
        assert body["total"] == 3
        assert len(body["items"]) == 3


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


class TestInfo:
    def test_info_by_id(self, cli_env, capsys):
        db, _, _ = cli_env
        track_id = next(iter(db.list_tracks(limit=1)[0]))["id"]
        rc = main(["info", str(track_id)])
        assert rc == 0
        out = capsys.readouterr().out
        assert f"ID:" in out
        assert "BPM:" in out
        assert "Key:" in out
        assert "Loudness" in out

    def test_info_by_path(self, cli_env, capsys):
        _db, _settings, lib = cli_env
        track_path = str(lib / "mid.flac")
        rc = main(["info", track_path])
        assert rc == 0
        out = capsys.readouterr().out
        assert track_path in out
        # mid was populated with bpm=120.
        assert "120" in out

    def test_info_missing_exits_1(self, cli_env, capsys):
        rc = main(["info", "/not/in/db.flac"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Not in database" in err

    def test_info_json(self, cli_env, capsys):
        _db, _settings, lib = cli_env
        track_path = str(lib / "mid.flac")
        rc = main(["info", track_path, "--json"])
        assert rc == 0
        body = json.loads(capsys.readouterr().out)
        assert body["bpm"] == 120
        assert body["path"] == track_path


# ---------------------------------------------------------------------------
# similar
# ---------------------------------------------------------------------------


class TestSimilar:
    def test_similar_returns_other_tracks(self, cli_env, capsys):
        db, _, _ = cli_env
        track_id = next(iter(db.list_tracks(limit=1)[0]))["id"]
        rc = main(["similar", str(track_id), "-n", "5"])
        assert rc == 0
        out = capsys.readouterr().out
        # Header-less output; one ranked line per match.
        # All embeddings are equal in our fixture, so every other track
        # comes back with score ~1.0.
        assert "1.0000" in out or "0.9999" in out
        # Self should not appear by default.
        for line in out.splitlines():
            assert f"[{track_id}]" not in line

    def test_similar_unknown_id_exits_1(self, cli_env, capsys):
        rc = main(["similar", "999999"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "999999" in err
