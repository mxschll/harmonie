"""Tests for the migration runner."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from harmonie import migrations as migmod
from harmonie.db import Database
from harmonie.migrations import (
    CURRENT_SCHEMA_VERSION,
    MIGRATIONS,
    MigrationError,
    get_schema_version,
    run_migrations,
)

# ---------------------------------------------------------------------------
# Fresh / idempotent paths
# ---------------------------------------------------------------------------


def test_fresh_db_applies_all_migrations(tmp_path: Path):
    db_path = tmp_path / "fresh.db"
    conn = sqlite3.connect(db_path)
    try:
        assert get_schema_version(conn) == 0
        run_migrations(conn)
        assert get_schema_version(conn) == CURRENT_SCHEMA_VERSION
        # The tracks table now exists.
        names = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "tracks" in names
        assert "meta" in names
    finally:
        conn.close()


def test_running_twice_is_a_noop(tmp_path: Path):
    db_path = tmp_path / "twice.db"
    conn = sqlite3.connect(db_path)
    try:
        run_migrations(conn)
        v1 = get_schema_version(conn)
        run_migrations(conn)
        v2 = get_schema_version(conn)
        assert v1 == v2 == CURRENT_SCHEMA_VERSION
    finally:
        conn.close()


def test_database_constructor_runs_migrations(tmp_path: Path):
    """Opening a Database on a fresh file should leave it at the current version."""
    db = Database(tmp_path / "via-class.db")
    try:
        assert get_schema_version(db._conn) == CURRENT_SCHEMA_VERSION
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Refusal to downgrade
# ---------------------------------------------------------------------------


def test_refuses_db_at_higher_version(tmp_path: Path):
    """If a newer harmonie wrote a DB at version N+1 and an older one tries
    to open it, we fail loudly rather than risk silent data loss."""
    db_path = tmp_path / "future.db"
    conn = sqlite3.connect(db_path)
    try:
        run_migrations(conn)  # bring to current
        # Pretend a future binary bumped it further.
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(CURRENT_SCHEMA_VERSION + 5),),
        )
        conn.commit()
        with pytest.raises(MigrationError):
            run_migrations(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Partial-failure rollback
# ---------------------------------------------------------------------------


def test_failed_migration_rolls_back(tmp_path: Path, monkeypatch):
    """If migration N raises mid-way, the runner rolls back its transaction
    and leaves the DB at version N-1. Subsequent migrations don't run."""
    db_path = tmp_path / "fail.db"

    # Bring the DB to current with the real migrations.
    conn = sqlite3.connect(db_path)
    try:
        run_migrations(conn)
    finally:
        conn.close()

    def good(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE good_added (id INTEGER PRIMARY KEY);")

    def bad(conn: sqlite3.Connection) -> None:
        # Make a partial change, then fail. The transactional runner must
        # ensure this change is rolled back.
        conn.execute("CREATE TABLE bad_added (id INTEGER PRIMARY KEY);")
        raise RuntimeError("something exploded")

    fake_migrations = list(MIGRATIONS) + [good, bad]
    monkeypatch.setattr(migmod, "MIGRATIONS", fake_migrations)
    monkeypatch.setattr(migmod, "CURRENT_SCHEMA_VERSION", len(fake_migrations))

    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(RuntimeError, match="something exploded"):
            run_migrations(conn)

        # The 'good' migration should have committed (it was a separate
        # transaction). The 'bad' migration's CREATE TABLE must be rolled
        # back, and the version should record only the successful step.
        version = get_schema_version(conn)
        assert version == CURRENT_SCHEMA_VERSION + 1

        names = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "good_added" in names  # the prior successful migration stuck
        assert "bad_added" not in names  # the failing migration was rolled back
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema sanity: migration 001 produces the expected columns
# ---------------------------------------------------------------------------


def test_migration_001_columns_match_db_layer_expectations(tmp_path: Path):
    """The columns referenced by db.upsert_track and friends must exist."""
    conn = sqlite3.connect(tmp_path / "schema.db")
    try:
        run_migrations(conn)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(tracks)")}
    finally:
        conn.close()
    expected = {
        "id", "path", "library_root", "relative_path",
        "size", "mtime", "duration",
        "embedding", "embedding_dim", "model",
        "descriptor_version",
        "bpm", "bpm_confidence", "key", "scale", "key_strength",
        "loudness", "danceability", "onset_rate",
        "artist", "album", "title", "track_number",
        "analyzed_at",
    }
    assert expected <= cols, f"missing columns: {expected - cols}"
