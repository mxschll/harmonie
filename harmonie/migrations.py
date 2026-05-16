"""Schema migrations.

Append-only list of migration functions. Each takes an open
:class:`sqlite3.Connection` and applies its DDL/DML. Migrations run in order
on every DB open; the runner is idempotent (a no-op when the DB is already
at the latest version).

The DB stores the version it has been migrated to in ``meta.schema_version``.
A migration runs in its own transaction — a partial failure rolls back and
leaves the DB at the previous version.

Refusing to run a binary older than the DB is a feature: if a newer harmonie
has migrated the DB to v5 and somebody starts an old harmonie that only
knows about v3, that older binary doesn't know how to read v5 and may
silently corrupt data. We fail loudly instead.

Adding a migration:

1. Write ``_migration_NNN_what_it_does(conn)`` below.
2. Append it to :data:`MIGRATIONS`.
3. Update any code in ``db.py`` that depends on the new shape (column lists
   in upserts, etc.).

Don't edit existing migration functions after they've shipped — they're a
historical record. New changes are new migrations.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable

logger = logging.getLogger("harmonie.migrations")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MigrationError(RuntimeError):
    """Raised when migrations cannot proceed (e.g. the DB is from a newer
    harmonie binary than the one trying to open it)."""


# ---------------------------------------------------------------------------
# Migration 001: initial schema
# ---------------------------------------------------------------------------


_M001_STATEMENTS = [
    """
    CREATE TABLE tracks (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        path                  TEXT    UNIQUE NOT NULL,
        -- Library-aware addressing for clients that mount paths differently.
        library_root          TEXT,
        relative_path         TEXT,
        size                  INTEGER NOT NULL,
        mtime                 REAL    NOT NULL,
        duration              REAL    NOT NULL,
        embedding             BLOB    NOT NULL,
        embedding_dim         INTEGER NOT NULL,
        model                 TEXT    NOT NULL,
        -- Versioning so a descriptor algo bump doesn't force re-running TF.
        descriptor_version    INTEGER NOT NULL,
        -- Musical descriptors. NULL = algorithm couldn't extract.
        bpm                   REAL,
        bpm_confidence        REAL,
        key                   TEXT,
        scale                 TEXT,
        key_strength          REAL,
        loudness_db           REAL,
        danceability          REAL,
        onset_rate            REAL,
        -- Tags from the file itself (mutagen). Used by external clients to
        -- match harmonie tracks back to their own catalog without
        -- filesystem walks.
        artist                TEXT,
        album                 TEXT,
        title                 TEXT,
        track_number          INTEGER,
        -- Full 400-d Discogs style activation vector (float32 BLOB) from
        -- the genre classifier head. Top-K labels broken out into the
        -- track_styles table for fast filtering. NULL = no styles extracted
        -- (e.g. musicextractor backend, or the head was unavailable at
        -- scan time).
        style_activations     BLOB,
        analyzed_at           REAL    NOT NULL
    )
    """,
    "CREATE INDEX idx_tracks_model       ON tracks(model)",
    "CREATE INDEX idx_tracks_bpm         ON tracks(bpm)",
    "CREATE INDEX idx_tracks_key_scale   ON tracks(key, scale)",
    "CREATE INDEX idx_tracks_dance       ON tracks(danceability)",
    "CREATE INDEX idx_tracks_loud        ON tracks(loudness_db)",
    "CREATE INDEX idx_tracks_descv       ON tracks(descriptor_version)",
    "CREATE INDEX idx_tracks_lib         ON tracks(library_root)",
    # Composite NOCASE index for the /tracks/lookup endpoint's tag triple.
    "CREATE INDEX idx_tracks_artist_album_title "
    "ON tracks(artist COLLATE NOCASE, album COLLATE NOCASE, title COLLATE NOCASE)",
    # Top-K style probabilities per track. Lookup by style is the common
    # filter direction; lookup by track is used to enrich track responses.
    """
    CREATE TABLE track_styles (
        track_id     INTEGER NOT NULL,
        style        TEXT    NOT NULL,
        probability  REAL    NOT NULL,
        PRIMARY KEY (track_id, style),
        FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX idx_track_styles_style ON track_styles(style)",
    "CREATE INDEX idx_track_styles_prob  ON track_styles(style, probability DESC)",
]


def _migration_001_initial(conn: sqlite3.Connection) -> None:
    """Initial schema. The tracks table with embedding + descriptors + tags +
    library-relative paths, plus all current indexes."""
    for stmt in _M001_STATEMENTS:
        conn.execute(stmt)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Append-only. Each entry's index + 1 is the version it brings the DB to.
MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    _migration_001_initial,
]

CURRENT_SCHEMA_VERSION = len(MIGRATIONS)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Return the version the DB has been migrated to. 0 for a fresh DB."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
    )
    if cur.fetchone() is None:
        return 0
    row = conn.execute(
        "SELECT value FROM meta WHERE key='schema_version'"
    ).fetchone()
    return int(row[0]) if row else 0


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta ("
        "    key TEXT PRIMARY KEY,"
        "    value TEXT NOT NULL"
        ")"
    )
    conn.commit()


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply pending migrations. Idempotent.

    Raises :class:`MigrationError` if the DB is at a version this binary
    doesn't know about (i.e. it was written by a newer harmonie).
    """
    _ensure_meta_table(conn)
    current = get_schema_version(conn)

    if current > CURRENT_SCHEMA_VERSION:
        raise MigrationError(
            f"database is at schema version {current}, but this harmonie "
            f"binary only supports up to version {CURRENT_SCHEMA_VERSION}. "
            f"Refusing to run — upgrade the binary, or restore an older "
            f"snapshot of the database."
        )

    if current == CURRENT_SCHEMA_VERSION:
        logger.debug("schema up to date at version %d", current)
        return

    pending = list(range(current, CURRENT_SCHEMA_VERSION))
    logger.info(
        "applying %d migration(s): %s",
        len(pending), ", ".join(str(i + 1) for i in pending),
    )

    # Python's sqlite3 driver, in its default isolation mode, implicitly
    # commits any pending transaction *before* executing a DDL statement
    # (CREATE TABLE, ALTER, DROP, …). That means a `CREATE TABLE` inside a
    # migration that later fails is already committed — `conn.rollback()`
    # has nothing to undo. To get real transactional DDL we switch to
    # autocommit mode (isolation_level=None) for the duration of the
    # migration loop and bracket each migration with explicit BEGIN/COMMIT
    # /ROLLBACK statements.
    original_isolation = conn.isolation_level
    conn.isolation_level = None
    try:
        for i in pending:
            version = i + 1
            fn = MIGRATIONS[i]
            logger.info("migration %d → applying %s", version, fn.__name__)
            try:
                conn.execute("BEGIN")
                fn(conn)
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) "
                    "VALUES('schema_version', ?)",
                    (str(version),),
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    # No active transaction (unlikely, but harmless).
                    pass
                logger.exception(
                    "migration %d (%s) failed; rolled back to version %d",
                    version, fn.__name__, version - 1,
                )
                raise
    finally:
        conn.isolation_level = original_isolation

    logger.info("schema migrated to version %d", CURRENT_SCHEMA_VERSION)
