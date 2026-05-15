"""SQLite storage for analyzed tracks. Single-writer, many-reader.

Schema design notes:

* Tracks are addressed by integer ``id`` (autoincrement). The path is exposed
  as a property of the row — clients should not depend on path stability.
* ``embedding`` is a contiguous ``float32`` blob; ``embedding_dim`` records
  the length so we can reshape on read without binding to a particular model.
* ``model`` and ``descriptor_version`` are tracked separately, so a descriptor
  algorithm bump doesn't require re-running the embedding model. Existing rows
  with older ``descriptor_version`` are topped up cheaply.
* When a file disappears between scans, its row is deleted (we don't try to
  detect moves — a moved file looks like delete+add, which is fine).
* Filter columns (bpm, key, scale, danceability, loudness_db) are indexed
  individually so playlist-style range queries stay fast on large libraries.
* WAL mode lets the analyzer write while the API serves reads.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from .features import Descriptors
from .tags import Tags

SCHEMA_VERSION = 1


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks (
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
    -- match harmonie tracks back to their own catalog without filesystem walks.
    artist                TEXT,
    album                 TEXT,
    title                 TEXT,
    track_number          INTEGER,
    musicbrainz_track_id  TEXT,
    analyzed_at           REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tracks_model       ON tracks(model);
CREATE INDEX IF NOT EXISTS idx_tracks_bpm         ON tracks(bpm);
CREATE INDEX IF NOT EXISTS idx_tracks_key_scale   ON tracks(key, scale);
CREATE INDEX IF NOT EXISTS idx_tracks_dance       ON tracks(danceability);
CREATE INDEX IF NOT EXISTS idx_tracks_loud        ON tracks(loudness_db);
CREATE INDEX IF NOT EXISTS idx_tracks_descv       ON tracks(descriptor_version);
CREATE INDEX IF NOT EXISTS idx_tracks_lib         ON tracks(library_root);
CREATE INDEX IF NOT EXISTS idx_tracks_mbid        ON tracks(musicbrainz_track_id);
"""


# ---------------------------------------------------------------------------
# Filter spec used by API and similarity
# ---------------------------------------------------------------------------


class TrackFilter:
    """Optional descriptor-based filters for list / similar / playlist queries.

    All fields are inclusive ranges or set memberships. ``None`` = no filter.
    """

    __slots__ = (
        "bpm_min", "bpm_max", "key", "scale",
        "danceability_min", "danceability_max",
        "loudness_min", "loudness_max",
    )

    def __init__(
        self,
        *,
        bpm_min: Optional[float] = None,
        bpm_max: Optional[float] = None,
        key: Optional[list[str]] = None,
        scale: Optional[str] = None,
        danceability_min: Optional[float] = None,
        danceability_max: Optional[float] = None,
        loudness_min: Optional[float] = None,
        loudness_max: Optional[float] = None,
    ) -> None:
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.key = list(key) if key else None
        self.scale = scale
        self.danceability_min = danceability_min
        self.danceability_max = danceability_max
        self.loudness_min = loudness_min
        self.loudness_max = loudness_max

    def to_sql(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.bpm_min is not None:
            clauses.append("bpm >= ?")
            params.append(float(self.bpm_min))
        if self.bpm_max is not None:
            clauses.append("bpm <= ?")
            params.append(float(self.bpm_max))
        if self.key:
            placeholders = ",".join("?" * len(self.key))
            clauses.append(f"key IN ({placeholders})")
            params.extend(self.key)
        if self.scale:
            clauses.append("scale = ?")
            params.append(self.scale)
        if self.danceability_min is not None:
            clauses.append("danceability >= ?")
            params.append(float(self.danceability_min))
        if self.danceability_max is not None:
            clauses.append("danceability <= ?")
            params.append(float(self.danceability_max))
        if self.loudness_min is not None:
            clauses.append("loudness_db >= ?")
            params.append(float(self.loudness_min))
        if self.loudness_max is not None:
            clauses.append("loudness_db <= ?")
            params.append(float(self.loudness_max))
        if not clauses:
            return "", []
        return " AND ".join(clauses), params

    def is_empty(self) -> bool:
        return all(getattr(self, slot) is None for slot in self.__slots__)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def _row_to_dict(row: sqlite3.Row, *, include_embedding: bool = False) -> dict:
    d = dict(row)
    if not include_embedding:
        d.pop("embedding", None)
    return d


def _is_under(path: Path, root: Path) -> bool:
    """True if ``path`` is the same as ``root`` or a descendant of it.

    Uses Path.is_relative_to where available (3.9+); falls back to a string
    prefix check that respects path separators.
    """
    try:
        return path == root or path.is_relative_to(root)
    except AttributeError:  # pragma: no cover - Python < 3.9
        ps = str(path)
        rs = str(root)
        return ps == rs or ps.startswith(rs + os.sep)


class Database:
    """SQLite wrapper. Construct one per process; share read-only by passing
    the same path. Writes are serialized internally."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # -- schema --------------------------------------------------------

    def _init_schema(self) -> None:
        # All DDL is idempotent; safe to run on every open. The meta row
        # records the schema version so that, when the service ships and we
        # need a real migration story, this is the hook point. Until then
        # we don't ship breaking changes — schema edits during development
        # are accompanied by deleting the local DB file.
        self._conn.executescript(SCHEMA_SQL)
        self._conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    # -- lifecycle -----------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # -- writes --------------------------------------------------------

    def upsert_track(
        self,
        *,
        path: str,
        size: int,
        mtime: float,
        duration: float,
        embedding: np.ndarray,
        model: str,
        descriptors: Descriptors,
        descriptor_version: int,
        tags: Optional[Tags] = None,
        library_root: Optional[str] = None,
        relative_path: Optional[str] = None,
    ) -> int:
        emb = np.ascontiguousarray(embedding.astype(np.float32, copy=False))
        t = tags or Tags()
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO tracks (
                    path, library_root, relative_path,
                    size, mtime, duration, embedding, embedding_dim, model,
                    descriptor_version,
                    bpm, bpm_confidence, key, scale, key_strength,
                    loudness_db, danceability, onset_rate,
                    artist, album, title, track_number, musicbrainz_track_id,
                    analyzed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    library_root         = excluded.library_root,
                    relative_path        = excluded.relative_path,
                    size                 = excluded.size,
                    mtime                = excluded.mtime,
                    duration             = excluded.duration,
                    embedding            = excluded.embedding,
                    embedding_dim        = excluded.embedding_dim,
                    model                = excluded.model,
                    descriptor_version   = excluded.descriptor_version,
                    bpm                  = excluded.bpm,
                    bpm_confidence       = excluded.bpm_confidence,
                    key                  = excluded.key,
                    scale                = excluded.scale,
                    key_strength         = excluded.key_strength,
                    loudness_db          = excluded.loudness_db,
                    danceability         = excluded.danceability,
                    onset_rate           = excluded.onset_rate,
                    artist               = excluded.artist,
                    album                = excluded.album,
                    title                = excluded.title,
                    track_number         = excluded.track_number,
                    musicbrainz_track_id = excluded.musicbrainz_track_id,
                    analyzed_at          = excluded.analyzed_at
                """,
                (
                    path,
                    library_root,
                    relative_path,
                    int(size),
                    float(mtime),
                    float(duration),
                    emb.tobytes(),
                    int(emb.shape[0]),
                    model,
                    int(descriptor_version),
                    descriptors.bpm,
                    descriptors.bpm_confidence,
                    descriptors.key,
                    descriptors.scale,
                    descriptors.key_strength,
                    descriptors.loudness_db,
                    descriptors.danceability,
                    descriptors.onset_rate,
                    t.artist,
                    t.album,
                    t.title,
                    t.track_number,
                    t.musicbrainz_track_id,
                    time.time(),
                ),
            )
            cur.execute("SELECT id FROM tracks WHERE path = ?", (path,))
            return int(cur.fetchone()[0])

    def update_descriptors(
        self,
        path: str,
        *,
        descriptors: Descriptors,
        descriptor_version: int,
        duration: Optional[float] = None,
        tags: Optional[Tags] = None,
    ) -> bool:
        """Refresh descriptor + tag columns without touching the embedding."""
        t = tags or Tags()
        sets = [
            "descriptor_version = ?",
            "bpm                = ?",
            "bpm_confidence     = ?",
            "key                = ?",
            "scale              = ?",
            "key_strength       = ?",
            "loudness_db        = ?",
            "danceability       = ?",
            "onset_rate         = ?",
            "artist             = ?",
            "album              = ?",
            "title              = ?",
            "track_number       = ?",
            "musicbrainz_track_id = ?",
            "analyzed_at        = ?",
        ]
        params: list = [
            int(descriptor_version),
            descriptors.bpm,
            descriptors.bpm_confidence,
            descriptors.key,
            descriptors.scale,
            descriptors.key_strength,
            descriptors.loudness_db,
            descriptors.danceability,
            descriptors.onset_rate,
            t.artist,
            t.album,
            t.title,
            t.track_number,
            t.musicbrainz_track_id,
            time.time(),
        ]
        if duration is not None:
            sets.insert(0, "duration = ?")
            params.insert(0, float(duration))
        params.append(path)
        with self.transaction() as cur:
            cur.execute(
                f"UPDATE tracks SET {', '.join(sets)} WHERE path = ?",
                tuple(params),
            )
            return cur.rowcount > 0

    def remove_by_path(self, path: str) -> int:
        with self.transaction() as cur:
            cur.execute("DELETE FROM tracks WHERE path = ?", (path,))
            return cur.rowcount

    def remove_by_id(self, track_id: int) -> int:
        with self.transaction() as cur:
            cur.execute("DELETE FROM tracks WHERE id = ?", (int(track_id),))
            return cur.rowcount

    def prune_missing_under_roots(
        self, *, roots: list[Path], keep: set[str]
    ) -> int:
        """Delete every row whose path is under one of ``roots`` but not in
        ``keep``. Rows for paths outside the given roots are left alone.

        ``roots`` is the list of *reachable* library paths in this scan —
        passing the configured library list would risk wiping the index when
        a network mount is briefly unavailable.

        Paths are compared as absolute, resolved strings to avoid mismatches
        from trailing slashes or relative components.
        """
        if not roots:
            return 0
        # Resolve roots once so the in-Python prefix check matches the way
        # paths were stored (also resolved at scan time).
        resolved_roots = [Path(r).expanduser().resolve() for r in roots]
        cur = self._conn.execute("SELECT id, path FROM tracks")
        to_delete: list[int] = []
        for row in cur:
            p = row["path"]
            if p in keep:
                continue
            try:
                pp = Path(p)
            except Exception:
                continue
            if any(_is_under(pp, root) for root in resolved_roots):
                to_delete.append(int(row["id"]))
        if not to_delete:
            return 0
        with self.transaction() as cur:
            cur.executemany(
                "DELETE FROM tracks WHERE id = ?", [(i,) for i in to_delete]
            )
        return len(to_delete)

    # -- reads ---------------------------------------------------------

    def get_track_by_id(self, track_id: int) -> Optional[dict]:
        cur = self._conn.execute("SELECT * FROM tracks WHERE id = ?", (int(track_id),))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def get_track_by_path(self, path: str) -> Optional[dict]:
        cur = self._conn.execute("SELECT * FROM tracks WHERE path = ?", (path,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None

    def get_tracks_by_ids(self, ids: list[int]) -> dict[int, dict]:
        """Return ``{id: row}`` for the given IDs, in one query.

        Used by route handlers to enrich similarity-search and playlist
        results with tag and library-relative-path metadata.
        """
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        cur = self._conn.execute(
            f"""
            SELECT id, path, library_root, relative_path,
                   duration, model, embedding_dim, descriptor_version,
                   bpm, bpm_confidence, key, scale, key_strength,
                   loudness_db, danceability, onset_rate,
                   artist, album, title, track_number, musicbrainz_track_id,
                   size, mtime, analyzed_at
              FROM tracks WHERE id IN ({placeholders})
            """,
            tuple(int(i) for i in ids),
        )
        return {int(r["id"]): dict(r) for r in cur}

    def get_embedding_by_id(self, track_id: int) -> Optional[tuple[np.ndarray, str]]:
        cur = self._conn.execute(
            "SELECT embedding, embedding_dim, model FROM tracks WHERE id = ?",
            (int(track_id),),
        )
        row = cur.fetchone()
        if row is None:
            return None
        emb = np.frombuffer(row["embedding"], dtype=np.float32).reshape(
            int(row["embedding_dim"])
        )
        return emb.copy(), str(row["model"])

    def harmonic_compatible_ids(
        self, *, model: str, pairs: list[tuple[str, str]]
    ) -> set[int]:
        """Return IDs of tracks whose ``(key, scale)`` is in ``pairs``.

        Used by playlist generation when applying a harmonic-mix constraint —
        keeping the SQL here instead of in the playlist module preserves the
        encapsulation of the storage layer.
        """
        if not pairs:
            return set()
        placeholders = " OR ".join("(key = ? AND scale = ?)" for _ in pairs)
        params: list = [model]
        for key, scale in pairs:
            params.extend([key, scale])
        cur = self._conn.execute(
            f"SELECT id FROM tracks WHERE model = ? AND ({placeholders})",
            tuple(params),
        )
        return {int(r["id"]) for r in cur}

    def bpm_by_id_for_model(self, model: str) -> dict[int, Optional[float]]:
        """Return a ``{track_id: bpm}`` map for one model. Used by the
        similar-seeded playlist for the bpm-drift constraint."""
        cur = self._conn.execute(
            "SELECT id, bpm FROM tracks WHERE model = ?", (model,)
        )
        return {int(r["id"]): r["bpm"] for r in cur}

    def needs_embedding(self, path: str, size: int, mtime: float, model: str) -> bool:
        meta = self.get_track_by_path(path)
        if meta is None:
            return True
        if meta["model"] != model:
            return True
        if int(meta["size"]) != int(size):
            return True
        if abs(float(meta["mtime"]) - float(mtime)) > 1.0:
            return True
        return False

    def needs_descriptor_refresh(self, path: str, current_version: int) -> bool:
        meta = self.get_track_by_path(path)
        if meta is None:
            return False  # caller should check needs_embedding first
        return int(meta["descriptor_version"]) < current_version

    def get_paths(self) -> set[str]:
        cur = self._conn.execute("SELECT path FROM tracks")
        return {row["path"] for row in cur}

    def list_tracks(
        self,
        *,
        filter: Optional[TrackFilter] = None,
        model: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "id",
    ) -> tuple[list[dict], int]:
        """Return (rows, total_count). Rows do not include the embedding blob."""
        if order_by not in {"id", "path", "bpm", "duration", "analyzed_at"}:
            order_by = "id"

        clauses: list[str] = []
        params: list[Any] = []
        if model:
            clauses.append("model = ?")
            params.append(model)
        if filter is not None:
            f_sql, f_params = filter.to_sql()
            if f_sql:
                clauses.append(f_sql)
                params.extend(f_params)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        total = self._conn.execute(
            f"SELECT COUNT(*) FROM tracks {where}", tuple(params)
        ).fetchone()[0]

        cur = self._conn.execute(
            f"""
            SELECT id, path, library_root, relative_path,
                   size, mtime, duration, embedding_dim, model,
                   descriptor_version, bpm, bpm_confidence, key, scale,
                   key_strength, loudness_db, danceability, onset_rate,
                   artist, album, title, track_number, musicbrainz_track_id,
                   analyzed_at
              FROM tracks {where}
              ORDER BY {order_by}
              LIMIT ? OFFSET ?
            """,
            tuple(params) + (int(limit), int(offset)),
        )
        rows = [dict(r) for r in cur]
        return rows, int(total)

    def filtered_ids(
        self,
        *,
        filter: Optional[TrackFilter] = None,
        model: Optional[str] = None,
    ) -> set[int]:
        """Set of track IDs matching the given filter. Used by similarity
        search to gate candidates without loading the embedding matrix."""
        clauses: list[str] = []
        params: list[Any] = []
        if model:
            clauses.append("model = ?")
            params.append(model)
        if filter is not None:
            f_sql, f_params = filter.to_sql()
            if f_sql:
                clauses.append(f_sql)
                params.extend(f_params)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = self._conn.execute(
            f"SELECT id FROM tracks {where}", tuple(params)
        )
        return {int(r["id"]) for r in cur}

    def all_embeddings(
        self, model: Optional[str] = None
    ) -> tuple[list[int], list[str], np.ndarray]:
        """Return (ids, paths, NxD matrix) for use in similarity search."""
        if model is None:
            cur = self._conn.execute(
                "SELECT id, path, embedding, embedding_dim FROM tracks ORDER BY id"
            )
        else:
            cur = self._conn.execute(
                "SELECT id, path, embedding, embedding_dim FROM tracks "
                "WHERE model = ? ORDER BY id",
                (model,),
            )
        ids: list[int] = []
        paths: list[str] = []
        vectors: list[np.ndarray] = []
        dim: Optional[int] = None
        for row in cur:
            d = int(row["embedding_dim"])
            if dim is None:
                dim = d
            elif d != dim:
                continue
            ids.append(int(row["id"]))
            paths.append(row["path"])
            vectors.append(
                np.frombuffer(row["embedding"], dtype=np.float32).reshape(d)
            )
        if not vectors:
            return [], [], np.empty((0, 0), dtype=np.float32)
        mat = np.stack(vectors).astype(np.float32, copy=False)
        return ids, paths, mat

    def stats(self) -> dict:
        cur = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(duration), 0), COALESCE(SUM(size), 0) FROM tracks"
        )
        n, total_duration, total_bytes = cur.fetchone()
        by_model = {
            r["model"]: r["c"]
            for r in self._conn.execute(
                "SELECT model, COUNT(*) AS c FROM tracks GROUP BY model ORDER BY model"
            )
        }
        return {
            "tracks": int(n or 0),
            "total_duration_sec": float(total_duration or 0.0),
            "total_bytes": int(total_bytes or 0),
            "by_model": by_model,
            "db_path": str(self.path),
            "db_size_bytes": (
                os.path.getsize(self.path) if self.path.exists() else 0
            ),
        }
