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
from .migrations import run_migrations
from .tags import Tags


# Schema lives in :mod:`harmonie.migrations` — see migrations.py for the
# canonical definition and the rules for adding a new migration.


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
        "styles", "style_min_probability", "style_match",
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
        styles: Optional[list[str]] = None,
        style_min_probability: float = 0.0,
        style_match: str = "any",
    ) -> None:
        self.bpm_min = bpm_min
        self.bpm_max = bpm_max
        self.key = list(key) if key else None
        self.scale = scale
        self.danceability_min = danceability_min
        self.danceability_max = danceability_max
        self.loudness_min = loudness_min
        self.loudness_max = loudness_max
        self.styles = list(styles) if styles else None
        self.style_min_probability = float(style_min_probability)
        self.style_match = style_match

    def to_sql(self) -> tuple[str, list[Any]]:
        """SQL fragment for the *tracks* table only. Style filtering is
        applied separately via :meth:`Database.filter_ids_by_style` because
        it lives in a child table."""
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

    def has_style_filter(self) -> bool:
        return bool(self.styles)

    def is_empty(self) -> bool:
        if self.styles:
            return False
        return all(
            getattr(self, slot) is None
            for slot in self.__slots__
            if slot not in ("styles", "style_min_probability", "style_match")
        )


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
        run_migrations(self._conn)

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
        style_activations: Optional[np.ndarray] = None,
        top_styles: Optional[list[tuple[str, float]]] = None,
    ) -> int:
        emb = np.ascontiguousarray(embedding.astype(np.float32, copy=False))
        t = tags or Tags()
        style_blob: Optional[bytes] = None
        if style_activations is not None:
            sa = np.ascontiguousarray(
                style_activations.astype(np.float32, copy=False)
            )
            style_blob = sa.tobytes()
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO tracks (
                    path, library_root, relative_path,
                    size, mtime, duration, embedding, embedding_dim, model,
                    descriptor_version,
                    bpm, bpm_confidence, key, scale, key_strength,
                    loudness_db, danceability, onset_rate,
                    artist, album, title, track_number,
                    style_activations,
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
                    style_activations    = excluded.style_activations,
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
                    style_blob,
                    time.time(),
                ),
            )
            cur.execute("SELECT id FROM tracks WHERE path = ?", (path,))
            track_id = int(cur.fetchone()[0])

            # Replace the per-track style rows wholesale. ON DELETE CASCADE
            # covers the case where the track row was just inserted; for
            # an upsert we still need to clear the old rows explicitly.
            cur.execute(
                "DELETE FROM track_styles WHERE track_id = ?", (track_id,)
            )
            if top_styles:
                cur.executemany(
                    "INSERT INTO track_styles (track_id, style, probability) "
                    "VALUES (?, ?, ?)",
                    [
                        (track_id, str(label), float(prob))
                        for label, prob in top_styles
                    ],
                )
            return track_id

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

    def find_track(
        self,
        *,
        path: Optional[str] = None,
        artist: Optional[str] = None,
        album: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Optional[dict]:
        """Best-effort lookup of a single track by path and/or tags.

        Strategies are tried in order; the first hit wins. Returns the row
        with the smallest ``id`` if multiple match (deterministic).

        1. Exact match on ``path``.
        2. Exact match on ``relative_path`` — useful when the caller's mount
           point differs from harmonie's view of the same library.
        3. Case-insensitive match on the (artist, album, title) triple.
        4. Case-insensitive match on (title, artist) or (title, album) when
           one tag is missing.

        Returns ``None`` if no strategy matches.
        """
        if path:
            row = self.get_track_by_path(path)
            if row is not None:
                return row
            cur = self._conn.execute(
                "SELECT * FROM tracks WHERE relative_path = ? "
                "ORDER BY id LIMIT 1",
                (path,),
            )
            row = cur.fetchone()
            if row is not None:
                return _row_to_dict(row)

        if artist and album and title:
            cur = self._conn.execute(
                "SELECT * FROM tracks "
                "WHERE artist = ? COLLATE NOCASE "
                "  AND album = ? COLLATE NOCASE "
                "  AND title = ? COLLATE NOCASE "
                "ORDER BY id LIMIT 1",
                (artist, album, title),
            )
            row = cur.fetchone()
            if row is not None:
                return _row_to_dict(row)

        if title and (artist or album):
            clauses = ["title = ? COLLATE NOCASE"]
            params: list = [title]
            if artist:
                clauses.append("artist = ? COLLATE NOCASE")
                params.append(artist)
            if album:
                clauses.append("album = ? COLLATE NOCASE")
                params.append(album)
            cur = self._conn.execute(
                f"SELECT * FROM tracks WHERE {' AND '.join(clauses)} "
                "ORDER BY id LIMIT 1",
                tuple(params),
            )
            row = cur.fetchone()
            if row is not None:
                return _row_to_dict(row)

        return None

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
                   artist, album, title, track_number,
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

    def bpm_key_by_id_for_model(
        self, model: str
    ) -> dict[int, tuple[Optional[float], Optional[str], Optional[str]]]:
        """Return ``{track_id: (bpm, key, scale)}`` for one model. Used by the
        chained playlist generator to check each candidate's BPM and key
        against the previous pick without re-querying the DB per candidate."""
        cur = self._conn.execute(
            "SELECT id, bpm, key, scale FROM tracks WHERE model = ?",
            (model,),
        )
        return {int(r["id"]): (r["bpm"], r["key"], r["scale"]) for r in cur}

    # -- styles --------------------------------------------------------

    def get_track_styles(self, track_id: int) -> list[tuple[str, float]]:
        """Top-K ``(style, probability)`` rows for one track, highest first."""
        cur = self._conn.execute(
            "SELECT style, probability FROM track_styles "
            "WHERE track_id = ? ORDER BY probability DESC",
            (int(track_id),),
        )
        return [(str(r["style"]), float(r["probability"])) for r in cur]

    def get_styles_by_ids(
        self, ids: list[int]
    ) -> dict[int, list[tuple[str, float]]]:
        """Bulk version of :meth:`get_track_styles`. One query for all IDs."""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        cur = self._conn.execute(
            f"SELECT track_id, style, probability FROM track_styles "
            f"WHERE track_id IN ({placeholders}) "
            f"ORDER BY track_id, probability DESC",
            tuple(int(i) for i in ids),
        )
        out: dict[int, list[tuple[str, float]]] = {i: [] for i in ids}
        for r in cur:
            out[int(r["track_id"])].append(
                (str(r["style"]), float(r["probability"]))
            )
        return out

    def filter_ids_by_style(
        self,
        styles: list[str],
        *,
        min_probability: float = 0.0,
        match: str = "any",
    ) -> set[int]:
        """Track IDs whose ``track_styles`` rows match the given style names.

        ``match='any'`` (default): track has at least one of the styles above
        ``min_probability``.
        ``match='all'``: track has every style above ``min_probability``.

        Style names match either exactly or as a prefix when the caller
        passes a bare genre like ``"Electronic"`` (matches all
        ``"Electronic---*"``). Passing the full ``"Genre---Style"`` form is
        always exact.
        """
        if not styles:
            return set()
        # Build an OR over `style = ?` (exact) or `style LIKE ?` (prefix).
        clauses: list[str] = []
        params: list[Any] = []
        for s in styles:
            if "---" in s:
                clauses.append("style = ?")
                params.append(s)
            else:
                clauses.append("style LIKE ?")
                params.append(f"{s}---%")
        where_styles = " OR ".join(clauses)
        params.append(float(min_probability))
        cur = self._conn.execute(
            f"SELECT track_id, COUNT(DISTINCT style) AS hits "
            f"FROM track_styles "
            f"WHERE ({where_styles}) AND probability >= ? "
            f"GROUP BY track_id",
            tuple(params),
        )
        if match == "all":
            need = len(styles)
            return {int(r["track_id"]) for r in cur if int(r["hits"]) >= need}
        return {int(r["track_id"]) for r in cur}

    def list_styles(
        self, *, min_probability: float = 0.0
    ) -> list[dict]:
        """Enumerate every style currently present in the DB.

        Returns a list of ``{style, track_count, mean_probability,
        max_probability}`` dicts ordered by ``track_count`` descending.
        Useful for building a UI of available filters and for sanity-checking
        what the model is confident about across the library.
        """
        cur = self._conn.execute(
            """
            SELECT style,
                   COUNT(*) AS track_count,
                   AVG(probability) AS mean_probability,
                   MAX(probability) AS max_probability
              FROM track_styles
             WHERE probability >= ?
             GROUP BY style
             ORDER BY track_count DESC, style ASC
            """,
            (float(min_probability),),
        )
        return [
            {
                "style": str(r["style"]),
                "track_count": int(r["track_count"]),
                "mean_probability": float(r["mean_probability"]),
                "max_probability": float(r["max_probability"]),
            }
            for r in cur
        ]

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
            if filter.has_style_filter():
                style_ids = self.filter_ids_by_style(
                    filter.styles or [],
                    min_probability=filter.style_min_probability,
                    match=filter.style_match,
                )
                if not style_ids:
                    # Style filter rules everything out — short-circuit.
                    return [], 0
                placeholders = ",".join("?" * len(style_ids))
                clauses.append(f"id IN ({placeholders})")
                params.extend(int(i) for i in style_ids)
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
                   artist, album, title, track_number,
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
        ids = {int(r["id"]) for r in cur}
        if filter is not None and filter.has_style_filter():
            style_ids = self.filter_ids_by_style(
                filter.styles or [],
                min_probability=filter.style_min_probability,
                match=filter.style_match,
            )
            ids &= style_ids
        return ids

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
