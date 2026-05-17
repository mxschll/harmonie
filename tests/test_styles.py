"""Tests for the Discogs-400 style classification feature.

Covers:

* :func:`harmonie.features.top_styles` (label/probability mapping).
* Migration shape — ``style_activations`` column + ``track_styles`` table.
* DB roundtrip — write/read of activation BLOB and top-K style rows.
* Filter helpers — ``filter_ids_by_style``, ``list_styles``.
* API filter — ``GET /tracks?styles=...`` and the ``GET /styles`` endpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from harmonie.features import (
    GENRE_NUM_CLASSES,
    STYLE_MIN_PROB,
    STYLE_TOP_K,
    Descriptors,
    top_styles,
)

# ---------------------------------------------------------------------------
# top_styles helper
# ---------------------------------------------------------------------------


def test_top_styles_returns_top_k_above_threshold():
    labels = [f"L{i}" for i in range(GENRE_NUM_CLASSES)]
    activations = np.zeros(GENRE_NUM_CLASSES, dtype=np.float32)
    # Plant 12 distinct probabilities, four below the default threshold.
    activations[5] = 0.95
    activations[10] = 0.80
    activations[15] = 0.60
    activations[20] = 0.40
    activations[25] = 0.20
    activations[30] = 0.10
    activations[35] = 0.06
    activations[40] = 0.04  # below threshold
    activations[45] = 0.03
    activations[50] = 0.02
    activations[55] = 0.01
    activations[60] = 0.005

    top = top_styles(activations, labels)
    # Should stop at the threshold (default 0.05) — 7 entries above it.
    assert len(top) == 7
    assert top[0] == ("L5", pytest.approx(0.95))
    assert top[-1][0] == "L35"
    # All probabilities above the threshold and in descending order.
    probs = [p for _, p in top]
    assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))
    assert all(p >= STYLE_MIN_PROB for p in probs)


def test_top_styles_caps_at_top_k():
    labels = [f"L{i}" for i in range(GENRE_NUM_CLASSES)]
    # Every label well above the threshold.
    activations = np.full(GENRE_NUM_CLASSES, 0.5, dtype=np.float32)
    top = top_styles(activations, labels)
    assert len(top) == STYLE_TOP_K  # never more than top_k


def test_top_styles_empty_when_all_below_threshold():
    labels = [f"L{i}" for i in range(GENRE_NUM_CLASSES)]
    activations = np.full(GENRE_NUM_CLASSES, 0.001, dtype=np.float32)
    assert top_styles(activations, labels) == []


def test_top_styles_rejects_wrong_shape():
    labels = [f"L{i}" for i in range(GENRE_NUM_CLASSES)]
    with pytest.raises(ValueError):
        top_styles(np.zeros(10, dtype=np.float32), labels)


# ---------------------------------------------------------------------------
# Migration shape
# ---------------------------------------------------------------------------


def test_migration_creates_style_columns_and_table(tmp_db_path):
    from harmonie.db import Database

    db = Database(tmp_db_path)
    try:
        # tracks.style_activations exists and is a BLOB.
        cols = {
            r["name"]: r["type"] for r in db._conn.execute("PRAGMA table_info(tracks)")
        }
        assert "style_activations" in cols
        assert cols["style_activations"].upper() == "BLOB"

        # track_styles table exists with the expected columns.
        sty_cols = {
            r["name"]: r["type"]
            for r in db._conn.execute("PRAGMA table_info(track_styles)")
        }
        assert sty_cols == {
            "track_id": "INTEGER",
            "style": "TEXT",
            "probability": "REAL",
        }

        # The lookup index on (style, probability DESC) is in place.
        idxs = {
            r["name"]
            for r in db._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'index' AND tbl_name = 'track_styles'"
            )
        }
        assert "idx_track_styles_style" in idxs
        assert "idx_track_styles_prob" in idxs
    finally:
        db.close()


# ---------------------------------------------------------------------------
# DB roundtrip
# ---------------------------------------------------------------------------


def _upsert_with_styles(
    db, path, embedding, *, top_styles_rows, style_activations=None
):
    """Helper: upsert a track with a style preview + activation vector."""
    from harmonie.tags import Tags

    if style_activations is None:
        # Fabricate a reasonable activation vector so the BLOB column is
        # populated. Set the entries for the labels we list to their probs.
        style_activations = np.zeros(GENRE_NUM_CLASSES, dtype=np.float32)
    return db.upsert_track(
        path=path,
        size=1,
        mtime=1.0,
        duration=1.0,
        embedding=embedding,
        model="m1",
        descriptors=Descriptors(),
        descriptor_version=1,
        tags=Tags(),
        style_activations=style_activations,
        top_styles=top_styles_rows,
    )


def test_db_roundtrip_styles(tmp_db_path):
    from harmonie.db import Database

    db = Database(tmp_db_path)
    try:
        emb = np.zeros(4, dtype=np.float32)
        tid = _upsert_with_styles(
            db,
            "/a.flac",
            emb,
            top_styles_rows=[
                ("Electronic---House", 0.91),
                ("Electronic---Techno", 0.42),
                ("Rock---Punk", 0.07),
            ],
        )
        rows = db.get_track_styles(tid)
        # Stored highest-first.
        assert rows == [
            ("Electronic---House", pytest.approx(0.91)),
            ("Electronic---Techno", pytest.approx(0.42)),
            ("Rock---Punk", pytest.approx(0.07)),
        ]

        # Bulk version returns the same data keyed by id.
        bulk = db.get_styles_by_ids([tid])
        assert bulk[tid] == rows

        # Activation BLOB was stored with the right size.
        cur = db._conn.execute(
            "SELECT length(style_activations) AS n FROM tracks WHERE id = ?",
            (tid,),
        )
        n = cur.fetchone()["n"]
        # 400 floats * 4 bytes = 1600.
        assert n == GENRE_NUM_CLASSES * 4
    finally:
        db.close()


def test_db_upsert_replaces_styles(tmp_db_path):
    """Re-scanning a track should clobber its old style rows, not append."""
    from harmonie.db import Database

    db = Database(tmp_db_path)
    try:
        emb = np.zeros(4, dtype=np.float32)
        tid = _upsert_with_styles(
            db,
            "/a.flac",
            emb,
            top_styles_rows=[("Electronic---House", 0.9)],
        )
        _upsert_with_styles(
            db,
            "/a.flac",
            emb,
            top_styles_rows=[("Rock---Punk", 0.7)],
        )
        rows = db.get_track_styles(tid)
        assert rows == [("Rock---Punk", pytest.approx(0.7))]
        # And only one row in the table.
        n = db._conn.execute(
            "SELECT COUNT(*) FROM track_styles WHERE track_id = ?", (tid,)
        ).fetchone()[0]
        assert n == 1
    finally:
        db.close()


def test_db_remove_cascades_to_styles(tmp_db_path):
    from harmonie.db import Database

    db = Database(tmp_db_path)
    try:
        tid = _upsert_with_styles(
            db,
            "/a.flac",
            np.zeros(4, dtype=np.float32),
            top_styles_rows=[("Electronic---House", 0.9)],
        )
        db.remove_by_id(tid)
        n = db._conn.execute(
            "SELECT COUNT(*) FROM track_styles WHERE track_id = ?", (tid,)
        ).fetchone()[0]
        assert n == 0
    finally:
        db.close()


# ---------------------------------------------------------------------------
# filter_ids_by_style and list_styles
# ---------------------------------------------------------------------------


def test_filter_ids_by_style_exact_and_prefix(tmp_db_path):
    from harmonie.db import Database

    db = Database(tmp_db_path)
    try:
        emb = np.zeros(4, dtype=np.float32)
        a = _upsert_with_styles(
            db,
            "/a",
            emb,
            top_styles_rows=[("Electronic---House", 0.9)],
        )
        b = _upsert_with_styles(
            db,
            "/b",
            emb,
            top_styles_rows=[("Electronic---Techno", 0.8)],
        )
        c = _upsert_with_styles(
            db,
            "/c",
            emb,
            top_styles_rows=[("Rock---Punk", 0.7)],
        )

        # Exact match: just track A.
        assert db.filter_ids_by_style(["Electronic---House"]) == {a}

        # Prefix match: A and B (whole Electronic branch).
        assert db.filter_ids_by_style(["Electronic"]) == {a, b}

        # min_probability gate excludes B (0.8) but keeps A (0.9).
        assert db.filter_ids_by_style(["Electronic"], min_probability=0.85) == {a}

        # Multiple needles, "any" mode.
        assert db.filter_ids_by_style(["Electronic---House", "Rock---Punk"]) == {a, c}

        # "all" mode requires every needle to be present on a track.
        # A has House but not Punk → empty.
        assert (
            db.filter_ids_by_style(["Electronic---House", "Rock---Punk"], match="all")
            == set()
        )
    finally:
        db.close()


def test_list_styles_aggregates_by_count(tmp_db_path):
    from harmonie.db import Database

    db = Database(tmp_db_path)
    try:
        emb = np.zeros(4, dtype=np.float32)
        _upsert_with_styles(
            db,
            "/a",
            emb,
            top_styles_rows=[("Electronic---House", 0.9)],
        )
        _upsert_with_styles(
            db,
            "/b",
            emb,
            top_styles_rows=[("Electronic---House", 0.5), ("Rock---Punk", 0.4)],
        )
        _upsert_with_styles(
            db,
            "/c",
            emb,
            top_styles_rows=[("Electronic---House", 0.7)],
        )

        styles = db.list_styles()
        by_style = {s["style"]: s for s in styles}
        # House appears on three tracks, Punk on one.
        assert by_style["Electronic---House"]["track_count"] == 3
        assert by_style["Rock---Punk"]["track_count"] == 1
        # mean_probability aggregates probabilities, not just counts.
        assert by_style["Electronic---House"]["mean_probability"] == pytest.approx(
            (0.9 + 0.5 + 0.7) / 3,
            abs=1e-6,
        )
        # min_probability gate filters out the Punk row (0.4 < 0.6).
        styles_high = db.list_styles(min_probability=0.6)
        names_high = {s["style"] for s in styles_high}
        assert "Rock---Punk" not in names_high
        assert "Electronic---House" in names_high
    finally:
        db.close()


# ---------------------------------------------------------------------------
# API integration: GET /tracks?styles=... and GET /styles
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client_with_styles(tmp_path, make_db):
    """Mount the app on a populated DB and return a TestClient."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from harmonie.api.routes import api_router, public_router
    from harmonie.config import Settings

    db, index = make_db("api.db")
    emb = np.zeros(4, dtype=np.float32)
    _upsert_with_styles(
        db,
        "/lib/house.flac",
        emb,
        top_styles_rows=[("Electronic---House", 0.9)],
    )
    _upsert_with_styles(
        db,
        "/lib/techno.flac",
        emb,
        top_styles_rows=[("Electronic---Techno", 0.8)],
    )
    _upsert_with_styles(
        db,
        "/lib/punk.flac",
        emb,
        top_styles_rows=[("Rock---Punk", 0.7)],
    )

    # Build the app without the production lifespan to skip TF and the
    # worker pool.
    settings = Settings(libraries=[tmp_path], data_dir=tmp_path)
    app = FastAPI()
    app.include_router(public_router)
    app.include_router(api_router, prefix="/api/v1")

    class StubAnalyzer:
        def __init__(self):
            self.db = db
            self.index = index
            self.settings = settings
            self.embedding_dim = 4

            class _S:
                state = "idle"

                def snapshot(self):
                    return {"state": self.state}

            self.status = _S()

    app.state.analyzer = StubAnalyzer()
    app.state.settings = settings
    return TestClient(app), db


def test_api_list_tracks_filters_by_style_prefix(api_client_with_styles):
    client, _db = api_client_with_styles
    r = client.get("/api/v1/tracks", params=[("style", "Electronic")])
    assert r.status_code == 200, r.text
    body = r.json()
    titles = sorted(item["path"] for item in body["items"])
    assert titles == ["/lib/house.flac", "/lib/techno.flac"]
    # Each item carries its own style preview.
    by_path = {it["path"]: it for it in body["items"]}
    assert by_path["/lib/house.flac"]["styles"][0]["style"] == "Electronic---House"


def test_api_list_tracks_filters_by_exact_style(api_client_with_styles):
    client, _db = api_client_with_styles
    r = client.get(
        "/api/v1/tracks",
        params=[("style", "Electronic---Techno")],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [it["path"] for it in body["items"]] == ["/lib/techno.flac"]


def test_api_styles_endpoint_lists_known_styles(api_client_with_styles):
    client, _db = api_client_with_styles
    r = client.get("/api/v1/styles")
    assert r.status_code == 200, r.text
    body = r.json()
    names = sorted(s["style"] for s in body["items"])
    assert names == [
        "Electronic---House",
        "Electronic---Techno",
        "Rock---Punk",
    ]
    assert body["total"] == 3


def test_api_styles_endpoint_respects_min_probability(api_client_with_styles):
    client, _db = api_client_with_styles
    r = client.get(
        "/api/v1/styles",
        params={"style_min": 0.85},
    )
    assert r.status_code == 200, r.text
    names = [s["style"] for s in r.json()["items"]]
    assert names == ["Electronic---House"]  # only the 0.9 row clears 0.85
