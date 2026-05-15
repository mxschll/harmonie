"""Tests for playlist generation and the Camelot wheel helpers."""

from __future__ import annotations

import numpy as np

from harmonie.db import TrackFilter
from harmonie.features import DESCRIPTOR_VERSION
from harmonie.playlist import (
    ChainedPlaylistRequest,
    SimilarPlaylistRequest,
    VibePlaylistRequest,
    camelot_of,
    compatible_camelot,
    compatible_keys_for,
    generate_chained_playlist,
    generate_similar_playlist,
    generate_vibe_playlist,
)


# ---------------------------------------------------------------------------
# Camelot
# ---------------------------------------------------------------------------


def test_camelot_basic():
    assert camelot_of("A", "minor") == "8A"
    assert camelot_of("C", "major") == "8B"
    assert camelot_of("F#", "major") == "2B"
    assert camelot_of("Gb", "major") == "2B"  # alternate spelling
    assert camelot_of(None, "major") is None
    assert camelot_of("X", "minor") is None


def test_compatible_camelot_wraparound():
    assert compatible_camelot("1A") == {"1A", "12A", "2A", "1B"}
    assert compatible_camelot("12B") == {"12B", "11B", "1B", "12A"}


def test_compatible_keys_for_round_trip():
    keys = compatible_keys_for("A", "minor")
    assert ("A", "minor") in keys
    assert ("C", "major") in keys
    assert ("D", "minor") in keys
    assert ("E", "minor") in keys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add(db, path, emb, descriptors, model="m1"):
    return db.upsert_track(
        path=path,
        size=1,
        mtime=1.0,
        duration=1.0,
        embedding=emb,
        model=model,
        descriptors=descriptors,
        descriptor_version=DESCRIPTOR_VERSION,
    )


# ---------------------------------------------------------------------------
# Similar-seeded playlist
# ---------------------------------------------------------------------------


def test_similar_playlist_returns_n(make_db, fake_descriptors):
    db, index = make_db()
    rng = np.random.default_rng(0)
    seed_emb = rng.standard_normal(8).astype(np.float32)
    seed = _add(db, "/seed", seed_emb, fake_descriptors())
    for i in range(20):
        v = seed_emb + 0.1 * rng.standard_normal(8).astype(np.float32)
        _add(db, f"/n{i}", v, fake_descriptors(bpm=128 + (i % 3)))

    items = generate_similar_playlist(
        db, index, SimilarPlaylistRequest(seed_ids=[seed], n=5)
    )
    assert len(items) == 5
    assert all(m.track_id != seed for m in items)


def test_similar_playlist_bpm_drift(make_db, fake_descriptors):
    db, index = make_db()
    rng = np.random.default_rng(0)
    seed_emb = rng.standard_normal(8).astype(np.float32)
    seed = _add(db, "/seed", seed_emb, fake_descriptors(bpm=128))
    for i in range(30):
        v = seed_emb + 0.1 * rng.standard_normal(8).astype(np.float32)
        bpm = 128 + (i - 15) * 4
        _add(db, f"/n{i}", v, fake_descriptors(bpm=bpm))

    items = generate_similar_playlist(
        db, index, SimilarPlaylistRequest(seed_ids=[seed], n=10, bpm_drift=8),
    )
    bpms = [db.get_track_by_id(m.track_id)["bpm"] for m in items]
    prev = 128
    for b in bpms:
        assert abs(b - prev) <= 8
        prev = b


def test_similar_playlist_harmonic_mix(make_db, fake_descriptors):
    """Harmonic-mix should restrict candidates to Camelot-compatible keys."""
    db, index = make_db()
    rng = np.random.default_rng(7)
    seed_emb = rng.standard_normal(4).astype(np.float32)
    # Seed: A minor (Camelot 8A). Compatible: A minor, C major, D minor, E minor.
    seed = _add(db, "/seed", seed_emb, fake_descriptors(key="A", scale="minor"))
    # One compatible track, one incompatible.
    compat = _add(
        db, "/compat",
        seed_emb + 0.05 * rng.standard_normal(4).astype(np.float32),
        fake_descriptors(key="C", scale="major"),
    )
    _add(
        db, "/incompat",
        seed_emb + 0.05 * rng.standard_normal(4).astype(np.float32),
        fake_descriptors(key="F#", scale="major"),
    )
    items = generate_similar_playlist(
        db, index,
        SimilarPlaylistRequest(seed_ids=[seed], n=10, harmonic_mix=True),
    )
    ids = {m.track_id for m in items}
    assert compat in ids
    # /incompat must not appear because F# major isn't in 8A's compatible set.
    incompat_paths = {m.path for m in items if m.track_id != compat}
    assert "/incompat" not in incompat_paths


# ---------------------------------------------------------------------------
# Vibe playlist
# ---------------------------------------------------------------------------


def test_vibe_playlist_filters_and_targets(make_db, fake_descriptors):
    db, _index = make_db()
    rng = np.random.default_rng(0)
    for i in range(10):
        v = rng.standard_normal(4).astype(np.float32)
        _add(db, f"/{i}", v, fake_descriptors(bpm=120 + i, danceability=1.0 + i * 0.1))

    items = generate_vibe_playlist(
        db,
        VibePlaylistRequest(
            n=3,
            target_bpm=125,
            shuffle=False,
            descriptor_filter=TrackFilter(bpm_min=120, bpm_max=130),
        ),
    )
    assert len(items) == 3
    for m in items:
        row = db.get_track_by_id(m.track_id)
        assert 120 <= row["bpm"] <= 130


# ---------------------------------------------------------------------------
# Chained playlist
# ---------------------------------------------------------------------------


def test_chained_playlist_no_duplicates(make_db, fake_descriptors):
    db, index = make_db()
    rng = np.random.default_rng(0)
    seed_emb = rng.standard_normal(8).astype(np.float32)
    seed = _add(db, "/seed", seed_emb, fake_descriptors())
    for i in range(40):
        v = seed_emb + 0.3 * rng.standard_normal(8).astype(np.float32)
        _add(db, f"/n{i}", v, fake_descriptors())

    items = generate_chained_playlist(
        db, index, ChainedPlaylistRequest(seed_id=seed, chunk_size=5, n=25)
    )
    assert len(items) == 25
    ids = [m.track_id for m in items]
    assert len(set(ids)) == len(ids)
    assert seed not in ids


def test_chained_anchor_actually_changes(make_db, fake_descriptors):
    db, index = make_db()
    rng = np.random.default_rng(1)
    a_center = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b_center = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    seed = _add(db, "/seed", a_center, fake_descriptors())
    for i in range(10):
        v = a_center + 0.05 * rng.standard_normal(4).astype(np.float32)
        _add(db, f"/a{i}", v, fake_descriptors())
    for i in range(10):
        v = b_center + 0.05 * rng.standard_normal(4).astype(np.float32)
        _add(db, f"/b{i}", v, fake_descriptors())

    items = generate_chained_playlist(
        db, index, ChainedPlaylistRequest(seed_id=seed, chunk_size=8, n=16)
    )
    first_chunk = [m.path for m in items[:8]]
    second_chunk = [m.path for m in items[8:16]]
    assert all(p.startswith("/a") for p in first_chunk)
    assert any(p.startswith("/b") for p in second_chunk)


def test_chained_chunk_size_one_is_greedy_walk(make_db, fake_descriptors):
    db, index = make_db()
    rng = np.random.default_rng(2)
    seed_emb = rng.standard_normal(8).astype(np.float32)
    seed = _add(db, "/seed", seed_emb, fake_descriptors())
    for i in range(15):
        v = seed_emb + 0.2 * rng.standard_normal(8).astype(np.float32)
        _add(db, f"/n{i}", v, fake_descriptors())

    items = generate_chained_playlist(
        db, index, ChainedPlaylistRequest(seed_id=seed, chunk_size=1, n=10)
    )
    assert len(items) == 10
    assert len({m.track_id for m in items}) == 10


def test_chained_stops_when_candidates_exhausted(make_db, fake_descriptors):
    db, index = make_db()
    seed = _add(db, "/seed",
                np.array([1.0, 0, 0, 0], dtype=np.float32),
                fake_descriptors())
    for i in range(3):
        _add(db, f"/n{i}",
             np.array([0.9, 0.1 * (i + 1), 0, 0], dtype=np.float32),
             fake_descriptors())
    items = generate_chained_playlist(
        db, index, ChainedPlaylistRequest(seed_id=seed, chunk_size=5, n=100)
    )
    assert len(items) == 3


def test_chained_filter_applied(make_db, fake_descriptors):
    db, index = make_db()
    rng = np.random.default_rng(3)
    seed_emb = rng.standard_normal(4).astype(np.float32)
    seed = _add(db, "/seed", seed_emb, fake_descriptors(bpm=128))
    for i in range(20):
        v = seed_emb + 0.1 * rng.standard_normal(4).astype(np.float32)
        bpm = 100 + i * 4
        _add(db, f"/n{i}", v, fake_descriptors(bpm=bpm))

    items = generate_chained_playlist(
        db, index,
        ChainedPlaylistRequest(
            seed_id=seed, chunk_size=3, n=10,
            descriptor_filter=TrackFilter(bpm_min=120, bpm_max=140),
        ),
    )
    for m in items:
        row = db.get_track_by_id(m.track_id)
        assert 120 <= row["bpm"] <= 140


def test_chained_include_seed(make_db, fake_descriptors):
    db, index = make_db()
    rng = np.random.default_rng(4)
    seed_emb = rng.standard_normal(4).astype(np.float32)
    seed = _add(db, "/seed", seed_emb, fake_descriptors())
    for i in range(5):
        v = seed_emb + 0.1 * rng.standard_normal(4).astype(np.float32)
        _add(db, f"/n{i}", v, fake_descriptors())
    items = generate_chained_playlist(
        db, index,
        ChainedPlaylistRequest(seed_id=seed, chunk_size=2, n=4, include_seed=True),
    )
    assert len(items) == 4
    assert items[0].track_id == seed
    ids = [m.track_id for m in items]
    assert len(set(ids)) == len(ids)
