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
        db, index, ChainedPlaylistRequest(seed_ids=[seed], chunk_size=5, n=25)
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
        db, index, ChainedPlaylistRequest(seed_ids=[seed], chunk_size=8, n=16)
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
        db, index, ChainedPlaylistRequest(seed_ids=[seed], chunk_size=1, n=10)
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
        db, index, ChainedPlaylistRequest(seed_ids=[seed], chunk_size=5, n=100)
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
            seed_ids=[seed], chunk_size=3, n=10,
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
        ChainedPlaylistRequest(seed_ids=[seed], chunk_size=2, n=4, include_seed=True),
    )
    assert len(items) == 4
    assert items[0].track_id == seed
    ids = [m.track_id for m in items]
    assert len(set(ids)) == len(ids)


def test_chained_respects_bpm_tolerance(make_db, fake_descriptors):
    """drift mode: consecutive picks (incl. seed→first) must stay within
    bpm_tolerance of the previous track."""
    db, index = make_db()
    rng = np.random.default_rng(11)
    seed_emb = rng.standard_normal(8).astype(np.float32)
    seed = _add(db, "/seed", seed_emb, fake_descriptors(bpm=128))
    # 30 tracks with BPMs spread out wildly so the constraint actually bites.
    for i in range(30):
        v = seed_emb + 0.1 * rng.standard_normal(8).astype(np.float32)
        bpm = 60 + i * 5  # 60..205
        _add(db, f"/n{i}", v, fake_descriptors(bpm=bpm))

    items = generate_chained_playlist(
        db, index,
        ChainedPlaylistRequest(
            seed_ids=[seed], chunk_size=3, n=10, bpm_drift=6,
        ),
    )
    assert len(items) > 0
    bpms = [db.get_track_by_id(m.track_id)["bpm"] for m in items]
    prev = 128  # seed BPM
    for b in bpms:
        assert abs(b - prev) <= 6, f"gap {b}-{prev} exceeds tolerance"
        prev = b


def test_chained_respects_key_compatible(make_db, fake_descriptors):
    """drift mode: every consecutive transition is harmonically compatible."""
    from harmonie.playlist import compatible_keys_for

    db, index = make_db()
    rng = np.random.default_rng(22)
    seed_emb = rng.standard_normal(4).astype(np.float32)
    seed = _add(
        db, "/seed", seed_emb, fake_descriptors(key="A", scale="minor"),
    )
    # Mix compatible (A minor, C major, D minor, E minor) and incompatible
    # (F# major, B minor in 10A) keys.
    keys_to_use = [
        ("A", "minor"), ("C", "major"), ("D", "minor"), ("E", "minor"),
        ("F#", "major"), ("F", "minor"), ("G", "major"), ("Bb", "major"),
    ]
    for i in range(40):
        v = seed_emb + 0.1 * rng.standard_normal(4).astype(np.float32)
        k, s = keys_to_use[i % len(keys_to_use)]
        _add(db, f"/n{i}", v, fake_descriptors(key=k, scale=s))

    items = generate_chained_playlist(
        db, index,
        ChainedPlaylistRequest(
            seed_ids=[seed], chunk_size=4, n=12, harmonic_mix=True,
        ),
    )
    assert len(items) > 0
    prev_key, prev_scale = "A", "minor"
    for m in items:
        row = db.get_track_by_id(m.track_id)
        ok = compatible_keys_for(prev_key, prev_scale)
        assert (row["key"], row["scale"]) in ok, (
            f"track {row['key']} {row['scale']} not compatible with "
            f"previous {prev_key} {prev_scale}"
        )
        prev_key, prev_scale = row["key"], row["scale"]


def test_chained_combines_bpm_and_key_constraints(make_db, fake_descriptors):
    """Both constraints applied at once — every consecutive pair must
    satisfy both."""
    from harmonie.playlist import compatible_keys_for

    db, index = make_db()
    rng = np.random.default_rng(33)
    seed_emb = rng.standard_normal(4).astype(np.float32)
    seed = _add(
        db, "/seed", seed_emb,
        fake_descriptors(key="A", scale="minor", bpm=128),
    )
    keys = [("A", "minor"), ("C", "major"), ("D", "minor"), ("E", "minor")]
    for i in range(20):
        v = seed_emb + 0.1 * rng.standard_normal(4).astype(np.float32)
        k, s = keys[i % len(keys)]
        _add(db, f"/n{i}", v, fake_descriptors(key=k, scale=s, bpm=124 + (i % 5)))

    items = generate_chained_playlist(
        db, index,
        ChainedPlaylistRequest(
            seed_ids=[seed], chunk_size=3, n=8,
            bpm_drift=4, harmonic_mix=True,
        ),
    )
    prev = (128, "A", "minor")
    for m in items:
        row = db.get_track_by_id(m.track_id)
        bpm, key, scale = row["bpm"], row["key"], row["scale"]
        assert abs(bpm - prev[0]) <= 4
        assert (key, scale) in compatible_keys_for(prev[1], prev[2])
        prev = (bpm, key, scale)



def test_chained_multiseed_uses_centroid_anchor(make_db, fake_descriptors):
    """Drift mode with multiple seeds anchors on the centroid of their
    embeddings, not on any single seed. Verify that picks come from the
    centroid neighborhood — i.e. close to the average of the seed vectors."""
    db, index = make_db()
    rng = np.random.default_rng(101)

    # Two seeds that span two distinct clusters in embedding space.
    seed_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    seed_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    sid_a = _add(db, "/seed_a", seed_a, fake_descriptors())
    sid_b = _add(db, "/seed_b", seed_b, fake_descriptors())

    # The centroid lies at (0.5, 0.5, 0, 0); normalized that's
    # (~0.707, ~0.707, 0, 0). Plant some candidates near the centroid and
    # some far away; near-centroid ones should win.
    centroid = np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float32)
    near_ids = []
    for i in range(5):
        v = centroid + 0.05 * rng.standard_normal(4).astype(np.float32)
        near_ids.append(_add(db, f"/near{i}", v, fake_descriptors()))
    # Tracks far from centroid (orthogonal direction).
    far = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    for i in range(5):
        v = far + 0.05 * rng.standard_normal(4).astype(np.float32)
        _add(db, f"/far{i}", v, fake_descriptors())

    items = generate_chained_playlist(
        db, index,
        ChainedPlaylistRequest(
            seed_ids=[sid_a, sid_b], chunk_size=3, n=3,
        ),
    )
    assert len(items) == 3
    chosen_paths = {m.path for m in items}
    # Every pick should be from the near-centroid cluster, not the far one.
    assert all(p.startswith("/near") for p in chosen_paths)


def test_chained_multiseed_include_seed_emits_each(make_db, fake_descriptors):
    """When include_seed is True with multiple seeds, every seed appears
    in the output (in input order), and the rest of the playlist follows."""
    db, index = make_db()
    rng = np.random.default_rng(202)
    seed_emb = rng.standard_normal(4).astype(np.float32)
    sid_x = _add(db, "/seed_x", seed_emb, fake_descriptors())
    sid_y = _add(db, "/seed_y", seed_emb + 0.1 * rng.standard_normal(4),
                 fake_descriptors())
    for i in range(8):
        v = seed_emb + 0.2 * rng.standard_normal(4).astype(np.float32)
        _add(db, f"/n{i}", v, fake_descriptors())

    items = generate_chained_playlist(
        db, index,
        ChainedPlaylistRequest(
            seed_ids=[sid_x, sid_y], chunk_size=3, n=6, include_seed=True,
        ),
    )
    assert len(items) == 6
    # Seeds are first two, in input order.
    assert items[0].track_id == sid_x
    assert items[1].track_id == sid_y
    # No duplicates.
    assert len({m.track_id for m in items}) == 6
