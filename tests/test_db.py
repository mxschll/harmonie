"""Tests for the SQLite layer."""

from __future__ import annotations

import numpy as np

from harmonie.db import Database, TrackFilter
from harmonie.features import DESCRIPTOR_VERSION


def _insert(db, *, path, embedding, descriptors, model="m1", size=100, mtime=1.0):
    return db.upsert_track(
        path=path,
        size=size,
        mtime=mtime,
        duration=180.0,
        embedding=embedding,
        model=model,
        descriptors=descriptors,
        descriptor_version=DESCRIPTOR_VERSION,
    )


def test_upsert_and_get(tmp_db_path, random_embedding, fake_descriptors):
    db = Database(tmp_db_path)
    try:
        emb = random_embedding(dim=8)
        tid = _insert(db, path="/a.flac", embedding=emb, descriptors=fake_descriptors())
        assert isinstance(tid, int) and tid > 0

        row = db.get_track_by_id(tid)
        assert row is not None
        assert row["path"] == "/a.flac"
        assert row["bpm"] == 128.0
        assert row["key"] == "C"
        assert row["scale"] == "major"
        assert row["embedding_dim"] == 8

        emb_back = db.get_embedding_by_id(tid)
        assert emb_back is not None
        v, model = emb_back
        assert model == "m1"
        np.testing.assert_allclose(v, emb)
    finally:
        db.close()


def test_upsert_replaces_existing(tmp_db_path, random_embedding, fake_descriptors):
    db = Database(tmp_db_path)
    try:
        e1 = random_embedding(dim=4, seed=1)
        e2 = random_embedding(dim=4, seed=2)
        tid1 = _insert(db, path="/a.flac", embedding=e1, descriptors=fake_descriptors(bpm=120))
        tid2 = _insert(db, path="/a.flac", embedding=e2, descriptors=fake_descriptors(bpm=140))
        assert tid1 == tid2  # same row, same id
        row = db.get_track_by_id(tid1)
        assert row["bpm"] == 140
    finally:
        db.close()


def test_filter_by_bpm_range(tmp_db_path, random_embedding, fake_descriptors):
    db = Database(tmp_db_path)
    try:
        for i, bpm in enumerate([100, 120, 128, 140, 160]):
            _insert(
                db,
                path=f"/{i}.flac",
                embedding=random_embedding(dim=4, seed=i),
                descriptors=fake_descriptors(bpm=bpm),
            )
        rows, total = db.list_tracks(filter=TrackFilter(bpm_min=120, bpm_max=140))
        assert total == 3
        bpms = sorted(r["bpm"] for r in rows)
        assert bpms == [120, 128, 140]
    finally:
        db.close()


def test_filter_by_key_set(tmp_db_path, random_embedding, fake_descriptors):
    db = Database(tmp_db_path)
    try:
        for i, key in enumerate(["C", "G", "D", "A"]):
            _insert(
                db,
                path=f"/{key}.flac",
                embedding=random_embedding(dim=4, seed=i),
                descriptors=fake_descriptors(key=key),
            )
        rows, total = db.list_tracks(filter=TrackFilter(key=["C", "G"]))
        assert total == 2
        assert sorted(r["key"] for r in rows) == ["C", "G"]
    finally:
        db.close()


def test_prune_missing_under_roots(tmp_path, random_embedding, fake_descriptors):
    from pathlib import Path

    db_path = tmp_path / "test.db"
    db = Database(db_path)
    try:
        # Two libraries: /lib_a (reachable in this test) and /lib_b (would be
        # unreachable IRL — we simulate that by passing only /lib_a as a root).
        lib_a = tmp_path / "lib_a"
        lib_b = tmp_path / "lib_b"
        lib_a.mkdir()
        lib_b.mkdir()
        for sub, name in [(lib_a, "kept.flac"), (lib_a, "gone.flac"),
                          (lib_b, "untouched.flac")]:
            _insert(
                db,
                path=str(sub / name),
                embedding=random_embedding(dim=4),
                descriptors=fake_descriptors(),
            )
        keep = {str(lib_a / "kept.flac")}
        # Only lib_a is "reachable" — lib_b should be untouched.
        removed = db.prune_missing_under_roots(roots=[lib_a], keep=keep)
        assert removed == 1
        paths = {r["path"] for r, _ in [(r, None) for r in db.list_tracks()[0]]}
        assert str(lib_a / "kept.flac") in paths
        assert str(lib_b / "untouched.flac") in paths
        assert str(lib_a / "gone.flac") not in paths
    finally:
        db.close()


def test_prune_with_no_roots_is_noop(tmp_path, random_embedding, fake_descriptors):
    db = Database(tmp_path / "test.db")
    try:
        _insert(
            db,
            path=str(tmp_path / "x.flac"),
            embedding=random_embedding(dim=4),
            descriptors=fake_descriptors(),
        )
        removed = db.prune_missing_under_roots(roots=[], keep=set())
        assert removed == 0
        _, total = db.list_tracks()
        assert total == 1
    finally:
        db.close()


def test_descriptor_version_split(tmp_db_path, random_embedding, fake_descriptors):
    db = Database(tmp_db_path)
    try:
        # Insert at descriptor_version-1 to force needs_descriptor_refresh.
        emb = random_embedding(dim=4)
        db.upsert_track(
            path="/a.flac",
            size=100,
            mtime=1.0,
            duration=120.0,
            embedding=emb,
            model="m1",
            descriptors=fake_descriptors(),
            descriptor_version=DESCRIPTOR_VERSION - 1
            if DESCRIPTOR_VERSION > 0
            else 0,
        )
        # If DESCRIPTOR_VERSION is 0 the test is trivially false, skip.
        if DESCRIPTOR_VERSION > 0:
            assert db.needs_descriptor_refresh("/a.flac", DESCRIPTOR_VERSION)
            assert not db.needs_embedding("/a.flac", 100, 1.0, "m1")
            ok = db.update_descriptors(
                "/a.flac",
                descriptors=fake_descriptors(bpm=99),
                descriptor_version=DESCRIPTOR_VERSION,
            )
            assert ok
            assert not db.needs_descriptor_refresh("/a.flac", DESCRIPTOR_VERSION)
            row = db.get_track_by_path("/a.flac")
            assert row["bpm"] == 99
    finally:
        db.close()


def test_tags_and_library_columns_round_trip(
    tmp_db_path, random_embedding, fake_descriptors
):
    """Tags + library_root + relative_path get stored and read back."""
    from harmonie.tags import Tags

    db = Database(tmp_db_path)
    try:
        tid = db.upsert_track(
            path="/lib/artist/album/01.flac",
            size=100,
            mtime=1.0,
            duration=180.0,
            embedding=random_embedding(dim=4),
            model="m1",
            descriptors=fake_descriptors(),
            descriptor_version=DESCRIPTOR_VERSION,
            tags=Tags(
                artist="Person",
                album="Record",
                title="Song",
                track_number=1,
            ),
            library_root="/lib",
            relative_path="artist/album/01.flac",
        )
        row = db.get_track_by_id(tid)
        assert row["artist"] == "Person"
        assert row["album"] == "Record"
        assert row["title"] == "Song"
        assert row["track_number"] == 1
        assert row["library_root"] == "/lib"
        assert row["relative_path"] == "artist/album/01.flac"
    finally:
        db.close()


def test_get_tracks_by_ids_bulk_lookup(
    tmp_db_path, random_embedding, fake_descriptors
):
    from harmonie.tags import Tags

    db = Database(tmp_db_path)
    try:
        ids = []
        for i in range(3):
            tid = db.upsert_track(
                path=f"/{i}.flac",
                size=100,
                mtime=1.0,
                duration=120.0,
                embedding=random_embedding(dim=4, seed=i),
                model="m1",
                descriptors=fake_descriptors(),
                descriptor_version=DESCRIPTOR_VERSION,
                tags=Tags(title=f"Track {i}"),
            )
            ids.append(tid)
        rows = db.get_tracks_by_ids(ids)
        assert set(rows.keys()) == set(ids)
        assert rows[ids[0]]["title"] == "Track 0"
        assert rows[ids[2]]["title"] == "Track 2"

        # Empty input returns empty dict, not an error.
        assert db.get_tracks_by_ids([]) == {}
    finally:
        db.close()


def test_descriptor_refresh_updates_tags(
    tmp_db_path, random_embedding, fake_descriptors
):
    """A descriptor-only refresh should also pick up new tags from the file."""
    from harmonie.tags import Tags

    db = Database(tmp_db_path)
    try:
        db.upsert_track(
            path="/a.flac",
            size=100,
            mtime=1.0,
            duration=180.0,
            embedding=random_embedding(dim=4),
            model="m1",
            descriptors=fake_descriptors(),
            descriptor_version=DESCRIPTOR_VERSION,
            tags=Tags(title="Old Title"),
        )
        ok = db.update_descriptors(
            "/a.flac",
            descriptors=fake_descriptors(bpm=140),
            descriptor_version=DESCRIPTOR_VERSION,
            tags=Tags(title="New Title", artist="New Artist"),
        )
        assert ok
        row = db.get_track_by_path("/a.flac")
        assert row["title"] == "New Title"
        assert row["artist"] == "New Artist"
        assert row["bpm"] == 140
    finally:
        db.close()
