"""Regression test for the cache-invalidation contract:
adding a new track and invalidating the index must let similarity search
find that track without restarting the process."""

from __future__ import annotations

import numpy as np

from harmonie.features import DESCRIPTOR_VERSION
from harmonie.similarity import find_similar_to_id


def _add(db, path, emb, fake_descriptors, model="m1"):
    return db.upsert_track(
        path=path,
        size=1,
        mtime=1.0,
        duration=1.0,
        embedding=emb,
        model=model,
        descriptors=fake_descriptors(),
        descriptor_version=DESCRIPTOR_VERSION,
    )


def test_similar_sees_new_tracks_after_invalidate(make_db, fake_descriptors):
    db, index = make_db()
    seed_emb = np.array([1.0, 0, 0, 0], dtype=np.float32)
    seed = _add(db, "/seed", seed_emb, fake_descriptors)
    other = _add(db, "/other",
                 np.array([0.9, 0.1, 0, 0], dtype=np.float32),
                 fake_descriptors)

    # First query — only /other is a candidate.
    matches = find_similar_to_id(db, index, seed, n=10)
    assert {m.path for m in matches} == {"/other"}

    # Add a new track that should be more similar to the seed.
    very_similar = _add(
        db, "/very-similar",
        np.array([0.99, 0.01, 0, 0], dtype=np.float32),
        fake_descriptors,
    )

    # Without invalidation, the cached matrix still has only the original 2.
    stale = find_similar_to_id(db, index, seed, n=10)
    assert {m.path for m in stale} == {"/other"}

    # After invalidate (which the analyzer does at end of every scan), the
    # rebuilt matrix includes the new track and ranks it correctly.
    index.invalidate()
    fresh = find_similar_to_id(db, index, seed, n=10)
    paths = [m.path for m in fresh]
    assert "/very-similar" in paths
    # /very-similar should outrank /other since cosine to seed is closer.
    assert paths.index("/very-similar") < paths.index("/other")
