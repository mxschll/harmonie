"""Tests for the in-memory embedding cache."""

from __future__ import annotations

import numpy as np
import pytest

from harmonie.features import DESCRIPTOR_VERSION
from harmonie.index import l2_normalize_vec


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


def test_l2_normalize_vec_unit_length():
    v = np.array([3.0, 4.0], dtype=np.float32)
    n = l2_normalize_vec(v)
    assert np.isclose(np.linalg.norm(n), 1.0)


def test_index_lazy_build(make_db, fake_descriptors):
    db, index = make_db()
    # Empty model -> empty cache returns empty CachedMatrix.
    cached = index.get("m1")
    assert cached.empty

    _add(db, "/a", np.ones(4, dtype=np.float32), fake_descriptors)
    # Same cached instance, still empty (we haven't invalidated yet).
    assert index.get("m1").empty

    index.invalidate()
    after = index.get("m1")
    assert not after.empty
    assert after.dim == 4
    assert after.ids == (1,)


def test_index_invalidate_specific_model(make_db, fake_descriptors):
    db, index = make_db()
    _add(db, "/a", np.ones(4, dtype=np.float32), fake_descriptors, model="m1")
    _add(db, "/b", np.ones(4, dtype=np.float32), fake_descriptors, model="m2")
    c1 = index.get("m1")
    c2 = index.get("m2")
    assert not c1.empty and not c2.empty

    index.invalidate("m1")
    # m2 still cached (same object identity).
    assert index.get("m2") is c2
    # m1 was rebuilt — different CachedMatrix instance.
    assert index.get("m1") is not c1


def test_index_normalised_rows(make_db, fake_descriptors):
    db, index = make_db()
    _add(db, "/a", np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float32), fake_descriptors)
    _add(db, "/b", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), fake_descriptors)
    cached = index.get("m1")
    # Each row has L2-norm 1.
    norms = np.linalg.norm(cached.matrix, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)


def test_index_search_orders_by_score(make_db, fake_descriptors):
    db, index = make_db()
    _add(db, "/v1", np.array([1.0, 0, 0, 0], dtype=np.float32), fake_descriptors)
    _add(db, "/v2", np.array([0.95, 0.05, 0, 0], dtype=np.float32), fake_descriptors)
    _add(db, "/v3", np.array([0.0, 0, 1.0, 0], dtype=np.float32), fake_descriptors)
    q = np.array([0.99, 0.01, 0, 0], dtype=np.float32)
    matches = index.search(q, model="m1", n=3)
    assert [m.path for m in matches[:2]] == ["/v1", "/v2"] or [
        m.path for m in matches[:2]
    ] == ["/v2", "/v1"]
    # /v3 is orthogonal — last.
    assert matches[-1].path == "/v3"


def test_index_search_exclude_and_allowed(make_db, fake_descriptors):
    db, index = make_db()
    a = _add(db, "/a", np.ones(4, dtype=np.float32), fake_descriptors)
    b = _add(db, "/b", np.ones(4, dtype=np.float32), fake_descriptors)
    c = _add(db, "/c", -np.ones(4, dtype=np.float32), fake_descriptors)
    q = np.ones(4, dtype=np.float32)

    # Exclude a -> b is the best match.
    out = index.search(q, model="m1", n=10, exclude_ids={a})
    assert out[0].track_id == b
    assert all(m.track_id != a for m in out)

    # Restrict to {c} only -> only c returned.
    out2 = index.search(q, model="m1", n=10, allowed_ids={c})
    assert [m.track_id for m in out2] == [c]


def test_index_search_dim_mismatch(make_db, fake_descriptors):
    db, index = make_db()
    _add(db, "/a", np.ones(4, dtype=np.float32), fake_descriptors)
    with pytest.raises(ValueError):
        index.search(np.ones(3, dtype=np.float32), model="m1", n=1)


def test_index_separate_models_separate_matrices(make_db, fake_descriptors):
    db, index = make_db()
    _add(db, "/a", np.ones(4, dtype=np.float32), fake_descriptors, model="modelA")
    _add(db, "/b", np.ones(8, dtype=np.float32), fake_descriptors, model="modelB")
    a = index.get("modelA")
    b = index.get("modelB")
    assert a.dim == 4
    assert b.dim == 8
    assert a is not b


def test_invalidate_releases_old_matrix_for_callers(make_db, fake_descriptors):
    """A query that obtained a cached matrix before invalidate keeps using
    its own reference; subsequent get() returns a fresh build."""
    db, index = make_db()
    _add(db, "/a", np.ones(4, dtype=np.float32), fake_descriptors)
    first = index.get("m1")
    _add(db, "/b", np.ones(4, dtype=np.float32), fake_descriptors)
    index.invalidate()
    second = index.get("m1")
    # Caller still holds a valid reference to the old matrix.
    assert first.matrix.shape[0] == 1
    assert second.matrix.shape[0] == 2
