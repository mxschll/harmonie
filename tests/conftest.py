"""Test fixtures."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make sure the package is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tests don't touch Essentia. Stub out the heavy imports so importing
# harmonie.features at module load time doesn't try to load TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def fake_descriptors():
    from harmonie.features import Descriptors

    def _make(**overrides):
        d = Descriptors(
            bpm=128.0,
            bpm_confidence=2.5,
            key="C",
            scale="major",
            key_strength=0.8,
            loudness_db=-12.0,
            danceability=1.5,
            onset_rate=4.2,
        )
        for k, v in overrides.items():
            setattr(d, k, v)
        return d

    return _make


@pytest.fixture
def random_embedding():
    rng = np.random.default_rng(42)

    def _make(dim: int = 1280, seed: int | None = None) -> np.ndarray:
        local = rng if seed is None else np.random.default_rng(seed)
        v = local.standard_normal(dim).astype(np.float32)
        return v

    return _make


@pytest.fixture
def make_db(tmp_path: Path):
    """Build a Database + paired EmbeddingIndex pointing at a temp file.

    Returns a factory so tests can build multiple isolated DBs if needed.
    """
    from harmonie.db import Database
    from harmonie.index import EmbeddingIndex

    created: list[Database] = []

    def _factory(name: str = "test.db"):
        db = Database(tmp_path / name)
        created.append(db)
        return db, EmbeddingIndex(db)

    yield _factory
    for db in created:
        try:
            db.close()
        except Exception:
            pass
