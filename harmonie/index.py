"""In-memory cache of embedding matrices for fast similarity search.

The DB is the source of truth. :class:`EmbeddingIndex` keeps an
L2-normalised matrix resident in process memory; queries become a single
matrix-vector multiply.

The cache is keyed by model name (a library that mixes models keeps
separate matrices) and is invalidated wholesale after every scan. It
rebuilds lazily on the next access.

Memory: at 1280-d / float32 the matrix is roughly 5 KB per track; 100k
tracks ≈ 500 MB resident.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

from .db import Database

logger = logging.getLogger("harmonie.index")


# ---------------------------------------------------------------------------
# Cached matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedMatrix:
    """L2-normalised embeddings for one model, plus row metadata.

    ``matrix[i]`` corresponds to ``ids[i]`` and ``paths[i]``. ``id_to_row``
    maps track id to row index.
    """

    ids: tuple[int, ...]
    paths: tuple[str, ...]
    matrix: np.ndarray  # shape (N, D), float32, L2-norm 1 per row
    id_to_row: dict[int, int]

    @property
    def empty(self) -> bool:
        return len(self.ids) == 0

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.size else 0


# ---------------------------------------------------------------------------
# Vector helpers
# ---------------------------------------------------------------------------


def _l2_normalize_matrix(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    return mat / norms


def l2_normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise a 1-D vector."""
    n = float(np.linalg.norm(v))
    return v / max(n, eps)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class EmbeddingIndex:
    """Thread-safe cache of L2-normalised embedding matrices per model.
    Build is serialised through a lock; queries don't hold the lock.
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._cache: dict[str, CachedMatrix] = {}
        self._lock = threading.Lock()

    # -- cache management ---------------------------------------------

    def invalidate(self, model: str | None = None) -> None:
        """Drop cached matrices. Called by the analyzer after every scan."""
        with self._lock:
            if model is None:
                self._cache.clear()
                logger.debug("index: invalidated all models")
            else:
                if self._cache.pop(model, None) is not None:
                    logger.debug("index: invalidated model %s", model)

    def get(self, model: str) -> CachedMatrix:
        """Return the matrix for ``model``, building it from the DB if needed."""
        with self._lock:
            cached = self._cache.get(model)
            if cached is None:
                cached = self._build(model)
                self._cache[model] = cached
        return cached

    def _build(self, model: str) -> CachedMatrix:
        ids, paths, mat = self._db.all_embeddings(model=model)
        if mat.size == 0:
            return CachedMatrix(ids=(), paths=(), matrix=mat, id_to_row={})
        norm = _l2_normalize_matrix(mat.astype(np.float32, copy=False))
        cached = CachedMatrix(
            ids=tuple(ids),
            paths=tuple(paths),
            matrix=norm,
            id_to_row={tid: i for i, tid in enumerate(ids)},
        )
        logger.info(
            "index: built model=%s tracks=%d dim=%d size=%.1fMiB",
            model,
            len(ids),
            norm.shape[1],
            norm.nbytes / (1024 * 1024),
        )
        return cached

    # -- search --------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        *,
        model: str,
        n: int,
        allowed_ids: set[int] | None = None,
        exclude_ids: set[int] | None = None,
    ) -> list[IndexMatch]:
        """Return the top-``n`` matches against ``query`` (a raw, un-normalised
        embedding) for the given ``model``.

        ``allowed_ids`` (if given) restricts results to that set; ``exclude_ids``
        always omits those tracks. Results are sorted by score descending.
        """
        cached = self.get(model)
        if cached.empty:
            return []
        if query.shape[0] != cached.dim:
            raise ValueError(f"query dim {query.shape[0]} != index dim {cached.dim}")

        q = l2_normalize_vec(query.astype(np.float32, copy=False))
        scores = cached.matrix @ q  # (N,)

        excluded = exclude_ids or set()
        out: list[IndexMatch] = []
        # Argsort once; iterate until n matches survive filtering.
        for idx in np.argsort(-scores):
            tid = cached.ids[idx]
            if tid in excluded:
                continue
            if allowed_ids is not None and tid not in allowed_ids:
                continue
            out.append(
                IndexMatch(
                    track_id=tid,
                    path=cached.paths[idx],
                    score=float(scores[idx]),
                )
            )
            if len(out) >= n:
                break
        return out


@dataclass(frozen=True)
class IndexMatch:
    track_id: int
    path: str
    score: float
