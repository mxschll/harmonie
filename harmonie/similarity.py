"""Cosine-similarity search over the in-memory embedding index.

This module is a thin layer over :class:`harmonie.index.EmbeddingIndex`.
The actual matmul lives in the index; here we resolve track IDs to query
vectors and compose the optional descriptor-filter gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .db import Database, TrackFilter
from .index import EmbeddingIndex, IndexMatch


@dataclass
class Match:
    track_id: int
    path: str
    score: float  # cosine similarity in [-1, 1]


def _to_match(m: IndexMatch) -> Match:
    return Match(track_id=m.track_id, path=m.path, score=m.score)


def find_similar_to_id(
    db: Database,
    index: EmbeddingIndex,
    track_id: int,
    *,
    n: int = 10,
    filter: Optional[TrackFilter] = None,
    include_self: bool = False,
) -> list[Match]:
    """Top-N similar to the track with the given id.

    Restricts to the query track's own model so we never compare embeddings
    from different feature spaces.
    """
    row = db.get_track_by_id(track_id)
    if row is None:
        raise KeyError(f"track {track_id} not in database")
    model = row["model"]

    cached = index.get(model)
    row_idx = cached.id_to_row.get(track_id)
    if row_idx is None:
        return []
    # Cached matrix rows are already L2-normalised, so we hand them straight
    # to search() — search() will re-normalise (cheap) and run the matmul.
    query_vec = cached.matrix[row_idx]

    allowed_ids: Optional[set[int]] = None
    if filter is not None and not filter.is_empty():
        allowed_ids = db.filtered_ids(filter=filter, model=model)

    matches = index.search(
        query_vec,
        model=model,
        n=n,
        allowed_ids=allowed_ids,
        exclude_ids=set() if include_self else {track_id},
    )
    return [_to_match(m) for m in matches]


def find_similar_to_vector(
    db: Database,
    index: EmbeddingIndex,
    embedding: np.ndarray,
    *,
    model: str,
    n: int = 10,
    filter: Optional[TrackFilter] = None,
    exclude_ids: Optional[set[int]] = None,
) -> list[Match]:
    """Top-N similar to a pre-computed embedding (e.g. from an external
    request such as 'play me music like this audio I just uploaded')."""
    allowed_ids: Optional[set[int]] = None
    if filter is not None and not filter.is_empty():
        allowed_ids = db.filtered_ids(filter=filter, model=model)

    matches = index.search(
        embedding,
        model=model,
        n=n,
        allowed_ids=allowed_ids,
        exclude_ids=exclude_ids or set(),
    )
    return [_to_match(m) for m in matches]
