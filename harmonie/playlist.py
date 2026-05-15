"""Playlist generation: similar-seeded, chained, and vibe-based.

All three generators read embeddings from the in-memory :class:`EmbeddingIndex`.
The DB is consulted only for descriptor metadata and filter gating.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .db import Database, TrackFilter
from .index import EmbeddingIndex, l2_normalize_vec


# ---------------------------------------------------------------------------
# Result type (kept distinct from index.IndexMatch to decouple the public API)
# ---------------------------------------------------------------------------


@dataclass
class Match:
    track_id: int
    path: str
    score: float


# ---------------------------------------------------------------------------
# Camelot wheel (DJ harmonic-mixing chart)
# ---------------------------------------------------------------------------

# Map (key, scale) -> Camelot code. Both sharp and flat spellings are
# registered so we don't depend on which Essentia profile produced the key.
_CAMELOT: dict[tuple[str, str], str] = {
    ("Ab", "minor"): "1A",  ("G#", "minor"): "1A",
    ("B",  "major"): "1B",
    ("Eb", "minor"): "2A",  ("D#", "minor"): "2A",
    ("F#", "major"): "2B",  ("Gb", "major"): "2B",
    ("Bb", "minor"): "3A",  ("A#", "minor"): "3A",
    ("Db", "major"): "3B",  ("C#", "major"): "3B",
    ("F",  "minor"): "4A",
    ("Ab", "major"): "4B",  ("G#", "major"): "4B",
    ("C",  "minor"): "5A",
    ("Eb", "major"): "5B",  ("D#", "major"): "5B",
    ("G",  "minor"): "6A",
    ("Bb", "major"): "6B",  ("A#", "major"): "6B",
    ("D",  "minor"): "7A",
    ("F",  "major"): "7B",
    ("A",  "minor"): "8A",
    ("C",  "major"): "8B",
    ("E",  "minor"): "9A",
    ("G",  "major"): "9B",
    ("B",  "minor"): "10A",
    ("D",  "major"): "10B",
    ("F#", "minor"): "11A", ("Gb", "minor"): "11A",
    ("A",  "major"): "11B",
    ("C#", "minor"): "12A", ("Db", "minor"): "12A",
    ("E",  "major"): "12B",
}


def camelot_of(key: Optional[str], scale: Optional[str]) -> Optional[str]:
    if not key or not scale:
        return None
    return _CAMELOT.get((key, scale.lower()))


def compatible_camelot(code: str) -> set[str]:
    """Codes that mix harmonically with ``code``: itself, ±1 number on the
    same letter, and the parallel mode (same number, opposite letter)."""
    n = int(code[:-1])
    letter = code[-1]
    other = "B" if letter == "A" else "A"
    nums_neighbor = (((n - 1 - 1) % 12) + 1, ((n - 1 + 1) % 12) + 1)
    return {
        f"{n}{letter}",
        f"{nums_neighbor[0]}{letter}",
        f"{nums_neighbor[1]}{letter}",
        f"{n}{other}",
    }


def compatible_keys_for(
    key: Optional[str], scale: Optional[str]
) -> set[tuple[str, str]]:
    """The (key, scale) pairs harmonically compatible with the given one."""
    code = camelot_of(key, scale)
    if code is None:
        return set()
    targets = compatible_camelot(code)
    return {pair for pair, c in _CAMELOT.items() if c in targets}


# ---------------------------------------------------------------------------
# Similar-seeded playlist
# ---------------------------------------------------------------------------


@dataclass
class SimilarPlaylistRequest:
    seed_ids: list[int]
    n: int = 20
    bpm_drift: Optional[float] = None
    harmonic_mix: bool = False
    descriptor_filter: Optional[TrackFilter] = None
    include_seeds: bool = False


def generate_similar_playlist(
    db: Database, index: EmbeddingIndex, req: SimilarPlaylistRequest
) -> list[Match]:
    if not req.seed_ids:
        raise ValueError("seed_ids must contain at least one track id")
    if req.n <= 0:
        return []

    # Resolve seed metadata (model, key, bpm) — embeddings come from the index.
    seed_rows = []
    for sid in req.seed_ids:
        row = db.get_track_by_id(sid)
        if row is None:
            raise KeyError(f"seed track {sid} not in database")
        seed_rows.append(row)

    models = {r["model"] for r in seed_rows}
    if len(models) > 1:
        raise ValueError(f"seed tracks span multiple models: {models}")
    model = next(iter(models))

    cached = index.get(model)
    if cached.empty:
        return []

    # Seed embeddings, taken from the cached (already-normalised) matrix.
    seed_indices: list[int] = []
    for r in seed_rows:
        idx = cached.id_to_row.get(int(r["id"]))
        if idx is None:
            return []  # stale state; bail
        seed_indices.append(idx)
    centroid = cached.matrix[seed_indices].mean(axis=0)

    # Allowed-ID gate: descriptor filter plus optional harmonic-mix restriction.
    allowed_ids: Optional[set[int]] = None
    if req.descriptor_filter is not None and not req.descriptor_filter.is_empty():
        allowed_ids = db.filtered_ids(filter=req.descriptor_filter, model=model)

    if req.harmonic_mix:
        # Harmonic constraint applies to the first seed's key.
        first = seed_rows[0]
        ok_keys = list(compatible_keys_for(first.get("key"), first.get("scale")))
        if ok_keys:
            harmonic_ids = db.harmonic_compatible_ids(model=model, pairs=ok_keys)
            allowed_ids = (
                harmonic_ids if allowed_ids is None else (allowed_ids & harmonic_ids)
            )

    seed_id_set = set(req.seed_ids)
    seed_bpms = [r["bpm"] for r in seed_rows if r.get("bpm") is not None]
    bpm_lookup = (
        db.bpm_by_id_for_model(model) if req.bpm_drift is not None else {}
    )

    # Score every track against the centroid and oversample for the walk.
    centroid_n = l2_normalize_vec(centroid.astype(np.float32, copy=False))
    scores = cached.matrix @ centroid_n
    candidates: list[tuple[int, str, float, np.ndarray]] = []
    for idx in np.argsort(-scores):
        tid = cached.ids[idx]
        if not req.include_seeds and tid in seed_id_set:
            continue
        if allowed_ids is not None and tid not in allowed_ids:
            continue
        candidates.append(
            (tid, cached.paths[idx], float(scores[idx]), cached.matrix[idx])
        )
        if len(candidates) >= req.n * 6:  # oversample for the greedy walk
            break

    if not candidates:
        return []

    # Greedy nearest-neighbour walk for smooth transitions. Treat the
    # centroid (and one seed BPM) as the "previous" state for the first pick
    # so bpm_drift applies between the seed and the first selected track.
    chosen: list[tuple[int, str, float, np.ndarray]] = []
    prev_emb: np.ndarray = centroid_n
    prev_bpm: Optional[float] = seed_bpms[0] if seed_bpms else None

    while candidates and len(chosen) < req.n:
        best_idx = -1
        best_score = -2.0
        for i, (tid, _path, _seed_score, emb) in enumerate(candidates):
            sim = float(emb @ prev_emb)
            if req.bpm_drift is not None:
                bpm = bpm_lookup.get(tid)
                if prev_bpm is not None and bpm is not None:
                    if abs(bpm - prev_bpm) > req.bpm_drift:
                        continue
                if seed_bpms and bpm is not None:
                    if min(abs(bpm - s) for s in seed_bpms) > req.bpm_drift * 2:
                        continue
            if sim > best_score:
                best_score = sim
                best_idx = i
        if best_idx < 0:
            break
        picked = candidates.pop(best_idx)
        chosen.append(picked)
        prev_emb = picked[3]
        prev_bpm = bpm_lookup.get(picked[0])

    return [
        Match(track_id=tid, path=path, score=score)
        for (tid, path, score, _emb) in chosen
    ]


# ---------------------------------------------------------------------------
# Chained ("smart") playlist
# ---------------------------------------------------------------------------


@dataclass
class ChainedPlaylistRequest:
    """Walk the embedding space in chunks.

    Take the top ``chunk_size`` similar tracks to the seed; re-anchor on the
    last track of that chunk and take the next ``chunk_size`` similars; repeat
    until the playlist has ``n`` tracks (or unique candidates run out). No
    track is ever repeated, so the chain can't loop back on itself.
    """

    seed_id: int
    chunk_size: int = 5
    n: int = 20
    descriptor_filter: Optional[TrackFilter] = None
    include_seed: bool = False


def generate_chained_playlist(
    db: Database, index: EmbeddingIndex, req: ChainedPlaylistRequest
) -> list[Match]:
    if req.chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if req.n < 1:
        return []

    seed_row = db.get_track_by_id(req.seed_id)
    if seed_row is None:
        raise KeyError(f"seed track {req.seed_id} not in database")
    model = seed_row["model"]

    cached = index.get(model)
    if cached.empty:
        return []
    seed_idx = cached.id_to_row.get(req.seed_id)
    if seed_idx is None:
        return []

    allowed_ids: Optional[set[int]] = None
    if req.descriptor_filter is not None and not req.descriptor_filter.is_empty():
        allowed_ids = db.filtered_ids(filter=req.descriptor_filter, model=model)

    visited: set[int] = {req.seed_id}
    chosen: list[Match] = []
    if req.include_seed:
        chosen.append(
            Match(track_id=req.seed_id, path=cached.paths[seed_idx], score=1.0)
        )

    anchor_emb: np.ndarray = cached.matrix[seed_idx]

    while len(chosen) < req.n:
        scores = cached.matrix @ anchor_emb  # cached rows are normalised
        chunk: list[Match] = []
        for idx in np.argsort(-scores):
            tid = cached.ids[idx]
            if tid in visited:
                continue
            if allowed_ids is not None and tid not in allowed_ids:
                continue
            chunk.append(
                Match(
                    track_id=tid,
                    path=cached.paths[idx],
                    score=float(scores[idx]),
                )
            )
            if len(chunk) >= req.chunk_size:
                break
        if not chunk:
            break  # ran out of candidates

        # Don't overshoot the requested length.
        chunk = chunk[: req.n - len(chosen)]
        for m in chunk:
            chosen.append(m)
            visited.add(m.track_id)

        anchor_emb = cached.matrix[cached.id_to_row[chosen[-1].track_id]]

    return chosen


# ---------------------------------------------------------------------------
# Vibe-based playlist
# ---------------------------------------------------------------------------


@dataclass
class VibePlaylistRequest:
    n: int = 20
    descriptor_filter: Optional[TrackFilter] = None
    target_bpm: Optional[float] = None
    target_danceability: Optional[float] = None
    shuffle: bool = True
    seed: Optional[int] = None


def generate_vibe_playlist(
    db: Database, req: VibePlaylistRequest, *, model: Optional[str] = None
) -> list[Match]:
    rows, _total = db.list_tracks(
        filter=req.descriptor_filter, model=model, limit=10_000, offset=0
    )
    if not rows:
        return []

    def fitness(row: dict) -> float:
        score = 0.0
        if req.target_bpm is not None:
            bpm = row.get("bpm")
            score += -10.0 if bpm is None else -abs(bpm - req.target_bpm) / 5.0
        if req.target_danceability is not None:
            d = row.get("danceability")
            score += -1.0 if d is None else -abs(d - req.target_danceability) * 2.0
        return score

    has_targets = req.target_bpm is not None or req.target_danceability is not None
    if has_targets:
        rows.sort(key=fitness, reverse=True)
        pool = rows[: max(req.n * 5, req.n)]
    else:
        pool = rows

    if req.shuffle:
        random.Random(req.seed).shuffle(pool)

    return [
        Match(track_id=int(r["id"]), path=r["path"], score=float(fitness(r)))
        for r in pool[: req.n]
    ]
