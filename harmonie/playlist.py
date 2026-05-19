"""Playlist generation: similar-seeded, chained, and vibe-based.

All three generators read embeddings from the in-memory :class:`EmbeddingIndex`.
The DB is consulted only for descriptor metadata, filter gating, and the
artist/title tags that drive the diversity rules.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .db import Database, TrackFilter
from .index import EmbeddingIndex, l2_normalize_vec

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class Match:
    track_id: int
    path: str
    score: float


# ---------------------------------------------------------------------------
# Camelot wheel (DJ harmonic-mixing chart)
# ---------------------------------------------------------------------------

# Map (key, scale) -> Camelot code. Sharp and flat spellings both
# registered.
_CAMELOT: dict[tuple[str, str], str] = {
    ("Ab", "minor"): "1A",
    ("G#", "minor"): "1A",
    ("B", "major"): "1B",
    ("Eb", "minor"): "2A",
    ("D#", "minor"): "2A",
    ("F#", "major"): "2B",
    ("Gb", "major"): "2B",
    ("Bb", "minor"): "3A",
    ("A#", "minor"): "3A",
    ("Db", "major"): "3B",
    ("C#", "major"): "3B",
    ("F", "minor"): "4A",
    ("Ab", "major"): "4B",
    ("G#", "major"): "4B",
    ("C", "minor"): "5A",
    ("Eb", "major"): "5B",
    ("D#", "major"): "5B",
    ("G", "minor"): "6A",
    ("Bb", "major"): "6B",
    ("A#", "major"): "6B",
    ("D", "minor"): "7A",
    ("F", "major"): "7B",
    ("A", "minor"): "8A",
    ("C", "major"): "8B",
    ("E", "minor"): "9A",
    ("G", "major"): "9B",
    ("B", "minor"): "10A",
    ("D", "major"): "10B",
    ("F#", "minor"): "11A",
    ("Gb", "minor"): "11A",
    ("A", "major"): "11B",
    ("C#", "minor"): "12A",
    ("Db", "minor"): "12A",
    ("E", "major"): "12B",
}


def camelot_of(key: str | None, scale: str | None) -> str | None:
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


def compatible_keys_for(key: str | None, scale: str | None) -> set[tuple[str, str]]:
    """The (key, scale) pairs harmonically compatible with the given one."""
    code = camelot_of(key, scale)
    if code is None:
        return set()
    targets = compatible_camelot(code)
    return {pair for pair, c in _CAMELOT.items() if c in targets}


# ---------------------------------------------------------------------------
# Diversity (artist cooldown + same-song deduplication)
# ---------------------------------------------------------------------------

# Verdicts returned by :meth:`_DiversityState.admit`. Strings rather than
# an Enum so they're easy to grep, easy to read in stack traces, and
# trivially serialisable if we ever want to expose them.
_AdmitVerdict = Literal["ok", "duplicate", "cooldown"]


@dataclass(frozen=True)
class _DiversityPolicy:
    """How aggressively a playlist mixes artists and dedupes duplicates.

    ``artist_cooldown`` is the number of picks that must pass before the
    same artist can appear again. With ``cooldown=2`` (the default), a
    playlist sequence like ``A, B, C, A, B, C, A`` is fine but
    ``A, A, ...`` and ``A, B, A`` are not. Set to ``0`` or ``None`` to
    allow back-to-back repeats. The cooldown is a sliding window over
    the playlist's tail: it does not bound the total number of picks
    per artist over the whole playlist, only their spacing.

    ``dedupe_titles=False`` lets the same ``(artist, title)`` tag pair
    appear multiple times — useful when paths matter more than tags
    (e.g. studio + live versions of the same song that happen to share
    a title). The dedup rule is permanent for the playlist; unlike the
    cooldown, it never relaxes.

    The defaults (cooldown=2, dedupe on) target the typical case: a
    personal library where you don't want the same artist clustered
    back-to-back and don't want the same song from three different
    compilations. The cooldown lets a single artist appear many times
    in a long playlist as long as other artists appear between them.
    """

    artist_cooldown: int | None = 2
    dedupe_titles: bool = True

    @classmethod
    def disabled(cls) -> _DiversityPolicy:
        """Pre-built policy with all rules off — equivalent to the
        pre-diversity behaviour. Useful in tests and as an opt-out."""
        return cls(artist_cooldown=None, dedupe_titles=False)


class _DiversityState:
    """Mutable book-keeping for one playlist's diversity decisions.

    Construct fresh per playlist; thread the same instance through the
    full pick loop. The state separates from :class:`_DiversityPolicy`
    so policies stay immutable and shareable.

    Tracks are addressed by their ``(artist, title)`` tag pair, both
    case-folded and stripped. Tracks with no artist tag are always
    admissible — the cooldown can only act on what's identifiable.
    """

    def __init__(self, policy: _DiversityPolicy) -> None:
        self.policy = policy
        # Ordered history of normalised artist keys, one entry per
        # recorded pick. Empty strings are appended for tag-less tracks
        # so the cooldown window advances with playlist position.
        self._history: list[str] = []
        self._seen_titles: set[tuple[str, str]] = set()

    @staticmethod
    def _norm(s: str | None) -> str:
        return (s or "").strip().lower()

    def admit(self, artist: str | None, title: str | None) -> _AdmitVerdict:
        """Classify a candidate without recording it.

        Returns:
            ``"ok"`` — admit; the caller should also call :meth:`record`.
            ``"duplicate"`` — same ``(artist, title)`` already in the
                playlist; reject permanently.
            ``"cooldown"`` — artist appeared within the cooldown window
                of recent picks; reject in the constrained pass, accept
                in the relaxation pass.
        """
        a = self._norm(artist)
        t = self._norm(title)
        if self.policy.dedupe_titles and a and t and (a, t) in self._seen_titles:
            return "duplicate"
        cooldown = self.policy.artist_cooldown
        # `a and ...` short-circuits so tag-less tracks are always ok.
        if cooldown and a and a in self._history[-cooldown:]:
            return "cooldown"
        return "ok"

    def record(self, artist: str | None, title: str | None) -> None:
        """Mark a candidate as picked. Caller is responsible for calling
        this exactly once per accepted track."""
        a = self._norm(artist)
        t = self._norm(title)
        if a and t:
            self._seen_titles.add((a, t))
        # Append unconditionally — the slot exists in the playlist even
        # when the artist tag is empty, and the cooldown window measures
        # positions, not entries.
        self._history.append(a)


def _tags_lookup(
    db: Database, model: str, policy: _DiversityPolicy
) -> dict[int, tuple[str | None, str | None]]:
    """Bulk-fetch ``{track_id: (artist, title)}`` for the playlist's model.
    Skips the DB call entirely when the policy is fully disabled."""
    if policy.artist_cooldown is None and not policy.dedupe_titles:
        return {}
    return db.artist_title_by_id_for_model(model)


# ---------------------------------------------------------------------------
# Similar-seeded playlist
# ---------------------------------------------------------------------------


@dataclass
class SimilarPlaylistRequest:
    seed_ids: list[int]
    n: int = 20
    bpm_drift: float | None = None
    harmonic_mix: bool = False
    descriptor_filter: TrackFilter | None = None
    include_seeds: bool = False
    diversity: _DiversityPolicy = field(default_factory=_DiversityPolicy)


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
    allowed_ids: set[int] | None = None
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
    # bpm_drift gate uses just the BPM column from the (bpm, key, scale)
    # bulk lookup. Cheaper than a second query.
    bpm_lookup: dict[int, float | None] = (
        {tid: meta[0] for tid, meta in db.bpm_key_by_id_for_model(model).items()}
        if req.bpm_drift is not None
        else {}
    )
    tags_lookup = _tags_lookup(db, model, req.diversity)

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
    state = _DiversityState(req.diversity)
    # When seeds are excluded from the output, their (artist, title) tags
    # must still count toward dedup — otherwise a different file of the
    # same song slips in as the first pick. When seeds are included, the
    # natural picking flow records them as they're picked.
    if not req.include_seeds:
        for r in seed_rows:
            state.record(r.get("artist"), r.get("title"))
    chosen: list[tuple[int, str, float, np.ndarray]] = []
    prev_emb: np.ndarray = centroid_n
    prev_bpm: float | None = seed_bpms[0] if seed_bpms else None

    def _passes_smoothness(tid: int, bpm: float | None) -> bool:
        if req.bpm_drift is None:
            return True
        if (
            prev_bpm is not None
            and bpm is not None
            and abs(bpm - prev_bpm) > req.bpm_drift
        ):
            return False
        return not (
            seed_bpms
            and bpm is not None
            and min(abs(bpm - s) for s in seed_bpms) > req.bpm_drift * 2
        )

    def _best_pick(allow_cooldown: bool) -> int:
        """Index into ``candidates`` of the best admissible pick, or -1."""
        best_idx = -1
        best_score = -2.0
        for i, (tid, _path, _seed_score, emb) in enumerate(candidates):
            if not _passes_smoothness(tid, bpm_lookup.get(tid)):
                continue
            artist, title = tags_lookup.get(tid, (None, None))
            verdict = state.admit(artist, title)
            if verdict == "duplicate":
                continue
            if verdict == "cooldown" and not allow_cooldown:
                continue
            sim = float(emb @ prev_emb)
            if sim > best_score:
                best_score = sim
                best_idx = i
        return best_idx

    while candidates and len(chosen) < req.n:
        # Constrained pass: respect both dedup and artist cap.
        best_idx = _best_pick(allow_cooldown=False)
        # Relaxation pass: cap dropped, dedup still applies. Lets the
        # playlist reach `n` even when one artist dominates the pool.
        if best_idx < 0:
            best_idx = _best_pick(allow_cooldown=True)
        if best_idx < 0:
            break
        picked = candidates.pop(best_idx)
        chosen.append(picked)
        artist, title = tags_lookup.get(picked[0], (None, None))
        state.record(artist, title)
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

    Take the top ``chunk_size`` similar tracks to the anchor; re-anchor on
    the last track of that chunk and take the next ``chunk_size`` similars;
    repeat until the playlist has ``n`` tracks (or unique candidates run
    out). No track is ever repeated, so the chain can't loop back on itself.

    Multiple seeds are allowed. The starting anchor is the seeds' embedding
    centroid; the consecutive-transition baseline (``bpm_drift``,
    ``harmonic_mix``) starts from the *first* seed, mirroring how
    ``generate_similar_playlist`` treats its seed list.

    ``bpm_drift`` and ``harmonic_mix`` enforce smooth transitions between
    *consecutive* picks (each new track is compatible with the immediately
    previous pick). They apply both within and across chunks.
    """

    seed_ids: list[int]
    chunk_size: int = 5
    n: int = 20
    descriptor_filter: TrackFilter | None = None
    include_seed: bool = False
    bpm_drift: float | None = None
    harmonic_mix: bool = False
    diversity: _DiversityPolicy = field(default_factory=_DiversityPolicy)


def generate_chained_playlist(
    db: Database, index: EmbeddingIndex, req: ChainedPlaylistRequest
) -> list[Match]:
    if req.chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if not req.seed_ids:
        raise ValueError("seed_ids must contain at least one track id")
    if req.n < 1:
        return []

    # Resolve all seed rows up front and verify they share a model.
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

    seed_indices: list[int] = []
    for r in seed_rows:
        idx = cached.id_to_row.get(int(r["id"]))
        if idx is None:
            return []  # stale state; bail
        seed_indices.append(idx)

    allowed_ids: set[int] | None = None
    if req.descriptor_filter is not None and not req.descriptor_filter.is_empty():
        allowed_ids = db.filtered_ids(filter=req.descriptor_filter, model=model)

    # Pre-fetch BPM/key metadata for every candidate. One query, used by
    # the per-pick consecutive-transition checks below.
    needs_meta = req.bpm_drift is not None or req.harmonic_mix
    track_meta = db.bpm_key_by_id_for_model(model) if needs_meta else {}
    tags_lookup = _tags_lookup(db, model, req.diversity)

    visited: set[int] = set(req.seed_ids)
    chosen: list[Match] = []
    state = _DiversityState(req.diversity)
    if req.include_seed:
        # Emit each seed in input order, with score 1.0 (perfect self-match).
        # Seeds count toward the diversity budget — if you seed three Aphex
        # Twin tracks with cap=2, the cap is already exceeded and the
        # relaxation pass takes over for further picks.
        for sid, idx in zip(req.seed_ids, seed_indices):
            chosen.append(Match(track_id=sid, path=cached.paths[idx], score=1.0))
            artist, title = tags_lookup.get(sid, (None, None))
            state.record(artist, title)
    else:
        # Seeds don't appear in the output but their (artist, title) tags
        # still anchor dedup. Otherwise a different file of the same song
        # gets picked first.
        for sid in req.seed_ids:
            artist, title = tags_lookup.get(sid, (None, None))
            state.record(artist, title)

    # L2-normalized centroid of the seed embeddings, used as the starting
    # anchor. Collapses to the single seed's vector when there's one seed.
    anchor_emb = cached.matrix[seed_indices].mean(axis=0)
    anchor_emb = l2_normalize_vec(anchor_emb.astype(np.float32, copy=False))

    # Consecutive-transition baseline starts at the first seed. Updates to
    # the most recent pick after each iteration.
    first_seed = seed_rows[0]
    prev_bpm: float | None = first_seed.get("bpm")
    prev_key: str | None = first_seed.get("key")
    prev_scale: str | None = first_seed.get("scale")

    def _find_one_pick(
        allow_cooldown: bool,
        chunk_prev_bpm: float | None,
        chunk_prev_key: str | None,
        chunk_prev_scale: str | None,
        in_chunk_ids: set[int],
    ) -> tuple[int, int, str | None, str | None] | None:
        """Walk the sorted candidates and return the first admissible
        pick as ``(idx_in_argsort, track_id, artist, title)``. Returns
        ``None`` if nothing qualifies. Smoothness state (bpm/key) is
        passed in so the caller advances it after each pick."""
        for idx in np.argsort(-scores):
            tid = cached.ids[idx]
            if tid in visited or tid in in_chunk_ids:
                continue
            if allowed_ids is not None and tid not in allowed_ids:
                continue

            cand_bpm = cand_key = cand_scale = None
            if needs_meta:
                meta = track_meta.get(tid)
                if meta is not None:
                    cand_bpm, cand_key, cand_scale = meta

            # Harmonic-mix: candidate's key must be Camelot-compatible
            # with the previous pick. Strict — tracks without key info
            # are excluded when this constraint is on.
            if req.harmonic_mix:
                if not cand_key or not cand_scale:
                    continue
                if chunk_prev_key and chunk_prev_scale:
                    ok_pairs = compatible_keys_for(chunk_prev_key, chunk_prev_scale)
                    if (cand_key, cand_scale) not in ok_pairs:
                        continue

            # bpm_drift: candidate's BPM within tolerance of previous pick.
            # Lenient — skip the check if either side has no BPM info.
            if req.bpm_drift is not None and (
                chunk_prev_bpm is not None
                and cand_bpm is not None
                and abs(cand_bpm - chunk_prev_bpm) > req.bpm_drift
            ):
                continue

            artist, title = tags_lookup.get(tid, (None, None))
            verdict = state.admit(artist, title)
            if verdict == "duplicate":
                continue
            if verdict == "cooldown" and not allow_cooldown:
                continue

            return (int(idx), tid, artist, title)
        return None

    def _scan_chunk() -> list[tuple[Match, dict]]:
        """Pick up to ``chunk_size`` tracks. Each pick is constrained-
        first, relaxed-only-if-needed, so we never produce back-to-back
        same artist when there's any other-artist candidate available."""
        chunk: list[tuple[Match, dict]] = []
        chunk_prev_bpm = prev_bpm
        chunk_prev_key = prev_key
        chunk_prev_scale = prev_scale
        in_chunk_ids: set[int] = set()

        while len(chunk) < req.chunk_size:
            pick = _find_one_pick(
                allow_cooldown=False,
                chunk_prev_bpm=chunk_prev_bpm,
                chunk_prev_key=chunk_prev_key,
                chunk_prev_scale=chunk_prev_scale,
                in_chunk_ids=in_chunk_ids,
            )
            if pick is None:
                # No constrained pick. Try once with cooldown relaxed.
                pick = _find_one_pick(
                    allow_cooldown=True,
                    chunk_prev_bpm=chunk_prev_bpm,
                    chunk_prev_key=chunk_prev_key,
                    chunk_prev_scale=chunk_prev_scale,
                    in_chunk_ids=in_chunk_ids,
                )
            if pick is None:
                break
            idx, tid, artist, title = pick
            cand_bpm = cand_key = cand_scale = None
            if needs_meta:
                meta = track_meta.get(tid)
                if meta is not None:
                    cand_bpm, cand_key, cand_scale = meta
            in_chunk_ids.add(tid)
            chunk.append(
                (
                    Match(
                        track_id=tid,
                        path=cached.paths[idx],
                        score=float(scores[idx]),
                    ),
                    {
                        "bpm": cand_bpm,
                        "key": cand_key,
                        "scale": cand_scale,
                        "artist": artist,
                        "title": title,
                    },
                )
            )
            # Record inline so the next pick in this chunk sees this
            # one in the cooldown window and dedup state.
            state.record(artist, title)
            chunk_prev_bpm = cand_bpm if cand_bpm is not None else chunk_prev_bpm
            chunk_prev_key = cand_key if cand_key else chunk_prev_key
            chunk_prev_scale = cand_scale if cand_scale else chunk_prev_scale
        return chunk

    while len(chosen) < req.n:
        scores = cached.matrix @ anchor_emb  # cached rows are normalised
        chunk = _scan_chunk()
        if not chunk:
            break  # no admissible candidates anywhere

        # Don't overshoot the requested length.
        chunk = chunk[: req.n - len(chosen)]
        for match, _meta in chunk:
            chosen.append(match)
            visited.add(match.track_id)
            # State was already recorded inline during the scan so
            # within-chunk admission decisions stay consistent.

        # Re-anchor on the last track and update the running prev_* state.
        last_id = chosen[-1].track_id
        anchor_emb = cached.matrix[cached.id_to_row[last_id]]
        if needs_meta:
            last_meta = track_meta.get(last_id)
            if last_meta is not None:
                last_bpm, last_key, last_scale = last_meta
                if last_bpm is not None:
                    prev_bpm = last_bpm
                if last_key:
                    prev_key = last_key
                if last_scale:
                    prev_scale = last_scale

    return chosen


# ---------------------------------------------------------------------------
# Vibe-based playlist
# ---------------------------------------------------------------------------


@dataclass
class VibePlaylistRequest:
    n: int = 20
    descriptor_filter: TrackFilter | None = None
    target_bpm: float | None = None
    target_danceability: float | None = None
    shuffle: bool = True
    seed: int | None = None
    diversity: _DiversityPolicy = field(default_factory=_DiversityPolicy)


def generate_vibe_playlist(
    db: Database, req: VibePlaylistRequest, *, model: str | None = None
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

    # Apply diversity in pool order. The cooldown is a sliding window:
    # a row that was on cooldown at iteration i may become admissible at
    # iteration j>i once enough other-artist picks shift the window. So
    # we do repeated constrained passes, only relaxing when a full pass
    # makes no progress. Dedup is permanent and never relaxes.
    state = _DiversityState(req.diversity)
    chosen: list[dict] = []
    remaining = list(pool)

    while len(chosen) < req.n and remaining:
        chose_this_iter = False
        # Phase 1: constrained — pick the first row that fully satisfies
        # the policy. Drop duplicates as we encounter them.
        i = 0
        while i < len(remaining):
            row = remaining[i]
            verdict = state.admit(row.get("artist"), row.get("title"))
            if verdict == "duplicate":
                remaining.pop(i)
                continue
            if verdict == "ok":
                remaining.pop(i)
                chosen.append(row)
                state.record(row.get("artist"), row.get("title"))
                chose_this_iter = True
                break
            i += 1  # cooldown — keep for a later pass
        if chose_this_iter:
            continue
        # Phase 2: relaxation for one pick. Take the first non-duplicate
        # row even if it's still on cooldown. After the pick, the next
        # iteration goes back to constrained mode.
        i = 0
        while i < len(remaining):
            row = remaining[i]
            verdict = state.admit(row.get("artist"), row.get("title"))
            if verdict == "duplicate":
                remaining.pop(i)
                continue
            remaining.pop(i)
            chosen.append(row)
            state.record(row.get("artist"), row.get("title"))
            chose_this_iter = True
            break
        if not chose_this_iter:
            break

    return [
        Match(track_id=int(r["id"]), path=r["path"], score=float(fitness(r)))
        for r in chosen
    ]
