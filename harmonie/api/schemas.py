"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Track resource
# ---------------------------------------------------------------------------


class StyleScore(BaseModel):
    """One Discogs-400 style classification with its probability."""

    style: str = Field(
        ...,
        description=(
            "Discogs-400 label, formatted as ``Genre---Style`` "
            "(e.g. ``Electronic---House``)."
        ),
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Sigmoid output from the classifier head.",
    )


class TrackSummary(BaseModel):
    """Lightweight track row used in list responses."""

    id: int
    path: str
    library_root: Optional[str] = None
    relative_path: Optional[str] = None
    duration: float
    model: str
    # Tags from the file itself — useful for matching against external catalogs.
    artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None
    track_number: Optional[int] = None
    # Musical descriptors.
    bpm: Optional[float] = None
    key: Optional[str] = None
    scale: Optional[str] = None
    danceability: Optional[float] = None
    loudness_db: Optional[float] = None
    # Top Discogs-400 style activations, highest first. Empty list when the
    # track was indexed without the genre classifier head.
    styles: list[StyleScore] = Field(default_factory=list)


class Track(TrackSummary):
    """Full track record."""

    size: int
    mtime: float
    embedding_dim: int
    descriptor_version: int
    bpm_confidence: Optional[float] = None
    key_strength: Optional[float] = None
    onset_rate: Optional[float] = None
    analyzed_at: float


class TrackList(BaseModel):
    items: list[TrackSummary]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Filter query (used by list, similar, playlist)
# ---------------------------------------------------------------------------


class FilterParams(BaseModel):
    bpm_min: Optional[float] = None
    bpm_max: Optional[float] = None
    key: Optional[list[str]] = None
    scale: Optional[str] = None
    danceability_min: Optional[float] = None
    danceability_max: Optional[float] = None
    loudness_min: Optional[float] = None
    loudness_max: Optional[float] = None
    # Discogs-400 style filter. Each entry is either a full ``Genre---Style``
    # label (exact match) or a bare genre like ``Electronic`` (prefix match,
    # so ``Electronic`` matches ``Electronic---House``,
    # ``Electronic---Techno``, …).
    styles: Optional[list[str]] = Field(
        None,
        description=(
            "Restrict to tracks whose top styles include any of these. "
            "Use ``Genre---Style`` for an exact match or a bare ``Genre`` "
            "to match the whole branch."
        ),
    )
    style_min_probability: float = Field(
        0.0, ge=0.0, le=1.0,
        description=(
            "Minimum classifier probability the matching style row must "
            "have to count. 0 = any top-K hit; raise to demand confidence."
        ),
    )
    style_match: str = Field(
        "any",
        pattern="^(any|all)$",
        description=(
            "``any``: track must match at least one requested style "
            "(default). ``all``: every requested style must be present."
        ),
    )


class StyleEnumeration(BaseModel):
    """One row of GET /styles output."""

    style: str
    track_count: int
    mean_probability: float
    max_probability: float


class StyleList(BaseModel):
    items: list[StyleEnumeration]
    total: int


# ---------------------------------------------------------------------------
# Similarity / playlist
# ---------------------------------------------------------------------------


class MatchOut(BaseModel):
    """A similarity-search or playlist hit, enriched with the metadata an
    external client needs to map it back to its own catalog without doing a
    filesystem walk.
    """

    track_id: int
    path: str
    score: float
    library_root: Optional[str] = None
    relative_path: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None
    track_number: Optional[int] = None
    styles: list[StyleScore] = Field(default_factory=list)


class SimilarResult(BaseModel):
    query_id: int
    matches: list[MatchOut]


class PlaylistBody(BaseModel):
    """Single endpoint for all playlist generation. The mode is implicit
    from the parameters you set — there's no separate "mode" enum to choose.

    Modes:

    * **No seeds** → descriptor-driven playlist. ``filter`` constrains the
      candidate pool; ``target_bpm`` / ``target_danceability`` pull tracks
      toward those values; ``shuffle`` randomises the order.

    * **One or more seeds** → similarity-driven playlist. The seed track(s)
      anchor the playlist; the result stays in the same neighbourhood of
      embedding space. ``bpm_tolerance`` and ``key_compatible`` add
      smooth-transition constraints.

    * **One seed + drift=true** → "drifting" playlist. Each new track's
      anchor is the previous selection, so the playlist gradually walks
      away from the seed in style. Useful for long mixes that evolve.
    """

    n: int = Field(20, ge=1, le=500, description="Number of tracks to return.")

    # ---- anchoring ---------------------------------------------------
    seeds: list[int] = Field(
        default_factory=list,
        description=(
            "Track IDs to anchor on. With seeds the playlist is built from "
            "cosine similarity in the embedding space. Without seeds the "
            "playlist is built from descriptor targets and filters."
        ),
    )
    drift: bool = Field(
        False,
        description=(
            "Walk away from the seed track instead of staying near it. "
            "Each new selection becomes the anchor for the next, so the "
            "playlist drifts in style. Requires exactly one seed."
        ),
    )
    chunk_size: int = Field(
        5,
        ge=1,
        le=100,
        description=(
            "How many tracks to take per anchor in drift mode. The "
            "playlist takes the top-N most similar to the seed, then "
            "re-anchors on the last of those and takes its top-N, and so "
            "on. Larger chunks stay closer to the seed; smaller chunks "
            "drift faster. Only used when drift is true."
        ),
    )

    # ---- candidate filter --------------------------------------------
    filter: Optional[FilterParams] = Field(
        None,
        description=(
            "Hard constraints on candidate tracks (BPM range, key, "
            "scale, danceability range, loudness range)."
        ),
    )

    # ---- smooth-transition rules (similarity modes only) -------------
    bpm_tolerance: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Maximum BPM gap allowed between consecutive tracks. Honored in "
            "both default and drift modes when seeds are provided."
        ),
    )
    key_compatible: bool = Field(
        False,
        description=(
            "Restrict candidates to keys that mix harmonically with the "
            "previous track (Camelot wheel: same key, ±1 number on the "
            "wheel, or parallel mode). Honored in both default and drift "
            "modes when seeds are provided. In default mode the constraint "
            "is anchored on the first seed; in drift mode it follows the "
            "running anchor."
        ),
    )

    # ---- descriptor targets (descriptor-only mode) -------------------
    target_bpm: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Rank candidates by closeness to this BPM. Only applied when "
            "seeds is empty."
        ),
    )
    target_danceability: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Rank candidates by closeness to this danceability score. Only "
            "applied when seeds is empty."
        ),
    )

    # ---- output --------------------------------------------------------
    include_seeds: bool = Field(
        False,
        description="Include the seed tracks in the result.",
    )
    shuffle: bool = Field(
        True,
        description=(
            "Shuffle the result. Only applied when seeds is empty — "
            "similarity- and drift-driven playlists are always ordered."
        ),
    )
    rng_seed: Optional[int] = Field(
        None,
        description="RNG seed for reproducible shuffling.",
    )


class TrackLookupBody(BaseModel):
    """Body for POST /tracks/lookup. All fields optional, but at least one
    must be provided. The handler tries path → relative_path → tag triple →
    looser tag pair, returning the first match (smallest id wins on ties)."""

    path: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None


class PlaylistResult(BaseModel):
    items: list[MatchOut]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class ScanStatusOut(BaseModel):
    state: str
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    last_duration_sec: Optional[float] = None
    last_error: Optional[str] = None
    discovered: int = 0
    full: int = 0
    descriptors_only: int = 0
    skipped: int = 0
    failed: int = 0
    removed: int = 0
    recent_failures: list[tuple[str, str]] = Field(default_factory=list)


class ServiceStatus(BaseModel):
    version: str
    backend: str
    embedding_dim: int
    libraries: list[str]
    workers: int
    db_path: str
    db_size_bytes: int
    tracks: int
    total_duration_sec: float
    by_model: dict[str, int]
    scan: ScanStatusOut


class ScanTriggerBody(BaseModel):
    force: bool = False


class ScanTriggerResult(BaseModel):
    triggered: bool
    state: str
