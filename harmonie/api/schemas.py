"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

from .filters import FilterBody


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
    artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None
    track_number: Optional[int] = None
    bpm: Optional[float] = None
    key: Optional[str] = None
    scale: Optional[str] = None
    danceability: Optional[float] = None
    loudness: Optional[float] = None
    styles: list[StyleScore] = Field(default_factory=list)


class Track(TrackSummary):
    """Full track record with file metadata."""

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
# Style enumeration
# ---------------------------------------------------------------------------


class StyleEnumeration(BaseModel):
    style: str
    track_count: int
    mean_probability: float
    max_probability: float


class StyleList(BaseModel):
    items: list[StyleEnumeration]
    total: int


# ---------------------------------------------------------------------------
# Resolve (find one track by path/tags)
# ---------------------------------------------------------------------------
#
# No request schema — all four fields go in the query string.


# ---------------------------------------------------------------------------
# Similarity / playlist matches
# ---------------------------------------------------------------------------


class MatchOut(BaseModel):
    """A similarity-search or playlist hit, enriched with the metadata an
    external client needs to map it back to its own catalog without doing
    a filesystem walk."""

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


class PlaylistResult(BaseModel):
    items: list[MatchOut]


# ---------------------------------------------------------------------------
# Playlist body — discriminated by ``mode``
# ---------------------------------------------------------------------------
#
# Each mode has its own validated schema. The OpenAPI document renders these
# as a ``oneOf`` so the published contract makes the three shapes explicit
# rather than relying on parameter-implicit dispatch.
#


class _SmoothTransitions(BaseModel):
    """Consecutive-pair smoothness rules. Used by ``similar`` and ``drift``
    modes; ignored by ``vibe`` mode (no anchor pair to smooth between)."""

    bpm_tolerance: Optional[float] = Field(
        None, ge=0,
        description=(
            "Maximum BPM gap allowed between consecutive picks. Lenient on "
            "missing BPMs (skip the check rather than reject the candidate)."
        ),
    )
    key_compatible: bool = Field(
        False,
        description=(
            "Restrict consecutive picks to keys that mix harmonically "
            "(Camelot wheel: same key, ±1 wheel position, or parallel mode). "
            "Strict — tracks without key info are excluded when on."
        ),
    )


class _DescriptorTarget(BaseModel):
    """Soft preferences for ``vibe`` mode: tracks closer to these values rank
    higher. Pure ranking — does not gate the candidate pool."""

    bpm: Optional[float] = Field(None, gt=0)
    danceability: Optional[float] = Field(None, ge=0)


class _PlaylistCommon(BaseModel):
    n: int = Field(20, ge=1, le=500, description="Number of tracks to return.")
    filter: Optional[FilterBody] = Field(
        None,
        description="Hard constraints on the candidate pool.",
    )


class SimilarPlaylist(_PlaylistCommon):
    """Mode 1: similarity-anchored. The seeds' embedding centroid is the
    target; results stay close to it."""

    mode: Literal["similar"]
    seeds: list[int] = Field(..., min_length=1, description="Seed track IDs.")
    smooth_transitions: _SmoothTransitions = Field(
        default_factory=_SmoothTransitions,
        description="Optional consecutive-pair smoothness rules.",
    )
    include_seeds: bool = Field(
        False, description="Include the seed tracks in the result.",
    )


class DriftPlaylist(_PlaylistCommon):
    """Mode 2: drifting walk. Take ``chunk_size`` similar to the seed,
    re-anchor on the last pick, repeat."""

    mode: Literal["drift"]
    seeds: list[int] = Field(
        ..., min_length=1, max_length=1,
        description="Single-element list. The drift walk needs one anchor.",
    )
    chunk_size: int = Field(
        5, ge=1, le=100,
        description=(
            "Tracks per anchor. Larger = stays closer to the seed; smaller "
            "= drifts faster."
        ),
    )
    smooth_transitions: _SmoothTransitions = Field(
        default_factory=_SmoothTransitions,
    )
    include_seeds: bool = Field(False)


class VibePlaylist(_PlaylistCommon):
    """Mode 3: descriptor-driven. No seeds; a filter narrows the pool and a
    target ranks within it."""

    mode: Literal["vibe"]
    target: _DescriptorTarget = Field(
        default_factory=_DescriptorTarget,
        description="Soft preferences. Higher closeness ranks higher.",
    )
    shuffle: bool = Field(
        True,
        description="Shuffle the (post-target) pool before truncating to n.",
    )
    rng_seed: Optional[int] = Field(
        None, description="Seed for the shuffle. Null = fresh randomness.",
    )


PlaylistBody = Annotated[
    Union[SimilarPlaylist, DriftPlaylist, VibePlaylist],
    Field(discriminator="mode"),
]


# ---------------------------------------------------------------------------
# Service info, stats, scan resource
# ---------------------------------------------------------------------------


class ServiceInfo(BaseModel):
    """Mostly-static service info. Cache-friendly."""

    version: str
    backend: str
    embedding_dim: int
    libraries: list[str]
    workers: int
    db_path: str
    schema_version: int
    descriptor_version: int


class ServiceStats(BaseModel):
    """Dynamic numbers. Polls happily."""

    tracks: int
    total_duration_sec: float
    db_size_bytes: int
    by_model: dict[str, int]


class ScanState(BaseModel):
    """Scan-resource representation. Same shape for ``GET /scan`` (current
    state) and ``POST /scan`` (immediately after triggering)."""

    state: str = Field(
        ...,
        description="``idle`` | ``scanning`` | ``error``.",
    )
    phase: str = Field(
        "idle",
        description=(
            "Sub-phase visible while ``state == 'scanning'``: "
            "``enumerating`` (walking the filesystem), ``extracting`` "
            "(running TF inference and writing tracks), or ``pruning`` "
            "(removing rows for files that disappeared). ``idle`` when "
            "no scan is running."
        ),
    )
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
