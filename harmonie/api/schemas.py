"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

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
    """A similarity-search or playlist hit, enriched with tag and
    library-relative metadata."""

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


# ---------------------------------------------------------------------------
# Seed references (used by ``similar`` and ``drift`` playlist modes)
# ---------------------------------------------------------------------------


class SeedRef(BaseModel):
    """Reference to a track by path or tags, resolved server-side using the
    same ladder as ``GET /tracks/resolve``. At least one field must be set.
    """

    path: Optional[str] = Field(
        None,
        description=(
            "Absolute or library-relative path. Tried before tag matching."
        ),
    )
    artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None

    @model_validator(mode="after")
    def _at_least_one_field(self) -> "SeedRef":
        if not (self.path or self.artist or self.album or self.title):
            raise ValueError(
                "seed_refs entries must set at least one of: "
                "path, artist, album, title"
            )
        return self


class UnresolvedSeedRef(BaseModel):
    """One ``seed_refs`` entry that could not be matched to a track."""

    ref: SeedRef
    reason: Literal["no_match"] = Field(
        "no_match",
        description=(
            "Why this ref didn't resolve. ``no_match`` is the only "
            "current value."
        ),
    )


class PlaylistResult(BaseModel):
    items: list[MatchOut]
    unresolved_seed_refs: list[UnresolvedSeedRef] = Field(
        default_factory=list,
        description=(
            "``seed_refs`` entries that didn't match any track. Empty when "
            "the request had no ``seed_refs`` or every ref resolved."
        ),
    )


# ---------------------------------------------------------------------------
# Playlist body — discriminated by ``mode``
# ---------------------------------------------------------------------------


class _SmoothTransitions(BaseModel):
    """Consecutive-pair smoothness rules used by ``similar`` and ``drift``
    modes; ignored by ``vibe`` mode."""

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
    """Soft ranking preferences for ``vibe`` mode."""

    bpm: Optional[float] = Field(None, gt=0)
    danceability: Optional[float] = Field(None, ge=0)


class _PlaylistCommon(BaseModel):
    n: int = Field(20, ge=1, le=500, description="Number of tracks to return.")
    filter: Optional[FilterBody] = Field(
        None,
        description="Hard constraints on the candidate pool.",
    )


class _SeededPlaylist(_PlaylistCommon):
    """Base for modes that anchor on seed tracks (``similar``, ``drift``).

    A request may carry pre-resolved IDs in ``seeds``, inline references in
    ``seed_refs``, or both. The merged set must be non-empty.
    """

    seeds: list[int] = Field(
        default_factory=list,
        description="Pre-resolved seed track IDs.",
    )
    seed_refs: list[SeedRef] = Field(
        default_factory=list,
        description=(
            "Inline path/tag references resolved server-side via the same "
            "ladder as ``GET /tracks/resolve``. Refs that don't match a "
            "track are reported in ``unresolved_seed_refs``."
        ),
    )

    @model_validator(mode="after")
    def _at_least_one_seed(self) -> "_SeededPlaylist":
        if not self.seeds and not self.seed_refs:
            raise ValueError(
                "playlist requires at least one of: seeds, seed_refs"
            )
        return self


class SimilarPlaylist(_SeededPlaylist):
    """Similarity-anchored playlist. Results stay close to the seeds'
    embedding centroid."""

    mode: Literal["similar"]
    smooth_transitions: _SmoothTransitions = Field(
        default_factory=_SmoothTransitions,
        description="Optional consecutive-pair smoothness rules.",
    )
    include_seeds: bool = Field(
        False, description="Include the seed tracks in the result.",
    )


class DriftPlaylist(_SeededPlaylist):
    """Chunked drift playlist. Takes the top ``chunk_size`` tracks similar
    to the current anchor, re-anchors on the last pick, and repeats.

    The starting anchor is the seeds' embedding centroid. Consecutive-pair
    constraints baseline against the first seed in the merged list
    (``seeds`` first, then resolved ``seed_refs``).
    """

    mode: Literal["drift"]
    chunk_size: int = Field(
        5, ge=1, le=100,
        description=(
            "Tracks per anchor. Larger stays closer to the seeds; smaller "
            "drifts faster."
        ),
    )
    smooth_transitions: _SmoothTransitions = Field(
        default_factory=_SmoothTransitions,
    )
    include_seeds: bool = Field(False)


class VibePlaylist(_PlaylistCommon):
    """Descriptor-driven playlist. No seeds; the filter narrows the pool
    and the target ranks within it."""

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


class ServiceStatus(BaseModel):
    """Service overview: configuration plus library counters. Returned by
    ``GET /api/v1/status``. Live scan state lives at ``GET /api/v1/scan``."""

    # Service identity.
    version: str
    backend: str
    embedding_dim: int
    libraries: list[str]
    workers: int
    db_path: str
    schema_version: int
    descriptor_version: int

    # Library counters.
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
            "``enumerating`` (walking the filesystem), ``classifying`` "
            "(stat'ing each file to decide what work it needs), "
            "``extracting`` (running TF inference and writing tracks), "
            "or ``pruning`` (removing rows for files that disappeared). "
            "``idle`` when no scan is running."
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
