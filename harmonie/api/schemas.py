"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Track resource
# ---------------------------------------------------------------------------


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
    musicbrainz_track_id: Optional[str] = None
    # Musical descriptors.
    bpm: Optional[float] = None
    key: Optional[str] = None
    scale: Optional[str] = None
    danceability: Optional[float] = None
    loudness_db: Optional[float] = None


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
    musicbrainz_track_id: Optional[str] = None


class SimilarResult(BaseModel):
    query_id: int
    matches: list[MatchOut]


class SimilarPlaylistBody(BaseModel):
    seed_ids: list[int] = Field(..., min_length=1)
    n: int = Field(20, ge=1, le=500)
    bpm_drift: Optional[float] = Field(None, ge=0)
    harmonic_mix: bool = False
    include_seeds: bool = False
    filter: Optional[FilterParams] = None


class VibePlaylistBody(BaseModel):
    n: int = Field(20, ge=1, le=500)
    target_bpm: Optional[float] = Field(None, gt=0)
    target_danceability: Optional[float] = Field(None, ge=0)
    shuffle: bool = True
    seed: Optional[int] = None
    filter: Optional[FilterParams] = None


class ChainedPlaylistBody(BaseModel):
    seed_id: int
    chunk_size: int = Field(5, ge=1, le=100)
    n: int = Field(20, ge=1, le=500)
    include_seed: bool = False
    filter: Optional[FilterParams] = None


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
