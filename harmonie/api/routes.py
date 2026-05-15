"""FastAPI routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.security import APIKeyHeader

from .. import __version__
from ..analyzer import Analyzer
from ..db import Database, TrackFilter
from ..index import EmbeddingIndex
from ..playlist import (
    ChainedPlaylistRequest,
    SimilarPlaylistRequest,
    VibePlaylistRequest,
    generate_chained_playlist,
    generate_similar_playlist,
    generate_vibe_playlist,
)
from ..similarity import find_similar_to_id
from .schemas import (
    ChainedPlaylistBody,
    FilterParams,
    MatchOut,
    PlaylistResult,
    ScanStatusOut,
    ScanTriggerBody,
    ScanTriggerResult,
    ServiceStatus,
    SimilarPlaylistBody,
    SimilarResult,
    Track,
    TrackList,
    TrackSummary,
    VibePlaylistBody,
)

logger = logging.getLogger("harmonie.api")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_analyzer(request: Request) -> Analyzer:
    analyzer: Optional[Analyzer] = getattr(request.app.state, "analyzer", None)
    if analyzer is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "service initializing")
    return analyzer


def get_db(analyzer: Analyzer = Depends(get_analyzer)) -> Database:
    return analyzer.db


def get_index(analyzer: Analyzer = Depends(get_analyzer)) -> EmbeddingIndex:
    return analyzer.index


def require_api_key(
    request: Request, key: Annotated[Optional[str], Depends(api_key_header)]
) -> None:
    expected = request.app.state.settings.api_key
    if not expected:
        return
    if key != expected:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_from_params(p: Optional[FilterParams]) -> Optional[TrackFilter]:
    if p is None:
        return None
    return TrackFilter(
        bpm_min=p.bpm_min,
        bpm_max=p.bpm_max,
        key=p.key,
        scale=p.scale,
        danceability_min=p.danceability_min,
        danceability_max=p.danceability_max,
        loudness_min=p.loudness_min,
        loudness_max=p.loudness_max,
    )


def _row_to_summary(row: dict) -> TrackSummary:
    return TrackSummary(
        id=row["id"],
        path=row["path"],
        library_root=row.get("library_root"),
        relative_path=row.get("relative_path"),
        duration=row["duration"],
        model=row["model"],
        artist=row.get("artist"),
        album=row.get("album"),
        title=row.get("title"),
        track_number=row.get("track_number"),
        musicbrainz_track_id=row.get("musicbrainz_track_id"),
        bpm=row.get("bpm"),
        key=row.get("key"),
        scale=row.get("scale"),
        danceability=row.get("danceability"),
        loudness_db=row.get("loudness_db"),
    )


def _enrich_matches(db: Database, matches) -> list[MatchOut]:
    """Bulk-fetch tag + library metadata for the matched IDs and build the
    enriched response objects. One SQL query regardless of N."""
    if not matches:
        return []
    ids = [m.track_id for m in matches]
    rows = db.get_tracks_by_ids(ids)
    out: list[MatchOut] = []
    for m in matches:
        row = rows.get(m.track_id) or {}
        out.append(
            MatchOut(
                track_id=m.track_id,
                path=m.path,
                score=m.score,
                library_root=row.get("library_root"),
                relative_path=row.get("relative_path"),
                artist=row.get("artist"),
                album=row.get("album"),
                title=row.get("title"),
                track_number=row.get("track_number"),
                musicbrainz_track_id=row.get("musicbrainz_track_id"),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# Public routes (no auth) — health is always reachable.
public_router = APIRouter()

# Authenticated routes.
api_router = APIRouter(dependencies=[Depends(require_api_key)])


@public_router.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@api_router.get("/status", response_model=ServiceStatus)
def get_status(analyzer: Analyzer = Depends(get_analyzer)) -> ServiceStatus:
    s = analyzer.db.stats()
    settings = analyzer.settings
    return ServiceStatus(
        version=__version__,
        backend=settings.backend,
        embedding_dim=analyzer.embedding_dim,
        libraries=[str(p) for p in settings.libraries],
        workers=settings.worker_count,
        db_path=s["db_path"],
        db_size_bytes=s["db_size_bytes"],
        tracks=s["tracks"],
        total_duration_sec=s["total_duration_sec"],
        by_model=s["by_model"],
        scan=ScanStatusOut(**analyzer.status.snapshot()),
    )


# ---- tracks --------------------------------------------------------------


@api_router.get("/tracks", response_model=TrackList)
def list_tracks(
    db: Database = Depends(get_db),
    bpm_min: Optional[float] = Query(None),
    bpm_max: Optional[float] = Query(None),
    key: Optional[list[str]] = Query(None),
    scale: Optional[str] = Query(None),
    danceability_min: Optional[float] = Query(None),
    danceability_max: Optional[float] = Query(None),
    loudness_min: Optional[float] = Query(None),
    loudness_max: Optional[float] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    order_by: str = Query("id", pattern="^(id|path|bpm|duration|analyzed_at)$"),
    model: Optional[str] = Query(None),
) -> TrackList:
    f = TrackFilter(
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        key=key,
        scale=scale,
        danceability_min=danceability_min,
        danceability_max=danceability_max,
        loudness_min=loudness_min,
        loudness_max=loudness_max,
    )
    rows, total = db.list_tracks(
        filter=f, model=model, limit=limit, offset=offset, order_by=order_by
    )
    return TrackList(
        items=[_row_to_summary(r) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@api_router.get("/tracks/{track_id}", response_model=Track)
def get_track(track_id: int, db: Database = Depends(get_db)) -> Track:
    row = db.get_track_by_id(track_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"track {track_id} not found")
    return Track(**row)


@api_router.get("/tracks/{track_id}/similar", response_model=SimilarResult)
def similar_to(
    track_id: int,
    db: Database = Depends(get_db),
    index: EmbeddingIndex = Depends(get_index),
    n: int = Query(10, ge=1, le=500),
    bpm_min: Optional[float] = Query(None),
    bpm_max: Optional[float] = Query(None),
    key: Optional[list[str]] = Query(None),
    scale: Optional[str] = Query(None),
    danceability_min: Optional[float] = Query(None),
    danceability_max: Optional[float] = Query(None),
    loudness_min: Optional[float] = Query(None),
    loudness_max: Optional[float] = Query(None),
    include_self: bool = Query(False),
) -> SimilarResult:
    f = TrackFilter(
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        key=key,
        scale=scale,
        danceability_min=danceability_min,
        danceability_max=danceability_max,
        loudness_min=loudness_min,
        loudness_max=loudness_max,
    )
    try:
        matches = find_similar_to_id(
            db, index, track_id, n=n, filter=f, include_self=include_self
        )
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"track {track_id} not found")
    return SimilarResult(
        query_id=track_id,
        matches=_enrich_matches(db, matches),
    )


# ---- playlists -----------------------------------------------------------


@api_router.post("/playlists/similar", response_model=PlaylistResult)
def similar_playlist(
    body: SimilarPlaylistBody,
    db: Database = Depends(get_db),
    index: EmbeddingIndex = Depends(get_index),
) -> PlaylistResult:
    req = SimilarPlaylistRequest(
        seed_ids=body.seed_ids,
        n=body.n,
        bpm_drift=body.bpm_drift,
        harmonic_mix=body.harmonic_mix,
        descriptor_filter=_filter_from_params(body.filter),
        include_seeds=body.include_seeds,
    )
    try:
        items = generate_similar_playlist(db, index, req)
    except KeyError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return PlaylistResult(items=_enrich_matches(db, items))


@api_router.post("/playlists/vibe", response_model=PlaylistResult)
def vibe_playlist(
    body: VibePlaylistBody, db: Database = Depends(get_db)
) -> PlaylistResult:
    req = VibePlaylistRequest(
        n=body.n,
        descriptor_filter=_filter_from_params(body.filter),
        target_bpm=body.target_bpm,
        target_danceability=body.target_danceability,
        shuffle=body.shuffle,
        seed=body.seed,
    )
    items = generate_vibe_playlist(db, req)
    return PlaylistResult(items=_enrich_matches(db, items))


@api_router.post("/playlists/chained", response_model=PlaylistResult)
def chained_playlist(
    body: ChainedPlaylistBody,
    db: Database = Depends(get_db),
    index: EmbeddingIndex = Depends(get_index),
) -> PlaylistResult:
    req = ChainedPlaylistRequest(
        seed_id=body.seed_id,
        chunk_size=body.chunk_size,
        n=body.n,
        descriptor_filter=_filter_from_params(body.filter),
        include_seed=body.include_seed,
    )
    try:
        items = generate_chained_playlist(db, index, req)
    except KeyError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return PlaylistResult(items=_enrich_matches(db, items))


# ---- admin ---------------------------------------------------------------


@api_router.post("/scan", response_model=ScanTriggerResult)
async def trigger_scan(
    body: ScanTriggerBody, analyzer: Analyzer = Depends(get_analyzer)
) -> ScanTriggerResult:
    if analyzer.status.state == "scanning":
        return ScanTriggerResult(triggered=False, state="scanning")
    # Run in thread so the API doesn't block the event loop.
    asyncio.create_task(asyncio.to_thread(analyzer.scan, force=body.force))
    # Yield briefly so the task picks up the lock and flips state to "scanning"
    # before we report back.
    await asyncio.sleep(0)
    return ScanTriggerResult(triggered=True, state=analyzer.status.state)


@api_router.get("/scan/status", response_model=ScanStatusOut)
def scan_status(analyzer: Analyzer = Depends(get_analyzer)) -> ScanStatusOut:
    return ScanStatusOut(**analyzer.status.snapshot())
