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
    FilterParams,
    MatchOut,
    PlaylistBody,
    PlaylistResult,
    ScanStatusOut,
    ScanTriggerBody,
    ScanTriggerResult,
    ServiceStatus,
    SimilarResult,
    StyleEnumeration,
    StyleList,
    StyleScore,
    Track,
    TrackList,
    TrackLookupBody,
    TrackSummary,
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
        styles=p.styles,
        style_min_probability=p.style_min_probability,
        style_match=p.style_match,
    )


def _styles_to_schema(rows: list[tuple[str, float]]) -> list[StyleScore]:
    return [StyleScore(style=s, probability=p) for s, p in rows]


def _row_to_summary(
    row: dict, styles: Optional[list[tuple[str, float]]] = None
) -> TrackSummary:
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
        bpm=row.get("bpm"),
        key=row.get("key"),
        scale=row.get("scale"),
        danceability=row.get("danceability"),
        loudness_db=row.get("loudness_db"),
        styles=_styles_to_schema(styles or []),
    )


def _enrich_matches(db: Database, matches) -> list[MatchOut]:
    """Bulk-fetch tag + library + style metadata for the matched IDs and
    build the enriched response objects. Two SQL queries regardless of N."""
    if not matches:
        return []
    ids = [m.track_id for m in matches]
    rows = db.get_tracks_by_ids(ids)
    styles_by_id = db.get_styles_by_ids(ids)
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
                styles=_styles_to_schema(styles_by_id.get(m.track_id, [])),
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
    styles: Optional[list[str]] = Query(
        None,
        description=(
            "Filter by Discogs-400 style. Repeat the param to pass several. "
            "Use ``Genre---Style`` for an exact match (e.g. "
            "``Electronic---House``) or just the genre prefix "
            "(e.g. ``Electronic``) to match the whole branch."
        ),
    ),
    style_min_probability: float = Query(
        0.0, ge=0.0, le=1.0,
        description=(
            "Minimum classifier probability the matched style row must have."
        ),
    ),
    style_match: str = Query(
        "any", pattern="^(any|all)$",
        description="``any`` (default) or ``all`` of the requested styles.",
    ),
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
        styles=styles,
        style_min_probability=style_min_probability,
        style_match=style_match,
    )
    rows, total = db.list_tracks(
        filter=f, model=model, limit=limit, offset=offset, order_by=order_by
    )
    styles_by_id = db.get_styles_by_ids([int(r["id"]) for r in rows])
    return TrackList(
        items=[_row_to_summary(r, styles_by_id.get(int(r["id"]))) for r in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@api_router.post("/tracks/lookup", response_model=Track)
def lookup_track(
    body: TrackLookupBody, db: Database = Depends(get_db)
) -> Track:
    """Find a single track by path and/or tags.

    Useful for external clients (e.g. media-server plugins) that want to
    map a track from their own catalog onto a harmonie track without doing
    a filesystem walk. See :class:`TrackLookupBody` for matching strategy.
    """
    if not (body.path or body.artist or body.album or body.title):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "at least one of path, artist, album, title must be provided",
        )
    row = db.find_track(
        path=body.path,
        artist=body.artist,
        album=body.album,
        title=body.title,
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no matching track")
    styles = db.get_track_styles(int(row["id"]))
    return Track(**row, styles=_styles_to_schema(styles))


@api_router.get("/tracks/{track_id}", response_model=Track)
def get_track(track_id: int, db: Database = Depends(get_db)) -> Track:
    row = db.get_track_by_id(track_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"track {track_id} not found")
    styles = db.get_track_styles(track_id)
    return Track(**row, styles=_styles_to_schema(styles))


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
    styles: Optional[list[str]] = Query(None),
    style_min_probability: float = Query(0.0, ge=0.0, le=1.0),
    style_match: str = Query("any", pattern="^(any|all)$"),
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
        styles=styles,
        style_min_probability=style_min_probability,
        style_match=style_match,
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


@api_router.get("/styles", response_model=StyleList)
def list_styles(
    db: Database = Depends(get_db),
    min_probability: float = Query(
        0.0, ge=0.0, le=1.0,
        description=(
            "Only count style rows whose probability is at least this high. "
            "0 (default) returns every style any track was tagged with."
        ),
    ),
) -> StyleList:
    """Enumerate every Discogs-400 style currently present in the database.

    Useful for building a UI of available filters and for sanity-checking
    the classifier's distribution across the library.
    """
    rows = db.list_styles(min_probability=min_probability)
    return StyleList(
        items=[StyleEnumeration(**r) for r in rows],
        total=len(rows),
    )


# ---- playlists -----------------------------------------------------------


@api_router.post("/playlists", response_model=PlaylistResult)
def make_playlist(
    body: PlaylistBody,
    db: Database = Depends(get_db),
    index: EmbeddingIndex = Depends(get_index),
) -> PlaylistResult:
    """Build a playlist. The mode is implicit from the parameters:

    * No seeds → descriptor-driven (filters + targets + shuffle).
    * Seeds + drift=true → drifting walk anchored on the previous selection.
    * Seeds + drift=false → similarity-anchored on the seed centroid, with
      optional smooth-transition rules (bpm_tolerance, key_compatible).
    """
    descriptor_filter = _filter_from_params(body.filter)

    try:
        if not body.seeds:
            # Descriptor-driven mode.
            items = generate_vibe_playlist(
                db,
                VibePlaylistRequest(
                    n=body.n,
                    descriptor_filter=descriptor_filter,
                    target_bpm=body.target_bpm,
                    target_danceability=body.target_danceability,
                    shuffle=body.shuffle,
                    seed=body.rng_seed,
                ),
            )
        elif body.drift:
            if len(body.seeds) != 1:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    "drift mode requires exactly one seed",
                )
            items = generate_chained_playlist(
                db, index,
                ChainedPlaylistRequest(
                    seed_id=body.seeds[0],
                    chunk_size=body.chunk_size,
                    n=body.n,
                    descriptor_filter=descriptor_filter,
                    include_seed=body.include_seeds,
                    bpm_drift=body.bpm_tolerance,
                    harmonic_mix=body.key_compatible,
                ),
            )
        else:
            items = generate_similar_playlist(
                db, index,
                SimilarPlaylistRequest(
                    seed_ids=body.seeds,
                    n=body.n,
                    bpm_drift=body.bpm_tolerance,
                    harmonic_mix=body.key_compatible,
                    descriptor_filter=descriptor_filter,
                    include_seeds=body.include_seeds,
                ),
            )
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
