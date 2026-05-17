"""FastAPI routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.security import APIKeyHeader

from .. import __version__
from ..analyzer import Analyzer
from ..db import Database, TrackFilter
from ..index import EmbeddingIndex
from ..migrations import CURRENT_SCHEMA_VERSION
from ..features import DESCRIPTOR_VERSION
from ..playlist import (
    ChainedPlaylistRequest,
    SimilarPlaylistRequest,
    VibePlaylistRequest,
    generate_chained_playlist,
    generate_similar_playlist,
    generate_vibe_playlist,
)
from ..similarity import find_similar_to_id
from .filters import build_track_filter
from .schemas import (
    DriftPlaylist,
    MatchOut,
    PlaylistBody,
    PlaylistResult,
    ScanState,
    SeedRef,
    ServiceStatus,
    SimilarPlaylist,
    SimilarResult,
    StyleEnumeration,
    StyleList,
    StyleScore,
    Track,
    TrackList,
    TrackSummary,
    UnresolvedSeedRef,
    VibePlaylist,
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
        loudness=row.get("loudness"),
        styles=_styles_to_schema(styles or []),
    )


def _enrich_matches(db: Database, matches) -> list[MatchOut]:
    """Bulk-fetch tag, library, and style metadata for the matched IDs in
    two SQL queries and build the enriched response objects."""
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


def filter_query(
    bpm: Optional[str] = Query(
        None,
        description=(
            "BPM filter. ``120..130`` (closed range), ``120..`` (lower "
            "only), ``..130`` (upper only), or ``128`` (exact)."
        ),
        examples=["120..130"],
    ),
    danceability: Optional[str] = Query(
        None, description="Same range syntax as ``bpm``.",
    ),
    loudness: Optional[str] = Query(
        None,
        description="ReplayGain in dB; same range syntax. e.g. ``..-10``.",
    ),
    key: Optional[list[str]] = Query(
        None, description="Filter by key. Repeat the param for OR.",
    ),
    scale: Optional[str] = Query(
        None, description="``major`` or ``minor``.",
    ),
    style: Optional[list[str]] = Query(
        None,
        description=(
            "Discogs-400 style filter. ``Genre---Style`` matches exactly; "
            "a bare ``Genre`` matches the whole branch. Repeatable."
        ),
    ),
    style_min: float = Query(
        0.0, ge=0.0, le=1.0,
        description="Minimum classifier probability for a style row to count.",
    ),
    style_mode: str = Query(
        "any", pattern="^(any|all)$",
        description="``any`` (default) or ``all`` of the requested styles.",
    ),
) -> TrackFilter:
    """Compose a :class:`TrackFilter` from the eight query-string filter
    parameters shared by ``GET /tracks`` and ``GET /tracks/{id}/similar``.
    """
    try:
        return build_track_filter(
            bpm=bpm,
            danceability=danceability,
            loudness=loudness,
            key=key,
            scale=scale,
            style=style,
            style_min=style_min,
            style_mode=style_mode,
        )
    except ValueError as e:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"invalid range filter: {e}"
        )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# Public: only the liveness probe.
public_router = APIRouter()

# Authenticated.
api_router = APIRouter(dependencies=[Depends(require_api_key)])


# ---- service info / health ----------------------------------------------


@public_router.get("/health")
def health() -> dict:
    """Liveness probe. Always returns ``{"status": "ok"}`` when reachable."""
    return {"status": "ok"}


@api_router.get("/status", response_model=ServiceStatus)
def get_status(analyzer: Analyzer = Depends(get_analyzer)) -> ServiceStatus:
    """Service overview: configuration plus library counters. Live scan
    state lives at ``GET /api/v1/scan``."""
    settings = analyzer.settings
    s = analyzer.db.stats()
    return ServiceStatus(
        version=__version__,
        backend=settings.backend,
        embedding_dim=analyzer.embedding_dim,
        libraries=[str(p) for p in settings.libraries],
        workers=settings.worker_count,
        db_path=s["db_path"],
        schema_version=CURRENT_SCHEMA_VERSION,
        descriptor_version=DESCRIPTOR_VERSION,
        tracks=s["tracks"],
        total_duration_sec=s["total_duration_sec"],
        db_size_bytes=s["db_size_bytes"],
        by_model=s["by_model"],
    )


# ---- tracks --------------------------------------------------------------


@api_router.get("/tracks", response_model=TrackList)
def list_tracks(
    db: Database = Depends(get_db),
    f: TrackFilter = Depends(filter_query),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    order_by: str = Query("id", pattern="^(id|path|bpm|duration|analyzed_at)$"),
    model: Optional[str] = Query(None),
) -> TrackList:
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


@api_router.get("/tracks/resolve", response_model=Track)
def resolve_track(
    db: Database = Depends(get_db),
    path: Optional[str] = Query(
        None,
        description=(
            "Absolute or library-relative path. Tried first; falls through "
            "to tag matching if not found."
        ),
    ),
    artist: Optional[str] = Query(None),
    album: Optional[str] = Query(None),
    title: Optional[str] = Query(None),
) -> Track:
    """Find one track by path and/or tags. Strategies, in order — first
    hit wins:

    1. Exact match on absolute ``path``.
    2. Exact match on ``relative_path`` (for mount-point mismatches).
    3. Case-insensitive match on (artist, album, title).
    4. Case-insensitive match on (title, artist) or (title, album).

    400 if every parameter is missing. 404 if no strategy matches.
    """
    if not (path or artist or album or title):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "at least one of path, artist, album, title must be provided",
        )
    row = db.find_track(path=path, artist=artist, album=album, title=title)
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
    f: TrackFilter = Depends(filter_query),
    limit: int = Query(10, ge=1, le=500),
    include_self: bool = Query(False),
) -> SimilarResult:
    try:
        matches = find_similar_to_id(
            db, index, track_id, n=limit, filter=f, include_self=include_self
        )
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"track {track_id} not found")
    return SimilarResult(
        query_id=track_id,
        matches=_enrich_matches(db, matches),
    )


# ---- styles --------------------------------------------------------------


@api_router.get("/styles", response_model=StyleList)
def list_styles(
    db: Database = Depends(get_db),
    style_min: float = Query(
        0.0, ge=0.0, le=1.0,
        description=(
            "Only count style rows whose probability is at least this high."
        ),
    ),
) -> StyleList:
    """Enumerate every Discogs-400 style currently present in the
    database."""
    rows = db.list_styles(min_probability=style_min)
    return StyleList(
        items=[StyleEnumeration(**r) for r in rows],
        total=len(rows),
    )


# ---- playlists -----------------------------------------------------------


def _resolve_seed_refs(
    db: Database, refs: list[SeedRef]
) -> tuple[list[int], list[UnresolvedSeedRef]]:
    """Resolve each ``SeedRef`` via :meth:`Database.find_track`. Returns
    ``(resolved_ids, unresolved)``; refs that don't match are returned in
    ``unresolved`` rather than raising.
    """
    resolved: list[int] = []
    unresolved: list[UnresolvedSeedRef] = []
    for ref in refs:
        row = db.find_track(
            path=ref.path,
            artist=ref.artist,
            album=ref.album,
            title=ref.title,
        )
        if row is None:
            unresolved.append(UnresolvedSeedRef(ref=ref))
        else:
            resolved.append(int(row["id"]))
    return resolved, unresolved


def _merge_seed_ids(seeds: list[int], resolved: list[int]) -> list[int]:
    """Concatenate ``seeds`` and ``resolved``, preserving order, deduping."""
    out: list[int] = []
    seen: set[int] = set()
    for sid in (*seeds, *resolved):
        if sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out


@api_router.post("/playlists", response_model=PlaylistResult)
def make_playlist(
    body: Annotated[PlaylistBody, Body(...)],
    db: Database = Depends(get_db),
    index: EmbeddingIndex = Depends(get_index),
) -> PlaylistResult:
    """Generate a playlist. The body's ``mode`` field selects the strategy:

    * ``similar``: anchored on the seeds' embedding centroid.
    * ``drift``: walks away from the seeds' centroid in chunks,
      re-anchoring on the most recent pick each chunk.
    * ``vibe``: descriptor-driven; ``filter`` narrows the pool and
      ``target`` ranks within it.

    ``similar`` and ``drift`` accept seeds as resolved IDs in ``seeds``,
    inline references in ``seed_refs``, or both. Unmatched refs are
    returned in ``unresolved_seed_refs``. 400 if every supplied seed fails
    to resolve.
    """
    descriptor_filter = (
        body.filter.to_track_filter() if body.filter is not None else None
    )

    # Vibe mode has no seeds; resolution is a no-op for it.
    unresolved: list[UnresolvedSeedRef] = []
    if isinstance(body, (SimilarPlaylist, DriftPlaylist)):
        resolved_ids, unresolved = _resolve_seed_refs(db, body.seed_refs)
        merged_seed_ids = _merge_seed_ids(body.seeds, resolved_ids)
        if not merged_seed_ids:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "no seeds resolved; check seed_refs against /tracks/resolve",
            )
    else:
        merged_seed_ids = []

    try:
        if isinstance(body, SimilarPlaylist):
            items = generate_similar_playlist(
                db, index,
                SimilarPlaylistRequest(
                    seed_ids=merged_seed_ids,
                    n=body.n,
                    bpm_drift=body.smooth_transitions.bpm_tolerance,
                    harmonic_mix=body.smooth_transitions.key_compatible,
                    descriptor_filter=descriptor_filter,
                    include_seeds=body.include_seeds,
                ),
            )
        elif isinstance(body, DriftPlaylist):
            items = generate_chained_playlist(
                db, index,
                ChainedPlaylistRequest(
                    seed_ids=merged_seed_ids,
                    chunk_size=body.chunk_size,
                    n=body.n,
                    descriptor_filter=descriptor_filter,
                    include_seed=body.include_seeds,
                    bpm_drift=body.smooth_transitions.bpm_tolerance,
                    harmonic_mix=body.smooth_transitions.key_compatible,
                ),
            )
        elif isinstance(body, VibePlaylist):
            items = generate_vibe_playlist(
                db,
                VibePlaylistRequest(
                    n=body.n,
                    descriptor_filter=descriptor_filter,
                    target_bpm=body.target.bpm,
                    target_danceability=body.target.danceability,
                    shuffle=body.shuffle,
                    seed=body.rng_seed,
                ),
            )
        else:  # pragma: no cover - exhaustive
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"unknown playlist mode: {body!r}"
            )
    except KeyError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e))
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    return PlaylistResult(
        items=_enrich_matches(db, items),
        unresolved_seed_refs=unresolved,
    )


# ---- scan resource -------------------------------------------------------


@api_router.post("/scan", response_model=ScanState)
async def trigger_scan(
    analyzer: Analyzer = Depends(get_analyzer),
    force: bool = Query(
        False,
        description=(
            "Re-extract embeddings for every track even if size + mtime "
            "match an existing row."
        ),
    ),
) -> ScanState:
    """Trigger a scan in the background. Returns the current scan state.
    No-op when a scan is already running.
    """
    if analyzer.status.state != "scanning":
        # Run in a thread to keep the event loop free.
        asyncio.create_task(asyncio.to_thread(analyzer.scan, force=force))
        # Yield so the task can acquire the scan lock and update state
        # before we snapshot it.
        await asyncio.sleep(0)
    return ScanState(**analyzer.status.snapshot())


@api_router.get("/scan", response_model=ScanState)
def get_scan(analyzer: Analyzer = Depends(get_analyzer)) -> ScanState:
    """Current scan state and counters."""
    return ScanState(**analyzer.status.snapshot())
