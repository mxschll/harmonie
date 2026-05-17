"""FastAPI app factory + lifespan management."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ..analyzer import Analyzer, scheduler_loop
from ..config import Settings, get_settings
from .routes import api_router, public_router

logger = logging.getLogger("harmonie.api")
access_logger = logging.getLogger("harmonie.api.requests")


# Paths logged at DEBUG instead of INFO. Liveness probes hit /health
# constantly.
_QUIET_PATHS = frozenset({"/health"})


async def _log_requests(request: Request, call_next):
    """Log one line per HTTP request through the ``harmonie.api.requests``
    logger.

    Format: ``<client> <method> <path>[?<query>] -> <status> (<ms>ms)``.
    Logs even when the handler raises, with ``status=500`` as the
    fallback.
    """
    start = time.monotonic()
    status: int = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        client = request.client.host if request.client else "-"
        path = request.url.path
        if request.url.query:
            path = f"{path}?{request.url.query}"
        level = (
            logging.DEBUG
            if request.url.path in _QUIET_PATHS
            else logging.INFO
        )
        access_logger.log(
            level, "%s %s %s -> %d (%.1fms)",
            client, request.method, path, status, duration_ms,
        )


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        analyzer = Analyzer(settings)
        # Worker pool is created on demand at the start of the first scan
        # to avoid the multi-second TF model load at app startup.
        app.state.analyzer = analyzer
        app.state.settings = settings

        scheduler_task: Optional[asyncio.Task] = None
        if settings.libraries:
            scheduler_task = asyncio.create_task(
                scheduler_loop(analyzer, settings),
                name="harmonie.scheduler",
            )
        else:
            logger.warning(
                "no libraries configured (HARMONIE_LIBRARIES) — scheduler not started"
            )

        try:
            yield
        finally:
            if scheduler_task is not None:
                scheduler_task.cancel()
                try:
                    await scheduler_task
                except (asyncio.CancelledError, Exception):
                    pass
            analyzer.stop()

    app = FastAPI(
        title="harmonie",
        version="0.1.0",
        description="Audio similarity service.",
        lifespan=lifespan,
    )

    app.middleware("http")(_log_requests)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(public_router)
    app.include_router(api_router, prefix="/api/v1")

    return app
