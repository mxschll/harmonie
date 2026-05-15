"""FastAPI app factory + lifespan management."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..analyzer import Analyzer, scheduler_loop
from ..config import Settings, get_settings
from .routes import api_router, public_router

logger = logging.getLogger("harmonie.api")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        analyzer = Analyzer(settings)
        analyzer.start()
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
