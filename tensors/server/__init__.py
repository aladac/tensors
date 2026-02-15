"""Tensors server — FastAPI app for gallery and CivitAI management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tensors.server.civitai_routes import create_civitai_router
from tensors.server.db_routes import create_db_router
from tensors.server.download_routes import create_download_router
from tensors.server.gallery_routes import create_gallery_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["app", "create_app"]

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the FastAPI application for gallery and model management."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info("Tensors server starting")
        yield

    app = FastAPI(title="tensors", lifespan=lifespan)

    # Serve Vue UI static files
    static_dir = Path(__file__).parent / "static"
    assets_dir = static_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    async def gallery_ui() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/vite.svg", include_in_schema=False)
    async def vite_icon() -> FileResponse:
        return FileResponse(static_dir / "vite.svg")

    @app.get("/status")
    async def status() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(create_civitai_router())
    app.include_router(create_db_router())
    app.include_router(create_gallery_router())
    app.include_router(create_download_router())
    return app


# Module-level app instance for uvicorn
app = create_app()
