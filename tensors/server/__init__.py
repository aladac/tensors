"""sd-server wrapper — FastAPI app for managing sd-server process."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI

from tensors.server.process import ProcessManager
from tensors.server.routes import create_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = ["ProcessManager", "create_app"]


def create_app() -> FastAPI:
    """Build the FastAPI application with process manager."""
    pm = ProcessManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        pm.stop()

    app = FastAPI(title="sd-server wrapper", lifespan=lifespan)
    app.include_router(create_router(pm))
    app.state.pm = pm
    return app
