"""Pydantic request models for the wrapper API."""

from __future__ import annotations

from pydantic import BaseModel

DEFAULT_PORT = 1234


class StartRequest(BaseModel):
    model: str
    port: int = DEFAULT_PORT
    args: list[str] = []


class RestartRequest(BaseModel):
    model: str | None = None
    port: int | None = None
    args: list[str] | None = None
