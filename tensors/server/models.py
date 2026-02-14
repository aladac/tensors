"""Pydantic models for the sd-server wrapper API."""

from __future__ import annotations

# Note: ServerConfig and ReloadRequest were removed since we no longer manage
# sd-server processes internally. The wrapper now proxies to an external sd-server.
