"""Authentication for tensors API."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

from tensors.config import get_server_api_key

# API key can be passed via header or query param
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


def verify_api_key(
    header_key: Annotated[str | None, Security(api_key_header)] = None,
    query_key: Annotated[str | None, Security(api_key_query)] = None,
) -> str | None:
    """Verify API key from header or query parameter.

    If no server API key is configured, authentication is disabled.
    If configured, the key must match.
    """
    server_key = get_server_api_key()

    # No auth required if no key configured
    if not server_key:
        return None

    # Check header first, then query
    provided_key = header_key or query_key

    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide via X-API-Key header or api_key query param.",
        )

    if provided_key != server_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return provided_key


# Dependency for protected routes
RequireAuth = Annotated[str | None, Depends(verify_api_key)]
