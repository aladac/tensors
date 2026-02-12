"""HTTP transport layer wrapping httpx."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class HttpTransport:
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        logger.debug("transport ready: %s", base_url)

    def get(self, path: str) -> Any:
        logger.debug("GET %s", path)
        try:
            r = self._client.get(path)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("GET %s → %d: %s", path, e.response.status_code, e.response.text[:200])
            raise
        except httpx.RequestError as e:
            logger.error("GET %s connection failed: %s", path, e)
            raise
        return r.json()

    def post(self, path: str, json: dict[str, Any]) -> Any:
        logger.debug("POST %s", path)
        try:
            r = self._client.post(path, json=json)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error("POST %s → %d: %s", path, e.response.status_code, e.response.text[:200])
            raise
        except httpx.RequestError as e:
            logger.error("POST %s connection failed: %s", path, e)
            raise
        return r.json()

    def close(self) -> None:
        self._client.close()
        logger.debug("transport closed")
