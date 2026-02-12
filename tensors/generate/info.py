"""Model and server info endpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensors.generate._http import HttpTransport

logger = logging.getLogger(__name__)


class InfoAPI:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    def models(self) -> list[dict[str, Any]]:
        """List loaded models (OpenAI /v1/models)."""
        return self._http.get("/v1/models")["data"]  # type: ignore[no-any-return]

    def sd_models(self) -> list[dict[str, Any]]:
        """Detailed model info (sdapi)."""
        return self._http.get("/sdapi/v1/sd-models")  # type: ignore[no-any-return]

    def options(self) -> dict[str, Any]:
        """Current server options."""
        return self._http.get("/sdapi/v1/options")  # type: ignore[no-any-return]

    def loras(self) -> list[dict[str, Any]]:
        """Available LoRAs from --lora-model-dir."""
        result: list[dict[str, Any]] = self._http.get("/sdapi/v1/loras")
        logger.info("found %d lora(s)", len(result))
        return result

    def samplers(self) -> list[str]:
        """Available sampler names."""
        return [s["name"] for s in self._http.get("/sdapi/v1/samplers")]

    def schedulers(self) -> list[str]:
        """Available scheduler names."""
        return [s["name"] for s in self._http.get("/sdapi/v1/schedulers")]
