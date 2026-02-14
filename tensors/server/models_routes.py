"""FastAPI route handlers for model management endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel as PydanticBaseModel

from tensors.config import MODELS_DIR

if TYPE_CHECKING:
    from tensors.server.process import ProcessManager

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================


class SwitchModelRequest(PydanticBaseModel):
    """Request body for switching models."""

    model: str  # Path to model file


# =============================================================================
# Helper Functions
# =============================================================================


def scan_models(directory: Path, extensions: tuple[str, ...] = (".safetensors", ".gguf")) -> list[dict[str, Any]]:
    """Scan directory for model files."""
    models: list[dict[str, Any]] = []

    if not directory.exists():
        return models

    for ext in extensions:
        for path in directory.rglob(f"*{ext}"):
            stat = path.stat()
            models.append(
                {
                    "name": path.stem,
                    "path": str(path),
                    "filename": path.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                }
            )

    # Sort by name
    models.sort(key=lambda x: x["name"].lower())
    return models


def scan_loras(directory: Path | None = None) -> list[dict[str, Any]]:
    """Scan for LoRA files."""
    lora_dir = directory or MODELS_DIR / "loras"
    return scan_models(lora_dir, extensions=(".safetensors",))


def scan_checkpoints(directory: Path | None = None) -> list[dict[str, Any]]:
    """Scan for checkpoint files."""
    checkpoint_dir = directory or MODELS_DIR / "checkpoints"
    return scan_models(checkpoint_dir, extensions=(".safetensors", ".gguf"))


# =============================================================================
# Router Factory
# =============================================================================


def create_models_router(pm: ProcessManager) -> APIRouter:
    """Build a router with /api/models/* endpoints."""
    router = APIRouter(prefix="/api/models", tags=["models"])

    @router.get("")
    def list_models() -> dict[str, Any]:
        """List available checkpoint models."""
        checkpoints = scan_checkpoints()
        return {
            "models": checkpoints,
            "total": len(checkpoints),
        }

    @router.get("/active")
    def get_active_model() -> dict[str, Any]:
        """Get information about the currently loaded model."""
        status = pm.status()
        config = pm.config

        if config is None:
            return {
                "loaded": False,
                "model": None,
                "status": status.get("status"),
            }

        return {
            "loaded": status.get("status") == "running",
            "model": config.model,
            "pid": status.get("pid"),
            "port": config.port,
            "status": status.get("status"),
        }

    @router.post("/switch")
    async def switch_model(req: SwitchModelRequest) -> JSONResponse:
        """Switch to a different model (hot reload)."""
        model_path = Path(req.model)

        # Validate model exists
        if not model_path.exists():
            raise HTTPException(status_code=400, detail=f"Model not found: {req.model}")

        # Use existing reload logic
        from tensors.server.models import ServerConfig  # noqa: PLC0415

        new_config = ServerConfig(
            model=req.model,
            port=pm.config.port if pm.config else 1234,
            args=pm.config.args if pm.config else [],
        )

        pm.stop()
        pm.start(new_config)
        ready = await pm.wait_ready()

        if not ready:
            return JSONResponse(
                {"error": "sd-server failed to become ready", "model": req.model},
                status_code=503,
            )

        return JSONResponse(
            {
                "ok": True,
                "model": req.model,
                "pid": pm.proc.pid if pm.proc else None,
            }
        )

    @router.get("/loras")
    def list_loras() -> dict[str, Any]:
        """List available LoRA files."""
        loras = scan_loras()
        return {
            "loras": loras,
            "total": len(loras),
        }

    @router.get("/scan")
    def scan_all_models() -> dict[str, Any]:
        """Scan all model directories."""
        checkpoints = scan_checkpoints()
        loras = scan_loras()

        return {
            "checkpoints": checkpoints,
            "loras": loras,
            "total_checkpoints": len(checkpoints),
            "total_loras": len(loras),
        }

    return router
