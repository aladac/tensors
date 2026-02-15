"""ComfyUI integration routes for model management and generation."""

from __future__ import annotations

import base64
import logging
import random
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from tensors.comfy import ComfyClient, get_last_checkpoint, save_last_checkpoint

logger = logging.getLogger(__name__)

COMFY_URL = "http://junkpile:8188"


def get_comfy_client() -> ComfyClient:
    """Get a ComfyUI client instance."""
    return ComfyClient(base_url=COMFY_URL)


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    path: str
    filename: str
    size_mb: float = 0
    modified: int = 0
    category: str = "sd15"


class LoRAInfo(BaseModel):
    """LoRA information."""

    name: str
    path: str
    filename: str
    size_mb: float = 0
    modified: int = 0
    category: str = "sd15"


class SwitchModelRequest(BaseModel):
    """Request to switch models."""

    model: str


class LoraConfig(BaseModel):
    """LoRA configuration for generation."""

    path: str
    multiplier: float = 0.8


class GenerateRequest(BaseModel):
    """Image generation request."""

    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    sampler: str = "euler_ancestral"
    save_to_gallery: bool = True
    lora: LoraConfig | None = None


class GeneratedImage(BaseModel):
    """Generated image info."""

    id: str
    data: str  # base64 encoded
    seed: int


def create_comfy_router() -> APIRouter:  # noqa: PLR0915
    """Create the ComfyUI integration router."""
    router = APIRouter(prefix="/api", tags=["comfy"])

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        """List available checkpoint models."""
        try:
            client = get_comfy_client()
            checkpoints = client.get_checkpoints()

            models = []
            for ckpt in checkpoints:
                # Determine category based on filename
                lower = ckpt.lower()
                category = "large" if "xl" in lower or "pony" in lower else "sd15"

                models.append(
                    ModelInfo(
                        name=ckpt.replace(".safetensors", "").replace(".ckpt", ""),
                        path=ckpt,
                        filename=ckpt,
                        category=category,
                    )
                )

            return {"models": [m.model_dump() for m in models], "total": len(models)}
        except httpx.HTTPError as e:
            logger.exception("Failed to connect to ComfyUI")
            raise HTTPException(status_code=503, detail=f"ComfyUI not available: {e}") from e

    @router.get("/models/active")
    async def get_active_model() -> dict[str, Any]:
        """Get the currently active/selected model."""
        last = get_last_checkpoint()
        return {"loaded": last is not None, "model": last}

    @router.post("/models/switch")
    async def switch_model(request: SwitchModelRequest) -> dict[str, Any]:
        """Switch to a different model (saves preference, actual load happens on generation)."""
        old_model = get_last_checkpoint()
        save_last_checkpoint(request.model)
        return {"ok": True, "old_model": old_model, "new_model": request.model}

    @router.get("/models/status")
    async def get_status() -> dict[str, Any]:
        """Get ComfyUI server status."""
        try:
            resp = httpx.get(f"{COMFY_URL}/system_stats", timeout=5)
            resp.raise_for_status()
            stats = resp.json()

            device = stats.get("devices", [{}])[0]
            return {
                "service": "comfyui",
                "active": True,
                "status": "running",
                "current_model": get_last_checkpoint(),
                "host": "junkpile",
                "port": "8188",
                "version": stats.get("system", {}).get("comfyui_version"),
                "gpu": device.get("name"),
                "vram_total": device.get("vram_total"),
                "vram_free": device.get("vram_free"),
            }
        except httpx.HTTPError:
            return {
                "service": "comfyui",
                "active": False,
                "status": "offline",
                "current_model": None,
                "host": "junkpile",
                "port": "8188",
            }

    @router.get("/models/loras")
    async def list_loras() -> dict[str, Any]:
        """List available LoRAs."""
        try:
            client = get_comfy_client()
            loras = client.get_loras()

            lora_list = []
            for lora in loras:
                lower = lora.lower()
                category = "large" if "xl" in lower or "pony" in lower else "sd15"

                lora_list.append(
                    LoRAInfo(
                        name=lora.replace(".safetensors", ""),
                        path=lora,
                        filename=lora,
                        category=category,
                    )
                )

            return {"loras": [lo.model_dump() for lo in lora_list], "total": len(lora_list)}
        except httpx.HTTPError as e:
            logger.exception("Failed to connect to ComfyUI")
            raise HTTPException(status_code=503, detail=f"ComfyUI not available: {e}") from e

    @router.post("/generate")
    async def generate_image(request: GenerateRequest) -> dict[str, Any]:
        """Generate an image using ComfyUI."""
        try:
            client = get_comfy_client()

            # Use last checkpoint or first available
            checkpoint = get_last_checkpoint()
            if not checkpoint:
                checkpoints = client.get_checkpoints()
                if not checkpoints:
                    raise HTTPException(status_code=400, detail="No checkpoints available")
                checkpoint = checkpoints[0]
                save_last_checkpoint(checkpoint)

            # Generate random seed if not specified
            seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)

            # Build generation params
            gen_kwargs: dict[str, Any] = {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "checkpoint": checkpoint,
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "cfg": request.cfg_scale,
                "seed": seed,
                "sampler": request.sampler,
                "auto_restart": False,  # Don't restart container from web UI
            }

            # Add LoRA if specified
            if request.lora:
                gen_kwargs["lora"] = request.lora.path
                gen_kwargs["lora_strength"] = request.lora.multiplier

            result = client.generate(**gen_kwargs)

            # Get image data
            images = []
            for img_info in result.get("images", []):
                img_data = client.get_image(
                    img_info["filename"],
                    img_info.get("subfolder", ""),
                    img_info.get("type", "output"),
                )
                images.append(
                    GeneratedImage(
                        id=img_info["filename"],
                        data=base64.b64encode(img_data).decode(),
                        seed=result.get("seed", seed),
                    )
                )

            return {"images": [img.model_dump() for img in images]}

        except httpx.HTTPError as e:
            logger.exception("Failed to connect to ComfyUI")
            raise HTTPException(status_code=503, detail=f"ComfyUI not available: {e}") from e
        except TimeoutError as e:
            logger.exception("Generation timed out")
            raise HTTPException(status_code=504, detail=str(e)) from e
        except Exception as e:
            logger.exception("Generation failed")
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router
