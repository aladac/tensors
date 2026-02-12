"""Generation parameter dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from tensors.generate.util import to_b64


@dataclass
class Txt2ImgParams:
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: int = -1
    batch_size: int = 1
    sampler_name: str = ""
    scheduler: str = ""
    clip_skip: int = -1
    lora: list[dict[str, Any]] | None = None

    def to_body(self) -> dict[str, Any]:
        body = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }
        if self.sampler_name:
            body["sampler_name"] = self.sampler_name
        if self.scheduler:
            body["scheduler"] = self.scheduler
        if self.clip_skip > 0:
            body["clip_skip"] = self.clip_skip
        if self.lora:
            body["lora"] = self.lora
        return body


@dataclass
class Img2ImgParams:
    prompt: str
    init_image: str | bytes | Path
    negative_prompt: str = ""
    width: int = -1
    height: int = -1
    steps: int = 20
    cfg_scale: float = 7.0
    denoising_strength: float = 0.75
    seed: int = -1
    batch_size: int = 1
    sampler_name: str = ""
    scheduler: str = ""
    clip_skip: int = -1
    mask: str | bytes | Path | None = None
    inpainting_mask_invert: bool = False
    lora: list[dict[str, Any]] | None = None
    extra_images: list[str | bytes | Path] = field(default_factory=list)

    def to_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "denoising_strength": self.denoising_strength,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "init_images": [to_b64(self.init_image)],
        }
        if self.width > 0:
            body["width"] = self.width
        if self.height > 0:
            body["height"] = self.height
        if self.mask is not None:
            body["mask"] = to_b64(self.mask)
        if self.inpainting_mask_invert:
            body["inpainting_mask_invert"] = 1
        if self.sampler_name:
            body["sampler_name"] = self.sampler_name
        if self.scheduler:
            body["scheduler"] = self.scheduler
        if self.clip_skip > 0:
            body["clip_skip"] = self.clip_skip
        if self.lora:
            body["lora"] = self.lora
        if self.extra_images:
            body["extra_images"] = [to_b64(img) for img in self.extra_images]
        return body
