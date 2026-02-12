"""Image generation endpoints."""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.generate._http import HttpTransport
    from tensors.generate.params import Img2ImgParams, Txt2ImgParams

logger = logging.getLogger(__name__)


class GenerationAPI:
    def __init__(self, http: HttpTransport) -> None:
        self._http = http

    def txt2img(self, params: Txt2ImgParams) -> list[bytes]:
        """Generate images from text prompt."""
        logger.info("txt2img: '%s' %dx%d steps=%d", params.prompt[:60], params.width, params.height, params.steps)
        data = self._http.post("/sdapi/v1/txt2img", params.to_body())
        images = [base64.b64decode(img) for img in data["images"]]
        logger.info("txt2img: got %d image(s)", len(images))
        return images

    def img2img(self, params: Img2ImgParams) -> list[bytes]:
        """Generate images from image + text prompt."""
        logger.info("img2img: '%s' strength=%.2f steps=%d", params.prompt[:60], params.denoising_strength, params.steps)
        data = self._http.post("/sdapi/v1/img2img", params.to_body())
        images = [base64.b64decode(img) for img in data["images"]]
        logger.info("img2img: got %d image(s)", len(images))
        return images
