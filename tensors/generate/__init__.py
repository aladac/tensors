"""sd-server Python client — modular, httpx-based."""

from __future__ import annotations

from typing import Any

from tensors.generate._http import HttpTransport
from tensors.generate.generation import GenerationAPI
from tensors.generate.info import InfoAPI
from tensors.generate.params import Img2ImgParams, Txt2ImgParams
from tensors.generate.util import save_images

__all__ = [
    "Img2ImgParams",
    "SDClient",
    "Txt2ImgParams",
    "save_images",
]


class SDClient:
    """Composite client for sd-server.

    Usage::

        with SDClient() as c:
            c.info.models()
            images = c.generate.txt2img(Txt2ImgParams(prompt="a cat"))
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 1234) -> None:
        self._http = HttpTransport(f"http://{host}:{port}")
        self.info = InfoAPI(self._http)
        self.generate = GenerationAPI(self._http)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> SDClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
