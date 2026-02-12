"""Utility functions for image encoding and file I/O."""

import base64
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def to_b64(image: str | bytes | Path) -> str:
    """Convert a file path, raw bytes, or base64 string to base64."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.exists():
            logger.debug("encoding file: %s", path)
            return base64.b64encode(path.read_bytes()).decode()
        return str(image)
    if isinstance(image, bytes):
        return base64.b64encode(image).decode()
    raise TypeError(f"unsupported image type: {type(image)}")


def save_images(
    images: list[bytes],
    output_dir: str = ".",
    prefix: str = "output",
) -> list[Path]:
    """Write raw PNG bytes to numbered files. Returns saved paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, data in enumerate(images):
        path = out / f"{prefix}_{i:04d}.png"
        path.write_bytes(data)
        logger.info("saved: %s", path)
        paths.append(path)
    return paths
