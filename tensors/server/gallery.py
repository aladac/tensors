"""Image gallery management for generated images."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tensors.config import GALLERY_DIR

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class GalleryImage:
    """Represents an image in the gallery."""

    id: str
    path: Path
    created_at: float
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def meta_path(self) -> Path:
        """Path to the sidecar metadata JSON file."""
        return self.path.with_suffix(".json")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "path": str(self.path),
            "filename": self.path.name,
            "created_at": self.created_at,
            "width": self.width,
            "height": self.height,
            "has_metadata": self.meta_path.exists(),
        }


class Gallery:
    """Manages the image gallery directory."""

    def __init__(self, gallery_dir: Path | None = None) -> None:
        """Initialize gallery with directory path."""
        self.gallery_dir = gallery_dir or GALLERY_DIR
        self.gallery_dir.mkdir(parents=True, exist_ok=True)

    def list_images(
        self,
        limit: int = 50,
        offset: int = 0,
        newest_first: bool = True,
    ) -> list[GalleryImage]:
        """List images in the gallery, paginated."""
        images: list[GalleryImage] = []

        for path in self.gallery_dir.glob("*.png"):
            img = self._load_image(path)
            if img:
                images.append(img)

        # Sort by creation time
        images.sort(key=lambda x: x.created_at, reverse=newest_first)

        # Apply pagination
        return images[offset : offset + limit]

    def get_image(self, image_id: str) -> GalleryImage | None:
        """Get an image by ID."""
        # ID is the filename stem
        path = self.gallery_dir / f"{image_id}.png"
        if not path.exists():
            return None
        return self._load_image(path)

    def get_metadata(self, image_id: str) -> dict[str, Any] | None:
        """Get metadata for an image."""
        meta_path = self.gallery_dir / f"{image_id}.json"
        if not meta_path.exists():
            return None
        result: dict[str, Any] = json.loads(meta_path.read_text())
        return result

    def update_metadata(self, image_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        """Update metadata for an image (merge with existing)."""
        meta_path = self.gallery_dir / f"{image_id}.json"
        img_path = self.gallery_dir / f"{image_id}.png"

        if not img_path.exists():
            return None

        # Load existing or create new
        metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

        # Merge updates
        metadata.update(updates)
        metadata["updated_at"] = time.time()

        # Save
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata

    def delete_image(self, image_id: str) -> bool:
        """Delete an image and its metadata."""
        img_path = self.gallery_dir / f"{image_id}.png"
        meta_path = self.gallery_dir / f"{image_id}.json"

        if not img_path.exists():
            return False

        img_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        return True

    def save_image(
        self,
        image_data: bytes,
        metadata: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> GalleryImage:
        """Save an image to the gallery with optional metadata."""
        timestamp = int(time.time() * 1000)  # milliseconds
        seed_str = str(seed) if seed is not None else "0"
        image_id = f"{timestamp}_{seed_str}"

        img_path = self.gallery_dir / f"{image_id}.png"
        img_path.write_bytes(image_data)

        # Save metadata if provided
        if metadata:
            meta = metadata.copy()
            meta["created_at"] = time.time()
            meta["seed"] = seed
            meta_path = img_path.with_suffix(".json")
            meta_path.write_text(json.dumps(meta, indent=2))

        return self._load_image(img_path) or GalleryImage(
            id=image_id,
            path=img_path,
            created_at=time.time(),
        )

    def _load_image(self, path: Path) -> GalleryImage | None:
        """Load image info from path."""
        if not path.exists():
            return None

        image_id = path.stem
        stat = path.stat()

        # Try to get dimensions from metadata or PIL
        width: int | None = None
        height: int | None = None
        metadata: dict[str, Any] = {}

        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
                width = metadata.get("width")
                height = metadata.get("height")
            except json.JSONDecodeError:
                pass

        return GalleryImage(
            id=image_id,
            path=path,
            created_at=stat.st_mtime,
            width=width,
            height=height,
            metadata=metadata,
        )

    def count(self) -> int:
        """Count total images in gallery."""
        return len(list(self.gallery_dir.glob("*.png")))
