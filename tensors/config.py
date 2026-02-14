"""Configuration, constants, and enums for tsr CLI."""

from __future__ import annotations

import os
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

# ============================================================================
# XDG Base Directory Configuration
# ============================================================================

# Config: ~/.config/tensors/config.toml
# Data:   ~/.local/share/tensors/models/, ~/.local/share/tensors/metadata/
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "tensors"
CONFIG_FILE = CONFIG_DIR / "config.toml"

DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "tensors"
MODELS_DIR = DATA_DIR / "models"
METADATA_DIR = DATA_DIR / "metadata"
GALLERY_DIR = DATA_DIR / "gallery"

# Legacy config for migration
LEGACY_RC_FILE = Path.home() / ".sftrc"

# Default download paths by model type
DEFAULT_PATHS: dict[str, Path] = {
    "Checkpoint": MODELS_DIR / "checkpoints",
    "LORA": MODELS_DIR / "loras",
    "LoCon": MODELS_DIR / "loras",
}

CIVITAI_API_BASE = "https://civitai.com/api/v1"
CIVITAI_DOWNLOAD_BASE = "https://civitai.com/api/download/models"


# ============================================================================
# Enums for CLI
# ============================================================================


class ModelType(str, Enum):
    """CivitAI model types."""

    checkpoint = "checkpoint"
    lora = "lora"
    embedding = "embedding"
    vae = "vae"
    controlnet = "controlnet"
    locon = "locon"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        mapping = {
            "checkpoint": "Checkpoint",
            "lora": "LORA",
            "embedding": "TextualInversion",
            "vae": "VAE",
            "controlnet": "Controlnet",
            "locon": "LoCon",
        }
        return mapping[self.value]


class BaseModel(str, Enum):
    """Common base models."""

    sd15 = "sd15"
    sdxl = "sdxl"
    pony = "pony"
    flux = "flux"
    illustrious = "illustrious"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        mapping = {
            "sd15": "SD 1.5",
            "sdxl": "SDXL 1.0",
            "pony": "Pony",
            "flux": "Flux.1 D",
            "illustrious": "Illustrious",
        }
        return mapping[self.value]


class SortOrder(str, Enum):
    """Sort options for search."""

    downloads = "downloads"
    rating = "rating"
    newest = "newest"

    def to_api(self) -> str:
        """Convert to CivitAI API value."""
        mapping = {
            "downloads": "Most Downloaded",
            "rating": "Highest Rated",
            "newest": "Newest",
        }
        return mapping[self.value]


# ============================================================================
# Config Functions
# ============================================================================


def load_config() -> dict[str, Any]:
    """Load configuration from TOML config file."""
    if CONFIG_FILE.exists():
        with CONFIG_FILE.open("rb") as f:
            return tomllib.load(f)
    return {}


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to TOML config file."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for k, v in value.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f"{k} = {v}")
            lines.append("")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {value}")

    CONFIG_FILE.write_text("\n".join(lines) + "\n")


def load_api_key() -> str | None:
    """Load API key from config file or CIVITAI_API_KEY env var."""
    # Check environment variable first
    env_key = os.environ.get("CIVITAI_API_KEY")
    if env_key:
        return env_key

    # Check TOML config file
    config = load_config()
    api_section = config.get("api", {})
    if isinstance(api_section, dict):
        key = api_section.get("civitai_key")
        if key:
            return str(key)

    # Fall back to legacy RC file for migration
    if LEGACY_RC_FILE.exists():
        content = LEGACY_RC_FILE.read_text().strip()
        if content:
            return content
    return None


def get_default_output_path(model_type: str | None) -> Path | None:
    """Get default output path based on model type."""
    if model_type and model_type in DEFAULT_PATHS:
        return DEFAULT_PATHS[model_type]
    return None
