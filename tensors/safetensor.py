"""Safetensor file reading functions."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

if TYPE_CHECKING:
    from rich.console import Console

# Safetensor format constants
HEADER_SIZE_BYTES = 8  # u64 little-endian
MAX_HEADER_SIZE = 100_000_000  # 100MB sanity check


def read_safetensor_metadata(file_path: Path) -> dict[str, Any]:
    """Read metadata from a safetensor file header."""
    with file_path.open("rb") as f:
        header_size_bytes = f.read(HEADER_SIZE_BYTES)
        if len(header_size_bytes) < HEADER_SIZE_BYTES:
            raise ValueError("Invalid safetensor file: too short")

        header_size = struct.unpack("<Q", header_size_bytes)[0]

        if header_size > MAX_HEADER_SIZE:
            raise ValueError(f"Invalid header size: {header_size}")

        header_bytes = f.read(header_size)
        if len(header_bytes) < header_size:
            raise ValueError("Invalid safetensor file: header truncated")

        header: dict[str, Any] = json.loads(header_bytes.decode("utf-8"))

    # Extract __metadata__ if present
    metadata: dict[str, Any] = header.get("__metadata__", {})

    # Count tensors (keys that aren't __metadata__)
    tensor_count = sum(1 for k in header if k != "__metadata__")

    return {
        "metadata": metadata,
        "tensor_count": tensor_count,
        "header_size": header_size,
    }


def compute_sha256(file_path: Path, console: Console) -> str:
    """Compute SHA256 hash of a file with progress display."""
    file_size = file_path.stat().st_size
    sha256 = hashlib.sha256()
    chunk_size = 1024 * 1024 * 8  # 8MB chunks

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Hashing {file_path.name}...", total=file_size)

        with file_path.open("rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
                progress.update(task, advance=len(chunk))

    return sha256.hexdigest().upper()


def get_base_name(file_path: Path) -> str:
    """Get base filename without .safetensors extension."""
    name = file_path.name
    for ext in (".safetensors", ".sft"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return file_path.stem
