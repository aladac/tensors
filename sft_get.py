#!/usr/bin/env python3
"""
sft-get: Read safetensor metadata and fetch CivitAI model information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
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
from rich.table import Table

console = Console()

CIVITAI_API_BASE = "https://civitai.com/api/v1"


def read_safetensor_metadata(file_path: Path) -> dict[str, Any]:
    """Read metadata from a safetensor file header."""
    with file_path.open("rb") as f:
        # First 8 bytes are the header size (little-endian u64)
        header_size_bytes = f.read(8)
        if len(header_size_bytes) < 8:
            raise ValueError("Invalid safetensor file: too short")

        header_size = struct.unpack("<Q", header_size_bytes)[0]

        if header_size > 100_000_000:  # 100MB sanity check
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


def compute_sha256(file_path: Path) -> str:
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


def fetch_civitai_by_hash(sha256_hash: str, api_key: str | None = None) -> dict[str, Any] | None:
    """Fetch model information from CivitAI by SHA256 hash."""
    url = f"{CIVITAI_API_BASE}/model-versions/by-hash/{sha256_hash}"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("[cyan]Fetching from CivitAI...", total=None)

        try:
            response = httpx.get(url, headers=headers, timeout=30.0)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            console.print(f"[red]Request error: {e}[/red]")
            return None


def _display_file_info(file_path: Path, local_metadata: dict[str, Any], sha256_hash: str) -> None:
    """Display file information table."""
    file_table = Table(title="File Information", show_header=True, header_style="bold magenta")
    file_table.add_column("Property", style="cyan")
    file_table.add_column("Value", style="green")

    file_table.add_row("File", str(file_path.name))
    file_table.add_row("Path", str(file_path.parent))
    file_table.add_row("Size", f"{file_path.stat().st_size / (1024**3):.2f} GB")
    file_table.add_row("SHA256", sha256_hash)
    file_table.add_row("Header Size", f"{local_metadata['header_size']:,} bytes")
    file_table.add_row("Tensor Count", str(local_metadata["tensor_count"]))

    console.print()
    console.print(file_table)


def _display_local_metadata(local_metadata: dict[str, Any]) -> None:
    """Display local safetensor metadata table."""
    if local_metadata["metadata"]:
        meta_table = Table(
            title="Safetensor Metadata", show_header=True, header_style="bold magenta"
        )
        meta_table.add_column("Key", style="cyan")
        meta_table.add_column("Value", style="green", max_width=80)

        for key, value in sorted(local_metadata["metadata"].items()):
            display_value = str(value)
            if len(display_value) > 200:
                display_value = display_value[:200] + "..."
            meta_table.add_row(key, display_value)

        console.print()
        console.print(meta_table)
    else:
        console.print()
        console.print("[yellow]No embedded metadata found in safetensor file.[/yellow]")


def _display_civitai_data(civitai_data: dict[str, Any] | None) -> None:
    """Display CivitAI model information table."""
    if not civitai_data:
        console.print()
        console.print("[yellow]Model not found on CivitAI.[/yellow]")
        return

    civit_table = Table(
        title="CivitAI Model Information", show_header=True, header_style="bold magenta"
    )
    civit_table.add_column("Property", style="cyan")
    civit_table.add_column("Value", style="green", max_width=80)

    civit_table.add_row("Model ID", str(civitai_data.get("modelId", "N/A")))
    civit_table.add_row("Version ID", str(civitai_data.get("id", "N/A")))
    civit_table.add_row("Version Name", str(civitai_data.get("name", "N/A")))
    civit_table.add_row("Base Model", str(civitai_data.get("baseModel", "N/A")))
    civit_table.add_row("Created At", str(civitai_data.get("createdAt", "N/A")))

    # Trained words
    trained_words: list[str] = civitai_data.get("trainedWords", [])
    if trained_words:
        civit_table.add_row("Trigger Words", ", ".join(trained_words))

    # Download URL
    download_url = str(civitai_data.get("downloadUrl", "N/A"))
    civit_table.add_row("Download URL", download_url)

    # File info from CivitAI
    files: list[dict[str, Any]] = civitai_data.get("files", [])
    for f in files:
        if f.get("primary"):
            civit_table.add_row("Primary File", str(f.get("name", "N/A")))
            civit_table.add_row(
                "File Size (CivitAI)",
                f"{f.get('sizeKB', 0) / 1024:.2f} MB",
            )
            meta: dict[str, Any] = f.get("metadata", {})
            if meta:
                civit_table.add_row("Format", str(meta.get("format", "N/A")))
                civit_table.add_row("Precision", str(meta.get("fp", "N/A")))
                civit_table.add_row("Size Type", str(meta.get("size", "N/A")))

    console.print()
    console.print(civit_table)

    # Model page link
    model_id = civitai_data.get("modelId")
    if model_id:
        console.print()
        console.print(
            f"[bold blue]View on CivitAI:[/bold blue] https://civitai.com/models/{model_id}"
        )


def display_results(
    file_path: Path,
    local_metadata: dict[str, Any],
    sha256_hash: str,
    civitai_data: dict[str, Any] | None,
) -> None:
    """Display results in rich tables."""
    _display_file_info(file_path, local_metadata, sha256_hash)
    _display_local_metadata(local_metadata)
    _display_civitai_data(civitai_data)


def get_base_name(file_path: Path) -> str:
    """Get base filename without .safetensors extension."""
    name = file_path.name
    for ext in (".safetensors", ".sft"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return file_path.stem


def save_metadata(
    file_path: Path,
    sha256_hash: str,
    local_metadata: dict[str, Any],
    civitai_data: dict[str, Any] | None,
) -> tuple[Path, Path]:
    """Save metadata JSON and SHA256 hash to files alongside the model."""
    base_name = get_base_name(file_path)
    parent = file_path.parent

    # Save JSON metadata
    json_path = parent / f"{base_name}-xm.json"
    output = {
        "file": str(file_path),
        "sha256": sha256_hash,
        "header_size": local_metadata["header_size"],
        "tensor_count": local_metadata["tensor_count"],
        "metadata": local_metadata["metadata"],
        "civitai": civitai_data,
    }
    json_path.write_text(json.dumps(output, indent=2))

    # Save SHA256 hash
    sha_path = parent / f"{base_name}-xm.sha256"
    sha_path.write_text(f"{sha256_hash}  {file_path.name}\n")

    return json_path, sha_path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Read safetensor metadata and fetch CivitAI model information.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the safetensor file",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="CivitAI API key for authenticated requests",
    )
    parser.add_argument(
        "--skip-civitai",
        action="store_true",
        help="Skip CivitAI API lookup",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save metadata JSON and SHA256 hash alongside the model file",
    )

    args = parser.parse_args()

    file_path: Path = args.file.resolve()

    if not file_path.exists():
        console.print(f"[red]Error: File not found: {file_path}[/red]")
        return 1

    if file_path.suffix.lower() not in (".safetensors", ".sft"):
        console.print("[yellow]Warning: File does not have .safetensors extension[/yellow]")

    try:
        # Read local metadata
        console.print(f"[bold]Reading safetensor file:[/bold] {file_path.name}")
        local_metadata = read_safetensor_metadata(file_path)

        # Compute SHA256
        sha256_hash = compute_sha256(file_path)

        # Fetch from CivitAI
        civitai_data = None
        if not args.skip_civitai:
            civitai_data = fetch_civitai_by_hash(sha256_hash, args.api_key)

        if args.json_output:
            output = {
                "file": str(file_path),
                "sha256": sha256_hash,
                "header_size": local_metadata["header_size"],
                "tensor_count": local_metadata["tensor_count"],
                "metadata": local_metadata["metadata"],
                "civitai": civitai_data,
            }
            console.print_json(data=output)
        else:
            display_results(file_path, local_metadata, sha256_hash, civitai_data)

        # Save files if requested
        if args.save:
            json_path, sha_path = save_metadata(
                file_path, sha256_hash, local_metadata, civitai_data
            )
            console.print()
            console.print(f"[green]Saved:[/green] {json_path}")
            console.print(f"[green]Saved:[/green] {sha_path}")

        return 0

    except ValueError as e:
        console.print(f"[red]Error reading safetensor: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
