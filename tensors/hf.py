"""Hugging Face Hub integration for safetensor files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.errors import RepositoryNotFoundError

if TYPE_CHECKING:
    from rich.console import Console

# Shared API instance
_api: HfApi | None = None


def _get_api(token: str | None = None) -> HfApi:
    """Get or create HfApi instance."""
    global _api  # noqa: PLW0603
    if _api is None:
        _api = HfApi(token=token)
    return _api


def search_hf_models(
    query: str | None = None,
    *,
    author: str | None = None,
    tags: list[str] | None = None,
    pipeline_tag: str | None = None,
    sort: str | None = None,
    limit: int = 25,
    token: str | None = None,
    console: Console | None = None,
) -> list[dict[str, Any]]:
    """Search Hugging Face models with safetensor files.

    Args:
        query: Search query string
        author: Filter by author/organization
        tags: Additional tags to filter by
        pipeline_tag: Pipeline type (text-generation, text-to-image, etc.)
        sort: Sort field (downloads, likes, created_at, trending_score)
        limit: Maximum results
        token: HuggingFace API token
        console: Rich console for output

    Returns:
        List of model info dictionaries with safetensor files
    """
    api = _get_api(token)

    # Build filter list - always include safetensors
    filters = ["safetensors"]
    if tags:
        filters.extend(tags)

    try:
        models = api.list_models(
            search=query,
            author=author,
            filter=filters,
            pipeline_tag=pipeline_tag,
            sort=sort or "downloads",
            limit=limit,
            expand=["siblings", "downloads", "likes", "author", "lastModified", "createdAt", "tags"],
        )

        results = []
        for model in models:
            model_dict = model.__dict__.copy()

            # Get safetensor files from siblings
            siblings = getattr(model, "siblings", None) or []
            safetensor_files = [
                {"rfilename": s.rfilename, "size": getattr(s, "size", None)}
                for s in siblings
                if s.rfilename.endswith(".safetensors")
            ]

            if safetensor_files:
                model_dict["_safetensor_files"] = safetensor_files
                results.append(model_dict)

        return results

    except Exception as e:
        if console:
            console.print(f"[red]Error searching models: {e}[/red]")
        return []


def get_hf_model(
    model_id: str,
    token: str | None = None,
    console: Console | None = None,
) -> dict[str, Any] | None:
    """Get detailed model information from Hugging Face.

    Args:
        model_id: Model ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
        token: HuggingFace API token
        console: Rich console for output

    Returns:
        Model info dictionary or None if not found
    """
    api = _get_api(token)

    try:
        model = api.model_info(model_id, files_metadata=True)
        model_dict = model.__dict__.copy()

        # Get safetensor files
        siblings = getattr(model, "siblings", None) or []
        safetensor_files = [
            {"rfilename": s.rfilename, "size": getattr(s, "size", None)} for s in siblings if s.rfilename.endswith(".safetensors")
        ]
        model_dict["_safetensor_files"] = safetensor_files

        return model_dict

    except RepositoryNotFoundError:
        if console:
            console.print(f"[red]Model not found: {model_id}[/red]")
        return None
    except Exception as e:
        if console:
            console.print(f"[red]Error fetching model: {e}[/red]")
        return None


def list_safetensor_files(
    model_id: str,
    token: str | None = None,
    console: Console | None = None,
) -> list[str]:
    """List all safetensor files in a Hugging Face model.

    Args:
        model_id: Model ID
        token: HuggingFace API token
        console: Rich console for output

    Returns:
        List of safetensor filenames
    """
    try:
        files = list_repo_files(model_id, token=token)
        return [f for f in files if f.endswith(".safetensors")]
    except RepositoryNotFoundError:
        if console:
            console.print(f"[red]Model not found: {model_id}[/red]")
        return []
    except Exception as e:
        if console:
            console.print(f"[red]Error listing files: {e}[/red]")
        return []


def download_hf_safetensor(
    model_id: str,
    filename: str,
    output_dir: Path,
    token: str | None = None,
    console: Console | None = None,
    *,
    resume: bool = True,
) -> Path | None:
    """Download a safetensor file from Hugging Face.

    Args:
        model_id: Model ID (e.g., "stabilityai/stable-diffusion-xl-base-1.0")
        filename: File name within the model repo
        output_dir: Directory to save the file
        token: HuggingFace API token
        console: Rich console for progress output
        resume: Whether to resume partial downloads

    Returns:
        Path to downloaded file, or None on failure
    """
    if not filename.endswith(".safetensors"):
        if console:
            console.print("[red]Only .safetensors files are supported[/red]")
        return None

    try:
        # hf_hub_download handles caching and resume automatically
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            local_dir=output_dir,
            token=token,
            force_download=not resume,
        )

        if console:
            console.print(f"[green]Downloaded: {downloaded_path}[/green]")

        return Path(downloaded_path)

    except RepositoryNotFoundError:
        if console:
            console.print(f"[red]Model not found: {model_id}[/red]")
        return None
    except Exception as e:
        if console:
            console.print(f"[red]Download failed: {e}[/red]")
        return None


def download_all_safetensors(
    model_id: str,
    output_dir: Path,
    token: str | None = None,
    console: Console | None = None,
) -> list[Path]:
    """Download all safetensor files from a model.

    Args:
        model_id: Model ID
        output_dir: Directory to save files
        token: HuggingFace API token
        console: Rich console for output

    Returns:
        List of downloaded file paths
    """
    files = list_safetensor_files(model_id, token, console)
    if not files:
        return []

    downloaded = []
    for filename in files:
        if console:
            console.print(f"[dim]Downloading {filename}...[/dim]")

        path = download_hf_safetensor(model_id, filename, output_dir, token, console)
        if path:
            downloaded.append(path)

    return downloaded
