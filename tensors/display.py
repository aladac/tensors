"""Rich table display functions for tsr CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from rich.table import Table

if TYPE_CHECKING:
    from rich.console import Console

# Size formatting constants
KB = 1024
MB_IN_KB = KB * KB
THOUSAND = 1000
MILLION = 1_000_000
MAX_TAGS_DISPLAY = 10


def _format_size(size_kb: float) -> str:
    """Format size in KB to human-readable string."""
    if size_kb < KB:
        return f"{size_kb:.0f} KB"
    if size_kb < MB_IN_KB:
        return f"{size_kb / KB:.1f} MB"
    return f"{size_kb / KB / KB:.2f} GB"


def _format_count(count: int) -> str:
    """Format large numbers with K/M suffix."""
    if count < THOUSAND:
        return str(count)
    if count < MILLION:
        return f"{count / THOUSAND:.1f}K"
    return f"{count / MILLION:.1f}M"


def display_file_info(file_path: Path, local_metadata: dict[str, Any], sha256_hash: str, console: Console) -> None:
    """Display file information table."""
    prop_width = 12

    file_table = Table(title="File Information", show_header=True, header_style="bold magenta", expand=True)
    file_table.add_column("Property", style="cyan", width=prop_width, no_wrap=True)
    file_table.add_column("Value", style="green", no_wrap=True, overflow="ellipsis")

    file_table.add_row("File", str(file_path.name))
    file_table.add_row("Path", str(file_path.parent))
    file_table.add_row("Size", f"{file_path.stat().st_size / (1024**3):.2f} GB")
    file_table.add_row("SHA256", sha256_hash)
    file_table.add_row("Header Size", f"{local_metadata['header_size']:,} bytes")
    file_table.add_row("Tensor Count", str(local_metadata["tensor_count"]))

    console.print()
    console.print(file_table)


def display_local_metadata(local_metadata: dict[str, Any], console: Console, keys_filter: list[str] | None = None) -> None:
    """Display local safetensor metadata table."""
    if not local_metadata["metadata"]:
        console.print()
        console.print("[yellow]No embedded metadata found in safetensor file.[/yellow]")
        return

    metadata = local_metadata["metadata"]

    # If specific keys requested, show them in full
    if keys_filter:
        for key in keys_filter:
            if key in metadata:
                console.print(f"[cyan]{key}[/cyan]: {metadata[key]}")
            else:
                console.print(f"[yellow]{key}: not found[/yellow]")
        return

    # Find the longest key to set column width
    all_keys = list(metadata.keys())
    key_width = max(len(k) for k in all_keys) if all_keys else 20

    # Value width: terminal minus key column and table borders (7 chars)
    terminal_width = console.size.width
    value_width = terminal_width - key_width - 7

    meta_table = Table(
        title="Safetensor Metadata",
        show_header=True,
        header_style="bold magenta",
    )
    meta_table.add_column("Key", style="cyan", width=key_width, no_wrap=True)
    meta_table.add_column("Value", style="green", width=value_width, no_wrap=True, overflow="ellipsis")

    for key, value in sorted(metadata.items()):
        meta_table.add_row(key, str(value))

    console.print()
    console.print(meta_table)


def _build_civitai_table(console: Console) -> tuple[Table, int]:
    """Build CivitAI info table with proper column widths."""
    prop_width = 14
    terminal_width = console.size.width
    overhead = 7
    value_width = max(40, terminal_width - prop_width - overhead)

    table = Table(title="CivitAI Model Information", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=prop_width, no_wrap=True)
    table.add_column("Value", style="green", width=value_width, no_wrap=True, overflow="ellipsis")

    return table, value_width


def display_civitai_data(civitai_data: dict[str, Any] | None, console: Console) -> None:
    """Display CivitAI model information table."""
    if not civitai_data:
        console.print()
        console.print("[yellow]Model not found on CivitAI.[/yellow]")
        return

    civit_table, _ = _build_civitai_table(console)

    civit_table.add_row("Model ID", str(civitai_data.get("modelId", "N/A")))
    civit_table.add_row("Version ID", str(civitai_data.get("id", "N/A")))
    civit_table.add_row("Version Name", str(civitai_data.get("name", "N/A")))
    civit_table.add_row("Base Model", str(civitai_data.get("baseModel", "N/A")))
    civit_table.add_row("Created At", str(civitai_data.get("createdAt", "N/A")))

    trained_words: list[str] = civitai_data.get("trainedWords", [])
    if trained_words:
        civit_table.add_row("Trigger Words", ", ".join(trained_words))

    download_url = str(civitai_data.get("downloadUrl", "N/A"))
    civit_table.add_row("Download URL", download_url)

    files: list[dict[str, Any]] = civitai_data.get("files", [])
    for f in files:
        if f.get("primary"):
            civit_table.add_row("Primary File", str(f.get("name", "N/A")))
            civit_table.add_row("File Size", _format_size(f.get("sizeKB", 0)))
            meta: dict[str, Any] = f.get("metadata", {})
            if meta:
                civit_table.add_row("Format", str(meta.get("format", "N/A")))
                civit_table.add_row("Precision", str(meta.get("fp", "N/A")))
                civit_table.add_row("Size Type", str(meta.get("size", "N/A")))

    console.print()
    console.print(civit_table)

    model_id = civitai_data.get("modelId")
    if model_id:
        console.print()
        console.print(f"[bold blue]View on CivitAI:[/bold blue] https://civitai.com/models/{model_id}")


def _build_model_table(console: Console) -> Table:
    """Build model info table with proper column widths."""
    prop_width = 10
    terminal_width = console.size.width
    overhead = 7
    value_width = max(40, terminal_width - prop_width - overhead)

    table = Table(title="Model Information", show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=prop_width, no_wrap=True)
    table.add_column("Value", style="green", width=value_width, no_wrap=True, overflow="ellipsis")

    return table


def _add_model_basic_info(table: Table, model_data: dict[str, Any]) -> None:
    """Add basic model info rows to table."""
    table.add_row("ID", str(model_data.get("id", "N/A")))
    table.add_row("Name", str(model_data.get("name", "N/A")))
    table.add_row("Type", str(model_data.get("type", "N/A")))
    table.add_row("NSFW", str(model_data.get("nsfw", False)))

    creator = model_data.get("creator", {})
    if creator:
        table.add_row("Creator", str(creator.get("username", "N/A")))

    tags: list[str] = model_data.get("tags", [])
    if tags:
        table.add_row("Tags", ", ".join(tags[:MAX_TAGS_DISPLAY]) + ("..." if len(tags) > MAX_TAGS_DISPLAY else ""))

    stats: dict[str, Any] = model_data.get("stats", {})
    if stats:
        table.add_row("Downloads", f"{stats.get('downloadCount', 0):,}")
        table.add_row("Likes", f"{stats.get('thumbsUpCount', 0):,}")

    mode = model_data.get("mode")
    if mode:
        table.add_row("Status", str(mode))


def _build_versions_table(console: Console) -> Table:
    """Build model versions table with proper column widths."""
    id_width = 7
    base_width = 20
    created_width = 10
    size_width = 8

    terminal_width = console.size.width
    fixed_width = id_width + base_width + created_width + size_width
    overhead = 20
    remaining = max(40, terminal_width - fixed_width - overhead)
    name_width = remaining // 3
    file_width = remaining - name_width

    table = Table(title="Model Versions", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=id_width, no_wrap=True)
    table.add_column("Name", style="green", width=name_width, no_wrap=True, overflow="ellipsis")
    table.add_column("Base Model", style="yellow", width=base_width, no_wrap=True, overflow="ellipsis")
    table.add_column("Created", style="blue", width=created_width, no_wrap=True)
    table.add_column("Filename", style="white", width=file_width, no_wrap=True, overflow="ellipsis")
    table.add_column("Size", justify="right", width=size_width, no_wrap=True)

    return table


def _add_version_rows(table: Table, versions: list[dict[str, Any]]) -> None:
    """Add version rows to versions table."""
    for ver in versions:
        files: list[dict[str, Any]] = ver.get("files", [])
        primary_file = next((f for f in files if f.get("primary")), files[0] if files else None)
        filename = "N/A"
        size = "N/A"
        if primary_file:
            filename = primary_file.get("name", "N/A")
            size = _format_size(primary_file.get("sizeKB", 0))

        created = str(ver.get("createdAt", "N/A"))[:10]
        table.add_row(
            str(ver.get("id", "N/A")),
            str(ver.get("name", "N/A")),
            str(ver.get("baseModel", "N/A")),
            created,
            filename,
            size,
        )


def display_model_info(model_data: dict[str, Any], console: Console) -> None:
    """Display full CivitAI model information."""
    model_table = _build_model_table(console)
    _add_model_basic_info(model_table, model_data)

    console.print()
    console.print(model_table)

    versions: list[dict[str, Any]] = model_data.get("modelVersions", [])
    if versions:
        ver_table = _build_versions_table(console)
        _add_version_rows(ver_table, versions)
        console.print()
        console.print(ver_table)

    model_id = model_data.get("id")
    if model_id:
        console.print()
        console.print(f"[bold blue]View on CivitAI:[/bold blue] https://civitai.com/models/{model_id}")


def _build_search_table(console: Console) -> Table:
    """Build search results table with proper column widths."""
    id_width = 7
    type_width = 16
    base_width = 20
    size_width = 8
    dls_width = 6
    likes_width = 6

    terminal_width = console.size.width
    fixed_width = id_width + type_width + base_width + size_width + dls_width + likes_width
    overhead = 23
    name_width = max(20, terminal_width - fixed_width - overhead)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", justify="right", width=id_width, no_wrap=True)
    table.add_column("Name", style="green", width=name_width, no_wrap=True, overflow="ellipsis")
    table.add_column("Type", style="yellow", width=type_width, no_wrap=True)
    table.add_column("Base", style="blue", width=base_width, no_wrap=True, overflow="ellipsis")
    table.add_column("Size", justify="right", width=size_width, no_wrap=True)
    table.add_column("DLs", justify="right", width=dls_width, no_wrap=True)
    table.add_column("Likes", justify="right", width=likes_width, no_wrap=True)

    return table


def _add_search_rows(table: Table, items: list[dict[str, Any]]) -> None:
    """Add search result rows to table."""
    for model in items:
        model_id = str(model.get("id", ""))
        name = model.get("name", "N/A")
        model_type = model.get("type", "N/A")

        versions = model.get("modelVersions", [])
        base_model = "N/A"
        size = "N/A"
        if versions:
            latest = versions[0]
            base_model = latest.get("baseModel", "N/A")
            files = latest.get("files", [])
            primary = next((f for f in files if f.get("primary")), files[0] if files else None)
            if primary:
                size = _format_size(primary.get("sizeKB", 0))

        stats = model.get("stats", {})
        downloads = _format_count(stats.get("downloadCount", 0))
        likes = _format_count(stats.get("thumbsUpCount", 0))

        table.add_row(model_id, name, model_type, base_model, size, downloads, likes)


def display_search_results(results: dict[str, Any], console: Console) -> None:
    """Display search results in a table."""
    items = results.get("items", [])
    if not items:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = _build_search_table(console)
    _add_search_rows(table, items)

    console.print()
    console.print(table)

    metadata = results.get("metadata", {})
    total = metadata.get("totalItems", len(items))
    console.print(f"\n[dim]Showing {len(items)} of {total:,} results[/dim]")
    console.print("[dim]Use 'tsr get <id>' to view details or 'tsr dl -m <id>' to download[/dim]")
