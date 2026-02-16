"""ComfyUI API client for programmatic workflow execution."""

from __future__ import annotations

import copy
import json
import os
import random
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from rich.progress import Progress, SpinnerColumn, TextColumn

if TYPE_CHECKING:
    from rich.console import Console

# Default ComfyUI URL (same as comfyui_routes.py)
COMFYUI_DEFAULT_URL = "http://127.0.0.1:8188"

# Progress update throttle interval (seconds)
_PROGRESS_UPDATE_INTERVAL = 0.25


def _get_comfyui_url() -> str:
    """Get ComfyUI URL from environment or default."""
    return os.environ.get("COMFYUI_URL", COMFYUI_DEFAULT_URL)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class GenerationResult:
    """Result from image generation."""

    prompt_id: str
    images: list[Path] = field(default_factory=list)
    node_errors: dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass
class WorkflowResult:
    """Result from workflow execution."""

    prompt_id: str
    outputs: dict[str, Any] = field(default_factory=dict)
    node_errors: dict[str, Any] = field(default_factory=dict)
    success: bool = True


# ============================================================================
# Progress Callback Type
# ============================================================================

# (current_step, total_steps, status_message)
ProgressCallback = Callable[[int, int, str], None]


# ============================================================================
# Basic Query Functions
# ============================================================================


def get_system_stats(url: str | None = None, console: Console | None = None) -> dict[str, Any] | None:
    """Get ComfyUI system stats (GPU, RAM, etc.).

    Args:
        url: ComfyUI base URL (defaults to COMFYUI_URL env var or localhost:8188)
        console: Rich console for progress/error output

    Returns:
        System stats dict or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, Any] | None:
        try:
            response = httpx.get(f"{base_url}/system_stats", timeout=10.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching system stats...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def get_queue_status(url: str | None = None, console: Console | None = None) -> dict[str, Any] | None:
    """Get ComfyUI queue status.

    Args:
        url: ComfyUI base URL
        console: Rich console for output

    Returns:
        Queue status dict with 'queue_running' and 'queue_pending' lists, or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, Any] | None:
        try:
            response = httpx.get(f"{base_url}/queue", timeout=10.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching queue status...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def clear_queue(url: str | None = None, console: Console | None = None) -> bool:
    """Clear the ComfyUI queue.

    Args:
        url: ComfyUI base URL
        console: Rich console for output

    Returns:
        True if successful, False on error
    """
    base_url = url or _get_comfyui_url()

    try:
        # Clear both pending and running
        response = httpx.post(f"{base_url}/queue", json={"clear": True}, timeout=10.0)
        response.raise_for_status()
        if console:
            console.print("[green]Queue cleared[/green]")
        return True
    except httpx.HTTPStatusError as e:
        if console:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
        return False
    except httpx.RequestError as e:
        if console:
            console.print(f"[red]Connection error: {e}[/red]")
        return False


def get_object_info(url: str | None = None, console: Console | None = None) -> dict[str, Any] | None:
    """Get ComfyUI object info (available nodes and their configurations).

    Args:
        url: ComfyUI base URL
        console: Rich console for output

    Returns:
        Object info dict or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, Any] | None:
        try:
            response = httpx.get(f"{base_url}/object_info", timeout=30.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching object info...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def get_loaded_models(url: str | None = None, console: Console | None = None) -> dict[str, list[str]] | None:
    """Get list of loaded/available models (checkpoints, loras, etc.).

    Args:
        url: ComfyUI base URL
        console: Rich console for output

    Returns:
        Dict mapping model type to list of model names, or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, list[str]] | None:
        result: dict[str, list[str]] = {}

        # Model type to node class and input name mapping
        model_types = {
            "checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
            "loras": ("LoraLoader", "lora_name"),
            "vae": ("VAELoader", "vae_name"),
            "clip": ("CLIPLoader", "clip_name"),
            "controlnet": ("ControlNetLoader", "control_net_name"),
            "upscale_models": ("UpscaleModelLoader", "model_name"),
        }

        try:
            response = httpx.get(f"{base_url}/object_info", timeout=30.0)
            response.raise_for_status()
            object_info: dict[str, Any] = response.json()

            for model_type, (node_class, input_name) in model_types.items():
                if node_class in object_info:
                    node_info = object_info[node_class]
                    inputs = node_info.get("input", {}).get("required", {})
                    if input_name in inputs:
                        input_def = inputs[input_name]
                        if isinstance(input_def, list) and len(input_def) > 0 and isinstance(input_def[0], list):
                            result[model_type] = input_def[0]

            return result

        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching loaded models...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


def get_history(
    url: str | None = None,
    prompt_id: str | None = None,
    max_items: int = 100,
    console: Console | None = None,
) -> dict[str, Any] | None:
    """Get ComfyUI history.

    Args:
        url: ComfyUI base URL
        prompt_id: Specific prompt ID to fetch (if None, fetches recent history)
        max_items: Maximum number of history items to return
        console: Rich console for output

    Returns:
        History dict (keyed by prompt_id) or None on error
    """
    base_url = url or _get_comfyui_url()

    def _do_fetch() -> dict[str, Any] | None:
        try:
            endpoint = f"{base_url}/history/{prompt_id}" if prompt_id else f"{base_url}/history?max_items={max_items}"
            response = httpx.get(endpoint, timeout=30.0)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        except httpx.HTTPStatusError as e:
            if console:
                console.print(f"[red]API error: {e.response.status_code}[/red]")
            return None
        except httpx.RequestError as e:
            if console:
                console.print(f"[red]Connection error: {e}[/red]")
            return None

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("[cyan]Fetching history...", total=None)
            return _do_fetch()
    else:
        return _do_fetch()


# ============================================================================
# Workflow Execution
# ============================================================================


def queue_prompt(
    workflow: dict[str, Any],
    url: str | None = None,
    client_id: str | None = None,
    console: Console | None = None,
) -> dict[str, Any] | None:
    """Queue a workflow prompt for execution.

    Args:
        workflow: ComfyUI workflow dict (API format)
        url: ComfyUI base URL
        client_id: Client ID for WebSocket tracking
        console: Rich console for output

    Returns:
        Response dict with 'prompt_id' and 'number', or None on error
    """
    base_url = url or _get_comfyui_url()
    client_id = client_id or str(uuid.uuid4())

    try:
        payload = {"prompt": workflow, "client_id": client_id}
        response = httpx.post(f"{base_url}/prompt", json=payload, timeout=30.0)
        response.raise_for_status()
        result: dict[str, Any] = response.json()

        if "error" in result:
            if console:
                console.print(f"[red]Workflow error: {result['error']}[/red]")
                if "node_errors" in result:
                    for node_id, errors in result["node_errors"].items():
                        console.print(f"  [yellow]Node {node_id}:[/yellow] {errors}")
            return None

        return result
    except httpx.HTTPStatusError as e:
        if console:
            console.print(f"[red]API error: {e.response.status_code}[/red]")
            try:
                error_detail = e.response.json()
                if "error" in error_detail:
                    console.print(f"  [yellow]{error_detail['error']}[/yellow]")
            except Exception:
                pass
        return None
    except httpx.RequestError as e:
        if console:
            console.print(f"[red]Connection error: {e}[/red]")
        return None


def _poll_for_completion(
    prompt_id: str,
    url: str,
    timeout: float = 600.0,
    poll_interval: float = 0.5,
    on_progress: ProgressCallback | None = None,
) -> WorkflowResult:
    """Poll history endpoint for workflow completion.

    Args:
        prompt_id: The prompt ID to track
        url: ComfyUI base URL
        timeout: Maximum wait time in seconds
        poll_interval: Time between polls in seconds
        on_progress: Optional callback for progress updates

    Returns:
        WorkflowResult with outputs or errors
    """
    start_time = time.time()
    last_progress_time = 0.0

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(f"{url}/history/{prompt_id}", timeout=10.0)
            response.raise_for_status()
            history = response.json()

            if prompt_id in history:
                entry = history[prompt_id]
                outputs = entry.get("outputs", {})
                status_info = entry.get("status", {})

                # Check for errors
                if status_info.get("status_str") == "error":
                    return WorkflowResult(
                        prompt_id=prompt_id,
                        outputs=outputs,
                        node_errors=status_info.get("messages", {}),
                        success=False,
                    )

                # Success - return outputs
                return WorkflowResult(
                    prompt_id=prompt_id,
                    outputs=outputs,
                    success=True,
                )

            # Still running - check queue for progress
            if on_progress:
                now = time.time()
                if now - last_progress_time >= _PROGRESS_UPDATE_INTERVAL:
                    queue_response = httpx.get(f"{url}/queue", timeout=5.0)
                    if queue_response.status_code == HTTPStatus.OK:
                        queue_data = queue_response.json()
                        running = queue_data.get("queue_running", [])
                        pending = queue_data.get("queue_pending", [])
                        total = len(running) + len(pending)
                        on_progress(0, total, f"Queued ({len(pending)} pending)")
                    last_progress_time = now

        except httpx.RequestError:
            pass  # Connection error, keep polling

        time.sleep(poll_interval)

    # Timeout
    return WorkflowResult(
        prompt_id=prompt_id,
        node_errors={"timeout": f"Workflow did not complete within {timeout}s"},
        success=False,
    )


def run_workflow(
    workflow: dict[str, Any] | Path,
    url: str | None = None,
    console: Console | None = None,
    on_progress: ProgressCallback | None = None,
    timeout: float = 600.0,
) -> WorkflowResult | None:
    """Run a workflow and wait for completion.

    Args:
        workflow: ComfyUI workflow dict (API format) or path to JSON file
        url: ComfyUI base URL
        console: Rich console for progress output
        on_progress: Optional callback for progress updates
        timeout: Maximum wait time in seconds

    Returns:
        WorkflowResult with outputs, or None if queuing failed
    """
    base_url = url or _get_comfyui_url()

    # Load workflow from file if needed
    workflow_dict: dict[str, Any]
    if isinstance(workflow, Path):
        if not workflow.exists():
            if console:
                console.print(f"[red]Workflow file not found: {workflow}[/red]")
            return None
        workflow_dict = json.loads(workflow.read_text())
    else:
        workflow_dict = workflow

    # Queue the workflow
    if console:
        console.print("[cyan]Queueing workflow...[/cyan]")

    result = queue_prompt(workflow_dict, url=base_url, console=console)
    if not result:
        return None

    prompt_id = result["prompt_id"]
    if console:
        console.print(f"[dim]Prompt ID: {prompt_id}[/dim]")

    # Poll for completion with progress
    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running workflow...", total=None)

            def _console_progress(step: int, total: int, status: str) -> None:
                progress.update(task, description=f"[cyan]{status}[/cyan]")
                if on_progress:
                    on_progress(step, total, status)

            return _poll_for_completion(prompt_id, base_url, timeout, on_progress=_console_progress)
    else:
        return _poll_for_completion(prompt_id, base_url, timeout, on_progress=on_progress)


# ============================================================================
# Simple Text-to-Image Generation
# ============================================================================

# Default SDXL/Flux compatible workflow template
# This is a minimal text-to-image workflow that works with most models
DEFAULT_WORKFLOW_TEMPLATE: dict[str, Any] = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 0,
            "steps": 20,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": ""},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "", "clip": ["4", 1]},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "comfy", "images": ["8", 0]},
    },
}


def _build_workflow(
    prompt: str,
    negative_prompt: str = "",
    model: str | None = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
) -> dict[str, Any]:
    """Build a text-to-image workflow from parameters.

    Args:
        prompt: Positive prompt text
        negative_prompt: Negative prompt text
        model: Checkpoint filename (if None, uses first available)
        width: Image width
        height: Image height
        steps: Number of sampling steps
        cfg: CFG scale
        seed: Random seed (-1 for random)
        sampler: Sampler name
        scheduler: Scheduler name

    Returns:
        ComfyUI workflow dict
    """
    workflow = copy.deepcopy(DEFAULT_WORKFLOW_TEMPLATE)

    # Set seed (random if -1)
    actual_seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)

    # Update KSampler settings
    workflow["3"]["inputs"]["seed"] = actual_seed
    workflow["3"]["inputs"]["steps"] = steps
    workflow["3"]["inputs"]["cfg"] = cfg
    workflow["3"]["inputs"]["sampler_name"] = sampler
    workflow["3"]["inputs"]["scheduler"] = scheduler

    # Set model
    if model:
        workflow["4"]["inputs"]["ckpt_name"] = model

    # Set dimensions
    workflow["5"]["inputs"]["width"] = width
    workflow["5"]["inputs"]["height"] = height

    # Set prompts
    workflow["6"]["inputs"]["text"] = prompt
    workflow["7"]["inputs"]["text"] = negative_prompt

    return workflow


def generate_image(
    prompt: str,
    url: str | None = None,
    negative_prompt: str = "",
    model: str | None = None,
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    console: Console | None = None,
    on_progress: ProgressCallback | None = None,
    timeout: float = 600.0,
) -> GenerationResult | None:
    """Generate an image using a simple text-to-image workflow.

    Args:
        prompt: Positive prompt text
        url: ComfyUI base URL
        negative_prompt: Negative prompt text
        model: Checkpoint filename (if None, must be pre-loaded in ComfyUI)
        width: Image width
        height: Image height
        steps: Number of sampling steps
        cfg: CFG scale
        seed: Random seed (-1 for random)
        sampler: Sampler name (euler, dpm_2, etc.)
        scheduler: Scheduler name (normal, karras, etc.)
        console: Rich console for progress output
        on_progress: Optional callback for progress updates
        timeout: Maximum wait time in seconds

    Returns:
        GenerationResult with image paths, or None if generation failed
    """
    base_url = url or _get_comfyui_url()

    # Get available models if none specified
    if not model:
        models = get_loaded_models(url=base_url)
        if models and models.get("checkpoints"):
            model = models["checkpoints"][0]
            if console:
                console.print(f"[dim]Using model: {model}[/dim]")
        else:
            if console:
                console.print("[red]No checkpoints available. Specify a model with --model[/red]")
            return None

    # Build workflow
    workflow = _build_workflow(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        sampler=sampler,
        scheduler=scheduler,
    )

    # Run workflow
    result = run_workflow(
        workflow=workflow,
        url=base_url,
        console=console,
        on_progress=on_progress,
        timeout=timeout,
    )

    if not result:
        return None

    if not result.success:
        if console:
            console.print("[red]Generation failed[/red]")
            for node_id, errors in result.node_errors.items():
                console.print(f"  [yellow]Node {node_id}:[/yellow] {errors}")
        return GenerationResult(
            prompt_id=result.prompt_id,
            node_errors=result.node_errors,
            success=False,
        )

    # Extract image paths from outputs
    images: list[Path] = []
    for _node_id, output in result.outputs.items():
        if "images" in output:
            for img_info in output["images"]:
                filename = img_info.get("filename", "")
                subfolder = img_info.get("subfolder", "")
                img_type = img_info.get("type", "output")

                # Construct path (ComfyUI default output structure)
                if img_type == "output":
                    img_path = Path(subfolder) / filename if subfolder else Path(filename)
                    images.append(img_path)

    if console and images:
        console.print(f"[green]Generated {len(images)} image(s)[/green]")
        for img in images:
            console.print(f"  [dim]{img}[/dim]")

    return GenerationResult(
        prompt_id=result.prompt_id,
        images=images,
        success=True,
    )


def get_image(
    filename: str,
    url: str | None = None,
    subfolder: str = "",
    folder_type: str = "output",
) -> bytes | None:
    """Download a generated image from ComfyUI.

    Args:
        filename: Image filename
        url: ComfyUI base URL
        subfolder: Subfolder within the output directory
        folder_type: Folder type (output, input, temp)

    Returns:
        Image bytes or None on error
    """
    base_url = url or _get_comfyui_url()

    try:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        response = httpx.get(f"{base_url}/view", params=params, timeout=30.0)
        response.raise_for_status()
        return response.content
    except httpx.RequestError:
        return None
