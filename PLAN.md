# Plan: SD Image Generation (CLI + Web Gallery)

Add `tsr gen` CLI command and `tsr gallery` web UI for generating images using diffusers + PyTorch (ROCm) directly from local safetensor checkpoints. Gallery is mobile-first, generation-only — no search, no model library.

## Stack

- **diffusers** — load safetensor checkpoints via `from_single_file()`, LoRA via `load_lora_weights()`, all schedulers built-in
- **torch** (ROCm) — assumed pre-installed with ROCm support (`torch.device("cuda")` works on ROCm via HIP)
- **transformers** — CLIP text encoders (diffusers dependency)
- **accelerate** — device placement
- **safetensors** — already a project dependency

## Phase 1: Generation Engine Module

### Description
New `tensors/generate.py` module — wraps diffusers pipeline for txt2img from local safetensor files. Handles checkpoint loading, LoRA application, scheduler selection, and generation.

### Steps

#### Step 1.1: Create generate.py with pipeline management
- **Objective**: Load safetensor checkpoints into diffusers pipeline, generate images
- **Files**: `tensors/generate.py`
- **Dependencies**: None
- **Implementation**:
  - `ImageGenerator` class
  - `load_checkpoint(path)` — `StableDiffusionPipeline.from_single_file()` or `StableDiffusionXLPipeline.from_single_file()` for SDXL safetensors. Auto-detect SD1.5 vs SDXL from metadata. Move to ROCm device.
  - `load_lora(path, strength)` — `pipe.load_lora_weights()`, support multiple LoRAs with `pipe.fuse_lora(lora_scale=strength)`
  - `set_scheduler(name)` — map name strings to diffusers schedulers: EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, etc.
  - `generate()` — accepts prompt, negative_prompt, steps, cfg_scale, width, height, seed, batch_size. Returns list of PIL Images.
  - Keep pipeline loaded between calls (model stays in VRAM)
  - `unload()` — free VRAM

#### Step 1.2: Add config entries for generation defaults
- **Objective**: Configurable default generation params and model paths in config.toml
- **Files**: `tensors/config.py`
- **Dependencies**: Step 1.1
- **Implementation**:
  - Add `[generate]` section: `models_dir` (default `~/models`), `lora_dir`, `output_dir` (default `~/.local/share/tensors/gallery/`), `default_steps`, `default_cfg`, `default_sampler`, `default_scheduler`, `default_width`, `default_height`
  - Enum or list of available schedulers with friendly names

## Phase 2: CLI Generate Command

### Description
`tsr gen` command that loads a checkpoint, generates images, saves to gallery directory with metadata sidecar JSON.

### Steps

#### Step 2.1: Implement `tsr gen` command
- **Objective**: CLI command with all generation parameters as options
- **Files**: `tensors/cli.py`
- **Dependencies**: Phase 1
- **Implementation**:
  - `tsr gen "prompt text"` — positional prompt argument
  - Options: `--model/-m` (path or name in models_dir), `--negative/-n`, `--steps/-s`, `--cfg/-c`, `--width/-W`, `--height/-H`, `--sampler`, `--scheduler`, `--seed`, `--lora` (repeatable, format `name:strength`), `--batch/-b`, `--output/-o` (override output dir)
  - Rich progress: model loading spinner, then generation progress (diffusers callback for step progress)
  - Save output as `{timestamp}_{seed}.png` in gallery dir
  - Save sidecar `{timestamp}_{seed}.json` with all generation params + model name + time elapsed
  - Display: filename, resolution, seed, time elapsed
  - `--json` flag for machine-readable output

#### Step 2.2: Add `tsr gen-ls` subcommand
- **Objective**: List available models, LoRAs, and schedulers
- **Files**: `tensors/cli.py`
- **Dependencies**: Phase 1
- **Implementation**:
  - Scan models_dir for `.safetensors` files
  - Scan lora_dir for LoRA files
  - List available schedulers
  - Rich table output, `--json` flag

## Phase 3: Web Gallery UI

### Description
`tsr gallery` serves a mobile-first web app for generating images and browsing results. Single HTML file, no build tools. Dark theme.

### Steps

#### Step 3.1: Create gallery API server
- **Objective**: FastAPI app that runs generation and serves the gallery
- **Files**: `tensors/gallery.py`
- **Dependencies**: Phase 1, Phase 2
- **Implementation**:
  - Holds a single `ImageGenerator` instance (lazy-loaded on first generate)
  - `POST /api/generate` — accepts generation params JSON, runs pipeline, saves to gallery dir, returns image URL + metadata. Checkpoint loaded/swapped as needed.
  - `GET /api/images` — list gallery images (paginated, newest first), reads sidecar JSONs for metadata
  - `GET /api/images/{filename}` — serve image file
  - `DELETE /api/images/{filename}` — delete image + sidecar
  - `GET /api/models` — list available checkpoints in models_dir
  - `GET /api/loras` — list available LoRAs
  - `GET /api/schedulers` — list available scheduler names
  - `GET /api/config` — current default generation params
  - `GET /api/status` — is model loaded, which one, VRAM usage
  - Static file serving for the frontend
  - Add `fastapi`, `uvicorn` as optional dependencies (`[project.optional-dependencies] gallery = [...]`)

#### Step 3.2: Build mobile-first gallery frontend
- **Objective**: Single-page responsive UI for generation + browsing
- **Files**: `tensors/static/index.html`
- **Dependencies**: Step 3.1
- **Implementation**:
  - **Generate panel** (top on mobile, sidebar on desktop):
    - Model selector dropdown (populated from `/api/models`)
    - Prompt textarea, negative prompt textarea
    - Collapsible "Advanced" section: steps, cfg, sampler dropdown, scheduler dropdown, width, height, seed, LoRA selector with strength slider
    - Generate button with loading state + step progress
    - Dropdowns populated from API on load
  - **Gallery grid** (below/main area):
    - Masonry or uniform grid of generated images, newest first
    - Tap/click to view full size with metadata overlay (prompt, params, seed)
    - Swipe between images on mobile
    - Delete button on detail view
    - Infinite scroll / load more
  - **Design**:
    - Dark theme
    - CSS grid/flexbox, no framework
    - Touch-friendly (large tap targets, no hover-dependent UI)
    - `<meta name="viewport">` for mobile
  - Single HTML file with inline CSS/JS (no build step)

#### Step 3.3: Add `tsr gallery` CLI command
- **Objective**: Launch the gallery web server from CLI
- **Files**: `tensors/cli.py`
- **Dependencies**: Step 3.1, Step 3.2
- **Implementation**:
  - `tsr gallery` — starts uvicorn on `0.0.0.0:7860`
  - Options: `--port/-p`, `--host`, `--model/-m` (pre-load a checkpoint)
  - Auto-open browser with `--open` flag

## Phase 4: Tests

### Steps

#### Step 4.1: Test generate.py
- **Files**: `tests/test_generate.py`
- **Dependencies**: Phase 1
- **Implementation**:
  - Mock torch/diffusers (don't require GPU in CI)
  - Test scheduler mapping, parameter validation, config loading
  - Test checkpoint type detection (SD1.5 vs SDXL)

#### Step 4.2: Test gallery API
- **Files**: `tests/test_gallery.py`
- **Dependencies**: Phase 3
- **Implementation**:
  - Use FastAPI TestClient
  - Mock ImageGenerator
  - Test image listing, deletion, model/lora listing
