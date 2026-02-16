# TODO: ComfyUI CLI & API Integration

## Phase 1: Core Client (`tensors/comfyui.py`)
- [x] Step 1.1: Create ComfyUI client module with basic query functions
  - `get_system_stats()` - System stats (GPU, RAM)
  - `get_queue_status()` - Queue status
  - `get_loaded_models()` - List loaded checkpoints/loras
  - `get_history()` - View history
  - Also: `clear_queue()`, `get_object_info()`, `get_image()`
- [x] Step 1.2: Add workflow execution with WebSocket progress tracking
  - `queue_prompt()` - Queue a workflow
  - `run_workflow()` - Run workflow with progress callback (uses polling)
- [x] Step 1.3: Add simple text-to-image generation
  - `generate_image()` - Text-to-image with embedded workflow template
  - Include SDXL/Flux-compatible default workflow (DEFAULT_WORKFLOW_TEMPLATE)

## Phase 2: CLI Commands (`tensors/cli.py`)
- [ ] Step 2.1: Add `comfy` subcommand group with status commands
  - `tsr comfy status` - System stats
  - `tsr comfy queue` - Queue status
  - `tsr comfy queue --clear` - Clear queue
  - `tsr comfy models` - List loaded models
  - `tsr comfy history [PROMPT_ID]` - View history
- [ ] Step 2.2: Add generation commands
  - `tsr comfy generate "prompt"` - Simple text-to-image
  - `tsr comfy run workflow.json` - Run arbitrary workflow
  - Rich progress bar for generation

## Phase 3: Server API Routes (`tensors/server/comfyui_api_routes.py`)
- [ ] Step 3.1: Create new router with query endpoints
  - `GET /api/comfyui/status` - System stats
  - `GET /api/comfyui/queue` - Queue status
  - `DELETE /api/comfyui/queue` - Clear queue
  - `GET /api/comfyui/models` - List loaded models
  - `GET /api/comfyui/history` - List history
  - `GET /api/comfyui/history/{prompt_id}` - Get specific result
- [ ] Step 3.2: Add generation endpoints
  - `POST /api/comfyui/generate` - Text-to-image generation
  - `POST /api/comfyui/workflow` - Run arbitrary workflow
- [ ] Step 3.3: Register router in server/__init__.py

## Phase 4: Configuration (`tensors/config.py`)
- [ ] Step 4.1: Add ComfyUI config functions
  - `get_comfyui_url()` - Get ComfyUI backend URL
  - `get_comfyui_defaults()` - Get default generation settings
  - Environment variable: `COMFYUI_URL`
  - Config section: `[comfyui]`

---

## Architecture Reference

```
tensors/
├── comfyui.py              # Core client (NEW) - shared by CLI and server
├── cli.py                  # Add `tsr comfy` subcommands
└── server/
    ├── comfyui_routes.py       # Existing proxy (unchanged)
    └── comfyui_api_routes.py   # New programmatic API routes (NEW)
```

## CLI Commands

```
tsr comfy status              # System stats (GPU, RAM, queue)
tsr comfy queue               # Current queue status
tsr comfy queue --clear       # Clear queue
tsr comfy models              # List loaded checkpoints/loras
tsr comfy history [PROMPT_ID] # View history or specific result
tsr comfy generate "prompt"   # Simple text-to-image
tsr comfy run workflow.json   # Run arbitrary workflow
```

## Generate Options

```
tsr comfy generate "a cat" \
  -n "blurry, bad" \           # negative prompt
  -m "flux1-dev-fp8.safetensors" \  # model
  -W 1024 -H 1024 \            # dimensions
  --steps 20 --cfg 7.0 \       # sampling
  --seed 42 \                  # reproducibility
  -o ./output.png              # output path
```
