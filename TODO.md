# TODO

## SD Image Generation

### Phase 1: Generation Engine
- [ ] Step 1.1: Create `tensors/generate.py` — ImageGenerator class (diffusers pipeline, checkpoint loading, LoRA, schedulers)
- [ ] Step 1.2: Add `[generate]` config section (models_dir, lora_dir, output_dir, defaults)

### Phase 2: CLI Generate Command
- [ ] Step 2.1: `tsr gen` command (prompt, model, negative, steps, cfg, sampler, scheduler, seed, lora, resolution)
- [ ] Step 2.2: `tsr gen-ls` command (list models, LoRAs, schedulers)

### Phase 3: Web Gallery
- [ ] Step 3.1: `tensors/gallery.py` — FastAPI server (generate, images, models, loras, schedulers endpoints)
- [ ] Step 3.2: `tensors/static/index.html` — mobile-first dark gallery UI (generate panel + image grid)
- [ ] Step 3.3: `tsr gallery` CLI command (launch server)

### Phase 4: Tests
- [ ] Step 4.1: `tests/test_generate.py` (mocked diffusers, scheduler mapping, config)
- [ ] Step 4.2: `tests/test_gallery.py` (FastAPI TestClient, mocked generator)

## Web UI (Future)

### Model Library
- [ ] Browse downloaded models in `~/.local/share/tensors/models/`
- [ ] Display model metadata (from `.json` files in metadata dir)
- [ ] Show file info: size, hash, tensor count, base model
- [ ] Display CivitAI info if available (trigger words, ratings, download count)
- [ ] Preview images from CivitAI gallery
- [ ] Filter by type (checkpoint, lora, etc.) and base model
- [ ] Search local models by name

### CivitAI Search
- [ ] Search CivitAI models from the UI
- [ ] Filter by type, base model, sort order
- [ ] View model details and versions
- [ ] One-click download to appropriate directory
- [ ] Show download progress
