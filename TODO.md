# TODO

## Web UI

Add a web interface started from the CLI (`tsr serve` or `tsr ui`) to:

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

### Technical
- [ ] Use FastAPI + htmx or similar lightweight stack
- [ ] SQLite for local model index/cache
- [ ] Watch filesystem for new models
- [ ] Configurable port (`tsr serve --port 8080`)
