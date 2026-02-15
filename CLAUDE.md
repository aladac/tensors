# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`tensors` is a Python CLI + FastAPI server for:
- Reading safetensor file metadata
- Searching CivitAI and HuggingFace models
- Managing a local model database (SQLModel ORM)
- Image gallery management
- Model downloads with resume support

Built with: Typer, Rich, FastAPI, httpx, SQLModel.

## Deployment

### Production (junkpile server)

| Service | Port | URL | Status |
|---------|------|-----|--------|
| tensors API | 51200 | https://tensors-api.saiden.dev | `systemctl status tensors` |
| ComfyUI | 8188 | internal only | `systemctl status comfyui` |

**Note:** ComfyUI runs separately - no integration with tensors yet.

```bash
# Deploy to junkpile
/deploy  # or: ssh junkpile "cd /opt/tensors && git pull && pip install -e '.[server]' && systemctl restart tensors"

# Check status
curl https://tensors-api.saiden.dev/status
ssh junkpile "systemctl status tensors"
ssh junkpile "journalctl -u tensors -f"  # logs
```

### TypeScript Client

Package: `@saiden/tensors` (GitHub: saiden-dev/tensors-typescript)

```bash
# Regenerate client after API changes
ruby ~/.claude/scripts/commands/tensors/generate_client.rb
```

## Commands

```bash
# Install dependencies
uv sync --group dev

# Run everything (fix, check, test)
just

# Individual tasks
just check          # ruff check + mypy
just test           # pytest with coverage
just fix            # auto-fix lint + format
just types          # mypy only

# Run server locally
uv run tsr serve --port 8000

# Run a single test
uv run pytest tests/test_tensors.py::TestClassName::test_name -v
```

## Architecture

### CLI (`tsr`)

Entry point: `tsr = "tensors:main"` (pyproject.toml)

Commands: `info`, `search`, `get`, `dl`, `config`, `serve`

### Modules

| Module | Purpose |
|--------|---------|
| `cli.py` | Typer CLI commands |
| `api.py` | CivitAI REST API wrapper (httpx) |
| `hf.py` | HuggingFace Hub integration |
| `config.py` | XDG paths, enums, API key resolution |
| `safetensor.py` | Binary header parsing, SHA256 |
| `display.py` | Rich table formatting |
| `models.py` | SQLModel ORM (17 tables) |
| `db.py` | Database operations |

### Server (`/api/*`)

| Endpoint | Description |
|----------|-------------|
| `/api/search` | Unified CivitAI + HuggingFace search |
| `/api/civitai/model/{id}` | CivitAI model info |
| `/api/db/*` | Local database CRUD |
| `/api/images/*` | Gallery management |
| `/api/download/*` | Model downloads |

Auth: `X-API-Key` header (configured via `TENSORS_API_KEY` env var)

### Database

SQLite with SQLModel ORM. Tables:
- Local files: `local_files`, `safetensor_metadata`
- CivitAI cache: `models`, `model_versions`, `creators`, `tags`, `version_files`, `file_hashes`, `trained_words`, `version_images`, `image_generation_params`, `image_resources`
- HuggingFace cache: `hf_models`, `hf_model_tags`, `hf_safetensor_files`

## Code Standards

- Python 3.12+, strict mypy, line length 130
- Ruff with extended rule set (E, W, F, I, B, C4, UP, ARG, SIM, TCH, PTH, PL, RUF)
- PLR0913 (too many arguments) is intentionally ignored for CLI commands
- Tests use respx for HTTP mocking and pytest fixtures from conftest.py

## Testing

```bash
# Full test suite
just test

# Specific test file
uv run pytest tests/test_db.py -v

# Test with coverage
uv run pytest --cov=tensors --cov-report=term-missing

# Test API endpoints locally
uv run tsr serve --port 8000 &
curl http://localhost:8000/status
curl -H "X-API-Key: test" http://localhost:8000/api/db/stats
```

## Release

Tags trigger PyPI publish: `git tag v0.1.x && git push origin v0.1.x`

## ComfyUI Integration

ComfyUI GUI is accessible via tensors reverse proxy with session auth:

**URL:** https://tensors-api.saiden.dev/comfy/login

**Environment Variables:**
```bash
COMFYUI_URL=http://127.0.0.1:8188   # ComfyUI backend (localhost only)
COMFYUI_USER=admin                   # Login username
COMFYUI_PASS=<password>              # Login password
SESSION_SECRET=<random-string>       # Cookie signing secret
```

**Flow:**
1. User visits `/comfy/login` → dark mode login page
2. Auth with static user/pass → session cookie set
3. All `/comfy/*` requests proxied to ComfyUI (HTTP + WebSocket)
4. Full GUI works: queue, previews, node editor, etc.

**Security:**
- ComfyUI listens on localhost only (not exposed)
- tensors handles auth via session cookies
- WebSocket connections also authenticated
