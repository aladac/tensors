# Models Database Implementation

SQLite database (`models.db`) for local model metadata storage and CivitAI model information cache.

## Dual Purpose

1. **Metadata Storage** — Track local safetensor files with their SHA256 hashes and link them to CivitAI model info
2. **Information Source** — Cache CivitAI API responses for offline queries, search, and model discovery

## Schema Overview

### Local Files Tracking

```sql
-- Your local safetensor files
local_files (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,      -- Absolute path to file
    sha256 TEXT NOT NULL,                 -- Full SHA256 hash
    header_size INTEGER,                  -- Safetensor header size in bytes
    tensor_count INTEGER,                 -- Number of tensors in file
    civitai_model_id INTEGER,             -- Links to models.civitai_id
    civitai_version_id INTEGER,           -- Links to model_versions.civitai_id
    created_at TEXT,
    updated_at TEXT
)

-- Key-value metadata extracted from safetensor headers
safetensor_metadata (
    id INTEGER PRIMARY KEY,
    local_file_id INTEGER NOT NULL,       -- FK to local_files.id
    key TEXT NOT NULL,
    value TEXT,
    UNIQUE(local_file_id, key)
)
```

### CivitAI Model Cache

```sql
-- Model creators
creators (id, username, image_url)

-- Models from CivitAI
models (
    id INTEGER PRIMARY KEY,
    civitai_id INTEGER UNIQUE NOT NULL,   -- CivitAI model ID
    name TEXT NOT NULL,
    description TEXT,                      -- HTML description
    type TEXT NOT NULL,                    -- Checkpoint, LORA, etc.
    nsfw INTEGER,
    creator_id INTEGER,                    -- FK to creators
    download_count INTEGER,
    thumbs_up_count INTEGER,
    ...
)

-- Model versions (each model has multiple versions)
model_versions (
    id INTEGER PRIMARY KEY,
    civitai_id INTEGER UNIQUE NOT NULL,   -- CivitAI version ID
    model_id INTEGER NOT NULL,            -- FK to models.id
    name TEXT NOT NULL,
    base_model TEXT,                       -- "SD 1.5", "SDXL 1.0", "Pony", etc.
    download_url TEXT,
    version_index INTEGER,                 -- 0 = latest
    ...
)

-- Trigger words for LoRAs
trained_words (version_id, word, position)

-- Downloadable files for each version
version_files (
    civitai_id INTEGER UNIQUE,
    version_id INTEGER,                    -- FK to model_versions
    name TEXT,
    size_kb REAL,
    format TEXT,                           -- safetensors, ckpt, etc.
    fp TEXT,                               -- fp16, fp32, bf16
    is_primary INTEGER,
    download_url TEXT
)

-- File hashes (SHA256, AutoV1, AutoV2, etc.)
file_hashes (file_id, hash_type, hash_value)
```

### Tags and Images

```sql
tags (id, name)
model_tags (model_id, tag_id)

-- Preview images with generation params
version_images (version_id, url, width, height, nsfw_level, ...)
image_generation_params (image_id, key, value)  -- prompt, sampler, cfg, etc.
image_resources (image_id, name, type, hash, weight)  -- LoRAs used in image
```

## Views

```sql
-- Models with their latest version info
v_models_with_latest:
    id, civitai_id, name, type, nsfw, creator, latest_version, base_model, download_count, thumbs_up_count

-- Local files with linked CivitAI info
v_local_files_full:
    file_path, sha256, model_name, model_type, version_name, base_model, creator
```

## Implementation Strategy

### 1. Scan Command (`tsr scan`)

Scan local model directories and populate `local_files`:

```python
def scan_models(directory: Path, db: Connection) -> None:
    """Scan directory for safetensor files and add to database."""
    for path in directory.rglob("*.safetensors"):
        sha256 = compute_sha256(path)
        header = read_safetensor_header(path)

        # Insert or update local_files
        db.execute("""
            INSERT INTO local_files (file_path, sha256, header_size, tensor_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                sha256 = excluded.sha256,
                updated_at = datetime('now')
        """, (str(path), sha256, header['size'], header['tensor_count']))

        # Store metadata
        for key, value in header['metadata'].items():
            db.execute("""
                INSERT INTO safetensor_metadata (local_file_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT DO UPDATE SET value = excluded.value
            """, (file_id, key, json.dumps(value)))
```

### 2. Link Command (`tsr link`)

Match local files to CivitAI by hash lookup:

```python
def link_to_civitai(db: Connection, api_key: str | None) -> None:
    """Link local files to CivitAI models using hash matching."""
    unlinked = db.execute("""
        SELECT id, sha256 FROM local_files
        WHERE civitai_model_id IS NULL
    """).fetchall()

    for file_id, sha256 in unlinked:
        # Check local hash cache first
        version = db.execute("""
            SELECT mv.civitai_id, mv.model_id
            FROM file_hashes fh
            JOIN version_files vf ON fh.file_id = vf.id
            JOIN model_versions mv ON vf.version_id = mv.id
            WHERE fh.hash_value = ? AND fh.hash_type = 'SHA256'
        """, (sha256,)).fetchone()

        if not version:
            # Fall back to API lookup
            data = fetch_civitai_by_hash(sha256, api_key)
            if data:
                store_model_version(db, data)
                version = (data['id'], data['modelId'])

        if version:
            db.execute("""
                UPDATE local_files
                SET civitai_version_id = ?, civitai_model_id = ?
                WHERE id = ?
            """, (version[0], version[1], file_id))
```

### 3. Cache Command (`tsr cache`)

Fetch and store full model details from CivitAI:

```python
def cache_model(model_id: int, db: Connection, api_key: str | None) -> None:
    """Fetch and cache complete model data from CivitAI."""
    data = fetch_civitai_model(model_id, api_key)
    if not data:
        return

    # Upsert creator
    creator = data.get('creator', {})
    if creator:
        db.execute("""
            INSERT INTO creators (username, image_url) VALUES (?, ?)
            ON CONFLICT(username) DO UPDATE SET image_url = excluded.image_url
        """, (creator['username'], creator.get('image')))

    # Upsert model
    db.execute("""
        INSERT INTO models (civitai_id, name, description, type, nsfw, ...)
        VALUES (?, ?, ?, ?, ?, ...)
        ON CONFLICT(civitai_id) DO UPDATE SET ...
    """, ...)

    # Process versions, files, hashes, images, trained words
    for idx, version in enumerate(data.get('modelVersions', [])):
        store_version(db, model_id, version, version_index=idx)
```

### 4. Query Commands

**List local models with CivitAI info:**
```python
def list_local_models(db: Connection) -> list[dict]:
    """List all local files with their linked CivitAI metadata."""
    return db.execute("""
        SELECT * FROM v_local_files_full ORDER BY model_name
    """).fetchall()
```

**Search cached models:**
```python
def search_cached(query: str, model_type: str | None, db: Connection) -> list[dict]:
    """Search cached models without hitting the API."""
    sql = """
        SELECT m.*, mv.base_model, mv.download_url
        FROM models m
        JOIN model_versions mv ON mv.model_id = m.id AND mv.version_index = 0
        WHERE m.name LIKE ?
    """
    params = [f'%{query}%']

    if model_type:
        sql += " AND m.type = ?"
        params.append(model_type)

    return db.execute(sql, params).fetchall()
```

**Find trigger words for a local LoRA:**
```python
def get_trigger_words(file_path: str, db: Connection) -> list[str]:
    """Get trigger words for a local LoRA file."""
    return db.execute("""
        SELECT tw.word
        FROM trained_words tw
        JOIN model_versions mv ON tw.version_id = mv.id
        JOIN local_files lf ON lf.civitai_version_id = mv.civitai_id
        WHERE lf.file_path = ?
        ORDER BY tw.position
    """, (file_path,)).fetchall()
```

## Database Location

Following XDG conventions, the database should live at:

```python
from tensors.config import DATA_DIR

DB_PATH = DATA_DIR / "models.db"  # ~/.local/share/tensors/models.db
```

## CLI Integration

```bash
# Scan models directory
tsr db scan /models/

# Link local files to CivitAI (uses API for unknown hashes)
tsr db link

# Cache a specific model's full data
tsr db cache 999258

# List local models with CivitAI info
tsr db list

# Search cached models (offline)
tsr db search "bimbo" --type lora

# Show trigger words for a LoRA
tsr db triggers /models/loras/70s_VPMS.safetensors

# Show generation params from example images
tsr db prompts 999258
```

## Benefits

1. **Offline First** — Query cached data without API calls
2. **Hash Deduplication** — Detect duplicate files by SHA256
3. **Metadata Enrichment** — Combine safetensor header info with CivitAI metadata
4. **Trigger Word Lookup** — Find correct prompts for LoRAs
5. **Example Prompts** — Extract working prompts from preview images
6. **Version Tracking** — Know which version you have vs. latest available
