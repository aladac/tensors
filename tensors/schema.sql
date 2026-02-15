-- Models Database Schema
-- SQLite database for local model metadata storage and CivitAI model information cache.

-- ============================================================================
-- Core Tables: Local Files
-- ============================================================================

CREATE TABLE IF NOT EXISTS local_files (
    id INTEGER PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    header_size INTEGER,
    tensor_count INTEGER,
    civitai_model_id INTEGER,
    civitai_version_id INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_local_files_sha256 ON local_files(sha256);
CREATE INDEX IF NOT EXISTS idx_local_files_civitai_model ON local_files(civitai_model_id);

CREATE TABLE IF NOT EXISTS safetensor_metadata (
    id INTEGER PRIMARY KEY,
    local_file_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    FOREIGN KEY (local_file_id) REFERENCES local_files(id) ON DELETE CASCADE,
    UNIQUE(local_file_id, key)
);

CREATE INDEX IF NOT EXISTS idx_safetensor_metadata_file ON safetensor_metadata(local_file_id);

-- ============================================================================
-- CivitAI Cache Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS creators (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    image_url TEXT
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    civitai_id INTEGER UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    type TEXT NOT NULL,
    nsfw INTEGER DEFAULT 0,
    poi INTEGER DEFAULT 0,
    minor INTEGER DEFAULT 0,
    sfw_only INTEGER DEFAULT 0,
    nsfw_level INTEGER,
    availability TEXT,
    allow_no_credit INTEGER,
    allow_commercial_use TEXT,
    allow_derivatives INTEGER,
    allow_different_license INTEGER,
    supports_generation INTEGER DEFAULT 0,
    creator_id INTEGER,
    download_count INTEGER DEFAULT 0,
    thumbs_up_count INTEGER DEFAULT 0,
    thumbs_down_count INTEGER DEFAULT 0,
    comment_count INTEGER DEFAULT 0,
    tipped_amount_count INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    FOREIGN KEY (creator_id) REFERENCES creators(id)
);

CREATE INDEX IF NOT EXISTS idx_models_civitai ON models(civitai_id);
CREATE INDEX IF NOT EXISTS idx_models_type ON models(type);
CREATE INDEX IF NOT EXISTS idx_models_name ON models(name);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS model_tags (
    model_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    PRIMARY KEY (model_id, tag_id),
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY,
    civitai_id INTEGER UNIQUE NOT NULL,
    model_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    base_model TEXT,
    base_model_type TEXT,
    nsfw_level INTEGER,
    status TEXT,
    availability TEXT,
    upload_type TEXT,
    usage_control TEXT,
    air TEXT,
    training_status TEXT,
    training_details TEXT,
    early_access_ends_at TEXT,
    download_count INTEGER DEFAULT 0,
    thumbs_up_count INTEGER DEFAULT 0,
    thumbs_down_count INTEGER DEFAULT 0,
    supports_generation INTEGER DEFAULT 0,
    download_url TEXT,
    created_at TEXT,
    published_at TEXT,
    updated_at TEXT,
    version_index INTEGER,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_model_versions_civitai ON model_versions(civitai_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_model ON model_versions(model_id);
CREATE INDEX IF NOT EXISTS idx_model_versions_base ON model_versions(base_model);

CREATE TABLE IF NOT EXISTS trained_words (
    id INTEGER PRIMARY KEY,
    version_id INTEGER NOT NULL,
    word TEXT NOT NULL,
    position INTEGER,
    FOREIGN KEY (version_id) REFERENCES model_versions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_trained_words_version ON trained_words(version_id);

CREATE TABLE IF NOT EXISTS version_files (
    id INTEGER PRIMARY KEY,
    civitai_id INTEGER UNIQUE NOT NULL,
    version_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    size_kb REAL,
    format TEXT,
    size_type TEXT,
    fp TEXT,
    is_primary INTEGER DEFAULT 0,
    pickle_scan_result TEXT,
    pickle_scan_message TEXT,
    virus_scan_result TEXT,
    virus_scan_message TEXT,
    scanned_at TEXT,
    download_url TEXT,
    FOREIGN KEY (version_id) REFERENCES model_versions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_version_files_version ON version_files(version_id);

CREATE TABLE IF NOT EXISTS file_hashes (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    hash_type TEXT NOT NULL,
    hash_value TEXT NOT NULL,
    FOREIGN KEY (file_id) REFERENCES version_files(id) ON DELETE CASCADE,
    UNIQUE(file_id, hash_type)
);

CREATE INDEX IF NOT EXISTS idx_file_hashes_file ON file_hashes(file_id);
CREATE INDEX IF NOT EXISTS idx_file_hashes_value ON file_hashes(hash_value);

CREATE TABLE IF NOT EXISTS version_images (
    id INTEGER PRIMARY KEY,
    civitai_id INTEGER,
    version_id INTEGER NOT NULL,
    url TEXT NOT NULL,
    type TEXT,
    nsfw_level INTEGER,
    width INTEGER,
    height INTEGER,
    hash TEXT,
    has_meta INTEGER DEFAULT 0,
    has_positive_prompt INTEGER DEFAULT 0,
    on_site INTEGER DEFAULT 0,
    minor INTEGER DEFAULT 0,
    poi INTEGER DEFAULT 0,
    availability TEXT,
    remix_of_id INTEGER,
    FOREIGN KEY (version_id) REFERENCES model_versions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_version_images_version ON version_images(version_id);

CREATE TABLE IF NOT EXISTS image_video_metadata (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL UNIQUE,
    duration REAL,
    has_audio INTEGER DEFAULT 0,
    size_bytes INTEGER,
    FOREIGN KEY (image_id) REFERENCES version_images(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS image_generation_params (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    FOREIGN KEY (image_id) REFERENCES version_images(id) ON DELETE CASCADE,
    UNIQUE(image_id, key)
);

CREATE INDEX IF NOT EXISTS idx_image_params_image ON image_generation_params(image_id);

CREATE TABLE IF NOT EXISTS image_resources (
    id INTEGER PRIMARY KEY,
    image_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    type TEXT,
    hash TEXT,
    weight REAL,
    FOREIGN KEY (image_id) REFERENCES version_images(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_resources_image ON image_resources(image_id);

-- ============================================================================
-- HuggingFace Cache Tables
-- ============================================================================

CREATE TABLE IF NOT EXISTS hf_models (
    id INTEGER PRIMARY KEY,
    repo_id TEXT NOT NULL UNIQUE,
    author TEXT,
    model_name TEXT NOT NULL,
    pipeline_tag TEXT,
    library_name TEXT,
    downloads INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    trending_score REAL,
    is_private INTEGER DEFAULT 0,
    is_gated INTEGER DEFAULT 0,
    last_modified TEXT,
    created_at TEXT,
    cached_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_hf_models_repo ON hf_models(repo_id);
CREATE INDEX IF NOT EXISTS idx_hf_models_author ON hf_models(author);
CREATE INDEX IF NOT EXISTS idx_hf_models_downloads ON hf_models(downloads);

CREATE TABLE IF NOT EXISTS hf_model_tags (
    hf_model_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (hf_model_id, tag),
    FOREIGN KEY (hf_model_id) REFERENCES hf_models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_hf_model_tags_model ON hf_model_tags(hf_model_id);

CREATE TABLE IF NOT EXISTS hf_safetensor_files (
    id INTEGER PRIMARY KEY,
    hf_model_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    size_bytes INTEGER,
    FOREIGN KEY (hf_model_id) REFERENCES hf_models(id) ON DELETE CASCADE,
    UNIQUE(hf_model_id, filename)
);

CREATE INDEX IF NOT EXISTS idx_hf_files_model ON hf_safetensor_files(hf_model_id);

-- ============================================================================
-- Views
-- ============================================================================

CREATE VIEW IF NOT EXISTS v_models_with_latest AS
SELECT
    m.id,
    m.civitai_id,
    m.name,
    m.type,
    m.nsfw,
    c.username as creator,
    mv.name as latest_version,
    mv.base_model,
    m.download_count,
    m.thumbs_up_count
FROM models m
LEFT JOIN creators c ON m.creator_id = c.id
LEFT JOIN model_versions mv ON mv.model_id = m.id AND mv.version_index = 0;

CREATE VIEW IF NOT EXISTS v_hf_models AS
SELECT
    hm.id,
    hm.repo_id,
    hm.author,
    hm.model_name,
    hm.pipeline_tag,
    hm.downloads,
    hm.likes,
    hm.is_gated,
    hm.last_modified,
    GROUP_CONCAT(DISTINCT hmt.tag) as tags,
    COUNT(DISTINCT hsf.id) as safetensor_count
FROM hf_models hm
LEFT JOIN hf_model_tags hmt ON hm.id = hmt.hf_model_id
LEFT JOIN hf_safetensor_files hsf ON hm.id = hsf.hf_model_id
GROUP BY hm.id;

CREATE VIEW IF NOT EXISTS v_local_files_full AS
SELECT
    lf.id,
    lf.file_path,
    lf.sha256,
    lf.header_size,
    lf.tensor_count,
    lf.civitai_model_id,
    lf.civitai_version_id,
    m.name as model_name,
    m.type as model_type,
    mv.name as version_name,
    mv.base_model,
    c.username as creator,
    lf.created_at,
    lf.updated_at
FROM local_files lf
LEFT JOIN models m ON lf.civitai_model_id = m.civitai_id
LEFT JOIN model_versions mv ON lf.civitai_version_id = mv.civitai_id
LEFT JOIN creators c ON m.creator_id = c.id;
