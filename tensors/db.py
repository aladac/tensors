"""SQLite database for local model metadata and CivitAI cache."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tensors.config import DATA_DIR
from tensors.safetensor import compute_sha256, read_safetensor_metadata

if TYPE_CHECKING:
    from rich.console import Console

# Database location
DB_PATH = DATA_DIR / "models.db"

# Load schema from file
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """SQLite database wrapper for models metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize database connection."""
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def init_schema(self) -> None:
        """Initialize database schema from schema.sql."""
        schema = _SCHEMA_PATH.read_text()
        self.conn.executescript(schema)
        self.conn.commit()

    # =========================================================================
    # Local Files Operations
    # =========================================================================

    def scan_directory(
        self,
        directory: Path,
        console: Console | None = None,
    ) -> list[dict[str, Any]]:
        """Scan directory for safetensor files and add to database.

        Returns list of scanned file info dicts.
        """
        results: list[dict[str, Any]] = []
        safetensor_files = list(directory.rglob("*.safetensors"))

        for path in safetensor_files:
            if console:
                console.print(f"[dim]Scanning {path.name}...[/dim]")

            try:
                sha256 = compute_sha256(path)
                metadata = read_safetensor_metadata(path)

                file_info = self._upsert_local_file(
                    file_path=str(path.resolve()),
                    sha256=sha256,
                    header_size=metadata.get("header_size"),
                    tensor_count=metadata.get("tensor_count"),
                )

                # Store safetensor metadata
                self._store_safetensor_metadata(file_info["id"], metadata.get("metadata", {}))

                results.append(file_info)
                self.conn.commit()

            except Exception as e:
                if console:
                    console.print(f"[red]Error scanning {path.name}: {e}[/red]")

        return results

    def _upsert_local_file(
        self,
        file_path: str,
        sha256: str,
        header_size: int | None = None,
        tensor_count: int | None = None,
    ) -> dict[str, Any]:
        """Insert or update a local file record."""
        cur = self.conn.cursor()

        cur.execute("SELECT id FROM local_files WHERE file_path = ?", (file_path,))
        existing = cur.fetchone()

        if existing:
            cur.execute(
                """
                UPDATE local_files SET sha256 = ?, header_size = ?, tensor_count = ?,
                updated_at = datetime('now') WHERE id = ?
                """,
                (sha256, header_size, tensor_count, existing["id"]),
            )
            file_id = existing["id"]
        else:
            cur.execute(
                """
                INSERT INTO local_files (file_path, sha256, header_size, tensor_count)
                VALUES (?, ?, ?, ?)
                """,
                (file_path, sha256, header_size, tensor_count),
            )
            file_id = cur.lastrowid or 0  # lastrowid is always set after INSERT

        return {"id": file_id, "file_path": file_path, "sha256": sha256}

    def _store_safetensor_metadata(self, local_file_id: int, metadata: dict[str, Any]) -> None:
        """Store safetensor header metadata."""
        cur = self.conn.cursor()
        for key, value in metadata.items():
            str_value = json.dumps(value) if not isinstance(value, str) else value
            cur.execute(
                """
                INSERT INTO safetensor_metadata (local_file_id, key, value)
                VALUES (?, ?, ?)
                ON CONFLICT(local_file_id, key) DO UPDATE SET value = excluded.value
                """,
                (local_file_id, key, str_value),
            )

    def list_local_files(self) -> list[dict[str, Any]]:
        """List all local files with CivitAI info."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM v_local_files_full ORDER BY file_path")
        return [dict(row) for row in cur.fetchall()]

    def get_local_file_by_path(self, file_path: str) -> dict[str, Any] | None:
        """Get local file by path."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM v_local_files_full WHERE file_path = ?", (file_path,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_local_file_by_hash(self, sha256: str) -> dict[str, Any] | None:
        """Get local file by SHA256 hash."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM v_local_files_full WHERE sha256 = ?", (sha256.upper(),))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_unlinked_files(self) -> list[dict[str, Any]]:
        """Get local files not linked to CivitAI."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT id, file_path, sha256 FROM local_files
            WHERE civitai_model_id IS NULL
            """
        )
        return [dict(row) for row in cur.fetchall()]

    def link_file_to_civitai(
        self,
        file_id: int,
        model_id: int,
        version_id: int,
    ) -> None:
        """Link a local file to CivitAI model/version."""
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE local_files
            SET civitai_model_id = ?, civitai_version_id = ?, updated_at = datetime('now')
            WHERE id = ?
            """,
            (model_id, version_id, file_id),
        )
        self.conn.commit()

    # =========================================================================
    # CivitAI Cache Operations
    # =========================================================================

    def get_version_by_hash(self, sha256: str) -> dict[str, Any] | None:
        """Find cached version by file hash."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT mv.civitai_id as version_id, m.civitai_id as model_id,
                   m.name as model_name, mv.name as version_name
            FROM file_hashes fh
            JOIN version_files vf ON fh.file_id = vf.id
            JOIN model_versions mv ON vf.version_id = mv.id
            JOIN models m ON mv.model_id = m.id
            WHERE UPPER(fh.hash_value) = UPPER(?)
            """,
            (sha256,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def cache_model(self, data: dict[str, Any]) -> int:
        """Cache full model data from CivitAI API response.

        Returns the internal model ID.
        """
        cur = self.conn.cursor()

        # Get or create creator
        creator_id = self._get_or_create_creator(data.get("creator"))

        # Check if model exists
        civitai_id = data.get("id")
        cur.execute("SELECT id FROM models WHERE civitai_id = ?", (civitai_id,))
        existing = cur.fetchone()

        stats = data.get("stats", {})

        if existing:
            model_id = int(existing["id"])
            cur.execute(
                """
                UPDATE models SET
                    name = ?, description = ?, type = ?, nsfw = ?,
                    download_count = ?, thumbs_up_count = ?,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (
                    data.get("name"),
                    data.get("description"),
                    data.get("type"),
                    1 if data.get("nsfw") else 0,
                    stats.get("downloadCount", 0),
                    stats.get("thumbsUpCount", 0),
                    model_id,
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO models (
                    civitai_id, name, description, type, nsfw, poi, minor,
                    sfw_only, nsfw_level, availability, allow_no_credit,
                    allow_commercial_use, allow_derivatives, allow_different_license,
                    supports_generation, creator_id, download_count, thumbs_up_count,
                    thumbs_down_count, comment_count, tipped_amount_count,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    civitai_id,
                    data.get("name"),
                    data.get("description"),
                    data.get("type"),
                    1 if data.get("nsfw") else 0,
                    1 if data.get("poi") else 0,
                    1 if data.get("minor") else 0,
                    1 if data.get("sfwOnly") else 0,
                    data.get("nsfwLevel"),
                    data.get("availability"),
                    1 if data.get("allowNoCredit") else 0,
                    str(data.get("allowCommercialUse", "")),
                    1 if data.get("allowDerivatives") else 0,
                    1 if data.get("allowDifferentLicense") else 0,
                    1 if data.get("supportsGeneration") else 0,
                    creator_id,
                    stats.get("downloadCount", 0),
                    stats.get("thumbsUpCount", 0),
                    stats.get("thumbsDownCount", 0),
                    stats.get("commentCount", 0),
                    stats.get("tippedAmountCount", 0),
                    data.get("createdAt"),
                ),
            )
            model_id = cur.lastrowid or 0  # lastrowid is always set after INSERT

        # Cache tags
        for tag_name in data.get("tags", []):
            tag_id = self._get_or_create_tag(tag_name)
            cur.execute("INSERT OR IGNORE INTO model_tags (model_id, tag_id) VALUES (?, ?)", (model_id, tag_id))

        # Cache versions
        for idx, version in enumerate(data.get("modelVersions", [])):
            self._cache_version(model_id, version, idx)

        self.conn.commit()
        return model_id

    def _get_or_create_creator(self, creator_data: dict[str, Any] | None) -> int | None:
        """Get or create a creator record."""
        if not creator_data:
            return None
        username = creator_data.get("username")
        if not username:
            return None

        cur = self.conn.cursor()
        cur.execute("SELECT id FROM creators WHERE username = ?", (username,))
        row = cur.fetchone()
        if row:
            return int(row["id"])

        cur.execute(
            "INSERT INTO creators (username, image_url) VALUES (?, ?)",
            (username, creator_data.get("image")),
        )
        return cur.lastrowid or 0

    def _get_or_create_tag(self, tag_name: str) -> int:
        """Get or create a tag record."""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM tags WHERE name = ?", (tag_name,))
        row = cur.fetchone()
        if row:
            return int(row["id"])

        cur.execute("INSERT INTO tags (name) VALUES (?)", (tag_name,))
        return cur.lastrowid or 0  # lastrowid is always set after INSERT

    def _cache_version(self, model_id: int, version: dict[str, Any], index: int) -> int:
        """Cache a model version."""
        cur = self.conn.cursor()
        civitai_id = version.get("id")

        cur.execute("SELECT id FROM model_versions WHERE civitai_id = ?", (civitai_id,))
        existing = cur.fetchone()

        stats = version.get("stats", {})

        if existing:
            version_id = int(existing["id"])
        else:
            cur.execute(
                """
                INSERT INTO model_versions (
                    civitai_id, model_id, name, description, base_model,
                    base_model_type, nsfw_level, status, availability,
                    download_count, thumbs_up_count, thumbs_down_count,
                    supports_generation, download_url, created_at, published_at,
                    updated_at, version_index
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    civitai_id,
                    model_id,
                    version.get("name"),
                    version.get("description"),
                    version.get("baseModel"),
                    version.get("baseModelType"),
                    version.get("nsfwLevel"),
                    version.get("status"),
                    version.get("availability"),
                    stats.get("downloadCount", 0),
                    stats.get("thumbsUpCount", 0),
                    stats.get("thumbsDownCount", 0),
                    1 if version.get("supportsGeneration") else 0,
                    version.get("downloadUrl"),
                    version.get("createdAt"),
                    version.get("publishedAt"),
                    version.get("updatedAt"),
                    index,
                ),
            )
            version_id = cur.lastrowid or 0  # lastrowid is always set after INSERT

        # Cache trained words
        for pos, word in enumerate(version.get("trainedWords", [])):
            cur.execute(
                "INSERT OR IGNORE INTO trained_words (version_id, word, position) VALUES (?, ?, ?)",
                (version_id, word, pos),
            )

        # Cache files and hashes
        for file_data in version.get("files", []):
            self._cache_file(version_id, file_data)

        # Cache images
        for image_data in version.get("images", []):
            self._cache_image(version_id, image_data)

        return version_id

    def _cache_file(self, version_id: int, file_data: dict[str, Any]) -> int | None:
        """Cache a version file."""
        cur = self.conn.cursor()
        civitai_id = file_data.get("id")
        if not civitai_id:
            return None

        cur.execute("SELECT id FROM version_files WHERE civitai_id = ?", (civitai_id,))
        existing = cur.fetchone()

        if existing:
            return int(existing["id"])

        meta = file_data.get("metadata", {})
        cur.execute(
            """
            INSERT INTO version_files (
                civitai_id, version_id, name, type, size_kb, format,
                size_type, fp, is_primary, pickle_scan_result,
                virus_scan_result, scanned_at, download_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                civitai_id,
                version_id,
                file_data.get("name"),
                file_data.get("type"),
                file_data.get("sizeKB"),
                meta.get("format"),
                meta.get("size"),
                meta.get("fp"),
                1 if file_data.get("primary") else 0,
                file_data.get("pickleScanResult"),
                file_data.get("virusScanResult"),
                file_data.get("scannedAt"),
                file_data.get("downloadUrl"),
            ),
        )
        file_id = cur.lastrowid or 0  # lastrowid is always set after INSERT

        # Cache hashes
        for hash_type, hash_value in file_data.get("hashes", {}).items():
            cur.execute(
                "INSERT OR IGNORE INTO file_hashes (file_id, hash_type, hash_value) VALUES (?, ?, ?)",
                (file_id, hash_type, hash_value),
            )

        return file_id

    def _cache_image(self, version_id: int, image_data: dict[str, Any]) -> int | None:
        """Cache a version image."""
        cur = self.conn.cursor()
        url = image_data.get("url")
        if not url:
            return None

        cur.execute("SELECT id FROM version_images WHERE url = ?", (url,))
        existing = cur.fetchone()

        if existing:
            return int(existing["id"])

        cur.execute(
            """
            INSERT INTO version_images (
                civitai_id, version_id, url, type, nsfw_level, width,
                height, hash, has_meta, has_positive_prompt, on_site,
                minor, poi, availability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_data.get("id"),
                version_id,
                url,
                image_data.get("type"),
                image_data.get("nsfwLevel"),
                image_data.get("width"),
                image_data.get("height"),
                image_data.get("hash"),
                1 if image_data.get("hasMeta") else 0,
                1 if image_data.get("hasPositivePrompt") else 0,
                1 if image_data.get("onSite") else 0,
                1 if image_data.get("minor") else 0,
                1 if image_data.get("poi") else 0,
                image_data.get("availability"),
            ),
        )
        image_id = cur.lastrowid or 0  # lastrowid is always set after INSERT

        # Cache generation params
        meta = image_data.get("meta", {})
        for key, value in meta.items():
            if key == "resources":
                continue
            str_value = str(value) if value is not None else None
            cur.execute(
                "INSERT OR IGNORE INTO image_generation_params (image_id, key, value) VALUES (?, ?, ?)",
                (image_id, key, str_value),
            )

        # Cache resources
        for res in meta.get("resources", []):
            cur.execute(
                "INSERT INTO image_resources (image_id, name, type, hash, weight) VALUES (?, ?, ?, ?, ?)",
                (image_id, res.get("name"), res.get("type"), res.get("hash"), res.get("weight")),
            )

        return image_id

    # =========================================================================
    # Query Operations
    # =========================================================================

    def search_models(
        self,
        query: str | None = None,
        model_type: str | None = None,
        base_model: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search cached models."""
        cur = self.conn.cursor()

        sql = "SELECT * FROM v_models_with_latest WHERE 1=1"
        params: list[Any] = []

        if query:
            sql += " AND name LIKE ?"
            params.append(f"%{query}%")

        if model_type:
            sql += " AND type = ?"
            params.append(model_type)

        if base_model:
            sql += " AND base_model LIKE ?"
            params.append(f"%{base_model}%")

        sql += " ORDER BY download_count DESC LIMIT ?"
        params.append(limit)

        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def get_model(self, civitai_id: int) -> dict[str, Any] | None:
        """Get cached model by CivitAI ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM v_models_with_latest WHERE civitai_id = ?", (civitai_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_triggers(self, file_path: str) -> list[str]:
        """Get trigger words for a local file."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT tw.word
            FROM trained_words tw
            JOIN model_versions mv ON tw.version_id = mv.id
            JOIN local_files lf ON lf.civitai_version_id = mv.civitai_id
            WHERE lf.file_path = ?
            ORDER BY tw.position
            """,
            (file_path,),
        )
        return [row["word"] for row in cur.fetchall()]

    def get_triggers_by_version(self, version_id: int) -> list[str]:
        """Get trigger words for a version by CivitAI version ID."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT tw.word
            FROM trained_words tw
            JOIN model_versions mv ON tw.version_id = mv.id
            WHERE mv.civitai_id = ?
            ORDER BY tw.position
            """,
            (version_id,),
        )
        return [row["word"] for row in cur.fetchall()]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, int]:
        """Get database statistics."""
        cur = self.conn.cursor()
        stats = {}
        for table in [
            "local_files",
            "models",
            "model_versions",
            "version_files",
            "trained_words",
            "creators",
            "tags",
        ]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cur.fetchone()[0]
        return stats
