"""SQLModel database models for tensors."""

from datetime import datetime
from typing import Any, Optional

from sqlmodel import Field, Relationship, SQLModel

# =============================================================================
# Local Files
# =============================================================================


class LocalFile(SQLModel, table=True):
    """Local safetensor file."""

    __tablename__ = "local_files"

    id: int | None = Field(default=None, primary_key=True)
    file_path: str = Field(unique=True)
    sha256: str = Field(index=True)
    header_size: int | None = None
    tensor_count: int | None = None
    civitai_model_id: int | None = Field(default=None, index=True)
    civitai_version_id: int | None = None
    created_at: datetime | None = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = Field(default_factory=datetime.utcnow)

    metadata_entries: list["SafetensorMetadata"] = Relationship(back_populates="local_file")


class SafetensorMetadata(SQLModel, table=True):
    """Safetensor header metadata key-value pairs."""

    __tablename__ = "safetensor_metadata"

    id: int | None = Field(default=None, primary_key=True)
    local_file_id: int = Field(foreign_key="local_files.id", index=True)
    key: str
    value: str | None = None

    local_file: "LocalFile" = Relationship(back_populates="metadata_entries")


# =============================================================================
# CivitAI Models
# =============================================================================


class Creator(SQLModel, table=True):
    """CivitAI model creator."""

    __tablename__ = "creators"

    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True)
    image_url: str | None = None

    models: list["Model"] = Relationship(back_populates="creator")


class Tag(SQLModel, table=True):
    """Model tag."""

    __tablename__ = "tags"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)


class ModelTag(SQLModel, table=True):
    """Model-tag association."""

    __tablename__ = "model_tags"

    model_id: int = Field(foreign_key="models.id", primary_key=True)
    tag_id: int = Field(foreign_key="tags.id", primary_key=True)


class Model(SQLModel, table=True):
    """CivitAI model."""

    __tablename__ = "models"

    id: int | None = Field(default=None, primary_key=True)
    civitai_id: int = Field(unique=True, index=True)
    name: str = Field(index=True)
    description: str | None = None
    type: str = Field(index=True)
    nsfw: bool = False
    poi: bool = False
    minor: bool = False
    sfw_only: bool = False
    nsfw_level: int | None = None
    availability: str | None = None
    allow_no_credit: bool | None = None
    allow_commercial_use: str | None = None
    allow_derivatives: bool | None = None
    allow_different_license: bool | None = None
    supports_generation: bool = False
    creator_id: int | None = Field(default=None, foreign_key="creators.id")
    download_count: int = 0
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    comment_count: int = 0
    tipped_amount_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = Field(default_factory=datetime.utcnow)

    creator: Optional["Creator"] = Relationship(back_populates="models")
    versions: list["ModelVersion"] = Relationship(back_populates="model")


class ModelVersion(SQLModel, table=True):
    """CivitAI model version."""

    __tablename__ = "model_versions"

    id: int | None = Field(default=None, primary_key=True)
    civitai_id: int = Field(unique=True, index=True)
    model_id: int = Field(foreign_key="models.id", index=True)
    name: str
    description: str | None = None
    base_model: str | None = Field(default=None, index=True)
    base_model_type: str | None = None
    nsfw_level: int | None = None
    status: str | None = None
    availability: str | None = None
    upload_type: str | None = None
    usage_control: str | None = None
    air: str | None = None
    training_status: str | None = None
    training_details: str | None = None
    early_access_ends_at: datetime | None = None
    download_count: int = 0
    thumbs_up_count: int = 0
    thumbs_down_count: int = 0
    supports_generation: bool = False
    download_url: str | None = None
    created_at: datetime | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    version_index: int | None = None

    model: "Model" = Relationship(back_populates="versions")
    files: list["VersionFile"] = Relationship(back_populates="version")
    images: list["VersionImage"] = Relationship(back_populates="version")
    trained_words: list["TrainedWord"] = Relationship(back_populates="version")


class TrainedWord(SQLModel, table=True):
    """Trigger words for a model version."""

    __tablename__ = "trained_words"

    id: int | None = Field(default=None, primary_key=True)
    version_id: int = Field(foreign_key="model_versions.id", index=True)
    word: str
    position: int | None = None

    version: "ModelVersion" = Relationship(back_populates="trained_words")


class VersionFile(SQLModel, table=True):
    """Model version file."""

    __tablename__ = "version_files"

    id: int | None = Field(default=None, primary_key=True)
    civitai_id: int = Field(unique=True)
    version_id: int = Field(foreign_key="model_versions.id", index=True)
    name: str
    type: str | None = None
    size_kb: float | None = None
    format: str | None = None
    size_type: str | None = None
    fp: str | None = None
    is_primary: bool = False
    pickle_scan_result: str | None = None
    pickle_scan_message: str | None = None
    virus_scan_result: str | None = None
    virus_scan_message: str | None = None
    scanned_at: datetime | None = None
    download_url: str | None = None

    version: "ModelVersion" = Relationship(back_populates="files")
    hashes: list["FileHash"] = Relationship(back_populates="file")


class FileHash(SQLModel, table=True):
    """File hash values."""

    __tablename__ = "file_hashes"

    id: int | None = Field(default=None, primary_key=True)
    file_id: int = Field(foreign_key="version_files.id", index=True)
    hash_type: str
    hash_value: str = Field(index=True)

    file: "VersionFile" = Relationship(back_populates="hashes")


class VersionImage(SQLModel, table=True):
    """Model version example image."""

    __tablename__ = "version_images"

    id: int | None = Field(default=None, primary_key=True)
    civitai_id: int | None = None
    version_id: int = Field(foreign_key="model_versions.id", index=True)
    url: str
    type: str | None = None
    nsfw_level: int | None = None
    width: int | None = None
    height: int | None = None
    hash: str | None = None
    has_meta: bool = False
    has_positive_prompt: bool = False
    on_site: bool = False
    minor: bool = False
    poi: bool = False
    availability: str | None = None
    remix_of_id: int | None = None

    version: "ModelVersion" = Relationship(back_populates="images")
    generation_params: list["ImageGenerationParam"] = Relationship(back_populates="image")
    resources: list["ImageResource"] = Relationship(back_populates="image")


class ImageVideoMetadata(SQLModel, table=True):
    """Video metadata for animated images."""

    __tablename__ = "image_video_metadata"

    id: int | None = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="version_images.id", unique=True)
    duration: float | None = None
    has_audio: bool = False
    size_bytes: int | None = None


class ImageGenerationParam(SQLModel, table=True):
    """Image generation parameters."""

    __tablename__ = "image_generation_params"

    id: int | None = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="version_images.id", index=True)
    key: str
    value: str | None = None

    image: "VersionImage" = Relationship(back_populates="generation_params")


class ImageResource(SQLModel, table=True):
    """Resources used in image generation."""

    __tablename__ = "image_resources"

    id: int | None = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="version_images.id", index=True)
    name: str
    type: str | None = None
    hash: str | None = None
    weight: float | None = None

    image: "VersionImage" = Relationship(back_populates="resources")


# =============================================================================
# HuggingFace Models
# =============================================================================


class HFModel(SQLModel, table=True):
    """HuggingFace model."""

    __tablename__ = "hf_models"

    id: int | None = Field(default=None, primary_key=True)
    repo_id: str = Field(unique=True, index=True)
    author: str | None = Field(default=None, index=True)
    model_name: str
    pipeline_tag: str | None = None
    library_name: str | None = None
    downloads: int = Field(default=0, index=True)
    likes: int = 0
    trending_score: float | None = None
    is_private: bool = False
    is_gated: bool = False
    last_modified: datetime | None = None
    created_at: datetime | None = None
    cached_at: datetime | None = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = Field(default_factory=datetime.utcnow)

    tags: list["HFModelTag"] = Relationship(back_populates="model")
    safetensor_files: list["HFSafetensorFile"] = Relationship(back_populates="model")


class HFModelTag(SQLModel, table=True):
    """HuggingFace model tag."""

    __tablename__ = "hf_model_tags"

    hf_model_id: int = Field(foreign_key="hf_models.id", primary_key=True, index=True)
    tag: str = Field(primary_key=True)

    model: "HFModel" = Relationship(back_populates="tags")


class HFSafetensorFile(SQLModel, table=True):
    """Safetensor file in HuggingFace model."""

    __tablename__ = "hf_safetensor_files"

    id: int | None = Field(default=None, primary_key=True)
    hf_model_id: int = Field(foreign_key="hf_models.id", index=True)
    filename: str
    size_bytes: int | None = None

    model: "HFModel" = Relationship(back_populates="safetensor_files")


# =============================================================================
# Database Setup
# =============================================================================


def get_engine(db_path: str = "") -> Any:
    """Create database engine."""
    from sqlmodel import create_engine  # noqa: PLC0415

    from tensors.config import DATA_DIR  # noqa: PLC0415

    if not db_path:
        db_path = str(DATA_DIR / "models.db")

    return create_engine(f"sqlite:///{db_path}", echo=False)


def create_tables(engine: Any) -> None:
    """Create all tables."""
    SQLModel.metadata.create_all(engine)
