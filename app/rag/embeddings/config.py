"""Embedding service configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Settings for the embedding service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    model_name: str = Field(
        default="BAAI/bge-large-en-v1.5",
        alias="EMBEDDING_MODEL",
    )
    device: str = Field(default="auto", alias="EMBEDDING_DEVICE")
    batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    cache_dir: Path = Field(
        default=Path(".cache/embeddings"),
        alias="EMBEDDING_CACHE_DIR",
    )
    max_retries: int = Field(default=3, alias="EMBEDDING_MAX_RETRIES")
    retry_min_seconds: float = Field(default=1.0, alias="EMBEDDING_RETRY_MIN_SECONDS")
    retry_max_seconds: float = Field(default=10.0, alias="EMBEDDING_RETRY_MAX_SECONDS")
    normalize_embeddings: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    show_progress: bool = Field(default=True, alias="EMBEDDING_SHOW_PROGRESS")


@lru_cache
def get_embedding_settings() -> EmbeddingSettings:
    """Return cached embedding settings."""
    return EmbeddingSettings()
