"""Qdrant vector store configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    """Settings for the Qdrant vector store."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    collection_name: str = Field(default="prod_rag", alias="QDRANT_COLLECTION")
    vector_size: int = Field(default=1024, alias="QDRANT_VECTOR_SIZE")
    timeout_seconds: float = Field(default=30.0, alias="QDRANT_TIMEOUT_SECONDS")


@lru_cache
def get_qdrant_settings() -> QdrantSettings:
    """Return cached Qdrant settings."""
    return QdrantSettings()
