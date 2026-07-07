"""Reranker configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankerSettings(BaseSettings):
    """Settings for the cross-encoder reranker."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    model_name: str = Field(
        default="BAAI/bge-reranker-large",
        alias="RERANKER_MODEL",
    )
    device: str = Field(default="auto", alias="RERANKER_DEVICE")
    batch_size: int = Field(default=32, alias="RERANKER_BATCH_SIZE")
    top_k: int = Field(default=5, alias="RERANKER_TOP_K")
    max_retries: int = Field(default=3, alias="RERANKER_MAX_RETRIES")
    retry_min_seconds: float = Field(default=1.0, alias="RERANKER_RETRY_MIN_SECONDS")
    retry_max_seconds: float = Field(default=10.0, alias="RERANKER_RETRY_MAX_SECONDS")
    show_progress: bool = Field(default=True, alias="RERANKER_SHOW_PROGRESS")


@lru_cache
def get_reranker_settings() -> RerankerSettings:
    """Return cached reranker settings."""
    return RerankerSettings()
