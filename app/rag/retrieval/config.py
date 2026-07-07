"""Hybrid retrieval configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class HybridRetrievalSettings(BaseSettings):
    """Settings for hybrid BM25 + vector retrieval."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    bm25_weight: float = Field(default=0.5, alias="HYBRID_BM25_WEIGHT")
    vector_weight: float = Field(default=0.5, alias="HYBRID_VECTOR_WEIGHT")
    top_k: int = Field(default=10, alias="HYBRID_TOP_K")
    candidate_pool_size: int = Field(default=50, alias="HYBRID_CANDIDATE_POOL_SIZE")


@lru_cache
def get_hybrid_retrieval_settings() -> HybridRetrievalSettings:
    """Return cached hybrid retrieval settings."""
    return HybridRetrievalSettings()
