"""Lexical search configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LexicalSearchSettings(BaseSettings):
    """Settings for BM25 lexical search."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    top_k: int = Field(default=10, alias="LEXICAL_TOP_K")
    k1: float = Field(default=1.5, alias="LEXICAL_BM25_K1")
    b: float = Field(default=0.75, alias="LEXICAL_BM25_B")


@lru_cache
def get_lexical_search_settings() -> LexicalSearchSettings:
    """Return cached lexical search settings."""
    return LexicalSearchSettings()
