"""LLM generation configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "claude", "ollama"]


class GenerationSettings(BaseSettings):
    """Settings for LLM generation providers."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    provider: ProviderName = Field(default="openai", alias="LLM_PROVIDER")
    model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    retry_min_seconds: float = Field(default=1.0, alias="LLM_RETRY_MIN_SECONDS")
    retry_max_seconds: float = Field(default=10.0, alias="LLM_RETRY_MAX_SECONDS")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL",
    )
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        alias="ANTHROPIC_BASE_URL",
    )
    anthropic_version: str = Field(
        default="2023-06-01",
        alias="ANTHROPIC_VERSION",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )


@lru_cache
def get_generation_settings() -> GenerationSettings:
    """Return cached generation settings."""
    return GenerationSettings()
