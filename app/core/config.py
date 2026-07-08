"""Application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="prod-rag", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=False, alias="LOG_JSON")

    observability_enabled: bool = Field(default=True, alias="OBSERVABILITY_ENABLED")
    observability_backend: str = Field(
        default="memory",
        alias="OBSERVABILITY_BACKEND",
        description="Tracing backend: memory, langfuse, or none",
    )
    observability_service_name: str = Field(
        default="prod-rag",
        alias="OBSERVABILITY_SERVICE_NAME",
    )
    langfuse_public_key: str | None = Field(
        default=None,
        alias="LANGFUSE_PUBLIC_KEY",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        alias="LANGFUSE_SECRET_KEY",
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        alias="LANGFUSE_BASE_URL",
    )
    observability_flush_on_shutdown: bool = Field(
        default=True,
        alias="OBSERVABILITY_FLUSH_ON_SHUTDOWN",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
