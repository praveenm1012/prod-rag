"""Prompt builder configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PromptSettings(BaseSettings):
    """Settings for RAG prompt construction."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    max_context_tokens: int = Field(default=4096, alias="PROMPT_MAX_CONTEXT_TOKENS")
    reserved_response_tokens: int = Field(
        default=512,
        alias="PROMPT_RESERVED_RESPONSE_TOKENS",
    )
    token_encoding: str = Field(default="cl100k_base", alias="PROMPT_TOKEN_ENCODING")
    system_prompt: str = Field(
        default=(
            "You are a helpful assistant. Answer using only the provided context. "
            "Cite sources using [N] notation matching the numbered context blocks."
        ),
        alias="PROMPT_SYSTEM",
    )
    context_header: str = Field(default="Context:", alias="PROMPT_CONTEXT_HEADER")
    examples_header: str = Field(default="Examples:", alias="PROMPT_EXAMPLES_HEADER")
    history_header: str = Field(
        default="Previous conversation:",
        alias="PROMPT_HISTORY_HEADER",
    )
    question_prefix: str = Field(default="Question:", alias="PROMPT_QUESTION_PREFIX")
    answer_prefix: str = Field(default="Answer:", alias="PROMPT_ANSWER_PREFIX")


@lru_cache
def get_prompt_settings() -> PromptSettings:
    """Return cached prompt settings."""
    return PromptSettings()
