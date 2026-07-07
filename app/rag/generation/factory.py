"""LLM provider factory."""

from __future__ import annotations

from typing import cast

from app.rag.generation.base import BaseLLMProvider, LLMProvider
from app.rag.generation.claude import ClaudeProvider
from app.rag.generation.config import (
    GenerationSettings,
    ProviderName,
    get_generation_settings,
)
from app.rag.generation.exceptions import GenerationConfigError
from app.rag.generation.http import HttpClient, HttpxHttpClient
from app.rag.generation.ollama import OllamaProvider
from app.rag.generation.openai import OpenAIProvider

_PROVIDER_TYPES: dict[ProviderName, type[BaseLLMProvider]] = {
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
    "ollama": OllamaProvider,
}


def create_llm_provider(
    provider: ProviderName | None = None,
    settings: GenerationSettings | None = None,
    http_client: HttpClient | None = None,
) -> LLMProvider:
    """Create an LLM provider from settings."""
    resolved_settings = settings or get_generation_settings()
    resolved_provider = provider or resolved_settings.provider
    client: HttpClient = http_client or cast(
        HttpClient,
        HttpxHttpClient(resolved_settings.timeout_seconds),
    )

    provider_type = _PROVIDER_TYPES.get(resolved_provider)
    if provider_type is None:
        raise GenerationConfigError(f"Unsupported LLM provider: {resolved_provider}")

    return provider_type(resolved_settings, client)
