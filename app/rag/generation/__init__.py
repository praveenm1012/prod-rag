"""LLM generation utilities."""

from app.rag.generation.base import LLMProvider
from app.rag.generation.claude import ClaudeProvider
from app.rag.generation.config import (
    GenerationSettings,
    ProviderName,
    get_generation_settings,
)
from app.rag.generation.exceptions import (
    GenerationConfigError,
    GenerationError,
    GenerationInputError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from app.rag.generation.factory import create_llm_provider
from app.rag.generation.http import HttpClient, HttpxHttpClient
from app.rag.generation.models import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
    messages_from_prompt,
)
from app.rag.generation.ollama import OllamaProvider
from app.rag.generation.openai import OpenAIProvider

__all__ = [
    "ChatMessage",
    "ClaudeProvider",
    "GenerationConfigError",
    "GenerationError",
    "GenerationInputError",
    "GenerationRequest",
    "GenerationResponse",
    "GenerationSettings",
    "HttpClient",
    "HttpxHttpClient",
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "ProviderError",
    "ProviderName",
    "ProviderResponseError",
    "ProviderTimeoutError",
    "StreamChunk",
    "create_llm_provider",
    "get_generation_settings",
    "messages_from_prompt",
]
