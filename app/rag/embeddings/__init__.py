"""Embedding generation utilities."""

from app.rag.embeddings.cache import EmbeddingCache
from app.rag.embeddings.config import EmbeddingSettings, get_embedding_settings
from app.rag.embeddings.exceptions import (
    EmbeddingConfigError,
    EmbeddingError,
    EmbeddingModelError,
)
from app.rag.embeddings.models import EmbeddingBatchResult, EmbeddingRecord, PromptType
from app.rag.embeddings.service import EmbeddingService, apply_prompt, resolve_device

__all__ = [
    "EmbeddingBatchResult",
    "EmbeddingCache",
    "EmbeddingConfigError",
    "EmbeddingError",
    "EmbeddingModelError",
    "EmbeddingRecord",
    "EmbeddingService",
    "EmbeddingSettings",
    "PromptType",
    "apply_prompt",
    "get_embedding_settings",
    "resolve_device",
]
