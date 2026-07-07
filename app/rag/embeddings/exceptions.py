"""Embedding-specific exceptions."""


class EmbeddingError(Exception):
    """Base exception for embedding failures."""


class EmbeddingModelError(EmbeddingError):
    """Raised when the embedding model fails to load or encode."""


class EmbeddingConfigError(EmbeddingError):
    """Raised when embedding configuration is invalid."""
