"""Hybrid retrieval exceptions."""


class HybridRetrievalError(Exception):
    """Base exception for hybrid retrieval failures."""


class HybridRetrievalConfigError(HybridRetrievalError):
    """Raised when hybrid retrieval configuration is invalid."""


class HybridRetrievalStateError(HybridRetrievalError):
    """Raised when retrieval is attempted before indexing."""
