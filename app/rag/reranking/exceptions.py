"""Reranking-specific exceptions."""


class RerankerError(Exception):
    """Base exception for reranking failures."""


class RerankerModelError(RerankerError):
    """Raised when the reranker model fails to load or score."""


class RerankerConfigError(RerankerError):
    """Raised when reranker configuration is invalid."""


class RerankerInputError(RerankerError):
    """Raised when reranker input is invalid."""
