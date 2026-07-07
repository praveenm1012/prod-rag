"""LLM generation exceptions."""


class GenerationError(Exception):
    """Base exception for LLM generation failures."""


class GenerationConfigError(GenerationError):
    """Raised when generation configuration is invalid."""


class GenerationInputError(GenerationError):
    """Raised when generation input is invalid."""


class ProviderError(GenerationError):
    """Raised when an LLM provider request fails."""


class ProviderTimeoutError(ProviderError):
    """Raised when an LLM provider request times out."""


class ProviderResponseError(ProviderError):
    """Raised when an LLM provider returns an invalid response."""
