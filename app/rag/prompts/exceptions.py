"""Prompt builder exceptions."""


class PromptBuilderError(Exception):
    """Base exception for prompt builder failures."""


class PromptConfigError(PromptBuilderError):
    """Raised when prompt builder configuration is invalid."""


class PromptInputError(PromptBuilderError):
    """Raised when prompt builder input is invalid."""
