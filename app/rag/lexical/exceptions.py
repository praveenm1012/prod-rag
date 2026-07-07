"""Lexical search exceptions."""


class LexicalSearchError(Exception):
    """Base exception for lexical search failures."""


class LexicalSearchConfigError(LexicalSearchError):
    """Raised when lexical search configuration is invalid."""


class LexicalSearchStateError(LexicalSearchError):
    """Raised when search is attempted before indexing."""
