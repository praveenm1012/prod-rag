"""Chunking-specific exceptions."""


class ChunkingError(Exception):
    """Base exception for chunking failures."""


class InvalidChunkConfigError(ChunkingError):
    """Raised when splitter configuration is invalid."""
