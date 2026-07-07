"""Ingestion-specific exceptions."""


class IngestionError(Exception):
    """Base exception for document ingestion failures."""


class UnsupportedDocumentError(IngestionError):
    """Raised when no loader supports the given source."""


class DocumentLoadError(IngestionError):
    """Raised when a loader fails to read a supported document."""
