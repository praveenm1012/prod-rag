"""Document management exceptions."""


class DocumentServiceError(Exception):
    """Base exception for document service failures."""


class DocumentNotFoundError(DocumentServiceError):
    """Raised when a document cannot be found."""


class DocumentUploadError(DocumentServiceError):
    """Raised when document upload or indexing fails."""


class UnsupportedUploadError(DocumentServiceError):
    """Raised when an uploaded file type is not supported."""
