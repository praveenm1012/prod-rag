"""Vector store exceptions."""


class VectorStoreError(Exception):
    """Base exception for vector store failures."""


class VectorStoreConfigError(VectorStoreError):
    """Raised when vector store configuration is invalid."""


class VectorStoreOperationError(VectorStoreError):
    """Raised when a vector store operation fails."""
