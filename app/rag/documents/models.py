"""Document management data models."""

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class DocumentRecord:
    """Metadata for an uploaded and indexed document."""

    document_id: str
    filename: str
    source: str
    chunk_count: int
    uploaded_at: datetime


@dataclass(frozen=True)
class UploadResult:
    """Result of a document upload and indexing operation."""

    document: DocumentRecord
    chunks_indexed: int


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)
