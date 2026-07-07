"""Document management data models."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

DocumentStatus = Literal["processing", "indexed", "failed"]
ProcessingPhase = Literal[
    "queued",
    "ingesting",
    "chunking",
    "embedding",
    "indexing",
]


@dataclass(frozen=True)
class DocumentRecord:
    """Metadata for an uploaded and indexed document."""

    document_id: str
    filename: str
    source: str
    chunk_count: int
    uploaded_at: datetime
    status: DocumentStatus = "indexed"
    phase: ProcessingPhase | None = None
    error_message: str | None = None
    file_size_bytes: int = 0


@dataclass(frozen=True)
class UploadResult:
    """Result of a document upload and indexing operation."""

    document: DocumentRecord
    chunks_indexed: int


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)
