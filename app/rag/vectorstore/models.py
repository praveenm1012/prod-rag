"""Vector store data models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetadataFilter:
    """Metadata filter for vector search and delete operations."""

    document_id: str | None = None
    source: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None

    def is_empty(self) -> bool:
        return not any(
            value is not None
            for value in (
                self.document_id,
                self.source,
                self.page_number,
                self.chunk_id,
            )
        )


@dataclass(frozen=True)
class VectorRecord:
    """A vector record to store in Qdrant."""

    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    """A vector search result."""

    id: str
    score: float
    payload: dict[str, Any]
