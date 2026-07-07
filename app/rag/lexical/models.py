"""Lexical search data models."""

from dataclasses import dataclass, field
from typing import Any

from app.rag.chunking.models import Chunk


@dataclass(frozen=True)
class LexicalDocument:
    """A document indexed for lexical search."""

    chunk_id: str
    text: str
    document_id: str
    page_number: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> "LexicalDocument":
        """Build a lexical document from a chunk."""
        metadata = dict(chunk.metadata)
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            document_id=chunk.document_id,
            page_number=chunk.page_number,
            source=chunk.source,
            metadata=metadata,
        )


@dataclass(frozen=True)
class LexicalSearchResult:
    """A ranked lexical search hit."""

    chunk_id: str
    text: str
    score: float
    rank: int
    document_id: str
    page_number: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
