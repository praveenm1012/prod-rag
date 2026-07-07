"""Chunk models for the RAG pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A text chunk with identifiers and source provenance."""

    text: str
    chunk_id: str
    document_id: str
    page_number: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
