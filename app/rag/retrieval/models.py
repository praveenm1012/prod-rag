"""Hybrid retrieval data models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HybridSearchResult:
    """A ranked hybrid retrieval hit."""

    chunk_id: str
    text: str
    score: float
    rank: int
    bm25_score: float
    vector_score: float
    bm25_score_normalized: float
    vector_score_normalized: float
    document_id: str
    page_number: int
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
