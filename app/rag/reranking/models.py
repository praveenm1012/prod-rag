"""Reranker data models."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RerankCandidate:
    """A candidate chunk to rerank against a question."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    """A reranked candidate with cross-encoder score."""

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
