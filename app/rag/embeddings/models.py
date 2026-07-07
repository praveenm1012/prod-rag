"""Embedding data models."""

from dataclasses import dataclass, field
from typing import Literal

PromptType = Literal["query", "document"]


@dataclass(frozen=True)
class EmbeddingRecord:
    """A single embedding with provenance metadata."""

    text: str
    vector: list[float]
    model_name: str
    prompt_type: PromptType
    cached: bool = False


@dataclass(frozen=True)
class EmbeddingBatchResult:
    """Batch embedding result with cache statistics."""

    records: list[EmbeddingRecord]
    cache_hits: int = 0
    cache_misses: int = 0
    metadata: dict[str, object] = field(default_factory=dict)
