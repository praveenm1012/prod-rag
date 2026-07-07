"""Document models for the ingestion pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A loaded document with extracted text and source metadata."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
