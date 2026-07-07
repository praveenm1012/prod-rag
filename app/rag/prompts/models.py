"""Prompt builder data models."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ContextChunk:
    """A retrieved chunk formatted for prompt context."""

    chunk_id: str
    text: str
    citation_id: int
    source: str = ""
    page_number: int | None = None
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationTurn:
    """A single turn in conversation history."""

    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class FewShotExample:
    """A question-answer pair used as a few-shot example."""

    question: str
    answer: str


@dataclass(frozen=True)
class Citation:
    """Citation metadata for a context block."""

    citation_id: int
    chunk_id: str
    source: str
    page_number: int | None
    placeholder: str


@dataclass(frozen=True)
class BuiltPrompt:
    """A fully assembled prompt ready for the LLM."""

    system_prompt: str
    user_prompt: str
    token_count: int
    truncated: bool
    citations: tuple[Citation, ...]
    included_context_count: int
    dropped_context_count: int = 0
    dropped_history_turns: int = 0
