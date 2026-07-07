"""LLM generation data models."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ChatMessage:
    """A chat message for LLM providers."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    """A generation request for an LLM provider."""

    messages: tuple[ChatMessage, ...]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResponse:
    """A completed LLM generation response."""

    content: str
    model: str
    provider: str
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamChunk:
    """A streamed token chunk from an LLM provider."""

    content: str
    finish_reason: str | None = None
    is_final: bool = False


def messages_from_prompt(
    system_prompt: str,
    user_prompt: str,
) -> tuple[ChatMessage, ...]:
    """Build chat messages from system and user prompt strings."""
    return (
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    )
