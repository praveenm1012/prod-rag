"""Token counting utilities."""

from __future__ import annotations

from typing import Any, Protocol

from app.rag.prompts.exceptions import PromptConfigError


class TokenCounter(Protocol):
    """Count tokens in text for budgeting."""

    def count(self, text: str) -> int:
        """Return the token count for the given text."""
        ...


class CharTokenCounter:
    """Approximate token count using character length."""

    def __init__(self, chars_per_token: int = 4) -> None:
        if chars_per_token <= 0:
            raise PromptConfigError("chars_per_token must be greater than 0")
        self._chars_per_token = chars_per_token

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text) + self._chars_per_token - 1) // self._chars_per_token)


class TiktokenCounter:
    """Token counter backed by tiktoken."""

    def __init__(self, encoding_name: str) -> None:
        self._encoding_name = encoding_name
        self._encoding = _load_encoding(encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))


def create_token_counter(encoding_name: str) -> TokenCounter:
    """Create a token counter, falling back to char approximation."""
    try:
        return TiktokenCounter(encoding_name)
    except PromptConfigError:
        return CharTokenCounter()


def _load_encoding(encoding_name: str) -> Any:
    try:
        import tiktoken
    except ImportError as exc:
        raise PromptConfigError(
            "tiktoken is required for accurate token counting"
        ) from exc

    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:
        raise PromptConfigError(f"Unknown token encoding: {encoding_name}") from exc
