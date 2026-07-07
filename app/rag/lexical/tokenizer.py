"""Text tokenization for lexical search."""

import re

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""
    return TOKEN_PATTERN.findall(text.lower())
