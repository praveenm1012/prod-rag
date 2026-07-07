"""Citation accuracy metric for RAG responses."""

from __future__ import annotations

import json
import re

_CITATION_PATTERN = re.compile(r"\[\d+\]")


def score_citation_accuracy(response: str, expected_citations: list[str]) -> float:
    """Return the fraction of expected citation placeholders present in the response."""
    if not expected_citations:
        return 1.0

    actual = set(_CITATION_PATTERN.findall(response))
    expected = set(expected_citations)
    matched = expected.intersection(actual)
    return len(matched) / len(expected)


def expected_citations_from_rubric(rubric: dict[str, str] | None) -> list[str]:
    """Parse expected citations from a Ragas rubric dictionary."""
    if not rubric:
        return []
    raw = rubric.get("expected_citations", "[]")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]
