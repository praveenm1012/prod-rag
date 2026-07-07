"""Score normalization utilities."""

from collections.abc import Mapping


def min_max_normalize(scores: Mapping[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] using min-max scaling."""
    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)

    if max_score == min_score:
        return {key: (1.0 if value > 0 else 0.0) for key, value in scores.items()}

    scale = max_score - min_score
    return {key: (value - min_score) / scale for key, value in scores.items()}
