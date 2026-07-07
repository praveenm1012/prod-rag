"""Hybrid retrieval evaluation metrics."""

from collections.abc import Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute recall@k for a ranked result list."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute precision@k for a ranked result list."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / len(top_k)


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    """Compute reciprocal rank of the first relevant hit."""
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / index
    return 0.0
