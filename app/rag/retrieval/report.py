"""Hybrid retrieval evaluation report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QueryEvaluation:
    """Evaluation metrics for a single query."""

    query: str
    relevant_ids: set[str]
    bm25_top: list[str]
    vector_top: list[str]
    hybrid_top: list[str]
    bm25_recall_at_k: float
    vector_recall_at_k: float
    hybrid_recall_at_k: float
    bm25_mrr: float
    vector_mrr: float
    hybrid_mrr: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate evaluation metrics."""

    query_count: int
    top_k: int
    bm25_weight: float
    vector_weight: float
    avg_bm25_recall_at_k: float
    avg_vector_recall_at_k: float
    avg_hybrid_recall_at_k: float
    avg_bm25_mrr: float
    avg_vector_mrr: float
    avg_hybrid_mrr: float
    bm25_latency_ms: float
    vector_latency_ms: float
    hybrid_latency_ms: float


def render_evaluation_report(
    summary: EvaluationSummary,
    per_query: list[QueryEvaluation],
) -> str:
    """Render a markdown evaluation report."""
    lines = [
        "# Hybrid Retrieval Evaluation Report",
        "",
        "## Configuration",
        "",
        f"- Queries evaluated: **{summary.query_count}**",
        f"- Top K: **{summary.top_k}**",
        f"- BM25 weight: **{summary.bm25_weight:.2f}**",
        f"- Vector weight: **{summary.vector_weight:.2f}**",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | BM25 | Vector | Hybrid |",
        "|--------|------|--------|--------|",
        (
            f"| Recall@{summary.top_k} "
            f"| {summary.avg_bm25_recall_at_k:.3f} "
            f"| {summary.avg_vector_recall_at_k:.3f} "
            f"| {summary.avg_hybrid_recall_at_k:.3f} |"
        ),
        (
            f"| MRR "
            f"| {summary.avg_bm25_mrr:.3f} "
            f"| {summary.avg_vector_mrr:.3f} "
            f"| {summary.avg_hybrid_mrr:.3f} |"
        ),
        (
            f"| Avg latency (ms) "
            f"| {summary.bm25_latency_ms:.2f} "
            f"| {summary.vector_latency_ms:.2f} "
            f"| {summary.hybrid_latency_ms:.2f} |"
        ),
        "",
        "## Per-query Results",
        "",
    ]

    for index, result in enumerate(per_query, start=1):
        lines.extend(
            [
                f"### Query {index}",
                "",
                f"**Query:** `{result.query}`",
                "",
                f"**Relevant:** {sorted(result.relevant_ids)}",
                "",
                (
                    f"- BM25 recall@{summary.top_k}: {result.bm25_recall_at_k:.3f}, "
                    f"MRR: {result.bm25_mrr:.3f}, top: {result.bm25_top}"
                ),
                (
                    f"- Vector recall@{summary.top_k}: "
                    f"{result.vector_recall_at_k:.3f}, "
                    f"MRR: {result.vector_mrr:.3f}, top: {result.vector_top}"
                ),
                (
                    f"- Hybrid recall@{summary.top_k}: "
                    f"{result.hybrid_recall_at_k:.3f}, "
                    f"MRR: {result.hybrid_mrr:.3f}, top: {result.hybrid_top}"
                ),
                "",
            ]
        )

    hybrid_wins = sum(
        1
        for result in per_query
        if result.hybrid_recall_at_k
        >= max(result.bm25_recall_at_k, result.vector_recall_at_k)
    )
    lines.extend(
        [
            "## Summary",
            "",
            (
                f"Hybrid matched or beat the best single retriever on "
                f"**{hybrid_wins}/{summary.query_count}** queries."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_report(
    path: Path,
    summary: EvaluationSummary,
    per_query: list[QueryEvaluation],
) -> Path:
    """Write the evaluation report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_evaluation_report(summary, per_query), encoding="utf-8")
    return path
