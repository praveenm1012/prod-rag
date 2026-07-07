"""Evaluation dataset and report models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationSample:
    """A single RAG evaluation example."""

    id: str
    question: str
    reference: str
    response: str
    contexts: list[str]
    expected_citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate score for one metric."""

    name: str
    mean: float
    min_score: float
    max_score: float


@dataclass(frozen=True)
class SampleScore:
    """Per-sample metric scores."""

    sample_id: str
    question: str
    scores: dict[str, float | None]
    response_preview: str


@dataclass(frozen=True)
class EvaluationReport:
    """Completed Ragas evaluation report."""

    dataset_path: str
    sample_count: int
    metric_summaries: tuple[MetricSummary, ...]
    sample_scores: tuple[SampleScore, ...]
    raw_scores: list[dict[str, float | None]]

    def metric_means(self) -> dict[str, float]:
        return {summary.name: summary.mean for summary in self.metric_summaries}
