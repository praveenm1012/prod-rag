"""Evaluation observability hooks for RAG quality runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.observability.metrics import MetricsRecorder
from app.observability.tracing import Tracer, span


@dataclass
class EvaluationScore:
    """A recorded evaluation metric."""

    name: str
    value: float
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EvaluationRecorder:
    """Record RAG evaluation scores to tracing and metrics backends."""

    def __init__(self, tracer: Tracer, metrics: MetricsRecorder) -> None:
        self._tracer = tracer
        self._metrics = metrics
        self.scores: list[EvaluationScore] = []

    def record_score(
        self,
        name: str,
        value: float,
        *,
        sample_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationScore:
        """Record a single evaluation score."""
        score = EvaluationScore(
            name=name,
            value=value,
            sample_id=sample_id,
            metadata=dict(metadata or {}),
        )
        self.scores.append(score)

        attributes = {
            "metric": name,
            "value": value,
            "sample_id": sample_id,
        }
        with span(
            self._tracer,
            "rag.evaluation.score",
            kind="event",
            attributes={key: val for key, val in attributes.items() if val is not None},
        ) as active_span:
            active_span.set_input({"metric": name, "sample_id": sample_id})
            active_span.set_output({"value": value, **(metadata or {})})

        labels = {"metric": name}
        if sample_id:
            labels["sample_id"] = sample_id
        self._metrics.observe("rag.evaluation.score", value, labels=labels)
        return score

    def record_run_summary(
        self,
        *,
        dataset_name: str,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record aggregate evaluation metrics for a dataset run."""
        with span(
            self._tracer,
            "rag.evaluation.run",
            attributes={"dataset": dataset_name},
        ) as active_span:
            active_span.set_input({"dataset": dataset_name, "metadata": metadata or {}})
            active_span.set_output(metrics)
            for metric_name, metric_value in metrics.items():
                self.record_score(
                    metric_name,
                    metric_value,
                    metadata={"dataset": dataset_name, **(metadata or {})},
                )


_evaluation_recorder: EvaluationRecorder | None = None


def get_evaluation_recorder() -> EvaluationRecorder:
    """Return the configured evaluation recorder."""
    if _evaluation_recorder is None:
        msg = "Observability has not been initialized"
        raise RuntimeError(msg)
    return _evaluation_recorder


def set_evaluation_recorder(recorder: EvaluationRecorder) -> None:
    """Register the global evaluation recorder."""
    global _evaluation_recorder
    _evaluation_recorder = recorder


def reset_evaluation_recorder() -> None:
    """Clear the global evaluation recorder."""
    global _evaluation_recorder
    _evaluation_recorder = None
