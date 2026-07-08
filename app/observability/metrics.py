"""Metrics recording interfaces and lightweight in-process collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricPoint:
    """A single recorded metric sample."""

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: str = "counter"


class MetricsRecorder(ABC):
    """Metrics backend interface."""

    @abstractmethod
    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric."""

    @abstractmethod
    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a histogram or gauge sample."""

    @abstractmethod
    def snapshot(self) -> list[MetricPoint]:
        """Return recorded metrics for verification and debugging."""


class NoOpMetricsRecorder(MetricsRecorder):
    """Metrics recorder that discards all samples."""

    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        return None

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        return None

    def snapshot(self) -> list[MetricPoint]:
        return []


class InMemoryMetricsRecorder(MetricsRecorder):
    """In-memory metrics recorder for tests and local verification."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = (
            defaultdict(float)
        )
        self._observations: list[MetricPoint] = []

    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        key = (name, tuple(sorted((labels or {}).items())))
        self._counters[key] += value

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        self._observations.append(
            MetricPoint(
                name=name,
                value=value,
                labels=dict(labels or {}),
                metric_type="histogram",
            )
        )

    def snapshot(self) -> list[MetricPoint]:
        points = [
            MetricPoint(
                name=name,
                value=value,
                labels=dict(label_items),
                metric_type="counter",
            )
            for (name, label_items), value in self._counters.items()
        ]
        return points + list(self._observations)


class LangfuseMetricsRecorder(MetricsRecorder):
    """Metrics recorder that forwards counters to Langfuse scores."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._local = InMemoryMetricsRecorder()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        self._local.increment(name, value, labels=labels)
        trace_id = labels.get("trace_id") if labels else None
        if trace_id:
            self._client.create_score(
                name=name,
                value=value,
                trace_id=trace_id,
                data_type="NUMERIC",
            )

    def observe(
        self,
        name: str,
        value: float,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        self._local.observe(name, value, labels=labels)

    def snapshot(self) -> list[MetricPoint]:
        return self._local.snapshot()


def create_metrics_recorder(
    backend: str,
    *,
    langfuse_client: Any | None = None,
) -> MetricsRecorder:
    """Create a metrics recorder for the requested backend."""
    normalized = backend.lower().strip()
    if normalized in {"", "none", "disabled"}:
        return NoOpMetricsRecorder()

    if normalized == "memory":
        return InMemoryMetricsRecorder()

    if normalized == "langfuse" and langfuse_client is not None:
        return LangfuseMetricsRecorder(langfuse_client)

    return InMemoryMetricsRecorder()
