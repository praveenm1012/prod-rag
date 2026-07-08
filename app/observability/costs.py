"""LLM cost tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.observability.metrics import MetricsRecorder
from app.observability.tracing import SpanContext, Tracer, span

# USD per 1M tokens (input, output) for common models.
DEFAULT_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.00, 30.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
}


@dataclass
class CostRecord:
    """A recorded LLM usage cost event."""

    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    metadata: dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Track token usage and estimated model costs."""

    def __init__(
        self,
        tracer: Tracer,
        metrics: MetricsRecorder,
        *,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._tracer = tracer
        self._metrics = metrics
        self._pricing = pricing or DEFAULT_MODEL_PRICING
        self.records: list[CostRecord] = []

    def estimate_cost_usd(
        self,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Estimate USD cost from token usage."""
        input_rate, output_rate = self._pricing.get(model, (0.0, 0.0))
        prompt_cost = (prompt_tokens / 1_000_000) * input_rate
        completion_cost = (completion_tokens / 1_000_000) * output_rate
        return prompt_cost + completion_cost

    def record_generation(
        self,
        *,
        model: str,
        provider: str,
        usage: dict[str, int],
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """Record LLM usage and emit tracing/metrics signals."""
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
        estimated_cost = self.estimate_cost_usd(
            model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        record = CostRecord(
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
            metadata=dict(metadata or {}),
        )
        self.records.append(record)

        with span(
            self._tracer,
            "llm.generation.cost",
            kind="generation",
            attributes={
                "model": model,
                "provider": provider,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
            },
        ) as active_span:
            self._annotate_span(active_span, record)

        self._metrics.increment(
            "llm.tokens.total",
            float(total_tokens),
            labels={"model": model, "provider": provider},
        )
        self._metrics.observe(
            "llm.cost.usd",
            estimated_cost,
            labels={"model": model, "provider": provider},
        )
        return record

    @staticmethod
    def _annotate_span(active_span: SpanContext, record: CostRecord) -> None:
        active_span.set_input(
            {
                "model": record.model,
                "provider": record.provider,
                "prompt_tokens": record.prompt_tokens,
                "completion_tokens": record.completion_tokens,
            }
        )
        active_span.set_output(
            {
                "total_tokens": record.total_tokens,
                "estimated_cost_usd": record.estimated_cost_usd,
            }
        )


_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Return the configured cost tracker."""
    if _cost_tracker is None:
        msg = "Observability has not been initialized"
        raise RuntimeError(msg)
    return _cost_tracker


def set_cost_tracker(tracker: CostTracker) -> None:
    """Register the global cost tracker."""
    global _cost_tracker
    _cost_tracker = tracker


def reset_cost_tracker() -> None:
    """Clear the global cost tracker."""
    global _cost_tracker
    _cost_tracker = None
