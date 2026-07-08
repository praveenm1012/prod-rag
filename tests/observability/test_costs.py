"""Cost tracking tests."""

from app.observability.costs import CostTracker
from app.observability.metrics import InMemoryMetricsRecorder
from app.observability.tracing import InMemoryTracer


def test_record_generation_tracks_tokens_and_cost() -> None:
    tracer = InMemoryTracer()
    metrics = InMemoryMetricsRecorder()
    tracker = CostTracker(tracer, metrics)

    record = tracker.record_generation(
        model="gpt-4o-mini",
        provider="openai",
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
        },
    )

    assert record.total_tokens == 1500
    assert record.estimated_cost_usd > 0
    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "llm.generation.cost"
    assert any(point.name == "llm.tokens.total" for point in metrics.snapshot())
