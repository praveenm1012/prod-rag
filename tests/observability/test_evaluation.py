"""Evaluation observability tests."""

from app.observability.evaluation import EvaluationRecorder
from app.observability.metrics import InMemoryMetricsRecorder
from app.observability.tracing import InMemoryTracer


def test_record_score_emits_trace_and_metric() -> None:
    tracer = InMemoryTracer()
    metrics = InMemoryMetricsRecorder()
    recorder = EvaluationRecorder(tracer, metrics)

    score = recorder.record_score(
        "faithfulness",
        0.92,
        sample_id="sample-1",
        metadata={"dataset": "sample_rag_dataset"},
    )

    assert score.name == "faithfulness"
    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "rag.evaluation.score"
    assert any(
        point.name == "rag.evaluation.score" and point.value == 0.92
        for point in metrics.snapshot()
    )
