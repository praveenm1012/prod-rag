"""Observability initialization and global accessors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import Settings, get_settings
from app.observability.costs import (
    CostTracker,
    reset_cost_tracker,
    set_cost_tracker,
)
from app.observability.evaluation import (
    EvaluationRecorder,
    reset_evaluation_recorder,
    set_evaluation_recorder,
)
from app.observability.metrics import MetricsRecorder, create_metrics_recorder
from app.observability.tracing import LangfuseTracer, Tracer, create_tracer

if TYPE_CHECKING:
    from app.observability.tracing import InMemoryTracer

_tracer: Tracer | None = None
_metrics: MetricsRecorder | None = None


def setup_observability(settings: Settings | None = None) -> Tracer:
    """Initialize tracing, metrics, costs, and evaluation recorders."""
    global _tracer, _metrics

    resolved = settings or get_settings()
    backend = (
        "none" if not resolved.observability_enabled else resolved.observability_backend
    )

    _tracer = create_tracer(
        backend,
        langfuse_public_key=resolved.langfuse_public_key,
        langfuse_secret_key=resolved.langfuse_secret_key,
        langfuse_base_url=resolved.langfuse_base_url,
    )

    langfuse_client = (
        _tracer.client if isinstance(_tracer, LangfuseTracer) else None
    )
    _metrics = create_metrics_recorder(backend, langfuse_client=langfuse_client)
    set_cost_tracker(CostTracker(_tracer, _metrics))
    set_evaluation_recorder(EvaluationRecorder(_tracer, _metrics))
    return _tracer


def shutdown_observability() -> None:
    """Flush and release observability resources."""
    global _tracer, _metrics

    if _tracer is not None:
        _tracer.flush()
        _tracer.shutdown()

    _tracer = None
    _metrics = None
    reset_cost_tracker()
    reset_evaluation_recorder()


def reset_observability() -> None:
    """Reset observability state without flushing (for tests)."""
    global _tracer, _metrics
    _tracer = None
    _metrics = None
    reset_cost_tracker()
    reset_evaluation_recorder()


def get_tracer() -> Tracer:
    """Return the configured tracer."""
    if _tracer is None:
        return setup_observability()
    return _tracer


def get_metrics() -> MetricsRecorder:
    """Return the configured metrics recorder."""
    if _metrics is None:
        setup_observability()
    assert _metrics is not None
    return _metrics


def get_memory_tracer() -> InMemoryTracer:
    """Return the active in-memory tracer for verification."""
    from app.observability.tracing import InMemoryTracer

    tracer = get_tracer()
    if not isinstance(tracer, InMemoryTracer):
        msg = "Observability backend is not memory"
        raise TypeError(msg)
    return tracer
