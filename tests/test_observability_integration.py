"""Integration tests for application observability."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.observability.init import get_memory_tracer, setup_observability
from app.observability.tracing import NoOpTracer


@pytest.fixture
def observability_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "memory")
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        yield client


def test_startup_creates_trace(observability_client: TestClient) -> None:
    tracer = get_memory_tracer()
    span_names = [record.name for record in tracer.spans]

    assert "application.startup" in span_names
    startup = next(
        record for record in tracer.spans if record.name == "application.startup"
    )
    assert startup.output == {"status": "ready"}


def test_health_request_is_traced(observability_client: TestClient) -> None:
    response = observability_client.get("/api/v1/health")
    assert response.status_code == 200

    tracer = get_memory_tracer()
    http_spans = [
        record
        for record in tracer.spans
        if record.name.startswith("http get ")
    ]
    assert http_spans, "expected an HTTP span for the health request"

    health_span = next(
        span
        for span in http_spans
        if span.attributes.get("http.route") == "/api/v1/health"
    )
    assert health_span.output is not None
    assert health_span.output["status_code"] == 200


def test_observability_disabled_uses_noop_tracer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "false")
    get_settings.cache_clear()

    settings = Settings()
    tracer = setup_observability(settings)
    assert isinstance(tracer, NoOpTracer)


@pytest.mark.skipif(
    not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"),
    reason="Langfuse credentials are not configured",
)
def test_langfuse_backend_can_flush(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OBSERVABILITY_ENABLED", "true")
    monkeypatch.setenv("OBSERVABILITY_BACKEND", "langfuse")
    get_settings.cache_clear()

    tracer = setup_observability()
    assert tracer.backend_name == "langfuse"
    with tracer.start_span("observability.verify") as span:
        span.set_output({"status": "ok"})
        span.end()
    tracer.flush()
