"""Live Langfuse integration tests using a local mock server."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from scripts.langfuse_mock_server import STATE, start_mock_server

from app.core.config import get_settings
from app.main import create_app
from app.observability.init import reset_observability, setup_observability
from app.observability.tracing import LangfuseTracer, span

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_env_key(name: str) -> str:
    for line in (REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"')
    msg = f"{name} not found in .env"
    raise KeyError(msg)


@pytest.fixture
def langfuse_mock() -> dict[str, str]:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or _read_env_key(
        "LANGFUSE_PUBLIC_KEY"
    )
    secret_key = os.getenv("LANGFUSE_SECRET_KEY") or _read_env_key(
        "LANGFUSE_SECRET_KEY"
    )

    STATE.health_checks = 0
    STATE.trace_batches.clear()
    STATE.auth_failures = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    base_url = f"http://127.0.0.1:{port}"
    server = start_mock_server(
        host="127.0.0.1",
        port=port,
        public_key=public_key,
        secret_key=secret_key,
    )

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/api/public/health", timeout=1.0)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        server.shutdown()
        msg = "Mock Langfuse server did not start"
        raise RuntimeError(msg)

    yield {
        "base_url": base_url,
        "public_key": public_key,
        "secret_key": secret_key,
    }

    server.shutdown()


def test_langfuse_live_auth_health_and_trace_export(
    langfuse_mock: dict[str, str],
) -> None:
    os.environ["OBSERVABILITY_ENABLED"] = "true"
    os.environ["OBSERVABILITY_BACKEND"] = "langfuse"
    os.environ["LANGFUSE_BASE_URL"] = langfuse_mock["base_url"]
    os.environ["LANGFUSE_PUBLIC_KEY"] = langfuse_mock["public_key"]
    os.environ["LANGFUSE_SECRET_KEY"] = langfuse_mock["secret_key"]
    get_settings.cache_clear()
    reset_observability()

    tracer = setup_observability()
    assert isinstance(tracer, LangfuseTracer)
    assert tracer.client.auth_check() is True

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        tracer.flush()
        time.sleep(0.5)

    assert STATE.health_checks >= 1
    assert STATE.trace_batches, "expected OTLP trace batches to be exported"
    assert STATE.auth_failures == 0

    with span(tracer, "langfuse.integration.verify") as active_span:
        active_span.set_output({"verified": True})
    tracer.flush()
    time.sleep(0.5)
    assert len(STATE.trace_batches) >= 2
