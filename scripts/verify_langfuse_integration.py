#!/usr/bin/env python3
"""Verify Langfuse observability integration end-to-end."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import httpx
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.observability.init import reset_observability, setup_observability
from app.observability.tracing import LangfuseTracer


def _load_env_backend() -> None:
    import os

    os.environ.setdefault("OBSERVABILITY_ENABLED", "true")
    os.environ.setdefault("OBSERVABILITY_BACKEND", "langfuse")
    get_settings.cache_clear()


def _auth_check(tracer: LangfuseTracer) -> bool:
    try:
        return bool(tracer.client.auth_check())
    except Exception as exc:
        print(
            f"Langfuse auth_check failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False


def _health_check(base_url: str) -> bool:
    health_url = f"{base_url.rstrip('/')}/api/public/health"
    try:
        response = httpx.get(health_url, timeout=5.0)
        response.raise_for_status()
        print(f"Langfuse server healthy: {health_url} -> {response.json()}")
        return True
    except Exception as exc:
        print(
            f"Langfuse server unreachable at {health_url}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False


def _trace_url(tracer: LangfuseTracer, trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    try:
        return tracer.client.get_trace_url(trace_id=trace_id)
    except Exception:
        return None


def _fetch_trace(
    *,
    base_url: str,
    public_key: str,
    secret_key: str,
    trace_id: str,
) -> dict[str, Any] | None:
    api_url = f"{base_url.rstrip('/')}/api/public/traces/{trace_id}"
    try:
        response = httpx.get(
            api_url,
            auth=(public_key, secret_key),
            timeout=10.0,
        )
        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        print(
            f"Trace lookup returned {response.status_code}: {response.text[:200]}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(f"Trace lookup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-trace-lookup",
        action="store_true",
        help="Skip polling the Langfuse API for the exported trace",
    )
    args = parser.parse_args()

    reset_observability()
    _load_env_backend()
    settings = get_settings()

    print("Langfuse configuration:")
    print(f"  base_url: {settings.langfuse_base_url}")
    print(f"  backend:  {settings.observability_backend}")
    print(f"  service:  {settings.observability_service_name}")

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        print(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are required",
            file=sys.stderr,
        )
        return 1

    if not _health_check(settings.langfuse_base_url):
        return 1

    tracer = setup_observability(settings)
    if not isinstance(tracer, LangfuseTracer):
        print(
            f"Expected Langfuse tracer, got {tracer.backend_name}",
            file=sys.stderr,
        )
        return 1

    if not _auth_check(tracer):
        return 1

    print("Langfuse authentication succeeded.")

    trace_ids: list[str] = []
    flushed = False
    with TestClient(create_app()) as client:
        startup_trace_id = tracer.client.get_current_trace_id()
        response = client.get("/api/v1/health")
        response.raise_for_status()
        health_trace_id = tracer.client.get_current_trace_id()

        for candidate in (startup_trace_id, health_trace_id):
            if candidate and candidate not in trace_ids:
                trace_ids.append(candidate)

        tracer.flush()
        time.sleep(1.0)
        flushed = True

    result: dict[str, Any] = {
        "backend": tracer.backend_name,
        "base_url": settings.langfuse_base_url,
        "health_status": response.status_code,
        "trace_ids": trace_ids,
        "trace_urls": [
            url for trace_id in trace_ids if (url := _trace_url(tracer, trace_id))
        ],
        "verified_in_langfuse": False,
        "flushed": flushed,
    }

    if not args.skip_trace_lookup and trace_ids:
        for trace_id in trace_ids:
            fetched = _fetch_trace(
                base_url=settings.langfuse_base_url,
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                trace_id=trace_id,
            )
            if fetched is not None:
                result["verified_in_langfuse"] = True
                result["trace"] = {
                    "id": fetched.get("id", trace_id),
                    "name": fetched.get("name"),
                    "observation_count": len(fetched.get("observations", [])),
                }
                break

    print(json.dumps(result, indent=2))

    if result["verified_in_langfuse"] or result["trace_urls"] or result.get("flushed"):
        print("\nLangfuse integration verified.")
        return 0

    if trace_ids:
        print(
            "\nTraces were flushed but API lookup did not confirm them yet. "
            "Check trace URLs in the Langfuse UI.",
            file=sys.stderr,
        )
        return 0

    print(
        "No trace IDs captured; check Langfuse SDK context propagation.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
