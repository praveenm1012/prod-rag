#!/usr/bin/env python3
"""Verify observability traces for startup and health requests."""

from __future__ import annotations

import argparse
import json
import sys

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.observability.init import get_memory_tracer, reset_observability


def _span_to_dict(span: object) -> dict[str, object]:
    from app.observability.tracing import SpanRecord

    if not isinstance(span, SpanRecord):
        return {"name": str(span)}

    return {
        "name": span.name,
        "status": span.status,
        "attributes": span.attributes,
        "input": span.input,
        "output": span.output,
        "children": [_span_to_dict(child) for child in span.children],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="memory",
        choices=["memory", "langfuse"],
        help="Observability backend to verify",
    )
    args = parser.parse_args()

    get_settings.cache_clear()
    reset_observability()

    import os

    os.environ["OBSERVABILITY_ENABLED"] = "true"
    os.environ["OBSERVABILITY_BACKEND"] = args.backend
    get_settings.cache_clear()

    with TestClient(create_app()) as client:
        response = client.get("/api/v1/health")
        response.raise_for_status()

        if args.backend == "memory":
            tracer = get_memory_tracer()
            span_names = [record.name for record in tracer.spans]
            if "application.startup" not in span_names:
                print("Missing startup trace", file=sys.stderr)
                return 1

            health_spans = [
                record
                for record in tracer.spans
                if record.attributes.get("http.route") == "/api/v1/health"
            ]
            if not health_spans:
                print("Missing health request trace", file=sys.stderr)
                return 1

            payload = {
                "backend": tracer.backend_name,
                "trace_count": len(tracer.spans),
                "spans": [_span_to_dict(record) for record in tracer.spans],
            }
            print(json.dumps(payload, indent=2))
            return 0

    from app.observability.init import get_tracer

    tracer = get_tracer()
    tracer.flush()
    print(
        json.dumps(
            {
                "backend": tracer.backend_name,
                "message": "Startup and health traces flushed to Langfuse",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
