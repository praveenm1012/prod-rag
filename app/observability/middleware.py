"""HTTP request tracing middleware."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.observability.init import get_metrics, get_tracer
from app.observability.tracing import span


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Create a trace span for each incoming HTTP request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        tracer = get_tracer()
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        span_name = f"http {request.method.lower()} {route_path}"

        attributes = {
            "http.method": request.method,
            "http.route": route_path,
            "http.url": str(request.url),
            "http.scheme": request.url.scheme,
        }

        started_at = time.perf_counter()
        with span(tracer, span_name, attributes=attributes) as active_span:
            active_span.set_input(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "query": request.url.query,
                }
            )
            try:
                response = await call_next(request)
            except BaseException as exc:
                active_span.set_attribute("http.status_code", 500)
                raise exc from None
            else:
                duration_ms = (time.perf_counter() - started_at) * 1000
                active_span.set_attribute("http.status_code", response.status_code)
                active_span.set_output(
                    {
                        "status_code": response.status_code,
                        "duration_ms": round(duration_ms, 2),
                    }
                )
                get_metrics().observe(
                    "http.request.duration_ms",
                    duration_ms,
                    labels={
                        "method": request.method,
                        "route": route_path,
                        "status_code": str(response.status_code),
                    },
                )
                get_metrics().increment(
                    "http.requests.total",
                    labels={
                        "method": request.method,
                        "route": route_path,
                        "status_code": str(response.status_code),
                    },
                )
                return response
