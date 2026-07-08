"""Tracing interfaces, span utilities, and backend adapters."""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

SpanKind = Literal["span", "generation", "event"]


@dataclass
class SpanRecord:
    """Completed span captured by in-memory and test backends."""

    name: str
    kind: SpanKind = "span"
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list[SpanRecord] = field(default_factory=list)
    status: str = "ok"
    input: Any | None = None
    output: Any | None = None
    error: str | None = None
    duration_ms: float | None = None
    trace_id: str | None = None
    span_id: str | None = None


class SpanContext(ABC):
    """Active span handle used by application code."""

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """Attach metadata to the active span."""

    @abstractmethod
    def set_input(self, value: Any) -> None:
        """Record span input payload."""

    @abstractmethod
    def set_output(self, value: Any) -> None:
        """Record span output payload."""

    @abstractmethod
    def record_exception(self, exc: BaseException) -> None:
        """Mark the span as failed and capture exception details."""

    @abstractmethod
    def end(self, *, status: str = "ok") -> SpanRecord | None:
        """Finish the span and return a record when supported."""


class Tracer(ABC):
    """Tracing backend interface."""

    @abstractmethod
    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        attributes: dict[str, Any] | None = None,
    ) -> SpanContext:
        """Create a new span."""

    @abstractmethod
    def flush(self) -> None:
        """Flush pending telemetry to the backend."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release backend resources."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the configured backend identifier."""


class NoOpSpanContext(SpanContext):
    """Span context that discards all telemetry."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_input(self, value: Any) -> None:
        return None

    def set_output(self, value: Any) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def end(self, *, status: str = "ok") -> SpanRecord | None:
        return None


class NoOpTracer(Tracer):
    """Tracer that performs no work when observability is disabled."""

    @property
    def backend_name(self) -> str:
        return "none"

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        attributes: dict[str, Any] | None = None,
    ) -> SpanContext:
        return NoOpSpanContext()

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


class InMemorySpanContext(SpanContext):
    """Span context that records completed spans for tests and local verification."""

    def __init__(
        self,
        tracer: InMemoryTracer,
        name: str,
        *,
        kind: SpanKind,
        attributes: dict[str, Any] | None,
        parent: SpanRecord | None,
    ) -> None:
        self._tracer = tracer
        self._record = SpanRecord(
            name=name,
            kind=kind,
            attributes=dict(attributes or {}),
        )
        self._parent = parent
        self._started_at = time.perf_counter()
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self._record.attributes[key] = value

    def set_input(self, value: Any) -> None:
        self._record.input = value

    def set_output(self, value: Any) -> None:
        self._record.output = value

    def record_exception(self, exc: BaseException) -> None:
        self._record.status = "error"
        self._record.error = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )

    def end(self, *, status: str = "ok") -> SpanRecord:
        if self._ended:
            return self._record

        self._ended = True
        if status == "error":
            self._record.status = "error"
        self._record.duration_ms = (time.perf_counter() - self._started_at) * 1000
        self._record.trace_id = self._tracer.current_trace_id
        self._record.span_id = f"{self._record.name}:{len(self._tracer.spans)}"

        if self._parent is None:
            self._tracer.spans.append(self._record)
        else:
            self._parent.children.append(self._record)

        if self._tracer._active_stack and self._tracer._active_stack[-1] is self:
            self._tracer._active_stack.pop()
            if self._tracer._active_stack:
                top = self._tracer._active_stack[-1]
                self._tracer.current_trace_id = top._record.trace_id
            else:
                self._tracer.current_trace_id = None

        return self._record


class InMemoryTracer(Tracer):
    """In-memory tracer for tests and local trace verification."""

    def __init__(self) -> None:
        self.spans: list[SpanRecord] = []
        self.current_trace_id: str | None = None
        self._active_stack: list[InMemorySpanContext] = []
        self._trace_counter = 0

    @property
    def backend_name(self) -> str:
        return "memory"

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        attributes: dict[str, Any] | None = None,
    ) -> SpanContext:
        parent_record = self._active_stack[-1]._record if self._active_stack else None
        context = InMemorySpanContext(
            self,
            name,
            kind=kind,
            attributes=attributes,
            parent=parent_record,
        )
        self._active_stack.append(context)
        if parent_record is None:
            self._trace_counter += 1
            self.current_trace_id = f"trace-{self._trace_counter}"
        return context

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        self._active_stack.clear()


class LangfuseSpanContext(SpanContext):
    """Span context backed by the Langfuse Python SDK."""

    def __init__(self, manager: Any, observation: Any) -> None:
        self._manager = manager
        self._observation = observation
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self._observation.update(metadata={key: value})

    def set_input(self, value: Any) -> None:
        self._observation.update(input=value)

    def set_output(self, value: Any) -> None:
        self._observation.update(output=value)

    def record_exception(self, exc: BaseException) -> None:
        self._observation.update(
            level="ERROR",
            status_message=str(exc),
            metadata={"exception_type": type(exc).__name__},
        )

    def end(self, *, status: str = "ok") -> SpanRecord | None:
        if self._ended:
            return None

        self._ended = True
        if status == "error":
            self._observation.update(level="ERROR")
        self._observation.end()
        self._manager.__exit__(None, None, None)
        return None


class LangfuseTracer(Tracer):
    """Tracer that exports spans to Langfuse."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def client(self) -> Any:
        return self._client

    @property
    def backend_name(self) -> str:
        return "langfuse"

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "span",
        attributes: dict[str, Any] | None = None,
    ) -> SpanContext:
        manager = self._client.start_as_current_observation(
            as_type=kind,
            name=name,
            metadata=attributes or {},
        )
        observation = manager.__enter__()
        return LangfuseSpanContext(manager, observation)

    def flush(self) -> None:
        self._client.flush()

    def shutdown(self) -> None:
        self.flush()


@contextmanager
def span(
    tracer: Tracer,
    name: str,
    *,
    kind: SpanKind = "span",
    attributes: dict[str, Any] | None = None,
) -> Iterator[SpanContext]:
    """Context manager that starts and ends a span, recording exceptions."""
    active_span = tracer.start_span(name, kind=kind, attributes=attributes)
    try:
        yield active_span
    except BaseException as exc:
        active_span.record_exception(exc)
        active_span.end(status="error")
        raise
    else:
        active_span.end(status="ok")


def create_tracer(
    backend: str,
    *,
    langfuse_public_key: str | None = None,
    langfuse_secret_key: str | None = None,
    langfuse_base_url: str | None = None,
) -> Tracer:
    """Create a tracer for the requested backend."""
    normalized = backend.lower().strip()
    if normalized in {"", "none", "disabled"}:
        return NoOpTracer()

    if normalized == "memory":
        return InMemoryTracer()

    if normalized == "langfuse":
        if not langfuse_public_key or not langfuse_secret_key:
            return NoOpTracer()

        from langfuse import Langfuse

        client = Langfuse(
            public_key=langfuse_public_key,
            secret_key=langfuse_secret_key,
            base_url=langfuse_base_url,
        )
        return LangfuseTracer(client)

    msg = f"Unsupported observability backend: {backend}"
    raise ValueError(msg)
