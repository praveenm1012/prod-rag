"""Tracing utility tests."""

from app.observability.tracing import InMemoryTracer, span


def test_span_records_successful_child_span() -> None:
    tracer = InMemoryTracer()

    with span(tracer, "parent") as parent_span:
        parent_span.set_input({"step": "parent"})
        with span(tracer, "child") as child_span:
            child_span.set_output({"step": "child"})

    assert len(tracer.spans) == 1
    parent = tracer.spans[0]
    assert parent.name == "parent"
    assert parent.status == "ok"
    assert len(parent.children) == 1
    assert parent.children[0].name == "child"


def test_span_records_exception() -> None:
    tracer = InMemoryTracer()

    try:
        with span(tracer, "failing"):
            raise ValueError("boom")
    except ValueError:
        pass

    assert tracer.spans[0].status == "error"
    assert tracer.spans[0].error is not None
