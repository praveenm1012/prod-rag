"""RAG pipeline observability tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.observability.init import get_memory_tracer, setup_observability
from app.rag.generation.models import GenerationResponse
from app.rag.pipeline import RagPipeline
from app.rag.retrieval import HybridRetriever


@dataclass
class FakeEmbeddingRecord:
    vector: list[float]


@dataclass
class FakeEmbeddingBatch:
    records: list[FakeEmbeddingRecord]


class FakeEmbeddingService:
    def embed_text(
        self,
        text: str,
        prompt_type: str = "document",
    ) -> FakeEmbeddingRecord:
        return FakeEmbeddingRecord(vector=[1.0, 0.1, 0.1, 0.1])

    def embed_texts(self, texts: list[str]) -> FakeEmbeddingBatch:
        return FakeEmbeddingBatch(
            records=[FakeEmbeddingRecord(vector=[1.0, 0.1, 0.1, 0.1]) for _ in texts]
        )


class FakeLLMProvider:
    def complete(self, request) -> GenerationResponse:  # noqa: ANN001
        return GenerationResponse(
            content="Observed answer.",
            model="gpt-4o-mini",
            provider="openai",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


@pytest.fixture
def pipeline() -> RagPipeline:
    setup_observability()
    return RagPipeline(
        retriever=HybridRetriever(embedding_service=FakeEmbeddingService()),
        llm_provider=FakeLLMProvider(),
        use_reranker=False,
    )


def _span_names(tracer: object) -> list[str]:
    from app.observability.tracing import InMemoryTracer

    assert isinstance(tracer, InMemoryTracer)
    names: list[str] = []

    def walk(span: object) -> None:
        from app.observability.tracing import SpanRecord

        assert isinstance(span, SpanRecord)
        names.append(span.name)
        for child in span.children:
            walk(child)

    for record in tracer.spans:
        walk(record)
    return names


def test_pipeline_ask_emits_nested_spans(
    pipeline: RagPipeline,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "_retrieve", lambda question, top_k=None: [])
    monkeypatch.setattr(pipeline, "_rerank", lambda question, results: [])

    pipeline.ask("What is observability?")

    tracer = get_memory_tracer()
    span_names = _span_names(tracer)

    assert "rag.pipeline.ask" in span_names
    assert "rag.retrieve" in span_names
    assert "rag.rerank" in span_names
    assert "rag.prompt.build" in span_names
    assert "rag.generate" in span_names
    assert "llm.generation.cost" in span_names
