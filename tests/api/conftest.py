"""API test fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_document_service,
    get_embedding_service,
    get_rag_pipeline,
)
from app.main import create_app
from app.rag.documents import DocumentService
from app.rag.generation.models import GenerationResponse, StreamChunk
from app.rag.pipeline import RagPipeline
from app.rag.retrieval import HybridRetriever

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class FakeEmbeddingRecord:
    vector: list[float]


@dataclass
class FakeEmbeddingBatch:
    records: list[FakeEmbeddingRecord]


class FakeEmbeddingService:
    """Deterministic embedding service for API tests."""

    def embed_text(
        self,
        text: str,
        prompt_type: str = "document",
    ) -> FakeEmbeddingRecord:
        return FakeEmbeddingRecord(vector=_vector_for_text(text))

    def embed_texts(self, texts: list[str]) -> FakeEmbeddingBatch:
        return FakeEmbeddingBatch(
            records=[
                FakeEmbeddingRecord(vector=_vector_for_text(text)) for text in texts
            ]
        )


class FakeLLMProvider:
    def complete(self, request) -> GenerationResponse:  # noqa: ANN001
        return GenerationResponse(
            content="Test answer from LLM.",
            model="test-model",
            provider="openai",
            finish_reason="stop",
            usage={"total_tokens": 12},
        )

    def stream(self, request) -> Iterator[StreamChunk]:  # noqa: ANN001
        yield StreamChunk(content="Test ")
        yield StreamChunk(content="answer.")
        yield StreamChunk(content="", is_final=True, finish_reason="stop")


def _vector_for_text(text: str, size: int = 16) -> list[float]:
    base = float((len(text) % 7) + 1)
    return [base if index == 0 else 0.1 for index in range(size)]


@pytest.fixture
def fake_embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def document_service(
    fake_embedding_service: FakeEmbeddingService,
    tmp_path,
) -> DocumentService:
    upload_dir = tmp_path / "uploads"
    retriever = HybridRetriever(embedding_service=fake_embedding_service)
    return DocumentService(
        retriever=retriever,
        embedding_service=fake_embedding_service,
        upload_dir=upload_dir,
    )


@pytest.fixture
def rag_pipeline(document_service: DocumentService) -> RagPipeline:
    return RagPipeline(
        retriever=document_service.retriever,
        llm_provider=FakeLLMProvider(),
        use_reranker=False,
    )


@pytest.fixture
def client(
    document_service: DocumentService,
    rag_pipeline: RagPipeline,
) -> Iterator[TestClient]:
    get_embedding_service.cache_clear()
    get_document_service.cache_clear()

    application = create_app()
    application.dependency_overrides[get_document_service] = lambda: document_service
    application.dependency_overrides[get_rag_pipeline] = lambda: rag_pipeline

    with TestClient(application) as test_client:
        yield test_client

    application.dependency_overrides.clear()
    get_embedding_service.cache_clear()
    get_document_service.cache_clear()
