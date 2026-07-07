"""Integration tests for the embedding service."""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.rag.embeddings import EmbeddingService
from app.rag.embeddings.config import EmbeddingSettings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_EMBEDDING_INTEGRATION", "0") != "1",
        reason="Set RUN_EMBEDDING_INTEGRATION=1 to run embedding integration tests",
    ),
]

sentence_transformers = pytest.importorskip("sentence_transformers")


@pytest.fixture(scope="module")
def embedding_service(tmp_path_factory: pytest.TempPathFactory) -> EmbeddingService:
    cache_dir = tmp_path_factory.mktemp("embedding-cache")
    settings = EmbeddingSettings(
        EMBEDDING_MODEL="BAAI/bge-large-en-v1.5",
        EMBEDDING_DEVICE="auto",
        EMBEDDING_BATCH_SIZE=4,
        EMBEDDING_CACHE_DIR=cache_dir,
        EMBEDDING_SHOW_PROGRESS=False,
    )
    return EmbeddingService(settings=settings)


def test_embed_single_text(embedding_service: EmbeddingService) -> None:
    text = "The quick brown fox jumps over the lazy dog."
    result = embedding_service.embed_text(text)

    assert result.vector
    assert len(result.vector) == embedding_service.embedding_dimension
    assert result.model_name == "BAAI/bge-large-en-v1.5"
    assert result.cached is False


def test_embed_batch_returns_aligned_records(
    embedding_service: EmbeddingService,
) -> None:
    texts = [
        "First document about databases.",
        "Second document about retrieval systems.",
        "Third document about vector search.",
    ]
    batch = embedding_service.embed_texts(texts)

    assert len(batch.records) == 3
    assert batch.cache_misses == 3
    assert batch.cache_hits == 0
    assert [record.text for record in batch.records] == texts
    dimension = embedding_service.embedding_dimension
    assert all(len(record.vector) == dimension for record in batch.records)


def test_cache_hit_on_second_request(embedding_service: EmbeddingService) -> None:
    text = "Caching should avoid recomputation."

    first = embedding_service.embed_text(text)
    second = embedding_service.embed_text(text)

    assert first.cached is False
    assert second.cached is True
    assert first.vector == second.vector


def test_query_and_document_prompts_differ(embedding_service: EmbeddingService) -> None:
    text = "What is retrieval augmented generation?"

    document = embedding_service.embed_text(text, prompt_type="document")
    query = embedding_service.embed_text(text, prompt_type="query")

    assert document.vector != query.vector


def test_batch_embeddings_are_normalized(embedding_service: EmbeddingService) -> None:
    batch = embedding_service.embed_texts(
        ["Normalization check.", "Another sentence."],
        prompt_type="document",
    )

    for record in batch.records:
        norm = float(np.linalg.norm(np.asarray(record.vector, dtype=np.float32)))
        assert norm == pytest.approx(1.0, rel=1e-4, abs=1e-4)


def test_clear_cache_forces_recompute(embedding_service: EmbeddingService) -> None:
    text = "Clear cache test sentence."

    first = embedding_service.embed_text(text)
    cached = embedding_service.embed_text(text)
    embedding_service.clear_cache()
    recomputed = embedding_service.embed_text(text)

    assert cached.cached is True
    assert recomputed.cached is False
    assert first.vector == recomputed.vector
