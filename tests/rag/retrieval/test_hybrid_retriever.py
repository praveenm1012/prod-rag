"""Hybrid retrieval unit tests."""

import numpy as np
import pytest

from app.rag.lexical import LexicalDocument
from app.rag.retrieval import HybridRetriever, min_max_normalize
from app.rag.retrieval.exceptions import (
    HybridRetrievalConfigError,
    HybridRetrievalStateError,
)
from app.rag.vectorstore.models import MetadataFilter


def _document(
    chunk_id: str,
    text: str,
    *,
    document_id: str,
    vector: list[float],
) -> tuple[LexicalDocument, list[float]]:
    document = LexicalDocument(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        page_number=1,
        source=f"/docs/{document_id}.txt",
    )
    return document, vector


@pytest.fixture
def corpus() -> tuple[list[LexicalDocument], list[list[float]]]:
    documents_and_vectors = [
        _document(
            "lex:1",
            "BM25 lexical keyword search ranking function",
            document_id="lex",
            vector=[1.0, 0.0, 0.0],
        ),
        _document(
            "sem:1",
            "Semantic embeddings capture meaning for similarity retrieval",
            document_id="sem",
            vector=[0.0, 1.0, 0.0],
        ),
        _document(
            "hyb:1",
            "Hybrid retrieval combines lexical BM25 and vector search",
            document_id="hyb",
            vector=[0.7, 0.7, 0.0],
        ),
    ]
    documents = [item[0] for item in documents_and_vectors]
    vectors = [item[1] for item in documents_and_vectors]
    return documents, vectors


@pytest.fixture
def retriever(
    corpus: tuple[list[LexicalDocument], list[list[float]]],
) -> HybridRetriever:
    documents, vectors = corpus
    hybrid = HybridRetriever(bm25_weight=0.5, vector_weight=0.5)
    hybrid.index(documents, vectors)
    return hybrid


class TestScoreNormalization:
    def test_min_max_normalize_scales_to_unit_interval(self) -> None:
        normalized = min_max_normalize({"a": 1.0, "b": 3.0, "c": 5.0})

        assert normalized["a"] == pytest.approx(0.0)
        assert normalized["b"] == pytest.approx(0.5)
        assert normalized["c"] == pytest.approx(1.0)

    def test_min_max_normalize_handles_equal_scores(self) -> None:
        normalized = min_max_normalize({"a": 2.0, "b": 2.0})

        assert normalized["a"] == 1.0
        assert normalized["b"] == 1.0


class TestHybridRetriever:
    def test_requires_index_before_search(self) -> None:
        retriever = HybridRetriever()
        with pytest.raises(HybridRetrievalStateError, match="index must be built"):
            retriever.search("test", query_vector=[1.0, 0.0, 0.0])

    def test_rejects_invalid_weights(self) -> None:
        with pytest.raises(
            HybridRetrievalConfigError,
            match="at least one retrieval weight",
        ):
            HybridRetriever(bm25_weight=0.0, vector_weight=0.0)

    def test_lexical_query_ranks_bm25_hit(
        self,
        retriever: HybridRetriever,
    ) -> None:
        query_vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tolist()
        results = retriever.search(
            "BM25 lexical keyword search",
            top_k=2,
            query_vector=query_vector,
        )

        assert results[0].chunk_id == "lex:1"
        assert results[0].bm25_score > 0
        assert results[0].score > 0

    def test_semantic_query_ranks_vector_hit(
        self,
        retriever: HybridRetriever,
    ) -> None:
        query_vector = np.asarray([0.05, 0.99, 0.0], dtype=np.float32).tolist()
        results = retriever.search(
            "meaning similarity embeddings",
            top_k=2,
            query_vector=query_vector,
        )

        assert results[0].chunk_id == "sem:1"
        assert results[0].vector_score > 0

    def test_hybrid_beats_single_signal_for_mixed_query(
        self,
        retriever: HybridRetriever,
    ) -> None:
        query_vector = np.asarray([0.6, 0.6, 0.0], dtype=np.float32).tolist()
        results = retriever.search(
            "hybrid BM25 vector retrieval",
            top_k=1,
            query_vector=query_vector,
        )

        assert results[0].chunk_id == "hyb:1"
        assert results[0].bm25_score_normalized > 0
        assert results[0].vector_score_normalized > 0

    def test_configurable_weights_change_ranking(
        self,
        corpus: tuple[list[LexicalDocument], list[list[float]]],
    ) -> None:
        documents, vectors = corpus
        lexical_only = HybridRetriever(bm25_weight=1.0, vector_weight=0.0)
        vector_only = HybridRetriever(bm25_weight=0.0, vector_weight=1.0)
        lexical_only.index(documents, vectors)
        vector_only.index(documents, vectors)

        query_vector = np.asarray([0.05, 0.99, 0.0], dtype=np.float32).tolist()
        lexical_results = lexical_only.search(
            "BM25 lexical keyword search",
            top_k=1,
            query_vector=query_vector,
        )
        vector_results = vector_only.search(
            "semantic embeddings meaning",
            top_k=1,
            query_vector=query_vector,
        )

        assert lexical_results[0].chunk_id == "lex:1"
        assert vector_results[0].chunk_id == "sem:1"

    def test_metadata_filter_is_applied(
        self,
        retriever: HybridRetriever,
    ) -> None:
        query_vector = np.asarray([0.0, 1.0, 0.0], dtype=np.float32).tolist()
        results = retriever.search(
            "semantic embeddings meaning",
            top_k=5,
            filters=MetadataFilter(document_id="sem"),
            query_vector=query_vector,
        )

        assert results
        assert all(result.document_id == "sem" for result in results)

    def test_result_contains_component_scores(
        self,
        retriever: HybridRetriever,
    ) -> None:
        query_vector = np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tolist()
        result = retriever.search("BM25", top_k=1, query_vector=query_vector)[0]

        assert result.rank == 1
        assert 0.0 <= result.bm25_score_normalized <= 1.0
        assert 0.0 <= result.vector_score_normalized <= 1.0
        assert result.score <= 1.0
