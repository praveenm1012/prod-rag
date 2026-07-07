"""BM25 lexical search unit tests."""

import pytest

from app.rag.chunking.models import Chunk
from app.rag.lexical import (
    BM25Searcher,
    LexicalDocument,
    LexicalSearchConfigError,
    LexicalSearchStateError,
)
from app.rag.vectorstore.models import MetadataFilter


@pytest.fixture
def documents() -> list[LexicalDocument]:
    return [
        LexicalDocument(
            chunk_id="doc-a:1:0",
            text="Retrieval augmented generation combines search and language models.",
            document_id="doc-a",
            page_number=1,
            source="/docs/rag.txt",
        ),
        LexicalDocument(
            chunk_id="doc-a:1:1",
            text="Vector databases store embeddings for semantic search.",
            document_id="doc-a",
            page_number=1,
            source="/docs/rag.txt",
        ),
        LexicalDocument(
            chunk_id="doc-b:1:0",
            text="BM25 is a lexical ranking function for keyword search.",
            document_id="doc-b",
            page_number=1,
            source="/docs/bm25.txt",
        ),
    ]


@pytest.fixture
def searcher(documents: list[LexicalDocument]) -> BM25Searcher:
    service = BM25Searcher(top_k=3)
    service.index(documents)
    return service


class TestBM25SearcherConfig:
    def test_rejects_invalid_k1(self) -> None:
        with pytest.raises(LexicalSearchConfigError, match="k1 must be greater"):
            BM25Searcher(k1=0)

    def test_rejects_invalid_b(self) -> None:
        with pytest.raises(LexicalSearchConfigError, match="b must be between"):
            BM25Searcher(b=1.5)


class TestBM25Indexer:
    def test_index_sets_document_count(self, documents: list[LexicalDocument]) -> None:
        searcher = BM25Searcher()
        count = searcher.index(documents)
        assert count == 3
        assert searcher.document_count == 3

    def test_search_before_index_raises(self) -> None:
        searcher = BM25Searcher()
        with pytest.raises(LexicalSearchStateError, match="index must be built"):
            searcher.search("retrieval")


class TestBM25Search:
    def test_returns_top_k_results(self, searcher: BM25Searcher) -> None:
        results = searcher.search("search", top_k=2)

        assert 1 <= len(results) <= 2
        assert results[0].rank == 1
        if len(results) > 1:
            assert results[1].rank == 2

    def test_results_include_metadata_and_score(self, searcher: BM25Searcher) -> None:
        results = searcher.search("BM25 keyword search")

        assert results
        top = results[0]
        assert top.chunk_id == "doc-b:1:0"
        assert top.document_id == "doc-b"
        assert top.source == "/docs/bm25.txt"
        assert top.page_number == 1
        assert isinstance(top.score, float)
        assert top.score > 0

    def test_results_are_ranked_by_descending_score(
        self,
        searcher: BM25Searcher,
    ) -> None:
        results = searcher.search("search embeddings vector", top_k=3)

        assert len(results) >= 2
        assert results[0].score >= results[1].score

    def test_metadata_filter_limits_results(self, searcher: BM25Searcher) -> None:
        results = searcher.search(
            "search",
            top_k=5,
            filters=MetadataFilter(document_id="doc-b"),
        )

        assert results
        assert all(result.document_id == "doc-b" for result in results)

    def test_page_number_metadata_filter(self, searcher: BM25Searcher) -> None:
        results = searcher.search(
            "search",
            filters=MetadataFilter(page_number=1, source="/docs/rag.txt"),
        )

        assert results
        assert all(result.source == "/docs/rag.txt" for result in results)

    def test_empty_query_returns_no_results(self, searcher: BM25Searcher) -> None:
        assert searcher.search("   ") == []

    def test_from_chunk_helper(self) -> None:
        chunk = Chunk(
            text="Chunk text",
            chunk_id="doc:1:0",
            document_id="doc",
            page_number=1,
            source="/doc.txt",
            metadata={"chunk_index": 0},
        )
        document = LexicalDocument.from_chunk(chunk)

        assert document.chunk_id == "doc:1:0"
        assert document.metadata["chunk_index"] == 0
