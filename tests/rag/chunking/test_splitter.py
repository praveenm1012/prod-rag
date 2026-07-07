"""Chunking splitter tests."""

import pytest

from app.rag.chunking import InvalidChunkConfigError, RecursiveCharacterTextSplitter
from app.rag.ingestion.models import Document


@pytest.fixture
def splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)


class TestRecursiveCharacterTextSplitterConfig:
    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(self) -> None:
        with pytest.raises(InvalidChunkConfigError, match="chunk_overlap must be less"):
            RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)

    def test_rejects_non_positive_chunk_size(self) -> None:
        with pytest.raises(InvalidChunkConfigError, match="chunk_size must be greater"):
            RecursiveCharacterTextSplitter(chunk_size=0, chunk_overlap=0)


class TestSplitText:
    def test_splits_on_paragraph_boundaries(
        self,
        splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        text = (
            "Paragraph one.\n\nParagraph two is a bit longer than the first.\n\nThird."
        )
        chunks = splitter.split_text(text)

        assert chunks
        assert all(len(chunk) <= splitter.chunk_size for chunk in chunks)

    def test_recursive_split_on_long_unbroken_text(self) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=5)
        text = "A" * 55
        chunks = splitter.split_text(text)

        assert len(chunks) >= 3
        assert all(len(chunk) <= 20 for chunk in chunks)

    def test_overlap_is_applied_between_chunks(self) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=10)
        text = "word " * 40
        chunks = splitter.split_text(text)

        assert len(chunks) > 1
        assert chunks[0][-10:] in chunks[1]

    def test_empty_text_returns_no_chunks(
        self,
        splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []


class TestSplitDocument:
    def test_chunk_contains_required_fields(
        self,
        splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        document = Document(
            text="Section one.\n\nSection two with more detail.\n\nSection three.",
            metadata={"source": "/tmp/report.txt", "document_id": "doc-123"},
        )

        chunks = splitter.split_document(document)

        assert chunks
        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.document_id == "doc-123"
            assert chunk.page_number == 1
            assert chunk.source == "/tmp/report.txt"
            assert chunk.text

    def test_document_id_is_derived_from_source_when_missing(
        self,
        splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        document = Document(
            text="Some content for hashing.",
            metadata={"source": "/tmp/source-a.txt"},
        )
        chunks = splitter.split_document(document)

        assert chunks
        assert chunks[0].document_id
        assert chunks[0].document_id != "doc-123"

    def test_page_numbers_assigned_from_page_texts(
        self,
        splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        document = Document(
            text="ignored combined text",
            metadata={
                "source": "/tmp/book.pdf",
                "document_id": "book-1",
                "page_texts": [
                    "Page one " * 20,
                    "Page two " * 20,
                ],
            },
        )

        chunks = splitter.split_document(document)

        page_numbers = {chunk.page_number for chunk in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert all(chunk.document_id == "book-1" for chunk in chunks)
        assert all(chunk.source == "/tmp/book.pdf" for chunk in chunks)

    def test_chunk_ids_are_unique_per_document(
        self,
        splitter: RecursiveCharacterTextSplitter,
    ) -> None:
        document = Document(
            text="Chunk " * 100,
            metadata={"source": "/tmp/long.txt", "document_id": "doc-xyz"},
        )
        chunks = splitter.split_document(document)

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))
