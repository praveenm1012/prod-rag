"""Document service unit tests."""

import pytest

from app.rag.documents import DocumentNotFoundError, DocumentService
from tests.api.conftest import FakeEmbeddingService


def test_upload_and_delete_document(tmp_path) -> None:
    service = DocumentService(
        embedding_service=FakeEmbeddingService(),
        upload_dir=tmp_path / "uploads",
    )

    result = service.upload_bytes("notes.txt", b"Python is a programming language.")
    assert result.chunks_indexed >= 1
    assert len(service.list_documents()) == 1

    deleted = service.delete_document(result.document.document_id)
    assert deleted.document_id == result.document.document_id
    assert service.list_documents() == []


def test_delete_missing_document_raises() -> None:
    service = DocumentService(embedding_service=FakeEmbeddingService())
    with pytest.raises(DocumentNotFoundError):
        service.delete_document("missing")
