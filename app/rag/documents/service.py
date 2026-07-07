"""Document upload, listing, and deletion service."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from app.core.logging import get_logger
from app.rag.chunking import RecursiveCharacterTextSplitter
from app.rag.documents.exceptions import (
    DocumentNotFoundError,
    DocumentUploadError,
    UnsupportedUploadError,
)
from app.rag.documents.models import DocumentRecord, UploadResult, utc_now
from app.rag.embeddings import EmbeddingService
from app.rag.ingestion import load_document
from app.rag.ingestion.exceptions import IngestionError
from app.rag.lexical.models import LexicalDocument
from app.rag.retrieval import HybridRetriever

logger = get_logger(__name__)


class EmbeddingLike(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> Any:
        """Embed texts and return a batch object with records."""


class DocumentService:
    """Manage document uploads and indexing for retrieval."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        embedding_service: EmbeddingLike | None = None,
        splitter: RecursiveCharacterTextSplitter | None = None,
        upload_dir: Path | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._splitter = splitter or RecursiveCharacterTextSplitter()
        self._upload_dir = upload_dir or Path(".cache/uploads")
        self._retriever = retriever or HybridRetriever(
            embedding_service=cast(EmbeddingService | None, embedding_service),
        )
        self._documents: dict[str, DocumentRecord] = {}
        self._lexical_documents: list[LexicalDocument] = []
        self._vectors: list[list[float]] = []

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    def list_documents(self) -> list[DocumentRecord]:
        """Return all uploaded documents sorted by upload time."""
        return sorted(
            self._documents.values(),
            key=lambda record: record.uploaded_at,
            reverse=True,
        )

    def get_document(self, document_id: str) -> DocumentRecord:
        """Return a document record or raise if it does not exist."""
        record = self._documents.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")
        return record

    def upload_bytes(self, filename: str, content: bytes) -> UploadResult:
        """Save, ingest, chunk, embed, and index an uploaded file."""
        if not filename.strip():
            raise DocumentUploadError("filename must not be empty")
        if not content.strip():
            raise DocumentUploadError("uploaded file is empty")

        self._upload_dir.mkdir(parents=True, exist_ok=True)
        document_id = uuid.uuid4().hex
        destination = self._upload_dir / f"{document_id}_{Path(filename).name}"

        try:
            destination.write_bytes(content)
            document = load_document(destination)
        except IngestionError as exc:
            destination.unlink(missing_ok=True)
            raise UnsupportedUploadError(str(exc)) from exc
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise DocumentUploadError(f"Failed to process upload: {filename}") from exc

        chunks = self._splitter.split_document(document)
        if not chunks:
            destination.unlink(missing_ok=True)
            raise DocumentUploadError("upload produced no indexable chunks")

        lexical_documents = [LexicalDocument.from_chunk(chunk) for chunk in chunks]
        vectors = self._embed_chunks([item.text for item in lexical_documents])

        self._lexical_documents.extend(lexical_documents)
        self._vectors.extend(vectors)
        self._reindex()

        record = DocumentRecord(
            document_id=document_id,
            filename=Path(filename).name,
            source=str(destination),
            chunk_count=len(chunks),
            uploaded_at=utc_now(),
        )
        self._documents[document_id] = record

        logger.info(
            "document_uploaded",
            document_id=document_id,
            filename=record.filename,
            chunk_count=record.chunk_count,
        )
        return UploadResult(document=record, chunks_indexed=len(chunks))

    def delete_document(self, document_id: str) -> DocumentRecord:
        """Delete a document and its chunks from the index."""
        record = self.get_document(document_id)

        remaining_documents: list[LexicalDocument] = []
        remaining_vectors: list[list[float]] = []
        for lexical_document, vector in zip(
            self._lexical_documents,
            self._vectors,
            strict=True,
        ):
            if lexical_document.document_id != document_id:
                remaining_documents.append(lexical_document)
                remaining_vectors.append(vector)

        self._lexical_documents = remaining_documents
        self._vectors = remaining_vectors
        del self._documents[document_id]
        self._reindex()

        Path(record.source).unlink(missing_ok=True)
        logger.info("document_deleted", document_id=document_id)
        return record

    def _embed_chunks(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_service is None:
            raise DocumentUploadError(
                "embedding service is required for document indexing"
            )
        batch = self._embedding_service.embed_texts(texts)
        records = batch.records
        vectors = [record.vector for record in records]
        if len(vectors) != len(texts):
            raise DocumentUploadError(
                "embedding service returned unexpected batch size"
            )
        return vectors

    def _reindex(self) -> None:
        if self._lexical_documents:
            self._retriever.index(self._lexical_documents, self._vectors)
        else:
            self._retriever.index([], [])
