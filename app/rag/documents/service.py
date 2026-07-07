"""Document upload, listing, and deletion service."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

from app.core.logging import get_logger
from app.rag.chunking import RecursiveCharacterTextSplitter
from app.rag.documents.exceptions import (
    DocumentNotFoundError,
    DocumentUploadError,
    UnsupportedUploadError,
)
from app.rag.documents.models import (
    DocumentRecord,
    ProcessingPhase,
    UploadResult,
    utc_now,
)
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
        self._lock = threading.Lock()

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    def list_documents(self) -> list[DocumentRecord]:
        """Return all uploaded documents sorted by upload time."""
        with self._lock:
            records = list(self._documents.values())
        return sorted(records, key=lambda record: record.uploaded_at, reverse=True)

    def get_document(self, document_id: str) -> DocumentRecord:
        """Return a document record or raise if it does not exist."""
        with self._lock:
            record = self._documents.get(document_id)
        if record is None:
            raise DocumentNotFoundError(f"Document not found: {document_id}")
        return record

    def get_upload_status(self, document_id: str) -> DocumentRecord:
        """Return the current upload/indexing status for a document."""
        return self.get_document(document_id)

    def upload_bytes(self, filename: str, content: bytes) -> UploadResult:
        """Save, ingest, chunk, embed, and index an uploaded file."""
        record = self.start_background_upload(filename, content)
        self.process_upload(record.document_id)
        return UploadResult(
            document=self.get_document(record.document_id),
            chunks_indexed=self.get_document(record.document_id).chunk_count,
        )

    def start_background_upload(self, filename: str, content: bytes) -> DocumentRecord:
        """Persist an upload and return immediately while indexing continues."""
        if not filename.strip():
            raise DocumentUploadError("filename must not be empty")
        if not content.strip():
            raise DocumentUploadError("uploaded file is empty")

        self._upload_dir.mkdir(parents=True, exist_ok=True)
        document_id = uuid.uuid4().hex
        destination = self._upload_dir / f"{document_id}_{Path(filename).name}"
        destination.write_bytes(content)

        record = DocumentRecord(
            document_id=document_id,
            filename=Path(filename).name,
            source=str(destination),
            chunk_count=0,
            uploaded_at=utc_now(),
            status="processing",
            phase="queued",
            file_size_bytes=len(content),
        )
        with self._lock:
            self._documents[document_id] = record

        logger.info(
            "document_upload_queued",
            document_id=document_id,
            filename=record.filename,
            file_size_bytes=record.file_size_bytes,
        )
        return record

    def process_upload(self, document_id: str) -> None:
        """Ingest, chunk, embed, and index a queued upload."""
        record = self.get_document(document_id)
        destination = Path(record.source)

        try:
            self._update_record(document_id, phase="ingesting")
            document = load_document(destination)

            self._update_record(document_id, phase="chunking")
            chunks = self._splitter.split_document(document)
            if not chunks:
                raise DocumentUploadError("upload produced no indexable chunks")

            lexical_documents = [LexicalDocument.from_chunk(chunk) for chunk in chunks]

            self._update_record(document_id, phase="embedding")
            vectors = self._embed_chunks([item.text for item in lexical_documents])

            self._update_record(document_id, phase="indexing")
            with self._lock:
                current = self._documents.get(document_id)
                if current is None:
                    raise DocumentNotFoundError(f"Document not found: {document_id}")
                if current.status == "failed":
                    return
                self._lexical_documents.extend(lexical_documents)
                self._vectors.extend(vectors)
                self._reindex()
                self._documents[document_id] = replace(
                    current,
                    status="indexed",
                    phase=None,
                    chunk_count=len(chunks),
                    error_message=None,
                )

            logger.info(
                "document_uploaded",
                document_id=document_id,
                filename=record.filename,
                chunk_count=len(chunks),
            )
        except IngestionError as exc:
            destination.unlink(missing_ok=True)
            self._mark_failed(document_id, str(exc))
            raise UnsupportedUploadError(str(exc)) from exc
        except Exception as exc:
            destination.unlink(missing_ok=True)
            self._mark_failed(document_id, str(exc))
            if isinstance(exc, DocumentUploadError):
                raise
            raise DocumentUploadError(
                f"Failed to process upload: {record.filename}"
            ) from exc

    def delete_document(self, document_id: str) -> DocumentRecord:
        """Delete a document and its chunks from the index."""
        with self._lock:
            record = self._documents.get(document_id)
            if record is None:
                raise DocumentNotFoundError(f"Document not found: {document_id}")

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
            if record.status == "indexed":
                self._reindex()

        Path(record.source).unlink(missing_ok=True)
        logger.info("document_deleted", document_id=document_id)
        return record

    def _update_record(
        self,
        document_id: str,
        *,
        phase: ProcessingPhase | None = None,
    ) -> None:
        with self._lock:
            record = self._documents.get(document_id)
            if record is None:
                raise DocumentNotFoundError(f"Document not found: {document_id}")
            self._documents[document_id] = replace(record, phase=phase)

    def _mark_failed(self, document_id: str, error_message: str) -> None:
        with self._lock:
            record = self._documents.get(document_id)
            if record is None:
                return
            self._documents[document_id] = replace(
                record,
                status="failed",
                phase=None,
                error_message=error_message,
            )
        logger.error(
            "document_upload_failed",
            document_id=document_id,
            error=error_message,
        )

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
