"""API model mapping helpers."""

from app.api.schemas import DocumentResponse, UploadStatusResponse
from app.rag.documents.models import DocumentRecord


def to_document_response(record: DocumentRecord) -> DocumentResponse:
    """Convert a document record to an API response model."""
    return DocumentResponse(
        document_id=record.document_id,
        filename=record.filename,
        source=record.source,
        chunk_count=record.chunk_count,
        uploaded_at=record.uploaded_at,
        status=record.status,
        phase=record.phase,
        error_message=record.error_message,
        file_size_bytes=record.file_size_bytes,
    )


def to_upload_status_response(record: DocumentRecord) -> UploadStatusResponse:
    """Convert a document record to an upload status response."""
    return UploadStatusResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status,
        phase=record.phase,
        chunk_count=record.chunk_count,
        file_size_bytes=record.file_size_bytes,
        error_message=record.error_message,
        uploaded_at=record.uploaded_at,
    )
