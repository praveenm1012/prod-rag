"""API model mapping helpers."""

from app.api.schemas import DocumentResponse
from app.rag.documents.models import DocumentRecord


def to_document_response(record: DocumentRecord) -> DocumentResponse:
    """Convert a document record to an API response model."""
    return DocumentResponse(
        document_id=record.document_id,
        filename=record.filename,
        source=record.source,
        chunk_count=record.chunk_count,
        uploaded_at=record.uploaded_at,
    )
