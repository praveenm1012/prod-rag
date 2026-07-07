"""Document deletion endpoint."""

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import DocumentServiceDep
from app.api.schemas import DeleteRequest, DeleteResponse, ErrorResponse
from app.rag.documents.exceptions import DocumentNotFoundError

router = APIRouter(tags=["documents"])


@router.delete(
    "/delete",
    response_model=DeleteResponse,
    responses={
        200: {"description": "Document deleted"},
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
)
async def delete_document(
    request: DeleteRequest,
    document_service: DocumentServiceDep,
) -> DeleteResponse:
    """Delete a document and remove its chunks from the index."""
    try:
        record = document_service.delete_document(request.document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return DeleteResponse(
        document_id=record.document_id,
        filename=record.filename,
    )
