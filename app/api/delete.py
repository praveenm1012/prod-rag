"""Document deletion endpoint."""

from fastapi import APIRouter, HTTPException, Response, status

from app.api.dependencies import DocumentServiceDep
from app.api.schemas import DeleteRequest, ErrorResponse
from app.rag.documents.exceptions import DocumentNotFoundError

router = APIRouter(tags=["documents"])


@router.delete(
    "/delete",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Document deleted"},
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
)
async def delete_document(
    request: DeleteRequest,
    document_service: DocumentServiceDep,
) -> Response:
    """Delete a document and remove its chunks from the index."""
    try:
        document_service.delete_document(request.document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)
