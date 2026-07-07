"""Document listing endpoint."""

from fastapi import APIRouter

from app.api.dependencies import DocumentServiceDep
from app.api.mappers import to_document_response
from app.api.schemas import DocumentListResponse

router = APIRouter(tags=["documents"])


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    responses={200: {"description": "Document list returned"}},
)
async def list_documents(
    document_service: DocumentServiceDep,
) -> DocumentListResponse:
    """List uploaded documents."""
    documents = document_service.list_documents()
    return DocumentListResponse(
        documents=[to_document_response(record) for record in documents],
        total=len(documents),
    )
