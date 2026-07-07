"""Document upload endpoint."""

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.dependencies import DocumentServiceDep
from app.api.mappers import to_document_response
from app.api.schemas import ErrorResponse, UploadResponse
from app.rag.documents.exceptions import (
    DocumentUploadError,
    UnsupportedUploadError,
)

router = APIRouter(tags=["documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Document uploaded and indexed"},
        400: {"model": ErrorResponse, "description": "Invalid upload"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        500: {"model": ErrorResponse, "description": "Upload processing failed"},
    },
)
async def upload_document(
    document_service: DocumentServiceDep,
    file: UploadFile = File(...),
) -> UploadResponse:
    """Upload and index a document for retrieval."""
    if file.filename is None or not file.filename.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="filename is required",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file is empty",
        )

    try:
        result = document_service.upload_bytes(file.filename, content)
    except UnsupportedUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc
    except DocumentUploadError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return UploadResponse(
        document=to_document_response(result.document),
        chunks_indexed=result.chunks_indexed,
    )
