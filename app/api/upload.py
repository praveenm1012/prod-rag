"""Document upload endpoint."""

from __future__ import annotations

import asyncio

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import JSONResponse

from app.api.dependencies import DocumentServiceDep
from app.api.mappers import to_document_response, to_upload_status_response
from app.api.schemas import (
    ErrorResponse,
    UploadAcceptedResponse,
    UploadResponse,
    UploadStatusResponse,
)
from app.rag.documents.exceptions import (
    DocumentNotFoundError,
    DocumentUploadError,
    UnsupportedUploadError,
)

router = APIRouter(tags=["documents"])


async def _process_upload_in_background(
    document_service: DocumentServiceDep,
    document_id: str,
) -> None:
    await asyncio.to_thread(document_service.process_upload, document_id)


@router.get(
    "/upload/{document_id}/status",
    response_model=UploadStatusResponse,
    responses={
        200: {"description": "Upload status returned"},
        404: {"model": ErrorResponse, "description": "Document not found"},
    },
)
async def upload_status(
    document_id: str,
    document_service: DocumentServiceDep,
) -> UploadStatusResponse:
    """Return background upload and indexing progress."""
    try:
        record = document_service.get_upload_status(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return to_upload_status_response(record)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Document uploaded and indexed"},
        202: {"model": UploadAcceptedResponse, "description": "Upload accepted"},
        400: {"model": ErrorResponse, "description": "Invalid upload"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        500: {"model": ErrorResponse, "description": "Upload processing failed"},
    },
)
async def upload_document(
    document_service: DocumentServiceDep,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    background: bool = Query(
        default=False,
        description="Process ingest/embed/index in the background and return 202",
    ),
) -> UploadResponse | JSONResponse:
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
        if background:
            record = document_service.start_background_upload(file.filename, content)
            background_tasks.add_task(
                _process_upload_in_background,
                document_service,
                record.document_id,
            )
            payload = UploadAcceptedResponse(
                document=to_document_response(record),
                message=(
                    "Upload accepted. Embedding and indexing are running in the "
                    "background. Poll /api/v1/upload/{document_id}/status for progress."
                ),
            )
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=payload.model_dump(mode="json"),
            )

        result = await asyncio.to_thread(
            document_service.upload_bytes,
            file.filename,
            content,
        )
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
