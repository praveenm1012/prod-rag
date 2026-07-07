"""Shared API request and response schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)
    use_rag: bool = False
    top_k: int | None = Field(default=None, ge=1, le=50)


class ChatResponse(BaseModel):
    answer: str
    model: str
    provider: str
    usage: dict[str, int] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)


class ProviderStatusResponse(BaseModel):
    provider: str
    model: str
    configured: bool


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    source: str
    chunk_count: int
    uploaded_at: datetime
    status: Literal["processing", "indexed", "failed"] = "indexed"
    phase: str | None = None
    error_message: str | None = None
    file_size_bytes: int = 0


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class UploadResponse(BaseModel):
    document: DocumentResponse
    chunks_indexed: int


class UploadAcceptedResponse(BaseModel):
    document: DocumentResponse
    status: Literal["processing"] = "processing"
    message: str


class UploadStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: Literal["processing", "indexed", "failed"]
    phase: str | None = None
    chunk_count: int
    file_size_bytes: int
    error_message: str | None = None
    uploaded_at: datetime


class DeleteRequest(BaseModel):
    document_id: str = Field(min_length=1)


class DeleteResponse(BaseModel):
    document_id: str
    filename: str
    deleted: bool = True


class ErrorResponse(BaseModel):
    detail: str
