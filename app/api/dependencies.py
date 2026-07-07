"""FastAPI dependency injection providers."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.rag.documents import DocumentService
from app.rag.embeddings import EmbeddingService
from app.rag.generation import create_llm_provider
from app.rag.pipeline import RagPipeline
from app.rag.reranking import CrossEncoderReranker


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return a cached embedding service."""
    return EmbeddingService()


@lru_cache
def get_document_service() -> DocumentService:
    """Return a cached document service."""
    return DocumentService(embedding_service=get_embedding_service())


def get_rag_pipeline(
    document_service: Annotated[DocumentService, Depends(get_document_service)],
) -> RagPipeline:
    """Return a RAG pipeline bound to the shared retriever."""
    return RagPipeline(
        retriever=document_service.retriever,
        reranker=CrossEncoderReranker(),
        llm_provider=create_llm_provider(),
    )


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
RagPipelineDep = Annotated[RagPipeline, Depends(get_rag_pipeline)]
