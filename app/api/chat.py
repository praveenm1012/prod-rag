"""Chat API endpoints."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import RagPipelineDep
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ProviderStatusResponse,
)
from app.rag.generation import ChatMessage, GenerationRequest
from app.rag.generation.config import get_generation_settings
from app.rag.prompts.models import ConversationTurn

router = APIRouter(prefix="/chat", tags=["chat"])


def _provider_configured() -> bool:
    settings = get_generation_settings()
    if settings.provider == "ollama":
        return True
    if settings.provider == "openai":
        return bool(settings.openai_api_key)
    if settings.provider == "claude":
        return bool(settings.anthropic_api_key)
    return False


@router.get(
    "/status",
    response_model=ProviderStatusResponse,
    responses={200: {"description": "Provider status returned"}},
)
async def provider_status() -> ProviderStatusResponse:
    """Return configured LLM provider details."""
    settings = get_generation_settings()
    return ProviderStatusResponse(
        provider=settings.provider,
        model=settings.model,
        configured=_provider_configured(),
    )


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        200: {"description": "Chat completion returned"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "LLM provider not configured"},
    },
)
async def chat(request: ChatRequest, pipeline: RagPipelineDep) -> ChatResponse:
    """Generate a chat response, optionally grounded in uploaded documents."""
    if not _provider_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured",
        )

    history = [
        ConversationTurn(role=message.role, content=message.content)
        for message in request.history
    ]

    if request.use_rag:
        answer = pipeline.ask(
            request.question,
            history=history,
            top_k=request.top_k,
        )
        citations = [citation.placeholder for citation in answer.prompt.citations]
        return ChatResponse(
            answer=answer.answer,
            model=answer.model,
            provider=answer.provider,
            usage=answer.usage,
            citations=citations,
        )

    response = pipeline.chat(request.question, history=history)
    return ChatResponse(
        answer=response.content,
        model=response.model,
        provider=response.provider,
        usage=response.usage,
    )


@router.post(
    "/stream",
    responses={
        200: {"description": "Streaming chat response"},
        503: {"model": ErrorResponse, "description": "LLM provider not configured"},
    },
)
async def chat_stream(
    request: ChatRequest,
    pipeline: RagPipelineDep,
) -> StreamingResponse:
    """Stream a chat response."""
    if not _provider_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider is not configured",
        )

    if request.use_rag:

        def rag_stream() -> Iterator[str]:
            for chunk in pipeline.stream(
                request.question,
                history=[
                    ConversationTurn(role=message.role, content=message.content)
                    for message in request.history
                ],
                top_k=request.top_k,
            ):
                if chunk.content:
                    yield chunk.content

        return StreamingResponse(rag_stream(), media_type="text/plain")

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content="You are a helpful assistant."),
    ]
    for message in request.history:
        messages.append(ChatMessage(role=message.role, content=message.content))
    messages.append(ChatMessage(role="user", content=request.question))

    def direct_stream() -> Iterator[str]:
        for chunk in pipeline.llm_provider.stream(
            GenerationRequest(messages=tuple(messages))
        ):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(direct_stream(), media_type="text/plain")
