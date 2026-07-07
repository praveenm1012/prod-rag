"""Chat and RAG API endpoints."""

from collections.abc import Iterator
from typing import Literal

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag.generation import create_llm_provider
from app.rag.generation.config import get_generation_settings
from app.rag.pipeline import RagPipeline
from app.rag.prompts.models import ConversationTurn

router = APIRouter(prefix="/chat", tags=["chat"])

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    """Return a cached RAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline(llm_provider=create_llm_provider())
    return _pipeline


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    history: list[HistoryMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    model: str
    provider: str
    usage: dict[str, int] = Field(default_factory=dict)


class ProviderStatusResponse(BaseModel):
    provider: str
    model: str
    configured: bool


@router.get("/status")
async def provider_status() -> ProviderStatusResponse:
    """Return configured LLM provider details."""
    settings = get_generation_settings()
    configured = (
        settings.provider == "ollama"
        or (settings.provider == "openai" and bool(settings.openai_api_key))
        or (settings.provider == "claude" and bool(settings.anthropic_api_key))
    )
    return ProviderStatusResponse(
        provider=settings.provider,
        model=settings.model,
        configured=configured,
    )


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Generate a direct LLM response without retrieval."""
    history = [
        ConversationTurn(role=message.role, content=message.content)
        for message in request.history
    ]
    pipeline = get_pipeline()
    response = pipeline.chat(request.question, history=history)
    return ChatResponse(
        answer=response.content,
        model=response.model,
        provider=response.provider,
        usage=response.usage,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream a direct LLM response without retrieval."""
    from app.rag.generation import ChatMessage, GenerationRequest

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content="You are a helpful assistant."),
    ]
    for message in request.history:
        messages.append(ChatMessage(role=message.role, content=message.content))
    messages.append(ChatMessage(role="user", content=request.question))

    pipeline = get_pipeline()

    def event_stream() -> Iterator[str]:
        for chunk in pipeline.llm_provider.stream(
            GenerationRequest(messages=tuple(messages))
        ):
            if chunk.content:
                yield chunk.content

    return StreamingResponse(event_stream(), media_type="text/plain")
