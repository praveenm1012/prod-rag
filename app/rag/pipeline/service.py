"""RAG pipeline orchestration."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.observability.costs import get_cost_tracker
from app.observability.init import get_tracer
from app.observability.tracing import span
from app.rag.generation import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    StreamChunk,
    create_llm_provider,
    messages_from_prompt,
)
from app.rag.prompts import BuiltPrompt, PromptBuilder
from app.rag.prompts.models import ConversationTurn, FewShotExample
from app.rag.reranking import CrossEncoderReranker, RerankCandidate, RerankResult
from app.rag.retrieval import HybridRetriever, HybridSearchResult

logger = get_logger(__name__)


@dataclass(frozen=True)
class RagAnswer:
    """A completed RAG answer."""

    question: str
    answer: str
    prompt: BuiltPrompt
    retrieval_results: tuple[HybridSearchResult, ...] = field(default_factory=tuple)
    rerank_results: tuple[RerankResult, ...] = field(default_factory=tuple)
    model: str = ""
    provider: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class RagPipeline:
    """Orchestrate retrieval, reranking, prompt building, and generation."""

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        prompt_builder: PromptBuilder | None = None,
        llm_provider: LLMProvider | None = None,
        *,
        use_reranker: bool = True,
    ) -> None:
        self._retriever = retriever or HybridRetriever()
        self._reranker = reranker or CrossEncoderReranker()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._llm = llm_provider or create_llm_provider()
        self._use_reranker = use_reranker

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    @property
    def llm_provider(self) -> LLMProvider:
        return self._llm

    def ask(
        self,
        question: str,
        *,
        history: Sequence[ConversationTurn] | None = None,
        few_shot_examples: Sequence[FewShotExample] | None = None,
        top_k: int | None = None,
    ) -> RagAnswer:
        """Run the full RAG pipeline and return a completed answer."""
        tracer = get_tracer()
        with span(
            tracer,
            "rag.pipeline.ask",
            attributes={"top_k": top_k},
        ) as root_span:
            root_span.set_input({"question": question, "top_k": top_k})

            with span(tracer, "rag.retrieve") as retrieve_span:
                retrieval_results = self._retrieve(question, top_k=top_k)
                retrieve_span.set_output({"result_count": len(retrieval_results)})

            with span(tracer, "rag.rerank") as rerank_span:
                rerank_results = self._rerank(question, retrieval_results)
                rerank_span.set_output({"result_count": len(rerank_results)})

            with span(tracer, "rag.prompt.build") as prompt_span:
                prompt = self._prompt_builder.build(
                    question=question,
                    context=list(rerank_results),
                    history=history,
                    few_shot_examples=few_shot_examples,
                )
                prompt_span.set_output(
                    {
                        "citation_count": len(prompt.citations),
                        "token_count": prompt.token_count,
                    }
                )

            with span(tracer, "rag.generate", kind="generation") as generate_span:
                generation = self._generate(prompt)
                generate_span.set_output(
                    {
                        "model": generation.model,
                        "provider": generation.provider,
                        "usage": generation.usage,
                    }
                )

            if generation.usage:
                get_cost_tracker().record_generation(
                    model=generation.model,
                    provider=generation.provider,
                    usage=generation.usage,
                )

            answer = RagAnswer(
                question=question,
                answer=generation.content,
                prompt=prompt,
                retrieval_results=tuple(retrieval_results),
                rerank_results=tuple(rerank_results),
                model=generation.model,
                provider=generation.provider,
                usage=generation.usage,
            )
            root_span.set_output(
                {
                    "answer_length": len(answer.answer),
                    "citation_count": len(answer.prompt.citations),
                }
            )
            return answer

    def stream(
        self,
        question: str,
        *,
        history: Sequence[ConversationTurn] | None = None,
        few_shot_examples: Sequence[FewShotExample] | None = None,
        top_k: int | None = None,
    ) -> Iterator[StreamChunk]:
        """Stream the generated answer for a RAG question."""
        retrieval_results = self._retrieve(question, top_k=top_k)
        rerank_results = self._rerank(question, retrieval_results)
        prompt = self._prompt_builder.build(
            question=question,
            context=list(rerank_results),
            history=history,
            few_shot_examples=few_shot_examples,
        )
        request = GenerationRequest(
            messages=messages_from_prompt(
                prompt.system_prompt,
                prompt.user_prompt,
            )
        )
        yield from self._llm.stream(request)

    def chat(
        self,
        question: str,
        *,
        history: Sequence[ConversationTurn] | None = None,
    ) -> GenerationResponse:
        """Generate an answer without retrieval (direct LLM chat)."""
        tracer = get_tracer()
        with span(tracer, "rag.chat", kind="generation") as active_span:
            active_span.set_input({"question": question})
            messages: list[ChatMessage] = [
                ChatMessage(
                    role="system",
                    content="You are a helpful assistant.",
                ),
            ]
            if history:
                for turn in history:
                    messages.append(
                        ChatMessage(role=turn.role, content=turn.content),
                    )
            messages.append(ChatMessage(role="user", content=question))
            response = self._llm.complete(GenerationRequest(messages=tuple(messages)))
            active_span.set_output(
                {
                    "model": response.model,
                    "provider": response.provider,
                    "usage": response.usage,
                }
            )
            if response.usage:
                get_cost_tracker().record_generation(
                    model=response.model,
                    provider=response.provider,
                    usage=response.usage,
                )
            return response

    def _retrieve(
        self,
        question: str,
        *,
        top_k: int | None,
    ) -> list[HybridSearchResult]:
        return self._retriever.search(question, top_k=top_k)

    def _rerank(
        self,
        question: str,
        results: Sequence[HybridSearchResult],
    ) -> list[RerankResult]:
        if not results:
            return []
        if not self._use_reranker:
            return [
                RerankResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    score=result.score,
                    rank=result.rank,
                    metadata={
                        **result.metadata,
                        "source": result.source,
                        "page_number": result.page_number,
                    },
                )
                for result in results
            ]

        candidates = [
            RerankCandidate(
                chunk_id=result.chunk_id,
                text=result.text,
                metadata={
                    **result.metadata,
                    "source": result.source,
                    "page_number": result.page_number,
                },
            )
            for result in results
        ]
        return self._reranker.rerank(question, candidates)

    def _generate(self, prompt: BuiltPrompt) -> GenerationResponse:
        request = GenerationRequest(
            messages=messages_from_prompt(
                prompt.system_prompt,
                prompt.user_prompt,
            )
        )
        return self._llm.complete(request)
