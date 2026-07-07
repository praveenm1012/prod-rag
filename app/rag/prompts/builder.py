"""RAG prompt builder."""

from __future__ import annotations

from collections.abc import Sequence

from app.core.logging import get_logger
from app.rag.prompts.config import PromptSettings, get_prompt_settings
from app.rag.prompts.exceptions import PromptConfigError, PromptInputError
from app.rag.prompts.models import (
    BuiltPrompt,
    Citation,
    ContextChunk,
    ConversationTurn,
    FewShotExample,
)
from app.rag.prompts.token_counter import TokenCounter, create_token_counter
from app.rag.reranking.models import RerankResult
from app.rag.retrieval.models import HybridSearchResult

logger = get_logger(__name__)

type ContextInput = ContextChunk | HybridSearchResult | RerankResult


class PromptBuilder:
    """Build LLM prompts with context, history, few-shot examples, and citations."""

    def __init__(
        self,
        settings: PromptSettings | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._settings = settings or get_prompt_settings()
        if self._settings.max_context_tokens <= 0:
            raise PromptConfigError("max_context_tokens must be greater than 0")
        if self._settings.reserved_response_tokens < 0:
            raise PromptConfigError(
                "reserved_response_tokens must be greater than or equal to 0"
            )
        if self._settings.reserved_response_tokens >= self._settings.max_context_tokens:
            raise PromptConfigError(
                "reserved_response_tokens must be less than max_context_tokens"
            )

        self._token_counter = token_counter or create_token_counter(
            self._settings.token_encoding
        )

    @property
    def token_budget(self) -> int:
        return (
            self._settings.max_context_tokens - self._settings.reserved_response_tokens
        )

    def build(
        self,
        question: str,
        context: Sequence[ContextInput],
        *,
        history: Sequence[ConversationTurn] | None = None,
        few_shot_examples: Sequence[FewShotExample] | None = None,
    ) -> BuiltPrompt:
        """Assemble a prompt with optional truncation to fit the token budget."""
        normalized_question = question.strip()
        if not normalized_question:
            raise PromptInputError("question must not be empty")

        context_chunks = _normalize_context(context)
        history_turns = list(history or [])
        examples = list(few_shot_examples or [])

        system_prompt = self._settings.system_prompt
        fixed_sections = self._join_sections(
            [
                self._format_examples(examples),
                self._format_history(history_turns),
                self._format_question(normalized_question),
            ]
        )
        fixed_tokens = self._count_sections(system_prompt, fixed_sections)

        if fixed_tokens > self.token_budget:
            history_turns, dropped_history = self._truncate_history(
                history_turns,
                budget=self.token_budget - self._count_sections(system_prompt, ""),
            )
            fixed_sections = self._join_sections(
                [
                    self._format_examples(examples),
                    self._format_history(history_turns),
                    self._format_question(normalized_question),
                ]
            )
            fixed_tokens = self._count_sections(system_prompt, fixed_sections)
            if fixed_tokens > self.token_budget:
                raise PromptInputError(
                    "question and fixed prompt sections exceed the token budget"
                )
        else:
            dropped_history = 0

        context_budget = self.token_budget - fixed_tokens
        selected_chunks, dropped_context = self._fit_context(
            context_chunks,
            budget=context_budget,
        )

        context_section = self._format_context(selected_chunks)
        user_prompt = self._join_sections([context_section, fixed_sections])
        token_count = self._count_sections(system_prompt, user_prompt)
        citations = tuple(_build_citations(selected_chunks))

        logger.info(
            "prompt_built",
            token_count=token_count,
            token_budget=self.token_budget,
            included_context=len(selected_chunks),
            dropped_context=dropped_context,
            dropped_history=dropped_history,
            truncated=bool(dropped_context or dropped_history),
        )

        return BuiltPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            token_count=token_count,
            truncated=bool(dropped_context or dropped_history),
            citations=citations,
            included_context_count=len(selected_chunks),
            dropped_context_count=dropped_context,
            dropped_history_turns=dropped_history,
        )

    def count_tokens(self, text: str) -> int:
        """Return the token count for a text fragment."""
        return self._token_counter.count(text)

    def _fit_context(
        self,
        chunks: Sequence[ContextChunk],
        budget: int,
    ) -> tuple[list[ContextChunk], int]:
        if budget <= 0 or not chunks:
            return [], len(chunks)

        selected: list[ContextChunk] = []
        used_tokens = 0
        header_tokens = self._token_counter.count(f"{self._settings.context_header}\n")

        for chunk in chunks:
            block = _format_context_block(chunk)
            block_tokens = self._token_counter.count(block)
            extra = header_tokens if not selected else 0
            if used_tokens + extra + block_tokens > budget:
                break
            if not selected:
                used_tokens += header_tokens
            selected.append(chunk)
            used_tokens += block_tokens

        dropped = len(chunks) - len(selected)
        return selected, dropped

    def _truncate_history(
        self,
        history: list[ConversationTurn],
        budget: int,
    ) -> tuple[list[ConversationTurn], int]:
        if budget <= 0 or not history:
            return [], len(history)

        kept: list[ConversationTurn] = []
        for turn in reversed(history):
            candidate = [turn, *kept]
            section = self._format_history(candidate)
            if self._token_counter.count(section) > budget:
                break
            kept = candidate

        kept.reverse()
        return kept, len(history) - len(kept)

    def _format_context(self, chunks: Sequence[ContextChunk]) -> str:
        if not chunks:
            return ""
        blocks = [_format_context_block(chunk) for chunk in chunks]
        return f"{self._settings.context_header}\n" + "\n\n".join(blocks)

    def _format_examples(self, examples: Sequence[FewShotExample]) -> str:
        if not examples:
            return ""
        lines = [self._settings.examples_header, ""]
        for example in examples:
            lines.extend(
                [
                    f"{self._settings.question_prefix} {example.question}",
                    f"{self._settings.answer_prefix} {example.answer}",
                    "",
                ]
            )
        return "\n".join(lines).rstrip()

    def _format_history(self, history: Sequence[ConversationTurn]) -> str:
        if not history:
            return ""
        lines = [self._settings.history_header, ""]
        for turn in history:
            role = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)

    def _format_question(self, question: str) -> str:
        return (
            f"{self._settings.question_prefix} {question}\n"
            f"{self._settings.answer_prefix}"
        )

    def _count_sections(self, system_prompt: str, user_prompt: str) -> int:
        total = 0
        if system_prompt:
            total += self._token_counter.count(system_prompt)
        if user_prompt:
            total += self._token_counter.count(user_prompt)
        return total

    @staticmethod
    def _join_sections(sections: Sequence[str]) -> str:
        return "\n\n".join(section for section in sections if section).strip()


def context_from_hybrid_result(
    result: HybridSearchResult,
    citation_id: int,
) -> ContextChunk:
    """Convert a hybrid retrieval result into a context chunk."""
    return ContextChunk(
        chunk_id=result.chunk_id,
        text=result.text,
        citation_id=citation_id,
        source=result.source,
        page_number=result.page_number,
        rank=result.rank,
        metadata=dict(result.metadata),
    )


def context_from_rerank_result(
    result: RerankResult,
    citation_id: int,
) -> ContextChunk:
    """Convert a rerank result into a context chunk."""
    metadata = dict(result.metadata)
    source = str(metadata.pop("source", ""))
    page_number = metadata.pop("page_number", None)
    page = int(page_number) if isinstance(page_number, int) else None
    return ContextChunk(
        chunk_id=result.chunk_id,
        text=result.text,
        citation_id=citation_id,
        source=source,
        page_number=page,
        rank=result.rank,
        metadata=metadata,
    )


def _normalize_context(context: Sequence[ContextInput]) -> list[ContextChunk]:
    normalized: list[ContextChunk] = []
    for index, item in enumerate(context, start=1):
        if isinstance(item, ContextChunk):
            normalized.append(item)
            continue

        if isinstance(item, HybridSearchResult):
            normalized.append(context_from_hybrid_result(item, citation_id=index))
            continue
        if isinstance(item, RerankResult):
            normalized.append(context_from_rerank_result(item, citation_id=index))
            continue

        raise PromptInputError(f"Unsupported context type: {type(item).__name__}")

    normalized.sort(key=lambda chunk: chunk.rank or chunk.citation_id)
    for index, chunk in enumerate(normalized, start=1):
        if chunk.citation_id != index:
            normalized[index - 1] = ContextChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                citation_id=index,
                source=chunk.source,
                page_number=chunk.page_number,
                rank=chunk.rank,
                metadata=dict(chunk.metadata),
            )
    return normalized


def _format_context_block(chunk: ContextChunk) -> str:
    header_parts = [f"[{chunk.citation_id}]"]
    if chunk.source:
        header_parts.append(f"source: {chunk.source}")
    if chunk.page_number is not None:
        header_parts.append(f"page: {chunk.page_number}")
    header = " ".join(header_parts)
    return f"{header}\n{chunk.text}"


def _build_citations(chunks: Sequence[ContextChunk]) -> list[Citation]:
    return [
        Citation(
            citation_id=chunk.citation_id,
            chunk_id=chunk.chunk_id,
            source=chunk.source,
            page_number=chunk.page_number,
            placeholder=f"[{chunk.citation_id}]",
        )
        for chunk in chunks
    ]
