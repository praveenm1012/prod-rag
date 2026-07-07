"""Prompt builder unit tests."""

import pytest

from app.rag.prompts import (
    CharTokenCounter,
    ContextChunk,
    ConversationTurn,
    FewShotExample,
    PromptBuilder,
    PromptConfigError,
    PromptInputError,
)
from app.rag.prompts.config import PromptSettings
from app.rag.reranking.models import RerankResult
from app.rag.retrieval.models import HybridSearchResult


@pytest.fixture
def builder() -> PromptBuilder:
    settings = PromptSettings(
        PROMPT_MAX_CONTEXT_TOKENS=500,
        PROMPT_RESERVED_RESPONSE_TOKENS=20,
        PROMPT_SYSTEM="You are a helpful assistant.",
        PROMPT_CONTEXT_HEADER="Context:",
        PROMPT_EXAMPLES_HEADER="Examples:",
        PROMPT_HISTORY_HEADER="Previous conversation:",
        PROMPT_QUESTION_PREFIX="Question:",
        PROMPT_ANSWER_PREFIX="Answer:",
    )
    return PromptBuilder(
        settings=settings,
        token_counter=CharTokenCounter(chars_per_token=1),
    )


@pytest.fixture
def context_chunks() -> list[ContextChunk]:
    return [
        ContextChunk(
            chunk_id="a",
            text="Python is a programming language.",
            citation_id=1,
            source="/docs/python.txt",
            page_number=1,
            rank=1,
        ),
        ContextChunk(
            chunk_id="b",
            text="Cats enjoy sunny windowsills.",
            citation_id=2,
            source="/docs/cats.txt",
            page_number=2,
            rank=2,
        ),
    ]


class TestPromptBuilderConfig:
    def test_rejects_invalid_token_budget(self) -> None:
        settings = PromptSettings(
            PROMPT_MAX_CONTEXT_TOKENS=100,
            PROMPT_RESERVED_RESPONSE_TOKENS=100,
        )
        with pytest.raises(PromptConfigError, match="reserved_response_tokens"):
            PromptBuilder(settings=settings, token_counter=CharTokenCounter())


class TestPromptBuilder:
    def test_rejects_empty_question(
        self,
        builder: PromptBuilder,
        context_chunks: list[ContextChunk],
    ) -> None:
        with pytest.raises(PromptInputError, match="question must not be empty"):
            builder.build("  ", context_chunks)

    def test_formats_context_with_citation_placeholders(
        self,
        builder: PromptBuilder,
        context_chunks: list[ContextChunk],
    ) -> None:
        prompt = builder.build("What is Python?", context_chunks)

        assert "[1] source: /docs/python.txt page: 1" in prompt.user_prompt
        assert "Python is a programming language." in prompt.user_prompt
        assert "[2] source: /docs/cats.txt page: 2" in prompt.user_prompt
        assert prompt.included_context_count == 2
        assert prompt.citations[0].placeholder == "[1]"
        assert prompt.citations[1].placeholder == "[2]"

    def test_includes_conversation_history(
        self,
        builder: PromptBuilder,
        context_chunks: list[ContextChunk],
    ) -> None:
        history = [
            ConversationTurn(role="user", content="Tell me about animals."),
            ConversationTurn(role="assistant", content="Cats are popular pets."),
        ]

        prompt = builder.build(
            "What about Python?",
            context_chunks,
            history=history,
        )

        assert "Previous conversation:" in prompt.user_prompt
        assert "User: Tell me about animals." in prompt.user_prompt
        assert "Assistant: Cats are popular pets." in prompt.user_prompt

    def test_includes_few_shot_examples(
        self,
        builder: PromptBuilder,
        context_chunks: list[ContextChunk],
    ) -> None:
        examples = [
            FewShotExample(
                question="What is Python?",
                answer="Python is a programming language [1].",
            ),
        ]

        prompt = builder.build(
            "What is Python?",
            context_chunks,
            few_shot_examples=examples,
        )

        assert "Examples:" in prompt.user_prompt
        assert "Question: What is Python?" in prompt.user_prompt
        assert "Answer: Python is a programming language [1]." in prompt.user_prompt

    def test_truncates_context_when_over_budget(self) -> None:
        settings = PromptSettings(
            PROMPT_MAX_CONTEXT_TOKENS=120,
            PROMPT_RESERVED_RESPONSE_TOKENS=10,
            PROMPT_SYSTEM="System",
        )
        builder = PromptBuilder(
            settings=settings,
            token_counter=CharTokenCounter(chars_per_token=1),
        )
        chunks = [
            ContextChunk(
                chunk_id="a",
                text="A" * 40,
                citation_id=1,
                rank=1,
            ),
            ContextChunk(
                chunk_id="b",
                text="B" * 40,
                citation_id=2,
                rank=2,
            ),
            ContextChunk(
                chunk_id="c",
                text="C" * 40,
                citation_id=3,
                rank=3,
            ),
        ]

        prompt = builder.build("Short question?", chunks)

        assert prompt.truncated is True
        assert prompt.dropped_context_count >= 1
        assert prompt.included_context_count < len(chunks)
        assert "CCC" not in prompt.user_prompt or prompt.included_context_count == 3

    def test_truncates_history_before_context(self) -> None:
        settings = PromptSettings(
            PROMPT_MAX_CONTEXT_TOKENS=90,
            PROMPT_RESERVED_RESPONSE_TOKENS=5,
            PROMPT_SYSTEM="Sys",
        )
        builder = PromptBuilder(
            settings=settings,
            token_counter=CharTokenCounter(chars_per_token=1),
        )
        history = [
            ConversationTurn(role="user", content="U" * 30),
            ConversationTurn(role="assistant", content="A" * 30),
            ConversationTurn(role="user", content="U" * 30),
        ]
        chunks = [
            ContextChunk(chunk_id="a", text="X" * 20, citation_id=1, rank=1),
        ]

        prompt = builder.build("Q?", chunks, history=history)

        assert prompt.dropped_history_turns >= 1 or prompt.truncated

    def test_adapts_hybrid_search_result(self, builder: PromptBuilder) -> None:
        result = HybridSearchResult(
            chunk_id="hyb:1",
            text="Hybrid retrieval combines BM25 and vectors.",
            score=0.9,
            rank=1,
            bm25_score=1.0,
            vector_score=0.8,
            bm25_score_normalized=1.0,
            vector_score_normalized=0.8,
            document_id="hyb",
            page_number=3,
            source="/docs/hyb.txt",
        )

        prompt = builder.build("How does hybrid retrieval work?", [result])

        assert "[1] source: /docs/hyb.txt page: 3" in prompt.user_prompt
        assert prompt.citations[0].chunk_id == "hyb:1"

    def test_adapts_rerank_result(self, builder: PromptBuilder) -> None:
        result = RerankResult(
            chunk_id="rerank:1",
            text="Reranked passage text.",
            score=0.95,
            rank=1,
            metadata={"source": "/docs/rerank.txt", "page_number": 4},
        )

        prompt = builder.build("Question?", [result])

        assert "[1] source: /docs/rerank.txt page: 4" in prompt.user_prompt
        assert prompt.citations[0].placeholder == "[1]"

    def test_count_tokens(self, builder: PromptBuilder) -> None:
        assert builder.count_tokens("abcd") == 4

    def test_system_prompt_is_separate(self, builder: PromptBuilder) -> None:
        prompt = builder.build(
            "Question?",
            [ContextChunk(chunk_id="a", text="Context text.", citation_id=1)],
        )

        assert prompt.system_prompt == "You are a helpful assistant."
        assert "You are a helpful assistant." not in prompt.user_prompt

    def test_keeps_higher_ranked_context_first_when_truncating(self) -> None:
        settings = PromptSettings(
            PROMPT_MAX_CONTEXT_TOKENS=80,
            PROMPT_RESERVED_RESPONSE_TOKENS=5,
            PROMPT_SYSTEM="S",
        )
        builder = PromptBuilder(
            settings=settings,
            token_counter=CharTokenCounter(chars_per_token=1),
        )
        chunks = [
            ContextChunk(chunk_id="first", text="F" * 25, citation_id=1, rank=1),
            ContextChunk(chunk_id="second", text="S" * 25, citation_id=2, rank=2),
            ContextChunk(chunk_id="third", text="T" * 25, citation_id=3, rank=3),
        ]

        prompt = builder.build("Q?", chunks)

        assert "FFFF" in prompt.user_prompt
        assert prompt.dropped_context_count >= 1
