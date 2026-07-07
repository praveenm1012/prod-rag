"""LLM provider unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.rag.generation import (
    ChatMessage,
    ClaudeProvider,
    GenerationConfigError,
    GenerationRequest,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    ProviderTimeoutError,
    create_llm_provider,
    messages_from_prompt,
)
from app.rag.generation.config import GenerationSettings


class MockHttpResponse:
    def __init__(
        self,
        data: dict[str, Any],
        status_code: int = 200,
        *,
        should_fail: bool = False,
    ) -> None:
        self._data = data
        self.status_code = status_code
        self._should_fail = should_fail

    def json(self) -> dict[str, Any]:
        return self._data

    def raise_for_status(self) -> None:
        if self._should_fail:
            raise ProviderError("HTTP 500")


class MockHttpClient:
    def __init__(
        self,
        post_responses: list[MockHttpResponse] | None = None,
        stream_lines_data: list[str] | None = None,
    ) -> None:
        self.post_responses = list(post_responses or [])
        self._stream_lines_data = list(stream_lines_data or [])
        self.post_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self._post_index = 0

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> MockHttpResponse:
        self.post_calls.append({"url": url, "headers": headers, "json": json})
        if not self.post_responses:
            raise ProviderError("No mock POST response configured")
        response = self.post_responses[
            min(self._post_index, len(self.post_responses) - 1)
        ]
        self._post_index += 1
        return response

    def stream_lines(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> Iterator[str]:
        self.stream_calls.append({"url": url, "headers": headers, "json": json})
        yield from self._stream_lines_data


@pytest.fixture
def openai_settings() -> GenerationSettings:
    return GenerationSettings(
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4o-mini",
        OPENAI_API_KEY="test-openai-key",
        LLM_MAX_RETRIES=3,
        LLM_RETRY_MIN_SECONDS=0.01,
        LLM_RETRY_MAX_SECONDS=0.02,
    )


@pytest.fixture
def claude_settings() -> GenerationSettings:
    return GenerationSettings(
        LLM_PROVIDER="claude",
        LLM_MODEL="claude-3-5-sonnet-20241022",
        ANTHROPIC_API_KEY="test-anthropic-key",
        LLM_MAX_RETRIES=3,
        LLM_RETRY_MIN_SECONDS=0.01,
        LLM_RETRY_MAX_SECONDS=0.02,
    )


@pytest.fixture
def ollama_settings() -> GenerationSettings:
    return GenerationSettings(
        LLM_PROVIDER="ollama",
        LLM_MODEL="llama3.2",
        OLLAMA_BASE_URL="http://localhost:11434",
        LLM_MAX_RETRIES=1,
    )


@pytest.fixture
def sample_messages() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="What is Python?"),
    )


class TestProviderFactory:
    def test_openai_requires_api_key(self, openai_settings: GenerationSettings) -> None:
        settings = GenerationSettings(
            LLM_PROVIDER="openai",
            OPENAI_API_KEY=None,
        )
        with pytest.raises(GenerationConfigError, match="OPENAI_API_KEY"):
            OpenAIProvider(settings, MockHttpClient())

    def test_claude_requires_api_key(self) -> None:
        settings = GenerationSettings(
            LLM_PROVIDER="claude",
            ANTHROPIC_API_KEY=None,
        )
        with pytest.raises(GenerationConfigError, match="ANTHROPIC_API_KEY"):
            ClaudeProvider(settings, MockHttpClient())

    def test_factory_creates_openai_provider(
        self,
        openai_settings: GenerationSettings,
    ) -> None:
        provider = create_llm_provider(
            provider="openai",
            settings=openai_settings,
            http_client=MockHttpClient(),
        )
        assert isinstance(provider, OpenAIProvider)

    def test_factory_creates_ollama_provider(
        self,
        ollama_settings: GenerationSettings,
    ) -> None:
        provider = create_llm_provider(
            provider="ollama",
            settings=ollama_settings,
            http_client=MockHttpClient(),
        )
        assert isinstance(provider, OllamaProvider)


class TestOpenAIProvider:
    def test_complete(
        self,
        openai_settings: GenerationSettings,
        sample_messages: tuple[ChatMessage, ...],
    ) -> None:
        client = MockHttpClient(
            post_responses=[
                MockHttpResponse(
                    {
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "message": {"content": "Python is a language."},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                )
            ]
        )
        provider = OpenAIProvider(openai_settings, client)
        response = provider.complete(GenerationRequest(messages=sample_messages))

        assert response.content == "Python is a language."
        assert response.provider == "openai"
        assert response.usage["total_tokens"] == 15
        assert client.post_calls[0]["json"]["stream"] is False

    def test_stream(
        self,
        openai_settings: GenerationSettings,
        sample_messages: tuple[ChatMessage, ...],
    ) -> None:
        client = MockHttpClient(
            stream_lines_data=[
                'data: {"choices":[{"delta":{"content":"Py"}}]}',
                'data: {"choices":[{"delta":{"content":"thon"}}]}',
                "data: [DONE]",
            ]
        )
        provider = OpenAIProvider(openai_settings, client)
        chunks = list(provider.stream(GenerationRequest(messages=sample_messages)))

        assert "".join(chunk.content for chunk in chunks) == "Python"
        assert client.stream_calls[0]["json"]["stream"] is True

    def test_retries_transient_failure(
        self,
        openai_settings: GenerationSettings,
        sample_messages: tuple[ChatMessage, ...],
    ) -> None:
        client = MockHttpClient(
            post_responses=[
                MockHttpResponse({}, should_fail=True),
                MockHttpResponse(
                    {
                        "model": "gpt-4o-mini",
                        "choices": [
                            {
                                "message": {"content": "Recovered."},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {},
                    }
                ),
            ]
        )
        provider = OpenAIProvider(openai_settings, client)
        response = provider.complete(GenerationRequest(messages=sample_messages))

        assert response.content == "Recovered."
        assert len(client.post_calls) == 2


class TestClaudeProvider:
    def test_complete_splits_system_message(
        self,
        claude_settings: GenerationSettings,
        sample_messages: tuple[ChatMessage, ...],
    ) -> None:
        client = MockHttpClient(
            post_responses=[
                MockHttpResponse(
                    {
                        "model": "claude-3-5-sonnet-20241022",
                        "content": [{"type": "text", "text": "Python is great."}],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 8, "output_tokens": 4},
                    }
                )
            ]
        )
        provider = ClaudeProvider(claude_settings, client)
        response = provider.complete(GenerationRequest(messages=sample_messages))

        assert response.content == "Python is great."
        assert client.post_calls[0]["json"]["system"] == "You are helpful."
        assert client.post_calls[0]["json"]["messages"] == [
            {"role": "user", "content": "What is Python?"}
        ]

    def test_stream(
        self,
        claude_settings: GenerationSettings,
        sample_messages: tuple[ChatMessage, ...],
    ) -> None:
        client = MockHttpClient(
            stream_lines_data=[
                'data: {"type":"content_block_delta","delta":{"text":"Cla"}}',
                'data: {"type":"content_block_delta","delta":{"text":"ude"}}',
                'data: {"type":"message_stop"}',
            ]
        )
        provider = ClaudeProvider(claude_settings, client)
        chunks = list(provider.stream(GenerationRequest(messages=sample_messages)))

        assert "".join(chunk.content for chunk in chunks) == "Claude"
        assert chunks[-1].is_final is True


class TestOllamaProvider:
    def test_complete(
        self,
        ollama_settings: GenerationSettings,
        sample_messages: tuple[ChatMessage, ...],
    ) -> None:
        client = MockHttpClient(
            post_responses=[
                MockHttpResponse(
                    {
                        "model": "llama3.2",
                        "message": {"role": "assistant", "content": "Local answer."},
                        "done": True,
                        "prompt_eval_count": 6,
                        "eval_count": 3,
                    }
                )
            ]
        )
        provider = OllamaProvider(ollama_settings, client)
        response = provider.complete(GenerationRequest(messages=sample_messages))

        assert response.content == "Local answer."
        assert response.provider == "ollama"
        assert response.usage["total_tokens"] == 9
        assert client.post_calls[0]["url"].endswith("/api/chat")

    def test_stream(
        self,
        ollama_settings: GenerationSettings,
        sample_messages: tuple[ChatMessage, ...],
    ) -> None:
        client = MockHttpClient(
            stream_lines_data=[
                '{"message":{"content":"Ol"},"done":false}',
                '{"message":{"content":"lama"},"done":true}',
            ]
        )
        provider = OllamaProvider(ollama_settings, client)
        chunks = list(provider.stream(GenerationRequest(messages=sample_messages)))

        assert "".join(chunk.content for chunk in chunks) == "Ollama"
        assert chunks[-1].is_final is True


class TestHelpers:
    def test_messages_from_prompt(self) -> None:
        messages = messages_from_prompt("System text", "User text")
        assert messages[0].role == "system"
        assert messages[1].role == "user"
        assert messages[1].content == "User text"


class TestTimeout:
    def test_timeout_error_from_http_client(
        self,
        openai_settings: GenerationSettings,
        sample_messages: tuple[ChatMessage, ...],
    ) -> None:
        client = MagicMock()
        client.post.side_effect = ProviderTimeoutError("timed out")
        provider = OpenAIProvider(openai_settings, client)

        with pytest.raises(ProviderTimeoutError):
            provider.complete(GenerationRequest(messages=sample_messages))
