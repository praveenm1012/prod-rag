"""Chat API tests."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import chat as chat_module
from app.main import create_app
from app.rag.generation.models import GenerationResponse, StreamChunk


@pytest.fixture(autouse=True)
def reset_pipeline() -> None:
    chat_module._pipeline = None
    yield
    chat_module._pipeline = None


def test_provider_status() -> None:
    client = TestClient(create_app())
    response = client.get("/api/v1/chat/status")
    assert response.status_code == 200
    body = response.json()
    assert "provider" in body
    assert "model" in body
    assert "configured" in body


@patch("app.api.chat.create_llm_provider")
def test_chat_endpoint(mock_create_provider: MagicMock) -> None:
    mock_provider = MagicMock()
    mock_provider.complete.return_value = GenerationResponse(
        content="Hello from the LLM.",
        model="gpt-4o-mini",
        provider="openai",
        finish_reason="stop",
        usage={"total_tokens": 10},
    )
    mock_create_provider.return_value = mock_provider

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/chat",
        json={"question": "Say hello"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hello from the LLM."
    assert body["provider"] == "openai"
    assert body["model"] == "gpt-4o-mini"


@patch("app.api.chat.create_llm_provider")
def test_chat_stream_endpoint(mock_create_provider: MagicMock) -> None:
    mock_provider = MagicMock()
    mock_provider.stream.return_value = [
        StreamChunk(content="Hel"),
        StreamChunk(content="lo"),
        StreamChunk(content="", is_final=True, finish_reason="stop"),
    ]
    mock_create_provider.return_value = mock_provider

    client = TestClient(create_app())
    response = client.post(
        "/api/v1/chat/stream",
        json={"question": "Say hello"},
    )

    assert response.status_code == 200
    assert response.text == "Hello"
