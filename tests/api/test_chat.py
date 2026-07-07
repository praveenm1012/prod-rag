"""Chat API tests."""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_provider_status(client: TestClient) -> None:
    response = client.get("/api/v1/chat/status")
    assert response.status_code == 200
    body = response.json()
    assert "provider" in body
    assert "model" in body
    assert "configured" in body


def test_chat_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"question": "Say hello"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Test answer from LLM."
    assert body["provider"] == "openai"
    assert body["model"] == "test-model"


def test_chat_stream_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/v1/chat/stream",
        json={"question": "Say hello"},
    )

    assert response.status_code == 200
    assert response.text == "Test answer."


def test_chat_with_rag_after_upload(client: TestClient) -> None:
    upload = client.post(
        "/api/v1/upload",
        files={
            "file": ("notes.txt", b"Python is a programming language.", "text/plain")
        },
    )
    assert upload.status_code == 201

    response = client.post(
        "/api/v1/chat",
        json={"question": "What is Python?", "use_rag": True, "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Test answer from LLM."
    assert isinstance(body["citations"], list)


@patch("app.api.chat._provider_configured", return_value=False)
def test_chat_returns_503_when_provider_not_configured(
    _mock_configured,
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/chat",
        json={"question": "hello"},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "LLM provider is not configured"


def test_chat_validation_error(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"question": ""})
    assert response.status_code == 422
