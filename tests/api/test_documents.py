"""Documents API tests."""

from fastapi.testclient import TestClient


def test_list_documents_empty(client: TestClient) -> None:
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0
    assert body["documents"] == []


def test_list_documents_after_upload(client: TestClient) -> None:
    upload = client.post(
        "/api/v1/upload",
        files={
            "file": ("notes.txt", b"Python is a programming language.", "text/plain")
        },
    )
    assert upload.status_code == 201

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["documents"][0]["filename"] == "notes.txt"
