"""Upload API tests."""

from fastapi.testclient import TestClient


def test_upload_text_document(client: TestClient) -> None:
    response = client.post(
        "/api/v1/upload",
        files={
            "file": ("notes.txt", b"Python is a programming language.", "text/plain")
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["chunks_indexed"] >= 1
    assert body["document"]["filename"] == "notes.txt"
    assert body["document"]["document_id"]


def test_upload_empty_file_returns_400(client: TestClient) -> None:
    response = client.post(
        "/api/v1/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "uploaded file is empty"


def test_upload_unsupported_type_returns_415(client: TestClient) -> None:
    response = client.post(
        "/api/v1/upload",
        files={"file": ("data.unknown", b"binary", "application/octet-stream")},
    )
    assert response.status_code == 415
