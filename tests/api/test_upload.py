"""Upload API tests."""

import time

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
    assert body["document"]["status"] == "indexed"


def test_upload_background_returns_202_and_indexes(client: TestClient) -> None:
    response = client.post(
        "/api/v1/upload?background=true",
        files={"file": ("notes.txt", b"Background upload test content.", "text/plain")},
    )

    assert response.status_code == 202
    body = response.json()
    document_id = body["document"]["document_id"]
    assert body["status"] == "processing"
    assert body["document"]["status"] == "processing"

    final_status = None
    for _ in range(50):
        status_response = client.get(f"/api/v1/upload/{document_id}/status")
        assert status_response.status_code == 200
        final_status = status_response.json()
        if final_status["status"] == "indexed":
            break
        time.sleep(0.1)

    assert final_status is not None
    assert final_status["status"] == "indexed"
    assert final_status["chunk_count"] >= 1


def test_upload_status_missing_document_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/upload/missing-document/status")
    assert response.status_code == 404


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
