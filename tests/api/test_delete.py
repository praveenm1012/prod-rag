"""Delete API tests."""

from fastapi.testclient import TestClient


def test_delete_document(client: TestClient) -> None:
    upload = client.post(
        "/api/v1/upload",
        files={
            "file": ("notes.txt", b"Python is a programming language.", "text/plain")
        },
    )
    assert upload.status_code == 201
    document_id = upload.json()["document"]["document_id"]

    response = client.request(
        "DELETE",
        "/api/v1/delete",
        json={"document_id": document_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == document_id
    assert body["deleted"] is True

    listing = client.get("/api/v1/documents")
    assert listing.json()["total"] == 0


def test_delete_missing_document_returns_404(client: TestClient) -> None:
    response = client.request(
        "DELETE",
        "/api/v1/delete",
        json={"document_id": "missing-document-id"},
    )
    assert response.status_code == 404
