#!/usr/bin/env python3
"""Run live API smoke tests for upload, chat, and delete."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


def main() -> int:
    client = TestClient(create_app())
    results: list[tuple[str, int, int, bool]] = []

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("Python is a programming language used for RAG testing.")
        upload_path = Path(handle.name)

    try:
        with upload_path.open("rb") as handle:
            upload = client.post(
                "/api/v1/upload",
                files={"file": ("notes.txt", handle, "text/plain")},
            )
        results.append(("POST /api/v1/upload", upload.status_code, 201, upload.status_code == 201))
        print(f"POST /api/v1/upload -> {upload.status_code}")
        if upload.status_code != 201:
            print(upload.text)
            return 1

        document_id = upload.json()["document"]["document_id"]
        print(json.dumps(upload.json(), indent=2))

        chat = client.post(
            "/api/v1/chat",
            json={"question": "What is Python?", "use_rag": True, "top_k": 3},
        )
        results.append(("POST /api/v1/chat", chat.status_code, 200, chat.status_code == 200))
        print(f"\nPOST /api/v1/chat -> {chat.status_code}")
        if chat.status_code != 200:
            print(chat.text)
            return 1
        print(json.dumps(chat.json(), indent=2))

        deleted = client.request(
            "DELETE",
            "/api/v1/delete",
            json={"document_id": document_id},
        )
        results.append(
            ("DELETE /api/v1/delete", deleted.status_code, 204, deleted.status_code == 204)
        )
        print(f"\nDELETE /api/v1/delete -> {deleted.status_code}")
        if deleted.status_code != 204:
            print(deleted.text)
            return 1

        print("\nSummary")
        print("-" * 48)
        all_passed = True
        for name, actual, expected, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"{status} {name}: expected {expected}, got {actual}")
            all_passed = all_passed and passed

        return 0 if all_passed else 1
    finally:
        upload_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
