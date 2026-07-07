"""End-to-end smoke tests for the RAG workflow."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "e2e_smoke_test.py"


def _load_smoke_module():
    spec = importlib.util.spec_from_file_location("e2e_smoke_test", SCRIPTS)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load e2e smoke script")
    module = importlib.util.module_from_spec(spec)
    sys.modules["e2e_smoke_test"] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke_module()


@pytest.mark.integration
def test_e2e_smoke_pdf_workflow_with_fakes(client: TestClient) -> None:
    """Fast in-process API smoke test with deterministic fakes."""
    pdf_bytes = smoke.create_sample_pdf_bytes()

    upload = client.post(
        "/api/v1/upload",
        files={
            "file": (
                "sample_smoke.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )
    assert upload.status_code == 201
    body = upload.json()
    document_id = body["document"]["document_id"]
    assert body["chunks_indexed"] >= 1

    chat = client.post(
        "/api/v1/chat",
        json={
            "question": smoke.SAMPLE_QUESTION,
            "use_rag": True,
            "top_k": 3,
        },
    )
    assert chat.status_code == 200
    chat_body = chat.json()
    assert chat_body["answer"]
    assert smoke.EXPECTED_CITATION in chat_body["citations"]

    deleted = client.request(
        "DELETE",
        "/api/v1/delete",
        json={"document_id": document_id},
    )
    assert deleted.status_code == 204


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_E2E_SMOKE") != "1",
    reason="Set RUN_E2E_SMOKE=1 to run live local pipeline smoke test",
)
def test_e2e_smoke_local_live() -> None:
    """Full local pipeline with real embeddings and configured LLM provider."""
    result = smoke.run_local_smoke(use_reranker=False)
    assert result.status == "pass", result.to_dict()
