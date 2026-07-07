"""Docker-based Qdrant repository integration tests."""

from __future__ import annotations

import os
import time
import uuid

import pytest

from app.rag.vectorstore import (
    MetadataFilter,
    QdrantRepository,
    QdrantSettings,
    VectorRecord,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.docker,
    pytest.mark.skipif(
        os.getenv("RUN_DOCKER_TESTS", "0") != "1",
        reason="Set RUN_DOCKER_TESTS=1 to run Docker-based Qdrant tests",
    ),
]

qdrant_client = pytest.importorskip("qdrant_client")
pytest.importorskip("testcontainers")


@pytest.fixture(scope="module")
def qdrant_url() -> str:
    """Start a Qdrant container for integration tests."""
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_for_logs

    container = DockerContainer("qdrant/qdrant:latest")
    container.with_exposed_ports(6333, 6334)
    container.start()

    try:
        wait_for_logs(container, "Qdrant HTTP listening", timeout=60)
        host = container.get_container_host_ip()
        port = container.get_exposed_port(6333)
        url = f"http://{host}:{port}"

        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                repo = QdrantRepository(
                    settings=QdrantSettings(
                        QDRANT_URL=url,
                        QDRANT_COLLECTION=f"test_{uuid.uuid4().hex[:8]}",
                    )
                )
                repo.health()
                break
            except Exception:
                time.sleep(1)
        else:
            pytest.fail("Qdrant container did not become healthy in time")

        yield url
    finally:
        container.stop()


@pytest.fixture
def repository(qdrant_url: str) -> QdrantRepository:
    collection = f"test_{uuid.uuid4().hex[:8]}"
    repo = QdrantRepository(
        settings=QdrantSettings(
            QDRANT_URL=qdrant_url,
            QDRANT_COLLECTION=collection,
            QDRANT_VECTOR_SIZE=8,
        )
    )
    repo.create_collection(recreate=True)
    return repo


VECTOR_A = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VECTOR_A_NEAR = [0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VECTOR_A_ALT = [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
VECTOR_B = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _record(
    record_id: str,
    vector: list[float],
    *,
    document_id: str,
    source: str,
    page_number: int,
    text: str,
) -> VectorRecord:
    return VectorRecord(
        id=record_id,
        vector=vector,
        payload={
            "chunk_id": record_id,
            "document_id": document_id,
            "source": source,
            "page_number": page_number,
            "text": text,
        },
    )


def test_health(repository: QdrantRepository) -> None:
    health = repository.health()

    assert health["status"] == "ok"
    assert health["collection_exists"] is True
    assert health["points_count"] == 0


def test_create_collection_is_idempotent(repository: QdrantRepository) -> None:
    repository.create_collection()
    repository.create_collection()
    health = repository.health()

    assert health["collection_exists"] is True


def test_insert_and_search(repository: QdrantRepository) -> None:
    records = [
        _record(
            "doc-a:1:0",
            VECTOR_A,
            document_id="doc-a",
            source="/a.txt",
            page_number=1,
            text="alpha",
        ),
        _record(
            "doc-a:1:1",
            VECTOR_A_NEAR,
            document_id="doc-a",
            source="/a.txt",
            page_number=1,
            text="alphabet",
        ),
        _record(
            "doc-b:1:0",
            VECTOR_B,
            document_id="doc-b",
            source="/b.txt",
            page_number=1,
            text="beta",
        ),
    ]
    inserted = repository.insert(records)

    assert inserted == 3
    health = repository.health()
    assert health["points_count"] == 3

    results = repository.search(vector=VECTOR_A, limit=2)

    assert len(results) == 2
    assert results[0].id == "doc-a:1:0"
    assert results[0].payload["text"] == "alpha"
    assert results[0].score >= results[1].score


def test_search_with_metadata_filter(repository: QdrantRepository) -> None:
    repository.insert(
        [
            _record(
                "doc-a:1:0",
                VECTOR_A,
                document_id="doc-a",
                source="/a.txt",
                page_number=1,
                text="alpha",
            ),
            _record(
                "doc-b:1:0",
                VECTOR_A_NEAR,
                document_id="doc-b",
                source="/b.txt",
                page_number=1,
                text="beta",
            ),
        ]
    )

    results = repository.search(
        vector=VECTOR_A,
        limit=5,
        filters=MetadataFilter(document_id="doc-b"),
    )

    assert len(results) == 1
    assert results[0].payload["document_id"] == "doc-b"


def test_delete_by_id(repository: QdrantRepository) -> None:
    repository.insert(
        [
            _record(
                "doc-a:1:0",
                VECTOR_A,
                document_id="doc-a",
                source="/a.txt",
                page_number=1,
                text="alpha",
            ),
            _record(
                "doc-b:1:0",
                VECTOR_B,
                document_id="doc-b",
                source="/b.txt",
                page_number=1,
                text="beta",
            ),
        ]
    )

    repository.delete(ids=["doc-a:1:0"])
    health = repository.health()

    assert health["points_count"] == 1
    results = repository.search(vector=VECTOR_A, limit=5)
    assert len(results) == 1
    assert results[0].id == "doc-b:1:0"


def test_delete_by_metadata_filter(repository: QdrantRepository) -> None:
    repository.insert(
        [
            _record(
                "doc-a:1:0",
                VECTOR_A,
                document_id="doc-a",
                source="/a.txt",
                page_number=1,
                text="alpha",
            ),
            _record(
                "doc-a:2:0",
                VECTOR_A_ALT,
                document_id="doc-a",
                source="/a.txt",
                page_number=2,
                text="second",
            ),
            _record(
                "doc-b:1:0",
                VECTOR_B,
                document_id="doc-b",
                source="/b.txt",
                page_number=1,
                text="beta",
            ),
        ]
    )

    repository.delete(filters=MetadataFilter(document_id="doc-a"))
    health = repository.health()

    assert health["points_count"] == 1
    results = repository.search(vector=VECTOR_B, limit=5)
    assert len(results) == 1
    assert results[0].payload["document_id"] == "doc-b"
