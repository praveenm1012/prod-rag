"""Qdrant vector store repository."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, cast

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.logging import get_logger
from app.rag.vectorstore.config import QdrantSettings, get_qdrant_settings
from app.rag.vectorstore.exceptions import (
    VectorStoreConfigError,
    VectorStoreOperationError,
)
from app.rag.vectorstore.models import MetadataFilter, SearchResult, VectorRecord

logger = get_logger(__name__)


class QdrantRepository:
    """Repository for Qdrant vector storage and retrieval."""

    def __init__(self, settings: QdrantSettings | None = None) -> None:
        self._settings = settings or get_qdrant_settings()
        if self._settings.vector_size <= 0:
            raise VectorStoreConfigError("vector_size must be greater than 0")

        self._client = QdrantClient(
            url=self._settings.url,
            api_key=self._settings.api_key,
            timeout=int(self._settings.timeout_seconds),
        )

    @property
    def collection_name(self) -> str:
        return self._settings.collection_name

    @property
    def vector_size(self) -> int:
        return self._settings.vector_size

    def create_collection(self, recreate: bool = False) -> None:
        """Create the collection if it does not already exist."""
        exists = self._client.collection_exists(self._settings.collection_name)
        if exists and not recreate:
            logger.info(
                "qdrant_collection_exists",
                collection=self._settings.collection_name,
            )
            return

        if exists and recreate:
            self._client.delete_collection(self._settings.collection_name)

        try:
            self._client.create_collection(
                collection_name=self._settings.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self._settings.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except Exception as exc:
            raise VectorStoreOperationError(
                f"Failed to create collection: {self._settings.collection_name}"
            ) from exc

        logger.info(
            "qdrant_collection_created",
            collection=self._settings.collection_name,
            vector_size=self._settings.vector_size,
        )

    def insert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or upsert vector records into the collection."""
        if not records:
            return 0

        self._ensure_collection_exists()
        points = [
            qmodels.PointStruct(
                id=_to_point_id(record.id),
                vector=record.vector,
                payload=_build_payload(record),
            )
            for record in records
        ]

        try:
            self._client.upsert(
                collection_name=self._settings.collection_name,
                points=points,
            )
        except Exception as exc:
            raise VectorStoreOperationError("Failed to insert vector records") from exc

        logger.info(
            "qdrant_insert_complete",
            collection=self._settings.collection_name,
            count=len(points),
        )
        return len(points)

    def delete(
        self,
        ids: Sequence[str] | None = None,
        filters: MetadataFilter | None = None,
    ) -> None:
        """Delete records by point IDs and/or metadata filters."""
        if not ids and (filters is None or filters.is_empty()):
            raise VectorStoreConfigError("delete requires ids and/or metadata filters")

        selector = _build_delete_selector(ids=ids, filters=filters)
        try:
            self._client.delete(
                collection_name=self._settings.collection_name,
                points_selector=selector,
            )
        except Exception as exc:
            raise VectorStoreOperationError("Failed to delete vector records") from exc

        logger.info(
            "qdrant_delete_complete",
            collection=self._settings.collection_name,
            ids_count=len(ids or []),
            filtered=filters is not None and not filters.is_empty(),
        )

    def search(
        self,
        vector: list[float],
        limit: int = 10,
        filters: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors with optional metadata filters."""
        if limit <= 0:
            raise VectorStoreConfigError("limit must be greater than 0")
        if len(vector) != self._settings.vector_size:
            message = (
                f"query vector size {len(vector)} does not match "
                f"configured size {self._settings.vector_size}"
            )
            raise VectorStoreConfigError(message)

        query_filter = _build_query_filter(filters)
        try:
            response = self._client.query_points(
                collection_name=self._settings.collection_name,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
            )
        except Exception as exc:
            raise VectorStoreOperationError("Vector search failed") from exc

        return [
            SearchResult(
                id=str((result.payload or {}).get("record_id", result.id)),
                score=float(result.score or 0.0),
                payload=dict(result.payload or {}),
            )
            for result in response.points
        ]

    def health(self) -> dict[str, Any]:
        """Return Qdrant health and collection status."""
        try:
            collections = self._client.get_collections()
            collection_name = self._settings.collection_name
            collection_exists = self._client.collection_exists(collection_name)
            points_count = 0
            if collection_exists:
                info = self._client.get_collection(self._settings.collection_name)
                points_count = int(info.points_count or 0)

            collection_names = [
                collection.name for collection in collections.collections
            ]
            return {
                "status": "ok",
                "url": self._settings.url,
                "collection": collection_name,
                "collection_exists": collection_exists,
                "points_count": points_count,
                "collections": collection_names,
            }
        except Exception as exc:
            raise VectorStoreOperationError("Qdrant health check failed") from exc

    def _ensure_collection_exists(self) -> None:
        if not self._client.collection_exists(self._settings.collection_name):
            self.create_collection()


def _to_point_id(record_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, record_id))


def _build_payload(record: VectorRecord) -> dict[str, Any]:
    payload = dict(record.payload)
    payload["record_id"] = record.id
    return payload


def _build_query_filter(filters: MetadataFilter | None) -> qmodels.Filter | None:
    if filters is None or filters.is_empty():
        return None

    conditions: list[qmodels.Condition] = []
    if filters.document_id is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="document_id",
                match=qmodels.MatchValue(value=filters.document_id),
            )
        )
    if filters.source is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="source",
                match=qmodels.MatchValue(value=filters.source),
            )
        )
    if filters.page_number is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="page_number",
                match=qmodels.MatchValue(value=filters.page_number),
            )
        )
    if filters.chunk_id is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="chunk_id",
                match=qmodels.MatchValue(value=filters.chunk_id),
            )
        )

    return qmodels.Filter(must=conditions)


def _build_delete_selector(
    ids: Sequence[str] | None,
    filters: MetadataFilter | None,
) -> qmodels.PointsSelector:
    if ids and filters is not None and not filters.is_empty():
        point_ids = [_to_point_id(record_id) for record_id in ids]
        must_conditions: list[qmodels.Condition] = [
            qmodels.HasIdCondition(has_id=point_ids)
        ]
        query_filter = _build_query_filter(filters)
        if query_filter is not None and query_filter.must is not None:
            metadata_conditions = cast(list[qmodels.Condition], query_filter.must)
            must_conditions.extend(metadata_conditions)
        return qmodels.FilterSelector(filter=qmodels.Filter(must=must_conditions))

    if ids:
        point_ids = [_to_point_id(record_id) for record_id in ids]
        return qmodels.PointIdsList(points=point_ids)

    query_filter = _build_query_filter(filters)
    if query_filter is None:
        raise VectorStoreConfigError("delete requires ids and/or metadata filters")
    return qmodels.FilterSelector(filter=query_filter)
