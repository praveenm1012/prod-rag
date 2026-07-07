"""Hybrid BM25 + vector retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from app.core.logging import get_logger
from app.rag.lexical.models import LexicalDocument
from app.rag.lexical.searcher import BM25Searcher
from app.rag.retrieval.config import (
    HybridRetrievalSettings,
    get_hybrid_retrieval_settings,
)
from app.rag.retrieval.exceptions import (
    HybridRetrievalConfigError,
    HybridRetrievalStateError,
)
from app.rag.retrieval.models import HybridSearchResult
from app.rag.retrieval.normalizer import min_max_normalize
from app.rag.vectorstore.models import MetadataFilter, VectorRecord

if TYPE_CHECKING:
    from app.rag.embeddings.service import EmbeddingService
    from app.rag.vectorstore.repository import QdrantRepository

logger = get_logger(__name__)


class HybridRetriever:
    """Combine BM25 lexical search and vector similarity with weighted score fusion."""

    def __init__(
        self,
        settings: HybridRetrievalSettings | None = None,
        *,
        bm25_searcher: BM25Searcher | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: QdrantRepository | None = None,
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> None:
        self._settings = settings or get_hybrid_retrieval_settings()
        self._bm25_weight = (
            bm25_weight if bm25_weight is not None else self._settings.bm25_weight
        )
        self._vector_weight = (
            vector_weight if vector_weight is not None else self._settings.vector_weight
        )

        if self._bm25_weight < 0 or self._vector_weight < 0:
            raise HybridRetrievalConfigError(
                "weights must be greater than or equal to 0"
            )
        if self._bm25_weight == 0 and self._vector_weight == 0:
            raise HybridRetrievalConfigError(
                "at least one retrieval weight must be > 0"
            )

        self._bm25 = bm25_searcher or BM25Searcher()
        self._embedding_service = embedding_service
        self._vector_store = vector_store

        self._documents: dict[str, LexicalDocument] = {}
        self._vectors: dict[str, list[float]] = {}

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def bm25_weight(self) -> float:
        return self._bm25_weight

    @property
    def vector_weight(self) -> float:
        return self._vector_weight

    def index(
        self,
        documents: Sequence[LexicalDocument],
        vectors: Sequence[list[float]] | None = None,
    ) -> int:
        """Index documents for BM25 and optional in-memory / Qdrant vector search."""
        if vectors is not None and len(vectors) != len(documents):
            raise HybridRetrievalConfigError("documents and vectors length must match")

        self._documents = {document.chunk_id: document for document in documents}
        self._vectors = {}
        if vectors is not None:
            self._vectors = {
                document.chunk_id: list(vector)
                for document, vector in zip(documents, vectors, strict=True)
            }

        self._bm25.index(documents)

        if self._vector_store is not None and vectors is not None:
            records = [
                VectorRecord(
                    id=document.chunk_id,
                    vector=list(vector),
                    payload=_document_payload(document),
                )
                for document, vector in zip(documents, vectors, strict=True)
            ]
            self._vector_store.create_collection(recreate=True)
            self._vector_store.insert(records)

        logger.info(
            "hybrid_index_built",
            document_count=len(documents),
            vector_backed=bool(vectors),
            qdrant_backed=self._vector_store is not None,
        )
        return len(documents)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilter | None = None,
        query_vector: list[float] | None = None,
    ) -> list[HybridSearchResult]:
        """Search using normalized BM25 and vector scores with weighted fusion."""
        if not self._documents:
            raise HybridRetrievalStateError("index must be built before searching")

        limit = top_k if top_k is not None else self._settings.top_k
        pool_size = max(limit, self._settings.candidate_pool_size)
        if limit <= 0:
            raise HybridRetrievalConfigError("top_k must be greater than 0")

        bm25_scores = _collect_bm25_scores(
            self._bm25,
            query=query,
            pool_size=pool_size,
            filters=filters,
        )
        vector_scores = self._collect_vector_scores(
            query=query,
            pool_size=pool_size,
            filters=filters,
            query_vector=query_vector,
        )

        bm25_normalized = min_max_normalize(bm25_scores)
        vector_normalized = min_max_normalize(vector_scores)

        candidate_ids = set(bm25_scores) | set(vector_scores)
        weight_total = self._bm25_weight + self._vector_weight
        merged: list[tuple[float, str]] = []

        for chunk_id in candidate_ids:
            bm25_component = bm25_normalized.get(chunk_id, 0.0)
            vector_component = vector_normalized.get(chunk_id, 0.0)
            hybrid_score = (
                self._bm25_weight * bm25_component
                + self._vector_weight * vector_component
            ) / weight_total
            merged.append((hybrid_score, chunk_id))

        merged.sort(key=lambda item: item[0], reverse=True)
        top_results = merged[:limit]

        results = [
            _build_result(
                chunk_id=chunk_id,
                hybrid_score=hybrid_score,
                rank=index,
                document=self._documents[chunk_id],
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                vector_score=vector_scores.get(chunk_id, 0.0),
                bm25_score_normalized=bm25_normalized.get(chunk_id, 0.0),
                vector_score_normalized=vector_normalized.get(chunk_id, 0.0),
            )
            for index, (hybrid_score, chunk_id) in enumerate(top_results, start=1)
            if chunk_id in self._documents
        ]

        logger.info(
            "hybrid_search_complete",
            query_length=len(query),
            bm25_candidates=len(bm25_scores),
            vector_candidates=len(vector_scores),
            hits=len(results),
            top_k=limit,
        )
        return results

    def _collect_vector_scores(
        self,
        query: str,
        pool_size: int,
        filters: MetadataFilter | None,
        query_vector: list[float] | None,
    ) -> dict[str, float]:
        if self._vector_store is not None:
            vector = query_vector or self._embed_query(query)
            hits = self._vector_store.search(
                vector=vector,
                limit=pool_size,
                filters=filters,
            )
            return {hit.id: hit.score for hit in hits}

        if not self._vectors:
            return {}

        vector = query_vector or self._embed_query(query)
        return _in_memory_vector_scores(
            query_vector=vector,
            vectors=self._vectors,
            documents=self._documents,
            pool_size=pool_size,
            filters=filters,
        )

    def _embed_query(self, query: str) -> list[float]:
        if self._embedding_service is None:
            raise HybridRetrievalStateError(
                "query_vector is required when no embedding service is configured"
            )
        record = self._embedding_service.embed_text(query, prompt_type="query")
        return record.vector


def _collect_bm25_scores(
    bm25: BM25Searcher,
    query: str,
    pool_size: int,
    filters: MetadataFilter | None,
) -> dict[str, float]:
    hits = bm25.search(query, top_k=pool_size, filters=filters)
    return {hit.chunk_id: hit.score for hit in hits}


def _in_memory_vector_scores(
    query_vector: list[float],
    vectors: dict[str, list[float]],
    documents: dict[str, LexicalDocument],
    pool_size: int,
    filters: MetadataFilter | None,
) -> dict[str, float]:
    query = np.asarray(query_vector, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0:
        return {}

    ranked: list[tuple[float, str]] = []
    for chunk_id, vector in vectors.items():
        document = documents[chunk_id]
        if filters is not None and not _matches_filter(document, filters):
            continue
        candidate = np.asarray(vector, dtype=np.float32)
        denominator = float(np.linalg.norm(candidate)) * query_norm
        if denominator == 0:
            continue
        score = float(np.dot(query, candidate) / denominator)
        if score > 0:
            ranked.append((score, chunk_id))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return {chunk_id: score for score, chunk_id in ranked[:pool_size]}


def _matches_filter(document: LexicalDocument, filters: MetadataFilter) -> bool:
    if filters.is_empty():
        return True
    if filters.document_id is not None and document.document_id != filters.document_id:
        return False
    if filters.source is not None and document.source != filters.source:
        return False
    if filters.page_number is not None and document.page_number != filters.page_number:
        return False
    return filters.chunk_id is None or document.chunk_id == filters.chunk_id


def _document_payload(document: LexicalDocument) -> dict[str, object]:
    payload: dict[str, object] = {
        "chunk_id": document.chunk_id,
        "document_id": document.document_id,
        "source": document.source,
        "page_number": document.page_number,
        "text": document.text,
    }
    payload.update(document.metadata)
    return payload


def _build_result(
    chunk_id: str,
    hybrid_score: float,
    rank: int,
    document: LexicalDocument,
    bm25_score: float,
    vector_score: float,
    bm25_score_normalized: float,
    vector_score_normalized: float,
) -> HybridSearchResult:
    return HybridSearchResult(
        chunk_id=chunk_id,
        text=document.text,
        score=hybrid_score,
        rank=rank,
        bm25_score=bm25_score,
        vector_score=vector_score,
        bm25_score_normalized=bm25_score_normalized,
        vector_score_normalized=vector_score_normalized,
        document_id=document.document_id,
        page_number=document.page_number,
        source=document.source,
        metadata=dict(document.metadata),
    )
