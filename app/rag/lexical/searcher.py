"""BM25 lexical search implementation."""

from __future__ import annotations

from collections.abc import Sequence

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.rag.lexical.config import LexicalSearchSettings, get_lexical_search_settings
from app.rag.lexical.exceptions import LexicalSearchConfigError, LexicalSearchStateError
from app.rag.lexical.models import LexicalDocument, LexicalSearchResult
from app.rag.lexical.tokenizer import tokenize
from app.rag.vectorstore.models import MetadataFilter

logger = get_logger(__name__)


class BM25Searcher:
    """In-memory BM25 lexical search over document chunks."""

    def __init__(
        self,
        settings: LexicalSearchSettings | None = None,
        *,
        k1: float | None = None,
        b: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self._settings = settings or get_lexical_search_settings()
        self._k1 = k1 if k1 is not None else self._settings.k1
        self._b = b if b is not None else self._settings.b
        self._default_top_k = top_k if top_k is not None else self._settings.top_k

        if self._k1 <= 0:
            raise LexicalSearchConfigError("k1 must be greater than 0")
        if not 0 <= self._b <= 1:
            raise LexicalSearchConfigError("b must be between 0 and 1")
        if self._default_top_k <= 0:
            raise LexicalSearchConfigError("top_k must be greater than 0")

        self._documents: list[LexicalDocument] = []
        self._corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def index(self, documents: Sequence[LexicalDocument]) -> int:
        """Build the BM25 index from documents."""
        self._documents = list(documents)
        self._corpus = [tokenize(document.text) for document in self._documents]
        if self._corpus:
            self._bm25 = BM25Okapi(self._corpus, k1=self._k1, b=self._b)
        else:
            self._bm25 = None

        logger.info("lexical_index_built", document_count=len(self._documents))
        return len(self._documents)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilter | None = None,
    ) -> list[LexicalSearchResult]:
        """Search the index and return top-k ranked results."""
        if self._bm25 is None:
            raise LexicalSearchStateError("index must be built before searching")

        limit = top_k if top_k is not None else self._default_top_k
        if limit <= 0:
            raise LexicalSearchConfigError("top_k must be greater than 0")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_candidates: list[tuple[float, LexicalDocument]] = []

        for score, document in zip(scores, self._documents, strict=True):
            if score <= 0:
                continue
            if filters is not None and not _matches_filter(document, filters):
                continue
            ranked_candidates.append((float(score), document))

        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        top_results = ranked_candidates[:limit]

        results = [
            LexicalSearchResult(
                chunk_id=document.chunk_id,
                text=document.text,
                score=score,
                rank=index,
                document_id=document.document_id,
                page_number=document.page_number,
                source=document.source,
                metadata=dict(document.metadata),
            )
            for index, (score, document) in enumerate(top_results, start=1)
        ]

        logger.info(
            "lexical_search_complete",
            query_terms=len(query_tokens),
            hits=len(results),
            top_k=limit,
        )
        return results


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
