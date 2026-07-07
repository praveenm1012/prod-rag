"""Cross-encoder reranking service."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from app.core.logging import get_logger
from app.rag.embeddings.service import resolve_device
from app.rag.reranking.config import RerankerSettings, get_reranker_settings
from app.rag.reranking.exceptions import (
    RerankerConfigError,
    RerankerInputError,
    RerankerModelError,
)
from app.rag.reranking.models import RerankCandidate, RerankResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Rerank candidate chunks with a BGE cross-encoder model."""

    def __init__(self, settings: RerankerSettings | None = None) -> None:
        self._settings = settings or get_reranker_settings()
        if self._settings.batch_size <= 0:
            raise RerankerConfigError("batch_size must be greater than 0")
        if self._settings.top_k <= 0:
            raise RerankerConfigError("top_k must be greater than 0")
        if self._settings.max_retries <= 0:
            raise RerankerConfigError("max_retries must be greater than 0")

        self._device = resolve_device(self._settings.device)
        self._model: CrossEncoder | None = None
        logger.info(
            "reranker_initialized",
            model=self._settings.model_name,
            device=self._device,
            batch_size=self._settings.batch_size,
            top_k=self._settings.top_k,
        )

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def default_top_k(self) -> int:
        return self._settings.top_k

    def rerank(
        self,
        question: str,
        candidates: Sequence[RerankCandidate],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Score question-candidate pairs and return the top-ranked results."""
        normalized_question = question.strip()
        if not normalized_question:
            raise RerankerInputError("question must not be empty")
        if not candidates:
            return []

        limit = top_k if top_k is not None else self._settings.top_k
        if limit <= 0:
            raise RerankerConfigError("top_k must be greater than 0")

        pairs = [(normalized_question, candidate.text) for candidate in candidates]
        scores = self._score_pairs(pairs)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[:limit]

        results = [
            RerankResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=float(score),
                rank=index,
                metadata=dict(candidate.metadata),
            )
            for index, (candidate, score) in enumerate(ranked, start=1)
        ]

        logger.info(
            "rerank_complete",
            question_length=len(normalized_question),
            candidate_count=len(candidates),
            returned=len(results),
            top_k=limit,
        )
        return results

    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        batch_size = self._settings.batch_size
        scores: list[float] = []
        batch_ranges = range(0, len(pairs), batch_size)
        progress = tqdm(
            batch_ranges,
            desc="Reranking candidates",
            disable=not self._settings.show_progress,
        )

        for start in progress:
            batch = pairs[start : start + batch_size]
            batch_scores = self._predict_with_retry(batch)
            scores.extend(batch_scores)
            progress.set_postfix(
                {
                    "batch": f"{start // batch_size + 1}",
                    "device": self._device,
                }
            )

        return scores

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerModelError(
                    "sentence-transformers is required for cross-encoder reranking"
                ) from exc

            try:
                logger.info(
                    "reranker_model_loading",
                    model=self._settings.model_name,
                    device=self._device,
                )
                self._model = CrossEncoder(
                    self._settings.model_name,
                    device=self._device,
                )
            except Exception as exc:
                raise RerankerModelError(
                    f"Failed to load reranker model: {self._settings.model_name}"
                ) from exc

        return self._model

    def _predict_with_retry(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[float]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self._settings.retry_min_seconds,
                max=self._settings.retry_max_seconds,
            ),
            retry=retry_if_exception_type(Exception),
        )
        def _predict() -> list[float]:
            model = self._get_model()
            try:
                raw_scores = model.predict(
                    list(pairs),
                    batch_size=len(pairs),
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise RerankerModelError("Cross-encoder scoring failed") from exc

            array = np.asarray(raw_scores, dtype=np.float32)
            if array.ndim == 0:
                return [float(array)]
            return [float(score) for score in array]

        return _predict()
