"""Reranking unit tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.reranking import (
    CrossEncoderReranker,
    RerankCandidate,
    RerankerConfigError,
    RerankerInputError,
)
from app.rag.reranking.config import RerankerSettings


@pytest.fixture
def reranker() -> CrossEncoderReranker:
    settings = RerankerSettings(
        RERANKER_MODEL="BAAI/bge-reranker-large",
        RERANKER_DEVICE="cpu",
        RERANKER_BATCH_SIZE=32,
        RERANKER_TOP_K=5,
        RERANKER_SHOW_PROGRESS=False,
    )
    return CrossEncoderReranker(settings=settings)


@pytest.fixture
def candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(chunk_id="a", text="Python is a programming language"),
        RerankCandidate(chunk_id="b", text="Cats enjoy sleeping on sunny windowsills"),
        RerankCandidate(
            chunk_id="c",
            text="Python supports machine learning libraries",
        ),
        RerankCandidate(chunk_id="d", text="Dogs like to play fetch in the park"),
        RerankCandidate(chunk_id="e", text="FastAPI is a Python web framework"),
        RerankCandidate(chunk_id="f", text="JavaScript runs in web browsers"),
        RerankCandidate(
            chunk_id="g",
            text="NumPy provides numerical computing in Python",
        ),
    ]


def _mock_cross_encoder(scores: list[float]) -> MagicMock:
    model = MagicMock()
    model.predict.return_value = np.asarray(scores, dtype=np.float32)
    return model


class TestCrossEncoderRerankerConfig:
    def test_rejects_invalid_batch_size(self) -> None:
        settings = RerankerSettings(RERANKER_BATCH_SIZE=0)
        with pytest.raises(RerankerConfigError, match="batch_size"):
            CrossEncoderReranker(settings=settings)

    def test_rejects_invalid_top_k(self) -> None:
        settings = RerankerSettings(RERANKER_TOP_K=0)
        with pytest.raises(RerankerConfigError, match="top_k"):
            CrossEncoderReranker(settings=settings)


class TestCrossEncoderReranker:
    def test_rejects_empty_question(
        self,
        reranker: CrossEncoderReranker,
        candidates: list[RerankCandidate],
    ) -> None:
        with pytest.raises(RerankerInputError, match="question must not be empty"):
            reranker.rerank("   ", candidates)

    def test_returns_empty_for_no_candidates(
        self,
        reranker: CrossEncoderReranker,
    ) -> None:
        assert reranker.rerank("What is Python?", []) == []

    @patch("app.rag.reranking.service.CrossEncoderReranker._get_model")
    def test_returns_top_5_by_score(
        self,
        mock_get_model: MagicMock,
        reranker: CrossEncoderReranker,
        candidates: list[RerankCandidate],
    ) -> None:
        scores = [0.1, 0.2, 0.9, 0.3, 0.85, 0.05, 0.95]
        mock_get_model.return_value = _mock_cross_encoder(scores)

        results = reranker.rerank("What is Python?", candidates)

        assert len(results) == 5
        assert [result.chunk_id for result in results] == ["g", "c", "e", "d", "b"]
        assert results[0].rank == 1
        assert results[0].score == pytest.approx(0.95)
        assert results[-1].rank == 5

    @patch("app.rag.reranking.service.CrossEncoderReranker._get_model")
    def test_returns_fewer_when_less_than_top_k(
        self,
        mock_get_model: MagicMock,
        reranker: CrossEncoderReranker,
    ) -> None:
        small_candidates = [
            RerankCandidate(chunk_id="a", text="first"),
            RerankCandidate(chunk_id="b", text="second"),
        ]
        mock_get_model.return_value = _mock_cross_encoder([0.4, 0.8])

        results = reranker.rerank("question", small_candidates)

        assert len(results) == 2
        assert results[0].chunk_id == "b"
        assert results[1].chunk_id == "a"

    @patch("app.rag.reranking.service.CrossEncoderReranker._get_model")
    def test_respects_custom_top_k(
        self,
        mock_get_model: MagicMock,
        reranker: CrossEncoderReranker,
        candidates: list[RerankCandidate],
    ) -> None:
        scores = [0.1, 0.2, 0.9, 0.3, 0.85, 0.05, 0.95]
        mock_get_model.return_value = _mock_cross_encoder(scores)

        results = reranker.rerank("What is Python?", candidates, top_k=2)

        assert len(results) == 2
        assert [result.chunk_id for result in results] == ["g", "c"]

    @patch("app.rag.reranking.service.CrossEncoderReranker._get_model")
    def test_results_sorted_descending_by_score(
        self,
        mock_get_model: MagicMock,
        reranker: CrossEncoderReranker,
        candidates: list[RerankCandidate],
    ) -> None:
        scores = [0.1, 0.2, 0.9, 0.3, 0.85, 0.05, 0.95]
        mock_get_model.return_value = _mock_cross_encoder(scores)

        results = reranker.rerank("What is Python?", candidates)

        observed_scores = [result.score for result in results]
        assert observed_scores == sorted(observed_scores, reverse=True)

    @patch("app.rag.reranking.service.CrossEncoderReranker._get_model")
    def test_passes_question_candidate_pairs_to_model(
        self,
        mock_get_model: MagicMock,
        reranker: CrossEncoderReranker,
    ) -> None:
        model = _mock_cross_encoder([0.7, 0.2])
        mock_get_model.return_value = model
        candidates = [
            RerankCandidate(chunk_id="a", text="alpha"),
            RerankCandidate(chunk_id="b", text="beta"),
        ]

        reranker.rerank("What is alpha?", candidates, top_k=2)

        model.predict.assert_called_once()
        pairs = model.predict.call_args.args[0]
        assert pairs == [
            ("What is alpha?", "alpha"),
            ("What is alpha?", "beta"),
        ]

    @patch("app.rag.reranking.service.CrossEncoderReranker._get_model")
    def test_preserves_candidate_metadata(
        self,
        mock_get_model: MagicMock,
        reranker: CrossEncoderReranker,
    ) -> None:
        mock_get_model.return_value = _mock_cross_encoder([0.9])
        candidates = [
            RerankCandidate(
                chunk_id="a",
                text="alpha",
                metadata={"source": "/docs/a.txt", "page_number": 2},
            ),
        ]

        results = reranker.rerank("question", candidates, top_k=1)

        assert results[0].metadata == {"source": "/docs/a.txt", "page_number": 2}
