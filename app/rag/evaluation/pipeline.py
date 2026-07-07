"""Ragas evaluation pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from app.rag.evaluation.dataset import load_evaluation_dataset, to_ragas_dataset
from app.rag.evaluation.metrics import score_citation_accuracy
from app.rag.evaluation.models import (
    EvaluationReport,
    EvaluationSample,
    MetricSummary,
    SampleScore,
)
from app.rag.generation.config import get_generation_settings


class RagasEvaluationPipeline:
    """Run Ragas metrics over a prepared evaluation dataset."""

    METRIC_NAMES = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "citation_accuracy",
    )

    def __init__(
        self,
        *,
        llm_model: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        settings = get_generation_settings()
        self._llm_model = llm_model or settings.model
        self._embedding_model = embedding_model
        self._openai_api_key = settings.openai_api_key

    def run(self, dataset_path: Path) -> EvaluationReport:
        """Evaluate the dataset and return structured results."""
        samples = load_evaluation_dataset(dataset_path)
        ragas_dataset = to_ragas_dataset(samples)
        result = evaluate(
            ragas_dataset,
            metrics=self._build_ragas_metrics(),
            llm=self._create_llm(),
            embeddings=self._create_embeddings(),
        )
        return self._build_report(dataset_path, samples, result)

    def _build_ragas_metrics(self) -> list[Any]:
        llm = self._create_llm()
        embeddings = self._create_embeddings()
        faithfulness.llm = llm
        answer_relevancy.llm = llm
        answer_relevancy.embeddings = embeddings
        context_precision.llm = llm
        return [
            faithfulness,
            answer_relevancy,
            context_precision,
        ]

    def _create_llm(self) -> Any:
        if not self._openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for Ragas evaluation metrics"
            )
        client = AsyncOpenAI(api_key=self._openai_api_key)
        return llm_factory(self._llm_model, client=client)

    def _create_embeddings(self) -> Any:
        if not self._openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for Ragas evaluation metrics"
            )
        from langchain_openai import OpenAIEmbeddings

        return LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=self._embedding_model,
                api_key=self._openai_api_key,  # type: ignore[call-arg]
            )
        )

    def _build_report(
        self,
        dataset_path: Path,
        samples: list[EvaluationSample],
        result: Any,
    ) -> EvaluationReport:
        dataframe = result.to_pandas()
        raw_scores: list[dict[str, float | None]] = []
        sample_scores: list[SampleScore] = []

        for index, sample in enumerate(samples):
            row_scores: dict[str, float | None] = {}
            ragas_metrics = (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
            )
            for metric_name in ragas_metrics:
                value = _safe_float(dataframe.iloc[index].get(metric_name))
                row_scores[metric_name] = value
            row_scores["citation_accuracy"] = score_citation_accuracy(
                sample.response,
                sample.expected_citations,
            )
            raw_scores.append(row_scores)
            sample_scores.append(
                SampleScore(
                    sample_id=sample.id,
                    question=sample.question,
                    scores=row_scores,
                    response_preview=_preview(sample.response),
                )
            )

        metric_summaries = tuple(
            _summarize_metric(metric_name, raw_scores)
            for metric_name in self.METRIC_NAMES
        )
        return EvaluationReport(
            dataset_path=str(dataset_path),
            sample_count=len(samples),
            metric_summaries=metric_summaries,
            sample_scores=tuple(sample_scores),
            raw_scores=raw_scores,
        )


def _summarize_metric(
    metric_name: str,
    raw_scores: list[dict[str, float | None]],
) -> MetricSummary:
    values = [
        score
        for row in raw_scores
        if (score := row.get(metric_name)) is not None and not math.isnan(score)
    ]
    if not values:
        return MetricSummary(
            name=metric_name,
            mean=0.0,
            min_score=0.0,
            max_score=0.0,
        )
    return MetricSummary(
        name=metric_name,
        mean=sum(values) / len(values),
        min_score=min(values),
        max_score=max(values),
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."
