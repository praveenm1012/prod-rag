"""Evaluation dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from app.rag.evaluation.models import EvaluationSample


def load_evaluation_dataset(path: Path) -> list[EvaluationSample]:
    """Load evaluation samples from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_samples = payload.get("samples", payload)
    if not isinstance(raw_samples, list):
        raise ValueError("evaluation dataset must contain a list of samples")

    samples: list[EvaluationSample] = []
    for index, item in enumerate(raw_samples):
        if not isinstance(item, dict):
            raise ValueError(f"sample at index {index} must be an object")
        samples.append(_parse_sample(item, index))
    return samples


def to_ragas_dataset(samples: list[EvaluationSample]) -> Dataset:
    """Convert evaluation samples to a Hugging Face dataset for Ragas."""
    records: dict[str, list[Any]] = {
        "sample_id": [],
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
        "rubrics": [],
    }
    for sample in samples:
        records["sample_id"].append(sample.id)
        records["user_input"].append(sample.question)
        records["response"].append(sample.response)
        records["retrieved_contexts"].append(sample.contexts)
        records["reference"].append(sample.reference)
        records["rubrics"].append(
            {
                "expected_citations": json.dumps(sample.expected_citations),
            }
        )
    return Dataset.from_dict(records)


def _parse_sample(item: dict[str, Any], index: int) -> EvaluationSample:
    sample_id = str(item.get("id") or f"sample-{index + 1}")
    question = _require_str(item, "question", sample_id)
    reference = _require_str(item, "reference", sample_id)
    response = _require_str(item, "response", sample_id)
    contexts = item.get("contexts", [])
    if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
        raise ValueError(f"sample '{sample_id}' contexts must be a list of strings")

    expected_citations = item.get("expected_citations", [])
    if not isinstance(expected_citations, list) or not all(
        isinstance(citation, str) for citation in expected_citations
    ):
        raise ValueError(
            f"sample '{sample_id}' expected_citations must be a list of strings"
        )

    metadata = item.get("metadata", {})
    if metadata and not isinstance(metadata, dict):
        raise ValueError(f"sample '{sample_id}' metadata must be an object")

    return EvaluationSample(
        id=sample_id,
        question=question,
        reference=reference,
        response=response,
        contexts=contexts,
        expected_citations=expected_citations,
        metadata=metadata,
    )


def _require_str(item: dict[str, Any], key: str, sample_id: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"sample '{sample_id}' requires a non-empty '{key}' field")
    return value.strip()
