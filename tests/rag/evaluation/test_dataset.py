"""Evaluation dataset loader tests."""

from pathlib import Path

import pytest

from app.rag.evaluation.dataset import load_evaluation_dataset, to_ragas_dataset


def test_load_sample_dataset() -> None:
    path = Path("data/evaluation/sample_rag_dataset.json")
    samples = load_evaluation_dataset(path)

    assert len(samples) == 5
    assert samples[0].id == "acme-hq"
    assert samples[0].expected_citations == ["[1]"]


def test_to_ragas_dataset_includes_rubric() -> None:
    path = Path("data/evaluation/sample_rag_dataset.json")
    samples = load_evaluation_dataset(path)
    dataset = to_ragas_dataset(samples)

    assert dataset[0]["user_input"].startswith("Where is the headquarters")
    assert dataset[0]["rubrics"]["expected_citations"] == '["[1]"]'


def test_load_dataset_rejects_invalid_sample(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bad.json"
    dataset_path.write_text('{"samples": [{"id": "x"}]}', encoding="utf-8")

    with pytest.raises(ValueError, match="requires a non-empty"):
        load_evaluation_dataset(dataset_path)
