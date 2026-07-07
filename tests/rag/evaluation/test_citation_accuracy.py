"""Citation accuracy metric tests."""

from app.rag.evaluation.metrics import score_citation_accuracy


def test_citation_accuracy_full_match() -> None:
    score = score_citation_accuracy(
        "Acme is based in Springfield [1].",
        ["[1]"],
    )
    assert score == 1.0


def test_citation_accuracy_partial_match() -> None:
    score = score_citation_accuracy(
        "Acme is based in Springfield [1].",
        ["[1]", "[2]"],
    )
    assert score == 0.5


def test_citation_accuracy_no_expected_citations() -> None:
    score = score_citation_accuracy("Plain answer without citations.", [])
    assert score == 1.0
