"""HTML report generation tests."""

from app.rag.evaluation.models import EvaluationReport, MetricSummary, SampleScore
from app.rag.evaluation.report import render_html_report, write_html_report


def test_render_html_report_contains_metrics(tmp_path) -> None:
    report = EvaluationReport(
        dataset_path="data/evaluation/sample_rag_dataset.json",
        sample_count=2,
        metric_summaries=(
            MetricSummary("faithfulness", 0.9, 0.8, 1.0),
            MetricSummary("answer_relevancy", 0.85, 0.7, 1.0),
            MetricSummary("context_precision", 0.95, 0.9, 1.0),
            MetricSummary("citation_accuracy", 0.75, 0.5, 1.0),
        ),
        sample_scores=(
            SampleScore(
                sample_id="acme-hq",
                question="Where is Acme headquartered?",
                scores={
                    "faithfulness": 1.0,
                    "answer_relevancy": 0.9,
                    "context_precision": 1.0,
                    "citation_accuracy": 1.0,
                },
                response_preview="Springfield [1].",
            ),
        ),
        raw_scores=[
            {
                "faithfulness": 1.0,
                "answer_relevancy": 0.9,
                "context_precision": 1.0,
                "citation_accuracy": 1.0,
            }
        ],
    )

    html = render_html_report(report)
    assert "Faithfulness" in html
    assert "Citation Accuracy" in html
    assert "acme-hq" in html

    output_path = tmp_path / "report.html"
    write_html_report(report, output_path)
    assert output_path.exists()
    assert "<!DOCTYPE html>" in output_path.read_text(encoding="utf-8")
