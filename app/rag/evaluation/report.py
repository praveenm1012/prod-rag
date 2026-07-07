"""HTML report generation for Ragas evaluation."""

from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path

from app.rag.evaluation.models import EvaluationReport, MetricSummary, SampleScore

_METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "citation_accuracy": "Citation Accuracy",
}


def render_html_report(report: EvaluationReport) -> str:
    """Render an HTML evaluation report."""
    generated_at = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    metric_cards = "\n".join(
        _metric_card(summary) for summary in report.metric_summaries
    )
    sample_rows = "\n".join(
        _sample_row(sample_score) for sample_score in report.sample_scores
    )
    json_payload = html.escape(json.dumps(_report_payload(report), indent=2))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG Evaluation Report</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0f172a;
      --panel: #111827;
      --card: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #38bdf8;
      --good: #34d399;
      --warn: #fbbf24;
      --bad: #f87171;
      --border: #374151;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
      background: linear-gradient(180deg, #020617 0%, var(--bg) 100%);
      color: var(--text);
      line-height: 1.5;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    h1 {{ font-size: 2rem; }}
    h2 {{ font-size: 1.25rem; margin-top: 32px; }}
    .meta {{
      color: var(--muted);
      margin-bottom: 24px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: rgba(31, 41, 55, 0.9);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
    }}
    .card h3 {{
      margin: 0 0 8px;
      font-size: 0.95rem;
      color: var(--muted);
      font-weight: 600;
    }}
    .score {{
      font-size: 2rem;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .bar {{
      height: 8px;
      border-radius: 999px;
      background: #111827;
      overflow: hidden;
      border: 1px solid var(--border);
    }}
    .bar > span {{
      display: block;
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--good));
    }}
    .range {{
      margin-top: 8px;
      font-size: 0.85rem;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      background: rgba(17, 24, 39, 0.75);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: rgba(15, 23, 42, 0.9);
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    tr:last-child td {{ border-bottom: none; }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.15);
      color: var(--accent);
    }}
    .metric-value {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
    .metric-good {{ color: var(--good); }}
    .metric-mid {{ color: var(--warn); }}
    .metric-bad {{ color: var(--bad); }}
    .json-block {{
      margin-top: 24px;
      background: rgba(15, 23, 42, 0.9);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 0.85rem;
      color: #cbd5e1;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>RAG Evaluation Report</h1>
    <div class="meta">
      Generated {html.escape(generated_at)} ·
      Dataset: <code>{html.escape(report.dataset_path)}</code> ·
      Samples: {report.sample_count}
    </div>

    <h2>Aggregate Metrics</h2>
    <div class="cards">
      {metric_cards}
    </div>

    <h2>Per-sample Results</h2>
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Question</th>
          <th>Faithfulness</th>
          <th>Answer Relevancy</th>
          <th>Context Precision</th>
          <th>Citation Accuracy</th>
        </tr>
      </thead>
      <tbody>
        {sample_rows}
      </tbody>
    </table>

    <h2>Raw Summary JSON</h2>
    <div class="json-block">{json_payload}</div>
  </div>
</body>
</html>
"""


def write_html_report(report: EvaluationReport, output_path: Path) -> Path:
    """Write the HTML report to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html_report(report), encoding="utf-8")
    return output_path


def _metric_card(summary: MetricSummary) -> str:
    label = _METRIC_LABELS.get(summary.name, summary.name)
    width = max(0.0, min(summary.mean, 1.0)) * 100
    return f"""
    <div class="card">
      <h3>{html.escape(label)}</h3>
      <div class="score">{summary.mean:.3f}</div>
      <div class="bar"><span style="width: {width:.1f}%"></span></div>
      <div class="range">min {summary.min_score:.3f} · max {summary.max_score:.3f}</div>
    </div>
    """


def _sample_row(sample_score: SampleScore) -> str:
    cells = [
        f'<span class="pill">{html.escape(sample_score.sample_id)}</span>',
        html.escape(sample_score.question),
    ]
    for metric_name in (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "citation_accuracy",
    ):
        cells.append(_metric_cell(sample_score.scores.get(metric_name)))
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _metric_cell(value: float | None) -> str:
    if value is None:
        return '<span class="metric-value metric-mid">n/a</span>'
    if value >= 0.8:
        css_class = "metric-good"
    elif value >= 0.5:
        css_class = "metric-mid"
    else:
        css_class = "metric-bad"
    return f'<span class="metric-value {css_class}">{value:.3f}</span>'


def _report_payload(report: EvaluationReport) -> dict[str, object]:
    return {
        "dataset_path": report.dataset_path,
        "sample_count": report.sample_count,
        "metrics": {
            summary.name: {
                "mean": round(summary.mean, 4),
                "min": round(summary.min_score, 4),
                "max": round(summary.max_score, 4),
            }
            for summary in report.metric_summaries
        },
        "samples": [
            {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "scores": sample.scores,
            }
            for sample in report.sample_scores
        ],
    }
