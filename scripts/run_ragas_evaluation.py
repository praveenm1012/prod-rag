#!/usr/bin/env python3
"""Run Ragas evaluation and generate an HTML report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.rag.evaluation import RagasEvaluationPipeline, write_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG quality with Ragas and generate an HTML report.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluation/sample_rag_dataset.json"),
        help="Path to the evaluation dataset JSON file",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/ragas_evaluation.html"),
        help="Output HTML report path",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the evaluator LLM model (defaults to LLM_MODEL from .env)",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model for answer relevancy",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON summary to stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    pipeline = RagasEvaluationPipeline(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
    )
    report = pipeline.run(args.dataset)
    report_path = write_html_report(report, args.report_path)

    payload = {
        "status": "pass",
        "dataset": str(args.dataset),
        "report_path": str(report_path),
        "sample_count": report.sample_count,
        "metrics": report.metric_means(),
    }

    if args.json_only:
        print(json.dumps(payload, indent=2))
    else:
        print("Ragas evaluation complete")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {report.sample_count}")
        print(f"Report: {report_path}")
        print("\nMetric means")
        print("-" * 40)
        for name, value in report.metric_means().items():
            print(f"{name}: {value:.4f}")
        print("\nJSON")
        print("-" * 40)
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
