#!/usr/bin/env python3
"""Benchmark hybrid retrieval against BM25-only and vector-only baselines."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.rag.lexical import LexicalDocument
from app.rag.retrieval.evaluation import mean_reciprocal_rank, recall_at_k
from app.rag.retrieval.hybrid import HybridRetriever
from app.rag.retrieval.report import (
    EvaluationSummary,
    QueryEvaluation,
    write_evaluation_report,
)


@dataclass(frozen=True)
class EvalQuery:
    query: str
    query_vector: list[float]
    relevant_ids: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark hybrid retrieval.")
    parser.add_argument("--top-k", type=int, default=3, help="Top K results")
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=0.5,
        help="BM25 weight for hybrid fusion",
    )
    parser.add_argument(
        "--vector-weight",
        type=float,
        default=0.5,
        help="Vector weight for hybrid fusion",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/hybrid_evaluation.md"),
        help="Output markdown evaluation report",
    )
    return parser.parse_args()


def build_corpus() -> tuple[list[LexicalDocument], list[list[float]], list[EvalQuery]]:
    documents_and_vectors = [
        (
            LexicalDocument(
                chunk_id="lex:1",
                text="BM25 lexical keyword search ranking function",
                document_id="lex",
                page_number=1,
                source="/docs/lex.txt",
            ),
            [1.0, 0.0, 0.0, 0.0],
        ),
        (
            LexicalDocument(
                chunk_id="sem:1",
                text="Semantic embeddings capture meaning for similarity retrieval",
                document_id="sem",
                page_number=1,
                source="/docs/sem.txt",
            ),
            [0.0, 1.0, 0.0, 0.0],
        ),
        (
            LexicalDocument(
                chunk_id="hyb:1",
                text="Hybrid retrieval combines lexical BM25 and vector search",
                document_id="hyb",
                page_number=1,
                source="/docs/hyb.txt",
            ),
            [0.7, 0.7, 0.0, 0.0],
        ),
        (
            LexicalDocument(
                chunk_id="ops:1",
                text="Operations dashboard monitors latency throughput and errors",
                document_id="ops",
                page_number=1,
                source="/docs/ops.txt",
            ),
            [0.0, 0.0, 1.0, 0.0],
        ),
    ]

    documents = [item[0] for item in documents_and_vectors]
    vectors = [item[1] for item in documents_and_vectors]
    eval_queries = [
        EvalQuery(
            query="BM25 lexical keyword search",
            query_vector=_unit_vector(0),
            relevant_ids={"lex:1"},
        ),
        EvalQuery(
            query="semantic embeddings meaning",
            query_vector=_unit_vector(1),
            relevant_ids={"sem:1"},
        ),
        EvalQuery(
            query="hybrid BM25 vector retrieval",
            query_vector=_normalize([0.7, 0.7, 0.0, 0.0]),
            relevant_ids={"hyb:1"},
        ),
        EvalQuery(
            query="latency throughput operations dashboard",
            query_vector=_unit_vector(2),
            relevant_ids={"ops:1"},
        ),
    ]
    return documents, vectors, eval_queries


def _unit_vector(index: int, size: int = 4) -> list[float]:
    vector = np.zeros(size, dtype=np.float32)
    vector[index] = 1.0
    return vector.tolist()


def _normalize(vector: list[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return array.tolist()
    return (array / norm).tolist()


def _timed_search(
    retriever: HybridRetriever,
    query: EvalQuery,
    top_k: int,
) -> tuple[list[str], float]:
    start = time.perf_counter()
    results = retriever.search(
        query.query,
        top_k=top_k,
        query_vector=query.query_vector,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return [result.chunk_id for result in results], elapsed_ms


def main() -> None:
    args = parse_args()
    documents, vectors, eval_queries = build_corpus()

    bm25_only = HybridRetriever(bm25_weight=1.0, vector_weight=0.0)
    vector_only = HybridRetriever(bm25_weight=0.0, vector_weight=1.0)
    hybrid = HybridRetriever(
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
    )

    for retriever in (bm25_only, vector_only, hybrid):
        retriever.index(documents, vectors)

    per_query: list[QueryEvaluation] = []
    bm25_latencies: list[float] = []
    vector_latencies: list[float] = []
    hybrid_latencies: list[float] = []

    for item in eval_queries:
        bm25_top, bm25_ms = _timed_search(bm25_only, item, args.top_k)
        vector_top, vector_ms = _timed_search(vector_only, item, args.top_k)
        hybrid_top, hybrid_ms = _timed_search(hybrid, item, args.top_k)

        bm25_latencies.append(bm25_ms)
        vector_latencies.append(vector_ms)
        hybrid_latencies.append(hybrid_ms)

        per_query.append(
            QueryEvaluation(
                query=item.query,
                relevant_ids=item.relevant_ids,
                bm25_top=bm25_top,
                vector_top=vector_top,
                hybrid_top=hybrid_top,
                bm25_recall_at_k=recall_at_k(bm25_top, item.relevant_ids, args.top_k),
                vector_recall_at_k=recall_at_k(
                    vector_top,
                    item.relevant_ids,
                    args.top_k,
                ),
                hybrid_recall_at_k=recall_at_k(
                    hybrid_top,
                    item.relevant_ids,
                    args.top_k,
                ),
                bm25_mrr=mean_reciprocal_rank(bm25_top, item.relevant_ids),
                vector_mrr=mean_reciprocal_rank(vector_top, item.relevant_ids),
                hybrid_mrr=mean_reciprocal_rank(hybrid_top, item.relevant_ids),
            )
        )

    summary = EvaluationSummary(
        query_count=len(per_query),
        top_k=args.top_k,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
        avg_bm25_recall_at_k=statistics.mean(row.bm25_recall_at_k for row in per_query),
        avg_vector_recall_at_k=statistics.mean(
            row.vector_recall_at_k for row in per_query
        ),
        avg_hybrid_recall_at_k=statistics.mean(
            row.hybrid_recall_at_k for row in per_query
        ),
        avg_bm25_mrr=statistics.mean(row.bm25_mrr for row in per_query),
        avg_vector_mrr=statistics.mean(row.vector_mrr for row in per_query),
        avg_hybrid_mrr=statistics.mean(row.hybrid_mrr for row in per_query),
        bm25_latency_ms=statistics.mean(bm25_latencies),
        vector_latency_ms=statistics.mean(vector_latencies),
        hybrid_latency_ms=statistics.mean(hybrid_latencies),
    )

    report_path = write_evaluation_report(args.report_path, summary, per_query)

    print("Hybrid retrieval benchmark")
    print(f"Queries: {summary.query_count}")
    print(f"Top K: {summary.top_k}")
    print(
        "Avg recall@k: "
        f"bm25={summary.avg_bm25_recall_at_k:.3f} "
        f"vector={summary.avg_vector_recall_at_k:.3f} "
        f"hybrid={summary.avg_hybrid_recall_at_k:.3f}"
    )
    print(
        "Avg MRR: "
        f"bm25={summary.avg_bm25_mrr:.3f} "
        f"vector={summary.avg_vector_mrr:.3f} "
        f"hybrid={summary.avg_hybrid_mrr:.3f}"
    )
    print(
        "Avg latency ms: "
        f"bm25={summary.bm25_latency_ms:.2f} "
        f"vector={summary.vector_latency_ms:.2f} "
        f"hybrid={summary.hybrid_latency_ms:.2f}"
    )
    print(f"Report written to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
