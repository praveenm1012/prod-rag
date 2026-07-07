#!/usr/bin/env python3
"""Benchmark BM25 lexical search indexing and query throughput."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from app.rag.lexical import BM25Searcher, LexicalDocument


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BM25 lexical search.")
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Optional text file; each non-empty line becomes a document",
    )
    parser.add_argument(
        "--document-count",
        type=int,
        default=1000,
        help="Synthetic documents to index when --text-file is not provided",
    )
    parser.add_argument(
        "--query-count",
        type=int,
        default=100,
        help="Number of benchmark queries to run",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top K results per query",
    )
    return parser.parse_args()


def build_documents(args: argparse.Namespace) -> list[LexicalDocument]:
    if args.text_file:
        lines = [
            line.strip()
            for line in args.text_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return [
            LexicalDocument(
                chunk_id=f"doc:{index}:0",
                text=line,
                document_id=f"doc-{index % 50}",
                page_number=1,
                source=str(args.text_file),
            )
            for index, line in enumerate(lines)
        ]

    return [
        LexicalDocument(
            chunk_id=f"synthetic:{index}:0",
            text=(
                f"Synthetic document {index} about retrieval, embeddings, "
                f"and lexical search with BM25 ranking."
            ),
            document_id=f"doc-{index % 50}",
            page_number=(index % 5) + 1,
            source="/synthetic/corpus.txt",
        )
        for index in range(args.document_count)
    ]


def build_queries(args: argparse.Namespace) -> list[str]:
    return [
        f"retrieval embeddings document {index}"
        for index in range(args.query_count)
    ]


def main() -> None:
    args = parse_args()
    documents = build_documents(args)
    queries = build_queries(args)
    searcher = BM25Searcher(top_k=args.top_k)

    print("Lexical search benchmark")
    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Top K: {args.top_k}")

    index_start = time.perf_counter()
    searcher.index(documents)
    index_time = time.perf_counter() - index_start

    latencies: list[float] = []
    hit_counts: list[int] = []
    for query in queries:
        start = time.perf_counter()
        results = searcher.search(query, top_k=args.top_k)
        latencies.append(time.perf_counter() - start)
        hit_counts.append(len(results))

    total_search_time = sum(latencies)
    print("\nSummary")
    print(f"Index time: {index_time:.4f}s")
    print(f"Search time total: {total_search_time:.4f}s")
    print(f"Search time avg: {statistics.mean(latencies) * 1000:.3f} ms")
    print(f"Search throughput: {len(queries) / total_search_time:.2f} qps")
    print(f"Avg hits per query: {statistics.mean(hit_counts):.2f}")
    if latencies:
        print(f"P95 latency: {statistics.quantiles(latencies, n=20)[-1] * 1000:.3f} ms")


if __name__ == "__main__":
    main()
