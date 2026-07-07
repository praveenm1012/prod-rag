#!/usr/bin/env python3
"""Benchmark the embedding service throughput and cache behavior."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from app.rag.embeddings import EmbeddingService
from app.rag.embeddings.config import EmbeddingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark embedding generation.")
    parser.add_argument(
        "--text-file",
        type=Path,
        help="Optional text file; each non-empty line is embedded",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=64,
        help="Number of synthetic sentences when --text-file is not provided",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, or cuda",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/embeddings-benchmark"),
        help="Cache directory for the benchmark run",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Number of benchmark repetitions",
    )
    return parser.parse_args()


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text_file:
        content = args.text_file.read_text(encoding="utf-8")
        return [line.strip() for line in content.splitlines() if line.strip()]

    return [
        f"Synthetic benchmark sentence number {index} about retrieval and embeddings."
        for index in range(args.sample_count)
    ]


def run_once(service: EmbeddingService, texts: list[str], use_cache: bool) -> float:
    start = time.perf_counter()
    service.embed_texts(texts, use_cache=use_cache)
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    texts = load_texts(args)
    settings = EmbeddingSettings(
        EMBEDDING_MODEL="BAAI/bge-large-en-v1.5",
        EMBEDDING_DEVICE=args.device,
        EMBEDDING_BATCH_SIZE=args.batch_size,
        EMBEDDING_CACHE_DIR=args.cache_dir,
        EMBEDDING_SHOW_PROGRESS=True,
    )
    service = EmbeddingService(settings=settings)

    print("Embedding benchmark")
    print(f"Model: {settings.model_name}")
    print(f"Device: {service.device}")
    print(f"Texts: {len(texts)}")
    print(f"Batch size: {settings.batch_size}")
    print(f"Cache dir: {settings.cache_dir}")

    cold_runs: list[float] = []
    warm_runs: list[float] = []

    for run_index in range(args.repeat):
        service.clear_cache()
        cold = run_once(service, texts, use_cache=True)
        warm = run_once(service, texts, use_cache=True)
        cold_runs.append(cold)
        warm_runs.append(warm)
        print(f"Run {run_index + 1}: cold={cold:.3f}s warm={warm:.3f}s")

    cold_avg = statistics.mean(cold_runs)
    warm_avg = statistics.mean(warm_runs)
    throughput = len(texts) / cold_avg if cold_avg > 0 else 0.0

    print("\nSummary")
    print(f"Cold avg: {cold_avg:.3f}s")
    print(f"Warm avg: {warm_avg:.3f}s")
    print(f"Cold throughput: {throughput:.2f} texts/sec")
    print(f"Embedding dimension: {service.embedding_dimension}")
    if warm_avg > 0:
        print(f"Cache speedup: {cold_avg / warm_avg:.2f}x")


if __name__ == "__main__":
    main()
