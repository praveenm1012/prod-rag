#!/usr/bin/env python3
"""Visualize chunk size distribution for a document."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from app.rag import RecursiveCharacterTextSplitter, load_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk a document and visualize chunk size distribution.",
    )
    parser.add_argument("source", type=Path, help="Path to a supported document file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters (default: 200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chunk_sizes.png"),
        help="Output image path (default: chunk_sizes.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = load_document(args.source)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = splitter.split_document(document)
    sizes = [len(chunk.text) for chunk in chunks]

    if not sizes:
        print("No chunks produced for the given document.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    histogram_bins = min(30, max(5, len(sizes) // 2))
    axes[0].hist(
        sizes,
        bins=histogram_bins,
        color="#2563eb",
        edgecolor="white",
    )
    axes[0].set_title("Chunk Size Distribution")
    axes[0].set_xlabel("Characters per chunk")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(
        args.chunk_size,
        color="#dc2626",
        linestyle="--",
        label="chunk_size limit",
    )
    axes[0].legend()

    axes[1].plot(
        range(1, len(sizes) + 1),
        sizes,
        marker="o",
        markersize=3,
        color="#059669",
    )
    axes[1].set_title("Chunk Size by Index")
    axes[1].set_xlabel("Chunk index")
    axes[1].set_ylabel("Characters")
    axes[1].axhline(
        args.chunk_size,
        color="#dc2626",
        linestyle="--",
        label="chunk_size limit",
    )
    axes[1].legend()

    fig.suptitle(
        f"{args.source.name} | chunks={len(sizes)} | "
        f"size={args.chunk_size} | overlap={args.chunk_overlap}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"Document: {args.source}")
    print(f"Chunks: {len(sizes)}")
    print(f"Min size: {min(sizes)}")
    print(f"Max size: {max(sizes)}")
    print(f"Avg size: {sum(sizes) / len(sizes):.1f}")
    print(f"Saved chart to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
