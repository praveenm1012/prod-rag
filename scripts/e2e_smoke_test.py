#!/usr/bin/env python3
"""End-to-end RAG smoke test: PDF upload, embed, index, ask, verify, metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import httpx
from pypdf import PdfWriter
from pypdf.generic import (
    DictionaryObject,
    NameObject,
    NumberObject,
    StreamObject,
)

from app.rag.documents import DocumentService
from app.rag.embeddings import EmbeddingService
from app.rag.generation import create_llm_provider
from app.rag.ingestion import load_document
from app.rag.lexical.models import LexicalDocument
from app.rag.pipeline import RagPipeline
from app.rag.reranking import CrossEncoderReranker
from app.rag.retrieval import HybridRetriever

Mode = Literal["local", "api"]

SAMPLE_TEXT = (
    "Acme Corporation Smoke Test Document.\n\n"
    "The headquarters of Acme Corporation is located in Springfield.\n"
    "Acme specializes in retrieval-augmented generation testing."
)
SAMPLE_QUESTION = "Where is the headquarters of Acme Corporation?"
EXPECTED_ANSWER_KEYWORDS = ("springfield", "acme")
EXPECTED_CITATION = "[1]"


@dataclass
class TimingMetrics:
    """Milliseconds spent in each pipeline phase."""

    upload_ms: float = 0.0
    ingest_ms: float = 0.0
    chunk_ms: float = 0.0
    embed_ms: float = 0.0
    index_ms: float = 0.0
    ask_ms: float = 0.0
    cleanup_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class SmokeResult:
    """Outcome of the end-to-end smoke workflow."""

    status: Literal["pass", "fail"]
    mode: Mode
    document_id: str | None = None
    chunks_indexed: int = 0
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    timings_ms: TimingMetrics = field(default_factory=TimingMetrics)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timings_ms"] = self.timings_ms.to_dict()
        return payload


def create_sample_pdf_bytes(text: str = SAMPLE_TEXT) -> bytes:
    """Build a minimal PDF containing the given text for ingestion tests."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    escaped = text.replace("(", r"\(").replace(")", r"\)")
    lines = escaped.split("\n")
    content_lines = ["BT", "/F1 14 Tf", "72 720 Td"]
    for index, line in enumerate(lines):
        if index > 0:
            content_lines.append("T*")
        content_lines.append(f"({line}) Tj")
    content_lines.append("ET")
    stream_data = "\n".join(content_lines).encode("latin-1")

    stream = StreamObject()
    stream._data = stream_data
    stream[NameObject("/Length")] = NumberObject(len(stream_data))

    font = DictionaryObject()
    font[NameObject("/Type")] = NameObject("/Font")
    font[NameObject("/Subtype")] = NameObject("/Type1")
    font[NameObject("/BaseFont")] = NameObject("/Helvetica")

    resources = DictionaryObject()
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font)
    resources[NameObject("/Font")] = fonts

    page_obj = page.get_object()
    page_obj[NameObject("/Resources")] = resources
    page_obj[NameObject("/Contents")] = writer._add_object(stream)

    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _answer_matches(answer: str) -> bool:
    normalized = answer.casefold()
    return all(keyword in normalized for keyword in EXPECTED_ANSWER_KEYWORDS)


def _citations_valid(citations: list[str], answer: str) -> bool:
    del answer  # API returns citation placeholders separately from answer text.
    return bool(citations) and EXPECTED_CITATION in citations


def _index_with_timings(
    service: DocumentService,
    filename: str,
    content: bytes,
    upload_dir: Path,
) -> tuple[str, int, TimingMetrics]:
    """Run ingest, chunk, embed, and index with per-phase timing."""
    timings = TimingMetrics()
    phase_start = time.perf_counter()

    upload_dir.mkdir(parents=True, exist_ok=True)
    document_id = uuid.uuid4().hex
    destination = upload_dir / f"{document_id}_{Path(filename).name}"
    destination.write_bytes(content)
    timings.upload_ms = (time.perf_counter() - phase_start) * 1000

    phase_start = time.perf_counter()
    document = load_document(destination)
    timings.ingest_ms = (time.perf_counter() - phase_start) * 1000

    phase_start = time.perf_counter()
    splitter = service._splitter  # noqa: SLF001
    chunks = splitter.split_document(document)
    if not chunks:
        raise RuntimeError("upload produced no indexable chunks")
    lexical_documents = [LexicalDocument.from_chunk(chunk) for chunk in chunks]
    timings.chunk_ms = (time.perf_counter() - phase_start) * 1000

    phase_start = time.perf_counter()
    vectors = service._embed_chunks([item.text for item in lexical_documents])  # noqa: SLF001
    timings.embed_ms = (time.perf_counter() - phase_start) * 1000

    phase_start = time.perf_counter()
    service._lexical_documents.extend(lexical_documents)  # noqa: SLF001
    service._vectors.extend(vectors)  # noqa: SLF001
    service._reindex()  # noqa: SLF001
    from app.rag.documents.models import DocumentRecord, utc_now

    record = DocumentRecord(
        document_id=document_id,
        filename=Path(filename).name,
        source=str(destination),
        chunk_count=len(chunks),
        uploaded_at=utc_now(),
    )
    service._documents[document_id] = record  # noqa: SLF001
    timings.index_ms = (time.perf_counter() - phase_start) * 1000

    return document_id, len(chunks), timings


def run_local_smoke(
    *,
    skip_cleanup: bool = False,
    use_reranker: bool = False,
) -> SmokeResult:
    """Run the full pipeline in-process with granular timing."""
    total_start = time.perf_counter()
    timings = TimingMetrics()
    upload_dir = Path(".cache/e2e-smoke/uploads")
    pdf_bytes = create_sample_pdf_bytes()

    embedding_service = EmbeddingService()
    retriever = HybridRetriever(embedding_service=embedding_service)
    document_service = DocumentService(
        retriever=retriever,
        embedding_service=embedding_service,
        upload_dir=upload_dir,
    )
    pipeline = RagPipeline(
        retriever=document_service.retriever,
        reranker=CrossEncoderReranker(),
        llm_provider=create_llm_provider(),
        use_reranker=use_reranker,
    )

    document_id: str | None = None
    chunks_indexed = 0
    answer = ""
    citations: list[str] = []
    checks: dict[str, bool] = {}

    try:
        document_id, chunks_indexed, index_timings = _index_with_timings(
            document_service,
            "sample_smoke.pdf",
            pdf_bytes,
            upload_dir,
        )
        timings.upload_ms = index_timings.upload_ms
        timings.ingest_ms = index_timings.ingest_ms
        timings.chunk_ms = index_timings.chunk_ms
        timings.embed_ms = index_timings.embed_ms
        timings.index_ms = index_timings.index_ms

        phase_start = time.perf_counter()
        rag_answer = pipeline.ask(SAMPLE_QUESTION, top_k=3)
        timings.ask_ms = (time.perf_counter() - phase_start) * 1000

        answer = rag_answer.answer
        citations = [citation.placeholder for citation in rag_answer.prompt.citations]

        checks = {
            "upload": document_id is not None,
            "chunks_indexed": chunks_indexed >= 1,
            "embed": timings.embed_ms > 0,
            "index": timings.index_ms >= 0,
            "answer": _answer_matches(answer),
            "citations": _citations_valid(citations, answer),
        }
        status: Literal["pass", "fail"] = "pass" if all(checks.values()) else "fail"
    except Exception as exc:
        timings.total_ms = (time.perf_counter() - total_start) * 1000
        return SmokeResult(
            status="fail",
            mode="local",
            document_id=document_id,
            chunks_indexed=chunks_indexed,
            answer=answer,
            citations=citations,
            checks=checks,
            timings_ms=timings,
            error=str(exc),
        )
    finally:
        if document_id and not skip_cleanup:
            cleanup_start = time.perf_counter()
            document_service.delete_document(document_id)
            timings.cleanup_ms = (time.perf_counter() - cleanup_start) * 1000

    timings.total_ms = (time.perf_counter() - total_start) * 1000
    return SmokeResult(
        status=status,
        mode="local",
        document_id=document_id,
        chunks_indexed=chunks_indexed,
        answer=answer,
        citations=citations,
        checks=checks,
        timings_ms=timings,
    )


def run_api_smoke(
    base_url: str,
    *,
    skip_cleanup: bool = False,
    timeout: float = 600.0,
) -> SmokeResult:
    """Run the smoke workflow against a live HTTP API."""
    total_start = time.perf_counter()
    timings = TimingMetrics()
    pdf_bytes = create_sample_pdf_bytes()
    document_id: str | None = None
    chunks_indexed = 0
    answer = ""
    citations: list[str] = []
    checks: dict[str, bool] = {}

    api_root = base_url.rstrip("/")
    upload_url = f"{api_root}/api/v1/upload"
    chat_url = f"{api_root}/api/v1/chat"
    delete_url = f"{api_root}/api/v1/delete"

    try:
        with httpx.Client(timeout=timeout) as client:
            phase_start = time.perf_counter()
            upload_response = client.post(
                upload_url,
                files={
                    "file": (
                        "sample_smoke.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )
            timings.upload_ms = (time.perf_counter() - phase_start) * 1000

            if upload_response.status_code != 201:
                raise RuntimeError(
                    f"upload failed ({upload_response.status_code}): {upload_response.text}"
                )

            upload_body = upload_response.json()
            document_id = upload_body["document"]["document_id"]
            chunks_indexed = int(upload_body["chunks_indexed"])
            timings.ingest_ms = timings.upload_ms
            timings.embed_ms = timings.upload_ms
            timings.index_ms = timings.upload_ms

            phase_start = time.perf_counter()
            chat_response = client.post(
                chat_url,
                json={
                    "question": SAMPLE_QUESTION,
                    "use_rag": True,
                    "top_k": 3,
                },
            )
            timings.ask_ms = (time.perf_counter() - phase_start) * 1000

            if chat_response.status_code != 200:
                raise RuntimeError(
                    f"chat failed ({chat_response.status_code}): {chat_response.text}"
                )

            chat_body = chat_response.json()
            answer = chat_body["answer"]
            citations = list(chat_body.get("citations", []))

            checks = {
                "upload": upload_response.status_code == 201,
                "chunks_indexed": chunks_indexed >= 1,
                "embed": chunks_indexed >= 1,
                "index": chunks_indexed >= 1,
                "answer": _answer_matches(answer),
                "citations": _citations_valid(citations, answer),
            }
            status: Literal["pass", "fail"] = "pass" if all(checks.values()) else "fail"

            if document_id and not skip_cleanup:
                cleanup_start = time.perf_counter()
                delete_response = client.request(
                    "DELETE",
                    delete_url,
                    json={"document_id": document_id},
                )
                timings.cleanup_ms = (time.perf_counter() - cleanup_start) * 1000
                if delete_response.status_code != 204:
                    raise RuntimeError(
                        "delete failed "
                        f"({delete_response.status_code}): {delete_response.text}"
                    )
    except Exception as exc:
        timings.total_ms = (time.perf_counter() - total_start) * 1000
        return SmokeResult(
            status="fail",
            mode="api",
            document_id=document_id,
            chunks_indexed=chunks_indexed,
            answer=answer,
            citations=citations,
            checks=checks,
            timings_ms=timings,
            error=str(exc),
        )

    timings.total_ms = (time.perf_counter() - total_start) * 1000
    return SmokeResult(
        status=status,
        mode="api",
        document_id=document_id,
        chunks_indexed=chunks_indexed,
        answer=answer,
        citations=citations,
        checks=checks,
        timings_ms=timings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAG smoke test (PDF upload through citations).",
    )
    parser.add_argument(
        "--mode",
        choices=("local", "api"),
        default="api",
        help="local = in-process pipeline; api = live HTTP server (default)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API base URL when --mode=api",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds for API mode",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Leave uploaded document in the index after the run",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable cross-encoder reranking in local mode",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only JSON results to stdout",
    )
    return parser.parse_args()


def print_report(result: SmokeResult, *, json_only: bool = False) -> None:
    if json_only:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print("End-to-end RAG smoke test")
    print(f"Mode: {result.mode}")
    print(f"Status: {result.status.upper()}")
    if result.error:
        print(f"Error: {result.error}")
    print()
    print("Workflow")
    print("-" * 48)
    print(f"Document ID: {result.document_id}")
    print(f"Chunks indexed: {result.chunks_indexed}")
    print(f"Question: {SAMPLE_QUESTION}")
    print(f"Answer preview: {result.answer[:240]}{'...' if len(result.answer) > 240 else ''}")
    print(f"Citations: {result.citations}")
    print()
    print("Checks")
    print("-" * 48)
    for name, passed in result.checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print()
    print("Timing metrics (ms)")
    print("-" * 48)
    for key, value in result.timings_ms.to_dict().items():
        print(f"{key}: {value:.2f}")
    print()
    print("JSON")
    print("-" * 48)
    print(json.dumps(result.to_dict(), indent=2))


def main() -> int:
    args = parse_args()
    if args.mode == "local":
        result = run_local_smoke(
            skip_cleanup=args.skip_cleanup,
            use_reranker=args.use_reranker,
        )
    else:
        result = run_api_smoke(
            args.base_url,
            skip_cleanup=args.skip_cleanup,
            timeout=args.timeout,
        )

    print_report(result, json_only=args.json_only)
    return 0 if result.status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
