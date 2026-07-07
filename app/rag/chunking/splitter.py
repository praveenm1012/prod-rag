"""Recursive character text splitter."""

from collections.abc import Callable, Sequence
from hashlib import sha256
from typing import Any

from app.rag.chunking.exceptions import InvalidChunkConfigError
from app.rag.chunking.models import Chunk
from app.rag.ingestion.models import Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ", "")


class RecursiveCharacterTextSplitter:
    """Split text recursively using a hierarchy of separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Sequence[str] | None = None,
        length_function: Callable[[str], int] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise InvalidChunkConfigError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            message = "chunk_overlap must be greater than or equal to 0"
            raise InvalidChunkConfigError(message)
        if chunk_overlap >= chunk_size:
            raise InvalidChunkConfigError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._separators = list(separators or DEFAULT_SEPARATORS)
        self._length_function = length_function or len

    def split_text(self, text: str) -> list[str]:
        """Split raw text into chunk strings."""
        normalized = text.strip()
        if not normalized:
            return []
        return self._split_text(normalized, self._separators)

    def split_document(self, document: Document) -> list[Chunk]:
        """Split a document into chunks with provenance metadata."""
        document_id = _resolve_document_id(document.metadata)
        source = str(document.metadata.get("source", ""))
        page_texts = document.metadata.get("page_texts")

        if isinstance(page_texts, list) and page_texts:
            return self._split_paginated_document(
                page_texts=page_texts,
                document_id=document_id,
                source=source,
            )

        return self._build_chunks(
            texts=self.split_text(document.text),
            document_id=document_id,
            source=source,
            page_number=1,
        )

    def _split_paginated_document(
        self,
        page_texts: list[Any],
        document_id: str,
        source: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0

        for page_number, page_text in enumerate(page_texts, start=1):
            if not isinstance(page_text, str):
                continue
            page_chunks = self._build_chunks(
                texts=self.split_text(page_text),
                document_id=document_id,
                source=source,
                page_number=page_number,
                start_index=chunk_index,
            )
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        return chunks

    def _build_chunks(
        self,
        texts: list[str],
        document_id: str,
        source: str,
        page_number: int,
        start_index: int = 0,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for offset, text in enumerate(texts):
            chunk_index = start_index + offset
            chunks.append(
                Chunk(
                    text=text,
                    chunk_id=f"{document_id}:{page_number}:{chunk_index}",
                    document_id=document_id,
                    page_number=page_number,
                    source=source,
                    metadata={"chunk_index": chunk_index},
                )
            )
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []

        for index, candidate in enumerate(separators):
            if candidate == "":
                separator = candidate
                break
            if candidate in text:
                separator = candidate
                new_separators = separators[index + 1 :]
                break

        splits = list(text) if separator == "" else text.split(separator)
        good_splits: list[str] = []

        for split in splits:
            if self._length_function(split) <= self.chunk_size:
                good_splits.append(split)
                continue

            if good_splits:
                final_chunks.extend(self._merge_splits(good_splits, separator))
                good_splits = []

            if not new_separators:
                final_chunks.append(split)
            else:
                final_chunks.extend(self._split_text(split, new_separators))

        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))

        return [chunk for chunk in final_chunks if chunk.strip()]

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        chunks: list[str] = []
        current_splits: list[str] = []
        current_length = 0

        for split in splits:
            split_length = self._length_function(split)
            separator_length = self._length_function(separator) if current_splits else 0
            projected_length = current_length + split_length + separator_length

            if projected_length <= self.chunk_size:
                current_splits.append(split)
                current_length = projected_length
                continue

            if current_splits:
                chunk = separator.join(current_splits).strip()
                if chunk:
                    chunks.append(chunk)

            while current_splits:
                pop_length = self._length_function(current_splits[0])
                has_multiple = len(current_splits) > 1
                separator_pop = self._length_function(separator) if has_multiple else 0
                current_length -= pop_length + separator_pop
                current_splits.pop(0)
                if current_length <= self.chunk_overlap:
                    break

            current_splits.append(split)
            current_length = sum(self._length_function(item) for item in current_splits)
            if len(current_splits) > 1:
                separator_len = self._length_function(separator)
                separators_total = separator_len * (len(current_splits) - 1)
                current_length += separators_total

        if current_splits:
            chunk = separator.join(current_splits).strip()
            if chunk:
                chunks.append(chunk)

        return chunks


def _resolve_document_id(metadata: dict[str, Any]) -> str:
    if document_id := metadata.get("document_id"):
        return str(document_id)

    source = str(metadata.get("source", ""))
    if source:
        return sha256(source.encode()).hexdigest()[:16]

    return sha256(repr(sorted(metadata.items())).encode()).hexdigest()[:16]
