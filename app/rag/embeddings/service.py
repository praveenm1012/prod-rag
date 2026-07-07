"""BGE embedding service."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from app.core.logging import get_logger
from app.rag.embeddings.cache import EmbeddingCache
from app.rag.embeddings.config import EmbeddingSettings, get_embedding_settings
from app.rag.embeddings.exceptions import EmbeddingConfigError, EmbeddingModelError
from app.rag.embeddings.models import EmbeddingBatchResult, EmbeddingRecord, PromptType

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def resolve_device(device: str) -> str:
    """Resolve auto device selection to a concrete torch device."""
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def apply_prompt(text: str, prompt_type: PromptType) -> str:
    """Apply BGE prompt formatting for query or document text."""
    if prompt_type == "query":
        return f"{QUERY_PREFIX}{text}"
    return text


class EmbeddingService:
    """Embedding service backed by sentence-transformers."""

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self._settings = settings or get_embedding_settings()
        if self._settings.batch_size <= 0:
            raise EmbeddingConfigError("batch_size must be greater than 0")
        if self._settings.max_retries <= 0:
            raise EmbeddingConfigError("max_retries must be greater than 0")

        self._device = resolve_device(self._settings.device)
        self._cache = EmbeddingCache(self._settings)
        self._model: SentenceTransformer | None = None
        logger.info(
            "embedding_service_initialized",
            model=self._settings.model_name,
            device=self._device,
            batch_size=self._settings.batch_size,
            cache_dir=str(self._cache.path),
        )

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def embedding_dimension(self) -> int:
        model = self._get_model()
        dimension = model.get_sentence_embedding_dimension()
        if dimension is None:
            raise EmbeddingModelError("Model did not report embedding dimension")
        return int(dimension)

    def embed_text(
        self,
        text: str,
        prompt_type: PromptType = "document",
        use_cache: bool = True,
    ) -> EmbeddingRecord:
        """Embed a single text."""
        batch = self.embed_texts([text], prompt_type=prompt_type, use_cache=use_cache)
        return batch.records[0]

    def embed_texts(
        self,
        texts: Sequence[str],
        prompt_type: PromptType = "document",
        use_cache: bool = True,
    ) -> EmbeddingBatchResult:
        """Embed multiple texts with batching, caching, retry, and progress logging."""
        if not texts:
            return EmbeddingBatchResult(records=[], cache_hits=0, cache_misses=0)

        normalized_texts = list(texts)
        records: list[EmbeddingRecord | None] = [None] * len(normalized_texts)
        pending_indices: list[int] = []
        pending_texts: list[str] = []
        cache_hits = 0

        for index, text in enumerate(normalized_texts):
            cache_key = self._cache.make_key(text, prompt_type)
            if use_cache:
                cached_vector = self._cache.get(cache_key)
                if cached_vector is not None:
                    records[index] = EmbeddingRecord(
                        text=text,
                        vector=cached_vector,
                        model_name=self._settings.model_name,
                        prompt_type=prompt_type,
                        cached=True,
                    )
                    cache_hits += 1
                    continue

            pending_indices.append(index)
            pending_texts.append(text)

        cache_misses = len(pending_texts)
        if pending_texts:
            encoded = self._encode_batches(pending_texts, prompt_type)
            for index, text, vector in zip(
                pending_indices,
                pending_texts,
                encoded,
                strict=True,
            ):
                cache_key = self._cache.make_key(text, prompt_type)
                if use_cache:
                    self._cache.set(cache_key, vector)
                records[index] = EmbeddingRecord(
                    text=text,
                    vector=vector,
                    model_name=self._settings.model_name,
                    prompt_type=prompt_type,
                    cached=False,
                )

        logger.info(
            "embedding_batch_complete",
            total=len(normalized_texts),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            device=self._device,
            prompt_type=prompt_type,
        )

        return EmbeddingBatchResult(
            records=[record for record in records if record is not None],
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            metadata={
                "device": self._device,
                "model_name": self._settings.model_name,
                "prompt_type": prompt_type,
            },
        )

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("embedding_cache_cleared", cache_dir=str(self._cache.path))

    def _get_model(self) -> "SentenceTransformer":
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingModelError(
                    "sentence-transformers is required for embedding generation"
                ) from exc

            try:
                logger.info(
                    "embedding_model_loading",
                    model=self._settings.model_name,
                    device=self._device,
                )
                self._model = SentenceTransformer(
                    self._settings.model_name,
                    device=self._device,
                )
            except Exception as exc:
                raise EmbeddingModelError(
                    f"Failed to load embedding model: {self._settings.model_name}"
                ) from exc

        return self._model

    def _encode_batches(
        self,
        texts: Sequence[str],
        prompt_type: PromptType,
    ) -> list[list[float]]:
        prepared = [apply_prompt(text, prompt_type) for text in texts]
        batch_size = self._settings.batch_size
        vectors: list[list[float]] = []

        batch_ranges = range(0, len(prepared), batch_size)
        progress = tqdm(
            batch_ranges,
            desc="Embedding texts",
            disable=not self._settings.show_progress,
        )

        for start in progress:
            batch = prepared[start : start + batch_size]
            batch_vectors = self._encode_with_retry(batch)
            vectors.extend(batch_vectors)
            progress.set_postfix(
                {
                    "batch": f"{start // batch_size + 1}",
                    "device": self._device,
                }
            )
            logger.info(
                "embedding_batch_progress",
                batch_number=start // batch_size + 1,
                batch_count=(len(prepared) + batch_size - 1) // batch_size,
                batch_size=len(batch),
                device=self._device,
            )

        return vectors

    def _encode_with_retry(self, texts: Sequence[str]) -> list[list[float]]:
        @retry(
            reraise=True,
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self._settings.retry_min_seconds,
                max=self._settings.retry_max_seconds,
            ),
            retry=retry_if_exception_type(Exception),
        )
        def _encode() -> list[list[float]]:
            model = self._get_model()
            try:
                embeddings = model.encode(
                    list(texts),
                    batch_size=len(texts),
                    convert_to_numpy=True,
                    normalize_embeddings=self._settings.normalize_embeddings,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise EmbeddingModelError("Embedding encoding failed") from exc

            array = np.asarray(embeddings, dtype=np.float32)
            if array.ndim == 1:
                return [array.tolist()]
            return [row.tolist() for row in array]

        return _encode()
