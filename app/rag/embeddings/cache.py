"""Disk-backed embedding cache."""

import hashlib
import json
from pathlib import Path

import diskcache

from app.rag.embeddings.config import EmbeddingSettings
from app.rag.embeddings.models import PromptType


class EmbeddingCache:
    """Persistent cache for embedding vectors."""

    def __init__(self, settings: EmbeddingSettings) -> None:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(settings.cache_dir))
        self._model_name = settings.model_name

    def make_key(self, text: str, prompt_type: PromptType) -> str:
        """Build a stable cache key for a text and prompt type."""
        payload = json.dumps(
            {
                "model": self._model_name,
                "prompt_type": prompt_type,
                "text": text,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> list[float] | None:
        """Return a cached vector if present."""
        value = self._cache.get(key)
        if value is None:
            return None
        if not isinstance(value, list):
            return None
        return [float(item) for item in value]

    def set(self, key: str, vector: list[float]) -> None:
        """Store a vector in the cache."""
        self._cache.set(key, vector)

    def clear(self) -> None:
        """Remove all cached embeddings."""
        self._cache.clear()

    @property
    def path(self) -> Path:
        """Return the cache directory path."""
        return Path(self._cache.directory)
