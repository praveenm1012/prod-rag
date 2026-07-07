"""Base LLM provider with retry support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.logging import get_logger
from app.rag.generation.config import GenerationSettings
from app.rag.generation.exceptions import ProviderError, ProviderResponseError
from app.rag.generation.http import HttpClient
from app.rag.generation.models import (
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
)

logger = get_logger(__name__)

T = TypeVar("T")


class LLMProvider(ABC):
    """Abstract LLM provider with completion and streaming APIs."""

    provider_name: str

    @abstractmethod
    def complete(self, request: GenerationRequest) -> GenerationResponse:
        """Generate a full completion."""

    @abstractmethod
    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        """Stream completion chunks."""


class BaseLLMProvider(LLMProvider):
    """Shared retry and validation logic for HTTP-backed providers."""

    def __init__(
        self,
        settings: GenerationSettings,
        http_client: HttpClient,
    ) -> None:
        self._settings = settings
        self._http = http_client

    def _resolve_model(self, request: GenerationRequest) -> str:
        return request.model or self._settings.model

    def _resolve_temperature(self, request: GenerationRequest) -> float:
        if request.temperature is not None:
            return request.temperature
        return self._settings.temperature

    def _resolve_max_tokens(self, request: GenerationRequest) -> int:
        if request.max_tokens is not None:
            return request.max_tokens
        return self._settings.max_tokens

    def _with_retry(self, operation: Callable[[], T]) -> T:
        if self._settings.max_retries <= 0:
            return operation()

        @retry(
            reraise=True,
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self._settings.retry_min_seconds,
                max=self._settings.retry_max_seconds,
            ),
            retry=retry_if_exception_type(ProviderError),
        )
        def _run() -> T:
            return operation()

        return _run()

    def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        def _request() -> dict[str, Any]:
            response = self._http.post(url, headers=headers, json=payload)
            try:
                response.raise_for_status()
            except Exception as exc:
                raise ProviderError(f"{self.provider_name} request failed") from exc
            data = response.json()
            if not isinstance(data, dict):
                raise ProviderResponseError(
                    f"{self.provider_name} returned a non-object JSON response"
                )
            return data

        return self._with_retry(_request)

    def _stream_lines(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> Iterator[str]:
        attempt = 0
        last_error: ProviderError | None = None
        while attempt < max(1, self._settings.max_retries):
            attempt += 1
            try:
                yield from self._http.stream_lines(url, headers=headers, json=payload)
                return
            except ProviderError as exc:
                last_error = exc
                logger.warning(
                    "provider_stream_retry",
                    provider=self.provider_name,
                    attempt=attempt,
                    error=str(exc),
                )
        if last_error is not None:
            raise last_error
        raise ProviderError(f"{self.provider_name} stream failed")
