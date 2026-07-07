"""HTTP client abstraction for LLM providers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import httpx

from app.rag.generation.exceptions import ProviderError, ProviderTimeoutError


class HttpResponse(Protocol):
    """Minimal HTTP response interface."""

    status_code: int

    def json(self) -> Any:
        """Parse the response body as JSON."""
        ...

    def raise_for_status(self) -> None:
        """Raise an error for non-success status codes."""
        ...


class HttpClient(Protocol):
    """HTTP client used by LLM providers."""

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> HttpResponse:
        """Send a JSON POST request."""
        ...

    def stream_lines(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> Iterator[str]:
        """Stream newline-delimited response lines."""
        ...


class HttpxHttpClient:
    """httpx-backed HTTP client with request timeout."""

    def __init__(self, timeout_seconds: float) -> None:
        self._timeout = timeout_seconds
        self._client = httpx.Client(timeout=timeout_seconds)

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> httpx.Response:
        try:
            return self._client.post(url, headers=headers, json=json)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP request failed: {exc}") from exc

    def stream_lines(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> Iterator[str]:
        try:
            with self._client.stream(
                "POST",
                url,
                headers=headers,
                json=json,
            ) as response:
                response.raise_for_status()
                yield from response.iter_lines()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(
                f"Request timed out after {self._timeout}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP stream failed: {exc}") from exc

    def close(self) -> None:
        self._client.close()
