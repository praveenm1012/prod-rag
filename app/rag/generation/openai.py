"""OpenAI chat completions provider."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from app.core.logging import get_logger
from app.rag.generation.base import BaseLLMProvider
from app.rag.generation.config import GenerationSettings
from app.rag.generation.exceptions import GenerationConfigError, ProviderResponseError
from app.rag.generation.http import HttpClient
from app.rag.generation.models import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
)

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible chat completions provider."""

    provider_name = "openai"

    def __init__(
        self,
        settings: GenerationSettings,
        http_client: HttpClient,
    ) -> None:
        super().__init__(settings, http_client)
        if not settings.openai_api_key:
            raise GenerationConfigError(
                "OPENAI_API_KEY is required for OpenAI provider"
            )

    def complete(self, request: GenerationRequest) -> GenerationResponse:
        payload = self._build_payload(request, stream=False)
        data = self._post_json(
            f"{self._settings.openai_base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            payload=payload,
        )
        return self._parse_response(data)

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        payload = self._build_payload(request, stream=True)
        url = f"{self._settings.openai_base_url.rstrip('/')}/chat/completions"
        for line in self._stream_lines(url, headers=self._headers(), payload=payload):
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                yield StreamChunk(content="", is_final=True, finish_reason="stop")
                return
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ProviderResponseError(
                    "OpenAI stream returned invalid JSON"
                ) from exc
            yield from self._parse_stream_event(event)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.openai_api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        request: GenerationRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self._resolve_model(request),
            "messages": [_message_to_dict(message) for message in request.messages],
            "temperature": self._resolve_temperature(request),
            "max_tokens": self._resolve_max_tokens(request),
            "stream": stream,
        }

    def _parse_response(self, data: dict[str, Any]) -> GenerationResponse:
        try:
            choice = data["choices"][0]
            message = choice["message"]
            content = str(message["content"])
            usage = {
                key: int(data.get("usage", {}).get(key, 0))
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderResponseError(
                "OpenAI response missing expected fields"
            ) from exc

        logger.info(
            "openai_generation_complete",
            model=data.get("model", self._settings.model),
            finish_reason=choice.get("finish_reason"),
        )
        return GenerationResponse(
            content=content,
            model=str(data.get("model", self._settings.model)),
            provider=self.provider_name,
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )

    def _parse_stream_event(self, event: dict[str, Any]) -> Iterator[StreamChunk]:
        choices = event.get("choices", [])
        if not choices:
            return
        choice = choices[0]
        delta = choice.get("delta", {})
        content = str(delta.get("content", ""))
        finish_reason = choice.get("finish_reason")
        if content:
            yield StreamChunk(content=content, finish_reason=finish_reason)
        elif finish_reason:
            yield StreamChunk(content="", finish_reason=finish_reason, is_final=True)


def _message_to_dict(message: ChatMessage) -> dict[str, str]:
    return {"role": message.role, "content": message.content}
