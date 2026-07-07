"""Ollama chat provider."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from app.core.logging import get_logger
from app.rag.generation.base import BaseLLMProvider
from app.rag.generation.exceptions import ProviderResponseError
from app.rag.generation.models import (
    ChatMessage,
    GenerationRequest,
    GenerationResponse,
    StreamChunk,
)

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama local chat API provider."""

    provider_name = "ollama"

    def complete(self, request: GenerationRequest) -> GenerationResponse:
        payload = self._build_payload(request, stream=False)
        data = self._post_json(
            f"{self._settings.ollama_base_url.rstrip('/')}/api/chat",
            headers={"Content-Type": "application/json"},
            payload=payload,
        )
        return self._parse_response(data, request)

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        payload = self._build_payload(request, stream=True)
        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/chat"
        for line in self._stream_lines(
            url,
            headers={"Content-Type": "application/json"},
            payload=payload,
        ):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProviderResponseError(
                    "Ollama stream returned invalid JSON"
                ) from exc
            yield from self._parse_stream_event(event)

    def _build_payload(
        self,
        request: GenerationRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": self._resolve_model(request),
            "messages": [_message_to_dict(message) for message in request.messages],
            "stream": stream,
            "options": {
                "temperature": self._resolve_temperature(request),
                "num_predict": self._resolve_max_tokens(request),
            },
        }

    def _parse_response(
        self,
        data: dict[str, Any],
        request: GenerationRequest,
    ) -> GenerationResponse:
        try:
            message = data["message"]
            content = str(message["content"])
        except (KeyError, TypeError) as exc:
            raise ProviderResponseError(
                "Ollama response missing expected fields"
            ) from exc

        logger.info(
            "ollama_generation_complete",
            model=self._resolve_model(request),
            done=data.get("done"),
        )
        return GenerationResponse(
            content=content,
            model=self._resolve_model(request),
            provider=self.provider_name,
            finish_reason="stop" if data.get("done") else None,
            usage={
                "prompt_tokens": int(data.get("prompt_eval_count", 0)),
                "completion_tokens": int(data.get("eval_count", 0)),
                "total_tokens": int(data.get("prompt_eval_count", 0))
                + int(data.get("eval_count", 0)),
            },
        )

    def _parse_stream_event(self, event: dict[str, Any]) -> Iterator[StreamChunk]:
        message = event.get("message", {})
        content = str(message.get("content", ""))
        if content:
            yield StreamChunk(content=content)
        if event.get("done"):
            yield StreamChunk(content="", finish_reason="stop", is_final=True)


def _message_to_dict(message: ChatMessage) -> dict[str, str]:
    return {"role": message.role, "content": message.content}
