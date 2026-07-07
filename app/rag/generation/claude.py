"""Anthropic Claude messages provider."""

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


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude messages API provider."""

    provider_name = "claude"

    def __init__(
        self,
        settings: GenerationSettings,
        http_client: HttpClient,
    ) -> None:
        super().__init__(settings, http_client)
        if not settings.anthropic_api_key:
            raise GenerationConfigError(
                "ANTHROPIC_API_KEY is required for Claude provider"
            )

    def complete(self, request: GenerationRequest) -> GenerationResponse:
        payload = self._build_payload(request, stream=False)
        data = self._post_json(
            f"{self._settings.anthropic_base_url.rstrip('/')}/v1/messages",
            headers=self._headers(),
            payload=payload,
        )
        return self._parse_response(data, request)

    def stream(self, request: GenerationRequest) -> Iterator[StreamChunk]:
        payload = self._build_payload(request, stream=True)
        url = f"{self._settings.anthropic_base_url.rstrip('/')}/v1/messages"
        for line in self._stream_lines(url, headers=self._headers(), payload=payload):
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            try:
                event = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ProviderResponseError(
                    "Claude stream returned invalid JSON"
                ) from exc
            yield from self._parse_stream_event(event)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._settings.anthropic_api_key or "",
            "anthropic-version": self._settings.anthropic_version,
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        request: GenerationRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        system_prompt, messages = _split_system_message(request.messages)
        payload: dict[str, Any] = {
            "model": self._resolve_model(request),
            "messages": [_message_to_dict(message) for message in messages],
            "temperature": self._resolve_temperature(request),
            "max_tokens": self._resolve_max_tokens(request),
            "stream": stream,
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def _parse_response(
        self,
        data: dict[str, Any],
        request: GenerationRequest,
    ) -> GenerationResponse:
        try:
            content_blocks = data["content"]
            content = "".join(
                str(block.get("text", ""))
                for block in content_blocks
                if block.get("type") == "text"
            )
            usage = {
                "prompt_tokens": int(data.get("usage", {}).get("input_tokens", 0)),
                "completion_tokens": int(data.get("usage", {}).get("output_tokens", 0)),
                "total_tokens": int(data.get("usage", {}).get("input_tokens", 0))
                + int(data.get("usage", {}).get("output_tokens", 0)),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderResponseError(
                "Claude response missing expected fields"
            ) from exc

        logger.info(
            "claude_generation_complete",
            model=data.get("model", self._resolve_model(request)),
            stop_reason=data.get("stop_reason"),
        )
        return GenerationResponse(
            content=content,
            model=str(data.get("model", self._resolve_model(request))),
            provider=self.provider_name,
            finish_reason=data.get("stop_reason"),
            usage=usage,
        )

    def _parse_stream_event(self, event: dict[str, Any]) -> Iterator[StreamChunk]:
        event_type = event.get("type")
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            text = str(delta.get("text", ""))
            if text:
                yield StreamChunk(content=text)
        elif event_type == "message_stop":
            yield StreamChunk(content="", finish_reason="end_turn", is_final=True)


def _split_system_message(
    messages: tuple[ChatMessage, ...],
) -> tuple[str | None, list[ChatMessage]]:
    if not messages:
        return None, []
    if messages[0].role == "system":
        return messages[0].content, list(messages[1:])
    return None, list(messages)


def _message_to_dict(message: ChatMessage) -> dict[str, str]:
    return {"role": message.role, "content": message.content}
