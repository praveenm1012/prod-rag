#!/usr/bin/env python3
"""Verify LLM provider configuration and connectivity."""

from __future__ import annotations

import argparse
import sys

from app.rag.generation import ChatMessage, GenerationRequest, create_llm_provider
from app.rag.generation.config import get_generation_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify LLM provider setup.")
    parser.add_argument(
        "--question",
        default="Reply with exactly: LLM provider is working.",
        help="Test prompt to send",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_generation_settings()

    print("LLM provider verification")
    print(f"Provider: {settings.provider}")
    print(f"Model: {settings.model}")
    print(f"Timeout: {settings.timeout_seconds}s")

    if settings.provider == "openai" and not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY is not set in .env")
        return 1
    if settings.provider == "claude" and not settings.anthropic_api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set in .env")
        return 1

    provider = create_llm_provider()
    request = GenerationRequest(
        messages=(
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content=args.question),
        )
    )

    print("\nSending test request...")
    try:
        response = provider.complete(request)
    except Exception as exc:
        print(f"ERROR: Provider request failed: {exc}")
        return 1

    print("\nSuccess!")
    print(f"Model used: {response.model}")
    print(f"Finish reason: {response.finish_reason}")
    print(f"Usage: {response.usage}")
    print(f"\nResponse:\n{response.content}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
