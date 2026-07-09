#!/usr/bin/env python3
"""Minimal Langfuse-compatible server for local integration testing."""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


class LangfuseMockState:
    def __init__(self) -> None:
      self.health_checks = 0
      self.trace_batches: list[dict[str, Any]] = []
      self.auth_failures = 0


STATE = LangfuseMockState()


class LangfuseMockHandler(BaseHTTPRequestHandler):
    expected_public_key = ""
    expected_secret_key = ""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            STATE.auth_failures += 1
            return False

        import base64

        encoded = header.removeprefix("Basic ").strip()
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except Exception:
            STATE.auth_failures += 1
            return False

        public_key, _, secret_key = decoded.partition(":")
        authorized = (
            public_key == self.expected_public_key
            and secret_key == self.expected_secret_key
        )
        if not authorized:
            STATE.auth_failures += 1
        return authorized

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/public/health":
            STATE.health_checks += 1
            self._json(200, {"status": "OK", "version": "mock"})
            return

        if path == "/api/public/projects":
            if not self._authorized():
                self._json(401, {"message": "Unauthorized"})
                return
            self._json(
                200,
                {
                    "data": [
                        {
                            "id": "proj-mock",
                            "name": "prod-rag",
                            "organization": {
                                "id": "org-mock",
                                "name": "prod-rag-org",
                            },
                            "metadata": {},
                        }
                    ]
                },
            )
            return

        if path.startswith("/api/public/traces/"):
            if not self._authorized():
                self._json(401, {"message": "Unauthorized"})
                return
            trace_id = path.rsplit("/", 1)[-1]
            self._json(
                200,
                {
                    "id": trace_id,
                    "name": "mock-trace",
                    "observations": [{"id": "obs-1", "name": "application.startup"}],
                },
            )
            return

        self._json(404, {"message": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/public/otel/v1/traces":
            if not self._authorized():
                self._json(401, {"message": "Unauthorized"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            STATE.trace_batches.append(
                {
                    "path": path,
                    "bytes": len(raw),
                    "content_type": self.headers.get("Content-Type", ""),
                }
            )
            self.send_response(200)
            self.end_headers()
            return

        self._json(404, {"message": "not found"})


def start_mock_server(
    *,
    host: str,
    port: int,
    public_key: str,
    secret_key: str,
) -> ThreadingHTTPServer:
    handler = type(
        "ConfiguredLangfuseMockHandler",
        (LangfuseMockHandler,),
        {
            "expected_public_key": public_key,
            "expected_secret_key": secret_key,
        },
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.allow_reuse_address = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--public-key", required=True)
    parser.add_argument("--secret-key", required=True)
    args = parser.parse_args()

    server = start_mock_server(
        host=args.host,
        port=args.port,
        public_key=args.public_key,
        secret_key=args.secret_key,
    )
    print(f"Mock Langfuse listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
