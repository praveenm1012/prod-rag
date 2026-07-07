#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RUN_DOCKER_TESTS=1
poetry run pytest tests/rag/vectorstore -v "$@"
