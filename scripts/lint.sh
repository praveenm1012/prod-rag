#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

poetry run ruff check app tests
poetry run ruff format --check app tests
poetry run black --check app tests
poetry run mypy app
poetry run pytest -v
