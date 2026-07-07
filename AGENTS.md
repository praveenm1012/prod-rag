# AGENTS.md

## Cursor Cloud specific instructions

`prod-rag` is a Python 3.12 FastAPI RAG scaffold managed with Poetry. There is a single service: the FastAPI HTTP API (`app.main:app`). The `app/rag/` package is an unimplemented placeholder. No database or external services are required to run or test it.

### Environment notes
- Poetry (1.8.x) is installed at `~/.local/bin` and added to `PATH` via `~/.bashrc`. Non-login shells may not have it on `PATH`; if `poetry` is not found, use `~/.local/bin/poetry` or `export PATH="$HOME/.local/bin:$PATH"`.
- Dependencies are installed into a Poetry-managed virtualenv, so run tools via `poetry run ...` (not the bare commands).
- The app reads config from a `.env` file (see `.env.example`). The update script copies `.env.example` to `.env` if missing; without it settings fall back to defaults, so the API still runs.

### Common commands (see `README.md` and `scripts/`)
- Run dev server: `poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` (health check at `GET /api/v1/health`, Swagger UI at `/docs`).
- Lint + tests together: `./scripts/lint.sh` (runs ruff, black, mypy, pytest).
- Tests only: `poetry run pytest -v`.
