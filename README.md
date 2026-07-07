# prod-rag

Production-grade RAG (Retrieval-Augmented Generation) application scaffold built with Python 3.12, FastAPI, and Poetry.

## Features

- **FastAPI** HTTP API with versioned routes
- **Poetry** dependency and packaging management
- **pydantic-settings** for typed configuration from environment variables
- **structlog** for structured logging (console or JSON)
- **Docker** and **Docker Compose** for containerized deployment
- **Ruff**, **Black**, **mypy**, **pytest**, and **pre-commit** for code quality

## Project structure

```
.
├── app/
│   ├── api/          # HTTP routes and handlers
│   ├── core/         # Config, logging, shared utilities
│   ├── rag/          # RAG pipeline (to be implemented)
│   └── main.py       # Application entry point
├── tests/
├── scripts/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

## Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) 1.8+
- Docker and Docker Compose (optional, for containerized runs)

## Quick start (local)

```bash
# 1. Clone and enter the project
cd prod-rag

# 2. Copy environment template
cp .env.example .env

# 3. Install dependencies and pre-commit hooks
chmod +x scripts/*.sh
./scripts/setup-dev.sh

# 4. Run the API
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Verify the health endpoint

```bash
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```

Expected response:

```json
{
  "status": "ok",
  "service": "prod-rag",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2026-07-07T12:00:00+00:00"
}
```

Interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Docker

```bash
# Build and start
cp .env.example .env
docker compose up --build -d

# Check health
curl -s http://localhost:8000/api/v1/health

# View logs
docker compose logs -f api

# Stop
docker compose down
```

## Development commands

```bash
# Run all linters and tests
./scripts/lint.sh

# Individual tools
poetry run ruff check app tests
poetry run ruff format app tests
poetry run black app tests
poetry run mypy app
poetry run pytest -v

# Pre-commit (runs on every commit after setup-dev.sh)
poetry run pre-commit run --all-files
```

## Configuration

All settings are defined in `app/core/config.py` and loaded from environment variables. See `.env.example` for available options.

| Variable    | Default       | Description                          |
|-------------|---------------|--------------------------------------|
| `APP_NAME`  | `prod-rag`    | Service name                         |
| `APP_ENV`   | `development` | Environment label                    |
| `APP_DEBUG` | `false`       | Enable debug mode and auto-reload    |
| `APP_HOST`  | `0.0.0.0`     | Bind host                            |
| `APP_PORT`  | `8000`        | Bind port                            |
| `LOG_LEVEL` | `INFO`        | Logging level                        |
| `LOG_JSON`  | `false`       | Emit JSON logs (set `true` in prod) |

## Next steps

- Implement document ingestion in `app/rag/`
- Add vector store and embedding integrations
- Wire retrieval and generation endpoints under `app/api/`
