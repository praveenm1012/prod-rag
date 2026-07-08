# prod-rag — Feature Reference

Complete list of features built in **prod-rag**, what each one does, and how to use it.

---

## Table of contents

1. [Overview](#overview)
2. [HTTP API](#http-api)
3. [Browser test console](#browser-test-console)
4. [Document ingestion](#document-ingestion)
5. [Chunking](#chunking)
6. [Embeddings](#embeddings)
7. [Lexical search (BM25)](#lexical-search-bm25)
8. [Vector store (Qdrant)](#vector-store-qdrant)
9. [Hybrid retrieval](#hybrid-retrieval)
10. [Cross-encoder reranking](#cross-encoder-reranking)
11. [Prompt builder](#prompt-builder)
12. [LLM generation](#llm-generation)
13. [RAG pipeline](#rag-pipeline)
14. [Document management](#document-management)
15. [Ragas evaluation](#ragas-evaluation)
16. [Scripts and CLI tools](#scripts-and-cli-tools)
17. [Testing](#testing)
18. [Configuration reference](#configuration-reference)
19. [Docker and development](#docker-and-development)

---

## Overview

**prod-rag** is a production-oriented Retrieval-Augmented Generation (RAG) application. It ingests documents, indexes them with hybrid lexical + vector search, reranks results, builds grounded prompts with citations, and generates answers via pluggable LLM providers.

**Typical flow:**

```
Upload → Ingest → Chunk → Embed → Index → Retrieve → Rerank → Prompt → Generate → Cite
```

**Stack:** Python 3.12 · FastAPI · Poetry · sentence-transformers · rank-bm25 · Qdrant (optional) · OpenAI / Claude / Ollama

---

## HTTP API

Base path: `/api/v1`  
Interactive docs: `http://localhost:8000/docs`

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Service health, version, environment |

**Use:** Liveness checks, load balancers, monitoring.

```bash
curl http://localhost:8000/api/v1/health
```

---

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/chat/status` | LLM provider name, model, configured flag |
| `POST` | `/api/v1/chat` | Chat completion (optional RAG) |
| `POST` | `/api/v1/chat/stream` | Streaming chat (optional RAG) |

**Request body (`POST /chat`):**

```json
{
  "question": "What is Acme Corporation known for?",
  "use_rag": true,
  "top_k": 3,
  "history": [
    { "role": "user", "content": "Previous question" },
    { "role": "assistant", "content": "Previous answer" }
  ]
}
```

**Response (`use_rag: true`):**

```json
{
  "answer": "Acme specializes in RAG testing [1].",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "usage": { "total_tokens": 110 },
  "citations": ["[1]"]
}
```

**Use:**

- `use_rag: false` — direct LLM chat without retrieval
- `use_rag: true` — full RAG pipeline (retrieve → rerank → prompt → generate)
- `/chat/stream` — token-by-token streaming for UI or CLI clients

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?", "use_rag": true, "top_k": 3}'
```

---

### Upload and indexing

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/upload` | Upload, embed, and index a document |
| `POST` | `/api/v1/upload?background=true` | Accept upload immediately; index in background (202) |
| `GET` | `/api/v1/upload/{document_id}/status` | Background upload progress and phase |

**Supported file types:** `.pdf`, `.txt`, `.md`, `.markdown`, `.docx`

**Sync upload (default)** — returns `201` when fully indexed:

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@document.pdf"
```

**Background upload** — returns `202` immediately; poll status for large files:

```bash
curl -X POST "http://localhost:8000/api/v1/upload?background=true" \
  -F "file=@large-document.pdf"

curl http://localhost:8000/api/v1/upload/{document_id}/status
```

**Status phases:** `queued` → `ingesting` → `chunking` → `embedding` → `indexing` → `indexed` (or `failed`)

**Use:** Background mode is recommended for multi-MB PDFs. Embedding on CPU can take several minutes for large documents.

---

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/documents` | List all uploaded documents |
| `DELETE` | `/api/v1/delete` | Delete a document and its index entries |

**List documents:**

```bash
curl http://localhost:8000/api/v1/documents
```

**Delete document:**

```bash
curl -X DELETE http://localhost:8000/api/v1/delete \
  -H "Content-Type: application/json" \
  -d '{"document_id": "abc123..."}'
```

---

## Browser test console

| Route | Description |
|-------|-------------|
| `GET /` | API test console (main UI) |
| `GET /ui` | Alias for test console |
| `GET /static/*` | CSS and JavaScript assets |

**Features:**

- Health and LLM provider status cards
- File upload with background processing and progress bar
- Document list with status (`processing`, `indexed`, `failed`), delete, and quick-ask
- Chat panel with `use_rag`, `top_k`, and streaming toggles
- Answer display with citation pills
- Request log for debugging API calls

**Use:** Manual end-to-end testing without curl or Postman.

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/
```

---

## Document ingestion

**Location:** `app/rag/ingestion/`

| Loader | Extensions | Use |
|--------|------------|-----|
| `PDFLoader` | `.pdf` | Extract text from PDF pages via pypdf |
| `TextLoader` | `.txt` | Plain text files |
| `MarkdownLoader` | `.md`, `.markdown` | Markdown files |
| `DocxLoader` | `.docx` | Word documents |

**API:** `load_document(path)` dispatches to the correct loader via extension registry.

**Use:** Called automatically during upload. Produces a normalized `Document` with text, metadata (format, filename, page count), and optional per-page text for chunking.

---

## Chunking

**Location:** `app/rag/chunking/`

**Component:** `RecursiveCharacterTextSplitter`

| Setting | Default | Description |
|---------|---------|-------------|
| `chunk_size` | 1000 | Target characters per chunk |
| `chunk_overlap` | 200 | Overlap between adjacent chunks |

**Output per chunk:** `chunk_id`, `text`, `source`, `page_number`, `document_id`, metadata

**Use:** Splits ingested documents into retrieval-sized pieces before embedding. Overlap preserves context across chunk boundaries.

---

## Embeddings

**Location:** `app/rag/embeddings/`

**Component:** `EmbeddingService` (sentence-transformers)

| Feature | Description |
|---------|-------------|
| Model | `BAAI/bge-large-en-v1.5` (configurable) |
| Query vs document prompts | BGE query prefix for search queries |
| Batching | Configurable batch size (default 32) |
| Disk cache | diskcache-backed embedding cache |
| Retry | Tenacity exponential backoff on failures |
| Normalization | L2-normalized vectors (cosine similarity) |
| Device | `auto` selects CUDA if available, else CPU |

**Use:** Converts chunk text and search queries into vectors for semantic retrieval.

```bash
poetry run python scripts/benchmark_embeddings.py --text-count 100 --repetitions 3
```

---

## Lexical search (BM25)

**Location:** `app/rag/lexical/`

**Component:** `BM25Searcher`

| Feature | Description |
|---------|-------------|
| Algorithm | Okapi BM25 via rank-bm25 |
| Tokenization | Custom tokenizer for indexing and search |
| Metadata filters | Filter by `document_id`, `page_number`, etc. |
| Configurable `k1`, `b` | BM25 tuning parameters |

**Use:** Keyword-based retrieval; strong on exact term matches. Used as one signal in hybrid retrieval.

```bash
poetry run python scripts/benchmark_lexical_search.py --query-count 5
```

---

## Vector store (Qdrant)

**Location:** `app/rag/vectorstore/`

**Component:** `QdrantRepository`

| Feature | Description |
|---------|-------------|
| Collection management | Create, recreate, health check |
| Insert / search | Vector similarity search with metadata filters |
| Delete | By ID or metadata filter |

**Use:** Optional persistent vector backend. Hybrid retriever can use in-memory vectors or Qdrant when configured.

```bash
./scripts/run-qdrant-tests.sh   # Requires Docker
```

---

## Hybrid retrieval

**Location:** `app/rag/retrieval/`

**Component:** `HybridRetriever`

| Feature | Description |
|---------|-------------|
| BM25 + vector fusion | Weighted score merge of lexical and semantic results |
| Score normalization | Min-max normalization before fusion |
| Configurable weights | `HYBRID_BM25_WEIGHT`, `HYBRID_VECTOR_WEIGHT` |
| Candidate pool | Retrieves a larger pool, returns top-k after fusion |
| Component scores | Each result includes BM25, vector, and fused scores |

**Use:** Combines keyword and semantic search for better recall on varied query types.

```bash
poetry run python scripts/benchmark_hybrid_retrieval.py \
  --report-path reports/hybrid_evaluation.md
```

Output: `reports/hybrid_evaluation.md` with per-query recall@k and MRR.

---

## Cross-encoder reranking

**Location:** `app/rag/reranking/`

**Component:** `CrossEncoderReranker`

| Feature | Description |
|---------|-------------|
| Model | `BAAI/bge-reranker-large` |
| Top-k | Returns top 5 candidates by default |
| Batching | Configurable batch size with progress |
| Retry | Tenacity backoff on model errors |

**Use:** Re-scores hybrid retrieval candidates with a cross-encoder for higher precision before prompt building.

---

## Prompt builder

**Location:** `app/rag/prompts/`

**Component:** `PromptBuilder`

| Feature | Description |
|---------|-------------|
| Context formatting | Numbered citation placeholders `[1]`, `[2]`, … |
| Conversation history | Prior user/assistant turns |
| Few-shot examples | Optional Q&A examples in prompt |
| Token budget | Truncates context/history to fit `PROMPT_MAX_CONTEXT_TOKENS` |
| tiktoken counting | Accurate token measurement |
| System prompt | Separate system and user prompt sections |

**Use:** Assembles the final LLM prompt from reranked chunks, history, and the user question. Citations in the answer map back to source chunks.

---

## LLM generation

**Location:** `app/rag/generation/`

**Providers:**

| Provider | Config | Use |
|----------|--------|-----|
| OpenAI | `LLM_PROVIDER=openai`, `OPENAI_API_KEY` | GPT models via OpenAI API |
| Claude | `LLM_PROVIDER=claude`, `ANTHROPIC_API_KEY` | Anthropic models |
| Ollama | `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL` | Local models via Ollama |

| Feature | Description |
|---------|-------------|
| Streaming | Token-by-token `stream()` generator |
| Retry | Exponential backoff on transient failures |
| Timeout | Configurable per-request timeout |
| Factory | `create_llm_provider()` selects provider from settings |

**Use:** Generates the final answer from the assembled prompt. Swappable via `.env` without code changes.

```bash
poetry run python scripts/verify_llm.py --question "Reply with exactly: LLM provider is working."
```

---

## RAG pipeline

**Location:** `app/rag/pipeline/`

**Component:** `RagPipeline`

**Orchestration steps:**

1. **Retrieve** — hybrid search for top-k chunks
2. **Rerank** — cross-encoder rescores candidates
3. **Build prompt** — context, history, citations, question
4. **Generate** — LLM completion or stream

| Method | Description |
|--------|-------------|
| `ask()` | Full RAG answer with metadata |
| `stream()` | Streaming RAG answer |
| `chat()` | Direct LLM without retrieval |

**Use:** Single entry point used by `/api/v1/chat` and `/api/v1/chat/stream`.

---

## Document management

**Location:** `app/rag/documents/`

**Component:** `DocumentService`

| Feature | Description |
|---------|-------------|
| Upload | Save, ingest, chunk, embed, index |
| Background upload | Queue indexing; return immediately |
| List | All documents with status and metadata |
| Delete | Remove document, chunks, vectors, and file |
| Thread-safe | Lock-protected index mutations |

**Document statuses:**

| Status | Meaning |
|--------|---------|
| `processing` | Ingest / chunk / embed / index in progress |
| `indexed` | Ready for RAG queries |
| `failed` | Error during processing (see `error_message`) |

---

## Ragas evaluation

**Location:** `app/rag/evaluation/`

**Metrics:**

| Metric | Source | What it measures |
|--------|--------|------------------|
| Faithfulness | Ragas | Answer claims supported by retrieved context |
| Answer Relevancy | Ragas | Answer alignment with the question |
| Context Precision | Ragas | Relevance/ranking of retrieved contexts |
| Citation Accuracy | Custom | Expected `[n]` placeholders present in response |

**Artifacts:**

| Path | Description |
|------|-------------|
| `data/evaluation/sample_rag_dataset.json` | 5-sample Acme Corporation eval set |
| `reports/ragas_evaluation.html` | HTML report with aggregate and per-sample scores |

**Use:** Offline quality measurement of RAG answers. Requires `OPENAI_API_KEY` for Ragas LLM metrics.

```bash
poetry run python scripts/run_ragas_evaluation.py
poetry run python scripts/run_ragas_evaluation.py --json-only
poetry run python scripts/run_ragas_evaluation.py \
  --dataset data/evaluation/sample_rag_dataset.json \
  --report-path reports/ragas_evaluation.html
```

---

## Scripts and CLI tools

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/setup-dev.sh` | Install deps and pre-commit hooks | `./scripts/setup-dev.sh` |
| `scripts/lint.sh` | Ruff, Black, mypy, pytest | `./scripts/lint.sh` |
| `scripts/verify_llm.py` | Test LLM provider connectivity | `poetry run python scripts/verify_llm.py` |
| `scripts/test_api_smoke.py` | Live API smoke (upload → chat → delete) | `poetry run python scripts/test_api_smoke.py` |
| `scripts/e2e_smoke_test.py` | End-to-end PDF workflow with timing metrics | `poetry run python scripts/e2e_smoke_test.py --mode api` |
| `scripts/run_ragas_evaluation.py` | Ragas eval + HTML report | `poetry run python scripts/run_ragas_evaluation.py` |
| `scripts/benchmark_embeddings.py` | Embedding throughput benchmark | `poetry run python scripts/benchmark_embeddings.py` |
| `scripts/benchmark_lexical_search.py` | BM25 search benchmark | `poetry run python scripts/benchmark_lexical_search.py` |
| `scripts/benchmark_hybrid_retrieval.py` | Hybrid vs BM25 vs vector benchmark | `poetry run python scripts/benchmark_hybrid_retrieval.py` |
| `scripts/visualize_chunks.py` | Chunk size distribution plot | `poetry run python scripts/visualize_chunks.py --file doc.pdf` |
| `scripts/run-qdrant-tests.sh` | Qdrant integration tests (Docker) | `./scripts/run-qdrant-tests.sh` |

### E2E smoke test options

```bash
# Against running API
poetry run python scripts/e2e_smoke_test.py --mode api --base-url http://127.0.0.1:8000

# In-process with per-phase timing
poetry run python scripts/e2e_smoke_test.py --mode local

# JSON output only
poetry run python scripts/e2e_smoke_test.py --mode api --json-only
```

**Workflow verified:** upload PDF → embed → index → ask → verify answer → verify citations → timing metrics.

---

## Testing

**Run all tests:**

```bash
poetry run pytest -v
# or
./scripts/lint.sh
```

| Suite | Location | Coverage |
|-------|----------|----------|
| API tests | `tests/api/` | Chat, upload, documents, delete, UI routes |
| E2E smoke | `tests/e2e/` | PDF workflow with fakes; optional live run |
| Ingestion | `tests/rag/ingestion/` | PDF, TXT, MD, DOCX loaders |
| Chunking | `tests/rag/chunking/` | Splitter config and output |
| Embeddings | `tests/rag/embeddings/` | Integration (gated by env) |
| Lexical | `tests/rag/lexical/` | BM25 search and filters |
| Retrieval | `tests/rag/retrieval/` | Hybrid fusion and ranking |
| Reranking | `tests/rag/reranking/` | Cross-encoder reranker |
| Prompts | `tests/rag/prompts/` | Prompt builder and truncation |
| Generation | `tests/rag/generation/` | OpenAI, Claude, Ollama providers |
| Evaluation | `tests/rag/evaluation/` | Citation accuracy, dataset, HTML report |
| Qdrant | `tests/rag/vectorstore/` | Docker integration (gated) |

**Optional integration flags:**

| Env var | Enables |
|---------|---------|
| `RUN_INTEGRATION_TESTS=1` | Embedding model integration tests |
| `RUN_E2E_SMOKE=1` | Live local e2e with real embeddings + LLM |
| `RUN_DOCKER_TESTS=1` | Qdrant Docker tests |

---

## Configuration reference

Copy `.env.example` to `.env` and set values as needed.

### Application

| Variable | Default | Use |
|----------|---------|-----|
| `APP_NAME` | `prod-rag` | Service name in health response |
| `APP_ENV` | `development` | Environment label |
| `APP_DEBUG` | `false` | Debug mode / auto-reload |
| `APP_HOST` | `0.0.0.0` | Bind host |
| `APP_PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LOG_JSON` | `false` | JSON log output (production) |

### Embeddings

| Variable | Default | Use |
|----------|---------|-----|
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Sentence-transformer model |
| `EMBEDDING_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `EMBEDDING_BATCH_SIZE` | `32` | Texts per embedding batch |
| `EMBEDDING_CACHE_DIR` | `.cache/embeddings` | Disk cache path |

### Retrieval and reranking

| Variable | Default | Use |
|----------|---------|-----|
| `HYBRID_BM25_WEIGHT` | `0.5` | Lexical weight in hybrid fusion |
| `HYBRID_VECTOR_WEIGHT` | `0.5` | Vector weight in hybrid fusion |
| `HYBRID_TOP_K` | `10` | Default retrieval top-k |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Cross-encoder model |
| `RERANKER_TOP_K` | `5` | Chunks passed to prompt builder |

### LLM

| Variable | Default | Use |
|----------|---------|-----|
| `LLM_PROVIDER` | `openai` | `openai`, `claude`, or `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `OPENAI_API_KEY` | — | Required for OpenAI provider |
| `ANTHROPIC_API_KEY` | — | Required for Claude provider |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### Prompts

| Variable | Default | Use |
|----------|---------|-----|
| `PROMPT_MAX_CONTEXT_TOKENS` | `4096` | Max prompt token budget |
| `PROMPT_RESERVED_RESPONSE_TOKENS` | `512` | Tokens reserved for LLM response |

See `.env.example` for the full list including Qdrant and lexical settings.

---

## Docker and development

| Feature | Location | Use |
|---------|----------|-----|
| Dockerfile | `Dockerfile` | Container image for the API |
| Docker Compose | `docker-compose.yml` | Local stack with API service |
| Pre-commit | `.pre-commit-config.yaml` | Git hooks for lint/format |
| Poetry | `pyproject.toml` | Dependency and tool management |

```bash
# Docker
docker compose up --build -d
curl http://localhost:8000/api/v1/health

# Local dev
cp .env.example .env
./scripts/setup-dev.sh
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Quick reference — common tasks

| Task | Command / URL |
|------|----------------|
| Start API | `poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| Open test UI | http://localhost:8000/ |
| API docs | http://localhost:8000/docs |
| Upload a file | `POST /api/v1/upload` or use the UI |
| Ask a RAG question | `POST /api/v1/chat` with `"use_rag": true` |
| Check LLM config | `GET /api/v1/chat/status` |
| Run all tests | `./scripts/lint.sh` |
| E2E smoke test | `poetry run python scripts/e2e_smoke_test.py --mode api` |
| Ragas evaluation | `poetry run python scripts/run_ragas_evaluation.py` |
| Verify LLM keys | `poetry run python scripts/verify_llm.py` |

---

## Performance notes

| Scenario | Expectation |
|----------|-------------|
| Small TXT upload (< 1 KB) | Seconds |
| Large PDF (20+ MB) on CPU | Several minutes (embedding is the bottleneck) |
| Background upload | HTTP returns in < 1s; indexing continues server-side |
| GPU (`EMBEDDING_DEVICE=cuda`) | Significantly faster embedding |
| Smaller model (`bge-small-en-v1.5`) | Faster, slightly lower retrieval quality |

For large files, use `?background=true` and poll `/api/v1/upload/{id}/status`, or use the browser test console which does this automatically.
