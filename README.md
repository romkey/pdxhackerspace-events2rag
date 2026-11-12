# events2rag

Event feed ingester for RAG. It fetches a JSON event feed (and optional ICS feed), turns events into embeddings, and upserts them into Qdrant on a recurring schedule.

## Features

- Environment-variable configuration
- Polling loop that rescans every hour by default
- JSON feed ingestion (`EVENTS_JSON_URL`)
- Optional ICS ingestion with recurrence expansion (`EVENTS_ICS_URL`)
- HTML stripping from event descriptions
- Human-friendly dates with day-of-week in embeddings
- Duration and temporal status (past/current/future) per occurrence
- Event-level summaries with frequency estimation
- Qdrant payload indexes for hybrid vector + filter search
- Resilient polling loop that survives transient errors
- Dockerized runtime
- GitHub Actions for CI and Docker image publishing

## Embedding backends

| Backend | Install | Image size | Description |
|---------|---------|------------|-------------|
| `openai` (default) | base | ~150 MB | Calls the OpenAI embeddings API (or any OpenAI-compatible server) |
| `onnx` | `pip install .[onnx]` | ~350 MB | In-process ONNX Runtime |
| `sentence-transformers` | `pip install .[torch]` | ~1+ GB | Full PyTorch stack |

## Configuration

Copy `.env.example` to `.env` and update values.

Key variables:

- `EVENTS_JSON_URL`: source JSON feed URL
- `EVENTS_ICS_URL`: optional ICS feed URL
- `POLL_INTERVAL_SECONDS`: scan interval (default `3600`)
- `QDRANT_URL`: Qdrant endpoint
- `QDRANT_COLLECTION`: collection name
- `EMBEDDING_BACKEND`: `openai`, `onnx`, or `sentence-transformers`
- `EMBEDDING_MODEL_NAME`: model name (default `text-embedding-3-small`)
- `OPENAI_BASE_URL`: OpenAI-compatible API base URL (default `https://api.openai.com/v1`)
- `OPENAI_API_KEY`: optional API key, sent as a bearer token (required for api.openai.com)

## Run with Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

This starts:

- `qdrant` on `localhost:6333`
- `ingester` that fetches and upserts every hour

The ingester calls the OpenAI embeddings API at `OPENAI_BASE_URL`. To use a local OpenAI-compatible server instead (such as Ollama), set `OPENAI_BASE_URL` to its `/v1` endpoint, e.g. `http://host.docker.internal:11434/v1`, leave `OPENAI_API_KEY` empty if the server does not need one, and set `EMBEDDING_MODEL_NAME` to a model it serves.

Changing the embedding model changes the vector dimension; point `QDRANT_COLLECTION` at a new collection (or drop the old one) when switching models.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
events2rag
```

## Run lint and tests in containers

```bash
docker compose -f docker-compose.dev.yml run --rm lint
docker compose -f docker-compose.dev.yml run --rm test
```

## GitHub Actions

- `.github/workflows/ci.yml` runs lint and tests on pushes/PRs.
- `.github/workflows/docker-publish.yml` builds and publishes to GHCR on semantic version tags.

### Semantic versioning and publishing

Create and push a git tag:

```bash
git tag v0.5.0
git push origin v0.5.0
```

This publishes image tags like:

- `ghcr.io/<owner>/<repo>:0.5.0`
- `ghcr.io/<owner>/<repo>:0.5`
- `ghcr.io/<owner>/<repo>:0`
- `ghcr.io/<owner>/<repo>:latest`
