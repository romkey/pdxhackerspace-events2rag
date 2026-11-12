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
| `openai` (default) | base | ~150 MB | Calls an OpenAI-compatible embeddings API (local provider or hosted) |
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
- `EMBEDDING_MODEL_NAME`: model name (default `nomic-ai/modernbert-embed-base`)
- `EMBEDDING_QUERY_PREFIX`: prompt prefix for search queries (default `search_query:`)
- `EMBEDDING_DOCUMENT_PREFIX`: prompt prefix for ingested events (default `search_document:`)
- `EMBEDDING_CONTEXT_LENGTH`: max tokens per input, prefix included (default `512`)
- `OPENAI_BASE_URL`: OpenAI-compatible API base URL (default `http://localhost:8080/v1`)
- `OPENAI_API_KEY`: optional API key, sent as a bearer token (required for api.openai.com)

`.env.example` lists the prefixes for common models (nomic/ModernBERT, e5, mxbai, Qwen3-Embedding, OpenAI).

## Run with Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

This starts:

- `qdrant` on `localhost:6333`
- `ingester` that fetches and upserts every hour

The ingester calls an OpenAI-compatible embeddings API at `OPENAI_BASE_URL`. The default is a local provider serving [ModernBERT Embed](https://huggingface.co/nomic-ai/modernbert-embed-base) on port 8080. For example, with Hugging Face text-embeddings-inference:

```bash
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:cpu-latest \
  --model-id nomic-ai/modernbert-embed-base
```

From inside the compose network, use `http://host.docker.internal:8080/v1` (or add the provider as a compose service). Other OpenAI-compatible servers such as vLLM or Ollama (`http://ollama:11434/v1`) work too. Set `EMBEDDING_MODEL_NAME` to a model they serve, and set the prefixes to match it.

Changing the embedding model changes the vector dimension, so point `QDRANT_COLLECTION` at a new collection (or drop the old one) when switching models. Changing only `EMBEDDING_DOCUMENT_PREFIX` keeps the dimension: every stored occurrence is re-embedded automatically on the next cycle.

## Querying the collection

Events are embedded with `EMBEDDING_DOCUMENT_PREFIX`. Any app that searches Qdrant directly must embed its query with the **same model** and the matching `EMBEDDING_QUERY_PREFIX`, or results get noticeably worse:

```python
import requests
from qdrant_client import QdrantClient

QUERY_PREFIX = "search_query: "  # must match EMBEDDING_QUERY_PREFIX

resp = requests.post(
    "http://localhost:8080/v1/embeddings",
    json={
        "model": "nomic-ai/modernbert-embed-base",
        "input": [QUERY_PREFIX + "What's happening this weekend?"],
    },
    timeout=30,
)
vector = resp.json()["data"][0]["embedding"]

hits = QdrantClient("http://localhost:6333").query_points(
    "events", query=vector, limit=5
).points
```

Python code that uses this package can call `build_embedder(Settings.from_env()).embed([text], kind="query")` from `events2rag.main`, which applies the configured query prefix and truncation for you.

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
