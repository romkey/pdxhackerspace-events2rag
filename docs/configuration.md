# Configuration

events2rag reads these environment variables at startup. Put them in `.env`
(copy `.env.example`); compose passes it to the container. An invalid
value stops the app with exit status 2 and a message naming the variable.
An empty value means the default.

## Feeds

| Variable | Default | Meaning |
|----------|---------|---------|
| `EVENTS_JSON_URL` | `https://events.pdxhackerspace/events.json` | JSON event feed (`http://` or `https://`) |
| `EVENTS_ICS_URL` | none | Optional ICS calendar feed, read as well as the JSON feed |
| `ICS_LOOKBACK_DAYS` | `30` | How far back to expand ICS recurrences, in days (0 or more) |
| `ICS_LOOKAHEAD_DAYS` | `365` | How far ahead to expand ICS recurrences, in days (0 or more) |
| `POLL_INTERVAL_SECONDS` | `3600` | Time between ingestion cycles (1 or more) |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Timeout for feed and embeddings API requests (1 or more) |

## Qdrant

| Variable | Default | Meaning |
|----------|---------|---------|
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant HTTP endpoint. The compose files set it to their own Qdrant. |
| `QDRANT_API_KEY` | none | Qdrant API key, if the server needs one. Secret. |
| `QDRANT_COLLECTION` | `events` | Collection to create and upsert into |
| `QDRANT_DISTANCE` | `Cosine` | Vector distance for a new collection: `Cosine`, `Dot`, `Euclid` or `Manhattan` |

## Embeddings

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_BACKEND` | `openai` | `openai`, `onnx` (needs the `onnx` extra) or `sentence-transformers` (needs the `torch` extra). The Docker image has only `openai`. |
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | Model name: one the API at `OPENAI_BASE_URL` serves, or a Hugging Face repo for `onnx` and `sentence-transformers`. Changing it changes the vector size, so use a new `QDRANT_COLLECTION` (or drop the old one). |
| `EMBEDDING_BATCH_SIZE` | `64` | Texts per embedding request (1 or more) |
| `EMBEDDING_CONTEXT_LENGTH` | `512` | The model's token limit; longer texts are truncated (1 or more) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API, for the `openai` backend: api.openai.com, or Ollama (`http://host:11434/v1`), vLLM, LM Studio |
| `OPENAI_API_KEY` | none | Sent as a bearer token. Required for api.openai.com. Secret. |

## Logging

| Variable | Default | Meaning |
|----------|---------|---------|
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

## Production compose (not read by the app)

| Variable | Default | Meaning |
|----------|---------|---------|
| `APP_VERSION` | `latest` | Image tag `docker-compose.prod.yaml` runs, e.g. `1.2.3` |

## Publishing (not read by the app)

| Variable | Meaning |
|----------|---------|
| `UV_PUBLISH_TOKEN` | PyPI API token, only for publishing by hand with `uv publish`. Releases from GitHub don't need it. |
