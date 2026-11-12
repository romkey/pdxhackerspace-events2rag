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
| `EMBEDDING_MODEL_NAME` | `nomic-ai/modernbert-embed-base` | Model name: one the API at `OPENAI_BASE_URL` serves, or a Hugging Face repo for `onnx` and `sentence-transformers`. Changing it changes the vector size, so use a new `QDRANT_COLLECTION` (or drop the old one). |
| `EMBEDDING_BATCH_SIZE` | `64` | Texts per embedding request (1 or more) |
| `EMBEDDING_CONTEXT_LENGTH` | `512` | The model's token limit; longer texts are truncated, with the prefix counted (1 or more) |
| `EMBEDDING_QUERY_PREFIX` | `search_query:` | Prompt prefix the model expects on search queries; see [Prompt prefixes](#prompt-prefixes). `none` for no prefix. |
| `EMBEDDING_DOCUMENT_PREFIX` | `search_document:` | Prompt prefix the model expects on documents; events are embedded with it. `none` for no prefix. |
| `OPENAI_BASE_URL` | `http://localhost:8080/v1` | OpenAI-compatible API, for the `openai` backend: a local server (text-embeddings-inference, vLLM, Ollama at `http://host:11434/v1`, LM Studio) or `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | none | Sent as a bearer token. Required for api.openai.com. Secret. |

### Prompt prefixes

Many embedding models were trained with a task prefix on search queries,
on the documents being searched, or both, and retrieve noticeably worse
without it. Set the two prefixes to match `EMBEDDING_MODEL_NAME`:

| Model | `EMBEDDING_QUERY_PREFIX` | `EMBEDDING_DOCUMENT_PREFIX` |
|-------|--------------------------|-----------------------------|
| `nomic-ai/modernbert-embed-base` (default), `nomic-embed-text` | `search_query:` | `search_document:` |
| e5 family (`intfloat/e5-*`, `multilingual-e5-*`) | `query:` | `passage:` |
| `mxbai-embed-large` | `Represent this sentence for searching relevant passages:` | `none` |
| Qwen3-Embedding and similar instruction models | an instruction line, e.g. `Instruct: Given a question, find matching events\nQuery:` | `none` |
| OpenAI `text-embedding-3-*` | `none` | `none` |

A space is added after a prefix unless it already ends in whitespace,
and `\n` becomes a newline. An empty value means the default
(the ModernBERT prefixes), so write `none` to send text without one.

The ingester embeds events with the document prefix. Anything that
searches the collection must embed its query with the same model and the
query prefix; see the README's "Searching the collection".

Each stored occurrence records the backend, model and document prefix it
was embedded with. Changing any of them re-embeds every occurrence on the
next cycle; event summaries are re-embedded every cycle anyway. The
query prefix doesn't affect stored vectors, so changing it needs no
re-embedding.

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
