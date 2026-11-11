# events2rag overview

**What:** polls event feeds (JSON, optional ICS), embeds events and their
occurrences, and upserts them into Qdrant for RAG.
**Runs as:** a long-running worker (`events2rag`); no listening port.
**Talks to:** the feed URLs, Qdrant, and Ollama (or an in-process model).

## Files

- `src/events2rag/__main__.py`: command line, logging, SIGTERM handling, wiring.
- `src/events2rag/config.py`: validated `Settings` from environment variables.
- `src/events2rag/service.py`: `IngestionService`: one cycle, and the polling loop.
- `src/events2rag/json_feed.py`, `ics_feed.py`: fetch and parse the feeds.
- `src/events2rag/embedder.py`: `Embedder` protocol; Ollama, ONNX, sentence-transformers.
- `src/events2rag/qdrant_store.py`: collection, payload indexes, upserts.
- `src/events2rag/models.py`: `EventOccurrence`, `EventSummary`, embedding text.
- `src/events2rag/text_utils.py`, `datetime_utils.py`: pure helpers.
- `tests/`: pytest; one file per module. Qdrant runs in-memory; the
  network and optional model libraries are faked.
- `Dockerfile`: `dev` and `production` stages.
- `docker-compose.yaml` (dev, with Qdrant), `.test`, `.lint`, `.prod`.

## Conventions

- All configuration from environment variables; see docs/configuration.md.
- Point IDs are UUIDv5 of the occurrence or `event:<id>`; never change the
  scheme without a migration plan, or points duplicate.
- Python 3.14, not 3.15, until onnxruntime ships 3.15 wheels.
- Rules: https://github.com/romkey/python-ai-rules/blob/main/RULES.md
