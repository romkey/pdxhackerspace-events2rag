# events2rag

[![CI](https://github.com/romkey/pdxhackerspace-events2rag/actions/workflows/ci.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-events2rag/actions/workflows/ci.yaml)
[![Lint](https://github.com/romkey/pdxhackerspace-events2rag/actions/workflows/lint.yaml/badge.svg)](https://github.com/romkey/pdxhackerspace-events2rag/actions/workflows/lint.yaml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Event feed ingester for retrieval-augmented generation (RAG). It fetches a
JSON event feed (and optionally an ICS calendar), turns each event and each
occurrence into an embedding, and upserts them into Qdrant, repeating every
hour, so a chatbot or search tool can answer "what's happening at the
hackerspace?" from current data.

> **AI disclosure:** This project was written by AI under human oversight,
> following [these rules](https://github.com/romkey/python-ai-rules/blob/main/RULES.md).

## What's here

| Path | What it is |
|------|------------|
| `src/events2rag/__main__.py` | Command line: reads settings, wires the parts together, handles SIGTERM |
| `src/events2rag/config.py` | `Settings` from environment variables, validated |
| `src/events2rag/json_feed.py` | Fetches the JSON feed and turns events into occurrences |
| `src/events2rag/ics_feed.py` | Fetches the ICS feed and expands recurring events |
| `src/events2rag/service.py` | One ingestion cycle: dedupe, enrich, skip unchanged, embed, upsert; the polling loop |
| `src/events2rag/embedder.py` | Embedding backends: OpenAI API, ONNX Runtime, sentence-transformers |
| `src/events2rag/qdrant_store.py` | Qdrant collection, payload indexes and upserts |
| `src/events2rag/models.py` | `EventOccurrence` and `EventSummary`, and the text that gets embedded |
| `src/events2rag/text_utils.py`, `datetime_utils.py` | HTML stripping, human-friendly dates and durations, frequency estimates |
| `tests/` | pytest tests, one file per module |
| `docs/` | [Configuration](docs/configuration.md) and an [overview](docs/overview.md) |

## Technical details

Each cycle:

1. Fetches `EVENTS_JSON_URL` and, if set, `EVENTS_ICS_URL`. The JSON parser
   accepts a list of events or an object holding one under `events`,
   `data`, `items` or `results`, and common field names for titles, dates,
   locations and tags. ICS recurrences are expanded from
   `ICS_LOOKBACK_DAYS` ago to `ICS_LOOKAHEAD_DAYS` ahead.
2. Strips HTML from descriptions, drops duplicates (preferring the JSON
   copy when both feeds list the same title at the same time), and adds
   each occurrence's duration and whether it's past, current or future.
3. Skips occurrences whose `last_modified` hasn't changed since they were
   last stored, embeds the rest, and upserts them with
   `record_type: occurrence`.
4. Builds one summary per event (next occurrence, locations, tags, an
   estimated frequency such as "weekly") and upserts it with
   `record_type: event_summary`.

Point IDs are UUIDs derived from the occurrence or event ID, so re-runs
update points instead of duplicating them. Payload indexes on
`record_type`, `temporal_status`, `tags`, `location`, `event_id`,
`start_time`, `end_time` and `has_future_occurrences` allow filtered
vector search. A failed cycle is logged and retried at the next interval.

Embedding backends, chosen with `EMBEDDING_BACKEND`:

| Backend | Install | What it does |
|---------|---------|--------------|
| `openai` (default) | included | Calls the OpenAI embeddings API, or any OpenAI-compatible server (Ollama, vLLM, LM Studio) |
| `onnx` | `pip install 'events2rag[onnx]'` | Runs the model in-process with ONNX Runtime |
| `sentence-transformers` | `pip install 'events2rag[torch]'` | Runs the model with PyTorch |

The Docker image includes only the `openai` backend. Changing the model
changes the vector size, so point `QDRANT_COLLECTION` at a new collection
(or drop the old one) when you switch.

```
events2rag [--version]
```

## Running it

Requires Docker, and either an OpenAI API key or a local
OpenAI-compatible server such as Ollama. Configuration comes from
environment variables, kept in `.env` (see
[docs/configuration.md](docs/configuration.md)):

```bash
cp .env.example .env   # then edit it: at least OPENAI_API_KEY
docker compose -f docker-compose.prod.yaml up -d
```

That runs the latest release from `ghcr.io/romkey/pdxhackerspace-events2rag`
next to a Qdrant container; set `APP_VERSION=X.Y.Z` to pin a release. To
use Ollama on the Docker host instead of OpenAI, set
`OPENAI_BASE_URL=http://host.docker.internal:11434/v1`, leave
`OPENAI_API_KEY` unset, and set `EMBEDDING_MODEL_NAME` to a model it serves.

### With pip

Needs Python 3.14 or newer. Install into a virtual environment from a
release's wheel:

```bash
python3 -m venv ~/.venvs/events2rag
~/.venvs/events2rag/bin/pip install https://github.com/romkey/pdxhackerspace-events2rag/releases/download/vX.Y.Z/events2rag-X.Y.Z-py3-none-any.whl
~/.venvs/events2rag/bin/events2rag --version
```

Set its environment variables yourself (see
[docs/configuration.md](docs/configuration.md)); it doesn't read `.env`.

## Development

With Docker, nothing else to install:

```bash
docker compose up --build                                  # run from the working tree, with Qdrant
docker compose -f docker-compose.test.yaml run --rm test   # tests with coverage
docker compose -f docker-compose.lint.yaml run --rm lint   # every lint check, as CI runs them
```

With [uv](https://docs.astral.sh/uv/), which installs the exact versions
CI tests (`uv.lock`):

```bash
uv sync
uv run pytest
uv run mypy
uv run events2rag
```

With pip (25.1 or newer, for `--group`). pip doesn't read `uv.lock`, so it
may pick newer dependency versions than CI tested:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e . --group dev
pytest
mypy
events2rag
```

Either way, `uvx pre-commit install` (or `pip install pre-commit &&
pre-commit install`) runs the lint checks on every commit. The tests fail
below 95% coverage.

Releases: run the **Release** workflow from the Actions tab, then merge the
release PR it opens; see [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
