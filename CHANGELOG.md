# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Releases up to 0.5.2 predate this
file.

## [Unreleased]

### Fixed

- Releases publish again: the release workflow no longer fails pushing
  the image when it can't save the build cache, which it can't on the run
  for a merged release PR. v0.6.0 was tagged but its image and GitHub
  Release weren't published because of this.

## [0.6.0] - 2026-11-14

### Added

- **Embedding prompt prefixes**: `EMBEDDING_QUERY_PREFIX` and
  `EMBEDDING_DOCUMENT_PREFIX`, for models trained with task prefixes
  (ModernBERT and nomic, e5, mxbai, Qwen3-Embedding). Events are embedded
  as documents; the prefix counts toward `EMBEDDING_CONTEXT_LENGTH`.
  `none` turns a prefix off. docs/configuration.md lists the prefixes for
  common models, and the README shows how searchers add the query prefix.
- Each stored occurrence records the backend, model and document prefix
  it was embedded with; changing any of them re-embeds it on the next
  cycle.

- **The `openai` embedding backend** replaces `ollama` and is the default.
  It calls `POST {OPENAI_BASE_URL}/embeddings`, so it works with
  api.openai.com and any OpenAI-compatible server (Ollama's `/v1`, vLLM,
  LM Studio). `OPENAI_BASE_URL` and `OPENAI_API_KEY` replace `OLLAMA_URL`
  and `OLLAMA_API_KEY`; the default model is `text-embedding-3-small`.
  Switching models changes the vector size: use a new `QDRANT_COLLECTION`.
- `events2rag --version`, and a clean shutdown on SIGTERM (`docker stop`):
  the current cycle finishes and the process exits instead of being killed.
- Every setting is validated at startup; a bad value exits with status 2
  and a message naming the variable.
- Embeddings API responses are checked: one vector of numbers per input,
  returned in input order.
- Tests for every module (99% coverage, enforced at 95%), strict mypy,
  and pre-commit checks for lint, formatting, YAML, workflows, the
  Dockerfile, shell scripts and secrets.
- Release workflow: run it from the Actions tab to open a release PR;
  merging it tags the release, pushes a multi-arch image with provenance
  and SBOM, and creates a GitHub Release with the wheel.
- Renovate configuration, `docs/`, `AGENTS.md`, `SECURITY.md` and an MIT
  `LICENSE`.

### Changed

- **The default model is ModernBERT Embed**
  (`nomic-ai/modernbert-embed-base`, 768 dimensions) served by a local
  OpenAI-compatible server: `OPENAI_BASE_URL` defaults to
  `http://localhost:8080/v1` and the prefixes to `search_query:` /
  `search_document:`. For OpenAI, set `OPENAI_BASE_URL`, the model, and
  both prefixes to `none`. Occurrences stored before this release are
  re-embedded once.

- **Requires Python 3.14** (was 3.11); the image runs Python 3.14 on
  Debian trixie. Python 3.15 waits for onnxruntime wheels.
- **Invalid settings now stop the app** where they used to be accepted:
  an unknown `QDRANT_DISTANCE` (was silently Cosine), a non-URL
  `QDRANT_URL`/`OPENAI_BASE_URL`/feed URL, a zero or negative interval, batch
  size, context length or timeout, and an unknown `LOG_LEVEL`. Empty
  values now mean "use the default".
- The production image runs as a non-root user (uid 10001), and the
  production compose file runs it read-only with no capabilities.
- Compose files follow the new layout: `docker-compose.yaml` (development,
  with Qdrant), `.test.yaml`, `.lint.yaml` and `.prod.yaml`. They replace
  `docker-compose.yml` and `docker-compose.dev.yml`; the service is now
  `app` (was `ingester`).
- Image tags: before 1.0.0, releases no longer move the major tag (`0`);
  use `X.Y.Z`, `X.Y` or `latest`.
- Development uses uv with a committed `uv.lock`; dev tools are in the
  `dev` dependency group (`pip install -e . --group dev`) instead of the
  `dev` extra.
- Dependencies raised to current releases: qdrant-client 1.20, icalendar 7,
  recurring-ical-events 3.8, requests 2.34, beautifulsoup4 4.15; the
  `onnx` and `torch` extras to numpy 2.5, onnxruntime 1.31, tokenizers
  0.23, huggingface-hub 2.4 and sentence-transformers 6.3. Compose uses
  Qdrant 1.20.
- Logs go to stdout.
