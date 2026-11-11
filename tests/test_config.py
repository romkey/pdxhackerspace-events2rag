from __future__ import annotations

import pytest

from events2rag.config import Settings

ENV_VARS = [
    "EVENTS_JSON_URL",
    "EVENTS_ICS_URL",
    "POLL_INTERVAL_SECONDS",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION",
    "QDRANT_DISTANCE",
    "EMBEDDING_BACKEND",
    "EMBEDDING_MODEL_NAME",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_CONTEXT_LENGTH",
    "OLLAMA_URL",
    "OLLAMA_API_KEY",
    "ICS_LOOKBACK_DAYS",
    "ICS_LOOKAHEAD_DAYS",
    "REQUEST_TIMEOUT_SECONDS",
    "LOG_LEVEL",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_from_env_uses_defaults(clean_env) -> None:
    assert Settings.from_env() == Settings()


def test_from_env_reads_all_variables(clean_env) -> None:
    values = {
        "EVENTS_JSON_URL": "https://example.com/events.json",
        "EVENTS_ICS_URL": "https://example.com/events.ics",
        "POLL_INTERVAL_SECONDS": "60",
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "qdrant-key",
        "QDRANT_COLLECTION": "my-events",
        "QDRANT_DISTANCE": "Dot",
        "EMBEDDING_BACKEND": "onnx",
        "EMBEDDING_MODEL_NAME": "some/model",
        "EMBEDDING_BATCH_SIZE": "8",
        "EMBEDDING_CONTEXT_LENGTH": "256",
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_API_KEY": "ollama-key",
        "ICS_LOOKBACK_DAYS": "7",
        "ICS_LOOKAHEAD_DAYS": "90",
        "REQUEST_TIMEOUT_SECONDS": "5",
        "LOG_LEVEL": "DEBUG",
    }
    for name, value in values.items():
        clean_env.setenv(name, value)

    assert Settings.from_env() == Settings(
        events_json_url="https://example.com/events.json",
        events_ics_url="https://example.com/events.ics",
        poll_interval_seconds=60,
        qdrant_url="http://localhost:6333",
        qdrant_api_key="qdrant-key",
        qdrant_collection="my-events",
        qdrant_distance="Dot",
        embedding_backend="onnx",
        embedding_model_name="some/model",
        embedding_batch_size=8,
        embedding_context_length=256,
        ollama_url="http://localhost:11434",
        ollama_api_key="ollama-key",
        ics_lookback_days=7,
        ics_lookahead_days=90,
        request_timeout_seconds=5,
        log_level="DEBUG",
    )


def test_from_env_treats_empty_optional_values_as_none(clean_env) -> None:
    clean_env.setenv("EVENTS_ICS_URL", "")
    clean_env.setenv("QDRANT_API_KEY", "")
    clean_env.setenv("OLLAMA_API_KEY", "")

    settings = Settings.from_env()

    assert settings.events_ics_url is None
    assert settings.qdrant_api_key is None
    assert settings.ollama_api_key is None


def test_from_env_rejects_non_integer(clean_env) -> None:
    clean_env.setenv("POLL_INTERVAL_SECONDS", "soon")
    with pytest.raises(ValueError):
        Settings.from_env()
