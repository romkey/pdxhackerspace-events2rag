import pytest

from events2rag.config import ConfigError, Settings


def test_from_env_uses_defaults() -> None:
    assert Settings.from_env({}) == Settings()


def test_from_env_reads_all_variables() -> None:
    env = {
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
        "EMBEDDING_QUERY_PREFIX": "query:",
        "EMBEDDING_DOCUMENT_PREFIX": "passage: ",
        "OPENAI_BASE_URL": "http://localhost:11434/v1",
        "OPENAI_API_KEY": "openai-key",
        "ICS_LOOKBACK_DAYS": "0",
        "ICS_LOOKAHEAD_DAYS": "90",
        "REQUEST_TIMEOUT_SECONDS": "5",
        "LOG_LEVEL": "debug",
    }

    assert Settings.from_env(env) == Settings(
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
        embedding_query_prefix="query: ",
        embedding_document_prefix="passage: ",
        openai_base_url="http://localhost:11434/v1",
        openai_api_key="openai-key",
        ics_lookback_days=0,
        ics_lookahead_days=90,
        request_timeout_seconds=5,
        log_level="DEBUG",
    )


def test_from_env_treats_empty_values_as_unset() -> None:
    env = {
        "EVENTS_JSON_URL": "",
        "EVENTS_ICS_URL": "",
        "POLL_INTERVAL_SECONDS": "",
        "QDRANT_API_KEY": "",
        "QDRANT_DISTANCE": "",
        "OPENAI_API_KEY": "",
        "EMBEDDING_QUERY_PREFIX": "",
        "EMBEDDING_DOCUMENT_PREFIX": "",
        "LOG_LEVEL": "",
    }
    assert Settings.from_env(env) == Settings()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("POLL_INTERVAL_SECONDS", "soon", "POLL_INTERVAL_SECONDS must be a whole"),
        ("POLL_INTERVAL_SECONDS", "0", "POLL_INTERVAL_SECONDS must be at least 1"),
        ("ICS_LOOKBACK_DAYS", "-1", "ICS_LOOKBACK_DAYS must be at least 0"),
        ("EMBEDDING_BACKEND", "magic", "EMBEDDING_BACKEND must be one of"),
        ("QDRANT_DISTANCE", "cosine", "QDRANT_DISTANCE must be one of"),
        ("QDRANT_URL", "qdrant:6333", "QDRANT_URL must be an http"),
        ("OPENAI_BASE_URL", "api.openai.com", "OPENAI_BASE_URL must be an http"),
        ("EVENTS_ICS_URL", "ftp://x", "EVENTS_ICS_URL must be an http"),
        ("LOG_LEVEL", "chatty", "LOG_LEVEL 'CHATTY' is not a logging level"),
    ],
)
def test_from_env_rejects_bad_values(name: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        Settings.from_env({name: value})


@pytest.mark.parametrize(
    ("value", "prefix"),
    [
        ("none", ""),
        ("None", ""),
        ("query:", "query: "),
        ("query: ", "query: "),
        (
            "Instruct: Find matching events\\nQuery:",
            "Instruct: Find matching events\nQuery: ",
        ),
    ],
)
def test_from_env_reads_prefixes(value: str, prefix: str) -> None:
    settings = Settings.from_env(
        {"EMBEDDING_QUERY_PREFIX": value, "EMBEDDING_DOCUMENT_PREFIX": value}
    )
    assert settings.embedding_query_prefix == prefix
    assert settings.embedding_document_prefix == prefix


def test_embedding_signature_tracks_how_documents_are_embedded() -> None:
    base = Settings()
    assert base.embedding_signature == (
        "openai|nomic-ai/modernbert-embed-base|search_document: "
    )
    for changed in (
        Settings(embedding_backend="onnx"),
        Settings(embedding_model_name="other"),
        Settings(embedding_document_prefix=""),
    ):
        assert changed.embedding_signature != base.embedding_signature
    assert (
        Settings(embedding_query_prefix="").embedding_signature
        == base.embedding_signature
    )
