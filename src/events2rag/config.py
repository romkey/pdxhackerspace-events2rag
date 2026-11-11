"""Settings, read once at startup from environment variables.

Every variable is documented in docs/configuration.md and listed, with a
safe example value, in .env.example.
"""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

EMBEDDING_BACKENDS = ("ollama", "onnx", "sentence-transformers")
QDRANT_DISTANCES = ("Cosine", "Dot", "Euclid", "Manhattan")


class ConfigError(ValueError):
    """An environment variable is set to something unusable."""


def _int(name: str, text: str | None, default: int, minimum: int) -> int:
    if text is None or text == "":
        return default
    try:
        value = int(text)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, not {text!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, not {value}")
    return value


def _choice(name: str, text: str | None, default: str, allowed: tuple[str, ...]) -> str:
    if text is None or text == "":
        return default
    if text not in allowed:
        raise ConfigError(f"{name} must be one of {', '.join(allowed)}, not {text!r}")
    return text


def _url(name: str, text: str | None, default: str) -> str:
    if text is None or text == "":
        return default
    if not text.startswith(("http://", "https://")):
        raise ConfigError(f"{name} must be an http:// or https:// URL, not {text!r}")
    return text


def _optional_url(name: str, text: str | None) -> str | None:
    return _url(name, text, "") or None


@dataclass(frozen=True)
class Settings:
    events_json_url: str = "https://events.pdxhackerspace/events.json"
    events_ics_url: str | None = None
    poll_interval_seconds: int = 3600
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "events"
    qdrant_distance: str = "Cosine"
    embedding_backend: str = "ollama"
    embedding_model_name: str = "qllama/bge-small-en-v1.5"
    embedding_batch_size: int = 64
    embedding_context_length: int = 512
    ollama_url: str = "http://ollama:11434"
    ollama_api_key: str | None = None
    ics_lookback_days: int = 30
    ics_lookahead_days: int = 365
    request_timeout_seconds: int = 30
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        log_level = (env.get("LOG_LEVEL") or cls.log_level).upper()
        if log_level not in logging.getLevelNamesMapping():
            raise ConfigError(f"LOG_LEVEL {log_level!r} is not a logging level")

        return cls(
            events_json_url=_url(
                "EVENTS_JSON_URL", env.get("EVENTS_JSON_URL"), cls.events_json_url
            ),
            events_ics_url=_optional_url("EVENTS_ICS_URL", env.get("EVENTS_ICS_URL")),
            poll_interval_seconds=_int(
                "POLL_INTERVAL_SECONDS",
                env.get("POLL_INTERVAL_SECONDS"),
                cls.poll_interval_seconds,
                minimum=1,
            ),
            qdrant_url=_url("QDRANT_URL", env.get("QDRANT_URL"), cls.qdrant_url),
            qdrant_api_key=env.get("QDRANT_API_KEY") or None,
            qdrant_collection=env.get("QDRANT_COLLECTION") or cls.qdrant_collection,
            qdrant_distance=_choice(
                "QDRANT_DISTANCE",
                env.get("QDRANT_DISTANCE"),
                cls.qdrant_distance,
                QDRANT_DISTANCES,
            ),
            embedding_backend=_choice(
                "EMBEDDING_BACKEND",
                env.get("EMBEDDING_BACKEND"),
                cls.embedding_backend,
                EMBEDDING_BACKENDS,
            ),
            embedding_model_name=(
                env.get("EMBEDDING_MODEL_NAME") or cls.embedding_model_name
            ),
            embedding_batch_size=_int(
                "EMBEDDING_BATCH_SIZE",
                env.get("EMBEDDING_BATCH_SIZE"),
                cls.embedding_batch_size,
                minimum=1,
            ),
            embedding_context_length=_int(
                "EMBEDDING_CONTEXT_LENGTH",
                env.get("EMBEDDING_CONTEXT_LENGTH"),
                cls.embedding_context_length,
                minimum=1,
            ),
            ollama_url=_url("OLLAMA_URL", env.get("OLLAMA_URL"), cls.ollama_url),
            ollama_api_key=env.get("OLLAMA_API_KEY") or None,
            ics_lookback_days=_int(
                "ICS_LOOKBACK_DAYS",
                env.get("ICS_LOOKBACK_DAYS"),
                cls.ics_lookback_days,
                minimum=0,
            ),
            ics_lookahead_days=_int(
                "ICS_LOOKAHEAD_DAYS",
                env.get("ICS_LOOKAHEAD_DAYS"),
                cls.ics_lookahead_days,
                minimum=0,
            ),
            request_timeout_seconds=_int(
                "REQUEST_TIMEOUT_SECONDS",
                env.get("REQUEST_TIMEOUT_SECONDS"),
                cls.request_timeout_seconds,
                minimum=1,
            ),
            log_level=log_level,
        )
