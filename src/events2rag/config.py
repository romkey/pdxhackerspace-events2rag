from __future__ import annotations

import os
from dataclasses import dataclass


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
    ics_lookback_days: int = 30
    ics_lookahead_days: int = 365
    request_timeout_seconds: int = 30
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            events_json_url=os.getenv(
                "EVENTS_JSON_URL", cls.events_json_url
            ),
            events_ics_url=os.getenv("EVENTS_ICS_URL") or None,
            poll_interval_seconds=int(
                os.getenv(
                    "POLL_INTERVAL_SECONDS",
                    str(cls.poll_interval_seconds),
                )
            ),
            qdrant_url=os.getenv("QDRANT_URL", cls.qdrant_url),
            qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
            qdrant_collection=os.getenv(
                "QDRANT_COLLECTION", cls.qdrant_collection
            ),
            qdrant_distance=os.getenv(
                "QDRANT_DISTANCE", cls.qdrant_distance
            ),
            embedding_backend=os.getenv(
                "EMBEDDING_BACKEND", cls.embedding_backend
            ),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME", cls.embedding_model_name
            ),
            embedding_batch_size=int(
                os.getenv(
                    "EMBEDDING_BATCH_SIZE",
                    str(cls.embedding_batch_size),
                )
            ),
            embedding_context_length=int(
                os.getenv(
                    "EMBEDDING_CONTEXT_LENGTH",
                    str(cls.embedding_context_length),
                )
            ),
            ollama_url=os.getenv("OLLAMA_URL", cls.ollama_url),
            ics_lookback_days=int(
                os.getenv(
                    "ICS_LOOKBACK_DAYS", str(cls.ics_lookback_days)
                )
            ),
            ics_lookahead_days=int(
                os.getenv(
                    "ICS_LOOKAHEAD_DAYS", str(cls.ics_lookahead_days)
                )
            ),
            request_timeout_seconds=int(
                os.getenv(
                    "REQUEST_TIMEOUT_SECONDS",
                    str(cls.request_timeout_seconds),
                )
            ),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
        )
