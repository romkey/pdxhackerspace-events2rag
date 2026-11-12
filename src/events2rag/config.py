from __future__ import annotations

import os
from dataclasses import dataclass

from events2rag.text_utils import normalize_prefix


@dataclass(frozen=True)
class Settings:
    events_json_url: str = "https://events.pdxhackerspace/events.json"
    events_ics_url: str | None = None
    poll_interval_seconds: int = 3600
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "events"
    qdrant_distance: str = "Cosine"
    embedding_backend: str = "openai"
    embedding_model_name: str = "nomic-ai/modernbert-embed-base"
    embedding_batch_size: int = 64
    embedding_context_length: int = 512
    embedding_query_prefix: str = "search_query: "
    embedding_document_prefix: str = "search_document: "
    openai_base_url: str = "http://localhost:8080/v1"
    openai_api_key: str | None = None
    ics_lookback_days: int = 30
    ics_lookahead_days: int = 365
    request_timeout_seconds: int = 30
    log_level: str = "INFO"

    @property
    def embedding_signature(self) -> str:
        """Identifies how stored document vectors were produced.

        Points whose stored signature differs are re-embedded, so
        changing the model or document prefix refreshes every vector.
        """
        return (
            f"{self.embedding_backend}|{self.embedding_model_name}|"
            f"{normalize_prefix(self.embedding_document_prefix)}"
        )

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
            embedding_query_prefix=os.getenv(
                "EMBEDDING_QUERY_PREFIX", cls.embedding_query_prefix
            ),
            embedding_document_prefix=os.getenv(
                "EMBEDDING_DOCUMENT_PREFIX", cls.embedding_document_prefix
            ),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL", cls.openai_base_url
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
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
