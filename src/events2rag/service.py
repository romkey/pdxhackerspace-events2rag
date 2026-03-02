from __future__ import annotations

import logging
import time
from collections.abc import Callable

from events2rag.config import Settings
from events2rag.embedder import Embedder
from events2rag.ics_feed import fetch_ics, parse_ics_occurrences
from events2rag.json_feed import fetch_json, parse_event_occurrences
from events2rag.models import EventOccurrence
from events2rag.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self, settings: Settings, store: QdrantStore, embedder: Embedder
    ) -> None:
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._store.ensure_collection(vector_size=self._embedder.dimension)

    def run_forever(
        self,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        while True:
            self.run_cycle()
            sleep_fn(self._settings.poll_interval_seconds)

    def run_cycle(self) -> int:
        occurrences = self._collect_occurrences()
        if not occurrences:
            logger.info("No events discovered; nothing to upsert.")
            return 0

        deduped = _dedupe_occurrences(occurrences)
        vectors = self._embed_batches(
            [occurrence.embedding_text() for occurrence in deduped]
        )
        upserted = self._store.upsert_occurrences(deduped, vectors)
        logger.info("Upserted %s event occurrences into Qdrant.", upserted)
        return upserted

    def _collect_occurrences(self) -> list[EventOccurrence]:
        payload = fetch_json(
            self._settings.events_json_url, self._settings.request_timeout_seconds
        )
        occurrences = parse_event_occurrences(payload)
        logger.info("Parsed %s occurrences from JSON feed.", len(occurrences))
        if self._settings.events_ics_url:
            ics_content = fetch_ics(
                self._settings.events_ics_url, self._settings.request_timeout_seconds
            )
            ics_occurrences = parse_ics_occurrences(
                ics_content,
                lookback_days=self._settings.ics_lookback_days,
                lookahead_days=self._settings.ics_lookahead_days,
            )
            occurrences.extend(ics_occurrences)
            logger.info("Parsed %s occurrences from ICS feed.", len(ics_occurrences))
        return occurrences

    def _embed_batches(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = max(1, self._settings.embedding_batch_size)
        for index in range(0, len(texts), batch_size):
            batch = texts[index : index + batch_size]
            vectors.extend(self._embedder.embed(batch))
        return vectors


def _dedupe_occurrences(occurrences: list[EventOccurrence]) -> list[EventOccurrence]:
    deduped_by_id: dict[str, EventOccurrence] = {}
    for occurrence in occurrences:
        deduped_by_id[occurrence.occurrence_id] = occurrence
    return list(deduped_by_id.values())

