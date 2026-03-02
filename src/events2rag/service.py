from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from events2rag.config import Settings
from events2rag.embedder import Embedder
from events2rag.ics_feed import fetch_ics, parse_ics_occurrences
from events2rag.json_feed import fetch_json, parse_event_occurrences
from events2rag.models import EventOccurrence, EventSummary
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
        occurrence_vectors = self._embed_batches(
            [occurrence.embedding_text() for occurrence in deduped]
        )
        upserted_occurrences = self._store.upsert_occurrences(
            deduped, occurrence_vectors
        )

        summaries = _build_event_summaries(deduped)
        summary_vectors = self._embed_batches(
            [summary.embedding_text() for summary in summaries]
        )
        upserted_summaries = self._store.upsert_event_summaries(
            summaries, summary_vectors
        )

        logger.info(
            "Upserted %s occurrences and %s event summaries into Qdrant.",
            upserted_occurrences,
            upserted_summaries,
        )
        return upserted_occurrences + upserted_summaries

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


def _build_event_summaries(occurrences: list[EventOccurrence]) -> list[EventSummary]:
    grouped: dict[str, list[EventOccurrence]] = {}
    for occurrence in occurrences:
        grouped.setdefault(occurrence.event_id, []).append(occurrence)

    now = datetime.now(tz=UTC)
    summaries: list[EventSummary] = []
    for event_id, event_occurrences in grouped.items():
        first = event_occurrences[0]
        next_start = _next_occurrence_start(event_occurrences, now)
        locations = sorted(
            {
                occurrence.location
                for occurrence in event_occurrences
                if occurrence.location
            }
        )
        tags = sorted(
            {
                tag
                for occurrence in event_occurrences
                for tag in occurrence.tags
                if tag and tag.strip()
            }
        )
        last_modified_candidates = [
            occurrence.last_modified
            for occurrence in event_occurrences
            if occurrence.last_modified is not None
        ]
        summaries.append(
            EventSummary(
                event_id=event_id,
                title=first.title,
                description=first.description,
                next_start_time=next_start,
                locations=locations,
                tags=tags,
                source_url=first.source_url,
                source_type=first.source_type,
                occurrence_count=len(event_occurrences),
                last_modified=max(last_modified_candidates)
                if last_modified_candidates
                else None,
            )
        )
    return summaries


def _next_occurrence_start(
    occurrences: list[EventOccurrence], now: datetime
) -> datetime | None:
    future_times = sorted(
        occurrence.start_time
        for occurrence in occurrences
        if occurrence.start_time >= now
    )
    if future_times:
        return future_times[0]
    all_times = sorted(occurrence.start_time for occurrence in occurrences)
    return all_times[-1] if all_times else None

