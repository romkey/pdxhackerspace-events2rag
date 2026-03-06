from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from events2rag.config import Settings
from events2rag.embedder import Embedder
from events2rag.ics_feed import fetch_ics, parse_ics_occurrences
from events2rag.json_feed import fetch_json, parse_event_occurrences
from events2rag.models import EventOccurrence, EventSummary
from events2rag.qdrant_store import QdrantStore, _to_point_id
from events2rag.text_utils import (
    estimate_frequency,
    human_duration,
    temporal_status,
    truncate_for_embedding,
)

logger = logging.getLogger(__name__)

_MIN_DATETIME = datetime.min.replace(tzinfo=UTC)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        store: QdrantStore,
        embedder: Embedder,
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
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                raise
            except Exception:
                logger.exception("Ingestion cycle failed; will retry.")
            sleep_fn(self._settings.poll_interval_seconds)

    def run_cycle(self) -> int:
        occurrences = self._collect_occurrences()
        if not occurrences:
            logger.info("No events discovered; nothing to upsert.")
            return 0

        now = datetime.now(tz=UTC)
        deduped = _dedupe_occurrences(occurrences)
        cross_deduped = _dedupe_by_time_and_title(deduped)
        enriched = _enrich_occurrences(cross_deduped, now)

        to_embed = self._filter_changed(enriched)
        if to_embed:
            vectors = self._embed_batches(
                [occ.embedding_text() for occ in to_embed]
            )
            upserted_occurrences = self._store.upsert_occurrences(
                to_embed, vectors
            )
        else:
            upserted_occurrences = 0

        summaries = _build_event_summaries(enriched, now)
        summary_vectors = self._embed_batches(
            [s.embedding_text() for s in summaries]
        )
        upserted_summaries = self._store.upsert_event_summaries(
            summaries, summary_vectors
        )

        logger.info(
            "Upserted %s occurrences (%s skipped unchanged) "
            "and %s event summaries.",
            upserted_occurrences,
            len(enriched) - len(to_embed),
            upserted_summaries,
        )
        return upserted_occurrences + upserted_summaries

    def _filter_changed(
        self, occurrences: list[EventOccurrence]
    ) -> list[EventOccurrence]:
        """Skip re-embedding occurrences whose last_modified hasn't changed."""
        point_ids = [
            _to_point_id(occ.occurrence_id) for occ in occurrences
        ]
        existing = self._store.get_existing_metadata(point_ids)

        changed: list[EventOccurrence] = []
        for occ, pid in zip(occurrences, point_ids, strict=True):
            meta = existing.get(pid)
            if meta is None:
                changed.append(occ)
                continue
            if occ.last_modified is None:
                continue
            ingested_at_str = meta.get("ingested_at")
            if ingested_at_str is None:
                changed.append(occ)
                continue
            ingested_at = datetime.fromisoformat(ingested_at_str)
            if occ.last_modified > ingested_at:
                changed.append(occ)
        return changed

    def _collect_occurrences(self) -> list[EventOccurrence]:
        payload = fetch_json(
            self._settings.events_json_url,
            self._settings.request_timeout_seconds,
        )
        occurrences = parse_event_occurrences(payload)
        logger.info(
            "Parsed %s occurrences from JSON feed.",
            len(occurrences),
        )
        if self._settings.events_ics_url:
            ics_content = fetch_ics(
                self._settings.events_ics_url,
                self._settings.request_timeout_seconds,
            )
            ics_occurrences = parse_ics_occurrences(
                ics_content,
                lookback_days=self._settings.ics_lookback_days,
                lookahead_days=self._settings.ics_lookahead_days,
                feed_url=self._settings.events_ics_url,
            )
            occurrences.extend(ics_occurrences)
            logger.info(
                "Parsed %s occurrences from ICS feed.",
                len(ics_occurrences),
            )
        return occurrences

    def _embed_batches(self, texts: list[str]) -> list[list[float]]:
        max_tokens = self._settings.embedding_context_length
        safe_texts = [
            truncate_for_embedding(t, max_tokens, logger) for t in texts
        ]
        vectors: list[list[float]] = []
        batch_size = max(1, self._settings.embedding_batch_size)
        for index in range(0, len(safe_texts), batch_size):
            batch = safe_texts[index : index + batch_size]
            vectors.extend(self._embedder.embed(batch))
        return vectors


def _dedupe_occurrences(
    occurrences: list[EventOccurrence],
) -> list[EventOccurrence]:
    deduped_by_id: dict[str, EventOccurrence] = {}
    for occurrence in occurrences:
        deduped_by_id[occurrence.occurrence_id] = occurrence
    return list(deduped_by_id.values())


def _dedupe_by_time_and_title(
    occurrences: list[EventOccurrence],
) -> list[EventOccurrence]:
    """Secondary dedup across sources by (normalized title, start_time).

    When both a JSON and ICS occurrence match, the JSON version is kept
    because it tends to have richer metadata.
    """
    seen: dict[tuple[str, str], EventOccurrence] = {}
    for occ in occurrences:
        key = (
            occ.title.strip().lower(),
            occ.start_time.isoformat(),
        )
        existing = seen.get(key)
        if existing is None or occ.source_type == "json":
            seen[key] = occ
    return list(seen.values())


def _enrich_occurrences(
    occurrences: list[EventOccurrence], now: datetime
) -> list[EventOccurrence]:
    enriched: list[EventOccurrence] = []
    for occ in occurrences:
        status = temporal_status(occ.start_time, occ.end_time, now)
        duration = human_duration(occ.start_time, occ.end_time)
        enriched.append(
            replace(occ, temporal_status=status, duration=duration)
        )
    return enriched


def _build_event_summaries(
    occurrences: list[EventOccurrence], now: datetime
) -> list[EventSummary]:
    grouped: dict[str, list[EventOccurrence]] = {}
    for occurrence in occurrences:
        grouped.setdefault(occurrence.event_id, []).append(occurrence)

    summaries: list[EventSummary] = []
    for event_id, event_occurrences in grouped.items():
        canonical = max(
            event_occurrences,
            key=lambda o: o.last_modified or _MIN_DATETIME,
        )
        next_start = _next_occurrence_start(event_occurrences, now)
        has_future = any(
            occ.temporal_status in ("future", "current")
            for occ in event_occurrences
        )
        locations = sorted(
            {
                occ.location
                for occ in event_occurrences
                if occ.location
            }
        )
        tags = sorted(
            {
                tag
                for occ in event_occurrences
                for tag in occ.tags
                if tag and tag.strip()
            }
        )
        last_modified_candidates = [
            occ.last_modified
            for occ in event_occurrences
            if occ.last_modified is not None
        ]
        freq = estimate_frequency(
            [occ.start_time for occ in event_occurrences]
        )
        summaries.append(
            EventSummary(
                event_id=event_id,
                title=canonical.title,
                description=canonical.description,
                next_start_time=next_start,
                locations=locations,
                tags=tags,
                source_url=canonical.source_url,
                source_type=canonical.source_type,
                occurrence_count=len(event_occurrences),
                last_modified=max(last_modified_candidates)
                if last_modified_candidates
                else None,
                frequency=freq,
                has_future_occurrences=has_future,
            )
        )
    return summaries


def _next_occurrence_start(
    occurrences: list[EventOccurrence], now: datetime
) -> datetime | None:
    future_times = sorted(
        occ.start_time
        for occ in occurrences
        if occ.start_time >= now
    )
    if future_times:
        return future_times[0]
    all_times = sorted(occ.start_time for occ in occurrences)
    return all_times[-1] if all_times else None
