from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime

from events2rag.config import Settings
from events2rag.models import EventOccurrence, EventSummary
from events2rag.service import (
    IngestionService,
    _dedupe_by_time_and_title,
)


@dataclass
class FakeEmbedder:
    dimension: int = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class FakeStore:
    def __init__(self) -> None:
        self.collection_size: int | None = None
        self.upserted_occurrences: list[EventOccurrence] = []
        self.occurrence_vectors: list[list[float]] = []
        self.upserted_summaries: list[EventSummary] = []
        self.summary_vectors: list[list[float]] = []

    def ensure_collection(self, vector_size: int) -> None:
        self.collection_size = vector_size

    def get_existing_metadata(
        self, point_ids: list[str]
    ) -> dict[str, dict]:
        return {}

    def upsert_occurrences(
        self,
        occurrences: list[EventOccurrence],
        vectors: list[list[float]],
    ) -> int:
        self.upserted_occurrences = list(occurrences)
        self.occurrence_vectors = list(vectors)
        return len(occurrences)

    def upsert_event_summaries(
        self,
        summaries: list[EventSummary],
        vectors: list[list[float]],
    ) -> int:
        self.upserted_summaries = list(summaries)
        self.summary_vectors = list(vectors)
        return len(summaries)


def test_run_cycle_dedupes_and_enriches(mocker) -> None:
    settings = Settings(
        events_json_url="https://example.com/events.json"
    )
    start = datetime(2026, 4, 1, 18, 0, tzinfo=UTC)
    end = datetime(2026, 4, 1, 20, 0, tzinfo=UTC)
    duplicate = EventOccurrence(
        occurrence_id="abc",
        event_id="event-1",
        title="Title",
        description="Description",
        start_time=start,
        end_time=end,
        location="Room A",
        source_url=None,
    )
    mocker.patch(
        "events2rag.service.fetch_json",
        return_value={"events": []},
    )
    mocker.patch(
        "events2rag.service.parse_event_occurrences",
        return_value=[duplicate, duplicate],
    )

    store = FakeStore()
    service = IngestionService(
        settings=settings, store=store, embedder=FakeEmbedder()
    )

    upserted = service.run_cycle()
    assert upserted == 2
    assert len(store.upserted_occurrences) == 1

    occ = store.upserted_occurrences[0]
    assert occ.temporal_status in ("past", "current", "future")
    assert occ.duration == "2 hours"
    assert "Status" not in occ.embedding_text()

    assert len(store.upserted_summaries) == 1
    summary = store.upserted_summaries[0]
    assert summary.event_id == "event-1"
    assert summary.frequency == "one-time"
    assert "Availability" not in summary.embedding_text()
    assert store.collection_size == 3


def test_run_forever_sleeps_between_cycles(mocker) -> None:
    settings = Settings(
        events_json_url="https://example.com/events.json",
        poll_interval_seconds=123,
    )
    mocker.patch(
        "events2rag.service.fetch_json",
        return_value={"events": []},
    )
    mocker.patch(
        "events2rag.service.parse_event_occurrences",
        return_value=[],
    )

    store = FakeStore()
    service = IngestionService(
        settings=settings, store=store, embedder=FakeEmbedder()
    )

    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise KeyboardInterrupt()

    with suppress(KeyboardInterrupt):
        service.run_forever(sleep_fn=fake_sleep)
    assert sleep_calls == [123]


def test_run_forever_survives_transient_errors(mocker) -> None:
    settings = Settings(
        events_json_url="https://example.com/events.json"
    )
    call_count = 0

    def flaky_fetch(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("network down")
        return {"events": []}

    mocker.patch(
        "events2rag.service.fetch_json", side_effect=flaky_fetch
    )
    mocker.patch(
        "events2rag.service.parse_event_occurrences", return_value=[]
    )

    store = FakeStore()
    service = IngestionService(
        settings=settings, store=store, embedder=FakeEmbedder()
    )

    sleep_count = 0

    def counted_sleep(seconds: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count >= 2:
            raise KeyboardInterrupt()

    with suppress(KeyboardInterrupt):
        service.run_forever(sleep_fn=counted_sleep)

    assert call_count == 2
    assert sleep_count == 2


def test_cross_source_dedup_prefers_json() -> None:
    start = datetime(2026, 3, 1, 18, 0, tzinfo=UTC)
    json_occ = EventOccurrence(
        occurrence_id="json-1:2026-03-01",
        event_id="json-1",
        title="Soldering Night",
        description="JSON description",
        start_time=start,
        end_time=None,
        location="Room A",
        source_url="https://example.com/soldering",
        source_type="json",
    )
    ics_occ = EventOccurrence(
        occurrence_id="uid-1:2026-03-01",
        event_id="uid-1",
        title="Soldering Night",
        description="ICS description",
        start_time=start,
        end_time=None,
        location=None,
        source_url=None,
        source_type="ics",
    )
    result = _dedupe_by_time_and_title([ics_occ, json_occ])
    assert len(result) == 1
    assert result[0].source_type == "json"
    assert result[0].description == "JSON description"


def test_summary_uses_canonical_occurrence(mocker) -> None:
    settings = Settings(
        events_json_url="https://example.com/events.json"
    )
    old_occ = EventOccurrence(
        occurrence_id="e1:occ1",
        event_id="e1",
        title="Old Title",
        description="Old desc",
        start_time=datetime(2026, 3, 1, tzinfo=UTC),
        end_time=datetime(2026, 3, 1, 2, 0, tzinfo=UTC),
        location="Room A",
        source_url=None,
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
    )
    new_occ = EventOccurrence(
        occurrence_id="e1:occ2",
        event_id="e1",
        title="Updated Title",
        description="Updated desc",
        start_time=datetime(2026, 3, 8, tzinfo=UTC),
        end_time=datetime(2026, 3, 8, 2, 0, tzinfo=UTC),
        location="Room A",
        source_url="https://example.com",
        last_modified=datetime(2026, 2, 15, tzinfo=UTC),
    )
    mocker.patch(
        "events2rag.service.fetch_json",
        return_value={"events": []},
    )
    mocker.patch(
        "events2rag.service.parse_event_occurrences",
        return_value=[old_occ, new_occ],
    )

    store = FakeStore()
    service = IngestionService(
        settings=settings, store=store, embedder=FakeEmbedder()
    )
    service.run_cycle()

    summary = store.upserted_summaries[0]
    assert summary.title == "Updated Title"
    assert summary.description == "Updated desc"
    assert summary.source_url == "https://example.com"
