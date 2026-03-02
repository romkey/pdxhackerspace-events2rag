from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime

from events2rag.config import Settings
from events2rag.models import EventOccurrence, EventSummary
from events2rag.service import IngestionService


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

    def upsert_occurrences(
        self, occurrences: list[EventOccurrence], vectors: list[list[float]]
    ) -> int:
        self.upserted_occurrences = list(occurrences)
        self.occurrence_vectors = list(vectors)
        return len(occurrences)

    def upsert_event_summaries(
        self, summaries: list[EventSummary], vectors: list[list[float]]
    ) -> int:
        self.upserted_summaries = list(summaries)
        self.summary_vectors = list(vectors)
        return len(summaries)


def test_run_cycle_dedupes_occurrences_and_creates_summary(mocker) -> None:
    settings = Settings(events_json_url="https://example.com/events.json")
    now = datetime(2026, 3, 1, tzinfo=UTC)
    duplicate = EventOccurrence(
        occurrence_id="abc",
        event_id="event-1",
        title="Title",
        description="Description",
        start_time=now,
        end_time=None,
        location=None,
        source_url=None,
    )
    mocker.patch("events2rag.service.fetch_json", return_value={"events": []})
    mocker.patch(
        "events2rag.service.parse_event_occurrences",
        return_value=[duplicate, duplicate],
    )

    store = FakeStore()
    service = IngestionService(settings=settings, store=store, embedder=FakeEmbedder())

    upserted = service.run_cycle()
    assert upserted == 2
    assert len(store.upserted_occurrences) == 1
    assert len(store.upserted_summaries) == 1
    assert store.upserted_summaries[0].event_id == "event-1"
    assert store.collection_size == 3


def test_run_forever_sleeps_between_cycles(mocker) -> None:
    settings = Settings(
        events_json_url="https://example.com/events.json", poll_interval_seconds=123
    )
    mocker.patch("events2rag.service.fetch_json", return_value={"events": []})
    mocker.patch("events2rag.service.parse_event_occurrences", return_value=[])

    store = FakeStore()
    service = IngestionService(settings=settings, store=store, embedder=FakeEmbedder())

    sleep_calls: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise KeyboardInterrupt()

    with suppress(KeyboardInterrupt):
        service.run_forever(sleep_fn=fake_sleep)
    assert sleep_calls == [123]

