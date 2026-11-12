import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

import pytest
from pytest_mock import MockerFixture

from events2rag.config import Settings
from events2rag.models import EventOccurrence, EventSummary
from events2rag.qdrant_store import _to_point_id
from events2rag.service import (
    IngestionService,
    _dedupe_by_time_and_title,
    _next_occurrence_start,
)


@dataclass
class FakeEmbedder:
    dimension: int = 3

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
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

    def get_existing_metadata(self, point_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {}

    def upsert_occurrences(
        self,
        occurrences: Sequence[EventOccurrence],
        vectors: Sequence[list[float]],
    ) -> int:
        self.upserted_occurrences = list(occurrences)
        self.occurrence_vectors = list(vectors)
        return len(occurrences)

    def upsert_event_summaries(
        self,
        summaries: Sequence[EventSummary],
        vectors: Sequence[list[float]],
    ) -> int:
        self.upserted_summaries = list(summaries)
        self.summary_vectors = list(vectors)
        return len(summaries)


def test_run_cycle_dedupes_and_enriches(mocker: MockerFixture) -> None:
    settings = Settings(events_json_url="https://example.com/events.json")
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
    service = IngestionService(settings=settings, store=store, embedder=FakeEmbedder())

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


class StopAfter(threading.Event):
    """A stop event that sets itself after `cycles` waits, recording them."""

    def __init__(self, cycles: int) -> None:
        super().__init__()
        self.cycles = cycles
        self.waits: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.waits.append(timeout)
        if len(self.waits) >= self.cycles:
            self.set()
        return self.is_set()


def test_run_forever_waits_between_cycles_until_stopped(
    mocker: MockerFixture,
) -> None:
    settings = Settings(poll_interval_seconds=123)
    service = IngestionService(
        settings=settings, store=FakeStore(), embedder=FakeEmbedder()
    )
    run_cycle = mocker.patch.object(service, "run_cycle", return_value=0)
    stop = StopAfter(cycles=2)

    service.run_forever(stop)

    assert run_cycle.call_count == 2
    assert stop.waits == [123, 123]


def test_run_forever_does_nothing_when_already_stopped(
    mocker: MockerFixture,
) -> None:
    service = IngestionService(
        settings=Settings(), store=FakeStore(), embedder=FakeEmbedder()
    )
    run_cycle = mocker.patch.object(service, "run_cycle")
    stop = threading.Event()
    stop.set()

    service.run_forever(stop)

    run_cycle.assert_not_called()


def test_run_forever_survives_transient_errors(mocker: MockerFixture) -> None:
    service = IngestionService(
        settings=Settings(), store=FakeStore(), embedder=FakeEmbedder()
    )
    run_cycle = mocker.patch.object(
        service, "run_cycle", side_effect=[ConnectionError("network down"), 0]
    )

    service.run_forever(StopAfter(cycles=2))

    assert run_cycle.call_count == 2


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


def test_summary_uses_canonical_occurrence(mocker: MockerFixture) -> None:
    settings = Settings(events_json_url="https://example.com/events.json")
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
    service = IngestionService(settings=settings, store=store, embedder=FakeEmbedder())
    service.run_cycle()

    summary = store.upserted_summaries[0]
    assert summary.title == "Updated Title"
    assert summary.description == "Updated desc"
    assert summary.source_url == "https://example.com"


_BASE_OCCURRENCE = EventOccurrence(
    occurrence_id="e1:1",
    event_id="e1",
    title="Title",
    description="Desc",
    start_time=datetime(2026, 3, 1, 18, tzinfo=UTC),
    end_time=None,
    location=None,
    source_url=None,
)


# Any: each override is a different EventOccurrence field type.
def _occ(**overrides: Any) -> EventOccurrence:
    return replace(_BASE_OCCURRENCE, **overrides)


class MetadataStore(FakeStore):
    def __init__(self, existing: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.existing = existing

    def get_existing_metadata(self, point_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {pid: self.existing[pid] for pid in point_ids if pid in self.existing}


# Any: each keyword is a different Settings field type.
def _service(store: FakeStore, **settings_kwargs: Any) -> IngestionService:
    settings = Settings(
        events_json_url="https://example.com/events.json", **settings_kwargs
    )
    return IngestionService(settings=settings, store=store, embedder=FakeEmbedder())


def test_run_cycle_returns_zero_when_no_events(mocker: MockerFixture) -> None:
    mocker.patch("events2rag.service.fetch_json", return_value=[])
    store = FakeStore()
    assert _service(store).run_cycle() == 0
    assert store.upserted_summaries == []


def test_run_forever_propagates_keyboard_interrupt(mocker: MockerFixture) -> None:
    service = _service(FakeStore())
    mocker.patch.object(service, "run_cycle", side_effect=KeyboardInterrupt)
    stop = StopAfter(cycles=1)

    with pytest.raises(KeyboardInterrupt):
        service.run_forever(stop)
    assert stop.waits == []


def test_filter_changed_skips_unchanged_occurrences(mocker: MockerFixture) -> None:
    ingested = datetime(2026, 2, 1, tzinfo=UTC)
    new = _occ(occurrence_id="new")
    no_last_modified = _occ(occurrence_id="no-lm")
    missing_ingested_at = _occ(
        occurrence_id="no-ingested",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
    )
    stale = _occ(
        occurrence_id="stale",
        last_modified=datetime(2026, 1, 1, tzinfo=UTC),
    )
    updated = _occ(
        occurrence_id="updated",
        last_modified=datetime(2026, 3, 1, tzinfo=UTC),
    )
    existing = {
        _to_point_id("no-lm"): {"ingested_at": ingested.isoformat()},
        _to_point_id("no-ingested"): {},
        _to_point_id("stale"): {"ingested_at": ingested.isoformat()},
        _to_point_id("updated"): {"ingested_at": ingested.isoformat()},
    }
    service = _service(MetadataStore(existing))

    changed = service._filter_changed(
        [new, no_last_modified, missing_ingested_at, stale, updated]
    )

    assert [o.occurrence_id for o in changed] == [
        "new",
        "no-ingested",
        "updated",
    ]


def test_run_cycle_with_nothing_changed_still_upserts_summaries(
    mocker: MockerFixture,
) -> None:
    occ = _occ(occurrence_id="same")
    mocker.patch("events2rag.service.fetch_json", return_value=[])
    mocker.patch("events2rag.service.parse_event_occurrences", return_value=[occ])
    store = MetadataStore(
        {_to_point_id("same"): {"ingested_at": "2026-01-01T00:00:00+00:00"}}
    )

    assert _service(store).run_cycle() == 1
    assert store.upserted_occurrences == []
    assert len(store.upserted_summaries) == 1


def test_collect_occurrences_includes_ics_feed(mocker: MockerFixture) -> None:
    json_occ = _occ(occurrence_id="json", title="From JSON")
    ics_occ = _occ(occurrence_id="ics", title="From ICS", source_type="ics")
    mocker.patch("events2rag.service.fetch_json", return_value=[])
    mocker.patch(
        "events2rag.service.parse_event_occurrences",
        return_value=[json_occ],
    )
    fetch_ics = mocker.patch("events2rag.service.fetch_ics", return_value="ICS")
    parse_ics = mocker.patch(
        "events2rag.service.parse_ics_occurrences",
        return_value=[ics_occ],
    )
    service = _service(
        FakeStore(),
        events_ics_url="https://example.com/feed.ics",
        ics_lookback_days=3,
        ics_lookahead_days=9,
        request_timeout_seconds=4,
    )

    result = service._collect_occurrences()

    assert result == [json_occ, ics_occ]
    fetch_ics.assert_called_once_with("https://example.com/feed.ics", 4)
    parse_ics.assert_called_once_with(
        "ICS",
        lookback_days=3,
        lookahead_days=9,
        feed_url="https://example.com/feed.ics",
    )


def test_embed_batches_splits_and_truncates() -> None:
    calls: list[list[str]] = []

    class RecordingEmbedder(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            calls.append(list(texts))
            return super().embed(texts)

    service = IngestionService(
        settings=Settings(embedding_batch_size=2, embedding_context_length=2),
        store=FakeStore(),
        embedder=RecordingEmbedder(),
    )

    vectors = service._embed_batches(["a", "b", "c", "word " * 10])

    assert len(vectors) == 4
    assert [len(batch) for batch in calls] == [2, 2]
    assert len(calls[1][1]) <= 6


def test_cross_source_dedup_keeps_first_non_json() -> None:
    first = _occ(occurrence_id="a", source_type="ics", description="first")
    second = _occ(occurrence_id="b", source_type="ics", description="second")
    result = _dedupe_by_time_and_title([first, second])
    assert [o.description for o in result] == ["first"]


def test_next_occurrence_start_prefers_future_else_latest() -> None:
    now = datetime(2026, 3, 15, tzinfo=UTC)
    past_a = _occ(start_time=datetime(2026, 3, 1, tzinfo=UTC))
    past_b = _occ(start_time=datetime(2026, 3, 8, tzinfo=UTC))
    future_a = _occ(start_time=datetime(2026, 3, 29, tzinfo=UTC))
    future_b = _occ(start_time=datetime(2026, 3, 22, tzinfo=UTC))

    assert _next_occurrence_start([past_a, past_b], now) == past_b.start_time
    assert (
        _next_occurrence_start([past_a, future_a, future_b], now) == future_b.start_time
    )
    assert _next_occurrence_start([], now) is None
