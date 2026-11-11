from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

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


def _occ(**overrides) -> EventOccurrence:
    values = {
        "occurrence_id": "e1:1",
        "event_id": "e1",
        "title": "Title",
        "description": "Desc",
        "start_time": datetime(2026, 3, 1, 18, tzinfo=UTC),
        "end_time": None,
        "location": None,
        "source_url": None,
    }
    values.update(overrides)
    return EventOccurrence(**values)


class MetadataStore(FakeStore):
    def __init__(self, existing: dict[str, dict]) -> None:
        super().__init__()
        self.existing = existing

    def get_existing_metadata(
        self, point_ids: list[str]
    ) -> dict[str, dict]:
        return {pid: self.existing[pid] for pid in point_ids if pid in self.existing}


def _service(store: FakeStore, **settings_kwargs) -> IngestionService:
    settings = Settings(
        events_json_url="https://example.com/events.json", **settings_kwargs
    )
    return IngestionService(
        settings=settings, store=store, embedder=FakeEmbedder()
    )


def test_run_cycle_returns_zero_when_no_events(mocker) -> None:
    mocker.patch("events2rag.service.fetch_json", return_value=[])
    store = FakeStore()
    assert _service(store).run_cycle() == 0
    assert store.upserted_summaries == []


def test_run_forever_propagates_keyboard_interrupt(mocker) -> None:
    store = FakeStore()
    service = _service(store)
    mocker.patch.object(
        service, "run_cycle", side_effect=KeyboardInterrupt
    )
    sleep = mocker.Mock()

    with pytest.raises(KeyboardInterrupt):
        service.run_forever(sleep_fn=sleep)
    sleep.assert_not_called()


def test_filter_changed_skips_unchanged_occurrences(mocker) -> None:
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
    mocker,
) -> None:
    occ = _occ(occurrence_id="same")
    mocker.patch("events2rag.service.fetch_json", return_value=[])
    mocker.patch(
        "events2rag.service.parse_event_occurrences", return_value=[occ]
    )
    store = MetadataStore(
        {_to_point_id("same"): {"ingested_at": "2026-01-01T00:00:00+00:00"}}
    )

    assert _service(store).run_cycle() == 1
    assert store.upserted_occurrences == []
    assert len(store.upserted_summaries) == 1


def test_collect_occurrences_includes_ics_feed(mocker) -> None:
    json_occ = _occ(occurrence_id="json", title="From JSON")
    ics_occ = _occ(
        occurrence_id="ics", title="From ICS", source_type="ics"
    )
    mocker.patch("events2rag.service.fetch_json", return_value=[])
    mocker.patch(
        "events2rag.service.parse_event_occurrences",
        return_value=[json_occ],
    )
    fetch_ics = mocker.patch(
        "events2rag.service.fetch_ics", return_value="ICS"
    )
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
        def embed(self, texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            return super().embed(texts)

    service = IngestionService(
        settings=Settings(
            embedding_batch_size=2, embedding_context_length=2
        ),
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
        _next_occurrence_start([past_a, future_a, future_b], now)
        == future_b.start_time
    )
    assert _next_occurrence_start([], now) is None
