import logging
import warnings
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import httpx
import pytest
from pytest_mock import MockerFixture
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from events2rag import qdrant_store
from events2rag.models import EventOccurrence, EventSummary
from events2rag.qdrant_store import PAYLOAD_INDEXES, QdrantStore, _to_point_id


def test_to_point_id_returns_uuid_and_is_stable() -> None:
    source_id = "occurrence-439@events.pdxhackerspace.org:2026-03-04T18:30:00+00:00"
    point_id_a = _to_point_id(source_id)
    point_id_b = _to_point_id(source_id)

    assert point_id_a == point_id_b
    assert str(UUID(point_id_a)) == point_id_a


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> QdrantStore:
    """A QdrantStore backed by qdrant-client's in-memory local mode."""
    seen: dict[str, str | None] = {}

    def fake_client(url: str, api_key: str | None) -> QdrantClient:
        seen["url"] = url
        seen["api_key"] = api_key
        return QdrantClient(location=":memory:")

    monkeypatch.setattr(qdrant_store, "QdrantClient", fake_client)
    result = QdrantStore(
        url="http://q:6333",
        api_key="secret",
        collection_name="events",
        distance_name="Dot",
    )
    assert seen == {"url": "http://q:6333", "api_key": "secret"}
    with warnings.catch_warnings():
        # Local mode warns that payload indexes have no effect.
        warnings.simplefilter("ignore", UserWarning)
        result.ensure_collection(vector_size=3)
    return result


_OCCURRENCE = EventOccurrence(
    occurrence_id="e1:2026-03-01",
    event_id="e1",
    title="Hack Night",
    description="Build stuff",
    start_time=datetime(2026, 3, 1, 18, tzinfo=UTC),
    end_time=datetime(2026, 3, 1, 20, tzinfo=UTC),
    location="Room A",
    source_url="https://example.com",
    tags=["social"],
    last_modified=datetime(2026, 2, 1, tzinfo=UTC),
    duration="2 hours",
)


# Any: each override is a different EventOccurrence field type.
def _occurrence(**overrides: Any) -> EventOccurrence:
    return replace(_OCCURRENCE, **overrides)


# Any: Qdrant payloads are JSON whose shape varies by field.
def _payload(store: QdrantStore, key: str) -> dict[str, Any]:
    point = store._client.retrieve("events", ids=[_to_point_id(key)])[0]
    assert point.payload is not None
    return point.payload


def test_ensure_collection_creates_with_configured_distance(store: QdrantStore) -> None:
    vectors = store._client.get_collection("events").config.params.vectors
    assert isinstance(vectors, VectorParams)
    assert vectors.size == 3
    assert vectors.distance == Distance.DOT


def test_ensure_collection_is_idempotent(store: QdrantStore) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        store.ensure_collection(vector_size=3)
    assert store._client.collection_exists("events")


def test_upsert_and_fetch_occurrence_metadata(store: QdrantStore) -> None:
    occ = _occurrence()
    bare = _occurrence(
        occurrence_id="e2:x",
        event_id="e2",
        end_time=None,
        last_modified=None,
    )

    assert store.upsert_occurrences([occ, bare], [[1, 0, 0], [0, 1, 0]]) == 2

    pid = _to_point_id(occ.occurrence_id)
    bare_pid = _to_point_id(bare.occurrence_id)
    meta = store.get_existing_metadata([pid, bare_pid, _to_point_id("nope")])

    assert set(meta) == {pid, bare_pid}
    assert meta[pid]["last_modified"] == "2026-02-01T00:00:00+00:00"
    assert "ingested_at" in meta[pid]
    assert meta[bare_pid]["last_modified"] is None

    payload = _payload(store, occ.occurrence_id)
    assert payload["record_type"] == "occurrence"
    assert payload["end_time"] == "2026-03-01T20:00:00+00:00"
    assert payload["embedding_text"] == occ.embedding_text()
    assert _payload(store, bare.occurrence_id)["end_time"] is None


def test_get_existing_metadata_empty(store: QdrantStore) -> None:
    assert store.get_existing_metadata([]) == {}


def test_upsert_event_summaries(store: QdrantStore) -> None:
    summary = EventSummary(
        event_id="e1",
        title="Hack Night",
        description="Build stuff",
        next_start_time=datetime(2026, 3, 1, 18, tzinfo=UTC),
        locations=["Room A"],
        occurrence_count=2,
        last_modified=datetime(2026, 2, 1, tzinfo=UTC),
        frequency="weekly",
        has_future_occurrences=True,
    )
    empty = EventSummary(
        event_id="e2",
        title="Mystery",
        description="",
        next_start_time=None,
    )

    assert store.upsert_event_summaries([summary, empty], [[1, 0, 0]] * 2) == 2

    payload = _payload(store, "event:e1")
    assert payload["record_type"] == "event_summary"
    assert payload["frequency"] == "weekly"
    assert payload["next_start_time"] == "2026-03-01T18:00:00+00:00"
    empty_payload = _payload(store, "event:e2")
    assert empty_payload["next_start_time"] is None
    assert empty_payload["last_modified"] is None


def test_upsert_length_mismatch_raises(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        store.upsert_occurrences([_occurrence()], [])
    with pytest.raises(ValueError, match="equal lengths"):
        store.upsert_event_summaries([], [[1, 0, 0]])


def _unexpected(message: str) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=400,
        reason_phrase="Bad Request",
        content=message.encode(),
        headers=httpx.Headers(),
    )


def test_payload_index_errors_are_logged_unless_already_exists(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    client = mocker.Mock()
    client.collection_exists.return_value = True
    errors = iter(
        [_unexpected("index already exists"), _unexpected("boom")]
        + [None] * len(PAYLOAD_INDEXES)
    )

    def create_payload_index(**_kwargs: object) -> None:
        error = next(errors)
        if error is not None:
            raise error

    client.create_payload_index.side_effect = create_payload_index
    mocker.patch.object(qdrant_store, "QdrantClient", return_value=client)
    store = QdrantStore(url="http://q", api_key=None, collection_name="c")

    with caplog.at_level(logging.WARNING):
        store.ensure_collection(vector_size=3)

    client.create_collection.assert_not_called()
    assert client.create_payload_index.call_count == len(PAYLOAD_INDEXES)
    warnings_logged = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings_logged) == 1
    assert "boom" in warnings_logged[0].getMessage()


def test_unknown_distance_defaults_to_cosine(mocker: MockerFixture) -> None:
    client = mocker.Mock()
    client.collection_exists.return_value = False
    mocker.patch.object(qdrant_store, "QdrantClient", return_value=client)
    store = QdrantStore(
        url="http://q",
        api_key=None,
        collection_name="c",
        distance_name="Bogus",
    )

    store.ensure_collection(vector_size=4)

    vectors_config = client.create_collection.call_args.kwargs["vectors_config"]
    assert vectors_config.distance == Distance.COSINE
    assert vectors_config.size == 4
