from uuid import UUID

from events2rag.qdrant_store import _to_point_id


def test_to_point_id_returns_uuid_and_is_stable() -> None:
    source_id = "occurrence-439@events.pdxhackerspace.org:2026-03-04T18:30:00+00:00"
    point_id_a = _to_point_id(source_id)
    point_id_b = _to_point_id(source_id)

    assert point_id_a == point_id_b
    assert str(UUID(point_id_a)) == point_id_a

