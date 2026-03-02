from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from events2rag.models import EventOccurrence, EventSummary

DISTANCE_BY_NAME: dict[str, Distance] = {
    "Cosine": Distance.COSINE,
    "Dot": Distance.DOT,
    "Euclid": Distance.EUCLID,
    "Manhattan": Distance.MANHATTAN,
}

PAYLOAD_INDEXES: dict[str, PayloadSchemaType] = {
    "record_type": PayloadSchemaType.KEYWORD,
    "temporal_status": PayloadSchemaType.KEYWORD,
    "tags": PayloadSchemaType.KEYWORD,
    "location": PayloadSchemaType.KEYWORD,
    "event_id": PayloadSchemaType.KEYWORD,
    "start_time": PayloadSchemaType.DATETIME,
}


class QdrantStore:
    def __init__(
        self,
        url: str,
        api_key: str | None,
        collection_name: str,
        distance_name: str = "Cosine",
    ) -> None:
        self._client = QdrantClient(url=url, api_key=api_key)
        self._collection_name = collection_name
        self._distance_name = distance_name

    def ensure_collection(self, vector_size: int) -> None:
        if not self._client.collection_exists(
            collection_name=self._collection_name
        ):
            distance = DISTANCE_BY_NAME.get(
                self._distance_name, Distance.COSINE
            )
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=distance
                ),
            )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        for field_name, schema_type in PAYLOAD_INDEXES.items():
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )

    def upsert_occurrences(
        self,
        occurrences: Sequence[EventOccurrence],
        vectors: Sequence[list[float]],
    ) -> int:
        if len(occurrences) != len(vectors):
            raise ValueError(
                "occurrences and vectors must have equal lengths"
            )
        now = datetime.now(tz=UTC).isoformat()
        points = [
            PointStruct(
                id=_to_point_id(occ.occurrence_id),
                vector=vector,
                payload={
                    "record_type": "occurrence",
                    "occurrence_id": occ.occurrence_id,
                    "event_id": occ.event_id,
                    "title": occ.title,
                    "description": occ.description,
                    "start_time": occ.start_time.isoformat(),
                    "end_time": occ.end_time.isoformat()
                    if occ.end_time
                    else None,
                    "location": occ.location,
                    "source_url": occ.source_url,
                    "tags": occ.tags,
                    "source_type": occ.source_type,
                    "temporal_status": occ.temporal_status,
                    "duration": occ.duration,
                    "last_modified": occ.last_modified.isoformat()
                    if occ.last_modified
                    else None,
                    "ingested_at": now,
                    "embedding_text": occ.embedding_text(),
                },
            )
            for occ, vector in zip(occurrences, vectors, strict=True)
        ]
        self._client.upsert(
            collection_name=self._collection_name, points=points
        )
        return len(points)

    def upsert_event_summaries(
        self,
        summaries: Sequence[EventSummary],
        vectors: Sequence[list[float]],
    ) -> int:
        if len(summaries) != len(vectors):
            raise ValueError(
                "summaries and vectors must have equal lengths"
            )
        now = datetime.now(tz=UTC).isoformat()
        points = [
            PointStruct(
                id=_to_point_id(f"event:{s.event_id}"),
                vector=vector,
                payload={
                    "record_type": "event_summary",
                    "event_id": s.event_id,
                    "title": s.title,
                    "description": s.description,
                    "next_start_time": s.next_start_time.isoformat()
                    if s.next_start_time
                    else None,
                    "locations": s.locations,
                    "tags": s.tags,
                    "source_url": s.source_url,
                    "source_type": s.source_type,
                    "occurrence_count": s.occurrence_count,
                    "frequency": s.frequency,
                    "has_future_occurrences": s.has_future_occurrences,
                    "last_modified": s.last_modified.isoformat()
                    if s.last_modified
                    else None,
                    "ingested_at": now,
                    "embedding_text": s.embedding_text(),
                },
            )
            for s, vector in zip(summaries, vectors, strict=True)
        ]
        self._client.upsert(
            collection_name=self._collection_name, points=points
        )
        return len(points)


def _to_point_id(occurrence_id: str) -> str:
    """Convert any occurrence key into a deterministic UUID."""
    return str(uuid5(NAMESPACE_URL, occurrence_id))
