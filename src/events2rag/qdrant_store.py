from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from events2rag.models import EventOccurrence, EventSummary

DISTANCE_BY_NAME: dict[str, Distance] = {
    "Cosine": Distance.COSINE,
    "Dot": Distance.DOT,
    "Euclid": Distance.EUCLID,
    "Manhattan": Distance.MANHATTAN,
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
        if self._client.collection_exists(collection_name=self._collection_name):
            return
        distance = DISTANCE_BY_NAME.get(self._distance_name, Distance.COSINE)
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

    def upsert_occurrences(
        self, occurrences: Sequence[EventOccurrence], vectors: Sequence[list[float]]
    ) -> int:
        if len(occurrences) != len(vectors):
            raise ValueError("occurrences and vectors must have equal lengths")
        now = datetime.now(tz=UTC).isoformat()
        points = [
            PointStruct(
                id=_to_point_id(occurrence.occurrence_id),
                vector=vector,
                payload={
                    "record_type": "occurrence",
                    "occurrence_id": occurrence.occurrence_id,
                    "event_id": occurrence.event_id,
                    "title": occurrence.title,
                    "description": occurrence.description,
                    "start_time": occurrence.start_time.isoformat(),
                    "end_time": occurrence.end_time.isoformat()
                    if occurrence.end_time
                    else None,
                    "location": occurrence.location,
                    "source_url": occurrence.source_url,
                    "tags": occurrence.tags,
                    "source_type": occurrence.source_type,
                    "last_modified": occurrence.last_modified.isoformat()
                    if occurrence.last_modified
                    else None,
                    "ingested_at": now,
                    "embedding_text": occurrence.embedding_text(),
                },
            )
            for occurrence, vector in zip(occurrences, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)
        return len(points)

    def upsert_event_summaries(
        self, summaries: Sequence[EventSummary], vectors: Sequence[list[float]]
    ) -> int:
        if len(summaries) != len(vectors):
            raise ValueError("summaries and vectors must have equal lengths")
        now = datetime.now(tz=UTC).isoformat()
        points = [
            PointStruct(
                id=_to_point_id(f"event:{summary.event_id}"),
                vector=vector,
                payload={
                    "record_type": "event_summary",
                    "event_id": summary.event_id,
                    "title": summary.title,
                    "description": summary.description,
                    "next_start_time": summary.next_start_time.isoformat()
                    if summary.next_start_time
                    else None,
                    "locations": summary.locations,
                    "tags": summary.tags,
                    "source_url": summary.source_url,
                    "source_type": summary.source_type,
                    "occurrence_count": summary.occurrence_count,
                    "last_modified": summary.last_modified.isoformat()
                    if summary.last_modified
                    else None,
                    "ingested_at": now,
                    "embedding_text": summary.embedding_text(),
                },
            )
            for summary, vector in zip(summaries, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection_name, points=points)
        return len(points)


def _to_point_id(occurrence_id: str) -> str:
    """Convert any occurrence key into a deterministic UUID for Qdrant."""
    return str(uuid5(NAMESPACE_URL, occurrence_id))

