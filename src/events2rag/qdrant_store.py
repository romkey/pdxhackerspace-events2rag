from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from events2rag.models import EventOccurrence

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
                id=occurrence.occurrence_id,
                vector=vector,
                payload={
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

