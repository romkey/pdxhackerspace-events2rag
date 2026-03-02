from __future__ import annotations

import logging

from events2rag.config import Settings
from events2rag.embedder import SentenceTransformerEmbedder
from events2rag.qdrant_store import QdrantStore
from events2rag.service import IngestionService


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    logging.getLogger(__name__).info(
        "Starting ingester with JSON URL: %s", settings.events_json_url
    )
    embedder = SentenceTransformerEmbedder(settings.embedding_model_name)
    store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        distance_name=settings.qdrant_distance,
    )
    service = IngestionService(settings=settings, store=store, embedder=embedder)
    service.run_forever()


if __name__ == "__main__":
    main()

