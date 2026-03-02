from __future__ import annotations

import logging

from events2rag.config import Settings
from events2rag.embedder import Embedder, OnnxEmbedder
from events2rag.qdrant_store import QdrantStore
from events2rag.service import IngestionService


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "sentence-transformers":
        from events2rag.embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(settings.embedding_model_name)
    return OnnxEmbedder(settings.embedding_model_name)


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    logger = logging.getLogger(__name__)
    logger.info(
        "Starting ingester (backend=%s, model=%s)",
        settings.embedding_backend,
        settings.embedding_model_name,
    )
    embedder = build_embedder(settings)
    store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        distance_name=settings.qdrant_distance,
    )
    service = IngestionService(
        settings=settings, store=store, embedder=embedder
    )
    service.run_forever()


if __name__ == "__main__":
    main()
