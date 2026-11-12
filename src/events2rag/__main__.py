"""Command line: `events2rag` or `python -m events2rag`.

Fetches the configured event feeds, embeds them and upserts them into
Qdrant, then repeats every POLL_INTERVAL_SECONDS until stopped.
"""

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Sequence
from types import FrameType

from events2rag import __version__
from events2rag.config import ConfigError, Settings
from events2rag.embedder import (
    EmbeddingBackend,
    OnnxEmbedder,
    OpenAIEmbedder,
    PromptedEmbedder,
    SentenceTransformerEmbedder,
)
from events2rag.qdrant_store import QdrantStore
from events2rag.service import IngestionService

logger = logging.getLogger("events2rag")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def build_embedder(settings: Settings) -> PromptedEmbedder:
    """The configured backend, with the configured prompt prefixes.

    Other code can embed a search query the same way the stored events
    were embedded: build_embedder(Settings.from_env()).embed([q], "query").
    """
    return PromptedEmbedder(
        build_backend(settings),
        query_prefix=settings.embedding_query_prefix,
        document_prefix=settings.embedding_document_prefix,
        max_tokens=settings.embedding_context_length,
    )


def build_backend(settings: Settings) -> EmbeddingBackend:
    backend = settings.embedding_backend

    if backend == "openai":
        return OpenAIEmbedder(
            model_name=settings.embedding_model_name,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout_seconds,
        )

    if backend == "onnx":
        return OnnxEmbedder(
            settings.embedding_model_name,
            max_length=settings.embedding_context_length,
        )

    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model_name)

    raise ValueError(f"Unknown embedding backend: {backend!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="events2rag", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.parse_args(argv)

    try:
        settings = Settings.from_env()
    except ConfigError as e:
        parser.exit(2, f"events2rag: {e}\n")

    configure_logging(settings.log_level)
    logger.info(
        "events2rag %s starting (backend=%s, model=%s, query prefix=%r, "
        "document prefix=%r)",
        __version__,
        settings.embedding_backend,
        settings.embedding_model_name,
        settings.embedding_query_prefix,
        settings.embedding_document_prefix,
    )
    store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
        distance_name=settings.qdrant_distance,
        embedding_signature=settings.embedding_signature,
    )
    service = IngestionService(
        settings=settings, store=store, embedder=build_embedder(settings)
    )

    # `docker stop` sends SIGTERM: finish the current cycle and exit instead
    # of being killed mid-upsert.
    stop = threading.Event()

    def on_signal(signum: int, frame: FrameType | None) -> None:
        logger.info("received signal %d, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    service.run_forever(stop)
    logger.info("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
