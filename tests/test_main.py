from __future__ import annotations

import logging

import pytest

from events2rag import embedder as embedder_module
from events2rag import main as main_module
from events2rag.config import Settings


class Recorder:
    """Stand-in class that records constructor arguments."""

    instances: list[Recorder] = []

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        Recorder.instances.append(self)


@pytest.fixture(autouse=True)
def reset_recorder() -> None:
    Recorder.instances = []


def test_configure_logging_sets_level(mocker) -> None:
    basic_config = mocker.patch.object(logging, "basicConfig")
    main_module.configure_logging("debug")
    assert basic_config.call_args.kwargs["level"] == logging.DEBUG


def test_configure_logging_unknown_level_falls_back_to_info(mocker) -> None:
    basic_config = mocker.patch.object(logging, "basicConfig")
    main_module.configure_logging("chatty")
    assert basic_config.call_args.kwargs["level"] == logging.INFO


def test_build_embedder_ollama(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "OllamaEmbedder", Recorder)
    settings = Settings(
        embedding_backend="ollama",
        embedding_model_name="m",
        ollama_url="http://o:1",
        ollama_api_key="k",
        request_timeout_seconds=9,
    )

    result = main_module.build_embedder(settings)

    assert isinstance(result, Recorder)
    assert result.kwargs == {
        "model_name": "m",
        "ollama_url": "http://o:1",
        "api_key": "k",
        "timeout": 9,
    }


def test_build_embedder_onnx(monkeypatch) -> None:
    monkeypatch.setattr(embedder_module, "OnnxEmbedder", Recorder)
    settings = Settings(
        embedding_backend="onnx",
        embedding_model_name="m",
        embedding_context_length=128,
    )

    result = main_module.build_embedder(settings)

    assert isinstance(result, Recorder)
    assert result.args == ("m",)
    assert result.kwargs == {"max_length": 128}


def test_build_embedder_sentence_transformers(monkeypatch) -> None:
    monkeypatch.setattr(
        embedder_module, "SentenceTransformerEmbedder", Recorder
    )
    settings = Settings(
        embedding_backend="sentence-transformers",
        embedding_model_name="m",
    )

    result = main_module.build_embedder(settings)

    assert isinstance(result, Recorder)
    assert result.args == ("m",)


def test_build_embedder_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        main_module.build_embedder(Settings(embedding_backend="nope"))


def test_main_wires_components(mocker) -> None:
    settings = Settings(
        qdrant_url="http://q:6333",
        qdrant_api_key="qk",
        qdrant_collection="c",
        qdrant_distance="Dot",
        log_level="WARNING",
    )
    mocker.patch.object(Settings, "from_env", return_value=settings)
    configure = mocker.patch.object(main_module, "configure_logging")
    fake_embedder = object()
    mocker.patch.object(
        main_module, "build_embedder", return_value=fake_embedder
    )
    store_cls = mocker.patch.object(main_module, "QdrantStore")
    service_cls = mocker.patch.object(main_module, "IngestionService")

    main_module.main()

    configure.assert_called_once_with("WARNING")
    store_cls.assert_called_once_with(
        url="http://q:6333",
        api_key="qk",
        collection_name="c",
        distance_name="Dot",
    )
    service_cls.assert_called_once_with(
        settings=settings,
        store=store_cls.return_value,
        embedder=fake_embedder,
    )
    service_cls.return_value.run_forever.assert_called_once_with()
