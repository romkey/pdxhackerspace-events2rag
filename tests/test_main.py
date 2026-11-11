import logging
import signal
import threading
from collections.abc import Callable
from typing import Any

import pytest
from pytest_mock import MockerFixture

from events2rag import __main__ as cli
from events2rag import __version__
from events2rag.config import Settings


class Recorder:
    """Stand-in for an embedder class: records its constructor arguments."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"events2rag {__version__}\n"


def test_bad_config_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "hourly")
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == 2
    assert "POLL_INTERVAL_SECONDS must be a whole number" in capsys.readouterr().err


def test_configure_logging_sets_level(mocker: MockerFixture) -> None:
    basic_config = mocker.patch.object(logging, "basicConfig")
    cli.configure_logging("DEBUG")
    assert basic_config.call_args.kwargs["level"] == "DEBUG"


def test_build_embedder_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "OllamaEmbedder", Recorder)
    settings = Settings(
        embedding_backend="ollama",
        embedding_model_name="m",
        ollama_url="http://o:1",
        ollama_api_key="k",
        request_timeout_seconds=9,
    )

    result = cli.build_embedder(settings)

    assert isinstance(result, Recorder)
    assert result.kwargs == {
        "model_name": "m",
        "ollama_url": "http://o:1",
        "api_key": "k",
        "timeout": 9,
    }


def test_build_embedder_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "OnnxEmbedder", Recorder)
    settings = Settings(
        embedding_backend="onnx",
        embedding_model_name="m",
        embedding_context_length=128,
    )

    result = cli.build_embedder(settings)

    assert isinstance(result, Recorder)
    assert result.args == ("m",)
    assert result.kwargs == {"max_length": 128}


def test_build_embedder_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", Recorder)
    settings = Settings(
        embedding_backend="sentence-transformers", embedding_model_name="m"
    )

    result = cli.build_embedder(settings)

    assert isinstance(result, Recorder)
    assert result.args == ("m",)


def test_build_embedder_unknown_backend() -> None:
    # Settings.from_env rejects this; build_embedder still guards itself.
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        cli.build_embedder(Settings(embedding_backend="nope"))


def test_main_wires_components_and_stops_on_sigterm(
    monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture
) -> None:
    for name, value in {
        "QDRANT_URL": "http://q:6333",
        "QDRANT_API_KEY": "qk",
        "QDRANT_COLLECTION": "c",
        "QDRANT_DISTANCE": "Dot",
        "LOG_LEVEL": "WARNING",
    }.items():
        monkeypatch.setenv(name, value)
    configure = mocker.patch.object(cli, "configure_logging")
    fake_embedder = mocker.sentinel.embedder
    mocker.patch.object(cli, "build_embedder", return_value=fake_embedder)
    store_cls = mocker.patch.object(cli, "QdrantStore")
    service_cls = mocker.patch.object(cli, "IngestionService")
    handlers: dict[int, Callable[..., Any]] = {}
    monkeypatch.setattr(signal, "signal", handlers.__setitem__)

    stops: list[threading.Event] = []

    def run_forever(stop: threading.Event) -> None:
        stops.append(stop)
        assert not stop.is_set()
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        assert stop.is_set()

    service_cls.return_value.run_forever.side_effect = run_forever

    assert cli.main([]) == 0

    configure.assert_called_once_with("WARNING")
    store_cls.assert_called_once_with(
        url="http://q:6333",
        api_key="qk",
        collection_name="c",
        distance_name="Dot",
    )
    service_cls.assert_called_once_with(
        settings=mocker.ANY, store=store_cls.return_value, embedder=fake_embedder
    )
    assert len(stops) == 1
    assert signal.SIGINT in handlers
