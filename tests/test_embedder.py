import sys
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import requests
from numpy.typing import NDArray

from events2rag.embedder import (
    OllamaEmbedder,
    OnnxEmbedder,
    SentenceTransformerEmbedder,
    _mean_pool,
)


class FakeResponse:
    # Any: stands in for an arbitrary JSON response body.
    def __init__(self, data: Any, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=None)

    def json(self) -> Any:
        return self._data


# Records each fake requests.post call: url, json body, headers.
Call = dict[str, Any]
Post = Callable[..., FakeResponse]


def fake_post(calls: list[Call], reply: Callable[[Any], Any]) -> Post:
    """A requests.post stand-in that records calls and answers with reply(body)."""

    def post(
        url: str,
        json: Any = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> FakeResponse:
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse(reply(json))

    return post


def _vectors_for(body: Any) -> Any:
    count = 1 if isinstance(body["input"], str) else len(body["input"])
    return {"embeddings": [[0.1, 0.2, 0.3]] * count}


def test_ollama_embedder_probe_and_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Call] = []
    monkeypatch.setattr(requests, "post", fake_post(calls, _vectors_for))

    embedder = OllamaEmbedder(
        model_name="test-model", ollama_url="http://localhost:11434/", timeout=7
    )

    assert embedder.dimension == 3
    vectors = embedder.embed(["hello", "world"])
    assert vectors == [[0.1, 0.2, 0.3]] * 2
    assert calls[0]["url"] == "http://localhost:11434/api/embed"
    assert calls[0]["json"] == {
        "model": "test-model",
        "input": "hello",
        "truncate": True,
    }
    assert calls[1]["json"]["input"] == ["hello", "world"]
    assert calls[1]["headers"] is None
    assert calls[1]["timeout"] == 7


def test_ollama_embedder_sends_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Call] = []
    monkeypatch.setattr(requests, "post", fake_post(calls, _vectors_for))

    OllamaEmbedder(model_name="m", api_key="secret-token").embed(["hello"])

    assert [c["headers"] for c in calls] == [
        {"Authorization": "Bearer secret-token"}
    ] * 2


def test_ollama_embedder_accepts_legacy_embedding_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        requests, "post", fake_post([], lambda _: {"embedding": [[1, 0.5]]})
    )

    embedder = OllamaEmbedder(model_name="m")

    assert embedder.dimension == 2
    assert embedder.embed(["x"]) == [[1.0, 0.5]]


def test_ollama_embedder_raises_when_no_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(requests, "post", fake_post([], lambda _: {"embeddings": []}))

    with pytest.raises(ValueError, match="no embeddings for model missing-model"):
        OllamaEmbedder(model_name="missing-model")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (["not", "an", "object"], "returned list, not an object"),
        ({}, "aren't lists of numbers"),
        ({"embeddings": [["a"]]}, "aren't lists of numbers"),
        ({"embeddings": "nope"}, "aren't lists of numbers"),
    ],
)
def test_ollama_embedder_rejects_malformed_responses(
    monkeypatch: pytest.MonkeyPatch, body: Any, message: str
) -> None:
    monkeypatch.setattr(requests, "post", fake_post([], lambda _: body))

    with pytest.raises(ValueError, match=message):
        OllamaEmbedder(model_name="m")


def test_ollama_embedder_propagates_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def post(*_args: object, **_kwargs: object) -> FakeResponse:
        return FakeResponse({}, status_code=500)

    monkeypatch.setattr(requests, "post", post)

    with pytest.raises(requests.HTTPError):
        OllamaEmbedder(model_name="m")


class _FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = [1 if i else 0 for i in ids]


class _FakeTokenizer:
    loaded_from: str | None = None

    def __init__(self) -> None:
        self.padding = False
        self.truncation: int | None = None

    @classmethod
    def from_file(cls, path: str) -> _FakeTokenizer:
        cls.loaded_from = path
        return cls()

    def enable_padding(self) -> None:
        self.padding = True

    def enable_truncation(self, max_length: int) -> None:
        self.truncation = max_length

    def encode(self, text: str) -> _FakeEncoding:
        return _FakeEncoding([1] * len(text.split()))

    def encode_batch(self, texts: list[str]) -> list[_FakeEncoding]:
        width = max(len(t.split()) for t in texts)
        return [
            _FakeEncoding([1] * len(t.split()) + [0] * (width - len(t.split())))
            for t in texts
        ]


class _FakeSession:
    DIM = 4

    def __init__(self, path: str) -> None:
        self.path = path
        self.feeds: list[dict[str, NDArray[np.int64]]] = []

    def run(
        self, _outputs: None, feed: dict[str, NDArray[np.int64]]
    ) -> list[NDArray[np.float32]]:
        self.feeds.append(feed)
        batch, seq = feed["input_ids"].shape
        # Token embedding = [position+1, 0, 0, 0] so pooling is checkable.
        out = np.zeros((batch, seq, self.DIM), dtype=np.float32)
        for pos in range(seq):
            out[:, pos, 0] = pos + 1
        return [out]


@pytest.fixture
def fake_onnx_stack(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    downloads: list[tuple[str, str]] = []

    def hf_hub_download(repo_id: str, filename: str) -> str:
        downloads.append((repo_id, filename))
        return f"/cache/{repo_id}/{filename}"

    modules = {
        "huggingface_hub": types.SimpleNamespace(hf_hub_download=hf_hub_download),
        "onnxruntime": types.SimpleNamespace(InferenceSession=_FakeSession),
        "tokenizers": types.SimpleNamespace(Tokenizer=_FakeTokenizer),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return downloads


def test_onnx_embedder(fake_onnx_stack: list[tuple[str, str]]) -> None:
    embedder = OnnxEmbedder("org/model", max_length=32)

    assert fake_onnx_stack == [
        ("org/model", "tokenizer.json"),
        ("org/model", "onnx/model.onnx"),
    ]
    assert _FakeTokenizer.loaded_from == "/cache/org/model/tokenizer.json"
    assert embedder._tokenizer.padding is True
    assert embedder._tokenizer.truncation == 32
    assert embedder._session.path == "/cache/org/model/onnx/model.onnx"
    assert embedder.dimension == _FakeSession.DIM

    vectors = embedder.embed(["one two three", "one"])

    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == _FakeSession.DIM
        assert np.linalg.norm(vector) == pytest.approx(1.0)
    feed = embedder._session.feeds[-1]
    assert feed["input_ids"].dtype == np.int64
    assert feed["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert not feed["token_type_ids"].any()


def test_mean_pool_ignores_padding() -> None:
    tokens = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.array([[1, 1, 0]], dtype=np.int64)
    assert _mean_pool(tokens, mask).tolist() == [[2.0, 3.0]]


class _FakeSentenceTransformer:
    dimension: int | None = 3

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encode_kwargs: dict[str, object] = {}

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: object) -> list[NDArray[np.float64]]:
        self.encode_kwargs = kwargs
        return [np.array([1.0, 0.0, 0.0]) for _ in texts]


@pytest.fixture
def fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    monkeypatch.setattr(_FakeSentenceTransformer, "dimension", 3)


@pytest.mark.usefixtures("fake_sentence_transformers")
def test_sentence_transformer_embedder() -> None:
    embedder = SentenceTransformerEmbedder("org/model")

    assert embedder.dimension == 3
    assert embedder._model.model_name == "org/model"
    assert embedder.embed(["a", "b"]) == [[1.0, 0.0, 0.0]] * 2
    assert embedder._model.encode_kwargs == {"normalize_embeddings": True}


@pytest.mark.usefixtures("fake_sentence_transformers")
def test_sentence_transformer_embedder_requires_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_FakeSentenceTransformer, "dimension", None)
    with pytest.raises(ValueError, match="dimension"):
        SentenceTransformerEmbedder("org/model")
