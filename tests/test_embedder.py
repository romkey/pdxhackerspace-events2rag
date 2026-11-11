from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import requests

from events2rag.embedder import (
    OllamaEmbedder,
    OnnxEmbedder,
    SentenceTransformerEmbedder,
    _mean_pool,
)


class FakeResponse:
    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> dict:
        return self._data


def test_ollama_embedder_probe_and_embed(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        calls.append({"url": url, "json": json, "headers": headers})
        input_val = json.get("input")
        count = 1 if isinstance(input_val, str) else len(input_val)
        return FakeResponse(
            {"embeddings": [[0.1, 0.2, 0.3]] * count}
        )

    monkeypatch.setattr(requests, "post", fake_post)

    embedder = OllamaEmbedder(
        model_name="test-model",
        ollama_url="http://localhost:11434",
    )

    assert embedder.dimension == 3

    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3

    assert calls[0]["json"]["model"] == "test-model"
    assert calls[1]["json"]["input"] == ["hello", "world"]
    assert calls[0]["headers"] is None
    assert calls[1]["headers"] is None


def test_ollama_embedder_uses_correct_url(monkeypatch) -> None:
    captured_urls: list[str] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        captured_urls.append(url)
        return FakeResponse({"embeddings": [[1.0, 2.0]]})

    monkeypatch.setattr(requests, "post", fake_post)

    OllamaEmbedder(
        model_name="m",
        ollama_url="http://myhost:9999",
    )
    assert captured_urls[0] == "http://myhost:9999/api/embed"


def test_ollama_embedder_sends_api_key(monkeypatch) -> None:
    headers_seen: list[dict | None] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        headers_seen.append(headers)
        return FakeResponse({"embeddings": [[1.0, 2.0]]})

    monkeypatch.setattr(requests, "post", fake_post)

    embedder = OllamaEmbedder(
        model_name="m",
        ollama_url="http://myhost:9999",
        api_key="secret-token",
    )
    embedder.embed(["hello"])

    assert headers_seen == [
        {"Authorization": "Bearer secret-token"},
        {"Authorization": "Bearer secret-token"},
    ]


def test_ollama_embedder_accepts_legacy_embedding_key(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        return FakeResponse({"embedding": [[0.5, 0.5]]})

    monkeypatch.setattr(requests, "post", fake_post)

    embedder = OllamaEmbedder(model_name="m")
    assert embedder.dimension == 2
    assert embedder.embed(["x"]) == [[0.5, 0.5]]


def test_ollama_embedder_raises_when_no_embeddings(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        return FakeResponse({"embeddings": []})

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(ValueError, match="no embeddings"):
        OllamaEmbedder(model_name="missing-model")


def test_ollama_embedder_propagates_http_errors(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        return FakeResponse({}, status_code=500)

    monkeypatch.setattr(requests, "post", fake_post)

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
            _FakeEncoding(
                [1] * len(t.split()) + [0] * (width - len(t.split()))
            )
            for t in texts
        ]


class _FakeSession:
    DIM = 4

    def __init__(self, path: str) -> None:
        self.path = path
        self.feeds: list[dict] = []

    def run(self, _outputs, feed):
        self.feeds.append(feed)
        ids = feed["input_ids"]
        batch, seq = ids.shape
        # Token embedding = [position+1, 0, 0, 0] so pooling is checkable.
        out = np.zeros((batch, seq, self.DIM), dtype=np.float32)
        for pos in range(seq):
            out[:, pos, 0] = pos + 1
        return [out]


@pytest.fixture
def fake_onnx_stack(monkeypatch):
    downloads: list[tuple[str, str]] = []

    def hf_hub_download(repo_id: str, filename: str) -> str:
        downloads.append((repo_id, filename))
        return f"/cache/{repo_id}/{filename}"

    modules = {
        "huggingface_hub": types.SimpleNamespace(
            hf_hub_download=hf_hub_download
        ),
        "onnxruntime": types.SimpleNamespace(InferenceSession=_FakeSession),
        "tokenizers": types.SimpleNamespace(Tokenizer=_FakeTokenizer),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return downloads


def test_onnx_embedder(fake_onnx_stack) -> None:
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
    tokens = np.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = np.array([[1, 1, 0]])
    pooled = _mean_pool(np, tokens, mask)
    assert pooled.tolist() == [[2.0, 3.0]]


class _FakeSentenceTransformer:
    dimension: int | None = 3

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encode_kwargs: dict = {}

    def get_sentence_embedding_dimension(self) -> int | None:
        return self.dimension

    def encode(self, texts, **kwargs):
        self.encode_kwargs = kwargs
        return [np.array([1.0, 0.0, 0.0]) for _ in texts]


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )
    monkeypatch.setattr(_FakeSentenceTransformer, "dimension", 3)


def test_sentence_transformer_embedder(fake_sentence_transformers) -> None:
    embedder = SentenceTransformerEmbedder("org/model")

    assert embedder.dimension == 3
    assert embedder._model.model_name == "org/model"
    assert embedder.embed(["a", "b"]) == [[1.0, 0.0, 0.0]] * 2
    assert embedder._model.encode_kwargs == {"normalize_embeddings": True}


def test_sentence_transformer_embedder_requires_dimension(
    fake_sentence_transformers, monkeypatch
) -> None:
    monkeypatch.setattr(_FakeSentenceTransformer, "dimension", None)
    with pytest.raises(ValueError, match="dimension"):
        SentenceTransformerEmbedder("org/model")
