from __future__ import annotations

import requests

from events2rag.embedder import OllamaEmbedder


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
