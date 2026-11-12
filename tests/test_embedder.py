from __future__ import annotations

import pytest
import requests

from events2rag.embedder import OpenAIEmbedder


class FakeResponse:
    def __init__(self, data: dict, status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self) -> dict:
        return self._data


def _openai_payload(count: int, vector: list[float]) -> dict:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vector}
            for i in range(count)
        ],
    }


def test_openai_embedder_probe_and_embed(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        calls.append({"url": url, "json": json, "headers": headers})
        return FakeResponse(_openai_payload(len(json["input"]), [0.1, 0.2, 0.3]))

    monkeypatch.setattr(requests, "post", fake_post)

    embedder = OpenAIEmbedder(model_name="test-model")

    assert embedder.dimension == 3

    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3

    assert calls[0]["url"] == "https://api.openai.com/v1/embeddings"
    assert calls[0]["json"]["model"] == "test-model"
    assert calls[1]["json"]["input"] == ["hello", "world"]
    assert calls[0]["headers"] is None
    assert calls[1]["headers"] is None


def test_openai_embedder_uses_custom_base_url(monkeypatch) -> None:
    captured_urls: list[str] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        captured_urls.append(url)
        return FakeResponse(_openai_payload(1, [1.0, 2.0]))

    monkeypatch.setattr(requests, "post", fake_post)

    OpenAIEmbedder(model_name="m", base_url="http://myhost:9999/v1/")
    assert captured_urls[0] == "http://myhost:9999/v1/embeddings"


def test_openai_embedder_sends_api_key(monkeypatch) -> None:
    headers_seen: list[dict | None] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        headers_seen.append(headers)
        return FakeResponse(_openai_payload(1, [1.0, 2.0]))

    monkeypatch.setattr(requests, "post", fake_post)

    embedder = OpenAIEmbedder(model_name="m", api_key="secret-token")
    embedder.embed(["hello"])

    assert headers_seen == [
        {"Authorization": "Bearer secret-token"},
        {"Authorization": "Bearer secret-token"},
    ]


def test_openai_embedder_orders_by_index(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        if len(json["input"]) == 1:
            return FakeResponse(_openai_payload(1, [0.0, 0.0]))
        return FakeResponse(
            {
                "data": [
                    {"index": 1, "embedding": [2.0, 2.0]},
                    {"index": 0, "embedding": [1.0, 1.0]},
                ]
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)

    embedder = OpenAIEmbedder(model_name="m")
    assert embedder.embed(["a", "b"]) == [[1.0, 1.0], [2.0, 2.0]]


def test_openai_embedder_rejects_short_response(monkeypatch) -> None:
    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        if len(json["input"]) == 1:
            return FakeResponse(_openai_payload(1, [0.0, 0.0]))
        return FakeResponse(_openai_payload(1, [0.0, 0.0]))

    monkeypatch.setattr(requests, "post", fake_post)

    embedder = OpenAIEmbedder(model_name="m")
    with pytest.raises(ValueError):
        embedder.embed(["a", "b"])


def test_openai_embedder_empty_input_skips_request(monkeypatch) -> None:
    calls: list[str] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: F811
        calls.append(url)
        return FakeResponse(_openai_payload(len(json["input"]), [1.0]))

    monkeypatch.setattr(requests, "post", fake_post)

    embedder = OpenAIEmbedder(model_name="m")
    assert embedder.embed([]) == []
    assert len(calls) == 1
