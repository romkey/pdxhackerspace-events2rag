from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal, Protocol

import requests

from events2rag.text_utils import normalize_prefix, prepare_embedding_text

logger = logging.getLogger(__name__)

EmbedKind = Literal["query", "document"]


class EmbeddingBackend(Protocol):
    """A raw embedding model: text in, vectors out, no prompting."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    @property
    def dimension(self) -> int:
        ...


class Embedder(Protocol):
    """An embedder that knows whether it is encoding queries or documents."""

    def embed(
        self, texts: Sequence[str], kind: EmbedKind = "document"
    ) -> list[list[float]]:
        ...

    @property
    def dimension(self) -> int:
        ...


class PromptedEmbedder:
    """Wraps a backend, adding query/document prompt prefixes.

    Many embedding models (nomic/ModernBERT, e5, mxbai, Qwen3-Embedding)
    are trained with a task prefix on queries and/or documents and
    retrieve noticeably worse without it. The prefix counts toward
    ``max_tokens`` when the text is truncated.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        query_prefix: str = "",
        document_prefix: str = "",
        max_tokens: int = 512,
    ) -> None:
        self._backend = backend
        self._prefixes: dict[str, str] = {
            "query": normalize_prefix(query_prefix),
            "document": normalize_prefix(document_prefix),
        }
        self._max_tokens = max_tokens

    def prepare(self, text: str, kind: EmbedKind = "document") -> str:
        """Return the exact string that will be sent to the model."""
        return prepare_embedding_text(
            text, self._prefixes[kind], self._max_tokens, logger
        )

    def embed(
        self, texts: Sequence[str], kind: EmbedKind = "document"
    ) -> list[list[float]]:
        if kind not in self._prefixes:
            raise ValueError(f"Unknown embed kind: {kind!r}")
        return self._backend.embed([self.prepare(t, kind) for t in texts])

    @property
    def dimension(self) -> int:
        return self._backend.dimension


class OpenAIEmbedder:
    """Embedding via the OpenAI ``/v1/embeddings`` API.

    Works against api.openai.com and any OpenAI-compatible server
    (e.g. Ollama, vLLM, LM Studio) by changing ``base_url``.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8080/v1",
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        self._model_name = model_name
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = (
            {"Authorization": f"Bearer {api_key}"} if api_key else None
        )
        self._timeout = timeout
        self._dimension = len(self._request(["hello"])[0])

    def _request(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self._url,
            json={"model": self._model_name, "input": texts},
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        items = response.json().get("data") or []
        if len(items) != len(texts) or not all(
            item.get("embedding") for item in items
        ):
            raise ValueError(
                f"OpenAI API returned {len(items)} embeddings for "
                f"{len(texts)} inputs (model {self._model_name})"
            )
        items = sorted(items, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._request(list(texts))

    @property
    def dimension(self) -> int:
        return self._dimension


class OnnxEmbedder:
    """Lightweight embedder using ONNX Runtime + HuggingFace tokenizers."""

    def __init__(
        self, model_name: str, max_length: int = 512
    ) -> None:
        from pathlib import Path

        import numpy as np
        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession
        from tokenizers import Tokenizer

        self._np = np

        model_dir = Path(
            hf_hub_download(
                repo_id=model_name,
                filename="tokenizer.json",
            )
        ).parent

        onnx_path = hf_hub_download(
            repo_id=model_name,
            filename="onnx/model.onnx",
        )

        self._tokenizer = Tokenizer.from_file(
            str(model_dir / "tokenizer.json")
        )
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=max_length)
        self._session = InferenceSession(onnx_path)
        self._dimension = self._detect_dimension()

    def _detect_dimension(self) -> int:
        np = self._np
        dummy = self._tokenizer.encode("hello")
        ids = np.array([[dummy.ids]], dtype=np.int64)
        mask = np.array([[dummy.attention_mask]], dtype=np.int64)
        token_type = np.zeros_like(ids)
        outputs = self._session.run(
            None,
            {
                "input_ids": ids[0],
                "attention_mask": mask[0],
                "token_type_ids": token_type[0],
            },
        )
        return int(outputs[0].shape[-1])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        np = self._np
        encoded = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array(
            [e.attention_mask for e in encoded], dtype=np.int64
        )
        token_type = np.zeros_like(ids)

        outputs = self._session.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": token_type,
            },
        )
        embeddings = _mean_pool(np, outputs[0], mask)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normalized = embeddings / norms
        return normalized.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerEmbedder:
    """Full sentence-transformers backend (requires torch)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        if self._dimension is None:
            raise ValueError(
                "Embedding model did not report a vector dimension"
            )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True
        )
        return [vector.tolist() for vector in vectors]

    @property
    def dimension(self) -> int:
        return self._dimension


def _mean_pool(np, token_embeddings, attention_mask):
    mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(
        np.float32
    )
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.maximum(np.sum(mask_expanded, axis=1), 1e-9)
    return summed / counts
