from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import requests


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    @property
    def dimension(self) -> int:
        ...


class OllamaEmbedder:
    """Embedding via Ollama's /api/embed endpoint."""

    def __init__(
        self,
        model_name: str,
        ollama_url: str = "http://ollama:11434",
        timeout: int = 120,
    ) -> None:
        self._model_name = model_name
        self._url = ollama_url.rstrip("/") + "/api/embed"
        self._timeout = timeout
        self._dimension = self._probe_dimension()

    def _probe_dimension(self) -> int:
        response = requests.post(
            self._url,
            json={
                "model": self._model_name,
                "input": "hello",
                "truncate": True,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings") or data.get("embedding")
        if not embeddings or not embeddings[0]:
            raise ValueError(
                f"Ollama returned no embeddings for model "
                f"{self._model_name}"
            )
        return len(embeddings[0])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = requests.post(
            self._url,
            json={
                "model": self._model_name,
                "input": list(texts),
                "truncate": True,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings") or data.get("embedding")

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
