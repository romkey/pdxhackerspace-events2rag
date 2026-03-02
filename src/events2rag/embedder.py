from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    @property
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        if self._dimension is None:
            raise ValueError("Embedding model did not report a vector dimension")

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    @property
    def dimension(self) -> int:
        return self._dimension

