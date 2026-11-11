from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import requests

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


def _parse_embeddings(data: object) -> list[list[float]]:
    """The vectors from an Ollama /api/embed response.

    Ollama returns {"embeddings": [[...], ...]}; older versions used
    "embedding". https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    if not isinstance(data, dict):
        raise ValueError(f"Ollama returned {type(data).__name__}, not an object")
    key = "embeddings" if "embeddings" in data else "embedding"
    embeddings = data.get(key)
    if not isinstance(embeddings, list) or not all(
        isinstance(vector, list) and all(isinstance(x, int | float) for x in vector)
        for vector in embeddings
    ):
        raise ValueError("Ollama returned embeddings that aren't lists of numbers")
    return [[float(x) for x in vector] for vector in embeddings]


class OllamaEmbedder:
    """Embedding via Ollama's /api/embed endpoint."""

    def __init__(
        self,
        model_name: str,
        ollama_url: str = "http://ollama:11434",
        api_key: str | None = None,
        timeout: int = 120,
    ) -> None:
        self._model_name = model_name
        self._url = ollama_url.rstrip("/") + "/api/embed"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._timeout = timeout
        self._dimension = self._probe_dimension()

    def _request(self, texts: str | list[str]) -> list[list[float]]:
        response = requests.post(
            self._url,
            json={"model": self._model_name, "input": texts, "truncate": True},
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return _parse_embeddings(response.json())

    def _probe_dimension(self) -> int:
        embeddings = self._request("hello")
        if not embeddings or not embeddings[0]:
            raise ValueError(
                f"Ollama returned no embeddings for model {self._model_name}"
            )
        return len(embeddings[0])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._request(list(texts))

    @property
    def dimension(self) -> int:
        return self._dimension


class OnnxEmbedder:
    """Lightweight embedder using ONNX Runtime + HuggingFace tokenizers.

    Needs the `onnx` extra; its imports are deferred so the base install
    (Ollama only) doesn't need numpy or onnxruntime.
    """

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        import numpy as np
        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession
        from tokenizers import Tokenizer

        self._np = np

        model_dir = Path(
            hf_hub_download(repo_id=model_name, filename="tokenizer.json")
        ).parent
        onnx_path = hf_hub_download(repo_id=model_name, filename="onnx/model.onnx")

        # huggingface_hub, onnxruntime and tokenizers aren't type-checked
        # (see pyproject.toml), so these are Any.
        self._tokenizer: Any = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=max_length)
        self._session: Any = InferenceSession(onnx_path)
        self._dimension = self._detect_dimension()

    def _run(
        self, ids: NDArray[np.int64], mask: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        outputs = self._session.run(
            None,
            {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": self._np.zeros_like(ids),
            },
        )
        token_embeddings: NDArray[np.float32] = outputs[0]
        return token_embeddings

    def _detect_dimension(self) -> int:
        np = self._np
        dummy = self._tokenizer.encode("hello")
        ids = np.array([dummy.ids], dtype=np.int64)
        mask = np.array([dummy.attention_mask], dtype=np.int64)
        return int(self._run(ids, mask).shape[-1])

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        np = self._np
        encoded = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        embeddings = _mean_pool(self._run(ids, mask), mask)
        norms = np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        vectors: list[list[float]] = (embeddings / norms).tolist()
        return vectors

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerEmbedder:
    """Full sentence-transformers backend (needs the `torch` extra)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        dimension: int | None = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise ValueError("Embedding model did not report a vector dimension")
        self._dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]

    @property
    def dimension(self) -> int:
        return self._dimension


def _mean_pool(
    token_embeddings: NDArray[np.float32], attention_mask: NDArray[np.int64]
) -> NDArray[np.float32]:
    """Average each text's token vectors, ignoring padding."""
    import numpy as np

    mask = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
    summed = np.sum(token_embeddings * mask, axis=1)
    counts = np.maximum(np.sum(mask, axis=1), 1e-9)
    pooled: NDArray[np.float32] = (summed / counts).astype(np.float32)
    return pooled
