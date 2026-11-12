import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import requests

from events2rag.text_utils import prepare_embedding_text

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

type EmbedKind = Literal["query", "document"]


class EmbeddingBackend(Protocol):
    """A model: texts in, vectors out, sent exactly as given."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


class Embedder(Protocol):
    """Embeds texts as search queries or as documents to be searched."""

    def embed(
        self, texts: Sequence[str], kind: EmbedKind = "document"
    ) -> list[list[float]]: ...

    @property
    def dimension(self) -> int: ...


class PromptedEmbedder:
    """A backend plus the prompt prefixes its model was trained with.

    Many embedding models expect a task prefix on queries and/or documents
    (nomic and ModernBERT: "search_query: " / "search_document: "; e5:
    "query: " / "passage: ") and retrieve worse without it. Texts are
    truncated to `max_tokens` with the prefix counted.
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
            "query": query_prefix,
            "document": document_prefix,
        }
        self._max_tokens = max_tokens

    def prepare(self, text: str, kind: EmbedKind = "document") -> str:
        """The exact text sent to the model for `text`."""
        if kind not in self._prefixes:
            raise ValueError(f"Unknown embed kind: {kind!r}")
        return prepare_embedding_text(
            text, self._prefixes[kind], self._max_tokens, logger
        )

    def embed(
        self, texts: Sequence[str], kind: EmbedKind = "document"
    ) -> list[list[float]]:
        return self._backend.embed([self.prepare(text, kind) for text in texts])

    @property
    def dimension(self) -> int:
        return self._backend.dimension


def _is_vector(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(x, int | float) and not isinstance(x, bool) for x in value
    )


def _parse_embeddings(data: object, count: int) -> list[list[float]]:
    """The vectors from an OpenAI /embeddings response, in input order.

    The response is {"data": [{"index": 0, "embedding": [...]}, ...]};
    https://platform.openai.com/docs/api-reference/embeddings/object
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Embeddings API returned {type(data).__name__}, not an object"
        )
    items = data.get("data")
    if not isinstance(items, list) or not all(
        isinstance(item, dict)
        and isinstance(item.get("index", 0), int)
        and _is_vector(item.get("embedding"))
        and item.get("embedding")
        for item in items
    ):
        raise ValueError("Embeddings API returned data that isn't a list of embeddings")
    if len(items) != count:
        raise ValueError(
            f"Embeddings API returned {len(items)} embeddings for {count} inputs"
        )
    items = sorted(items, key=lambda item: item.get("index", 0))
    return [[float(x) for x in item["embedding"]] for item in items]


class OpenAIEmbedder:
    """Embedding via the OpenAI /embeddings API.

    Works against a local OpenAI-compatible server (text-embeddings-inference,
    vLLM, Ollama's /v1, LM Studio) or api.openai.com, chosen by base_url.
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
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
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
        try:
            return _parse_embeddings(response.json(), len(texts))
        except ValueError as error:
            raise ValueError(f"{error} (model {self._model_name})") from error

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._request(list(texts))

    @property
    def dimension(self) -> int:
        return self._dimension


class OnnxEmbedder:
    """Lightweight embedder using ONNX Runtime + HuggingFace tokenizers.

    Needs the `onnx` extra; its imports are deferred so the base install
    (OpenAI-compatible API only) doesn't need numpy or onnxruntime.
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
