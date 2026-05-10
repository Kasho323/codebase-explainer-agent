"""Embedder protocol + two implementations.

The protocol exists so tests can swap in :class:`FakeEmbedder` (no torch,
no model download, no network) without monkey-patching. The real model
import is deferred until :class:`SentenceTransformerEmbedder` is actually
constructed, so importing this module is cheap and doesn't pull in
``sentence_transformers``/``torch`` for callers that only need the fake.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    pass


class Embedder(Protocol):
    """Encodes one or more strings into L2-normalised dense vectors."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 array, L2-normalised."""
        ...


class FakeEmbedder:
    """Deterministic char-hash embedder. Used in tests and as a fallback.

    Every character of every input contributes to a fixed-size accumulator
    indexed by ``ord(char) % dim``. Different strings produce different
    vectors (almost always), and the same string always produces the same
    vector — that's the only contract tests need.
    """

    DIM = 16
    _MODEL_NAME = "fake/deterministic-hash-v1"

    @property
    def model_name(self) -> str:
        return self._MODEL_NAME

    @property
    def dim(self) -> int:
        return self.DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for c in text:
                out[i, ord(c) % self.DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """Wraps any sentence-transformers model. Lazy-imports torch/ST."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        # Lazy import: this module loads cleanly in environments without
        # torch (CI tests use FakeEmbedder).
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False)
