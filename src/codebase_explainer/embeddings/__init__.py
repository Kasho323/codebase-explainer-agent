"""Symbol-level embeddings: chunking, persistence, and FAISS search.

Public surface:
    - :class:`Embedder` — protocol; any class with ``model_name``, ``dim``,
      and ``encode(list[str]) -> ndarray`` qualifies.
    - :class:`FakeEmbedder` — deterministic char-hash embedder for tests.
    - :class:`SentenceTransformerEmbedder` — wraps
      ``sentence-transformers/all-MiniLM-L6-v2`` (or any compatible model).
    - :func:`chunk_for_symbol` — turns one symbol into the text we embed.
    - :func:`write_embedding` / :func:`load_all_vectors` / :func:`search` —
      SQLite-backed storage and an in-memory FAISS index built per query.
"""

from __future__ import annotations

from codebase_explainer.embeddings.chunker import chunk_for_symbol
from codebase_explainer.embeddings.embedder import (
    Embedder,
    FakeEmbedder,
    SentenceTransformerEmbedder,
)
from codebase_explainer.embeddings.store import (
    embedding_count,
    load_all_vectors,
    search,
    write_embedding,
)

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "SentenceTransformerEmbedder",
    "chunk_for_symbol",
    "embedding_count",
    "load_all_vectors",
    "search",
    "write_embedding",
]
