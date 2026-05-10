"""SQLite-backed embedding storage and FAISS-backed search.

Vectors are written as raw float32 bytes (``vector.tobytes()``) into the
``embeddings.vector`` BLOB column — small, compact, no external file to
keep in sync with the SQLite DB. Search builds an in-memory FAISS index
per call: cheap for repos under ~10K symbols (the build is O(N) and
inner-product on normalised vectors == cosine), and avoids stale-index
bugs when re-embeddings happen between calls.
"""

from __future__ import annotations

import sqlite3

import numpy as np


def write_embedding(
    conn: sqlite3.Connection,
    *,
    symbol_id: int,
    vector: np.ndarray,
    model_name: str,
    chunk_text: str,
) -> None:
    """Insert or replace the embedding for one symbol.

    ``UNIQUE(symbol_id)`` makes this safe to call repeatedly during
    incremental re-indexing; the old vector is replaced atomically.
    """
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    conn.execute(
        """
        INSERT INTO embeddings (symbol_id, model_name, dim, vector, chunk_text)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol_id) DO UPDATE SET
            model_name = excluded.model_name,
            dim = excluded.dim,
            vector = excluded.vector,
            chunk_text = excluded.chunk_text
        """,
        (symbol_id, model_name, int(flat.shape[0]), flat.tobytes(), chunk_text),
    )


def embedding_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])


def load_all_vectors(
    conn: sqlite3.Connection,
) -> tuple[np.ndarray, list[int]]:
    """Return ``(matrix shape (N, dim), symbol_ids)`` in stable order."""
    rows = conn.execute(
        "SELECT symbol_id, dim, vector FROM embeddings ORDER BY id"
    ).fetchall()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32), []
    dim = int(rows[0]["dim"])
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    ids: list[int] = []
    for i, r in enumerate(rows):
        matrix[i] = np.frombuffer(r["vector"], dtype=np.float32)
        ids.append(int(r["symbol_id"]))
    return matrix, ids


def search(
    conn: sqlite3.Connection,
    query_vector: np.ndarray,
    *,
    k: int = 10,
) -> list[tuple[int, float]]:
    """Top-k nearest symbol ids by cosine similarity to ``query_vector``.

    Returns ``[(symbol_id, score), ...]`` sorted high-to-low. Empty list
    if the index has no embeddings.
    """
    matrix, ids = load_all_vectors(conn)
    if not ids:
        return []

    import faiss  # noqa: PLC0415 — lazy so callers without the wheel still import store

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    scores, indices = index.search(query, min(k, len(ids)))
    return [
        (ids[i], float(s))
        for s, i in zip(scores[0], indices[0], strict=True)
        if i >= 0
    ]
