"""Orchestrator: walk a repo, parse each file, persist into SQLite.

Optionally runs a third pass that embeds every symbol — controlled by
the ``embedder`` parameter so callers without sentence-transformers /
torch installed (e.g. CI) can index without the heavy ML deps.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codebase_explainer.embeddings import chunk_for_symbol, write_embedding
from codebase_explainer.indexer import extract_file
from codebase_explainer.persistence import hash_source, write_file_index
from codebase_explainer.repo_walker import relative_module_prefix, walk_python_files
from codebase_explainer.resolver import resolve_callees
from codebase_explainer.schema import connect, init_db


@dataclass
class IndexStats:
    files: int = 0
    symbols: int = 0
    calls: int = 0
    imports: int = 0
    skipped: int = 0
    resolved_calls: int = 0
    embedded_symbols: int = 0


def index_repo(
    repo_root: str | Path,
    db_path: str | Path,
    *,
    embedder: Any = None,
) -> IndexStats:
    """Walk a Python repo, parse every ``.py`` file, write the index to SQLite.

    If ``embedder`` is provided, runs an embedding pass over every
    extracted symbol after the resolver. The embedder must satisfy the
    :class:`Embedder` protocol (``model_name``, ``dim``, ``encode``).
    """
    repo_root = Path(repo_root).resolve()
    init_db(db_path)
    stats = IndexStats()

    with connect(db_path) as conn:
        for py_file in walk_python_files(repo_root):
            try:
                source = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                stats.skipped += 1
                continue

            prefix = relative_module_prefix(py_file, repo_root)
            file_index = extract_file(source, prefix=prefix)

            try:
                rel_path = str(py_file.relative_to(repo_root))
            except ValueError:
                rel_path = str(py_file)
            # Normalise to forward slashes so paths look the same on Windows
            # and Linux — important for portable diffs and recruiter-friendly
            # SQL output.
            rel_path = rel_path.replace("\\", "/")

            write_file_index(
                conn,
                path=rel_path,
                file_index=file_index,
                content_hash=hash_source(source),
            )

            stats.files += 1
            stats.symbols += len(file_index.symbols)
            stats.calls += len(file_index.calls)
            stats.imports += len(file_index.imports)

        conn.commit()
        stats.resolved_calls = resolve_callees(conn)

        if embedder is not None:
            stats.embedded_symbols = _embed_all_symbols(conn, embedder, repo_root)

    return stats


def _embed_all_symbols(
    conn: sqlite3.Connection,
    embedder: Any,
    repo_root: Path,
) -> int:
    """Build a chunk for every symbol, embed in one batch, persist."""
    symbol_ids = [row["id"] for row in conn.execute("SELECT id FROM symbols")]
    if not symbol_ids:
        return 0

    chunks: dict[int, str] = {}
    for sid in symbol_ids:
        chunk = chunk_for_symbol(conn, sid, repo_root=repo_root)
        if chunk is not None:
            chunks[sid] = chunk

    if not chunks:
        return 0

    ids = list(chunks)
    texts = [chunks[i] for i in ids]
    vectors = embedder.encode(texts)

    for sid, vec in zip(ids, vectors, strict=True):
        write_embedding(
            conn,
            symbol_id=sid,
            vector=vec,
            model_name=embedder.model_name,
            chunk_text=chunks[sid],
        )
    conn.commit()
    return len(ids)
