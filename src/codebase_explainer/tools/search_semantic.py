"""search_semantic tool: fuzzy lookup over symbol embeddings.

Different shape from the other tools — needs an :class:`Embedder` to
embed the user's query before hitting FAISS. The Agent injects the
embedder via kwargs; if it's missing or the index has no embeddings, we
return a clear error so the model knows to fall back to grep /
find_definition rather than retry blindly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from codebase_explainer.embeddings import embedding_count, search

MAX_K = 20


def handle_search_semantic(
    input: dict[str, Any],
    *,
    db_conn: sqlite3.Connection,
    repo_root: Path,  # noqa: ARG001 — kept for handler-signature uniformity
    embedder: Any = None,
    **_: Any,
) -> str:
    if embedder is None:
        return (
            "Error: search_semantic needs an embedder, which wasn't configured. "
            "Re-index with `--embed` to enable semantic search, or use "
            "find_definition / grep / view_symbol instead."
        )

    if embedding_count(db_conn) == 0:
        return (
            "Error: this index has no embeddings. Re-index with `--embed` to "
            "enable semantic search, or use find_definition / grep / view_symbol "
            "instead."
        )

    query = input["query"]
    k = min(max(1, int(input.get("k", 10) or 10)), MAX_K)

    query_vec = embedder.encode([query])[0]
    hits = search(db_conn, query_vec, k=k)
    if not hits:
        return f"No semantic matches for {query!r}."

    sym_ids = [h[0] for h in hits]
    placeholders = ",".join("?" * len(sym_ids))
    rows = db_conn.execute(
        f"""
        SELECT s.id, s.qualified_name, s.kind, s.signature, s.docstring,
               s.start_line, f.path
        FROM symbols s
        JOIN files f ON s.file_id = f.id
        WHERE s.id IN ({placeholders})
        """,
        sym_ids,
    ).fetchall()
    sym_by_id = {r["id"]: r for r in rows}

    parts = [f"# Top {len(hits)} semantic match(es) for {query!r}"]
    for sym_id, score in hits:
        r = sym_by_id.get(sym_id)
        if r is None:  # symbol deleted between embed and search; skip
            continue
        parts.append(
            f"  [{score:+.3f}] [{r['kind']}] {r['qualified_name']}  "
            f"({r['path']}:{r['start_line']})"
        )
        if r["signature"]:
            parts.append(f"    {r['signature']}")
        if r["docstring"]:
            doc = r["docstring"].strip().replace("\n", " ")
            if len(doc) > 140:
                doc = doc[:137] + "..."
            parts.append(f"    {doc}")
    return "\n".join(parts)
