"""Persist a FileIndex into the SQLite symbol-graph database.

Re-indexing a file is idempotent: the previous record is deleted (cascading
to symbols, imports, and calls) and re-inserted. Caller resolution
(filling ``calls.callee_id``) is intentionally deferred to a later pass —
this layer only writes textual ``callee_name``.
"""

from __future__ import annotations

import hashlib
import sqlite3

from codebase_explainer.indexer import FileIndex


def hash_source(source: str | bytes) -> str:
    """SHA-1 hex digest of source bytes; used as a cheap change detector."""
    if isinstance(source, str):
        source = source.encode("utf-8")
    return hashlib.sha1(source).hexdigest()


def write_file_index(
    conn: sqlite3.Connection,
    *,
    path: str,
    file_index: FileIndex,
    language: str = "python",
    content_hash: str | None = None,
) -> int:
    """Replace any existing record for ``path`` with the given index.

    Returns the new ``files.id``.
    """
    conn.execute("DELETE FROM files WHERE path = ?", (path,))

    cursor = conn.execute(
        "INSERT INTO files (path, language, content_hash) VALUES (?, ?, ?)",
        (path, language, content_hash),
    )
    file_id = cursor.lastrowid
    assert file_id is not None  # SQLite always populates lastrowid after INSERT

    # Symbols come out of extract_file in document order, so a parent symbol
    # is always inserted before its children. We map the in-memory Symbol
    # object identity to its newly-assigned row id and look up parents by
    # identity rather than qualified name (cheaper, and unaffected by name
    # collisions across nested defs).
    sym_to_row_id: dict[int, int] = {}
    for sym in file_index.symbols:
        parent_row_id = sym_to_row_id.get(id(sym.parent)) if sym.parent else None
        cur = conn.execute(
            """
            INSERT INTO symbols
                (file_id, kind, name, qualified_name, parent_id,
                 start_line, end_line, signature, docstring)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                sym.kind,
                sym.name,
                sym.qualified_name,
                parent_row_id,
                sym.start_line,
                sym.end_line,
                sym.signature,
                sym.docstring,
            ),
        )
        sym_to_row_id[id(sym)] = cur.lastrowid

    for imp in file_index.imports:
        conn.execute(
            "INSERT INTO imports (file_id, module, name, alias, line) VALUES (?, ?, ?, ?, ?)",
            (file_id, imp.module, imp.name, imp.alias, imp.line),
        )

    for call in file_index.calls:
        # Empty caller (module-level) stored as NULL for SQL-friendly queries.
        caller = call.caller_qualified_name or None
        conn.execute(
            """
            INSERT INTO calls (file_id, caller_qualified_name, callee_name, line)
            VALUES (?, ?, ?, ?)
            """,
            (file_id, caller, call.callee_name, call.line),
        )

    return file_id
