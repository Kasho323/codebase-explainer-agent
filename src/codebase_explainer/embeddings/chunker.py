"""Convert one symbol into the text we hand to the embedder.

The shape is intentionally simple — a four-line header (kind +
qualified_name + signature + docstring) followed by the source body.
This puts the most semantically dense content first so it lands inside
the model's truncation window even for long methods.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# all-MiniLM-L6-v2 truncates around 256 tokens (~1024 chars). We stay a
# touch under to leave room for tokeniser overhead without losing the
# header.
MAX_CHUNK_CHARS = 1500


def chunk_for_symbol(
    conn: sqlite3.Connection,
    symbol_id: int,
    *,
    repo_root: Path,
) -> str | None:
    """Return the embeddable text for one symbol, or ``None`` if missing.

    ``None`` means either the symbol id doesn't exist or the source file
    can't be decoded — caller should skip the symbol rather than write
    an empty embedding.
    """
    sym = conn.execute(
        """
        SELECT s.qualified_name, s.kind, s.signature, s.docstring,
               s.start_line, s.end_line, f.path
        FROM symbols s
        JOIN files f ON s.file_id = f.id
        WHERE s.id = ?
        """,
        (symbol_id,),
    ).fetchone()

    if sym is None:
        return None

    parts: list[str] = [f"{sym['kind']} {sym['qualified_name']}"]
    if sym["signature"]:
        parts.append(sym["signature"])
    if sym["docstring"]:
        parts.append(sym["docstring"])

    body = _read_body(repo_root, sym["path"], sym["start_line"], sym["end_line"])
    if body:
        parts.append("")
        parts.append(body)

    chunk = "\n".join(parts)
    if len(chunk) > MAX_CHUNK_CHARS:
        chunk = chunk[:MAX_CHUNK_CHARS]
    return chunk


def _read_body(
    repo_root: Path, rel_path: str, start: int, end: int
) -> str | None:
    full = repo_root / rel_path
    if not full.is_file():
        return None
    try:
        text = full.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    lines = text.splitlines()
    body_lines = lines[start - 1 : end]
    return "\n".join(body_lines)
