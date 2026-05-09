"""view_symbol tool: comprehensive lookup for one symbol.

Returns identity, signature, docstring, source body, parent, callers,
and callees in a single response. Designed so the agent can answer
"explain X" with one tool call instead of chaining
``find_definition`` + ``read_file`` + ``find_callers``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

# Bound the context budget. A symbol body longer than this is almost always
# a class — the agent can read the methods it cares about with read_file.
MAX_SOURCE_LINES = 200
MAX_CALLERS = 20
MAX_TEXTUAL_CALLERS = 10
MAX_CALLEES = 30
DOCSTRING_TRUNCATE = 500


def handle_view_symbol(
    input: dict[str, Any],
    *,
    db_conn: sqlite3.Connection,
    repo_root: Path,
    **_: Any,
) -> str:
    name = input["name"]
    sym = _lookup_symbol(db_conn, name)
    if sym is None:
        return (
            f"No symbol found matching {name!r}. "
            "Try find_definition first to discover candidate names."
        )

    parts: list[str] = [
        f"# [{sym['kind']}] {sym['qualified_name']}",
        f"  Location: {sym['path']}:{sym['start_line']}-{sym['end_line']}",
    ]

    parent_line = _format_parent(db_conn, sym["parent_id"])
    if parent_line:
        parts.append(parent_line)

    if sym["signature"]:
        parts.extend(["", f"  {sym['signature']}"])

    if sym["docstring"]:
        parts.extend(_format_docstring(sym["docstring"]))

    parts.extend(_format_source(repo_root, sym))
    parts.extend(_format_callers(db_conn, sym))
    parts.extend(_format_callees(db_conn, sym["qualified_name"]))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def _lookup_symbol(conn: sqlite3.Connection, name: str) -> sqlite3.Row | None:
    """Find the most specific symbol matching ``name``.

    Tries fully-qualified name first, then qualified suffix (``A.b`` matches
    ``mod.A.b``), then bare name. Picks the shortest qualified_name on ties so
    the agent gets the top-level definition rather than a nested namesake.
    """
    return conn.execute(
        """
        SELECT s.id, s.qualified_name, s.kind, s.signature, s.docstring,
               s.start_line, s.end_line, s.parent_id, f.path
        FROM symbols s
        JOIN files f ON s.file_id = f.id
        WHERE s.qualified_name = ?
           OR s.qualified_name LIKE '%.' || ?
           OR s.name = ?
        ORDER BY LENGTH(s.qualified_name)
        LIMIT 1
        """,
        (name, name, name),
    ).fetchone()


def _format_parent(conn: sqlite3.Connection, parent_id: int | None) -> str | None:
    if parent_id is None:
        return None
    row = conn.execute(
        "SELECT qualified_name, kind FROM symbols WHERE id = ?",
        (parent_id,),
    ).fetchone()
    if row is None:
        return None
    return f"  Parent:   [{row['kind']}] {row['qualified_name']}"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _format_docstring(docstring: str) -> list[str]:
    text = docstring.strip()
    if len(text) > DOCSTRING_TRUNCATE:
        text = text[: DOCSTRING_TRUNCATE - 3] + "..."
    out = ["", "  Docstring:"]
    out.extend(f"    {line}" for line in text.splitlines())
    return out


def _format_source(repo_root: Path, sym: sqlite3.Row) -> list[str]:
    full_path = repo_root / sym["path"]
    if not full_path.is_file():
        return ["", f"  [source unavailable: file not found at {sym['path']}]"]
    try:
        source_lines = full_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        return ["", f"  [could not read source: {type(e).__name__}: {e}]"]

    start = sym["start_line"]
    requested_end = sym["end_line"]
    end = min(requested_end, start + MAX_SOURCE_LINES - 1)
    selected = source_lines[start - 1 : end]
    width = len(str(end))

    out = ["", f"  Source ({sym['path']}:{start}-{end}):"]
    out.extend(f"    {i:>{width}}  {line}" for i, line in enumerate(selected, start=start))
    if end < requested_end:
        truncated_count = requested_end - end
        out.append(f"    ... ({truncated_count} more lines truncated; use read_file for the full body)")
    return out


def _format_callers(conn: sqlite3.Connection, sym: sqlite3.Row) -> list[str]:
    """Resolved callers (callee_id JOIN) plus textual matches as a fallback.

    We surface them separately so the agent doesn't conflate confirmed call
    sites with same-named external functions — same convention as
    find_callers."""
    resolved = conn.execute(
        """
        SELECT c.caller_qualified_name, f.path, c.line
        FROM calls c
        JOIN files f ON c.file_id = f.id
        WHERE c.callee_id = ?
        ORDER BY f.path, c.line
        LIMIT ?
        """,
        (sym["id"], MAX_CALLERS),
    ).fetchall()

    short = sym["qualified_name"].rsplit(".", 1)[-1]
    textual = conn.execute(
        """
        SELECT c.caller_qualified_name, f.path, c.line, c.callee_name
        FROM calls c
        JOIN files f ON c.file_id = f.id
        WHERE c.callee_id IS NULL
          AND (c.callee_name = ? OR c.callee_name LIKE '%.' || ?)
        ORDER BY f.path, c.line
        LIMIT ?
        """,
        (short, short, MAX_TEXTUAL_CALLERS),
    ).fetchall()

    out: list[str] = []
    if resolved:
        out.extend(["", f"  Called by ({len(resolved)}):"])
        for r in resolved:
            caller = r["caller_qualified_name"] or "<module>"
            out.append(f"    {caller}  ({r['path']}:{r['line']})")
    if textual:
        out.extend(["", f"  Possible textual callers ({len(textual)}, unresolved — may be external):"])
        for r in textual:
            caller = r["caller_qualified_name"] or "<module>"
            out.append(f"    {caller} -> {r['callee_name']}  ({r['path']}:{r['line']})")
    return out


def _format_callees(conn: sqlite3.Connection, qualified_name: str) -> list[str]:
    """All calls made FROM this symbol's body, in source order."""
    rows = conn.execute(
        """
        SELECT c.callee_name, c.line, c.callee_id, s.qualified_name AS resolved_qn
        FROM calls c
        LEFT JOIN symbols s ON c.callee_id = s.id
        WHERE c.caller_qualified_name = ?
        ORDER BY c.line
        LIMIT ?
        """,
        (qualified_name, MAX_CALLEES),
    ).fetchall()

    if not rows:
        return []

    out = ["", f"  Calls ({len(rows)}):"]
    for r in rows:
        target = r["resolved_qn"] or f"{r['callee_name']} (unresolved)"
        out.append(f"    L{r['line']:>4}: {target}")
    return out
