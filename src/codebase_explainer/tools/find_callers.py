"""find_callers tool: find every call site of a symbol."""

from __future__ import annotations

import sqlite3
from typing import Any

MAX_RESULTS = 50


def handle_find_callers(
    input: dict[str, Any], *, db_conn: sqlite3.Connection, **_: Any
) -> str:
    name = input["name"]

    # Resolved callers — calls.callee_id was wired up to a real symbol
    # by the resolution pass. These are reliable.
    resolved = db_conn.execute(
        """
        SELECT c.caller_qualified_name, f.path, c.line, s.qualified_name AS callee_qn
        FROM calls c
        JOIN symbols s ON c.callee_id = s.id
        JOIN files f ON c.file_id = f.id
        WHERE s.qualified_name = ?
           OR s.name = ?
           OR s.qualified_name LIKE '%.' || ?
        ORDER BY f.path, c.line
        LIMIT ?
        """,
        (name, name, name, MAX_RESULTS),
    ).fetchall()

    # Textual fallback — calls we couldn't resolve, but whose callee_name
    # matches by string. Likely external (stdlib, third-party) but might
    # also be in-repo calls the resolver missed (relative imports, deep
    # attribute chains). Surfaced separately so the agent doesn't conflate
    # them with confirmed call sites.
    textual = db_conn.execute(
        """
        SELECT c.caller_qualified_name, f.path, c.line, c.callee_name
        FROM calls c
        JOIN files f ON c.file_id = f.id
        WHERE c.callee_id IS NULL
          AND (c.callee_name = ? OR c.callee_name LIKE '%.' || ?)
        ORDER BY f.path, c.line
        LIMIT ?
        """,
        (name, name, MAX_RESULTS),
    ).fetchall()

    if not resolved and not textual:
        return f"No callers found for {name!r}."

    parts: list[str] = []

    if resolved:
        parts.append(f"# {len(resolved)} resolved caller(s) of {name!r}")
        for r in resolved:
            caller = r["caller_qualified_name"] or "<module>"
            parts.append(f"  {caller} -> {r['callee_qn']}  ({r['path']}:{r['line']})")

    if textual:
        if parts:
            parts.append("")
        parts.append(
            f"# {len(textual)} textual match(es) (unresolved — could be external "
            f"libraries with the same name)"
        )
        for r in textual:
            caller = r["caller_qualified_name"] or "<module>"
            parts.append(f"  {caller} -> {r['callee_name']}  ({r['path']}:{r['line']})")

    return "\n".join(parts)
