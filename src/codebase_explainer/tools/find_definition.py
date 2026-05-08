"""find_definition tool: look up where a symbol is defined."""

from __future__ import annotations

import sqlite3
from typing import Any

MAX_RESULTS = 20


def handle_find_definition(
    input: dict[str, Any], *, db_conn: sqlite3.Connection, **_: Any
) -> str:
    name = input["name"]
    rows = db_conn.execute(
        """
        SELECT s.qualified_name, s.kind, s.signature, s.docstring,
               s.start_line, s.end_line, f.path
        FROM symbols s
        JOIN files f ON s.file_id = f.id
        WHERE s.qualified_name = ?
           OR s.name = ?
           OR s.qualified_name LIKE '%.' || ?
        ORDER BY LENGTH(s.qualified_name)
        LIMIT ?
        """,
        (name, name, name, MAX_RESULTS),
    ).fetchall()

    if not rows:
        return f"No symbol found matching {name!r}."

    lines = [f"# {len(rows)} definition(s) for {name!r}"]
    for r in rows:
        lines.append(
            f"  [{r['kind']}] {r['qualified_name']}  "
            f"({r['path']}:{r['start_line']}-{r['end_line']})"
        )
        if r["signature"]:
            lines.append(f"    {r['signature']}")
        if r["docstring"]:
            doc = r["docstring"].strip().replace("\n", " ")
            if len(doc) > 200:
                doc = doc[:197] + "..."
            lines.append(f'    """{doc}"""')
        lines.append("")
    return "\n".join(lines).rstrip()
