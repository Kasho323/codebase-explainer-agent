"""Resolve textual callee names to ``symbols.id`` references.

Runs as a second pass after the indexer has populated ``symbols``,
``imports``, and ``calls`` for every file. Updates each call row's
``callee_id`` whenever the textual ``callee_name`` can be matched
unambiguously to a symbol in the index.

Resolution rules, applied in order. The first rule that matches a real
symbol wins; if a rule "claims" a callee but the candidate isn't in the
index (e.g. an external library call), we stop and leave ``callee_id``
NULL rather than fall through to a less specific rule.

    1. ``self.X`` / ``cls.X`` inside a method        -> {enclosing_class}.X
    2. Head matches an import alias                  -> {import_target}.{rest}
    3. Bare or dotted callee local to the file       -> {file_prefix}.{callee}
    4. Callee already happens to be a fully qualified
       name in the index                             -> {callee}

Intentional limitations (documented for the future-work pass):
    - Relative imports (``from . import X``, ``from ..pkg import Y``)
      are skipped. Their absolute target would require knowing the
      file's package depth, which is doable but not yet implemented.
    - Wildcard imports (``from X import *``) are skipped.
    - Deep attribute chains (``a.b.c.d()``) only resolve when the head
      directly matches an import alias; intermediate object types are
      not tracked.
"""

from __future__ import annotations

import sqlite3


def resolve_callees(conn: sqlite3.Connection) -> int:
    """Resolve textual callee names into symbol IDs.

    Returns the number of call rows that were newly resolved.
    """
    qn_to_id: dict[str, int] = {
        row[0]: row[1] for row in conn.execute("SELECT qualified_name, id FROM symbols")
    }
    qn_to_kind: dict[str, str] = {
        row[0]: row[1] for row in conn.execute("SELECT qualified_name, kind FROM symbols")
    }

    resolved = 0
    files = list(conn.execute("SELECT id, path FROM files"))

    for file_row in files:
        file_id = file_row[0]
        path = file_row[1]
        prefix = _prefix_from_path(path)
        alias_map = _build_alias_map(conn, file_id)

        calls = conn.execute(
            "SELECT id, caller_qualified_name, callee_name FROM calls "
            "WHERE file_id = ? AND callee_id IS NULL",
            (file_id,),
        ).fetchall()

        for call_id, caller_qn, callee in calls:
            target_qn = _try_resolve(
                callee=callee,
                caller_qn=caller_qn,
                file_prefix=prefix,
                alias_map=alias_map,
                qn_to_kind=qn_to_kind,
            )
            if target_qn is not None and target_qn in qn_to_id:
                conn.execute(
                    "UPDATE calls SET callee_id = ? WHERE id = ?",
                    (qn_to_id[target_qn], call_id),
                )
                resolved += 1

    conn.commit()
    return resolved


def _prefix_from_path(path: str) -> str:
    """Mirror ``repo_walker.relative_module_prefix`` but operate on stored
    POSIX paths rather than Path objects."""
    parts = path.split("/")
    if not parts:
        return ""
    last = parts[-1]
    if last == "__init__.py":
        parts = parts[:-1]
    elif last.endswith(".py"):
        parts[-1] = last[:-3]
    return ".".join(parts)


def _build_alias_map(conn: sqlite3.Connection, file_id: int) -> dict[str, str]:
    """For one file's imports, build ``{local_name: target_qualified_name}``.

    Examples:
        ``import os``                       -> {"os": "os"}
        ``import os.path as p``             -> {"p": "os.path"}
        ``from lib import tools``           -> {"tools": "lib.tools"}
        ``from lib import tools as t``      -> {"t": "lib.tools"}
        ``from lib.tools import hammer``    -> {"hammer": "lib.tools.hammer"}
    """
    alias_map: dict[str, str] = {}
    rows = conn.execute(
        "SELECT module, name, alias FROM imports WHERE file_id = ?", (file_id,)
    )
    for module, name, alias in rows:
        if module.startswith("."):
            continue  # relative imports — see module docstring
        if name is None:
            # ``import X`` / ``import X.Y`` / ``import X.Y as Z``
            if alias:
                alias_map[alias] = module
            else:
                top = module.split(".")[0]
                alias_map[top] = top
        elif name == "*":
            continue  # wildcard imports — see module docstring
        else:
            local = alias if alias else name
            alias_map[local] = f"{module}.{name}"
    return alias_map


def _enclosing_class(
    caller_qn: str | None, qn_to_kind: dict[str, str]
) -> str | None:
    """Walk up ``caller_qn`` and return the longest prefix that's a class.

    For ``caller_qn="myapp.models.User.save"`` this returns
    ``"myapp.models.User"``. For top-level functions it returns None.
    """
    if not caller_qn:
        return None
    parts = caller_qn.split(".")
    parts.pop()  # strip the leaf (the caller itself)
    while parts:
        candidate = ".".join(parts)
        if qn_to_kind.get(candidate) == "class":
            return candidate
        parts.pop()
    return None


def _try_resolve(
    *,
    callee: str,
    caller_qn: str | None,
    file_prefix: str,
    alias_map: dict[str, str],
    qn_to_kind: dict[str, str],
) -> str | None:
    parts = callee.split(".")
    head = parts[0]
    rest = ".".join(parts[1:])

    # Rule 1: self.X / cls.X inside a method.
    if head in ("self", "cls"):
        class_qn = _enclosing_class(caller_qn, qn_to_kind)
        if class_qn:
            cand = f"{class_qn}.{rest}" if rest else class_qn
            if cand in qn_to_kind:
                return cand
        return None  # don't fall through — self/cls scope is unambiguous

    # Rule 2: head matches an import alias.
    if head in alias_map:
        target_root = alias_map[head]
        cand = f"{target_root}.{rest}" if rest else target_root
        if cand in qn_to_kind:
            return cand
        return None  # don't fall through — alias namespace is unambiguous

    # Rule 3: bare or dotted callee local to the file.
    cand = f"{file_prefix}.{callee}" if file_prefix else callee
    if cand in qn_to_kind:
        return cand

    # Rule 4: a fully-qualified callee that happens to match.
    if callee in qn_to_kind:
        return callee

    return None
