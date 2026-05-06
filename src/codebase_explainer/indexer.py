"""Tree-sitter Python indexer.

Walks Python source and extracts a flat record of what the agent's tools
need to query against:

- Symbols: every function, class, and method (with parent links for
  nested defs and dotted qualified names).
- Calls: every function/method call site, attributed to the enclosing
  symbol (or the module, when called at top level).
- Imports: every ``import`` and ``from ... import`` statement, capturing
  module, name, and alias.

Persistence into SQLite happens in a separate layer; this module just
produces dataclasses.

Known limitations of this iteration (intentional, see Week 2 roadmap):
- Default arg values are not walked, so calls that only appear in
  defaults are not captured.
- Decorator calls are captured at the *enclosing* scope, which matches
  Python's evaluation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

PY_LANGUAGE = Language(tspython.language())


@dataclass
class Symbol:
    """A function, class, or method extracted from source.

    Lines are 1-indexed and inclusive on both ends, matching what an editor
    would show. ``qualified_name`` is dot-joined, e.g. ``app.auth.User.save``.
    """

    kind: str  # 'function' | 'class' | 'method'
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    parent: Symbol | None = field(default=None, repr=False)


@dataclass
class Call:
    """A single call site.

    ``caller_qualified_name`` is the qualified name of the enclosing
    function/method, or ``""`` for module-level calls (e.g. decorator
    arguments). ``callee_name`` is the textual form of what is being
    called: ``"foo"``, ``"self.save"``, ``"os.path.join"``.
    """

    caller_qualified_name: str
    callee_name: str
    line: int


@dataclass
class Import:
    """A single imported name.

    Examples:
        ``import os``                  -> module='os',      name=None,    alias=None
        ``import os.path as p``        -> module='os.path', name=None,    alias='p'
        ``from os.path import join``   -> module='os.path', name='join',  alias=None
        ``from os.path import join as j`` -> module='os.path', name='join', alias='j'
        ``from . import config``       -> module='.',       name='config', alias=None
        ``from os import *``           -> module='os',      name='*',     alias=None
    """

    module: str
    name: str | None
    alias: str | None
    line: int


@dataclass
class FileIndex:
    """Everything extracted from one source file."""

    symbols: list[Symbol] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)


def extract_file(source: str | bytes, prefix: str = "") -> FileIndex:
    """Parse Python source and return symbols, calls, and imports.

    ``prefix`` is an optional module path (e.g. ``"app.auth"``) used as
    the root of qualified names.
    """
    source_bytes = source.encode("utf-8") if isinstance(source, str) else source
    tree = _parser().parse(source_bytes)
    out = FileIndex()
    _walk(tree.root_node, source_bytes, parent=None, prefix=prefix, out=out)
    return out


def extract_symbols(source: str | bytes, prefix: str = "") -> list[Symbol]:
    """Backwards-compatible shortcut: only return symbols."""
    return extract_file(source, prefix).symbols


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _parser() -> Parser:
    return Parser(PY_LANGUAGE)


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _walk(
    node: Node,
    source: bytes,
    parent: Symbol | None,
    prefix: str,
    out: FileIndex,
) -> None:
    for child in node.named_children:
        if child.type == "function_definition":
            sym = _make_function(child, source, parent=parent, prefix=prefix)
            out.symbols.append(sym)
            body = child.child_by_field_name("body")
            if body is not None:
                _walk(body, source, parent=sym, prefix=prefix, out=out)

        elif child.type == "class_definition":
            sym = _make_class(child, source, parent=parent, prefix=prefix)
            out.symbols.append(sym)
            body = child.child_by_field_name("body")
            if body is not None:
                _walk(body, source, parent=sym, prefix=prefix, out=out)

        elif child.type == "call":
            callee = _callee_name(child, source)
            if callee is not None:
                out.calls.append(
                    Call(
                        caller_qualified_name=parent.qualified_name if parent else "",
                        callee_name=callee,
                        line=child.start_point[0] + 1,
                    )
                )
            # Recurse so calls nested inside arguments (f(g())) are caught.
            _walk(child, source, parent=parent, prefix=prefix, out=out)

        elif child.type == "import_statement":
            out.imports.extend(_parse_import(child, source))

        elif child.type == "import_from_statement":
            out.imports.extend(_parse_import_from(child, source))

        else:
            _walk(child, source, parent=parent, prefix=prefix, out=out)


# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------


def _make_function(
    node: Node, source: bytes, *, parent: Symbol | None, prefix: str
) -> Symbol:
    name = _node_text(node.child_by_field_name("name"), source)
    body = node.child_by_field_name("body")
    kind = "method" if parent and parent.kind == "class" else "function"
    return Symbol(
        kind=kind,
        name=name,
        qualified_name=_qualify(name, parent, prefix),
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        signature=_signature(node, source),
        docstring=_docstring(body, source) if body is not None else None,
        parent=parent,
    )


def _make_class(
    node: Node, source: bytes, *, parent: Symbol | None, prefix: str
) -> Symbol:
    name = _node_text(node.child_by_field_name("name"), source)
    body = node.child_by_field_name("body")
    return Symbol(
        kind="class",
        name=name,
        qualified_name=_qualify(name, parent, prefix),
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        signature=f"class {name}",
        docstring=_docstring(body, source) if body is not None else None,
        parent=parent,
    )


def _signature(func_node: Node, source: bytes) -> str:
    name_node = func_node.child_by_field_name("name")
    params_node = func_node.child_by_field_name("parameters")
    return_node = func_node.child_by_field_name("return_type")
    if name_node is None or params_node is None:
        return ""
    parts = ["def ", _node_text(name_node, source), _node_text(params_node, source)]
    if return_node is not None:
        parts.append(" -> ")
        parts.append(_node_text(return_node, source))
    return "".join(parts)


def _docstring(body: Node, source: bytes) -> str | None:
    for child in body.named_children:
        if child.type != "expression_statement":
            return None
        for sub in child.named_children:
            if sub.type == "string":
                return _strip_quotes(_node_text(sub, source))
        return None
    return None


def _strip_quotes(raw: str) -> str:
    text = raw.strip()
    for quote in ('"""', "'''", '"', "'"):
        if text.startswith(quote) and text.endswith(quote) and len(text) >= 2 * len(quote):
            return text[len(quote) : -len(quote)].strip()
    return text


def _qualify(name: str, parent: Symbol | None, prefix: str) -> str:
    if parent is not None:
        return f"{parent.qualified_name}.{name}"
    return f"{prefix}.{name}" if prefix else name


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def _callee_name(call_node: Node, source: bytes) -> str | None:
    """Return the textual callee, or None for unsupported call shapes."""
    func = call_node.child_by_field_name("function")
    if func is None:
        return None
    if func.type in ("identifier", "attribute"):
        return _node_text(func, source)
    # Subscripts (a[0]()), parenthesised calls, lambdas, etc — skip for v1.
    return None


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def _parse_import(node: Node, source: bytes) -> list[Import]:
    """Handle ``import X`` and ``import X as Y``, including comma-separated."""
    line = node.start_point[0] + 1
    out: list[Import] = []
    for name_node in node.children_by_field_name("name"):
        if name_node.type == "dotted_name":
            out.append(
                Import(
                    module=_node_text(name_node, source),
                    name=None,
                    alias=None,
                    line=line,
                )
            )
        elif name_node.type == "aliased_import":
            module_node = name_node.child_by_field_name("name")
            alias_node = name_node.child_by_field_name("alias")
            if module_node is not None:
                out.append(
                    Import(
                        module=_node_text(module_node, source),
                        name=None,
                        alias=_node_text(alias_node, source) if alias_node else None,
                        line=line,
                    )
                )
    return out


def _parse_import_from(node: Node, source: bytes) -> list[Import]:
    """Handle ``from X import a, b as c`` and ``from X import *``."""
    line = node.start_point[0] + 1
    module_node = node.child_by_field_name("module_name")
    if module_node is None:
        return []
    module = _node_text(module_node, source)

    # Wildcard: `from X import *`
    for child in node.named_children:
        if child.type == "wildcard_import":
            return [Import(module=module, name="*", alias=None, line=line)]

    out: list[Import] = []
    for name_node in node.children_by_field_name("name"):
        if name_node.type in ("identifier", "dotted_name"):
            out.append(
                Import(
                    module=module,
                    name=_node_text(name_node, source),
                    alias=None,
                    line=line,
                )
            )
        elif name_node.type == "aliased_import":
            inner_name = name_node.child_by_field_name("name")
            inner_alias = name_node.child_by_field_name("alias")
            if inner_name is not None:
                out.append(
                    Import(
                        module=module,
                        name=_node_text(inner_name, source),
                        alias=_node_text(inner_alias, source) if inner_alias else None,
                        line=line,
                    )
                )
    return out
