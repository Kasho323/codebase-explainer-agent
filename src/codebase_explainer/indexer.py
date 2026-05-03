"""Tree-sitter Python indexer.

Walks Python source and extracts a flat list of symbols (functions, classes,
methods, including nested defs). Persistence into SQLite happens in a
separate layer; this module just produces dataclasses.
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
    would show. `qualified_name` is dot-joined: 'app.auth.User.save'.
    """

    kind: str  # 'function' | 'class' | 'method'
    name: str
    qualified_name: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    parent: Symbol | None = field(default=None, repr=False)


def extract_symbols(source: str | bytes, prefix: str = "") -> list[Symbol]:
    """Parse Python source and return all symbols in source order.

    `prefix` is an optional module path (e.g. 'app.auth') used as the root
    of qualified names.
    """
    source_bytes = source.encode("utf-8") if isinstance(source, str) else source
    tree = _parser().parse(source_bytes)
    symbols: list[Symbol] = []
    _walk(tree.root_node, source_bytes, parent=None, prefix=prefix, out=symbols)
    return symbols


def _parser() -> Parser:
    return Parser(PY_LANGUAGE)


def _node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _walk(
    node: Node,
    source: bytes,
    parent: Symbol | None,
    prefix: str,
    out: list[Symbol],
) -> None:
    for child in node.named_children:
        if child.type == "function_definition":
            sym = _make_function(child, source, parent=parent, prefix=prefix)
            out.append(sym)
            body = child.child_by_field_name("body")
            if body is not None:
                _walk(body, source, parent=sym, prefix=prefix, out=out)
        elif child.type == "class_definition":
            sym = _make_class(child, source, parent=parent, prefix=prefix)
            out.append(sym)
            body = child.child_by_field_name("body")
            if body is not None:
                _walk(body, source, parent=sym, prefix=prefix, out=out)
        else:
            _walk(child, source, parent=parent, prefix=prefix, out=out)


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
    """Reproduce 'def name(params) -> Return:' from the AST nodes."""
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
    """Return the first string literal inside body, if it's the first statement."""
    for child in body.named_children:
        if child.type != "expression_statement":
            return None
        for sub in child.named_children:
            if sub.type == "string":
                raw = _node_text(sub, source)
                return _strip_quotes(raw)
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
