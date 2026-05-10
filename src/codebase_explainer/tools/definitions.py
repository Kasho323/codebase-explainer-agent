"""JSON-schema definitions for the agent's tools.

This module is intentionally pure data. The bytes of ``TOOL_DEFINITIONS``
must be deterministic across requests so that the prompt cache hits — any
non-determinism here (e.g. iterating a ``set``, embedding a timestamp)
silently invalidates the cache for every request.
"""

from __future__ import annotations

from typing import Any

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file in the indexed repo. Lines are 1-indexed. "
            "Use start_line and end_line to read a specific range and save tokens — "
            "prefer ranges over full reads when you know the location of what you want."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative path, e.g. 'covid_pipeline/main.py'.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Optional 1-indexed start line. Reads from beginning if omitted.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Optional 1-indexed end line, inclusive. Reads to end if omitted.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search the repo for a regex pattern. Returns matching lines with file paths "
            "and line numbers. Use this when you don't know where something is defined or "
            "want to find usages of a string."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regex pattern (re.search semantics).",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional rglob filter. Default '**/*.py'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max matching lines to return. Default 20, hard cap 100.",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "find_definition",
        "description": (
            "Find where a symbol (function, class, method) is defined in the repo. "
            "Pass the unqualified name (e.g. 'build_graph') or any suffix of the "
            "qualified name. Returns location, kind, signature, and docstring. Prefer "
            "this over grep when looking up a known symbol."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unqualified name ('foo'), qualified suffix ('module.foo'), or fully-qualified name.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "find_callers",
        "description": (
            "Find all call sites of a symbol. Returns caller, file, and line for each call. "
            "Resolved callers come from the symbol-graph index. Textual matches (callers we "
            "couldn't resolve to an in-repo symbol) are listed separately and may include "
            "external library calls of the same name — treat them as candidates, not certainties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unqualified name ('foo'), qualified suffix ('module.foo'), or fully-qualified name.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "view_symbol",
        "description": (
            "One-shot deep lookup for a symbol: returns location, signature, docstring, "
            "the source body, the parent class (if a method), every caller, and every "
            "callee — all in one call. Prefer this over chaining find_definition + "
            "read_file + find_callers when you want to understand a single symbol fully."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unqualified name ('foo'), qualified suffix ('module.foo'), or fully-qualified name.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_semantic",
        "description": (
            "Fuzzy semantic search over the symbol index. Returns the top-k "
            "symbols whose docstring + body are closest to the natural-language "
            "query, with similarity scores. Use this when you don't know the "
            "symbol's name — e.g. 'where is auth handled', 'anything related "
            "to retry logic'. Use grep when the query is a literal string and "
            "find_definition when you already know the symbol name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A short natural-language description of what you're looking for.",
                },
                "k": {
                    "type": "integer",
                    "description": "Max results to return. Default 10, hard cap 20.",
                },
            },
            "required": ["query"],
        },
    },
]
