"""Tool surface for the agent.

`TOOL_DEFINITIONS` is the JSON list passed to the Anthropic API as the
``tools`` parameter — its bytes must be stable across requests so the
prompt cache stays warm. `TOOL_HANDLERS` maps each tool's ``name`` to the
Python implementation that runs when Claude calls it.
"""

from __future__ import annotations

from codebase_explainer.tools.definitions import TOOL_DEFINITIONS
from codebase_explainer.tools.find_callers import handle_find_callers
from codebase_explainer.tools.find_definition import handle_find_definition
from codebase_explainer.tools.grep import handle_grep
from codebase_explainer.tools.read_file import handle_read_file
from codebase_explainer.tools.view_symbol import handle_view_symbol

TOOL_HANDLERS = {
    "read_file": handle_read_file,
    "grep": handle_grep,
    "find_definition": handle_find_definition,
    "find_callers": handle_find_callers,
    "view_symbol": handle_view_symbol,
}

__all__ = ["TOOL_DEFINITIONS", "TOOL_HANDLERS"]
