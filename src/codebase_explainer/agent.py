"""Agent loop: send a user message, dispatch tool calls until end_turn.

Uses Claude Sonnet 4.6 with adaptive thinking. Runs the agentic loop
manually rather than via the SDK's tool runner so the caller can observe
each tool invocation as it happens (useful for the CLI's transparency,
and for plugging in a custom logger or human-in-the-loop gate later).

Prompt-caching strategy: the top-level ``cache_control`` marker auto-
places on the last cacheable block of every request, which means the
tool-definition list, the system prompt, and every prior turn are all
cached after the first round-trip. Verify with
``response.usage.cache_read_input_tokens`` — a steady-state non-zero
value confirms the cache is hitting.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from codebase_explainer.embeddings import embedding_count
from codebase_explainer.tools import (
    EMBEDDING_DEPENDENT_TOOLS,
    TOOL_DEFINITIONS,
    TOOL_HANDLERS,
)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_EFFORT = "medium"
MAX_TOOL_ITERATIONS = 25  # safety net against runaway loops

SYSTEM_PROMPT = """\
You are a code-explorer agent. The user is asking questions about a Python repo \
that has been pre-indexed into a SQLite symbol graph. You have these tools:

  read_file(path, start_line, end_line)
      Read source by path. Always prefer line ranges over full reads when
      you know where to look.

  grep(pattern, glob, max_results)
      Search file contents with a Python regex. Use for literal strings.

  find_definition(name)
      Look up where a function/class/method is defined. Returns location,
      signature, and docstring. Use when you only need the headline.

  find_callers(name)
      Find every call site of a symbol. Use for "who calls X" or "what does
      removing X break".

  view_symbol(name)
      One-shot deep lookup: returns location, signature, docstring, source
      body, parent class, every caller, and every callee in a single call.
      Prefer this over chaining find_definition + read_file + find_callers
      when you want to understand a single symbol thoroughly.

  search_semantic(query, k)
      Fuzzy semantic search over symbol embeddings. Returns the top-k
      symbols whose docstring + body are closest to a natural-language
      query. Use when you don't know the symbol's name — e.g. "where is
      auth handled", "anything related to retry logic". Only available if
      the index was built with --embed; otherwise this tool is omitted
      from the tool list.

WORKFLOW

  1. Pick the tool that answers the question with the fewest calls. For
     "explain X" (you know the name), call view_symbol — usually one-shot.
     For "who calls X", use find_callers. For "where is X handled" (no
     symbol name), use search_semantic if available, else grep.
  2. Read just enough source to answer. Quote the specific lines that
     justify your answer, not whole files.
  3. ALWAYS cite locations as `path:line` (e.g. `covid_pipeline/main.py:42`).
     Lines come from the indexer; users can click them to verify.
  4. Don't fabricate code. If a tool returns no results, say so. Label
     inferences as inferences ("this is probably...").
  5. Be concise. End with a one-sentence summary if the answer involved
     more than one tool call.
"""


ToolCallback = Callable[[str, dict[str, Any]], None]
TextCallback = Callable[[str], None]


@dataclass
class Agent:
    """Multi-turn agent over the Claude Messages API.

    Holds the conversation in ``messages`` and reuses it across calls to
    :meth:`run_turn`. The DB connection, repo root, and optional embedder
    are passed to every tool handler — handlers that don't need a
    particular kwarg just swallow it via ``**_``.

    The active tool list (``self._tools``) is computed once at
    construction. Tools that depend on embeddings are dropped if no
    embedder is configured or the index has no embeddings, so Claude
    doesn't waste a tool call discovering they don't work — and so the
    prompt-cache key stays stable for the life of the session.
    """

    client: Anthropic
    db_conn: sqlite3.Connection
    repo_root: Path
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    effort: str = DEFAULT_EFFORT
    embedder: Any = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    _tools: list[dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._tools = self._compute_active_tools()

    def _compute_active_tools(self) -> list[dict[str, Any]]:
        if self.embedder is None or embedding_count(self.db_conn) == 0:
            return [t for t in TOOL_DEFINITIONS if t["name"] not in EMBEDDING_DEPENDENT_TOOLS]
        return list(TOOL_DEFINITIONS)

    def run_turn(
        self,
        user_message: str,
        *,
        on_tool_use: ToolCallback | None = None,
        on_text: TextCallback | None = None,
    ) -> str:
        """Send ``user_message`` and loop until Claude stops calling tools.

        Returns the concatenated text of the final assistant message.
        ``on_tool_use(name, input)`` fires once per tool call. ``on_text``
        fires once per text block in every assistant message (including
        intermediate ones between tool calls).
        """
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(MAX_TOOL_ITERATIONS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=self._tools,
                cache_control={"type": "ephemeral"},
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=self.messages,
            )

            # Preserve full content (including tool_use and thinking blocks)
            # so the next request gets a valid round-trip.
            self.messages.append({"role": "assistant", "content": response.content})

            if on_text is not None:
                for block in response.content:
                    if block.type == "text":
                        on_text(block.text)

            if response.stop_reason == "end_turn":
                return "\n".join(
                    b.text for b in response.content if b.type == "text"
                )

            if response.stop_reason != "tool_use":
                # max_tokens, refusal, or pause_turn — surface to the caller
                # so the CLI can show a useful diagnostic. The caller can
                # inspect self.messages for the partial state.
                return f"[stopped: {response.stop_reason}]"

            tool_results = self._dispatch_tools(response.content, on_tool_use)
            self.messages.append({"role": "user", "content": tool_results})

        return "[stopped: hit MAX_TOOL_ITERATIONS]"

    def _dispatch_tools(
        self,
        content: list[Any],
        on_tool_use: ToolCallback | None,
    ) -> list[dict[str, Any]]:
        """Execute every ``tool_use`` block in ``content`` and return the
        ``tool_result`` blocks to send back."""
        results: list[dict[str, Any]] = []
        for block in content:
            if block.type != "tool_use":
                continue
            if on_tool_use is not None:
                on_tool_use(block.name, block.input)
            results.append(self._call_one(block.id, block.name, block.input))
        return results

    def _call_one(
        self, tool_use_id: str, name: str, tool_input: dict[str, Any]
    ) -> dict[str, Any]:
        handler = TOOL_HANDLERS.get(name)
        if handler is None:
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Error: unknown tool {name!r}",
                "is_error": True,
            }
        try:
            result = handler(
                tool_input,
                db_conn=self.db_conn,
                repo_root=self.repo_root,
                embedder=self.embedder,
            )
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
            }
        except Exception as e:  # noqa: BLE001 — surface any handler failure to Claude
            return {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": f"Error: {type(e).__name__}: {e}",
                "is_error": True,
            }
