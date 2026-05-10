"""Interactive chat REPL backed by the Agent.

Lazily-imported from ``__main__`` so users running ``index`` don't pay for
``anthropic`` import-time work. Validates the index DB and repo root
*before* constructing the Anthropic client, so missing-file errors don't
require an API key.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from codebase_explainer.schema import connect

PROMPT = "You> "
EXIT_COMMANDS = frozenset({"/exit", "/quit", "/q"})
RESET_COMMAND = "/reset"
HELP_COMMAND = "/help"

HELP_TEXT = """\
Commands:
  /exit, /quit, /q   Leave the REPL.
  /reset             Clear the conversation history.
  /help              Show this help.
Anything else is sent to the agent as a message."""


def run_chat(
    *,
    db_path: Path,
    repo_root: Path,
    model: str,
    effort: str,
) -> int:
    """Entry point invoked from ``__main__``. Returns an exit code."""
    if not db_path.is_file():
        print(f"Error: index DB not found at {db_path}", file=sys.stderr)
        print(
            "Run `python -m codebase_explainer index <repo>` first.",
            file=sys.stderr,
        )
        return 1

    if not repo_root.is_dir():
        print(
            f"Error: --repo-root '{repo_root}' is not a directory",
            file=sys.stderr,
        )
        return 1

    # Lazy import: keeps the index path free of the anthropic dependency at
    # tool-import time, and lets the file-validation errors above run
    # without needing ANTHROPIC_API_KEY in the environment.
    from anthropic import Anthropic

    from codebase_explainer.agent import Agent

    try:
        client = Anthropic()
    except Exception as e:  # noqa: BLE001 — surface any init failure
        print(f"Error: could not initialise Anthropic client: {e}", file=sys.stderr)
        print(
            "Hint: set the ANTHROPIC_API_KEY environment variable.",
            file=sys.stderr,
        )
        return 1

    with connect(db_path) as conn:
        embedder = _maybe_load_embedder(conn)

        agent = Agent(
            client=client,
            db_conn=conn,
            repo_root=repo_root,
            model=model,
            effort=effort,
            embedder=embedder,
        )
        _print_banner(conn, db_path, repo_root, model, effort, embedder)
        _repl_loop(agent)

    return 0


def _maybe_load_embedder(conn):
    """If the index has embeddings, build the matching embedder; else None.

    Reads the most-recent ``model_name`` from the embeddings table so we
    use the same model that was used at index time. If the user re-indexed
    with a different model, the latest one wins. Failure to import the
    backend (e.g. torch not installed) returns ``None`` with a message —
    the chat REPL still works, semantic search is just unavailable.
    """
    from codebase_explainer.embeddings import embedding_count

    if embedding_count(conn) == 0:
        return None

    row = conn.execute(
        "SELECT model_name FROM embeddings ORDER BY id DESC LIMIT 1"
    ).fetchone()
    model_name = row["model_name"]

    if model_name.startswith("fake/"):
        # Test fixture left this in. Don't try to load a real model.
        from codebase_explainer.embeddings import FakeEmbedder

        return FakeEmbedder()

    try:
        from codebase_explainer.embeddings import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=model_name)
    except Exception as e:  # noqa: BLE001 — fall back to no semantic search
        print(
            f"# Note: index has embeddings but couldn't load model "
            f"({type(e).__name__}: {e}). Semantic search disabled.",
            file=sys.stderr,
        )
        return None


def _print_banner(conn, db_path, repo_root, model, effort, embedder) -> None:
    n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    n_symbols = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    n_calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    print(f"# Codebase Explainer Agent  —  model={model}, effort={effort}")
    print(f"# Index: {db_path} ({n_files} files / {n_symbols} symbols / {n_calls} calls)")
    print(f"# Repo:  {repo_root}")
    if embedder is not None:
        n_emb = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        print(f"# Semantic search: ON ({n_emb} embeddings, {embedder.model_name})")
    else:
        print("# Semantic search: OFF (re-index with --embed to enable)")
    print("# /exit to quit, /reset to clear conversation, /help for commands")
    print()


def _repl_loop(agent) -> None:
    while True:
        try:
            line = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not line:
            continue
        if line in EXIT_COMMANDS:
            return
        if line == RESET_COMMAND:
            agent.messages.clear()
            print("# Conversation cleared.\n")
            continue
        if line == HELP_COMMAND:
            print(HELP_TEXT, "\n")
            continue

        try:
            _run_one_turn(agent, line)
        except KeyboardInterrupt:
            # User aborted mid-turn. Conversation state is preserved up to
            # the last completed iteration; the partial assistant turn we
            # may have appended will get a tool_result on the next message.
            print("\n[interrupted]")
        except Exception as e:  # noqa: BLE001
            print(f"\n[error: {type(e).__name__}: {e}]")


def _run_one_turn(agent, user_message: str) -> None:
    def on_tool_use(name: str, tool_input: dict[str, Any]) -> None:
        formatted = ", ".join(f"{k}={_short_repr(v)}" for k, v in tool_input.items())
        print(f"  -> {name}({formatted})")

    def on_text(text: str) -> None:
        body = text.strip()
        if body:
            print()
            print(body)

    print()
    agent.run_turn(user_message, on_tool_use=on_tool_use, on_text=on_text)
    print()


def _short_repr(value: Any, max_len: int = 80) -> str:
    """Render a tool-input value for the ``-> tool(...)`` line.

    Strings get repr'd so newlines and quotes are visible; long values are
    truncated with an ellipsis so a 5KB regex doesn't blow up the screen.
    """
    rendered = repr(value)
    if len(rendered) > max_len:
        rendered = rendered[: max_len - 3] + "..."
    return rendered
