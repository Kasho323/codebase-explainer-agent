"""Local Gradio demo for the codebase-explainer agent.

Gradio (and the Anthropic SDK) are imported lazily inside the launch
functions so this module loads cleanly in environments that don't have
them — in particular, CI tests can import this file to verify shape
without paying for ``gradio`` / ``anthropic`` install.

Real demo only. No pre-recorded answers, no hardcoded responses. If the
user hasn't set credentials, the UI returns a clear warning when they
ask a question — never a fabricated answer.

Environment variables (all read directly by the Anthropic SDK when
``Anthropic()`` is constructed with no args):

    ANTHROPIC_API_KEY     Primary key, sent as ``x-api-key`` header.
    ANTHROPIC_AUTH_TOKEN  Alternative auth, sent as ``Authorization:
                          Bearer ...``. Some Anthropic-compatible
                          endpoints (e.g. DeepSeek's /anthropic compat
                          layer) prefer this form.
    ANTHROPIC_BASE_URL    Endpoint override (e.g.
                          https://api.deepseek.com/anthropic). Defaults
                          to Anthropic's official endpoint.
    ANTHROPIC_MODEL       Model id override. When set, takes precedence
                          over the --model CLI flag (the CLI flag's
                          default would otherwise win).

This is the chat REPL's web sibling: same Agent, same tool-call
transparency, served over HTTP for non-terminal users.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from codebase_explainer.schema import connect

DEFAULT_PORT = 7860


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_demo(
    *,
    db_path: Path,
    repo_root: Path,
    model: str,
    effort: str,
    port: int = DEFAULT_PORT,
    share: bool = False,
) -> int:
    """Launch the Gradio app. Returns exit code (0 on graceful exit).

    Validates inputs *before* importing gradio so missing-file errors don't
    require ``gradio`` or ``ANTHROPIC_API_KEY`` to surface usefully.
    """
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

    has_creds, base_url, model_from_env = _resolve_credentials()
    effective_model = model_from_env or model
    if not has_creds:
        print(
            "Warning: neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set. "
            "The UI will refuse to answer questions until you stop the demo, "
            "set credentials, and re-launch.",
            file=sys.stderr,
        )

    try:
        import gradio as gr  # noqa: PLC0415
    except ImportError as e:
        print(
            f"Error: gradio is not installed ({e}). "
            "Run `pip install -r requirements.txt` first.",
            file=sys.stderr,
        )
        return 1

    # Lazy import: lets test_demo.py import this module in CI without
    # anthropic available. (anthropic IS in CI today, but staying lazy
    # mirrors the pattern used by chat.py and keeps the door open.)
    from codebase_explainer.chat import _maybe_load_embedder  # noqa: PLC0415

    conn = connect(db_path)
    embedder = _maybe_load_embedder(conn)
    client = _build_client(has_credentials=has_creds)

    banner = _build_banner(
        repo_root=repo_root,
        db_path=db_path,
        model=effective_model,
        base_url=base_url,
        conn=conn,
        has_embeddings=embedder is not None,
        has_credentials=has_creds,
    )

    def respond(question: str, _history: list[Any]) -> str:
        """Gradio ChatInterface callback. History is ignored — each turn
        constructs a fresh Agent, so questions are independent. The chat
        scrollback in the UI is purely visual."""
        return run_single_turn(
            question=question,
            client=client,
            conn=conn,
            repo_root=repo_root,
            model=effective_model,
            effort=effort,
            embedder=embedder,
        )

    with gr.Blocks(title="Codebase Explainer Agent — Demo") as demo:
        gr.Markdown(banner)
        gr.ChatInterface(
            fn=respond,
            type="messages",
            examples=[
                "Where is build_graph defined and what does it return?",
                "Who calls recommend in this repo?",
                "Explain what dfs._visit does and why it is nested inside dfs.",
            ],
        )

    print(f"Demo running at http://localhost:{port}", flush=True)
    demo.launch(server_port=port, share=share)
    return 0


# ---------------------------------------------------------------------------
# Per-turn handler — extracted so it can be unit-tested with a mock client
# ---------------------------------------------------------------------------


def run_single_turn(
    *,
    question: str,
    client: Any,
    conn: Any,
    repo_root: Path,
    model: str,
    effort: str,
    embedder: Any,
) -> str:
    """One full agent turn: tool-use loop + final answer, formatted for chat.

    Returns the markdown string to display in the chat bubble. If
    ``client`` is ``None`` (no API key), returns a clear warning rather
    than calling out.
    """
    if client is None:
        return (
            "⚠️ **ANTHROPIC_API_KEY not set.** Stop the demo, set the env var, "
            "then re-run `python -m codebase_explainer demo`."
        )

    # Lazy import keeps demo.py loadable in environments without anthropic
    # (e.g. some smoke tests). CI has anthropic today, but we keep the
    # pattern consistent with chat.py.
    from codebase_explainer.agent import Agent  # noqa: PLC0415

    agent = Agent(
        client=client,
        db_conn=conn,
        repo_root=repo_root,
        model=model,
        effort=effort,
        embedder=embedder,
    )

    tool_calls: list[str] = []

    def on_tool_use(name: str, tool_input: dict[str, Any]) -> None:
        formatted_args = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
        tool_calls.append(f"→ {name}({formatted_args})")

    try:
        answer = agent.run_turn(question, on_tool_use=on_tool_use)
    except Exception as e:  # noqa: BLE001 — surface to UI, never crash the server
        return f"[error: {type(e).__name__}: {e}]"

    return _format_response(answer=answer, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Pure helpers — fully unit-tested, no SDK dependencies
# ---------------------------------------------------------------------------


def _format_response(*, answer: str, tool_calls: list[str]) -> str:
    """Render tool-call trace + answer as one markdown blob.

    Tool calls go in a fenced block so they read as a "transcript" of what
    the agent did before the answer.
    """
    if not tool_calls:
        return answer
    trace = "```\n" + "\n".join(tool_calls) + "\n```\n\n"
    return trace + answer


def _resolve_credentials() -> tuple[bool, str | None, str | None]:
    """Read Anthropic-compatible config from the environment.

    Returns ``(has_credentials, base_url, model_from_env)``:

    - ``has_credentials``: True if either ``ANTHROPIC_API_KEY`` or
      ``ANTHROPIC_AUTH_TOKEN`` is set. The Anthropic SDK picks whichever
      is present and uses the appropriate auth header.
    - ``base_url``: value of ``ANTHROPIC_BASE_URL`` if set, else ``None``.
      When ``None``, the SDK uses the official Anthropic endpoint.
    - ``model_from_env``: value of ``ANTHROPIC_MODEL`` if set, else
      ``None``. When set it takes precedence over the ``--model`` CLI
      flag (because we can't distinguish the user typing the default
      from accepting it).

    Pure read-only — no side effects, easy to unit-test with monkeypatch.
    """
    has_creds = bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    )
    base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    model_env = os.environ.get("ANTHROPIC_MODEL") or None
    return has_creds, base_url, model_env


def _build_client(*, has_credentials: bool) -> Any:
    """Construct an Anthropic client, or return None if unavailable.

    The SDK auto-reads ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` /
    ``ANTHROPIC_BASE_URL`` from env when constructed with no args, so we
    don't pass them explicitly. Never raises — caller routes the
    (possibly ``None``) client to :func:`run_single_turn` which handles
    the None case with a UI warning.
    """
    if not has_credentials:
        return None
    try:
        from anthropic import Anthropic  # noqa: PLC0415

        return Anthropic()
    except Exception as e:  # noqa: BLE001
        print(
            f"Warning: Anthropic client init failed ({type(e).__name__}: {e}). "
            "Demo will start but cannot answer questions.",
            file=sys.stderr,
        )
        return None


def _build_banner(
    *,
    repo_root: Path,
    db_path: Path,
    model: str,
    base_url: str | None,
    conn: Any,
    has_embeddings: bool,
    has_credentials: bool,
) -> str:
    n_files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    n_symbols = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
    n_calls = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    sem = "ON" if has_embeddings else "OFF (re-index with `--embed` to enable)"
    creds = "✅ set" if has_credentials else "❌ NOT SET — answers will be blocked"
    endpoint = base_url if base_url else "default (api.anthropic.com)"
    return (
        "## Codebase Explainer Agent — Local Demo\n\n"
        f"- **Indexed repo**: `{repo_root}`\n"
        f"- **Index**: `{db_path}` "
        f"({n_files} files / {n_symbols} symbols / {n_calls} calls)\n"
        f"- **Model**: `{model}` · **Semantic search**: {sem}\n"
        f"- **API endpoint**: `{endpoint}` · **Credentials**: {creds}\n\n"
        "Type a question or pick one of the examples below. Every answer "
        "is grounded in the indexed source and cites `path:line` references "
        "the user can click to verify."
    )
