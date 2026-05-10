"""CLI entry point: ``python -m codebase_explainer ...``"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Imported eagerly so ``--help`` shows the live defaults. The agent module
# pulls in ``anthropic`` at import time, but the cost is tolerable for a
# CLI startup, and centralising the defaults here avoids drift.
from codebase_explainer.agent import DEFAULT_EFFORT, DEFAULT_MODEL
from codebase_explainer.index_repo import index_repo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codebase_explainer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    index = sub.add_parser("index", help="Index a Python repo into a SQLite DB.")
    index.add_argument("path", type=Path, help="Path to a Python repo (directory).")
    index.add_argument(
        "--db",
        type=Path,
        default=Path(".codebase-index.sqlite3"),
        help="SQLite DB path (default: .codebase-index.sqlite3).",
    )
    index.add_argument(
        "--embed",
        action="store_true",
        help=(
            "Also embed every symbol with sentence-transformers/all-MiniLM-L6-v2 "
            "to enable the search_semantic tool. Requires sentence-transformers "
            "and faiss-cpu — pulls torch on first install (~2GB)."
        ),
    )

    chat = sub.add_parser(
        "chat",
        help="Interactive chat with the agent over an indexed repo.",
    )
    chat.add_argument(
        "--db",
        type=Path,
        default=Path(".codebase-index.sqlite3"),
        help="Index DB path (default: .codebase-index.sqlite3).",
    )
    chat.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Path to the repo the index was built against (default: cwd).",
    )
    chat.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model ID (default: {DEFAULT_MODEL}).",
    )
    chat.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "max"],
        help=f"Reasoning effort level (default: {DEFAULT_EFFORT}).",
    )

    eval_cmd = sub.add_parser(
        "eval",
        help="Run golden-case evaluation against the agent (5a: citation_match only).",
    )
    eval_cmd.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="Directory of .toml golden cases (recursive).",
    )
    eval_cmd.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Pre-built index DB the agent will query against.",
    )
    eval_cmd.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Repo root the index was built against.",
    )
    eval_cmd.add_argument(
        "--output",
        type=Path,
        default=Path("eval-report.md"),
        help="Where to write the markdown report (default: eval-report.md).",
    )
    eval_cmd.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude model ID (default: {DEFAULT_MODEL}).",
    )
    eval_cmd.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["low", "medium", "high", "max"],
        help=f"Reasoning effort level (default: {DEFAULT_EFFORT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "index":
        embedder = None
        if args.embed:
            # Lazy import so the non-embed path doesn't pull torch.
            from codebase_explainer.embeddings import SentenceTransformerEmbedder

            print(
                f"Loading embedder ({SentenceTransformerEmbedder.DEFAULT_MODEL})... "
                "first run downloads ~80MB.",
                flush=True,
            )
            embedder = SentenceTransformerEmbedder()

        stats = index_repo(args.path, args.db, embedder=embedder)
        print(f"Indexed {stats.files} files into {args.db}")
        print(f"  symbols:  {stats.symbols}")
        print(f"  imports:  {stats.imports}")
        print(
            f"  calls:    {stats.calls}  "
            f"(resolved to in-repo symbols: {stats.resolved_calls})"
        )
        if stats.embedded_symbols:
            print(f"  embedded: {stats.embedded_symbols}")
        if stats.skipped:
            print(f"  skipped:  {stats.skipped}")
        return 0

    if args.cmd == "chat":
        # Lazy import so tests of `index` don't pull the chat dependencies.
        from codebase_explainer.chat import run_chat

        return run_chat(
            db_path=args.db,
            repo_root=args.repo_root,
            model=args.model,
            effort=args.effort,
        )

    if args.cmd == "eval":
        # Lazy imports keep the index/chat paths free of eval-only modules.
        from codebase_explainer.eval.case import load_cases
        from codebase_explainer.eval.report import render_markdown
        from codebase_explainer.eval.runner import run_eval

        cases = load_cases(args.cases)
        if not cases:
            print(f"No .toml cases under {args.cases}", file=sys.stderr)
            return 1

        agent_factory = _build_real_agent_factory(
            db_path=args.db,
            repo_root=args.repo_root,
            model=args.model,
            effort=args.effort,
        )
        results = run_eval(cases, agent_factory=agent_factory)
        args.output.write_text(render_markdown(results), encoding="utf-8")

        passes = sum(1 for r in results if r.citation_pass)
        print(f"Wrote {args.output}  —  {passes}/{len(results)} citation pass.")
        return 0

    parser.print_help()
    return 2


def _build_real_agent_factory(
    *,
    db_path: Path,
    repo_root: Path,
    model: str,
    effort: str,
):
    """Construct a closure that builds a fresh Agent per case.

    Each case gets its own Agent so the conversation history doesn't leak
    between unrelated questions. The DB connection and Anthropic client
    are reused across cases for efficiency.
    """
    from anthropic import Anthropic

    from codebase_explainer.agent import Agent
    from codebase_explainer.chat import _maybe_load_embedder
    from codebase_explainer.schema import connect

    client = Anthropic()
    conn = connect(db_path)
    embedder = _maybe_load_embedder(conn)

    def factory(_case):
        return Agent(
            client=client,
            db_conn=conn,
            repo_root=repo_root,
            model=model,
            effort=effort,
            embedder=embedder,
        )

    return factory


if __name__ == "__main__":
    sys.exit(main())
