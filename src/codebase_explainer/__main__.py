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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "index":
        stats = index_repo(args.path, args.db)
        print(f"Indexed {stats.files} files into {args.db}")
        print(f"  symbols:  {stats.symbols}")
        print(f"  imports:  {stats.imports}")
        print(
            f"  calls:    {stats.calls}  "
            f"(resolved to in-repo symbols: {stats.resolved_calls})"
        )
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

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
