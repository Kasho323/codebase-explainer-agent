"""CLI entry point: ``python -m codebase_explainer ...``"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "index":
        stats = index_repo(args.path, args.db)
        print(f"Indexed {stats.files} files into {args.db}")
        print(f"  symbols: {stats.symbols}")
        print(f"  calls:   {stats.calls}")
        print(f"  imports: {stats.imports}")
        if stats.skipped:
            print(f"  skipped: {stats.skipped}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
