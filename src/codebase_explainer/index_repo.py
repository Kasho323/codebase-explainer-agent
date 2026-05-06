"""Orchestrator: walk a repo, parse each file, persist into SQLite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codebase_explainer.indexer import extract_file
from codebase_explainer.persistence import hash_source, write_file_index
from codebase_explainer.repo_walker import relative_module_prefix, walk_python_files
from codebase_explainer.resolver import resolve_callees
from codebase_explainer.schema import connect, init_db


@dataclass
class IndexStats:
    files: int = 0
    symbols: int = 0
    calls: int = 0
    imports: int = 0
    skipped: int = 0
    resolved_calls: int = 0


def index_repo(repo_root: str | Path, db_path: str | Path) -> IndexStats:
    """Walk a Python repo, parse every ``.py`` file, write the index to SQLite.

    Files that fail to read or decode are counted in ``skipped`` and not
    inserted (e.g. binary content with a ``.py`` extension by accident).
    """
    repo_root = Path(repo_root).resolve()
    init_db(db_path)
    stats = IndexStats()

    with connect(db_path) as conn:
        for py_file in walk_python_files(repo_root):
            try:
                source = py_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                stats.skipped += 1
                continue

            prefix = relative_module_prefix(py_file, repo_root)
            file_index = extract_file(source, prefix=prefix)

            try:
                rel_path = str(py_file.relative_to(repo_root))
            except ValueError:
                rel_path = str(py_file)
            # Normalise to forward slashes so paths look the same on Windows
            # and Linux — important for portable diffs and recruiter-friendly
            # SQL output.
            rel_path = rel_path.replace("\\", "/")

            write_file_index(
                conn,
                path=rel_path,
                file_index=file_index,
                content_hash=hash_source(source),
            )

            stats.files += 1
            stats.symbols += len(file_index.symbols)
            stats.calls += len(file_index.calls)
            stats.imports += len(file_index.imports)

        conn.commit()
        stats.resolved_calls = resolve_callees(conn)

    return stats
