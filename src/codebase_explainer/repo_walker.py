"""Walk a repo and produce per-file inputs for the indexer.

Skips directories that almost never contain useful source: VCS metadata,
build outputs, virtualenvs, and IDE/editor caches. The skip list is a set
membership check on path components, so it correctly handles nested cases
like ``vendor/.venv/...``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        "node_modules",
        "dist",
        "build",
        ".cache",
        ".idea",
        ".vscode",
    }
)


def walk_python_files(root: str | Path) -> Iterator[Path]:
    """Yield every ``.py`` file under ``root``, skipping uninteresting dirs."""
    root = Path(root)
    if root.is_file():
        if root.suffix == ".py":
            yield root
        return
    if not root.is_dir():
        return
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        yield path


def relative_module_prefix(file_path: Path, repo_root: Path) -> str:
    """Convert a file path under ``repo_root`` into a dotted module prefix.

    Examples (with repo_root='/r'):
        /r/app/__init__.py -> 'app'
        /r/app/auth.py     -> 'app.auth'
        /r/foo.py          -> 'foo'
    """
    try:
        rel = file_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return file_path.stem
    parts = list(rel.parts)
    if not parts:
        return ""
    last = parts[-1]
    if last == "__init__.py":
        parts = parts[:-1]
    elif last.endswith(".py"):
        parts[-1] = last[:-3]
    return ".".join(parts)
