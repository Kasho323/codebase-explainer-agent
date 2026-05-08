"""grep tool: regex search across the repo."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

DEFAULT_GLOB = "**/*.py"
HARD_MAX_RESULTS = 100
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        "dist",
        "build",
        ".cache",
    }
)


def handle_grep(input: dict[str, Any], *, repo_root: Path, **_: Any) -> str:
    pattern = input["pattern"]
    glob = input.get("glob") or DEFAULT_GLOB
    requested = int(input.get("max_results", 20) or 20)
    max_results = min(max(1, requested), HARD_MAX_RESULTS)

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex {pattern!r}: {e}"

    matches: list[str] = []
    for file_path in repo_root.rglob(glob):
        if not file_path.is_file():
            continue
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        rel = file_path.relative_to(repo_root).as_posix()
        for line_num, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{rel}:{line_num}: {line}")
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    if not matches:
        return f"No matches for pattern {pattern!r} in {glob}."

    truncated = " (truncated)" if len(matches) >= max_results else ""
    header = f"# {len(matches)} match(es){truncated}"
    return header + "\n" + "\n".join(matches)
