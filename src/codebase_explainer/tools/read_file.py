"""read_file tool: return source bytes with line numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Hard cap so a single tool call can't blow up the agent's context budget.
MAX_LINES = 500


def handle_read_file(input: dict[str, Any], *, repo_root: Path, **_: Any) -> str:
    path = input["path"]
    start_line = input.get("start_line")
    end_line = input.get("end_line")

    full_path = repo_root / path
    if not full_path.is_file():
        return f"Error: file not found: {path}"

    try:
        text = full_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: could not decode {path} as UTF-8"

    lines = text.splitlines()
    total = len(lines)

    start = max(1, start_line if start_line is not None else 1)
    end = min(end_line if end_line is not None else total, total)

    if start > total:
        return f"Error: start_line {start} exceeds file length ({total} lines)"
    if end < start:
        return f"Error: end_line ({end}) precedes start_line ({start})"

    span = end - start + 1
    if span > MAX_LINES:
        return (
            f"Error: requested range too large ({span} lines, max {MAX_LINES}). "
            f"Narrow the range — file has {total} lines total."
        )

    selected = lines[start - 1 : end]
    width = len(str(end))
    body = "\n".join(
        f"{i:>{width}}  {line}" for i, line in enumerate(selected, start=start)
    )
    header = f"# {path} (lines {start}-{end} of {total})"
    return f"{header}\n{body}"
