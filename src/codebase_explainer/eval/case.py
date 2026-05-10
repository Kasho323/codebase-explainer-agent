"""GoldenCase dataclass + TOML loader.

Cases live as one ``.toml`` per question under ``eval/golden_cases/``.
We use TOML rather than YAML because Python 3.11+ ships ``tomllib`` in
the stdlib — zero new dependencies, and TOML's multi-line ``\"\"\" \"\"\"``
strings cover the ``expected_gist`` use case cleanly.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

VALID_QUESTION_TYPES = frozenset(
    {
        "definition_lookup",
        "call_graph",
        "design_intent",
        "change_impact",
    }
)


@dataclass(frozen=True)
class GoldenCase:
    """One scored question against one repo."""

    id: str
    repo: str
    repo_sha: str
    question: str
    question_type: str
    expected_citations: tuple[str, ...]
    expected_gist: str
    source_path: Path | None = field(default=None, compare=False, repr=False)


def load_case(path: Path) -> GoldenCase:
    """Parse one ``.toml`` case file. Raises ``ValueError`` on malformed input."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return _from_dict(data, source_path=path)


def load_cases(dir_path: Path) -> list[GoldenCase]:
    """Recursively load every ``.toml`` under ``dir_path``, sorted by id.

    Empty directories return ``[]``. Files that fail to parse raise the
    underlying ``tomllib.TOMLDecodeError`` / ``ValueError`` — we don't
    silently skip, because a broken case is almost always a typo the user
    wants to know about.
    """
    if not dir_path.is_dir():
        raise FileNotFoundError(f"cases directory not found: {dir_path}")
    cases = [load_case(p) for p in sorted(dir_path.rglob("*.toml"))]
    cases.sort(key=lambda c: c.id)
    return cases


def _from_dict(data: dict, *, source_path: Path | None) -> GoldenCase:
    missing = [k for k in ("id", "repo", "question", "question_type", "expected_citations", "expected_gist") if k not in data]
    if missing:
        loc = f" ({source_path})" if source_path else ""
        raise ValueError(f"case missing required field(s): {missing}{loc}")

    qtype = data["question_type"]
    if qtype not in VALID_QUESTION_TYPES:
        raise ValueError(
            f"question_type {qtype!r} not in {sorted(VALID_QUESTION_TYPES)}"
        )

    citations = data["expected_citations"]
    if not isinstance(citations, list) or not all(isinstance(c, str) for c in citations):
        raise ValueError("expected_citations must be a list of strings")
    if not citations:
        raise ValueError("expected_citations cannot be empty")

    return GoldenCase(
        id=str(data["id"]),
        repo=str(data["repo"]),
        repo_sha=str(data.get("repo_sha", "HEAD")),
        question=str(data["question"]).strip(),
        question_type=qtype,
        expected_citations=tuple(citations),
        expected_gist=str(data["expected_gist"]).strip(),
        source_path=source_path,
    )
