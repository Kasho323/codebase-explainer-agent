"""Deterministic scorers for the eval harness.

5a only ships ``citation_match`` — a regex-based check that the agent's
final answer cites at least one expected ``path:line``. LLM-judge scorers
(faithfulness, gist_match) are 5b work and intentionally not in this
module yet.

Design note for 5b: when LLM-judge functions land, they MUST take the
judge model id as an explicit parameter (e.g. ``llm_judge_faithfulness(
..., judge_model: str)``) — never read a global constant or env var
inside this module. The CLI / runner reads the env var or flag and
threads it through, so tests stay deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Matches:  path/to/file.ext:42        path/to/file.ext:42-50
# Captures: ('path/to/file.ext', '42')
# Path chars allowed: letters, digits, _, -, ., /. Single backtick wrappers
# (common in markdown answers) are tolerated by the search but not captured.
_CITATION_RE = re.compile(r"([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+):(\d+)(?:-\d+)?")


def extract_citations(text: str) -> set[str]:
    """Pull every ``path:line`` citation from ``text``.

    A range like ``foo.py:42-50`` collapses to ``foo.py:42`` — matching
    is by start-line so an expected single-line citation still finds it.
    """
    return {f"{m.group(1)}:{m.group(2)}" for m in _CITATION_RE.finditer(text)}


def citation_match(actual_text: str, expected: Iterable[str]) -> bool:
    """True iff at least one expected citation appears in ``actual_text``.

    Each expected entry is normalised the same way as extracted citations
    (range -> start line), so case authors can write ``foo.py:42-50`` and
    it matches either ``foo.py:42`` or ``foo.py:42-50`` in the answer.
    """
    actual = extract_citations(actual_text)
    expected_set = {_normalise(e) for e in expected}
    return bool(actual & expected_set)


def _normalise(citation: str) -> str:
    """``foo.py:42-50`` -> ``foo.py:42``; pass-through for already-single."""
    if ":" not in citation:
        return citation
    path, line_part = citation.rsplit(":", 1)
    if "-" in line_part:
        line_part = line_part.split("-", 1)[0]
    return f"{path}:{line_part}"
