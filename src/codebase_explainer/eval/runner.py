"""Run golden cases through the agent and collect per-case results.

Agent construction is dependency-injected via ``agent_factory`` so tests
can swap in a stub without ANTHROPIC_API_KEY, and so the same runner
serves both the CLI (real Anthropic agent) and any future programmatic
caller.

A handler-level exception is captured as ``CaseResult.error`` rather than
propagated — one broken case shouldn't sink the rest of the eval run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from codebase_explainer.eval.case import GoldenCase
from codebase_explainer.eval.scorers import citation_match


@dataclass
class CaseResult:
    case: GoldenCase
    answer: str
    citation_pass: bool
    latency_seconds: float
    tool_calls: list[str] = field(default_factory=list)
    error: str | None = None


# An agent_factory takes a case and returns something with a
# ``.run_turn(question, on_tool_use=...) -> str`` method. Typed as Any to
# avoid pinning to ``Agent`` (so the test stub doesn't have to subclass it).
AgentFactory = Callable[[GoldenCase], Any]


def run_eval(
    cases: list[GoldenCase],
    *,
    agent_factory: AgentFactory,
) -> list[CaseResult]:
    """Run every case in order and return one CaseResult each."""
    return [_run_one(case, agent_factory) for case in cases]


def _run_one(case: GoldenCase, agent_factory: AgentFactory) -> CaseResult:
    try:
        agent = agent_factory(case)
    except Exception as e:  # noqa: BLE001 — surface to result, don't propagate
        return CaseResult(
            case=case,
            answer="",
            citation_pass=False,
            latency_seconds=0.0,
            error=f"agent_factory failed: {type(e).__name__}: {e}",
        )

    tool_calls: list[str] = []

    def on_tool_use(name: str, _input: dict[str, Any]) -> None:
        tool_calls.append(name)

    t0 = time.perf_counter()
    try:
        answer = agent.run_turn(case.question, on_tool_use=on_tool_use)
    except Exception as e:  # noqa: BLE001 — same rationale
        return CaseResult(
            case=case,
            answer="",
            citation_pass=False,
            latency_seconds=time.perf_counter() - t0,
            tool_calls=tool_calls,
            error=f"agent.run_turn failed: {type(e).__name__}: {e}",
        )
    elapsed = time.perf_counter() - t0

    return CaseResult(
        case=case,
        answer=answer,
        citation_pass=citation_match(answer, case.expected_citations),
        latency_seconds=elapsed,
        tool_calls=tool_calls,
    )
