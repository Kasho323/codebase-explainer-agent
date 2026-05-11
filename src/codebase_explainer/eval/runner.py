"""Run golden cases through the agent and collect per-case results.

Agent construction is dependency-injected via ``agent_factory`` so tests
can swap in a stub without ANTHROPIC_API_KEY, and so the same runner
serves both the CLI (real Anthropic agent) and any future programmatic
caller.

A handler-level exception is captured as ``CaseResult.error`` rather than
propagated — one broken case shouldn't sink the rest of the eval run.

LLM judges (5b) are optional: pass ``judge_client`` + ``judge_model`` to
enable. If either is None, judges are skipped and scores stay ``None``.
We never construct a client here or read env vars — the caller resolves
config and threads it through, keeping this module decoupled from any
particular SDK or default model.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from codebase_explainer.eval.case import GoldenCase
from codebase_explainer.eval.judges import judge_faithfulness, judge_gist
from codebase_explainer.eval.scorers import citation_match


@dataclass
class CaseResult:
    case: GoldenCase
    answer: str
    citation_pass: bool
    latency_seconds: float
    tool_calls: list[str] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)
    faithfulness_score: int | None = None
    gist_score: int | None = None
    error: str | None = None


# An agent_factory takes a case and returns something with a
# ``.run_turn(question, on_tool_use=...) -> str`` method. Typed as Any to
# avoid pinning to ``Agent`` (so the test stub doesn't have to subclass it).
AgentFactory = Callable[[GoldenCase], Any]


def run_eval(
    cases: list[GoldenCase],
    *,
    agent_factory: AgentFactory,
    judge_client: Any = None,
    judge_model: str | None = None,
) -> list[CaseResult]:
    """Run every case in order and return one CaseResult each.

    Judges run when both ``judge_client`` and ``judge_model`` are truthy.
    Otherwise they're skipped silently — citation_match still runs.
    """
    use_judges = judge_client is not None and bool(judge_model)
    return [
        _run_one(case, agent_factory, judge_client if use_judges else None, judge_model)
        for case in cases
    ]


def _run_one(
    case: GoldenCase,
    agent_factory: AgentFactory,
    judge_client: Any,
    judge_model: str | None,
) -> CaseResult:
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

    tool_outputs = _extract_tool_outputs(getattr(agent, "messages", []))

    result = CaseResult(
        case=case,
        answer=answer,
        citation_pass=citation_match(answer, case.expected_citations),
        latency_seconds=elapsed,
        tool_calls=tool_calls,
        tool_outputs=tool_outputs,
    )

    if judge_client is not None and judge_model:
        result.faithfulness_score = judge_faithfulness(
            client=judge_client,
            model=judge_model,
            question=case.question,
            answer=answer,
            tool_outputs=tool_outputs,
        )
        result.gist_score = judge_gist(
            client=judge_client,
            model=judge_model,
            question=case.question,
            answer=answer,
            expected_gist=case.expected_gist,
        )

    return result


def _extract_tool_outputs(messages: Any) -> list[str]:
    """Pull every tool_result content out of an Agent's ``messages`` list.

    ``messages`` follows the Anthropic API shape: a list of
    ``{role, content}`` dicts. Tool results live in user-role messages
    whose ``content`` is a list of dicts where ``type == 'tool_result'``.

    Safe against missing attribute (stub agents in tests have no
    ``.messages``) — returns an empty list rather than raising.
    """
    out: list[str] = []
    if not messages:
        return out
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                out.append(str(block.get("content", "")))
    return out
