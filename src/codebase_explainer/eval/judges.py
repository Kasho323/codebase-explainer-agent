"""LLM-based scorers (faithfulness + gist match).

These functions take an Anthropic-compatible client AND a judge model id
as explicit parameters — never read globals, env vars, or hardcoded
defaults inside this module. The CLI / runner is responsible for
resolving config (CLI flag or ``EVAL_JUDGE_MODEL`` env var) and threading
it through. That keeps tests deterministic and makes the "no hardcoded
judge model" rule trivially auditable.

A judge call returns ``int`` (1-5) on success or ``None`` if the API call
fails or the response can't be parsed — never raises. The runner records
``None`` so a flaky judge can't crash a whole eval run.
"""

from __future__ import annotations

import re
from typing import Any

# Match a standalone digit 1-5. Word boundaries prevent matching the first
# digit of multi-digit numbers ("10" should not score as 1). Examples:
#   "5"           -> 5
#   "Score: 4"    -> 4
#   "10/10"       -> no match (returns None)
#   "Answer: 3."  -> 3
_SCORE_RE = re.compile(r"\b([1-5])\b")


def parse_score(text: str | None) -> int | None:
    if not text:
        return None
    match = _SCORE_RE.search(text)
    return int(match.group(1)) if match else None


def build_faithfulness_prompt(
    *, question: str, answer: str, tool_outputs: list[str]
) -> str:
    """Faithfulness = is every claim in the answer grounded in tool output?"""
    if tool_outputs:
        joined = "\n\n---\n\n".join(tool_outputs)
        context_section = f"WHAT THE AGENT'S TOOLS RETURNED:\n{joined}"
    else:
        context_section = "WHAT THE AGENT'S TOOLS RETURNED:\n(no tool calls were made)"

    return (
        "You are scoring a code-explorer agent's answer for faithfulness — "
        "i.e. how much of the answer is grounded in what the agent's tools "
        "actually returned, versus invented or hallucinated.\n"
        "\n"
        f"QUESTION:\n{question}\n"
        "\n"
        f"{context_section}\n"
        "\n"
        f"THE AGENT'S ANSWER:\n{answer}\n"
        "\n"
        "Scoring scale (1-5):\n"
        "  5: every factual claim is directly supported by tool output\n"
        "  4: all major claims supported; minor unsupported details\n"
        "  3: most claims supported but some inferences not grounded\n"
        "  2: significant invention not in tool output\n"
        "  1: answer largely invented or contradicts tool output\n"
        "\n"
        "Respond with ONLY a single digit 1-5. No explanation."
    )


def build_gist_prompt(
    *, question: str, answer: str, expected_gist: str
) -> str:
    """Gist match = does the answer match the curated expected essence?"""
    return (
        "You are scoring a code-explorer agent's answer against the "
        "expected essence of a correct answer.\n"
        "\n"
        f"QUESTION:\n{question}\n"
        "\n"
        f"EXPECTED ESSENCE:\n{expected_gist}\n"
        "\n"
        f"THE AGENT'S ACTUAL ANSWER:\n{answer}\n"
        "\n"
        "Scoring scale (1-5):\n"
        "  5: answer covers all key points of the expected essence accurately\n"
        "  4: covers most key points; minor omissions or extra noise\n"
        "  3: partial coverage; some key points missing or wrong\n"
        "  2: marginal overlap; mostly off-topic or wrong\n"
        "  1: doesn't match the expected essence at all\n"
        "\n"
        "Respond with ONLY a single digit 1-5. No explanation."
    )


def judge_faithfulness(
    *,
    client: Any,
    model: str,
    question: str,
    answer: str,
    tool_outputs: list[str],
) -> int | None:
    prompt = build_faithfulness_prompt(
        question=question, answer=answer, tool_outputs=tool_outputs
    )
    return _call_judge(client=client, model=model, prompt=prompt)


def judge_gist(
    *,
    client: Any,
    model: str,
    question: str,
    answer: str,
    expected_gist: str,
) -> int | None:
    prompt = build_gist_prompt(
        question=question, answer=answer, expected_gist=expected_gist
    )
    return _call_judge(client=client, model=model, prompt=prompt)


def _call_judge(*, client: Any, model: str, prompt: str) -> int | None:
    """Single-shot call to the judge model; returns parsed score or ``None``.

    Catches any exception so a flaky judge doesn't crash the eval run. The
    runner records ``None`` and the report surfaces it as a missing score.
    """
    try:
        response = client.messages.create(
            model=model,
            max_tokens=8,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:  # noqa: BLE001 — judge failure is non-fatal
        return None

    text_parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", "") or "")
    return parse_score("".join(text_parts))
