"""Render a list of CaseResult into a single markdown string.

No HTML, no Jinja, no front-end. Plain markdown that opens cleanly in any
viewer and pastes into a PR description.
"""

from __future__ import annotations

from collections import defaultdict

from codebase_explainer.eval.runner import CaseResult


def render_markdown(results: list[CaseResult]) -> str:
    if not results:
        return "# Eval Report\n\n(no cases)\n"

    total = len(results)
    passes = sum(1 for r in results if r.citation_pass)
    errors = sum(1 for r in results if r.error is not None)
    pass_pct = (100 * passes) // total if total else 0
    avg_latency = sum(r.latency_seconds for r in results) / total

    lines = [
        "# Eval Report",
        "",
        "## Summary",
        f"- Cases: **{total}**",
        f"- Citation pass: **{passes}/{total}** ({pass_pct}%)",
        f"- Errors: **{errors}**",
        f"- Avg latency: **{avg_latency:.2f}s**",
        "",
        _render_by_type(results),
        "",
        "## Per-case",
        "",
    ]
    for r in results:
        lines.extend(_render_case(r))
        lines.append("")
    return "\n".join(lines)


def _render_by_type(results: list[CaseResult]) -> str:
    by_type: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_type[r.case.question_type].append(r)

    out = ["## By question type", "", "| Type | Pass | N |", "|---|---|---|"]
    for qtype in sorted(by_type):
        rs = by_type[qtype]
        p = sum(1 for r in rs if r.citation_pass)
        out.append(f"| `{qtype}` | {p}/{len(rs)} | {len(rs)} |")
    return "\n".join(out)


def _render_case(r: CaseResult) -> list[str]:
    badge = "✅" if r.citation_pass else "❌"
    out = [
        f"### {r.case.id} — {badge}",
        f"- **Type**: `{r.case.question_type}`",
        f"- **Question**: {r.case.question}",
        f"- **Expected**: {', '.join(f'`{c}`' for c in r.case.expected_citations)}",
        f"- **Tools used**: {', '.join(f'`{t}`' for t in r.tool_calls) or '(none)'}",
        f"- **Latency**: {r.latency_seconds:.2f}s",
    ]
    if r.error:
        out.append(f"- **Error**: `{r.error}`")
    elif r.answer:
        out.append("- **Answer**:")
        for line in r.answer.splitlines():
            out.append(f"  > {line}" if line else "  >")
    return out
