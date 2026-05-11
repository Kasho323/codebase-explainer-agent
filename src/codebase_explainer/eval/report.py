"""Render a list of CaseResult into a single markdown string.

No HTML, no Jinja, no front-end. Plain markdown that opens cleanly in any
viewer and pastes into a PR description.

If any result carries judge scores (faithfulness / gist), they appear in
the summary and per-case sections. If all scores are ``None`` (judges
skipped), the judge columns are omitted to keep reports tidy.
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

    judges_used = any(
        r.faithfulness_score is not None or r.gist_score is not None
        for r in results
    )

    lines = [
        "# Eval Report",
        "",
        "## Summary",
        f"- Cases: **{total}**",
        f"- Citation pass: **{passes}/{total}** ({pass_pct}%)",
        f"- Errors: **{errors}**",
        f"- Avg latency: **{avg_latency:.2f}s**",
    ]
    if judges_used:
        lines.extend(_summary_judge_lines(results))
    lines.append("")

    lines.append(_render_by_type(results))
    lines.append("")
    lines.append("## Per-case")
    lines.append("")
    for r in results:
        lines.extend(_render_case(r, judges_used=judges_used))
        lines.append("")
    return "\n".join(lines)


def _summary_judge_lines(results: list[CaseResult]) -> list[str]:
    out = []
    faith_scores = [r.faithfulness_score for r in results if r.faithfulness_score is not None]
    gist_scores = [r.gist_score for r in results if r.gist_score is not None]
    if faith_scores:
        avg = sum(faith_scores) / len(faith_scores)
        out.append(
            f"- Avg faithfulness: **{avg:.2f}/5** ({len(faith_scores)}/{len(results)} scored)"
        )
    if gist_scores:
        avg = sum(gist_scores) / len(gist_scores)
        out.append(
            f"- Avg gist match: **{avg:.2f}/5** ({len(gist_scores)}/{len(results)} scored)"
        )
    return out


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


def _render_case(r: CaseResult, *, judges_used: bool) -> list[str]:
    badge = "✅" if r.citation_pass else "❌"
    out = [
        f"### {r.case.id} — {badge}",
        f"- **Type**: `{r.case.question_type}`",
        f"- **Question**: {r.case.question}",
        f"- **Expected**: {', '.join(f'`{c}`' for c in r.case.expected_citations)}",
        f"- **Tools used**: {', '.join(f'`{t}`' for t in r.tool_calls) or '(none)'}",
        f"- **Latency**: {r.latency_seconds:.2f}s",
    ]
    if judges_used:
        out.append(
            f"- **Faithfulness**: {_score_repr(r.faithfulness_score)}  "
            f"·  **Gist match**: {_score_repr(r.gist_score)}"
        )
    if r.error:
        out.append(f"- **Error**: `{r.error}`")
    elif r.answer:
        out.append("- **Answer**:")
        for line in r.answer.splitlines():
            out.append(f"  > {line}" if line else "  >")
    return out


def _score_repr(score: int | None) -> str:
    return f"{score}/5" if score is not None else "n/a"
