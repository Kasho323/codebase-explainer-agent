"""Tests for the eval harness.

Covers the deterministic surface — case loading, citation parsing, report
rendering — plus the runner with a fake agent_factory so we don't touch
the Anthropic API in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codebase_explainer.eval.case import GoldenCase, load_case, load_cases
from codebase_explainer.eval.report import render_markdown
from codebase_explainer.eval.runner import CaseResult, run_eval
from codebase_explainer.eval.scorers import citation_match, extract_citations

# -- case loader -------------------------------------------------------------


def _write_case(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


VALID_CASE_TOML = """\
id = "test-01"
repo = "owner/test-repo"
repo_sha = "abc123"
question = "Where is foo defined?"
question_type = "definition_lookup"
expected_citations = ["foo.py:10"]
expected_gist = '''
foo is defined in foo.py around line 10.
'''
"""


def test_load_case_parses_all_fields(tmp_path):
    p = _write_case(tmp_path, "c.toml", VALID_CASE_TOML)
    case = load_case(p)
    assert case.id == "test-01"
    assert case.repo == "owner/test-repo"
    assert case.repo_sha == "abc123"
    assert case.question_type == "definition_lookup"
    assert case.expected_citations == ("foo.py:10",)
    assert case.expected_gist.startswith("foo is defined")
    assert case.source_path == p


def test_load_case_defaults_repo_sha_to_head(tmp_path):
    body = VALID_CASE_TOML.replace('repo_sha = "abc123"\n', "")
    case = load_case(_write_case(tmp_path, "c.toml", body))
    assert case.repo_sha == "HEAD"


def test_load_case_strips_question_whitespace(tmp_path):
    body = VALID_CASE_TOML.replace(
        'question = "Where is foo defined?"',
        'question = "   Where is foo defined?   "',
    )
    case = load_case(_write_case(tmp_path, "c.toml", body))
    assert case.question == "Where is foo defined?"


def test_load_case_rejects_missing_field(tmp_path):
    body = VALID_CASE_TOML.replace('id = "test-01"\n', "")
    with pytest.raises(ValueError, match="missing required field"):
        load_case(_write_case(tmp_path, "c.toml", body))


def test_load_case_rejects_unknown_question_type(tmp_path):
    body = VALID_CASE_TOML.replace(
        'question_type = "definition_lookup"',
        'question_type = "random_garbage"',
    )
    with pytest.raises(ValueError, match="question_type"):
        load_case(_write_case(tmp_path, "c.toml", body))


def test_load_case_rejects_empty_citations(tmp_path):
    body = VALID_CASE_TOML.replace(
        'expected_citations = ["foo.py:10"]',
        "expected_citations = []",
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        load_case(_write_case(tmp_path, "c.toml", body))


def test_load_cases_returns_sorted_list(tmp_path):
    _write_case(tmp_path, "b.toml", VALID_CASE_TOML.replace("test-01", "test-02"))
    _write_case(tmp_path, "a.toml", VALID_CASE_TOML.replace("test-01", "test-01"))
    cases = load_cases(tmp_path)
    assert [c.id for c in cases] == ["test-01", "test-02"]


def test_load_cases_empty_dir(tmp_path):
    assert load_cases(tmp_path) == []


def test_load_cases_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "no_such")


def test_load_cases_finds_nested_files(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    _write_case(sub, "deep.toml", VALID_CASE_TOML)
    cases = load_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0].id == "test-01"


def test_ships_basket_graph_cases(tmp_path):
    """Sanity-check: the shipped basket-graph cases all parse."""
    repo_root = Path(__file__).resolve().parents[1]
    cases_dir = repo_root / "eval" / "golden_cases" / "basket_graph_analytics"
    cases = load_cases(cases_dir)
    assert len(cases) == 3
    assert {c.id for c in cases} == {"basket-01", "basket-02", "basket-03"}
    assert all(c.repo == "Kasho323/basket-graph-analytics" for c in cases)


# -- citation extraction & matching -----------------------------------------


def test_extract_citations_basic_path_line():
    text = "See basket_graph/basket.py:49 for details."
    assert extract_citations(text) == {"basket_graph/basket.py:49"}


def test_extract_citations_range_normalises_to_start():
    text = "Lines basket_graph/basket.py:49-60 define build_graph."
    assert extract_citations(text) == {"basket_graph/basket.py:49"}


def test_extract_citations_multiple_in_one_text():
    text = "Compare foo.py:10 and bar/baz.py:200 for details."
    assert extract_citations(text) == {"foo.py:10", "bar/baz.py:200"}


def test_extract_citations_handles_backticks_around_path():
    text = "See `basket_graph/basket.py:49` in the answer."
    assert "basket_graph/basket.py:49" in extract_citations(text)


def test_extract_citations_does_not_eat_partial_line_numbers():
    """foo.py:42 must not match the prefix of foo.py:420."""
    text = "Look at foo.py:420 only."
    assert extract_citations(text) == {"foo.py:420"}
    assert "foo.py:42" not in extract_citations(text)


def test_extract_citations_empty_text():
    assert extract_citations("") == set()


def test_citation_match_pass():
    text = "The answer is at basket_graph/basket.py:49."
    assert citation_match(text, ["basket_graph/basket.py:49"]) is True


def test_citation_match_fail():
    text = "Some other file.py:99 not relevant."
    assert citation_match(text, ["basket_graph/basket.py:49"]) is False


def test_citation_match_accepts_range_in_expected():
    """Expected can list a range; matches the start line in the answer."""
    text = "See basket_graph/basket.py:49 there."
    assert citation_match(text, ["basket_graph/basket.py:49-60"]) is True


def test_citation_match_passes_if_any_one_expected_present():
    text = "Answer cites only basket_graph/basket.py:49."
    assert citation_match(text, ["basket_graph/basket.py:49", "wrong.py:1"]) is True


# -- runner with fake agent --------------------------------------------------


class _FakeAgent:
    """Returns a pre-baked answer; records the question + tool callbacks."""

    def __init__(self, answer: str, *, tool_names_to_emit=()):
        self.answer = answer
        self._tools = list(tool_names_to_emit)

    def run_turn(self, _question, *, on_tool_use=None, on_text=None):
        for t in self._tools:
            if on_tool_use is not None:
                on_tool_use(t, {})
        if on_text is not None:
            on_text(self.answer)
        return self.answer


def _case(id="x-01", citations=("foo.py:1",), qtype="definition_lookup"):
    return GoldenCase(
        id=id,
        repo="x/y",
        repo_sha="HEAD",
        question="q?",
        question_type=qtype,
        expected_citations=tuple(citations),
        expected_gist="g",
    )


def test_runner_passes_when_answer_cites_expected():
    case = _case(citations=("foo.py:42",))
    results = run_eval(
        [case],
        agent_factory=lambda _c: _FakeAgent("The answer is at foo.py:42."),
    )
    assert len(results) == 1
    assert results[0].citation_pass is True
    assert results[0].error is None
    assert results[0].latency_seconds >= 0.0


def test_runner_fails_when_citation_missing():
    case = _case(citations=("foo.py:42",))
    results = run_eval(
        [case],
        agent_factory=lambda _c: _FakeAgent("I don't know."),
    )
    assert results[0].citation_pass is False


def test_runner_captures_tool_names():
    case = _case()
    results = run_eval(
        [case],
        agent_factory=lambda _c: _FakeAgent(
            "see foo.py:1", tool_names_to_emit=("find_definition", "read_file")
        ),
    )
    assert results[0].tool_calls == ["find_definition", "read_file"]


def test_runner_records_factory_failure_as_error():
    def boom(_c):
        raise RuntimeError("init failed")

    results = run_eval([_case()], agent_factory=boom)
    assert results[0].citation_pass is False
    assert results[0].error is not None
    assert "init failed" in results[0].error


def test_runner_records_run_turn_failure_as_error():
    class _BoomAgent:
        def run_turn(self, *_a, **_kw):
            raise RuntimeError("api blew up")

    results = run_eval([_case()], agent_factory=lambda _c: _BoomAgent())
    assert results[0].citation_pass is False
    assert "api blew up" in results[0].error


def test_runner_one_bad_case_does_not_sink_others():
    case_a = _case(id="a-01", citations=("foo.py:1",))
    case_b = _case(id="b-02", citations=("bar.py:2",))

    def factory(case):
        if case.id == "a-01":
            raise RuntimeError("only A breaks")
        return _FakeAgent("see bar.py:2")

    results = run_eval([case_a, case_b], agent_factory=factory)
    assert results[0].error is not None
    assert results[1].error is None
    assert results[1].citation_pass is True


# -- report rendering --------------------------------------------------------


def test_report_empty_results_is_noted():
    out = render_markdown([])
    assert "no cases" in out.lower()


def test_report_includes_summary_counts():
    results = [
        CaseResult(case=_case(id="a"), answer="foo.py:1", citation_pass=True,
                   latency_seconds=1.0),
        CaseResult(case=_case(id="b"), answer="", citation_pass=False,
                   latency_seconds=2.0, error="boom"),
    ]
    out = render_markdown(results)
    assert "**2**" in out  # total
    assert "1/2" in out    # pass count
    assert "boom" in out


def test_report_groups_by_question_type():
    results = [
        CaseResult(case=_case(id="a", qtype="definition_lookup"),
                   answer="foo.py:1", citation_pass=True, latency_seconds=0.1),
        CaseResult(case=_case(id="b", qtype="call_graph"),
                   answer="bar.py:2", citation_pass=False, latency_seconds=0.2),
    ]
    out = render_markdown(results)
    assert "`definition_lookup`" in out
    assert "`call_graph`" in out


def test_report_renders_pass_and_fail_badges():
    results = [
        CaseResult(case=_case(id="pass"), answer="foo.py:1",
                   citation_pass=True, latency_seconds=0.1),
        CaseResult(case=_case(id="fail"), answer="nope", citation_pass=False,
                   latency_seconds=0.1),
    ]
    out = render_markdown(results)
    assert "✅" in out
    assert "❌" in out
