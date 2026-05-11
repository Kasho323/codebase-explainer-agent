"""Tests for the eval harness.

Covers the deterministic surface — case loading, citation parsing, report
rendering — plus the runner with a fake agent_factory so we don't touch
the Anthropic API in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codebase_explainer.eval.case import GoldenCase, load_case, load_cases
from codebase_explainer.eval.judges import (
    build_faithfulness_prompt,
    build_gist_prompt,
    judge_faithfulness,
    judge_gist,
    parse_score,
)
from codebase_explainer.eval.report import render_markdown
from codebase_explainer.eval.runner import CaseResult, _extract_tool_outputs, run_eval
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


# -- 5b: judge prompt builders + score parser -------------------------------


def test_parse_score_simple_digit():
    assert parse_score("3") == 3
    assert parse_score("5") == 5
    assert parse_score("1") == 1


def test_parse_score_inside_sentence():
    assert parse_score("Score: 4") == 4
    assert parse_score("The answer scores 2 out of 5.") == 2


def test_parse_score_rejects_out_of_range_digits():
    """0, 6, 7, 8, 9 are not valid judge scores; should return None."""
    assert parse_score("0") is None
    assert parse_score("6") is None
    assert parse_score("Score: 9") is None


def test_parse_score_does_not_match_inside_larger_number():
    """'10/10' should not score as 1 (word boundaries protect us)."""
    assert parse_score("10/10") is None
    assert parse_score("answer is 100% confident") is None


def test_parse_score_handles_none_and_empty():
    assert parse_score(None) is None
    assert parse_score("") is None
    assert parse_score("no digits at all") is None


def test_faithfulness_prompt_contains_question_answer_outputs():
    prompt = build_faithfulness_prompt(
        question="Where is foo defined?",
        answer="In foo.py:42.",
        tool_outputs=["foo defined at foo.py:42"],
    )
    assert "Where is foo defined?" in prompt
    assert "In foo.py:42." in prompt
    assert "foo defined at foo.py:42" in prompt
    assert "1-5" in prompt
    assert "single digit" in prompt


def test_faithfulness_prompt_handles_empty_tool_outputs():
    prompt = build_faithfulness_prompt(
        question="q", answer="a", tool_outputs=[]
    )
    assert "no tool calls were made" in prompt
    assert "q" in prompt
    assert "a" in prompt


def test_gist_prompt_contains_question_answer_gist():
    prompt = build_gist_prompt(
        question="Where is foo?",
        answer="foo.py:42",
        expected_gist="foo is at foo.py:42 and returns int.",
    )
    assert "Where is foo?" in prompt
    assert "foo.py:42" in prompt
    assert "foo is at foo.py:42 and returns int." in prompt
    assert "single digit" in prompt


# -- 5b: judge functions with mock client -----------------------------------


class _FakeBlock:
    """Mimics an Anthropic content block (text type)."""
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeClient:
    """Mimics ``Anthropic().messages.create(...)``."""
    def __init__(self, response_text):
        self._response_text = response_text
        self.calls = []  # records each create() call's kwargs

        class _Messages:
            def __init__(self, parent):
                self._parent = parent

            def create(self, **kw):
                self._parent.calls.append(kw)
                return _FakeResponse(self._parent._response_text)

        self.messages = _Messages(self)


class _BrokenClient:
    """Mimics a client whose .messages.create raises."""
    class _Messages:
        def create(self, **_kw):
            raise RuntimeError("api down")

    messages = _Messages()


def test_judge_faithfulness_returns_parsed_score():
    client = _FakeClient("4")
    score = judge_faithfulness(
        client=client, model="some-model",
        question="q", answer="a", tool_outputs=["out"],
    )
    assert score == 4
    # Model id passed through verbatim — never hardcoded inside judges.py
    assert client.calls[0]["model"] == "some-model"
    # Single-shot, low max_tokens
    assert client.calls[0]["max_tokens"] <= 16


def test_judge_gist_returns_parsed_score():
    client = _FakeClient("Score: 3")
    score = judge_gist(
        client=client, model="x", question="q", answer="a", expected_gist="g",
    )
    assert score == 3


def test_judge_returns_none_when_client_raises():
    """Non-fatal failure: a flaky judge call shouldn't crash an eval run."""
    score = judge_faithfulness(
        client=_BrokenClient(), model="x",
        question="q", answer="a", tool_outputs=[],
    )
    assert score is None


def test_judge_returns_none_when_response_has_no_digit():
    client = _FakeClient("I refuse to score this answer.")
    score = judge_faithfulness(
        client=client, model="x",
        question="q", answer="a", tool_outputs=[],
    )
    assert score is None


# -- 5b: runner integration with judges -------------------------------------


def test_runner_skips_judges_when_judge_client_is_none():
    """Default behaviour: no judge_client means no scores, no API calls."""
    results = run_eval(
        [_case(citations=("foo.py:1",))],
        agent_factory=lambda _c: _FakeAgent("see foo.py:1"),
        judge_client=None,
        judge_model="any-model",
    )
    assert results[0].faithfulness_score is None
    assert results[0].gist_score is None


def test_runner_skips_judges_when_judge_model_is_none():
    """Symmetric: client without model also skips."""
    results = run_eval(
        [_case(citations=("foo.py:1",))],
        agent_factory=lambda _c: _FakeAgent("see foo.py:1"),
        judge_client=_FakeClient("5"),
        judge_model=None,
    )
    assert results[0].faithfulness_score is None
    assert results[0].gist_score is None


def test_runner_calls_both_judges_when_configured():
    client = _FakeClient("4")  # both judges return 4
    results = run_eval(
        [_case(citations=("foo.py:1",))],
        agent_factory=lambda _c: _FakeAgent("see foo.py:1"),
        judge_client=client,
        judge_model="judge-x",
    )
    assert results[0].faithfulness_score == 4
    assert results[0].gist_score == 4
    # Two calls: faithfulness + gist
    assert len(client.calls) == 2
    # The model id we passed in was used in both calls
    assert all(c["model"] == "judge-x" for c in client.calls)


def test_runner_does_not_call_judges_when_case_errored():
    """If agent run failed, we don't waste judge calls on an empty answer."""
    client = _FakeClient("5")

    def boom_factory(_c):
        raise RuntimeError("agent init failed")

    results = run_eval(
        [_case()],
        agent_factory=boom_factory,
        judge_client=client,
        judge_model="judge-x",
    )
    assert results[0].error is not None
    assert results[0].faithfulness_score is None
    assert results[0].gist_score is None
    assert client.calls == []


def test_extract_tool_outputs_finds_results_in_user_messages():
    messages = [
        {"role": "user", "content": "question text"},
        {"role": "assistant", "content": [
            # Pydantic-shaped block we ignore
            type("X", (), {"type": "text", "text": "thinking..."})(),
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "abc", "content": "first output"},
            {"type": "tool_result", "tool_use_id": "def", "content": "second output"},
        ]},
    ]
    out = _extract_tool_outputs(messages)
    assert out == ["first output", "second output"]


def test_extract_tool_outputs_returns_empty_for_missing_attribute():
    """Stub agents in tests don't have .messages; runner must not crash."""
    assert _extract_tool_outputs(None) == []
    assert _extract_tool_outputs([]) == []


# -- 5b: report includes judge scores when present --------------------------


def test_report_includes_judge_summary_when_judges_ran():
    results = [
        CaseResult(case=_case(id="a"), answer="foo.py:1",
                   citation_pass=True, latency_seconds=0.1,
                   faithfulness_score=5, gist_score=4),
        CaseResult(case=_case(id="b"), answer="bar.py:1",
                   citation_pass=True, latency_seconds=0.1,
                   faithfulness_score=3, gist_score=3),
    ]
    out = render_markdown(results)
    assert "faithfulness" in out.lower()
    assert "gist match" in out.lower()
    assert "/5" in out  # score scale appears


def test_report_omits_judge_lines_when_judges_skipped():
    """Clean reports for citation-only runs."""
    results = [
        CaseResult(case=_case(id="a"), answer="foo.py:1",
                   citation_pass=True, latency_seconds=0.1),
    ]
    out = render_markdown(results)
    assert "faithfulness" not in out.lower()
    assert "gist match" not in out.lower()


def test_report_handles_partial_judge_scores():
    """One case scored, one not (judge failed) — both render fine."""
    results = [
        CaseResult(case=_case(id="a"), answer="x", citation_pass=False,
                   latency_seconds=0.1, faithfulness_score=4, gist_score=None),
        CaseResult(case=_case(id="b"), answer="y", citation_pass=False,
                   latency_seconds=0.1, faithfulness_score=None, gist_score=2),
    ]
    out = render_markdown(results)
    assert "n/a" in out  # missing score rendered
    assert "4/5" in out
    assert "2/5" in out
