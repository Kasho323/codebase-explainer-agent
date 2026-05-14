"""Tests for the minimal Anthropic-compatible-endpoint compat shim.

Covers:
  - ``_is_deepseek_endpoint()`` env-var heuristic in isolation
  - ``Agent.run_turn()`` strips the right kwargs when targeting DeepSeek,
    and leaves them alone when targeting Anthropic (default).

No real API calls — the Anthropic client is a ``MagicMock`` and we
inspect ``client.messages.create.call_args.kwargs``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from codebase_explainer.agent import (
    _DEEPSEEK_INCOMPATIBLE_PARAMS,
    Agent,
    _is_deepseek_endpoint,
)
from codebase_explainer.schema import connect

# -- _is_deepseek_endpoint() -----------------------------------------------


def test_returns_false_when_env_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert _is_deepseek_endpoint() is False


def test_returns_false_when_env_empty(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    assert _is_deepseek_endpoint() is False


def test_returns_false_for_anthropic_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    assert _is_deepseek_endpoint() is False


def test_returns_true_for_deepseek_compat_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    assert _is_deepseek_endpoint() is True


def test_returns_true_case_insensitive(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://API.DEEPSEEK.COM/anthropic")
    assert _is_deepseek_endpoint() is True


def test_returns_true_for_deepseek_with_port(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com:443/anthropic/v1")
    assert _is_deepseek_endpoint() is True


# -- Agent.run_turn() kwargs under each endpoint ---------------------------


def _mock_end_turn_response(text: str = "ok"):
    """Build a minimal Anthropic-shaped response that ends the run_turn loop."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


def test_anthropic_default_keeps_all_three_params(monkeypatch, indexed_repo):
    """Without ANTHROPIC_BASE_URL set, the original Anthropic behaviour is unchanged."""
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    repo, db_path = indexed_repo

    client = MagicMock()
    client.messages.create.return_value = _mock_end_turn_response()

    with connect(db_path) as conn:
        agent = Agent(
            client=client,
            db_conn=conn,
            repo_root=repo,
            model="claude-sonnet-4-6",
            effort="medium",
        )
        agent.run_turn("Where is build_graph?")

    kwargs = client.messages.create.call_args.kwargs
    # All three Anthropic-specific params survive
    assert "thinking" in kwargs
    assert "output_config" in kwargs
    assert "cache_control" in kwargs
    # And core params still present
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert "messages" in kwargs
    assert "tools" in kwargs


def test_deepseek_endpoint_strips_anthropic_only_params(monkeypatch, indexed_repo):
    """With DeepSeek base URL set, the three Anthropic-only kwargs are removed."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    repo, db_path = indexed_repo

    client = MagicMock()
    client.messages.create.return_value = _mock_end_turn_response()

    with connect(db_path) as conn:
        agent = Agent(
            client=client,
            db_conn=conn,
            repo_root=repo,
            model="deepseek-chat",
            effort="medium",
        )
        agent.run_turn("anything")

    kwargs = client.messages.create.call_args.kwargs
    # The three Anthropic-only params are gone
    for stripped in _DEEPSEEK_INCOMPATIBLE_PARAMS:
        assert stripped not in kwargs, f"{stripped!r} should have been stripped"
    # Core params still passed through verbatim — the DeepSeek model is honoured
    assert kwargs["model"] == "deepseek-chat"
    assert "messages" in kwargs
    assert "tools" in kwargs
    assert kwargs["max_tokens"] > 0


def test_deepseek_stripping_does_not_leak_into_next_call(monkeypatch, indexed_repo):
    """The kwargs dict is rebuilt per loop iteration. Switching env between
    turns must take effect on the next ``run_turn``."""
    repo, db_path = indexed_repo
    client = MagicMock()
    client.messages.create.return_value = _mock_end_turn_response()

    with connect(db_path) as conn:
        agent = Agent(
            client=client, db_conn=conn, repo_root=repo,
            model="x", effort="medium",
        )

        # First turn: DeepSeek
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
        agent.run_turn("q1")
        first_kwargs = client.messages.create.call_args.kwargs
        assert "thinking" not in first_kwargs

        # Second turn: switch to Anthropic. Stripped params must be back.
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        client.messages.create.reset_mock()
        client.messages.create.return_value = _mock_end_turn_response()
        agent.run_turn("q2")
        second_kwargs = client.messages.create.call_args.kwargs
        assert "thinking" in second_kwargs
        assert "output_config" in second_kwargs
        assert "cache_control" in second_kwargs
