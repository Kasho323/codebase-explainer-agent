"""Tests for the search_semantic tool dispatcher and Agent integration."""

from __future__ import annotations

from codebase_explainer.embeddings import FakeEmbedder
from codebase_explainer.schema import connect
from codebase_explainer.tools import EMBEDDING_DEPENDENT_TOOLS, TOOL_DEFINITIONS
from codebase_explainer.tools.search_semantic import handle_search_semantic


def test_returns_top_hits_with_scores_and_locations(embedded_repo):
    repo, db_path = embedded_repo
    with connect(db_path) as conn:
        out = handle_search_semantic(
            {"query": "persist user to database", "k": 5},
            db_conn=conn,
            repo_root=repo,
            embedder=FakeEmbedder(),
        )
    # Returns the standard agent-tool envelope.
    assert "semantic match(es)" in out
    # All result lines carry path:line citations.
    assert "models.py:" in out
    # Scores are signed floats with 3 decimals.
    assert "[+" in out or "[-" in out


def test_returns_clear_error_when_embedder_missing(embedded_repo):
    repo, db_path = embedded_repo
    with connect(db_path) as conn:
        out = handle_search_semantic(
            {"query": "anything"},
            db_conn=conn,
            repo_root=repo,
            embedder=None,
        )
    assert out.startswith("Error:")
    assert "embedder" in out.lower()


def test_returns_clear_error_when_index_has_no_embeddings(indexed_repo):
    repo, db_path = indexed_repo
    with connect(db_path) as conn:
        out = handle_search_semantic(
            {"query": "anything"},
            db_conn=conn,
            repo_root=repo,
            embedder=FakeEmbedder(),
        )
    assert out.startswith("Error:")
    assert "no embeddings" in out.lower() or "re-index" in out.lower()


def test_default_k_when_omitted(embedded_repo):
    """Caller can omit `k` and we use the default without crashing."""
    repo, db_path = embedded_repo
    with connect(db_path) as conn:
        out = handle_search_semantic(
            {"query": "save"},
            db_conn=conn,
            repo_root=repo,
            embedder=FakeEmbedder(),
        )
    assert not out.startswith("Error:")


def test_k_clamped_to_hard_max(embedded_repo):
    """Even a huge k value can't exceed MAX_K = 20."""
    repo, db_path = embedded_repo
    with connect(db_path) as conn:
        out = handle_search_semantic(
            {"query": "save", "k": 99999},
            db_conn=conn,
            repo_root=repo,
            embedder=FakeEmbedder(),
        )
    # We don't have 99999 symbols; but we shouldn't crash and the line
    # count should be modest.
    lines = [line for line in out.splitlines() if line.startswith("  [")]
    assert 0 < len(lines) <= 20


# -- Agent's conditional tool registration ------------------------------


def test_tool_definitions_include_search_semantic():
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert "search_semantic" in names


def test_search_semantic_is_in_embedding_dependent_set():
    assert "search_semantic" in EMBEDDING_DEPENDENT_TOOLS


def test_agent_drops_search_semantic_when_no_embedder(indexed_repo):
    """Without an embedder, the Agent shouldn't expose search_semantic.

    Tests the construction-time filter without making any API calls.
    """
    from unittest.mock import MagicMock

    from codebase_explainer.agent import Agent

    _repo, db_path = indexed_repo
    with connect(db_path) as conn:
        agent = Agent(
            client=MagicMock(),
            db_conn=conn,
            repo_root=db_path.parent / "repo",
            embedder=None,
        )
    active_names = {t["name"] for t in agent._tools}
    assert "search_semantic" not in active_names
    # All other tools still present.
    expected = {t["name"] for t in TOOL_DEFINITIONS} - EMBEDDING_DEPENDENT_TOOLS
    assert active_names == expected


def test_agent_keeps_search_semantic_when_embedded_and_embedder_set(embedded_repo):
    from unittest.mock import MagicMock

    from codebase_explainer.agent import Agent

    _repo, db_path = embedded_repo
    with connect(db_path) as conn:
        agent = Agent(
            client=MagicMock(),
            db_conn=conn,
            repo_root=db_path.parent / "repo",
            embedder=FakeEmbedder(),
        )
    active_names = {t["name"] for t in agent._tools}
    assert "search_semantic" in active_names


def test_agent_drops_search_semantic_if_embedder_set_but_no_embeddings(indexed_repo):
    """Embedder configured but DB has no rows -> still drop the tool, since
    every search would fail. Belt and braces against half-configured setups."""
    from unittest.mock import MagicMock

    from codebase_explainer.agent import Agent

    _repo, db_path = indexed_repo
    with connect(db_path) as conn:
        agent = Agent(
            client=MagicMock(),
            db_conn=conn,
            repo_root=db_path.parent / "repo",
            embedder=FakeEmbedder(),
        )
    active_names = {t["name"] for t in agent._tools}
    assert "search_semantic" not in active_names
