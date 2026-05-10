"""Shared pytest fixtures.

``indexed_repo`` builds a tiny Python repo and runs the indexer over it.
``embedded_repo`` does the same plus an embedding pass with the
deterministic :class:`FakeEmbedder` — no torch, no model download.
"""

from __future__ import annotations

import pytest

from codebase_explainer.embeddings import FakeEmbedder
from codebase_explainer.index_repo import index_repo


def _make_tiny_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "models.py").write_text(
        '"""User model."""\n'
        "\n"
        "class User:\n"
        '    """A user record."""\n'
        "\n"
        "    def save(self) -> bool:\n"
        '        """Persist the user to the database."""\n'
        "        return True\n"
        "\n"
        "    def flush(self) -> None:\n"
        "        pass\n"
    )
    (repo / "main.py").write_text(
        '"""Application entry point."""\n'
        "from models import User\n"
        "\n"
        "def run() -> None:\n"
        '    """Top-level orchestrator that creates a user and saves it."""\n'
        "    u = User()\n"
        "    u.save()\n"
        "    print('done')\n"
    )
    return repo


@pytest.fixture
def indexed_repo(tmp_path):
    """A small Python repo, indexed (no embeddings)."""
    repo = _make_tiny_repo(tmp_path)
    db_path = tmp_path / "idx.sqlite3"
    index_repo(repo, db_path)
    return repo, db_path


@pytest.fixture
def embedded_repo(tmp_path):
    """The same small repo, indexed and embedded with FakeEmbedder."""
    repo = _make_tiny_repo(tmp_path)
    db_path = tmp_path / "idx.sqlite3"
    index_repo(repo, db_path, embedder=FakeEmbedder())
    return repo, db_path
