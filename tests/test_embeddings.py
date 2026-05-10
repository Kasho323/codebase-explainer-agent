"""Tests for the embeddings module: embedder, chunker, store."""

from __future__ import annotations

import numpy as np

from codebase_explainer.embeddings import (
    FakeEmbedder,
    chunk_for_symbol,
    embedding_count,
    load_all_vectors,
    search,
    write_embedding,
)
from codebase_explainer.index_repo import index_repo
from codebase_explainer.schema import connect

# -- FakeEmbedder --------------------------------------------------------


def test_fake_embedder_is_deterministic():
    e = FakeEmbedder()
    a = e.encode(["hello world"])
    b = e.encode(["hello world"])
    assert np.allclose(a, b)


def test_fake_embedder_distinguishes_different_strings():
    e = FakeEmbedder()
    out = e.encode(["hello world", "goodbye world"])
    assert not np.allclose(out[0], out[1])


def test_fake_embedder_returns_unit_vectors():
    e = FakeEmbedder()
    out = e.encode(["one", "two", "three"])
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)


def test_fake_embedder_dim_matches_output_shape():
    e = FakeEmbedder()
    out = e.encode(["x", "y", "z"])
    assert out.shape == (3, e.dim)
    assert out.dtype == np.float32


def test_fake_embedder_handles_empty_string():
    e = FakeEmbedder()
    out = e.encode([""])
    # Zero vector should be normalised to itself, not NaN
    assert out.shape == (1, e.dim)
    assert not np.isnan(out).any()


# -- chunker -------------------------------------------------------------


def test_chunk_for_symbol_includes_qualified_name_and_signature(indexed_repo):
    repo, db_path = indexed_repo
    with connect(db_path) as conn:
        save_id = conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'models.User.save'"
        ).fetchone()["id"]
        chunk = chunk_for_symbol(conn, save_id, repo_root=repo)
    assert chunk is not None
    assert "models.User.save" in chunk
    assert "method " in chunk  # kind prefix
    assert "def save(self) -> bool" in chunk
    assert "Persist the user" in chunk
    assert "return True" in chunk  # body included


def test_chunk_for_missing_symbol_returns_none(indexed_repo):
    _, db_path = indexed_repo
    with connect(db_path) as conn:
        chunk = chunk_for_symbol(conn, 9999999, repo_root=db_path.parent)
    assert chunk is None


# -- store: write / load / search ---------------------------------------


def test_write_embedding_round_trips(indexed_repo):
    _, db_path = indexed_repo
    e = FakeEmbedder()
    with connect(db_path) as conn:
        save_id = conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'models.User.save'"
        ).fetchone()["id"]
        vec = e.encode(["test text"])[0]
        write_embedding(
            conn,
            symbol_id=save_id,
            vector=vec,
            model_name=e.model_name,
            chunk_text="test text",
        )
        conn.commit()

        matrix, ids = load_all_vectors(conn)
    assert ids == [save_id]
    assert matrix.shape == (1, e.dim)
    assert np.allclose(matrix[0], vec)


def test_write_embedding_replaces_on_conflict(indexed_repo):
    """Re-embedding the same symbol overwrites the prior vector, doesn't dup."""
    _, db_path = indexed_repo
    e = FakeEmbedder()
    with connect(db_path) as conn:
        save_id = conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'models.User.save'"
        ).fetchone()["id"]
        write_embedding(
            conn,
            symbol_id=save_id,
            vector=e.encode(["v1"])[0],
            model_name=e.model_name,
            chunk_text="v1",
        )
        write_embedding(
            conn,
            symbol_id=save_id,
            vector=e.encode(["v2"])[0],
            model_name=e.model_name,
            chunk_text="v2",
        )
        conn.commit()
        rows = conn.execute(
            "SELECT chunk_text FROM embeddings WHERE symbol_id = ?", (save_id,)
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["chunk_text"] == "v2"


def test_search_returns_top_k_in_descending_order(embedded_repo):
    """Querying with one symbol's chunk should rank that symbol first."""
    _, db_path = embedded_repo
    e = FakeEmbedder()
    with connect(db_path) as conn:
        save_id = conn.execute(
            "SELECT id FROM symbols WHERE qualified_name = 'models.User.save'"
        ).fetchone()["id"]
        chunk = chunk_for_symbol(conn, save_id, repo_root=db_path.parent / "repo")
        query_vec = e.encode([chunk])[0]
        hits = search(conn, query_vec, k=5)
    assert len(hits) > 0
    # Top hit should be the symbol whose own chunk we used as the query.
    assert hits[0][0] == save_id
    # Scores must be in descending order.
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_search_k_caps_results(embedded_repo):
    _, db_path = embedded_repo
    e = FakeEmbedder()
    with connect(db_path) as conn:
        hits = search(conn, e.encode(["save user"])[0], k=2)
    assert len(hits) <= 2


def test_search_returns_empty_when_no_embeddings(indexed_repo):
    _, db_path = indexed_repo
    e = FakeEmbedder()
    with connect(db_path) as conn:
        hits = search(conn, e.encode(["anything"])[0], k=5)
    assert hits == []


def test_embedding_count_matches_actual_rows(embedded_repo):
    _, db_path = embedded_repo
    with connect(db_path) as conn:
        n = embedding_count(conn)
        rows = conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]
    assert n == rows
    assert n > 0  # the fixture should have embedded at least the User.save method


# -- index_repo --embed integration -------------------------------------


def test_index_repo_embeds_when_embedder_provided(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a.py").write_text("def foo():\n    return 1\n")
    db = tmp_path / "idx.sqlite3"
    stats = index_repo(repo, db, embedder=FakeEmbedder())
    assert stats.embedded_symbols >= 1
    with connect(db) as conn:
        assert embedding_count(conn) == stats.embedded_symbols


def test_index_repo_skips_embedding_when_no_embedder(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a.py").write_text("def foo():\n    return 1\n")
    db = tmp_path / "idx.sqlite3"
    stats = index_repo(repo, db)
    assert stats.embedded_symbols == 0
    with connect(db) as conn:
        assert embedding_count(conn) == 0


def test_re_indexing_keeps_one_embedding_per_symbol(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "a.py").write_text("def foo():\n    return 1\n")
    db = tmp_path / "idx.sqlite3"
    index_repo(repo, db, embedder=FakeEmbedder())
    index_repo(repo, db, embedder=FakeEmbedder())
    with connect(db) as conn:
        # One embedding per symbol; cascading delete + INSERT OR REPLACE
        # keeps it that way across re-indexing.
        n_emb = embedding_count(conn)
        n_sym = conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"]
    assert n_emb == n_sym
