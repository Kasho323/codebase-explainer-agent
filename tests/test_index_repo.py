"""End-to-end test for the orchestrator: walk -> parse -> persist."""

from codebase_explainer.index_repo import index_repo
from codebase_explainer.schema import connect


def _make_tiny_repo(root):
    """Create a minimal repo with a package and a module that uses it."""
    (root / "myapp").mkdir()
    (root / "myapp" / "__init__.py").write_text("")
    (root / "myapp" / "models.py").write_text(
        "class User:\n"
        "    def save(self):\n"
        "        return True\n"
    )
    (root / "myapp" / "main.py").write_text(
        "from myapp.models import User\n"
        "\n"
        "def run():\n"
        "    u = User()\n"
        "    u.save()\n"
    )
    # Files that should be ignored by walker:
    (root / ".git").mkdir()
    (root / ".git" / "HEAD.py").write_text("garbage")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "stale.py").write_text("garbage")


def test_index_repo_returns_stats_for_each_layer(tmp_path):
    _make_tiny_repo(tmp_path)
    db_path = tmp_path / "idx.sqlite3"

    stats = index_repo(tmp_path, db_path)

    # 3 indexed: __init__.py (empty), models.py, main.py
    assert stats.files == 3
    assert stats.symbols >= 3  # User, save, run at minimum
    assert stats.calls >= 2  # User() and u.save()
    assert stats.imports >= 1  # from myapp.models import User


def test_index_repo_persists_data_queryable_by_caller(tmp_path):
    _make_tiny_repo(tmp_path)
    db_path = tmp_path / "idx.sqlite3"

    index_repo(tmp_path, db_path)

    with connect(db_path) as conn:
        # Symbol User.save lives in myapp/models.py
        save_row = conn.execute(
            "SELECT qualified_name, kind FROM symbols WHERE name='save'"
        ).fetchone()
        assert save_row["qualified_name"] == "myapp.models.User.save"
        assert save_row["kind"] == "method"

        # The call to u.save() inside main.run() is recorded
        save_calls = conn.execute(
            "SELECT caller_qualified_name FROM calls WHERE callee_name='u.save'"
        ).fetchall()
        assert any(r["caller_qualified_name"] == "myapp.main.run" for r in save_calls)


def test_index_repo_skips_directories_in_skip_list(tmp_path):
    _make_tiny_repo(tmp_path)
    db_path = tmp_path / "idx.sqlite3"
    index_repo(tmp_path, db_path)

    with connect(db_path) as conn:
        paths = {r["path"] for r in conn.execute("SELECT path FROM files")}

    assert all(".git" not in p for p in paths)
    assert all("__pycache__" not in p for p in paths)


def test_index_repo_uses_forward_slashes_in_path(tmp_path):
    _make_tiny_repo(tmp_path)
    db_path = tmp_path / "idx.sqlite3"
    index_repo(tmp_path, db_path)

    with connect(db_path) as conn:
        paths = {r["path"] for r in conn.execute("SELECT path FROM files")}

    assert "myapp/models.py" in paths
    assert "myapp/main.py" in paths
    assert all("\\" not in p for p in paths)


def test_re_running_index_repo_keeps_one_record_per_file(tmp_path):
    _make_tiny_repo(tmp_path)
    db_path = tmp_path / "idx.sqlite3"
    index_repo(tmp_path, db_path)
    index_repo(tmp_path, db_path)  # second pass

    with connect(db_path) as conn:
        n_files = conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()["n"]

    assert n_files == 3
