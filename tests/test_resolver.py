"""End-to-end tests for the callee resolution pass."""

from codebase_explainer.index_repo import index_repo
from codebase_explainer.resolver import resolve_callees
from codebase_explainer.schema import connect


def _setup_repo(tmp_path, files: dict[str, str]):
    repo = tmp_path / "repo"
    repo.mkdir()
    for rel, content in files.items():
        full = repo / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
    db = tmp_path / "idx.sqlite3"
    index_repo(repo, db)
    return db


def _callee_id(conn, callee_name: str):
    row = conn.execute(
        "SELECT callee_id FROM calls WHERE callee_name = ? LIMIT 1",
        (callee_name,),
    ).fetchone()
    return row["callee_id"] if row else None


def _symbol_id(conn, qualified_name: str):
    row = conn.execute(
        "SELECT id FROM symbols WHERE qualified_name = ?",
        (qualified_name,),
    ).fetchone()
    return row["id"] if row else None


def test_resolves_local_function_in_same_file(tmp_path):
    db = _setup_repo(
        tmp_path,
        {"app.py": "def helper():\n    return 1\n\ndef main():\n    return helper()\n"},
    )
    with connect(db) as conn:
        assert _callee_id(conn, "helper") == _symbol_id(conn, "app.helper")


def test_resolves_self_method_call_to_class_method(tmp_path):
    db = _setup_repo(
        tmp_path,
        {
            "models.py": (
                "class User:\n"
                "    def save(self):\n"
                "        self.flush()\n"
                "    def flush(self):\n"
                "        pass\n"
            )
        },
    )
    with connect(db) as conn:
        assert _callee_id(conn, "self.flush") == _symbol_id(conn, "models.User.flush")


def test_resolves_cls_method_call(tmp_path):
    db = _setup_repo(
        tmp_path,
        {
            "models.py": (
                "class User:\n"
                "    @classmethod\n"
                "    def create(cls):\n"
                "        return cls.build()\n"
                "    @classmethod\n"
                "    def build(cls):\n"
                "        return cls()\n"
            )
        },
    )
    with connect(db) as conn:
        assert _callee_id(conn, "cls.build") == _symbol_id(conn, "models.User.build")


def test_resolves_dotted_callee_via_module_import(tmp_path):
    db = _setup_repo(
        tmp_path,
        {
            "lib/__init__.py": "",
            "lib/tools.py": "def hammer(): pass\n",
            "main.py": "from lib import tools\n\ndef run():\n    tools.hammer()\n",
        },
    )
    with connect(db) as conn:
        assert _callee_id(conn, "tools.hammer") == _symbol_id(conn, "lib.tools.hammer")


def test_resolves_aliased_module_import(tmp_path):
    db = _setup_repo(
        tmp_path,
        {
            "lib/__init__.py": "",
            "lib/tools.py": "def hammer(): pass\n",
            "main.py": "from lib import tools as t\n\ndef run():\n    t.hammer()\n",
        },
    )
    with connect(db) as conn:
        assert _callee_id(conn, "t.hammer") == _symbol_id(conn, "lib.tools.hammer")


def test_resolves_direct_from_import(tmp_path):
    db = _setup_repo(
        tmp_path,
        {
            "lib/__init__.py": "",
            "lib/tools.py": "def hammer(): pass\n",
            "main.py": "from lib.tools import hammer\n\ndef run():\n    hammer()\n",
        },
    )
    with connect(db) as conn:
        assert _callee_id(conn, "hammer") == _symbol_id(conn, "lib.tools.hammer")


def test_resolves_class_constructor_imported_directly(tmp_path):
    db = _setup_repo(
        tmp_path,
        {
            "models.py": "class User:\n    pass\n",
            "main.py": "from models import User\n\ndef run():\n    User()\n",
        },
    )
    with connect(db) as conn:
        assert _callee_id(conn, "User") == _symbol_id(conn, "models.User")


def test_external_calls_left_unresolved(tmp_path):
    db = _setup_repo(
        tmp_path,
        {
            "main.py": (
                "import os\n\n"
                "def run():\n"
                "    print('hi')\n"
                "    os.path.join('a', 'b')\n"
            )
        },
    )
    with connect(db) as conn:
        assert _callee_id(conn, "print") is None
        assert _callee_id(conn, "os.path.join") is None


def test_running_resolve_pass_again_resolves_nothing_new(tmp_path):
    db = _setup_repo(tmp_path, {"app.py": "def f(): pass\ndef g(): f()\n"})
    with connect(db) as conn:
        # index_repo already ran the resolution pass. A second pass has
        # nothing left to do.
        assert resolve_callees(conn) == 0
        # And the original resolution still holds.
        assert _callee_id(conn, "f") == _symbol_id(conn, "app.f")


def test_index_repo_reports_resolved_count_in_stats(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    (repo / "app.py").write_text(
        "def helper(): pass\n"
        "def main():\n"
        "    helper()\n"   # resolvable: app.helper
        "    print('hi')\n"  # external: stays NULL
    )
    db = tmp_path / "idx.sqlite3"
    stats = index_repo(repo, db)
    assert stats.calls == 2
    assert stats.resolved_calls == 1
