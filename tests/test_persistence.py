"""Tests for the SQLite persistence layer."""

from codebase_explainer.indexer import extract_file
from codebase_explainer.persistence import hash_source, write_file_index
from codebase_explainer.schema import connect, init_db


def _setup(tmp_path):
    db_path = tmp_path / "idx.sqlite3"
    init_db(db_path)
    return db_path


def test_persists_symbols_with_correct_parent_links(tmp_path):
    db_path = _setup(tmp_path)
    src = "class Greeter:\n    def hello(self):\n        return 'hi'\n"
    fi = extract_file(src, prefix="mymod")

    with connect(db_path) as conn:
        write_file_index(conn, path="src/mymod.py", file_index=fi)
        conn.commit()
        rows = conn.execute(
            "SELECT id, qualified_name, kind, parent_id FROM symbols ORDER BY id"
        ).fetchall()

    by_qn = {r["qualified_name"]: r for r in rows}
    assert by_qn["mymod.Greeter"]["kind"] == "class"
    assert by_qn["mymod.Greeter"]["parent_id"] is None
    assert by_qn["mymod.Greeter.hello"]["kind"] == "method"
    assert by_qn["mymod.Greeter.hello"]["parent_id"] == by_qn["mymod.Greeter"]["id"]


def test_persists_imports(tmp_path):
    db_path = _setup(tmp_path)
    src = "import os\nfrom pathlib import Path as P\n"
    fi = extract_file(src)

    with connect(db_path) as conn:
        write_file_index(conn, path="a.py", file_index=fi)
        conn.commit()
        rows = conn.execute(
            "SELECT module, name, alias FROM imports ORDER BY id"
        ).fetchall()

    assert (rows[0]["module"], rows[0]["name"], rows[0]["alias"]) == ("os", None, None)
    assert (rows[1]["module"], rows[1]["name"], rows[1]["alias"]) == ("pathlib", "Path", "P")


def test_persists_calls_with_module_level_caller_as_null(tmp_path):
    db_path = _setup(tmp_path)
    src = '@app.route("/")\ndef hello():\n    bar()\n'
    fi = extract_file(src, prefix="m")

    with connect(db_path) as conn:
        write_file_index(conn, path="m.py", file_index=fi)
        conn.commit()
        rows = conn.execute(
            "SELECT caller_qualified_name, callee_name FROM calls ORDER BY id"
        ).fetchall()

    by_callee = {r["callee_name"]: r for r in rows}
    assert by_callee["app.route"]["caller_qualified_name"] is None
    assert by_callee["bar"]["caller_qualified_name"] == "m.hello"


def test_re_indexing_same_path_replaces_previous_data(tmp_path):
    db_path = _setup(tmp_path)
    fi_v1 = extract_file("def old(): pass\n", prefix="m")
    fi_v2 = extract_file("def fresh(): pass\n", prefix="m")

    with connect(db_path) as conn:
        write_file_index(conn, path="m.py", file_index=fi_v1)
        write_file_index(conn, path="m.py", file_index=fi_v2)
        conn.commit()
        rows = conn.execute("SELECT qualified_name FROM symbols").fetchall()
        files = conn.execute("SELECT COUNT(*) AS n FROM files").fetchone()

    qualified = [r["qualified_name"] for r in rows]
    assert qualified == ["m.fresh"]
    assert files["n"] == 1


def test_cascading_delete_drops_child_rows(tmp_path):
    db_path = _setup(tmp_path)
    src = "def f():\n    g()\n"
    fi = extract_file(src, prefix="m")

    with connect(db_path) as conn:
        write_file_index(conn, path="m.py", file_index=fi)
        conn.commit()
        # Sanity: child rows exist.
        assert conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"] == 1
        assert conn.execute("SELECT COUNT(*) AS n FROM calls").fetchone()["n"] == 1

        conn.execute("DELETE FROM files WHERE path = 'm.py'")
        conn.commit()

        assert conn.execute("SELECT COUNT(*) AS n FROM symbols").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM calls").fetchone()["n"] == 0


def test_hash_source_is_deterministic_across_str_and_bytes():
    assert hash_source("def x(): pass") == hash_source(b"def x(): pass")
    assert hash_source("a") != hash_source("b")


def test_content_hash_round_trips_through_db(tmp_path):
    db_path = _setup(tmp_path)
    src = "x = 1\n"
    fi = extract_file(src)
    digest = hash_source(src)

    with connect(db_path) as conn:
        write_file_index(conn, path="x.py", file_index=fi, content_hash=digest)
        conn.commit()
        row = conn.execute("SELECT content_hash FROM files WHERE path='x.py'").fetchone()

    assert row["content_hash"] == digest
