from codebase_explainer.schema import SCHEMA_VERSION, connect, init_db


def test_init_db_creates_expected_tables(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    init_db(db_path)
    with connect(db_path) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"schema_meta", "files", "symbols", "imports", "calls"}.issubset(tables)


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    init_db(db_path)
    init_db(db_path)
    with connect(db_path) as conn:
        versions = [r["version"] for r in conn.execute("SELECT version FROM schema_meta")]
    assert versions == [SCHEMA_VERSION]


def test_foreign_keys_enabled(tmp_path):
    db_path = tmp_path / "index.sqlite3"
    init_db(db_path)
    with connect(db_path) as conn:
        result = conn.execute("PRAGMA foreign_keys").fetchone()
    assert result[0] == 1
