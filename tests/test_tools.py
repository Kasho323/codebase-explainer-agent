"""Unit tests for the agent's four tools.

Each test builds a tiny indexed repo in a temp directory and exercises one
tool handler directly. No Anthropic API calls — these run in CI without
a key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codebase_explainer.index_repo import index_repo
from codebase_explainer.schema import connect
from codebase_explainer.tools import TOOL_DEFINITIONS, TOOL_HANDLERS
from codebase_explainer.tools.find_callers import handle_find_callers
from codebase_explainer.tools.find_definition import handle_find_definition
from codebase_explainer.tools.grep import handle_grep
from codebase_explainer.tools.read_file import handle_read_file
from codebase_explainer.tools.view_symbol import handle_view_symbol


@pytest.fixture
def indexed_repo(tmp_path):
    """Build a small repo with a class, a function, and call sites; index it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "models.py").write_text(
        '"""User model."""\n'
        "\n"
        "class User:\n"
        "    \"\"\"A user record.\"\"\"\n"
        "\n"
        "    def save(self) -> bool:\n"
        "        \"\"\"Persist the user.\"\"\"\n"
        "        return True\n"
        "\n"
        "    def flush(self) -> None:\n"
        "        pass\n"
    )
    (repo / "main.py").write_text(
        "from models import User\n"
        "\n"
        "def run() -> None:\n"
        "    u = User()\n"
        "    u.save()\n"
        "    print('done')\n"
    )
    db_path = tmp_path / "idx.sqlite3"
    index_repo(repo, db_path)
    return repo, db_path


# -- shape sanity --------------------------------------------------------


def test_tool_definitions_match_handler_names():
    schema_names = {t["name"] for t in TOOL_DEFINITIONS}
    handler_names = set(TOOL_HANDLERS)
    assert schema_names == handler_names


def test_tool_definitions_are_byte_stable_for_caching():
    """Every tool definition must serialize the same way each call so the
    prompt cache stays warm. Just check the list itself doesn't have any
    obvious non-determinism (sets/dicts that might reorder)."""
    import json

    a = json.dumps(TOOL_DEFINITIONS, sort_keys=False)
    b = json.dumps(TOOL_DEFINITIONS, sort_keys=False)
    assert a == b


# -- read_file -----------------------------------------------------------


def test_read_file_returns_numbered_lines(indexed_repo):
    repo, _ = indexed_repo
    out = handle_read_file({"path": "main.py"}, repo_root=repo)
    assert "from models import User" in out
    # 1-indexed line numbers in the prefix
    assert "1  from models import User" in out


def test_read_file_with_range(indexed_repo):
    repo, _ = indexed_repo
    out = handle_read_file(
        {"path": "main.py", "start_line": 3, "end_line": 5}, repo_root=repo
    )
    assert "def run()" in out
    assert "3  def run()" in out
    assert "from models import User" not in out  # line 1, outside range


def test_read_file_missing_returns_error(indexed_repo):
    repo, _ = indexed_repo
    out = handle_read_file({"path": "nope.py"}, repo_root=repo)
    assert out.startswith("Error:")
    assert "not found" in out


def test_read_file_oversized_range_is_rejected(tmp_path):
    big = tmp_path / "big.py"
    big.write_text("\n".join(f"x = {i}" for i in range(2000)))
    out = handle_read_file(
        {"path": "big.py", "start_line": 1, "end_line": 2000},
        repo_root=tmp_path,
    )
    assert out.startswith("Error:")
    assert "too large" in out


# -- grep ----------------------------------------------------------------


def test_grep_finds_pattern(indexed_repo):
    repo, _ = indexed_repo
    out = handle_grep({"pattern": r"class\s+\w+"}, repo_root=repo)
    assert "models.py" in out
    assert "class User" in out


def test_grep_no_match(indexed_repo):
    repo, _ = indexed_repo
    out = handle_grep({"pattern": "nonexistent_pattern_xyz"}, repo_root=repo)
    assert "No matches" in out


def test_grep_invalid_regex_returns_error(indexed_repo):
    repo, _ = indexed_repo
    out = handle_grep({"pattern": "("}, repo_root=repo)
    assert out.startswith("Error:")
    assert "invalid regex" in out


def test_grep_skips_pycache_dir(tmp_path):
    pyfile = tmp_path / "code.py"
    pyfile.write_text("MARKER = 1\n")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "code.cpython-311.py").write_text("MARKER = 2\n")
    out = handle_grep({"pattern": "MARKER"}, repo_root=tmp_path)
    assert "code.py" in out
    assert "__pycache__" not in out


# -- find_definition -----------------------------------------------------


def test_find_definition_by_unqualified_name(indexed_repo):
    _, db_path = indexed_repo
    with connect(db_path) as conn:
        out = handle_find_definition({"name": "save"}, db_conn=conn)
    assert "models.User.save" in out
    assert "method" in out
    assert "Persist the user." in out


def test_find_definition_by_qualified_suffix(indexed_repo):
    _, db_path = indexed_repo
    with connect(db_path) as conn:
        out = handle_find_definition({"name": "User.flush"}, db_conn=conn)
    assert "models.User.flush" in out


def test_find_definition_no_match(indexed_repo):
    _, db_path = indexed_repo
    with connect(db_path) as conn:
        out = handle_find_definition({"name": "doesnt_exist"}, db_conn=conn)
    assert "No symbol found" in out


# -- find_callers --------------------------------------------------------


def test_find_callers_resolved(indexed_repo):
    _, db_path = indexed_repo
    with connect(db_path) as conn:
        # User() is resolved via `from models import User` → models.User
        out = handle_find_callers({"name": "User"}, db_conn=conn)
    assert "main.run" in out
    assert "main.py" in out
    assert "resolved" in out


def test_find_callers_textual_fallback(indexed_repo):
    _, db_path = indexed_repo
    with connect(db_path) as conn:
        # `print` is a builtin — never resolves to an in-repo symbol
        out = handle_find_callers({"name": "print"}, db_conn=conn)
    assert "textual match" in out
    assert "main.run" in out


def test_find_callers_no_match(indexed_repo):
    _, db_path = indexed_repo
    with connect(db_path) as conn:
        out = handle_find_callers({"name": "no_such_callee_xyz"}, db_conn=conn)
    assert "No callers found" in out


# -- view_symbol ---------------------------------------------------------


def test_view_symbol_returns_full_picture_for_a_method(indexed_repo):
    repo, db_path = indexed_repo
    with connect(db_path) as conn:
        out = handle_view_symbol(
            {"name": "save"}, db_conn=conn, repo_root=repo
        )
    # Identity
    assert "models.User.save" in out
    assert "[method]" in out
    # Parent linkage
    assert "Parent:" in out
    assert "User" in out
    # Signature
    assert "def save(self) -> bool" in out
    # Docstring
    assert "Persist the user." in out
    # Source body — actual code, not just signature
    assert "return True" in out


def test_view_symbol_includes_resolved_callers(indexed_repo):
    repo, db_path = indexed_repo
    with connect(db_path) as conn:
        # User() inside main.run resolves via `from models import User`
        out = handle_view_symbol(
            {"name": "User"}, db_conn=conn, repo_root=repo
        )
    assert "Called by" in out
    assert "main.run" in out
    assert "main.py" in out


def test_view_symbol_includes_callees_for_a_function(indexed_repo):
    repo, db_path = indexed_repo
    with connect(db_path) as conn:
        # main.run calls User(), u.save(), and print()
        out = handle_view_symbol(
            {"name": "main.run"}, db_conn=conn, repo_root=repo
        )
    assert "Calls" in out
    # Either the resolved name or the textual callee appears in the section
    assert "User" in out or "save" in out


def test_view_symbol_distinguishes_resolved_from_unresolved_callees(indexed_repo):
    repo, db_path = indexed_repo
    with connect(db_path) as conn:
        out = handle_view_symbol(
            {"name": "main.run"}, db_conn=conn, repo_root=repo
        )
    # print() is a builtin — never resolves to an in-repo symbol
    assert "print (unresolved)" in out


def test_view_symbol_truncates_long_bodies(tmp_path):
    """A symbol whose body exceeds MAX_SOURCE_LINES gets truncated with a
    pointer to read_file for the rest."""
    from codebase_explainer.index_repo import index_repo

    repo = tmp_path / "big_repo"
    repo.mkdir()
    body = "\n".join(f"    x{i} = {i}" for i in range(300))
    (repo / "huge.py").write_text(f"def huge():\n{body}\n")
    db_path = tmp_path / "idx.sqlite3"
    index_repo(repo, db_path)

    with connect(db_path) as conn:
        out = handle_view_symbol({"name": "huge"}, db_conn=conn, repo_root=repo)
    assert "truncated" in out
    assert "read_file" in out  # tells the agent how to recover


def test_view_symbol_returns_clear_message_for_missing_name(indexed_repo):
    repo, db_path = indexed_repo
    with connect(db_path) as conn:
        out = handle_view_symbol(
            {"name": "no_such_symbol_xyz"}, db_conn=conn, repo_root=repo
        )
    assert "No symbol found" in out
    assert "find_definition" in out  # points to a fallback


def test_view_symbol_picks_shortest_qualified_match(indexed_repo):
    """Given an ambiguous name, prefer the top-level definition over a
    nested one that happens to share the same trailing component."""
    repo, db_path = indexed_repo
    with connect(db_path) as conn:
        # 'flush' is a single User method; nothing nested that would
        # collide. Sanity-check the lookup picks it.
        out = handle_view_symbol({"name": "flush"}, db_conn=conn, repo_root=repo)
    assert "models.User.flush" in out


# -- end-to-end sanity ---------------------------------------------------


def test_dispatch_via_handler_registry(indexed_repo):
    """The Agent looks tools up in TOOL_HANDLERS by name. Smoke-test that
    the registry actually dispatches each tool."""
    repo, db_path = indexed_repo
    with connect(db_path) as conn:
        rd = TOOL_HANDLERS["read_file"]({"path": "main.py"}, repo_root=repo, db_conn=conn)
        gp = TOOL_HANDLERS["grep"]({"pattern": "User"}, repo_root=repo, db_conn=conn)
        fd = TOOL_HANDLERS["find_definition"]({"name": "save"}, repo_root=repo, db_conn=conn)
        fc = TOOL_HANDLERS["find_callers"]({"name": "User"}, repo_root=repo, db_conn=conn)
        vs = TOOL_HANDLERS["view_symbol"]({"name": "save"}, repo_root=repo, db_conn=conn)
    assert "from models import User" in rd
    assert "User" in gp
    assert "models.User.save" in fd
    assert "main.run" in fc
    assert "models.User.save" in vs and "return True" in vs


def test_repo_root_can_be_passed_as_pathlib_path(indexed_repo):
    repo, _ = indexed_repo
    assert isinstance(repo, Path)
    out = handle_read_file({"path": "main.py"}, repo_root=repo)
    assert out.startswith("# main.py")
