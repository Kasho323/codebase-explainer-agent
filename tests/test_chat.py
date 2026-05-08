"""Tests for the chat CLI surface.

Exercise argparse and the file-existence guard rails — the agent loop
itself needs an Anthropic API key, so it isn't covered here. CI runs
without a key.
"""

from __future__ import annotations

from codebase_explainer.__main__ import build_parser
from codebase_explainer.chat import run_chat


def test_chat_subcommand_parses_with_defaults():
    parser = build_parser()
    args = parser.parse_args(["chat"])
    assert args.cmd == "chat"
    assert str(args.db).endswith("codebase-index.sqlite3")
    assert args.effort == "medium"


def test_chat_subcommand_accepts_overrides(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "chat",
            "--db",
            str(tmp_path / "x.sqlite3"),
            "--repo-root",
            str(tmp_path),
            "--model",
            "claude-haiku-4-5",
            "--effort",
            "low",
        ]
    )
    assert args.db.name == "x.sqlite3"
    assert args.repo_root == tmp_path
    assert args.model == "claude-haiku-4-5"
    assert args.effort == "low"


def test_chat_rejects_invalid_effort():
    parser = build_parser()
    try:
        parser.parse_args(["chat", "--effort", "extreme"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse should have rejected --effort extreme")


def test_run_chat_returns_error_when_db_missing(tmp_path, capsys):
    code = run_chat(
        db_path=tmp_path / "missing.sqlite3",
        repo_root=tmp_path,
        model="claude-sonnet-4-6",
        effort="medium",
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "index DB not found" in err


def test_run_chat_returns_error_when_repo_root_not_a_dir(tmp_path, capsys):
    db = tmp_path / "idx.sqlite3"
    db.write_bytes(b"")  # any non-empty file passes the is_file check
    code = run_chat(
        db_path=db,
        repo_root=tmp_path / "no_such_dir",
        model="claude-sonnet-4-6",
        effort="medium",
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "is not a directory" in err
