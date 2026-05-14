"""Tests for the demo module.

Hermetic: never imports gradio (it's lazy inside ``run_demo``), never
calls the Anthropic API. CI runs these without ``gradio`` installed.
"""

from __future__ import annotations

from codebase_explainer.__main__ import build_parser

# Plain import must succeed without gradio in the environment.
from codebase_explainer.demo import (
    _build_banner,
    _build_client,
    _format_response,
    _resolve_credentials,
    run_demo,
    run_single_turn,
)

# -- pure formatter ---------------------------------------------------------


def test_format_response_with_no_tools_returns_answer_verbatim():
    out = _format_response(answer="plain answer", tool_calls=[])
    assert out == "plain answer"


def test_format_response_with_tools_prepends_fenced_code_block():
    out = _format_response(
        answer="the answer",
        tool_calls=["→ find_definition(name='foo')"],
    )
    assert out.startswith("```\n")
    assert "→ find_definition(name='foo')" in out
    assert out.endswith("the answer")


def test_format_response_lists_each_tool_call_on_its_own_line():
    out = _format_response(
        answer="done",
        tool_calls=["→ a()", "→ b(x=1)", "→ c()"],
    )
    # All three trace lines visible
    assert "→ a()" in out
    assert "→ b(x=1)" in out
    assert "→ c()" in out
    # Trace block precedes answer
    assert out.index("→ a()") < out.index("done")


# -- run_single_turn: never-call-out cases ----------------------------------


def test_run_single_turn_no_client_returns_warning_not_silent(tmp_path):
    """No API key path: UI gets a clear English warning, not a fabricated answer."""
    out = run_single_turn(
        question="any question",
        client=None,
        conn=None,
        repo_root=tmp_path,
        model="x",
        effort="medium",
        embedder=None,
    )
    assert "ANTHROPIC_API_KEY" in out
    assert out.startswith("⚠️") or "warning" in out.lower() or "not set" in out.lower()


# -- run_demo input validation (fast-fail before gradio import) -------------


def test_run_demo_errors_when_db_missing(tmp_path, capsys):
    code = run_demo(
        db_path=tmp_path / "no_such.sqlite3",
        repo_root=tmp_path,
        model="x",
        effort="medium",
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "index DB not found" in err


def test_run_demo_errors_when_repo_root_is_not_a_directory(tmp_path, capsys):
    db = tmp_path / "idx.sqlite3"
    db.write_bytes(b"")  # any file passes is_file()
    code = run_demo(
        db_path=db,
        repo_root=tmp_path / "no_such_dir",
        model="x",
        effort="medium",
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "is not a directory" in err


# -- _build_client: defensive against env state -----------------------------


def test_build_client_returns_none_without_credentials():
    """Without the has_credentials flag set, we shouldn't even try Anthropic()."""
    assert _build_client(has_credentials=False) is None


# -- _resolve_credentials: env var resolution -------------------------------


def test_resolve_credentials_no_env_vars_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    has_creds, base_url, model_env = _resolve_credentials()
    assert has_creds is False
    assert base_url is None
    assert model_env is None


def test_resolve_credentials_recognises_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-anything")
    has_creds, _, _ = _resolve_credentials()
    assert has_creds is True


def test_resolve_credentials_recognises_auth_token(monkeypatch):
    """ANTHROPIC_AUTH_TOKEN alone should be enough — DeepSeek-compat path."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-deepseek-anything")
    has_creds, _, _ = _resolve_credentials()
    assert has_creds is True


def test_resolve_credentials_reads_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    _, base_url, _ = _resolve_credentials()
    assert base_url == "https://api.deepseek.com/anthropic"


def test_resolve_credentials_reads_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "deepseek-chat")
    _, _, model_env = _resolve_credentials()
    assert model_env == "deepseek-chat"


def test_resolve_credentials_treats_empty_string_as_unset(monkeypatch):
    """Empty env var should be treated as unset, not as a valid value."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    monkeypatch.setenv("ANTHROPIC_MODEL", "")
    _, base_url, model_env = _resolve_credentials()
    assert base_url is None
    assert model_env is None


# -- _build_banner: shows endpoint + credential state without leaking key ---


def test_banner_shows_default_endpoint_when_no_base_url(tmp_path):
    """When ANTHROPIC_BASE_URL is unset, banner says the default endpoint."""
    # Need a real sqlite conn for the COUNT queries
    from codebase_explainer.index_repo import index_repo
    from codebase_explainer.schema import connect

    (tmp_path / "a.py").write_text("def f(): pass\n")
    db = tmp_path / "idx.sqlite3"
    index_repo(tmp_path, db)
    with connect(db) as conn:
        banner = _build_banner(
            repo_root=tmp_path,
            db_path=db,
            model="claude-sonnet-4-6",
            base_url=None,
            conn=conn,
            has_embeddings=False,
            has_credentials=True,
        )
    assert "default (api.anthropic.com)" in banner
    assert "claude-sonnet-4-6" in banner


def test_banner_shows_custom_endpoint_when_base_url_set(tmp_path):
    from codebase_explainer.index_repo import index_repo
    from codebase_explainer.schema import connect

    (tmp_path / "a.py").write_text("def f(): pass\n")
    db = tmp_path / "idx.sqlite3"
    index_repo(tmp_path, db)
    with connect(db) as conn:
        banner = _build_banner(
            repo_root=tmp_path,
            db_path=db,
            model="deepseek-chat",
            base_url="https://api.deepseek.com/anthropic",
            conn=conn,
            has_embeddings=False,
            has_credentials=True,
        )
    assert "https://api.deepseek.com/anthropic" in banner
    assert "deepseek-chat" in banner
    assert "default" not in banner


def test_banner_never_includes_full_api_key(tmp_path, monkeypatch):
    """Even if a key is in env, the banner must never echo it."""
    from codebase_explainer.index_repo import index_repo
    from codebase_explainer.schema import connect

    secret = "sk-ant-this-must-not-appear-in-banner-1234567890"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    (tmp_path / "a.py").write_text("def f(): pass\n")
    db = tmp_path / "idx.sqlite3"
    index_repo(tmp_path, db)
    with connect(db) as conn:
        banner = _build_banner(
            repo_root=tmp_path,
            db_path=db,
            model="x",
            base_url=None,
            conn=conn,
            has_embeddings=False,
            has_credentials=True,
        )
    assert secret not in banner
    assert "this-must-not-appear" not in banner


def test_banner_signals_missing_credentials_clearly(tmp_path):
    from codebase_explainer.index_repo import index_repo
    from codebase_explainer.schema import connect

    (tmp_path / "a.py").write_text("def f(): pass\n")
    db = tmp_path / "idx.sqlite3"
    index_repo(tmp_path, db)
    with connect(db) as conn:
        banner = _build_banner(
            repo_root=tmp_path,
            db_path=db,
            model="x",
            base_url=None,
            conn=conn,
            has_embeddings=False,
            has_credentials=False,
        )
    assert "NOT SET" in banner
    assert "❌" in banner


# -- CLI wiring -------------------------------------------------------------


def test_demo_subcommand_parses_with_defaults():
    parser = build_parser()
    args = parser.parse_args(["demo"])
    assert args.cmd == "demo"
    assert args.port == 7860
    assert args.share is False
    assert args.effort == "medium"


def test_demo_subcommand_accepts_overrides(tmp_path):
    parser = build_parser()
    args = parser.parse_args(
        [
            "demo",
            "--db", str(tmp_path / "x.sqlite3"),
            "--repo-root", str(tmp_path),
            "--model", "claude-haiku-4-5",
            "--effort", "low",
            "--port", "8000",
            "--share",
        ]
    )
    assert args.db.name == "x.sqlite3"
    assert args.repo_root == tmp_path
    assert args.model == "claude-haiku-4-5"
    assert args.effort == "low"
    assert args.port == 8000
    assert args.share is True


def test_demo_subcommand_rejects_invalid_effort():
    parser = build_parser()
    try:
        parser.parse_args(["demo", "--effort", "absurd"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse should have rejected --effort absurd")
