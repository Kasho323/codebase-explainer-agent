"""Tests for repo walker: file discovery and module-prefix derivation."""

from pathlib import Path

from codebase_explainer.repo_walker import relative_module_prefix, walk_python_files


def _rel(paths, root):
    return sorted(p.relative_to(root).as_posix() for p in paths)


def test_walk_yields_all_python_files(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("")
    (tmp_path / "sub" / "data.txt").write_text("")
    assert _rel(walk_python_files(tmp_path), tmp_path) == ["a.py", "sub/b.py"]


def test_walk_skips_vcs_and_cache_dirs(tmp_path):
    (tmp_path / "a.py").write_text("")
    for d in [".git", "__pycache__", ".venv", ".pytest_cache", "node_modules"]:
        (tmp_path / d).mkdir()
        (tmp_path / d / "x.py").write_text("")
    assert _rel(walk_python_files(tmp_path), tmp_path) == ["a.py"]


def test_walk_skips_egg_info_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("")
    (tmp_path / "src" / "mypackage.egg-info").mkdir()
    (tmp_path / "src" / "mypackage.egg-info" / "PKG-INFO.py").write_text("")
    assert _rel(walk_python_files(tmp_path), tmp_path) == ["src/a.py"]


def test_walk_handles_single_file_argument(tmp_path):
    p = tmp_path / "lonely.py"
    p.write_text("")
    found = list(walk_python_files(p))
    assert found == [p]


def test_walk_returns_empty_for_nonexistent_path(tmp_path):
    assert list(walk_python_files(tmp_path / "does_not_exist")) == []


def test_module_prefix_for_init_drops_filename(tmp_path):
    p = tmp_path / "app" / "__init__.py"
    p.parent.mkdir()
    p.write_text("")
    assert relative_module_prefix(p, tmp_path) == "app"


def test_module_prefix_strips_py_extension(tmp_path):
    p = tmp_path / "src" / "auth" / "login.py"
    p.parent.mkdir(parents=True)
    p.write_text("")
    assert relative_module_prefix(p, tmp_path) == "src.auth.login"


def test_module_prefix_for_top_level_file(tmp_path):
    p = tmp_path / "foo.py"
    p.write_text("")
    assert relative_module_prefix(p, tmp_path) == "foo"


def test_module_prefix_falls_back_when_outside_root(tmp_path):
    file_outside = Path("/elsewhere/bar.py")
    assert relative_module_prefix(file_outside, tmp_path) == "bar"
