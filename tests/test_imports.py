"""Import-extraction tests for the indexer."""

from codebase_explainer.indexer import extract_file


def _imports(source: str):
    return extract_file(source).imports


def test_simple_import():
    imps = _imports("import os\n")
    assert len(imps) == 1
    assert imps[0].module == "os"
    assert imps[0].name is None
    assert imps[0].alias is None
    assert imps[0].line == 1


def test_dotted_import():
    imps = _imports("import os.path\n")
    assert imps[0].module == "os.path"
    assert imps[0].alias is None


def test_aliased_import():
    imps = _imports("import os.path as p\n")
    assert imps[0].module == "os.path"
    assert imps[0].alias == "p"


def test_multiple_imports_on_one_line_become_separate_records():
    imps = _imports("import os, sys\n")
    modules = sorted(i.module for i in imps)
    assert modules == ["os", "sys"]


def test_from_import_records_module_and_name():
    imps = _imports("from os.path import join\n")
    assert len(imps) == 1
    assert imps[0].module == "os.path"
    assert imps[0].name == "join"
    assert imps[0].alias is None


def test_from_import_with_multiple_names():
    imps = _imports("from os.path import join, dirname\n")
    names = sorted(i.name for i in imps)
    assert names == ["dirname", "join"]
    assert all(i.module == "os.path" for i in imps)


def test_from_import_with_alias():
    imps = _imports("from os.path import join as j\n")
    assert imps[0].name == "join"
    assert imps[0].alias == "j"


def test_relative_import_keeps_dot_in_module():
    imps = _imports("from . import config\n")
    assert imps[0].module == "."
    assert imps[0].name == "config"


def test_wildcard_import_records_star_as_name():
    imps = _imports("from os.path import *\n")
    assert len(imps) == 1
    assert imps[0].module == "os.path"
    assert imps[0].name == "*"


def test_lazy_import_inside_function_is_still_captured():
    imps = _imports("def f():\n    import json\n    return json.dumps({})\n")
    assert len(imps) == 1
    assert imps[0].module == "json"
    assert imps[0].line == 2
