"""Call-extraction tests for the indexer."""

from codebase_explainer.indexer import extract_file


def _calls(source: str):
    return extract_file(source).calls


def test_module_level_call_has_empty_caller():
    calls = _calls("foo()\n")
    assert len(calls) == 1
    assert calls[0].caller_qualified_name == ""
    assert calls[0].callee_name == "foo"
    assert calls[0].line == 1


def test_call_inside_function_attributes_to_function():
    calls = _calls("def f():\n    bar()\n    baz()\n")
    by_callee = {c.callee_name: c for c in calls}
    assert {"bar", "baz"} <= set(by_callee)
    assert all(c.caller_qualified_name == "f" for c in calls)


def test_method_call_uses_dotted_callee_text():
    calls = _calls(
        "class A:\n"
        "    def save(self):\n"
        "        self.flush()\n"
        "        os.path.join(a, b)\n"
    )
    callee_names = {c.callee_name for c in calls}
    assert "self.flush" in callee_names
    assert "os.path.join" in callee_names
    # All happen inside A.save:
    assert all(c.caller_qualified_name == "A.save" for c in calls)


def test_nested_calls_in_arguments_are_all_captured():
    calls = _calls("def f():\n    h(g(), k())\n")
    callee_names = sorted(c.callee_name for c in calls)
    assert callee_names == ["g", "h", "k"]


def test_decorator_call_is_attributed_to_enclosing_scope():
    # @app.route("/") at module level -> caller is empty (module).
    calls = _calls('@app.route("/")\ndef hello():\n    pass\n')
    decorator_calls = [c for c in calls if c.callee_name == "app.route"]
    assert len(decorator_calls) == 1
    assert decorator_calls[0].caller_qualified_name == ""


def test_call_inside_class_method_is_attributed_to_method():
    # `@classmethod` is a bare reference, not a call (no parens). The only
    # call here is `cls()` inside the method body, attributed to A.make.
    calls = _calls(
        "class A:\n"
        "    @classmethod\n"
        "    def make(cls):\n"
        "        return cls()\n"
    )
    assert len(calls) == 1
    assert calls[0].callee_name == "cls"
    assert calls[0].caller_qualified_name == "A.make"


def test_decorator_with_args_is_a_call_at_enclosing_scope():
    # `@register(name="x")` *is* a call, attributed to the enclosing scope
    # (the class, not the decorated method).
    calls = _calls(
        "class A:\n"
        '    @register(name="x")\n'
        "    def m(self):\n"
        "        pass\n"
    )
    register_calls = [c for c in calls if c.callee_name == "register"]
    assert len(register_calls) == 1
    assert register_calls[0].caller_qualified_name == "A"


def test_no_calls_in_empty_source():
    assert _calls("") == []
