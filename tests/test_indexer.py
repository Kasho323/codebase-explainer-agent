from codebase_explainer.indexer import extract_symbols

SAMPLE = '''\
"""Module-level docstring."""


def top_level_func(a, b: int) -> int:
    """Add two numbers."""
    return a + b


class Greeter:
    """A simple greeter."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}"


def outer():
    def inner():
        return 1
    return inner
'''


def _by_qualified(symbols):
    return {s.qualified_name: s for s in symbols}


def test_extracts_top_level_function_with_signature_and_docstring():
    syms = extract_symbols(SAMPLE)
    by_qual = _by_qualified(syms)
    f = by_qual["top_level_func"]
    assert f.kind == "function"
    assert f.docstring == "Add two numbers."
    assert "(a, b: int)" in f.signature
    assert "-> int" in f.signature


def test_class_and_methods_get_correct_kinds_and_qualified_names():
    syms = extract_symbols(SAMPLE)
    by_qual = _by_qualified(syms)
    assert by_qual["Greeter"].kind == "class"
    assert by_qual["Greeter"].docstring == "A simple greeter."
    assert by_qual["Greeter.__init__"].kind == "method"
    assert by_qual["Greeter.greet"].kind == "method"


def test_nested_functions_are_captured_with_dotted_qualified_name():
    syms = extract_symbols(SAMPLE)
    by_qual = _by_qualified(syms)
    assert "outer" in by_qual
    assert "outer.inner" in by_qual
    assert by_qual["outer.inner"].kind == "function"


def test_module_prefix_is_applied_to_top_level_symbols():
    syms = extract_symbols("def hello(): pass", prefix="myapp.utils")
    assert syms[0].qualified_name == "myapp.utils.hello"


def test_line_numbers_are_one_indexed():
    syms = extract_symbols(SAMPLE)
    by_qual = _by_qualified(syms)
    f = by_qual["top_level_func"]
    assert f.start_line == 4  # def starts on line 4 of SAMPLE
    assert f.end_line >= f.start_line


def test_function_without_docstring_returns_none():
    syms = extract_symbols("def f(x):\n    return x\n")
    assert syms[0].docstring is None


def test_empty_source_returns_no_symbols():
    assert extract_symbols("") == []
