"""Golden-case evaluation harness.

Loads TOML case files, runs the agent against each, scores answers, and
renders a markdown report. The agent is injected via a factory so tests
can swap in a fake without needing ANTHROPIC_API_KEY.

Public surface kept intentionally small — see ``case.py``, ``scorers.py``,
``runner.py``, ``report.py`` for the four moving parts.
"""
