# Golden Eval Cases

Frozen, hand-curated cases for measuring agent quality across releases.

## Layout

```
eval/golden_cases/
└── <repo_name>/
    ├── README.md
    └── NN_<type>_<short_name>.toml
```

Each `.toml` is one scored question. See [`basket_graph_analytics/01_def_build_graph.toml`](basket_graph_analytics/01_def_build_graph.toml) for an example.

## Case schema

```toml
id = "basket-01"                                    # globally unique
repo = "Kasho323/basket-graph-analytics"            # informational; runner uses --repo-root
repo_sha = "HEAD"                                   # commit pin (line numbers should hold)
question = "Where is the `build_graph` function defined?"
question_type = "definition_lookup"                 # or call_graph | design_intent | change_impact
expected_citations = ["basket_graph/basket.py:49"]  # any one match = pass
expected_gist = """
Free-form expected-essence text. Used by the LLM-judge in 5b for gist match.
"""
```

## How to run

```bash
# 1. Index the target repo
python -m codebase_explainer index ./basket-graph-analytics --db /tmp/basket.sqlite3

# 2. Run the eval over its cases
ANTHROPIC_API_KEY=sk-ant-... python -m codebase_explainer eval \
    --cases eval/golden_cases/basket_graph_analytics \
    --db /tmp/basket.sqlite3 \
    --repo-root ./basket-graph-analytics \
    --output /tmp/eval-report.md
```

## Roadmap

- **5a (now)**: 3 cases against basket-graph-analytics; deterministic citation_match scorer.
- **5b**: add LLM-judge scorers (faithfulness, gist match). Judge model is configurable, never hardcoded.
- **5c**: expand to ~20 cases across 5 repos (3 own + 2 external pinned commits).

Eval runs are **not** part of CI — they need an Anthropic API key and indexed repos.
