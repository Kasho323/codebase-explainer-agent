# Golden Cases — basket-graph-analytics

3 hand-curated cases against [`Kasho323/basket-graph-analytics`](https://github.com/Kasho323/basket-graph-analytics).
Line numbers verified against the actual source on the commit pinned in each
`.toml`'s `repo_sha` field (currently `HEAD`).

| File | Type | Probes |
|---|---|---|
| `01_def_build_graph.toml` | `definition_lookup` | basic find-definition |
| `02_def_recommend.toml` | `definition_lookup` | function with multiple parameters |
| `03_design_dfs_visit.toml` | `design_intent` | nested function and closure semantics |

This set is intentionally small for Week 5 increment 5a. Increments 5b (LLM
judges) and 5c (15 more cases across 4 more repos) expand from here.
