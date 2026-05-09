# Codebase Explainer Agent

> Point it at any Python repo. Ask engineering questions. Get answers grounded in the actual source code.

[![CI](https://github.com/Kasho323/codebase-explainer-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Kasho323/codebase-explainer-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

⚠️ **Status: under active development — first public milestone targeted for mid-June 2026.** Indexer and CLI chat are working today; embedding layer, multi-language grammars, eval harness, and Gradio UI ship in weeks 4–6. See [Roadmap](#roadmap).

---

## Why

When joining a new codebase — onboarding to a new job, contributing to open source, auditing a dependency — most of the work is *navigation*: figuring out what calls what, why a module exists, and where to make a change. LLM chat alone is not enough: it hallucinates without grounding. Naive RAG is not enough: it retrieves text but loses the call graph.

This project combines a **tree-sitter symbol graph** in SQLite with a **Claude agent that navigates it via tool use**, citing specific `path:line` for every claim.

---

## What it does

**Input**: a Python repo on disk.

**Output**: an interactive chat that can answer questions like:
- *"Who calls the `authenticate` function?"*
- *"What's the design intent behind the `Storage` abstraction?"*
- *"If I wanted to add OAuth support, which files would I need to touch?"*
- *"Walk me through what happens when a request hits `/api/users`."*

Every answer cites concrete `path:line` references that you can click to verify.

---

## Architecture

```mermaid
graph LR
    User[User] --> CLI[CLI: chat REPL]
    CLI --> Agent[Agent Loop<br/>Claude Sonnet 4.6<br/>+ adaptive thinking<br/>+ prompt caching]

    Agent <--> Tools{Tool dispatcher}
    Tools --> T1[read_file]
    Tools --> T2[grep]
    Tools --> T3[find_definition]
    Tools --> T4[find_callers]
    Tools --> T5[view_symbol]
    Tools -.-> T6[search_semantic — Week 4]
    Tools -.-> T7[git_log — Week 5]

    T1 --> Files[(Repo on disk)]
    T2 --> Files
    T3 --> Symbols[(SQLite symbol graph)]
    T4 --> Symbols
    T5 --> Symbols
    T5 --> Files
    T6 -.-> Vectors[(FAISS index — Week 4)]

    Indexer[Tree-sitter indexer<br/>+ resolver] --> Symbols
    Indexer -.-> Vectors

    style Agent fill:#e1f5ff
    style Indexer fill:#fff4e1
```

**How it answers**:
1. **Symbol layer** — tree-sitter extracts every function, class, method, import, and call edge into a SQLite graph. A resolution pass turns textual callees (`basket.build_graph`) into `symbols.id` foreign keys via import-aware lookup, so "who calls X" becomes a JOIN, not a regex.
2. **Embedding layer** *(Week 4)* — file and symbol-doc embeddings in FAISS for fuzzy lookups ("anything related to authentication").
3. **Agent layer** — Claude Sonnet 4.6 with five tools (`read_file`, `grep`, `find_definition`, `find_callers`, `view_symbol`) decides which layer to query. Adaptive thinking for multi-step navigation, prompt caching for the tool list and system prompt. Two more tools (`search_semantic`, `git_log`) ship in Weeks 4–5.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Anthropic Claude Sonnet 4.6 | Strong tool use, 200K context, balanced cost. Adaptive thinking + `effort=medium` by default. |
| Code parsing | tree-sitter (Python today; JavaScript + Go in Week 4) | Battle-tested AST extraction, language-agnostic. |
| Symbol store | SQLite | Zero infra, fast joins for call-graph queries, fits in a single file you can `scp` around. |
| Vector store | FAISS (local) — *Week 4* | No external service, fits in RAM for repos under 100k LOC. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` — *Week 4* | Free, runs on CPU, 384-dim. |
| UI | CLI today; Gradio + HF Spaces in Week 6 | One-file chat UI, zero-cost public demo. |

---

## Quick start

```bash
git clone https://github.com/Kasho323/codebase-explainer-agent
cd codebase-explainer-agent
pip install -r requirements.txt

# 1. Index a Python repo on disk. The DB is a single SQLite file.
python -m codebase_explainer index /path/to/some/repo --db /tmp/idx.sqlite3

# 2. Point chat at the indexed DB and the repo root.
export ANTHROPIC_API_KEY=sk-ant-...           # Linux/macOS
# $env:ANTHROPIC_API_KEY = "sk-ant-..."       # Windows PowerShell
python -m codebase_explainer chat \
    --db /tmp/idx.sqlite3 \
    --repo-root /path/to/some/repo
```

The chat REPL opens with a banner showing the index size, then accepts free-form questions. As the agent works, every tool call prints to the screen so you can see exactly what's being read:

```
You> who calls build_graph in this repo?

  -> find_callers(name='build_graph')
  -> read_file(path='basket_graph/main.py', start_line=60, end_line=70)

[agent's grounded answer with path:line citations]

You> /exit
```

Commands inside the REPL: `/exit`, `/reset` (clear conversation), `/help`.

### Try it on a small target

The smaller [`Kasho323/basket-graph-analytics`](https://github.com/Kasho323/basket-graph-analytics) repo (~500 LOC, classic graph-mining problem) is a quick way to see the agent in action without indexing something huge:

```bash
git clone https://github.com/Kasho323/basket-graph-analytics
python -m codebase_explainer index ./basket-graph-analytics --db /tmp/basket.sqlite3
python -m codebase_explainer chat --db /tmp/basket.sqlite3 --repo-root ./basket-graph-analytics
```

Good first questions to try:
- *"Who calls build_graph and what does it return?"*
- *"Explain what dfs._visit does and where it's defined."*
- *"Why does benchmark.py exist? What is it measuring?"*

---

## Evaluation

Quality is tracked against a frozen eval set: **5 real open-source repos × 4 question types = 20 golden cases**. See [`eval/golden_cases/`](eval/golden_cases/).

Each case has a question, an expected `file:line` citation that any correct answer must include, and a free-form "expected gist" judged by Claude as LLM judge. Metrics tracked per release: citation accuracy, answer faithfulness, end-to-end latency p50/p95, token cost per query (split into input / cached / output).

The harness lands in Week 5.

---

## Roadmap

**6 weeks, May 5 → June 15, 2026**

- [x] **Week 1** (4/28–5/4) — Repo scaffolding, README, CI, FastAPI hello-world endpoint.
- [x] **Week 2** (5/5–5/11) — Tree-sitter Python indexer; SQLite schema for symbols/imports/calls; `python -m codebase_explainer index <path>` walks a real repo, persists into SQLite, and runs a callee-resolution pass that fills `calls.callee_id` for in-repo references via self/cls scoping, import aliases, and same-file lookup.
- [x] **Week 3** (5/12–5/18) — Manual tool-use loop on Claude Sonnet 4.6 with four tools (`read_file`, `grep`, `find_definition`, `find_callers`). Adaptive thinking + prompt caching wired in. Interactive REPL via `python -m codebase_explainer chat --db <file> --repo-root <path>` with `/reset` / `/exit`, surfaces every tool call as the agent makes it, cites results as `path:line`. **84 tests passing.**
- [ ] **Week 4** (5/19–5/25) — partially shipped:
  - [x] `view_symbol` tool: one-shot deep lookup returning location, signature, docstring, source body, parent, callers, and callees. **91 tests passing.**
  - [ ] Embedding layer (FAISS + sentence-transformers/all-MiniLM-L6-v2) + `search_semantic` tool.
  - [ ] JavaScript and Go grammars (Go may slip to Week 5 depending on indexer dispatch refactor).
- [ ] **Week 5** (5/26–6/1) — Eval harness with 20 golden cases; `git_log` tool; deeper prompt-caching tuning across long sessions.
- [ ] **Week 6** (6/2–6/15) — Gradio UI; Hugging Face Spaces deploy; demo gif; polish README; companion blog post.

---

## Project layout

```
codebase-explainer-agent/
├── src/codebase_explainer/
│   ├── __init__.py
│   ├── __main__.py          # CLI: python -m codebase_explainer {index,chat}
│   ├── agent.py             # Manual tool-use loop with prompt caching
│   ├── chat.py              # Interactive REPL with /reset and /exit
│   ├── tools/
│   │   ├── __init__.py      # TOOL_HANDLERS registry
│   │   ├── definitions.py   # Tool JSON schemas (byte-stable for caching)
│   │   ├── read_file.py
│   │   ├── grep.py
│   │   ├── find_definition.py
│   │   ├── find_callers.py
│   │   └── view_symbol.py
│   ├── indexer.py           # Tree-sitter → Symbol / Call / Import dataclasses
│   ├── repo_walker.py       # File-tree walker with VCS/cache skip-list
│   ├── persistence.py       # Idempotent FileIndex → SQLite
│   ├── resolver.py          # Resolve textual callees → symbol_id
│   ├── index_repo.py        # Orchestrator: walk + parse + persist + resolve
│   ├── schema.py            # SQLite DDL (v2: cascading FKs)
│   └── main.py              # FastAPI placeholder (Week 6 deployment)
├── tests/                   # 84 pytest cases across 9 files
├── eval/golden_cases/       # Frozen eval set (Week 5)
├── .github/workflows/ci.yml
└── pyproject.toml
```

---

## License

MIT — see [LICENSE](LICENSE).
