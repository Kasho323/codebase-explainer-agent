# Golden Eval Cases

Frozen test set for measuring quality across releases.

**Structure (target by Week 5)**:
- 5 real open-source repos × 4 question types = 20 cases
- Question types: *symbol-lookup*, *call-graph*, *design-intent*, *change-impact*

Each case is a YAML file:

```yaml
id: httpx-001
repo: https://github.com/encode/httpx
question: "Where is the connection pool managed?"
expected_citations:
  - "httpx/_transports/default.py:120"
expected_gist: |
  The connection pool lives in HTTPTransport via httpcore.ConnectionPool.
  It is created in __init__ and disposed in close().
question_type: design-intent
```

To be populated in Week 5.
