---
title: Support for multi-file context tracking
labels: integration, scorer
---

## Summary

Extend `ChunkLog` to track context across multiple files or context sources, with per-file budgets and cross-file relevance scoring.

## Motivation

In real coding sessions, context comes from multiple files (source code, docs, test output, terminal logs). Currently, all chunks share a single flat budget. Multi-file tracking would allow per-source budgets and smarter cross-referencing.

## Proposed Approach

1. Add a `source` field to `ChunkEntry` (e.g., file path, "terminal", "user")
2. Support per-source token budgets in `ChunkLog` configuration
3. Implement cross-source relevance: when a user mentions a file, boost chunks from that source
4. Add source-aware compaction: evict from over-budget sources first

## Example

```python
log = ChunkLog(
    max_tokens=8000,
    source_budgets={
        "code": 4000,    # Half budget for code
        "docs": 2000,    # Quarter for docs
        "chat": 2000,    # Quarter for conversation
    }
)
log.append("user", "Check the auth module", source="chat")
log.append("system", open("auth.py").read(), source="code")
```

## Acceptance Criteria

- [ ] `ChunkEntry` has a `source` field
- [ ] Per-source budget enforcement during compaction
- [ ] Cross-source relevance boosting
- [ ] Backwards compatible: single-source mode works unchanged
- [ ] Unit tests for multi-source scenarios
