---
title: Implement accumulated entity frequency tracking
labels: scorer, good-first-issue
---

## Summary

Track entity mention frequency across turns and boost priority for entities that appear repeatedly. Currently, `EntityExtractor` identifies entities but doesn't track how often they recur across the conversation.

## Motivation

Entities mentioned multiple times (e.g., a table name referenced in 5 turns) are likely more important than one-off mentions. Accumulated frequency provides a strong signal for compaction decisions without any ML overhead.

## Proposed Approach

1. Add a frequency counter (dict) to `EntityAwareScorer` that accumulates entity counts across `score()` calls
2. Blend frequency signal into the existing entity score: entities seen 3+ times get a boost
3. Expose frequency data in `DecisionRecord` for auditability

## Acceptance Criteria

- [ ] Entity frequency persists across multiple `score()` calls within a session
- [ ] Chunks containing high-frequency entities score higher
- [ ] Frequency data appears in DecisionRecords
- [ ] Unit tests cover frequency accumulation and scoring boost
- [ ] No regression on existing benchmarks
