---
title: Visualization dashboard for DecisionRecords
labels: integration, good-first-issue
---

## Summary

Build an interactive visualization dashboard that renders DecisionRecords as a timeline, showing compaction events, chunk lifecycle, and scoring distributions.

## Motivation

DecisionRecords capture rich data about every append and eviction, but currently require manual JSON inspection. A visual dashboard would make it easy to understand compaction behavior, debug scoring issues, and present results.

## Proposed Approach

1. Create `dashboard.py` using a lightweight web framework (e.g., Flask or a static HTML page with embedded charts)
2. Visualizations:
   - **Timeline**: horizontal bar chart showing chunk lifetime (append → evict)
   - **Score distribution**: histogram of chunk scores at each compaction event
   - **Context usage**: line chart of token count over turns
   - **Eviction log**: table of evicted chunks with reasons and scores
3. Read DecisionRecords from SQLite or exported JSON
4. Support both live (connected to running session) and post-hoc (from saved data) modes

## Acceptance Criteria

- [ ] Dashboard renders at least 3 visualization types
- [ ] Works with existing DecisionRecord format
- [ ] Can load from exported JSON benchmark results
- [ ] No heavy dependencies (matplotlib or plotly for charts, no full SPA framework)
- [ ] Screenshot in PR description
