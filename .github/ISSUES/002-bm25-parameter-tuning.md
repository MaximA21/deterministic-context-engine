---
title: Add BM25 parameter tuning (k1, b) benchmark
labels: benchmark, scorer
---

## Summary

Create a benchmark that systematically evaluates different BM25 parameter combinations (k1 and b) against the existing keyword scorer to find optimal settings for context compaction.

## Motivation

The current `score_chunk()` function uses simple keyword matching with linear interpolation. BM25 is a well-studied ranking function that accounts for term frequency saturation (k1) and document length normalization (b). Tuning these parameters could improve scoring accuracy.

## Proposed Approach

1. Implement a `BM25Scorer` class with configurable k1 (default 1.2) and b (default 0.75) parameters
2. Create `benchmarks/niah_bm25.py` that tests a grid of (k1, b) values: k1 in [0.5, 1.0, 1.2, 1.5, 2.0], b in [0.0, 0.25, 0.5, 0.75, 1.0]
3. Compare recall scores against `GoalGuidedScorer` and keyword scorer baselines
4. Output results as a heatmap (k1 x b → recall score)

## Acceptance Criteria

- [ ] `BM25Scorer` class with configurable k1, b
- [ ] Grid search benchmark covering 25 parameter combinations
- [ ] Heatmap visualization of results
- [ ] Comparison table against existing scorers
- [ ] Results documented in PR description
