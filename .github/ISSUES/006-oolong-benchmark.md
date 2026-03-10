---
title: OOLONG benchmark integration
labels: benchmark, good-first-issue
---

## Summary

Integrate the OOLONG (Object-Oriented Long-context) benchmark suite to evaluate the engine against a standardized long-context benchmark.

## Motivation

Current benchmarks are custom NIAH variants. OOLONG provides a standardized, peer-reviewed benchmark for long-context models that would allow direct comparison with other context management approaches. This strengthens the engine's evaluation story.

## Proposed Approach

1. Review the OOLONG benchmark paper and identify applicable subtasks (those that test context retention over many turns)
2. Create `benchmarks/oolong_integration.py` that adapts OOLONG tasks to use `ChunkLog` + `CerebrasSession`
3. Run the engine against OOLONG tasks and report scores alongside existing benchmarks
4. Add OOLONG results to the README benchmark table

## References

- OOLONG benchmark: https://arxiv.org/abs/2412.18921

## Acceptance Criteria

- [ ] At least 3 OOLONG subtasks integrated
- [ ] Results compared against baseline (no engine) on same tasks
- [ ] Reproducible benchmark script in `benchmarks/`
- [ ] Results added to README
