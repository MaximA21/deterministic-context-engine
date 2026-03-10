---
title: "Adversarial scorer: improve 3.6/5 to 4.5+"
labels: scorer, benchmark
---

## Summary

The adversarial benchmark (where filler shares keywords with needles) currently scores 3.6/5 with keyword scoring. The goal is to reach 4.5+ by improving how the scorer distinguishes needles from keyword-matched filler.

## Context

In adversarial scenarios, filler text is deliberately crafted to share vocabulary with needles. Simple keyword matching fails because it can't distinguish "mentions the same terms" from "contains the actual critical information." The `GoalGuidedScorer` handles this via TF-IDF uniqueness, but the keyword scorer alone does not.

## Proposed Approaches

1. **Positional weighting**: Boost chunks where keywords appear in structured patterns (e.g., "bug in X at line Y") vs. casual mentions
2. **Keyword co-occurrence**: Score based on how many distinct keywords co-occur in a single chunk, not just individual matches
3. **Structural signals**: Detect structured content patterns (error messages, stack traces, action items) and boost them
4. **Hybrid approach**: Combine keyword score with lightweight structural analysis

## Acceptance Criteria

- [ ] Adversarial benchmark score >= 4.5/5 (average over 10 sessions)
- [ ] No regression on dense NIAH benchmark (must stay at 5.0/5)
- [ ] No regression on goal-guided fair benchmark
- [ ] Performance: scoring must complete in < 10ms per chunk
- [ ] Approach documented with rationale
