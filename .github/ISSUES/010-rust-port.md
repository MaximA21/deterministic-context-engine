---
title: Rust port of core engine for sub-1ms latency
labels: integration
---

## Summary

Port the core engine components (`ChunkLog`, keyword scorer, compaction logic) to Rust for sub-1ms scoring latency, with Python bindings via PyO3.

## Motivation

The Python engine achieves ~50ms compaction latency, which is acceptable but becomes noticeable at very high throughput. A Rust port of the hot path (scoring + compaction) would bring latency under 1ms while maintaining the Python API through PyO3 bindings.

## Proposed Approach

### Phase 1: Core Port
1. Port `ChunkLog` (append, compact, retrieve) to Rust
2. Port `score_chunk()` keyword scoring
3. Port `extract_keywords()` regex extraction
4. Use `rusqlite` for SQLite WAL storage

### Phase 2: Python Bindings
1. Use PyO3 + maturin to build Python bindings
2. Expose the same API as the Python `ChunkLog`
3. Transparent fallback: use Rust if available, Python otherwise

### Phase 3: TF-IDF Port
1. Port `GoalGuidedScorer` TF-IDF logic (or use a Rust TF-IDF crate)
2. Benchmark against scikit-learn implementation

## Non-Goals

- Not porting `CerebrasSession` (API client stays in Python)
- Not porting benchmark scripts
- Not changing the SQLite schema

## Acceptance Criteria

- [ ] Core Rust library with ChunkLog, keyword scorer, compaction
- [ ] PyO3 bindings exposing the same Python API
- [ ] Benchmark showing < 1ms p99 compaction latency
- [ ] All existing Python tests pass against Rust backend
- [ ] CI builds for Linux, macOS, Windows
