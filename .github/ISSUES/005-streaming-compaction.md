---
title: Add streaming support for real-time compaction
labels: integration
---

## Summary

Support streaming token output while simultaneously running compaction in the background, so that compaction latency doesn't block inference responses.

## Motivation

Currently, compaction runs synchronously when the token budget is exceeded. At high throughput (1,000+ tok/s), even 50ms compaction pauses can cause visible hitches in streaming output. Background compaction would allow the engine to maintain smooth streaming while managing context.

## Proposed Approach

1. Add an async variant of `ChunkLog.compact()` that runs scoring and eviction in a background thread
2. Use a double-buffer pattern: one buffer serves reads while the other compacts
3. Expose a `streaming=True` flag on `CerebrasSession` to enable async compaction
4. Ensure thread safety on the SQLite WAL (read-write separation)

## Technical Considerations

- SQLite WAL already supports concurrent reads + single writer
- Compaction is CPU-bound (TF-IDF scoring), not I/O-bound
- Need to handle the edge case where new chunks arrive during compaction
- Consider `asyncio` vs `threading` (engine is currently sync)

## Acceptance Criteria

- [ ] Async compaction doesn't block streaming output
- [ ] No data races or lost chunks during concurrent access
- [ ] Latency benchmark shows < 5ms p99 pause during streaming
- [ ] Backwards compatible: sync mode still works unchanged
- [ ] Thread safety tests
