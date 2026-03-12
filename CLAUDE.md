# Deterministic Context Engine

Deterministic context management for high-throughput LLM inference. No model calls in the scoring path -- all decisions are pure math, under 50ms.

## Project Layout

```
engine.py                          # Backwards-compat shim (re-exports from package)
deterministic_context_engine/
  __init__.py                      # Public API: ChunkLog, scorers, sessions
  engine.py                       # Core: ChunkLog, ChunkEntry, DecisionRecord
  _utils.py                       # SHA-256 hashing, token estimation
  scorers/
    bm25.py                       # BM25Scorer (default)
    tfidf.py                      # GoalGuidedScorer (deprecated, use BM25)
    semantic.py                   # SemanticScorer (sentence-transformers)
    entity_aware.py               # EntityAwareScorer
    entities.py                   # EntityExtractor
    keywords.py                   # extract_keywords, score_chunk
  sessions/
    cerebras.py                   # CerebrasSession (lazy import)
    gemini.py                     # GeminiSession (lazy import)
benchmarks/                       # NIAH benchmark scripts
results/                          # Benchmark outputs (JSON + PNG)
tests/                            # pytest suite
```

## Core Concepts

- **ChunkLog**: Append-only context log backed by SQLite WAL. Uses SHA-256 content-addressing, soft/hard threshold compaction, and DecisionRecords for auditability.
- **Scorers**: Rank chunks by goal relevance + corpus uniqueness. `paper_ensemble` is the best scorer (combines BM25 + structural density + MemFly redundancy + SWE-Pruner continuity + Active Context Compression recency). BM25 is the default (`scoring_mode="bm25"`). TF-IDF (`GoalGuidedScorer`) is deprecated.
- **Sessions**: Thin wrappers around Cerebras and Gemini SDKs. Imported lazily to avoid requiring all SDKs.

## Usage

```python
from deterministic_context_engine import ChunkLog

# Best scorer: paper_ensemble (novel combination of 4 papers)
log = ChunkLog(max_tokens=8000, scoring_mode="paper_ensemble")
log.append("user", "Hello!")
```

## Testing

```bash
pytest tests/
```

212 tests, 91% coverage. Always run pytest before committing.

## Adding a New Scorer

1. Create a new file in `deterministic_context_engine/scorers/`.
2. Implement a class with a `score_chunks(goal, chunks, keyword_scores=None) -> dict[str, float]` method. Follow the interface in `BM25Scorer` -- accepts `(goal, chunks, keyword_scores)`, returns `{chunk_hash: score}` with scores in `[0.5, 2.0]`.
3. Register the new `scoring_mode` string in `ChunkLog.__init__` and the compaction path in `ChunkLog._maybe_compact`.
4. Export from `deterministic_context_engine/scorers/__init__.py` and `deterministic_context_engine/__init__.py`.
5. Run all 4 Gemini BM25 benchmarks to validate:
   - `benchmarks/gemini_bm25_dense.py`
   - `benchmarks/gemini_bm25_adversarial.py`
   - `benchmarks/gemini_bm25_boilerplate.py`
   - `benchmarks/gemini_bm25_50turn.py`

## Hard Rules

- Never use model calls (LLM inference) in the scoring path.
- Keep scoring decisions under 50ms.
- Always run `pytest tests/` before committing.
- Always use `gemini/gemini-3.1-flash-lite-preview` for Google/Gemini API calls.

## Dependencies

Core: `numpy`, `rank-bm25`. Optional extras: `tfidf`, `semantic`, `cerebras`, `gemini`, `all`, `dev`. See `pyproject.toml` for details.
