# Deterministic Context Engine

Deterministic, model-free context management for LLM agents. Keeps important information in context, evicts noise, no model calls in the critical path.

```
pip install deterministic-context-engine
```

Or install from source:

```
git clone https://github.com/yourusername/deterministic-context-engine.git
cd deterministic-context-engine
pip install -e .
```

Optional dependencies for additional scorers and session backends:

```
pip install deterministic-context-engine[tfidf]     # TF-IDF scorer (scikit-learn)
pip install deterministic-context-engine[cerebras]   # Cerebras session wrapper
pip install deterministic-context-engine[gemini]     # Gemini session wrapper
pip install deterministic-context-engine[all]        # Everything
```

```python
from deterministic_context_engine import ChunkLog

log = ChunkLog(max_tokens=128_000)  # BM25 scoring by default

log.append("user", "auth.py line 42 has an off-by-one in validate_token()")
log.next_turn()
log.append("assistant", "Fixed. The comparison should use < instead of <=.")
log.next_turn()
log.append("user", "Also check the rate limiter in api_gateway.py")
log.next_turn()

# When context fills up, low-value chunks are evicted automatically.
# High-priority content survives. Every decision is logged.
messages = log.get_context()
log.close()
```

## Why

At 1,000+ tokens/second (Cerebras, Codex Spark), you fill a 128k context window in minutes, not hours. Something has to decide what stays and what goes.

Most agent stacks use the model itself for this — summarization, autonomous context selection, retrieval-augmented generation. Those approaches are expensive, non-reproducible, and hard to audit. When the model decides what to remember, you can't predict or debug what it forgets.

This engine takes the opposite approach: **deterministic, model-free compaction**. Every eviction decision is based on BM25 scoring, logged with full context, and reproducible. No GPU required. Each decision completes in under 50ms.

## How It Works

```
message ──▶ append ──▶ [SHA-256 dedup] ──▶ SQLite WAL
                │
                ▼
        token_count > soft_threshold (70%)?
                │ yes
                ▼
        BM25 score all chunks
        ├── 40% goal relevance (vs last user message)
        └── 60% uniqueness (inverse peer similarity)
                │
                ▼
        evict lowest-scored, oldest first
        log eviction to DecisionRecords
                │
                ▼
        token_count > hard_threshold (90%)?
                │ yes
                ▼
        aggressive eviction (priority floor)
```

**Step by step:**

1. **Append** — Every message is content-addressed with SHA-256 and stored in SQLite (WAL mode). Chunks are immutable once written. Duplicates are rejected.

2. **Threshold check** — After each append, the engine checks total token count against two thresholds:
   - **Soft (70%)**: triggers scored eviction of lowest-priority chunks
   - **Hard (90%)**: triggers aggressive eviction while protecting a priority floor

3. **BM25 scoring** — When compaction triggers, every chunk gets a score from two blended signals:
   - **Goal alignment (40%)**: BM25 relevance against the most recent user message
   - **Uniqueness (60%)**: inverse average BM25 similarity to peer chunks — rare, specific content scores higher

4. **Eviction** — Lowest-scored chunks are evicted first, oldest-first within the same score tier. Every eviction is logged in a `DecisionRecord` with timestamp, chunk hash, reason, and context size before/after.

5. **Auditability** — Nothing is silently discarded. Evicted chunks remain in SQLite storage. The full decision history is queryable.

## Benchmark Results

All benchmarks: 10 sessions, 5 needles (critical facts injected at random turns among filler), recall question on final turn.

### Gemini Flash (32k window, ~5x compression)

| Benchmark | BM25 (default) | TF-IDF | Hardcoded (ceiling) | Naive (sliding window) |
|---|---|---|---|---|
| Dense NIAH (30 turns) | **5.0/5** | 5.0/5 | 5.0/5 | 5.0/5 |
| Adversarial (shared keywords) | **3.6/5** | 1.2/5 | 5.0/5 | 1.4/5 |
| Boilerplate (repetitive content) | **4.6/5** | 1.0/5 | 5.0/5 | 2.1/5 |
| 50-Turn Extended (turn 25) | **5.0/5** | 5.0/5 | 5.0/5 | 5.0/5 |
| 50-Turn Extended (turn 50) | **5.0/5** | 5.0/5 | 5.0/5 | **0.0/5** |

BM25 beats TF-IDF by **+2.4** on adversarial and **+3.6** on boilerplate. Naive sliding window loses all needles by turn 50.

### Cerebras llama3.1-8b (8k window)

| Metric | Engine (compaction) | Naive (sliding window) |
|---|---:|---:|
| Avg recall | 5.0/5 | 2.1/5 |
| Context at recall | 4,967 tok | 7,717 tok |
| TTFT | 0.70s | 6.92s |
| Compaction events | 6.4 | 0 |

The engine kept ~5k tokens of context while preserving all 5 needles from ~15k total throughput. The naive baseline kept 7.7k tokens but lost early-planted needles.

### Cross-Model Comparison

| Benchmark | Gemini (32k) | Cerebras (8k) |
|---|---|---|
| Engine recall | 5.0/5 | 5.0/5 |
| Naive recall | 3.4/5 | 2.1/5 |
| 50-turn engine | 5.0/5 | — |
| 50-turn naive | 0.0/5 | — |

The engine maintains perfect recall regardless of model or context window size. Naive baseline degrades worse on smaller windows.

### Live Agent Demo (FastAPI repository)

Tested on a real coding session against FastAPI (~70k lines). An agent indexed 50 files, then answered 15 sequential questions about routing, security, error handling, and dependency injection.

- **15/15 turns completed**, 0 context errors
- 5 compaction events, 61 chunks evicted
- **4/4 recall checks passed** (referencing topics from earlier turns)
- Avg TTFT: 2.01s, ~99k total tokens

## The Scoring Progression

We tested five scoring approaches. Two worked, three failed.

### What worked

**Keywords → TF-IDF → BM25**

| Scorer | Dense | Adversarial | Boilerplate | Latency |
|---|---|---|---|---|
| Keyword extraction | 5.0/5 | 1.0/5 | 2.0/5 | <1ms |
| TF-IDF (goal-guided) | 5.0/5 | 5.0/5* | 3.5/5 | ~16ms |
| **BM25 (winner)** | **5.0/5** | **3.6/5** | **4.6/5** | **~16ms** |

*TF-IDF scored 5.0/5 adversarial on Cerebras 8k but dropped to 1.2/5 on Gemini 32k with higher compression ratios, revealing its fragility.

**Keyword extraction** works on clean conversations where needles have distinct terms (filenames, error messages), but fails when filler shares vocabulary with needles.

**TF-IDF** (goal-guided, with uniqueness signal) improved adversarial handling but two failure modes remained: term frequency inflation on repetitive content (boilerplate) and length bias on variable-length chunks.

**BM25** fixed both failure modes. Term frequency saturation (`k1=1.5`) prevents repetitive terms from dominating scores. Length normalization (`b=0.75`) handles variable-length chunks fairly. BM25 is now the default.

### What We Tried That Failed

**Dense embeddings (all-MiniLM-L6-v2)** — Performed *worse* than TF-IDF. Embedding-space pairwise similarity is 9x higher than TF-IDF (0.27 vs 0.03), compressing the distinctiveness signal that drives eviction decisions. Boilerplate recall: 2.3/5 vs TF-IDF's 3.5/5. Also 25x slower (~418ms vs ~16ms).

| Benchmark | Semantic | Hybrid (TF-IDF + Semantic) | TF-IDF |
|---|---|---|---|
| Boilerplate | 2.3/5 | 2.5/5 | 3.5/5 |
| Adversarial | 4.0/5 | 4.3/5 | 5.0/5 |
| Semantic gap | **4.7/5** | **4.8/5** | 3.4/5 |
| Latency | 418ms | ~400ms | 16ms |

Embeddings only won on the semantic gap benchmark (technical needles vs natural-language recall), but the 25x latency penalty makes them impractical for the compaction hot path.

**Entity-aware scoring** — Regex-based entity extraction (filenames, function names, IPs, dates) combined with TF-IDF. Matched TF-IDF performance exactly but never exceeded it. The fundamental problem: compaction decisions are made *before* the future recall query is known, so knowing which entities appear in a chunk doesn't help predict which chunks will be needed later.

| Benchmark | Entity-Aware | TF-IDF | Difference |
|---|---|---|---|
| Adversarial | 4.1/5 | 4.1/5 | +0.0 |
| Boilerplate | 3.5/5 | 3.5/5 | +0.0 |
| Semantic gap | 5.0/5 | 5.0/5 | +0.0 |
| Fair | 5.0/5 | 5.0/5 | +0.0 |

Added 9ms overhead for zero improvement. Entity extraction is a dead end for this use case.

## Limitations

**Synthetic benchmarks.** Results come from NIAH-style tests with injected needles and generated filler, plus one uncontrolled FastAPI session. Real coding sessions are messier: interleaved file reads, error traces, user corrections, tool outputs.

**Distinguishable needles.** Each needle contains unique, actionable details (line numbers, bug descriptions, employee IDs). Real important information may not separate from noise so cleanly.

**Not yet validated at scale.** Performance over 100+ turn real agent sessions with constant context pressure is unknown.

**Compaction is lossy.** Any eviction strategy can drop something important. The engine makes this tradeoff explicit and auditable through DecisionRecords, but does not eliminate it.

**BM25 is not perfect.** It scores 3.6/5 on adversarial (vs 5.0/5 hardcoded ceiling) and 4.6/5 on boilerplate. Perfect automatic scoring without caller-provided priority hints remains an open problem.

## API Reference

```python
from deterministic_context_engine import ChunkLog

log = ChunkLog(
    db_path=":memory:",       # SQLite path (":memory:" or file path)
    max_tokens=128_000,       # Context window size
    soft_threshold=0.7,       # Trigger scored eviction at 70%
    hard_threshold=0.9,       # Trigger aggressive eviction at 90%
    scoring_mode="bm25",      # "bm25" (default), "tfidf", or None
)

log.append(role, content, priority=1.0)  # Add a chunk
log.next_turn()                           # Advance turn counter
log.get_context()                         # Get priority-managed messages
log.get_context_tokens()                  # Current token count
log.compaction_count                      # Number of compactions
log.close()                               # Close SQLite connection
```

Set `priority > 1.0` for content you know is important. Set `scoring_mode=None` to disable automatic scoring and use manual priorities only.

## Architecture

```
deterministic_context_engine/
├── __init__.py          Public API: from deterministic_context_engine import ChunkLog
├── engine.py            ChunkLog, ChunkEntry, DecisionRecord
├── _utils.py            SHA-256 hashing, token estimation
├── scorers/
│   ├── bm25.py          BM25Scorer (default) — goal relevance + uniqueness
│   ├── tfidf.py         GoalGuidedScorer — TF-IDF variant
│   ├── keywords.py      AutoPriority — regex keyword extraction
│   ├── semantic.py      SemanticScorer — MiniLM embeddings (optional)
│   ├── entities.py      EntityExtractor — regex entity extraction
│   └── entity_aware.py  EntityAwareScorer — TF-IDF + entity extraction
└── sessions/
    ├── cerebras.py      CerebrasSession — Cerebras inference wrapper
    └── gemini.py        GeminiSession — Google Gemini wrapper
```

## Next Steps

- Real coding session evaluation over 100+ turns
- Structured compression (summarize evicted chunks instead of dropping them)
- Agent framework integration (embed in production coding agents)
- Accumulated entity frequency tracking across turns

## References

- **LCM: Lossless Context Management** — Ehrlich, Voltropy (2026). Deterministic context management outperforms model-autonomous memory. [papers.voltropy.com/LCM](https://papers.voltropy.com/LCM)
- **Evaluating AGENTS.md** — Gloaguen et al. (2026). LLM-generated context degrades multi-turn agent performance. [arxiv.org/abs/2602.11988](https://arxiv.org/abs/2602.11988)
- **SWE-Pruner** — Wang et al. (2026). Deterministic context pruning improves coding agent performance. [arxiv.org/abs/2601.16746](https://arxiv.org/abs/2601.16746)
- **Active Context Compression** — Verma (2026). Structured summarization vs raw truncation. [arxiv.org/abs/2601.07190](https://arxiv.org/abs/2601.07190)
