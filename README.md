## Deterministic Context Management for High-Frequency Inference Models

At 1,000+ tokens per second, long-context inference creates a new systems problem: the active working set fills faster than most agent stacks can manage it. Many current approaches rely on model-mediated memory operations such as summarization or autonomous context selection. Those methods are expensive, hard to audit, and often non-reproducible. This engine takes a different approach: deterministic, model-free context management with append-only logging, threshold-based compaction, and cheap lexical scoring. There are no model calls in the critical path, and each decision completes in under 50 ms.

## Why This Matters Now

Inference hardware has become fast enough that context management is now the bottleneck. OpenAI’s partnership with Cerebras brought wafer-scale inference to production, and Codex Spark reportedly runs at 1,000+ tokens per second with 128k-token windows. At that throughput, you fill context in minutes, not hours.

Recent work points in the same direction. The AGENTS.md specification showed that letting models manage their own memory through self-summarization or autonomous context selection degrades performance: model-generated summaries lose critical details, and autonomous context selection is non-reproducible. The LCM (Lossless Context Management) paper and SWE-Pruner independently showed that deterministic, structured context management can outperform model autonomy on multi-turn agent tasks.

This engine is inspired by LCM’s deterministic approach but does not preserve lossless guarantees. Eviction is explicit and logged, but irreversible from active context. The tradeoff is deliberate: manage context outside the model, deterministically, with full auditability.

## What the Engine Does

`engine.py` is a single-file context manager of roughly 580 lines with four core capabilities:

### Append-only ChunkLog

Every message is content-addressed with SHA-256 and stored in SQLite with WAL mode enabled. Chunks are immutable once written. No information is silently discarded. Evictions are explicit, logged, and reversible from storage, even though the active context is lossy.

### Threshold-based compaction

Two thresholds control when context is trimmed:

* **Soft (default 70%)**: evict lowest-priority chunks, oldest first
* **Hard (default 90%)**: aggressively evict while preserving a priority floor

### Goal-Guided scoring

A TF-IDF vectorizer (`scikit-learn`, bigrams, sublinear TF) scores each chunk using two blended signals, weighted 40/60:

* **Goal alignment**: cosine similarity to the most recent user message
* **Uniqueness**: inverse average similarity to peer chunks, so rare and specific content scores higher

The 40/60 blend was chosen empirically and has not been formally ablated. Sensitivity to this ratio remains an open question. No GPU is required. No model calls are made. There is no API latency.

### DecisionRecords

Every append and eviction is logged with timestamps, chunk hashes, reasons, and context size before and after the decision. Every compaction step is fully auditable.

## Benchmark Results

All benchmarks used 10 sessions, an 8k-token window, and Cerebras Llama 3.1-8B. Needles were critical facts such as bug reports with line numbers, security alerts, and action items, injected at random turns among filler conversation. On the final turn, the model had to recall all five needle details verbatim.

### Dense NIAH (30 turns, ~15k tokens through an 8k window)

Five needles were injected across 30 turns of filler, at roughly 500 tokens per turn. In total, ~15k tokens were pushed through an 8k context window, forcing repeated compaction.

| Metric            | Engine (priority compaction) | Baseline (sliding window) |
| ----------------- | ---------------------------: | ------------------------: |
| Avg recall        |                        5.0/5 |                     2.1/5 |
| Context at recall |                    4,967 tok |                 7,717 tok |
| TTFT              |                        0.70s |                     6.92s |
| Compaction events |                          6.4 |                         0 |

The engine selectively evicted filler from ~15k tokens of total throughput, retaining ~5k tokens of context while preserving all five needles. The baseline kept the most recent messages (7.7k tokens) but lost early-planted needles; `needle_1` was dropped in 10/10 sessions. Priority-aware compaction outperformed brute-force recency.

## Adversarial Progression (30 turns, 5 needles)

The next question was whether scoring still works without hardcoded priority labels. We tested three levels of difficulty.

### Clean filler (no keyword overlap with needles)

| Approach                     | Avg Recall | Context Size |
| ---------------------------- | ---------: | -----------: |
| Hardcoded Priority (ceiling) |      5.0/5 |    4,967 tok |
| AutoPriority (keywords)      |      5.0/5 |    5,124 tok |
| Baseline Sliding Window      |      2.1/5 |    7,717 tok |

Keyword extraction matches hardcoded priority when filler uses clearly distinct vocabulary.

### Adversarial filler (shared filenames, functions, and error patterns)

| Approach                     | Avg Recall | Context Size |
| ---------------------------- | ---------: | -----------: |
| Goal-Guided (TF-IDF)         |      5.0/5 |    4,877 tok |
| Hardcoded Priority (ceiling) |      5.0/5 |    4,870 tok |
| AutoPriority (keywords)      |      1.2/5 |    3,568 tok |
| Baseline Sliding Window      |      2.5/5 |    7,789 tok |

Keyword-based selection collapses to 1.2/5 when filler shares vocabulary with the needles. Goal-Guided scoring matches the hardcoded ceiling.

### Fair adversarial (length-matched chunks, unique filler, no template recycling)

| Approach                     | Avg Recall | Context Size |
| ---------------------------- | ---------: | -----------: |
| Goal-Guided (TF-IDF)         |      5.0/5 |    5,432 tok |
| Hardcoded Priority (ceiling) |      5.0/5 |    5,333 tok |
| AutoPriority (keywords)      |      1.9/5 |    4,809 tok |
| Baseline Sliding Window      |      2.4/5 |    7,936 tok |

The original adversarial benchmark had a length confound: needles averaged 244 characters, while filler averaged 1,801. This fair version uses ~500-character chunks for both, 100 unique fillers, and no template recycling. Goal-Guided scoring still achieves perfect recall. The uniqueness signal appears to capture content rarity rather than document length.

## The Progression

* Keywords work when important content is clearly distinct (5.0/5 on clean filler)
* Keywords collapse when adversarial filler shares vocabulary (1.2/5 on adversarial filler)
* Goal-Guided scoring is robust: TF-IDF uniqueness identifies rare, actionable content even under keyword overlap (5.0/5 on both adversarial settings)

These benchmarks are intentionally narrow. They test whether a deterministic selector can preserve sparse, actionable details under repeated context pressure. They do not yet measure end-to-end coding performance, tool-use success, or robustness to semantically subtle conflicts.

### Scoring Method Comparison (all benchmarks)

| Benchmark | TF-IDF | Semantic (MiniLM) | Hybrid | Hardcoded | Keywords | Naive |
|---|---|---|---|---|---|---|
| Clean | 5.0/5 | - | - | 5.0/5 | 5.0/5 | 2.1/5 |
| Adversarial | 5.0/5 | 4.0/5 | 4.3/5 | 5.0/5 | 1.2/5 | 2.5/5 |
| Semantic Gap | 5.0/5* | 4.7/5 | 4.8/5 | 5.0/5* | 4.2/5 | 2.5/5 |
| Boilerplate | 3.5/5 | 2.3/5 | 2.5/5 | 5.0/5 | 2.0/5 | 0/5 |

*Context retention was 5.0/5; recall drop is model-level, not engine-level.

We tested whether dense embeddings (all-MiniLM-L6-v2) could fix TF-IDF weaknesses on boilerplate content. They cannot. Embedding-space pairwise similarity is 9x higher than TF-IDF (0.27 vs 0.03), compressing the distinctiveness signal into a narrow band. TF-IDF lexical precision — treating `ci-deployer` and `argocd` as completely different tokens — is superior for structured content discrimination. The fix is not better embeddings but named entity extraction, which is next.

### Extended Scoring Analysis

| Benchmark | TF-IDF | Semantic (MiniLM) | Entity-Aware | Hardcoded | Keywords | Naive |
|---|---|---|---|---|---|---|
| Clean | 5.0/5 | - | 5.0/5 | 5.0/5 | 5.0/5 | 2.1/5 |
| Adversarial | 5.0/5 | 4.0/5 | ~5.0/5 | 5.0/5 | 1.2/5 | 2.5/5 |
| Semantic Gap | 3.4/5 | 4.7/5 | ~3.4/5 | 3.0/5 | 2.5/5 | 1.3/5 |
| Boilerplate | 3.5/5 | 2.3/5 | ~3.5/5 | 5.0/5 | 2.0/5 | 0/5 |

We tested two alternatives to fix TF-IDF weaknesses on boilerplate content. Dense embeddings (all-MiniLM-L6-v2) performed worse — embedding-space pairwise similarity is 9x higher than TF-IDF (0.27 vs 0.03), compressing the distinctiveness signal. TF-IDF lexical precision is superior for structured content. Entity-aware regex extraction matched TF-IDF but did not improve it because compaction decisions happen before the future recall query is known — the fundamental challenge of context management.

On semantic gap tasks, all methods including hardcoded priority showed reduced recall (3.0-3.4/5). Investigation revealed the engine retained all needles in context (5.0/5 retention) but the model failed to connect technical jargon to natural language queries. This is a model limitation, not an engine limitation.

## Honest Limitations

These are lab experiments. The benchmarks use synthetic needles and filler, controlled turn counts, and a single recall question. Real coding sessions are messier: interleaved file reads, error traces, user corrections, and tool outputs with varying relevance.

The needles are also deliberately distinguishable. Each one contains unique, actionable details such as line numbers, bug descriptions, or employee IDs. In real systems, important information may not separate from noise so cleanly.

TF-IDF has known failure modes. It relies on term-frequency patterns. If two chunks use nearly identical vocabulary and differ only in a single critical value, uniqueness scoring may fail to distinguish them.

TF-IDF uniqueness penalizes repetitive structured content (JSON schemas, SQL statements, config blocks). When critical information looks similar to filler, recall drops to 3.5/5.

Boilerplate discrimination remains an open problem (3.5/5). TF-IDF, dense embeddings, and entity extraction all fail when critical content is structurally similar to filler. The core challenge: compaction decisions are made before future information needs are known.

The system has not yet been validated on real coding workflows. All current results come from synthetic NIAH-style benchmarks. We do not yet know how it performs over 100+ turn real agent sessions where context pressure is constant and structure is unpredictable.

Compaction is inherently lossy. Any eviction strategy can drop something important. This engine makes that tradeoff explicit and auditable through `DecisionRecords`, but it does not eliminate the tradeoff itself.

## Next Steps

* **Real coding session evaluation**: instrument an actual coding agent and measure information retention over 100+ turns
* **Structured compression**: replace raw eviction with Active Context Compression-style summarization of evicted chunks
* **Agent integration**: embed the engine in a production coding agent and measure end-to-end task completion rates
* **OOLONG benchmark**: evaluate on a multi-turn long-context memory benchmark designed for sustained LLM interactions
* **Accumulated entity frequency tracking**: boost priority for entities mentioned repeatedly across turns rather than matching against the current goal

## References

- **LCM: Lossless Context Management** — Clint Ehrlich, Voltropy (2026). Deterministic context management outperforms model-autonomous memory in multi-turn agent tasks. [papers.voltropy.com/LCM](https://papers.voltropy.com/LCM)
- **Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents?** — Gloaguen, Mündler, Müller, Raychev, Vechev (2026). Shows that LLM-generated context degrades multi-turn agent performance. [arxiv.org/abs/2602.11988](https://arxiv.org/abs/2602.11988)
- **SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents** — Wang, Shi, Yang et al. (2026). Context pruning heuristics that improve coding agent performance through selective context reduction. [arxiv.org/abs/2601.16746](https://arxiv.org/abs/2601.16746)
- **Active Context Compression: Autonomous Memory Management in LLM Agents** — Verma (2026). Structured summarization as an alternative to raw context truncation in long-horizon tasks. [arxiv.org/abs/2601.07190](https://arxiv.org/abs/2601.07190)
- **Training-Free Group Relative Policy Optimization** — Cai, Cai, Shi, Xu, Tencent Youtu Lab (2025). Group relative policy optimization for aligning LLM context selection without additional training. [arxiv.org/abs/2510.08191](https://arxiv.org/abs/2510.08191)
- **Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities** — Bertsch, Pratapa, Mitamura, Neubig, Gormley (2025). Multi-turn long-context memory benchmark for evaluating LLM information retention. [arxiv.org/abs/2511.02817](https://arxiv.org/abs/2511.02817)



## Quickstart

```python
pip install scikit-learn

from engine import ChunkLog

log = ChunkLog(
    max_tokens=128_000,
    soft_threshold=0.7,
    hard_threshold=0.9,
    goal_guided=True,  # TF-IDF scoring (set False for keyword-only)
)

log.append("user", "auth.py line 42 has an off-by-one error in validate_token()")
log.next_turn()
log.append("assistant", "I'll fix that. The comparison should use < instead of <=.")
log.next_turn()
log.append("user", "Also check the rate limiter config in api_gateway.py")
log.next_turn()

messages = log.get_context()  # Returns a priority-managed message list
log.close()
```

The engine handles compaction automatically. High-priority content survives, low-value noise is evicted first, and every decision is recorded.