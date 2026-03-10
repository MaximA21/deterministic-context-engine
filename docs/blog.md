# Why Deterministic Context Management Beats LLM Summarization for Coding Agents

Every coding agent has the same dirty secret: it spends a shocking amount of its token budget talking to itself about what it already did.

Claude Code, Aider, Cursor, Copilot Workspace -- they all hit the same wall. After 20-30 turns of a coding session, the conversation history overflows the context window. The standard fix is to call the LLM to summarize prior turns, compressing history into a shorter form. This works, until it doesn't.

We replaced LLM summarization with a deterministic engine that decides what stays in context using BM25-style scoring. No model calls in the critical path. Decisions under 50ms. And it recalled 5.0/5 needles where a naive sliding window scored 2.1/5.

Here's what we tried, what failed, and what actually worked.

## The problem with LLM summarization

When a coding agent summarizes its own history, three things go wrong:

**It's expensive.** Each summarization call costs tokens -- sometimes more than the content it compresses. In a 50-turn session, summarization overhead can exceed the actual work.

**It's lossy in unpredictable ways.** The model decides what's "important" based on its current understanding. But context that seems irrelevant on turn 12 might be critical on turn 30. A file path mentioned early in the session, an error message from a failed test, a user preference stated once -- these get compressed away.

**It's non-deterministic.** Run the same session twice and you'll get different summaries, different context, different behavior. You can't debug what you can't reproduce.

We wanted something different: a context manager that keeps raw content, evicts what's least relevant using math, and never calls a model to make the decision.

## The experiment

We built an append-only context log backed by SQLite WAL. Every message gets chunked, hashed with SHA-256 for deduplication, and scored. When the context window fills up (soft threshold at 70%, hard at 90%), the engine evicts the lowest-scored chunks. Every append and eviction is logged as a DecisionRecord -- timestamp, chunk hash, reason, context size before and after.

To test, we used a Needle-in-a-Haystack (NIAH) benchmark: plant specific facts across 30 turns of conversation, fill the rest with realistic-looking filler, then ask the model to recall the planted facts. We ran this against Cerebras's llama3.1-8b with an 8k context window -- deliberately small to force compaction.

The question was: what scoring function should decide which chunks survive?

## What we tried

### 1. Keyword extraction

The simplest approach: regex-extract filenames (`auth.py`), function names (`def validate_token`), error indicators (`Exception`, `CRITICAL`), IP addresses, dates. Score each chunk by how many keywords it contains. Map the count to a priority from 0.5 to 2.0.

**Result: 5.0/5 on clean benchmarks.** When filler is generic ("let's discuss the architecture..."), keywords easily distinguish needles from noise.

**Result: 1.2/5 on adversarial benchmarks.** When filler shares keywords with the needles -- mentioning the same files, the same function names -- keyword scoring collapses completely. It can't tell "critical bug on line 42 of auth.py" from "refactored auth.py last week."

### 2. TF-IDF with uniqueness scoring

Vectorize all chunks using TF-IDF (bigrams, sublinear term frequency, 5000 max features). Then blend two signals:

- **Goal alignment (40%):** Cosine similarity between each chunk and the accumulated user messages (the "goal" of the session).
- **Uniqueness (60%):** For each chunk, compute the average cosine similarity to all other chunks. Uniqueness = 1.0 minus that average.

**Result: 5.0/5 on both clean and adversarial benchmarks.** More on why below.

### 3. Semantic embeddings (MiniLM)

Same architecture as TF-IDF but with `all-MiniLM-L6-v2` embeddings instead of sparse vectors. Same 40/60 goal/uniqueness blend.

**Result: 4.0/5 on adversarial, 2.3/5 on boilerplate.** The embeddings compress the distinctiveness signal. Pairwise similarity between chunks averaged 0.27 with embeddings vs 0.03 with TF-IDF -- a 9x compression. Two completely different JSON schemas look nearly identical to MiniLM. The narrow similarity band means the uniqueness score can't separate needles from filler reliably.

### 4. Entity-aware scoring

TF-IDF base plus an entity extraction bonus. We wrote 20+ regex patterns for SQL table names, CLI flags, environment variables, API endpoints, file paths, version numbers, error codes, JIRA tickets, JSON keys, snake_case identifiers. The entity bonus fires when >40% of a chunk's entities overlap with the current goal AND <10% of all chunks share those entities (specificity check).

**Result: matched TF-IDF exactly.** The fundamental problem: compaction happens *before* the future recall query is known. Entity bonus can only help if the goal already contains the needle's entities, but during filler turns, the goal *is* the filler. It can't predict what you'll need next.

### 5. Hybrid (TF-IDF + embeddings)

30% TF-IDF, 70% semantic blend. Trying to get the best of both.

**Result: 4.3-4.8/5.** Better than pure embeddings, worse than pure TF-IDF. The semantic signal adds noise rather than signal for structured content like code.

## What worked: TF-IDF uniqueness and why

The winner was pure TF-IDF with the 40/60 goal/uniqueness blend. But why does uniqueness work so well?

The mechanism is statistical. In a typical coding session, filler messages share vocabulary -- they talk about the same repo, the same framework, the same patterns. Needles, by definition, contain specific facts that appear rarely.

With TF-IDF, the pairwise similarity between filler chunks is measurably higher than between a needle and the filler. The uniqueness score amplifies this:

- 20 filler chunks: uniqueness gap ~0.003 (needle vs filler avg)
- 40 filler chunks: uniqueness gap ~0.020

The gap *grows dynamically* as more filler accumulates. Each new filler chunk raises the average peer similarity for all filler, while needles remain distinct. By the time compaction runs, the filler-to-needle ratio (typically 40:5) means even a small per-chunk gap creates enough separation to evict filler before needles.

We initially suspected a length artifact -- our needles averaged 244 characters while filler averaged 1,801. So we ran a "fair" benchmark: all chunks at ~500 characters, 100 unique filler messages. Result: still 5.0/5. The uniqueness signal is real.

## The honest results

Here's the full benchmark table:

| Benchmark | TF-IDF | Semantic | Hybrid | Keywords | Naive |
|-----------|--------|----------|--------|----------|-------|
| Clean (30 turns) | 5.0/5 | -- | -- | 5.0/5 | 2.1/5 |
| Adversarial | 5.0/5 | 4.0/5 | 4.3/5 | 1.2/5 | 2.5/5 |
| Fair (length-matched) | 5.0/5 | 4.7/5 | 4.8/5 | 1.9/5 | 2.5/5 |
| Boilerplate | 3.5/5 | 2.3/5 | 2.5/5 | 2.0/5 | 0/5 |

**Dense NIAH (30 turns, ~15k tokens through 8k window):**
- Engine: 5.0/5 recall, 4,967 tokens used, 0.70s time-to-first-token
- Baseline (sliding window): 2.1/5 recall, 7,717 tokens used, 6.92s TTFT

**Live test (15-turn session on a 70k-line FastAPI repo):**
- 15/15 turns completed, zero context errors
- 5 compaction events, 61 chunks evicted
- 4/4 recall checks passed
- Average TTFT: 2.01s

The numbers are real but the caveats matter.

## Open problems

**Boilerplate retention (3.5/5).** When the critical content is structurally similar to filler -- both are JSON schemas, both are SQL migrations, both are config blocks -- TF-IDF uniqueness breaks down. The needle *is* boilerplate. No sparse-vector method can distinguish "this particular migration matters" from "this is just another migration." This likely requires domain-specific heuristics or a fundamentally different signal.

**Adversarial keyword overlap.** TF-IDF handles this well (5.0/5), but it's possible to construct adversarial filler that matches TF-IDF patterns too. We haven't tested worst-case adversarial TF-IDF overlap yet.

**Scaling.** TF-IDF refit is O(n * vocab) on every compaction. At 200+ chunks, this could add meaningful latency. We haven't hit this in practice with our 8k window, but larger context windows (128k+) will need incremental TF-IDF or approximate scoring.

**The timing problem.** Entity extraction doesn't help because compaction decisions happen before future queries are known. Any scoring method that relies on predicting *future* relevance faces this fundamental constraint. The uniqueness signal works around it -- rare content is more *likely* to be relevant -- but it's a heuristic, not a guarantee.

## How to use it

```python
from engine import ChunkLog

log = ChunkLog(
    max_tokens=128_000,
    soft_threshold=0.7,
    hard_threshold=0.9,
    goal_guided=True,
)

log.append("user", "auth.py line 42 has an off-by-one error")
log.next_turn()
log.append("assistant", "I'll fix the bounds check in validate_token...")
log.next_turn()

messages = log.get_context()  # compaction runs automatically
```

Five lines to wire up. No API keys for scoring, no model calls, no async. The engine handles chunking, deduplication, scoring, eviction, and decision logging. You get back a list of messages that fit your context window, with the most relevant content preserved.

## What this means

Deterministic context management isn't a silver bullet. It fails on boilerplate-heavy sessions, and it can't truly predict what the user will ask next. But for coding agents, where conversations follow predictable patterns of "discuss, code, test, debug," TF-IDF uniqueness captures the right signal surprisingly well.

The deeper insight: you don't need a model to manage a model's memory. The statistical properties of the conversation itself -- what's unique, what's repetitive -- carry enough information to make good eviction decisions. And those decisions are fast, reproducible, and auditable.

Every eviction is logged. Every decision can be replayed. When your agent drops critical context on turn 47, you can look at the DecisionRecords and understand exactly why. Try doing that with an LLM summarizer.

The code is open source and the benchmarks are reproducible. We'd rather show you a 3.5/5 on boilerplate than pretend it doesn't exist.
