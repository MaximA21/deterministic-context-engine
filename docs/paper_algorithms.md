# Paper Algorithms — Scoring/Eviction Logic

Extracted from three papers for potential implementation as scorers in our context engine.

---

## 1. SWE-Pruner (arxiv 2601.16746)

**Core idea**: A lightweight neural skimmer (0.6B params) performs goal-guided line-level pruning of code context. Formulated as a reranking problem with CRF-based sequential labeling.

### Algorithm: Token Scoring + Line-Level Aggregation

```python
def swe_pruner_score(chunks: list[str], goal: str, threshold: float = 0.5) -> list[str]:
    """
    SWE-Pruner line-level pruning.

    1. Score each token given the goal query
    2. Aggregate to line-level scores (mean of token scores)
    3. Keep lines above threshold
    """
    # Step 1: Token-level scoring
    # s_i = F(goal, x_i | context; theta)
    # where F is a neural encoder (Qwen3-Reranker-0.6B backbone)
    # with multi-layer feature fusion from layers [7, 14, 28]
    token_scores = neural_encoder(goal, context_tokens)  # -> list[float]

    # Step 2: Line-level aggregation
    kept_lines = []
    for line in lines:
        tokens_in_line = get_tokens(line)
        # Line score = mean of constituent token scores
        line_score = sum(token_scores[t] for t in tokens_in_line) / len(tokens_in_line)

        # Step 3: Threshold decision (CRF Viterbi decoding in practice)
        if line_score > threshold:
            kept_lines.append(line)

    return kept_lines
```

### Training Objective

```python
def swe_pruner_loss(predictions, labels, lambda_weight=0.05):
    """
    Combined CRF + reranking loss.

    CRF models transition probabilities between retain/prune states:
      score(x, y) = start_y1 + sum(emission_t_yt) + sum(transition_yt_yt-1) + end_yt

    Loss:
      L_compress = (1/B) * sum_i [ CRF_NLL(x_i, y_i) / L_i ]
        where L_i = sequence length (normalization prevents bias toward aggressive pruning)

      L_rerank = MSE(s_pred, s_ref)  # document-level relevance

      L_total = (1 - lambda) * L_compress + lambda * L_rerank
    """
    crf_loss = mean(crf_nll(x, y) / seq_len for x, y, seq_len in zip(...))
    rerank_loss = mse(pred_scores, ref_scores)
    return (1 - lambda_weight) * crf_loss + lambda_weight * rerank_loss
```

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `threshold` (tau) | 0.5 | Line retention cutoff |
| `lambda` | 0.05 | CRF vs reranking loss balance |
| `dropout` | 0.4 | Regularization |
| `lr` | 3e-5 | Learning rate |
| Feature layers | 7, 14, 28 | Multi-layer fusion from encoder |

### Adaptation for Our Engine

```python
def swe_pruner_as_scorer(chunk: str, goal: str, all_chunks: list[str]) -> float:
    """
    Simplified SWE-Pruner scorer for ChunkLog.

    Without a trained neural model, we approximate the scoring:
    - Use TF-IDF similarity between goal and chunk as proxy for F(goal, x_i)
    - Line-level aggregation becomes chunk-level (our chunks are already atomic units)
    - Threshold becomes the compaction cutoff in ChunkLog
    """
    # Direct analog: goal-chunk similarity (replaces neural token scoring)
    goal_similarity = tfidf_cosine(goal, chunk)

    # The CRF sequential dependency could be approximated by
    # boosting scores of chunks adjacent to high-scoring chunks
    # (context continuity bonus)
    return goal_similarity
```

---

## 2. Active Context Compression (arxiv 2601.07190)

**Core idea**: A behavioral framework (not a computational algorithm) where an LLM agent autonomously decides when to consolidate key learnings into a persistent "Knowledge" block and prunes raw history. The agent uses tool calls (`start_focus`, `consolidate`, `withdraw`) rather than mathematical scoring.

### Algorithm: Focus Loop

```python
def active_context_compression(
    messages: list[Message],
    compression_interval: int = 12,  # every 10-15 tool calls
    max_uncompressed_calls: int = 15,  # system reminder trigger
) -> tuple[str, list[Message]]:
    """
    Active Context Compression — Focus Agent loop.

    No scoring function. Compression is a behavioral pattern:
    1. start_focus(topic) — declare investigation scope
    2. explore() — standard tool calls within focus
    3. consolidate() — LLM summarizes what was learned
    4. withdraw() — append summary to Knowledge block, delete raw messages
    """
    knowledge_block = ""
    checkpoint = 0
    tool_call_count = 0

    for i, msg in enumerate(messages):
        tool_call_count += 1

        # Trigger: compress every N tool calls (prompted behavior)
        if tool_call_count >= compression_interval:
            # CONSOLIDATE: LLM generates structured summary
            span = messages[checkpoint:i]
            summary = llm_summarize(
                span,
                prompt="What was attempted? What was learned (facts, file paths, bugs)? What is the outcome?"
            )

            # WITHDRAW: append summary, delete raw history
            knowledge_block += f"\n{summary}"
            messages = messages[:checkpoint] + messages[i:]  # delete span

            checkpoint = i
            tool_call_count = 0

    return knowledge_block, messages
```

### Key Properties

- **No scoring/ranking**: Binary keep-all-summary / delete-all-raw
- **No selective retention**: Everything in the span is summarized equally
- **Agent-driven timing**: The LLM decides when to compress (prompted to do so every 10-15 tool calls)
- **Lossy**: Raw messages are deleted; only the LLM summary survives

### Adaptation for Our Engine

```python
def active_compression_as_scorer(
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    knowledge_block: str,
) -> float:
    """
    Active Context Compression adapted as a scorer.

    Since the original has no scoring, we derive one from its principles:
    - Recent chunks score high (not yet compressed)
    - Chunks whose content is already captured in knowledge_block score low
    - Chunks near compression boundary get medium scores
    """
    # Recency signal (recent = high score, will be compressed later)
    recency = chunk_index / total_chunks  # 0.0 (oldest) to 1.0 (newest)

    # Redundancy with knowledge block (already captured = low score)
    redundancy = tfidf_cosine(chunk, knowledge_block)

    # Score: keep if recent OR not yet captured in knowledge
    score = 0.6 * recency + 0.4 * (1.0 - redundancy)
    return score
```

---

## 3. MemFly (arxiv 2602.07885)

**Core idea**: Information-bottleneck-based memory that minimizes compression entropy while maximizing relevance entropy. Uses a stratified structure (Notes -> Keywords -> Topics) with LLM-approximated divergence scores for merge/link/append decisions.

### Core Objective (Information Bottleneck)

```python
def memfly_objective(memory_state, raw_inputs, beta: float = 1.0) -> float:
    """
    Information Bottleneck Lagrangian (Eq. 1):

      L_IB(M_t) = I(X_{1:t}; M_t) - beta * I(M_t; Y)

    Where:
      I(X_{1:t}; M_t) = compression term (minimize: reduce redundancy)
      I(M_t; Y)        = relevance term (maximize: preserve task-useful info)
      beta > 0          = trade-off parameter
    """
    compression = mutual_information(raw_inputs, memory_state)
    relevance = mutual_information(memory_state, future_tasks)
    return compression - beta * relevance  # minimize this
```

### Algorithm: Gated Structural Update

```python
def memfly_ingest(
    new_chunk: str,
    memory: list[Note],
    merge_threshold: float = 0.7,  # tau_m
    link_threshold: float = 0.5,   # tau_l
) -> list[Note]:
    """
    MemFly gated structural update (Eq. 6).

    For each incoming chunk, compute redundancy and complementarity scores
    against existing memory notes, then decide: merge, link, or append.
    """
    new_note = Note(
        raw=new_chunk,
        context=llm_summarize(new_chunk),           # semantically denoised
        embedding=embed(new_chunk),                  # dense vector h_i
        keywords=llm_extract_keywords(new_chunk),    # symbolic anchors K_i
    )

    # Find best matching existing note
    best_note = None
    best_redundancy = 0.0
    best_complementarity = 0.0

    for note in memory:
        # LLM-approximated Jensen-Shannon Divergence:
        #   s_red(n_t, n_i) ≈ 1 - D_JS[p(Y|n_t), p(Y|n_i)]
        # In practice: LLM generates two scores via structured prompting
        s_red = llm_score_redundancy(new_note, note)    # 0-1, semantic overlap
        s_comp = llm_score_complementarity(new_note, note)  # 0-1, logical connection

        if s_red > best_redundancy:
            best_redundancy = s_red
            best_complementarity = s_comp
            best_note = note

    # Gated decision (Eq. 6)
    if best_redundancy > merge_threshold:
        # MERGE: high redundancy -> consolidate
        # Directly minimizes compression entropy I(X; M)
        best_note.raw = best_note.raw + "\n" + new_note.raw
        best_note.context = llm_merge_contexts(best_note.context, new_note.context)
        best_note.keywords = best_note.keywords | new_note.keywords
        best_note.embedding = embed(best_note.context)  # re-embed merged context

    elif best_complementarity > link_threshold:
        # LINK: complementary -> create directed edge
        # Preserves conditional dependencies for I(M; Y)
        memory.append(new_note)
        add_edge(new_note, best_note, edge_type="related")

    else:
        # APPEND: distinct content -> new autonomous unit
        # Preserves distributional diversity
        memory.append(new_note)

    return memory
```

### Stratified Memory Structure

```python
@dataclass
class Note:
    """Layer 1 — Fidelity layer. Preserves I(N; X) ≈ H(X)."""
    raw: str           # r_i: verbatim content
    context: str       # c_i: semantically denoised summary
    embedding: list[float]  # h_i ∈ R^d: dense embedding
    keywords: set[str]      # K_i ⊂ K: symbolic anchors

@dataclass
class Keyword:
    """Layer 2 — Anchoring layer. Stabilizes semantic proximity."""
    term: str
    embedding: list[float]  # e_j ∈ R^d
    co_occurrence_edges: set[str]  # other keywords seen together

@dataclass
class Topic:
    """Layer 3 — Navigation layer. O(1) macro navigation."""
    keywords: set[str]      # cluster of related keywords
    centroid: list[float]   # semantic centroid for retrieval
    # Derived via Leiden algorithm on keyword co-occurrence graph
```

### Hybrid Retrieval (Tri-Pathway)

```python
def memfly_retrieve(
    query: str,
    memory: list[Note],
    keywords: list[Keyword],
    topics: list[Topic],
    k_topic: int = 3,
    k_key: int = 10,
    k_final: int = 20,
    max_iterations: int = 3,
) -> list[Note]:
    """
    MemFly tri-pathway retrieval with iterative refinement.
    """
    query_embedding = embed(query)
    query_keywords = extract_keywords(query)

    # Pathway 1: Macro-Semantic Localization (topic -> notes)
    top_topics = topk(topics, k_topic, key=lambda t: cosine(query_embedding, t.centroid))
    r_topic = [n for n in memory if any(k in t.keywords for k in n.keywords for t in top_topics)]

    # Pathway 2: Micro-Symbolic Anchoring (keyword -> notes)
    top_keywords = topk(keywords, k_key, key=lambda kw: max(cosine(embed(qk), kw.embedding) for qk in query_keywords))
    r_key = [n for n in memory if n.keywords & {kw.term for kw in top_keywords}]

    # Pathway 3: Topological Expansion (follow edges from anchors)
    anchors = set(r_topic + r_key)
    expanded = anchors | {m for n in anchors for m in get_related(n)}

    # Reciprocal Rank Fusion across pathways
    pool = reciprocal_rank_fusion(r_topic, r_key, list(expanded))
    evidence = pool[:k_final]

    # Iterative Evidence Refinement (IER)
    for i in range(max_iterations):
        if llm_sufficiency_check(evidence, query):
            break
        refined_query = llm_refine_query(query, evidence)
        new_evidence = memfly_retrieve(refined_query, memory, keywords, topics)
        evidence = deduplicate(evidence + new_evidence)[:k_final]

    return evidence
```

### Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `tau_m` (merge threshold) | 0.7 | Redundancy cutoff for merging |
| `tau_l` (link threshold) | 0.5 | Complementarity cutoff for linking |
| `k_topic` | 3 | Topic candidates in retrieval |
| `k_key` | 10 | Keyword candidates in retrieval |
| `k_final` | 20 | Final evidence pool size |
| `max_iterations` (IER) | 3 | Iterative refinement rounds |
| `beta` | 1.0 | IB trade-off (compression vs relevance) |

### Adaptation for Our Engine

```python
def memfly_as_scorer(
    chunk: str,
    goal: str,
    all_chunks: list[str],
    merge_threshold: float = 0.7,
) -> float:
    """
    MemFly adapted as a ChunkLog scorer.

    Key insight: MemFly's scoring is bidimensional (redundancy + complementarity).
    We can use this to produce a single eviction score.

    High score = keep. Low score = evict.
    """
    chunk_embedding = tfidf_vector(chunk)
    goal_embedding = tfidf_vector(goal)

    # Relevance: how much does this chunk help with the goal?
    # Approximates I(M; Y) — the relevance entropy term
    relevance = cosine_similarity(chunk_embedding, goal_embedding)

    # Redundancy: how much overlap with other chunks?
    # Approximates I(X; M) — the compression entropy term
    peer_similarities = [
        cosine_similarity(chunk_embedding, tfidf_vector(other))
        for other in all_chunks if other != chunk
    ]
    avg_redundancy = sum(peer_similarities) / max(len(peer_similarities), 1)

    # MemFly merges when redundancy > 0.7, meaning highly redundant chunks
    # should be consolidated. For scoring, redundant chunks get lower scores.
    # IB objective: minimize compression (penalize redundancy),
    #               maximize relevance (reward goal alignment)
    score = 0.5 * relevance + 0.5 * (1.0 - avg_redundancy)
    return score
```

---

## Comparison Summary

| Dimension | SWE-Pruner | Active Context Compression | MemFly |
|-----------|-----------|---------------------------|--------|
| **Scoring** | Neural token-level, CRF line-level | None (behavioral) | LLM-scored redundancy + complementarity |
| **Granularity** | Line-level | Entire conversation span | Note-level (chunk) |
| **Goal-awareness** | Yes (explicit hint/query) | No (agent decides) | Implicit (IB relevance term) |
| **Decision** | Keep/prune per line (threshold) | Summarize-all / delete-all | Merge / link / append (dual threshold) |
| **Requires NN** | Yes (0.6B model) | No (LLM prompting) | No (LLM scoring, TF-IDF possible) |
| **Best for our engine** | Goal-guided pruning concept | Periodic compaction pattern | Redundancy+relevance scoring formula |

### What to Implement

1. **From SWE-Pruner**: Goal-guided scoring is already our `GoalGuidedScorer`. The CRF sequential dependency idea (boosting neighbors of important chunks) could improve our scorer.

2. **From Active Context Compression**: Our `ChunkLog.compact()` already does this. Could add periodic forced compaction (every N appends) as a complement to threshold-based compaction.

3. **From MemFly**: The dual-threshold merge/link/append gating and the IB-inspired `relevance - redundancy` formula are directly implementable. The `0.5 * goal_similarity + 0.5 * (1 - avg_peer_similarity)` is very close to our existing `0.4 * goal + 0.6 * uniqueness` weighting.
