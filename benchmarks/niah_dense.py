#!/usr/bin/env python3
"""Dense NIAH v2 — Engine (priority-aware compaction) vs Naive (sliding window).

Instead of Engine OFF (which just crashes on context overflow), the baseline
is now a NAIVE sliding window that drops oldest messages first (FIFO) to fit
within the 8k token limit. No priority awareness — just brute truncation.

This tests the interesting question: does smart compaction preserve critical
info better than dumb truncation?

Measures:
- Needle recall rate (0-5) for Engine vs Naive
- Which specific needles were lost by each approach
- At which turn compaction fired (Engine mode)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine import ChunkLog, _estimate_tokens

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --- 5 Needles (critical facts to remember) ---
NEEDLES = [
    {"id": "needle_1", "fact": "The secret project codename is Operation Maple Leaf.", "keyword": "maple leaf"},
    {"id": "needle_2", "fact": "The database migration is scheduled for March 15th at 3am UTC.", "keyword": "march 15"},
    {"id": "needle_3", "fact": "The API rate limit was changed from 100 to 250 requests per minute.", "keyword": "250"},
    {"id": "needle_4", "fact": "The new hire's name is Elena Vasquez and she starts on Monday.", "keyword": "elena"},
    {"id": "needle_5", "fact": "The production server IP address is 10.42.88.7.", "keyword": "10.42.88.7"},
]

# Filler templates (~500 tokens each when expanded)
FILLER_TEMPLATES = [
    """Here's the latest code review for the authentication module:

```python
class AuthManager:
    def __init__(self, config):
        self.config = config
        self.token_cache = {}
        self.session_store = SessionStore()
        self.rate_limiter = RateLimiter(max_requests=100)

    def authenticate(self, username, password):
        if not username or not password:
            raise ValueError("Username and password required")
        if self.rate_limiter.is_limited(username):
            raise RateLimitError("Too many attempts")
        hashed = self._hash_password(password, self._get_salt(username))
        stored = self.session_store.get_hash(username)
        if not constant_time_compare(hashed, stored):
            self.rate_limiter.record_failure(username)
            raise AuthenticationError("Invalid credentials")
        token = self._generate_token(username)
        self.token_cache[token] = {
            'username': username,
            'created': time.time(),
            'expires': time.time() + 3600
        }
        return token

    def _hash_password(self, password, salt):
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)

    def _get_salt(self, username):
        return self.session_store.get_salt(username)

    def _generate_token(self, username):
        return secrets.token_urlsafe(32)
```

The code looks good but we should add logging for failed attempts and consider using argon2 instead of pbkdf2.""",

    """Updated the deployment pipeline configuration:

```yaml
stages:
  - name: build
    steps:
      - checkout
      - restore_cache:
          keys:
            - deps-v2-{{ checksum "requirements.txt" }}
      - run: pip install -r requirements.txt
      - run: python -m pytest tests/ -v --cov=src
      - save_cache:
          key: deps-v2-{{ checksum "requirements.txt" }}
          paths:
            - .venv
  - name: deploy-staging
    requires: [build]
    steps:
      - run: aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_REPO
      - run: docker build -t $ECR_REPO:$COMMIT_SHA .
      - run: docker push $ECR_REPO:$COMMIT_SHA
      - run: kubectl set image deployment/app app=$ECR_REPO:$COMMIT_SHA -n staging
      - run: kubectl rollout status deployment/app -n staging --timeout=300s
  - name: deploy-production
    requires: [deploy-staging]
    approval: manual
    steps:
      - run: kubectl set image deployment/app app=$ECR_REPO:$COMMIT_SHA -n production
      - run: kubectl rollout status deployment/app -n production --timeout=300s
      - run: python scripts/smoke_test.py --env production
```

Remember to update the ECR repository URL when we migrate to the new AWS account next quarter.""",

    """Performance analysis from the load testing results:

The API endpoint /api/v2/search showed concerning latency under load:
- P50 latency: 45ms (acceptable, target < 100ms)
- P95 latency: 230ms (borderline, target < 250ms)
- P99 latency: 890ms (needs improvement, target < 500ms)
- Max observed: 2.3s (unacceptable)

Database query analysis:
- The main bottleneck is the full-text search query on the documents table
- Index scan is being used correctly but the table has grown to 12M rows
- Adding a composite index on (tenant_id, created_at, status) should help
- Consider implementing cursor-based pagination instead of offset

Memory usage:
- Average: 2.1GB per pod (within 4GB limit)
- Peak during search: 3.2GB (close to limit)
- GC pauses averaging 15ms, max observed 120ms

Recommendations:
1. Add the composite index (estimated 40% improvement)
2. Implement response caching with 30s TTL for repeated queries
3. Consider Elasticsearch for the search functionality if growth continues
4. Increase pod memory limit to 6GB as safety margin""",

    """Meeting notes from the architecture review:

Attendees: Team Alpha, Platform team, Security team

Discussion points:
1. Microservice decomposition plan
   - Extract user service from monolith (Q2 target)
   - Extract notification service (Q3 target)
   - Keep payment processing in monolith for now (regulatory concerns)

2. Event-driven architecture proposal
   - Use Apache Kafka for inter-service communication
   - Schema registry with Avro for message contracts
   - Dead letter queues for failed message processing
   - Retention policy: 7 days for most topics, 30 days for audit

3. Security review findings
   - JWT tokens need shorter expiry (currently 24h, recommend 1h)
   - Add refresh token rotation
   - Implement CORS policy for API gateway
   - Review third-party dependency versions (3 CVEs found)

4. Infrastructure cost optimization
   - Move to spot instances for non-critical workloads (est. 40% savings)
   - Implement auto-scaling based on queue depth
   - Consider reserved instances for databases

Action items assigned, follow-up meeting in two weeks.""",

    """Database schema migration plan for the multi-tenant feature:

```sql
-- Migration: Add tenant isolation
ALTER TABLE users ADD COLUMN tenant_id UUID NOT NULL DEFAULT gen_random_uuid();
ALTER TABLE documents ADD COLUMN tenant_id UUID NOT NULL DEFAULT gen_random_uuid();
ALTER TABLE audit_log ADD COLUMN tenant_id UUID NOT NULL DEFAULT gen_random_uuid();

CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    plan VARCHAR(50) NOT NULL DEFAULT 'free',
    max_users INTEGER NOT NULL DEFAULT 5,
    max_storage_gb INTEGER NOT NULL DEFAULT 10,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_users ON users
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_docs ON documents
    USING (tenant_id = current_setting('app.current_tenant')::UUID);

CREATE INDEX idx_users_tenant ON users(tenant_id);
CREATE INDEX idx_documents_tenant ON documents(tenant_id);
CREATE INDEX idx_audit_tenant ON audit_log(tenant_id);
```

This migration needs careful coordination. Run during maintenance window only.""",
]

MAX_CONTEXT_TOKENS = 8000
SYSTEM_PROMPT = "You are a helpful assistant with perfect memory. Answer questions based on the conversation history provided."
SYSTEM_PROMPT_TOKENS = _estimate_tokens(SYSTEM_PROMPT)

RECALL_QUESTION = """Please recall ALL important updates and facts mentioned during our conversation.
Specifically, I need you to tell me:
1. What is the secret project codename?
2. When is the database migration scheduled?
3. What was the API rate limit changed to?
4. What is the new hire's name and when does she start?
5. What is the production server IP address?

Answer each question based ONLY on what was mentioned in our conversation."""


def generate_needle_placements(num_sessions: int, num_turns: int = 30, num_needles: int = 5) -> list[list[int]]:
    """Generate different needle placement patterns across sessions."""
    placements = []
    rng = random.Random(42)
    for _ in range(num_sessions):
        turns = sorted(rng.sample(range(num_turns), num_needles))
        placements.append(turns)
    return placements


def sliding_window_truncate(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    """Naive sliding window: drop oldest messages first until we fit.

    Always keeps the LAST message (the recall question).
    Drops from the front (oldest) until total tokens <= max_tokens.
    """
    if not messages:
        return messages

    # Reserve the last message (recall question)
    last_msg = messages[-1]
    last_tokens = _estimate_tokens(last_msg["content"])

    budget = max_tokens - SYSTEM_PROMPT_TOKENS - last_tokens
    if budget <= 0:
        return [last_msg]

    # Walk backwards from second-to-last, accumulating messages that fit
    kept: list[dict[str, str]] = []
    used = 0
    for msg in reversed(messages[:-1]):
        msg_tokens = _estimate_tokens(msg["content"])
        if used + msg_tokens <= budget:
            kept.append(msg)
            used += msg_tokens
        else:
            break  # Stop at first message that doesn't fit (FIFO: drop all older)

    kept.reverse()
    kept.append(last_msg)
    return kept


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,  # "engine" or "naive"
    api_key: str,
    num_turns: int = 30,
) -> dict[str, Any]:
    """Run a single dense NIAH session."""
    from cerebras.cloud.sdk import Cerebras

    rng = random.Random(session_id * 1000 + hash(mode))
    compaction_log: list[dict] = []
    total_tokens_added = 0

    if mode == "engine":
        # Priority-aware compaction
        chunk_log = ChunkLog(db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS, soft_threshold=0.7, hard_threshold=0.9)
    else:
        # Naive mode: we accumulate messages in a plain list, truncate before API call
        chunk_log = None
        raw_messages: list[dict[str, str]] = []

    # Simulate 30 turns of conversation
    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            needle = NEEDLES[needle_idx]
            content = f"IMPORTANT UPDATE: {needle['fact']}"
            priority = 2.0
        else:
            filler = rng.choice(FILLER_TEMPLATES)
            unique_salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] {filler}\n\nAdditional context for this turn:\n{rng.choice(FILLER_TEMPLATES)}{unique_salt}"
            priority = 0.5

        total_tokens_added += _estimate_tokens(content)

        if mode == "engine":
            tokens_before = chunk_log.current_tokens()
            compactions_before = chunk_log.compaction_count
            chunk_log.append("user", content, priority=priority)
            chunk_log.next_turn()
            tokens_after = chunk_log.current_tokens()
            compactions_after = chunk_log.compaction_count
            if compactions_after > compactions_before:
                compaction_log.append({
                    "turn": turn,
                    "tokens_before": tokens_before + _estimate_tokens(content),
                    "tokens_after": tokens_after,
                    "events": compactions_after - compactions_before,
                })
        else:
            raw_messages.append({"role": "user", "content": content})

    # Add recall question
    if mode == "engine":
        chunk_log.append("user", RECALL_QUESTION, priority=2.0)
        messages = chunk_log.get_context()
        context_tokens = chunk_log.get_context_tokens()
        compaction_events = chunk_log.compaction_count
    else:
        raw_messages.append({"role": "user", "content": RECALL_QUESTION})
        full_tokens = sum(_estimate_tokens(m["content"]) for m in raw_messages)
        # Apply sliding window truncation
        messages = sliding_window_truncate(raw_messages, MAX_CONTEXT_TOKENS)
        context_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
        compaction_events = 0

    # Track which needles survived truncation (naive mode)
    needles_in_context = []
    context_text = " ".join(m["content"] for m in messages)
    for needle in NEEDLES:
        if needle["keyword"].lower() in context_text.lower():
            needles_in_context.append(needle["id"])

    # Call Cerebras API
    client = Cerebras(api_key=api_key)

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *messages,
            ],
            max_tokens=512,
            temperature=0.0,
        )
        ttft = time.time() - t0
        answer = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        error = None
    except Exception as e:
        answer = ""
        ttft = 0
        input_tokens = 0
        output_tokens = 0
        error = str(e)
    finally:
        if chunk_log:
            chunk_log.close()

    # Score: check which needles were recalled
    answer_lower = answer.lower()
    needles_recalled = []
    needles_lost = []
    for needle in NEEDLES:
        if needle["keyword"].lower() in answer_lower:
            needles_recalled.append(needle["id"])
        else:
            needles_lost.append(needle["id"])

    return {
        "session_id": session_id,
        "mode": mode,
        "error": error,
        "needle_turns": needle_turns,
        "needles_in_context": needles_in_context,
        "needles_recalled": needles_recalled,
        "needles_lost": needles_lost,
        "recall_score": len(needles_recalled),
        "total_needles": len(NEEDLES),
        "answer": answer,
        "ttft": round(ttft, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_tokens": context_tokens,
        "total_tokens_added": total_tokens_added,
        "compaction_events": compaction_events,
        "compaction_log": compaction_log,
    }


def generate_chart(results: list[dict], output_path: Path) -> None:
    """Generate comparison chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    engine_results = [r for r in results if r["mode"] == "engine"]
    naive_results = [r for r in results if r["mode"] == "naive"]

    engine_scores = [r["recall_score"] for r in engine_results]
    naive_scores = [r["recall_score"] for r in naive_results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Chart 1: Recall scores per session
    ax = axes[0]
    x = range(len(engine_scores))
    width = 0.35
    ax.bar([i - width/2 for i in x], engine_scores, width, label="Engine (priority compaction)", color="#2ecc71")
    ax.bar([i + width/2 for i in x], naive_scores, width, label="Naive (sliding window)", color="#e74c3c")
    ax.set_xlabel("Session")
    ax.set_ylabel("Needles Recalled (out of 5)")
    ax.set_title("Needle Recall per Session")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"S{i+1}" for i in x])
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=8)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)

    # Chart 2: Average recall comparison
    ax = axes[1]
    avg_engine = np.mean(engine_scores)
    avg_naive = np.mean(naive_scores)
    bars = ax.bar(["Engine\n(priority)", "Naive\n(sliding window)"], [avg_engine, avg_naive], color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Avg Needles Recalled")
    ax.set_title("Average Recall Score")
    ax.set_ylim(0, 5.5)
    for bar, val in zip(bars, [avg_engine, avg_naive]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.1f}/5", ha="center", fontsize=12)

    # Chart 3: Which needles were lost most often
    ax = axes[2]
    needle_ids = [n["id"] for n in NEEDLES]
    engine_lost = {nid: 0 for nid in needle_ids}
    naive_lost = {nid: 0 for nid in needle_ids}
    for r in engine_results:
        for nid in r.get("needles_lost", []):
            engine_lost[nid] += 1
    for r in naive_results:
        for nid in r.get("needles_lost", []):
            naive_lost[nid] += 1

    x = np.arange(len(needle_ids))
    ax.bar(x - width/2, [engine_lost[nid] for nid in needle_ids], width, label="Engine", color="#2ecc71")
    ax.bar(x + width/2, [naive_lost[nid] for nid in needle_ids], width, label="Naive", color="#e74c3c")
    ax.set_xlabel("Needle")
    ax.set_ylabel("Times Lost (out of 10)")
    ax.set_title("Needle Loss Frequency")
    ax.set_xticks(x)
    ax.set_xticklabels([f"N{i+1}" for i in range(len(needle_ids))], rotation=0)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Dense NIAH v2: Priority Compaction vs Sliding Window\n"
        "(30 turns, 5 needles, ~15k tokens through 8k window)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


def main():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("ERROR: CEREBRAS_API_KEY not set")
        sys.exit(1)

    num_sessions = 10
    num_turns = 30
    placements = generate_needle_placements(num_sessions, num_turns)

    # Estimate tokens (context capped at 8k per API call)
    estimated_api_tokens = MAX_CONTEXT_TOKENS * num_sessions * 2 + num_sessions * 2 * 600
    print("=" * 60)
    print("Dense NIAH v2: Engine vs Naive Sliding Window")
    print("=" * 60)
    print(f"Sessions: {num_sessions}")
    print(f"Turns per session: {num_turns}")
    print(f"Needles per session: {len(NEEDLES)}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Estimated API tokens: ~{estimated_api_tokens:,}")
    print()
    print("Engine mode: priority-aware compaction (needles=2.0, filler=0.5)")
    print("Naive mode:  sliding window, drop oldest first (FIFO)")
    print()
    print("Needle placements (turn numbers):")
    for i, p in enumerate(placements):
        print(f"  Session {i+1}: turns {p}")
    print("-" * 60)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    results: list[dict] = []

    total_tests = num_sessions * 2
    test_num = 0

    for session_id, needle_turns in enumerate(placements):
        for mode in ["engine", "naive"]:
            test_num += 1
            label = f"[{test_num}/{total_tests}] Session {session_id+1} {mode.upper()}"
            print(f"{label} (needles at turns {needle_turns})...", end=" ", flush=True)

            result = run_session(session_id, needle_turns, mode, api_key, num_turns)
            results.append(result)

            if result["error"]:
                print(f"ERR: {result['error'][:60]}")
            else:
                recalled = result["recall_score"]
                in_ctx = len(result.get("needles_in_context", []))
                lost = result.get("needles_lost", [])
                compact = result["compaction_events"]
                print(f"Recalled {recalled}/5 (in_context={in_ctx}/5, lost={lost}) ttft={result['ttft']:.2f}s compact={compact}")

            time.sleep(1)

    # Save results
    output = {
        "timestamp": timestamp,
        "model": "llama3.1-8b",
        "benchmark": "dense_niah_v2",
        "description": "Engine (priority compaction) vs Naive (sliding window truncation)",
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "needles": NEEDLES,
        "placements": placements,
        "results": results,
    }

    json_path = RESULTS_DIR / f"niah_dense_v2_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Generate chart
    png_path = RESULTS_DIR / f"niah_dense_v2_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print summary
    engine_results = [r for r in results if r["mode"] == "engine"]
    naive_results = [r for r in results if r["mode"] == "naive"]

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for label, mode_results in [("ENGINE (priority compaction)", engine_results), ("NAIVE (sliding window)", naive_results)]:
        scores = [r["recall_score"] for r in mode_results]
        errors = sum(1 for r in mode_results if r["error"])
        avg = sum(scores) / len(scores) if scores else 0
        compactions = [r["compaction_events"] for r in mode_results]
        avg_compact = sum(compactions) / len(compactions) if compactions else 0
        avg_ctx = sum(r["context_tokens"] for r in mode_results) / len(mode_results) if mode_results else 0
        print(f"\n{label}:")
        print(f"  Avg recall: {avg:.1f}/5 ({100*avg/5:.0f}%)")
        print(f"  Scores: {scores}")
        print(f"  API errors: {errors}")
        print(f"  Avg context tokens sent: {avg_ctx:.0f}")
        print(f"  Avg compaction events: {avg_compact:.1f}")

        lost_counts = {}
        for r in mode_results:
            for nid in r.get("needles_lost", []):
                lost_counts[nid] = lost_counts.get(nid, 0) + 1
        if lost_counts:
            print(f"  Needles lost: {dict(sorted(lost_counts.items(), key=lambda x: -x[1]))}")
        else:
            print(f"  Needles lost: none!")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
