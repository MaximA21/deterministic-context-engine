#!/usr/bin/env python3
"""Adversarial Dense NIAH — stress-test for AutoPriority.

The challenge: needles and filler share the SAME keywords (filenames, function
names, error indicators). Only the needles contain ACTIONABLE info (line numbers,
specific bugs, concrete fixes). Filler mentions the same files/functions but
with irrelevant context (refactoring notes, history, style comments).

This tests whether keyword-based scoring can distinguish signal from noise
when keyword overlap is high.

Three modes: AutoPriority vs Hardcoded priority vs Naive sliding window.
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

from engine import ChunkLog, _estimate_tokens, extract_keywords, score_chunk

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --- 5 Needles: ACTIONABLE critical facts ---
# Each contains specific bugs, line numbers, concrete fixes.
NEEDLES = [
    {
        "id": "needle_1",
        "fact": "CRITICAL BUG: auth.py line 42, function validate_token() has an off-by-one error. Token expiry check uses `<=` instead of `<`, causing tokens to be valid for one extra second. Fix: change line 42 from `if expires <= now` to `if expires < now`.",
        "keyword": "off-by-one",
    },
    {
        "id": "needle_2",
        "fact": "URGENT: database.py function migrate_schema() will corrupt data if run on tables with more than 10 million rows. The batch_size on line 187 must be changed from 100000 to 10000. Schedule the fix for March 15th 3am UTC maintenance window.",
        "keyword": "10 million",
    },
    {
        "id": "needle_3",
        "fact": "SECURITY ALERT: api_gateway.py class RateLimiter has a bypass vulnerability. Function check_rate() on line 93 doesn't validate the X-Forwarded-For header, allowing attackers to spoof IPs. Rate limit must change from 100 to 250 req/min after patching.",
        "keyword": "x-forwarded-for",
    },
    {
        "id": "needle_4",
        "fact": "ACTION REQUIRED: New hire Elena Vasquez (senior engineer) starts Monday. She needs admin access to deploy_pipeline.py and must review the Exception handling in provision_access() before her first deploy. Her employee ID is EV-2847.",
        "keyword": "ev-2847",
    },
    {
        "id": "needle_5",
        "fact": "PRODUCTION INCIDENT: Server 10.42.88.7 in config.yaml has a memory leak in worker_pool.py function spawn_workers() line 156. RSS grows 50MB/hour. Temporary fix: restart cron every 4 hours. Permanent fix requires refactoring the connection pool in line 156-180.",
        "keyword": "50mb/hour",
    },
]

# --- Adversarial filler: SAME keywords, IRRELEVANT info ---
# Mentions auth.py, validate_token, database.py, migrate_schema, api_gateway.py,
# RateLimiter, deploy_pipeline.py, config.yaml, worker_pool.py, etc.
# But with boring, non-actionable context.
ADVERSARIAL_FILLER = [
    """Code review notes for auth.py from last Thursday:

The validate_token() function was originally written two years ago during
the initial authentication sprint. It has gone through three major refactors
since then. The latest style pass cleaned up variable names and added type
hints throughout. The function signature now follows our new convention of
returning Optional[TokenData] instead of raising exceptions. The docstring
was updated to match the Google style guide. Overall the auth module is in
good shape — 94% test coverage, no open linting warnings. The team agreed
that validate_token is one of our better-tested functions. We should use
it as a reference implementation for other validation functions. The auth.py
module also contains refresh_token() and revoke_session() which follow
similar patterns. All three were reviewed and approved without changes.""",

    """Historical notes on database.py evolution:

The migrate_schema() function has been our workhorse for schema changes
since version 2.0. It was inspired by Rails migrations and adapted for our
Python stack. The function handles both forward and rollback migrations
gracefully. Last quarter we added transaction savepoints to improve safety.
The database.py module is well-documented with 47 inline comments explaining
the rationale behind each design decision. The test suite includes 23
integration tests that run against a real PostgreSQL instance. The team
is proud of the migrate_schema implementation — it has handled 156
production migrations without incident. We considered switching to Alembic
but decided our custom solution better fits our multi-tenant architecture.
The module also contains connection_pool() and health_check() utilities.""",

    """Architecture overview of api_gateway.py:

The RateLimiter class was designed with extensibility in mind. It supports
multiple backends: Redis for distributed deployments and in-memory for local
development. The check_rate() function uses a sliding window algorithm that
provides smoother rate limiting compared to fixed windows. The api_gateway.py
module also includes RequestRouter, ResponseCache, and CircuitBreaker classes.
Each class has comprehensive unit tests and integration tests. The team did
a thorough security audit last month and found no concerns. The RateLimiter
configuration is stored in rate_limits.yaml and supports per-endpoint
customization. The api_gateway module is considered mature and stable — it
hasn't needed significant changes in over six months. Documentation was
updated last week to reflect the latest configuration options.""",

    """Onboarding documentation for deploy_pipeline.py:

The deploy_pipeline.py module is the backbone of our CI/CD process. New team
members are encouraged to read through it during their first week. The module
contains the DeployManager class which orchestrates builds, tests, and
deployments. The provision_access() function handles IAM role creation and
permission assignment. Exception handling follows our standard patterns with
structured logging. The deploy pipeline has been running reliably for 18
months. It processes approximately 40 deployments per day across all
environments. The pipeline supports canary releases, blue-green deployments,
and feature flags. The team maintains a runbook in Confluence that covers
common deployment scenarios. New hires typically shadow a senior engineer
for their first three deployments before doing one independently.""",

    """Infrastructure monitoring for worker_pool.py and config.yaml:

The worker_pool.py module manages our background job processing. The
spawn_workers() function creates worker processes based on the configuration
in config.yaml. The module has been running smoothly in production with
predictable resource usage patterns. Server metrics are tracked via Prometheus
and visualized in Grafana dashboards. The config.yaml file contains settings
for worker count, queue priorities, and timeout values. The infrastructure
team reviews these settings quarterly and adjusts based on traffic patterns.
The worker pool currently handles approximately 500K jobs per day with
99.7% success rate. Memory usage is stable and well within allocated limits.
The spawn_workers function was recently refactored for clarity but the
behavior remains unchanged. Connection pooling is handled efficiently.""",

    """Sprint retrospective notes mentioning auth.py and database.py:

The team discussed the recent auth.py improvements during the retro. Everyone
agreed the validate_token refactor improved readability significantly. The
database.py changes for multi-tenant support were also highlighted as a
success. The migrate_schema function's new dry-run mode was well received
by QA. Looking ahead, the team plans to add metrics collection to auth.py
and performance benchmarks for database.py query patterns. The config.yaml
standardization effort is 80% complete. The api_gateway.py documentation
was praised for its thoroughness. Overall team velocity increased by 12%
this sprint. The worker_pool.py monitoring alerts were fine-tuned to reduce
noise. No blocking dependencies for the upcoming sprint were identified.""",

    """Code metrics report for the backend modules:

auth.py: 342 lines, 94% coverage, complexity score 12, last modified 3 days ago.
database.py: 891 lines, 87% coverage, complexity score 24, last modified 1 week ago.
api_gateway.py: 567 lines, 91% coverage, complexity score 18, last modified 2 weeks ago.
deploy_pipeline.py: 423 lines, 82% coverage, complexity score 15, last modified 5 days ago.
worker_pool.py: 278 lines, 89% coverage, complexity score 11, last modified 4 days ago.
config.yaml: 156 lines, N/A coverage, last modified 1 week ago.

All modules pass our quality gates. The validate_token function in auth.py
has the highest individual test coverage at 98%. The migrate_schema function
in database.py has the most extensive integration test suite. The RateLimiter
class in api_gateway.py was recently benchmarked at 50K lookups/second.
The provision_access function in deploy_pipeline.py handles 200+ role
assignments daily. spawn_workers in worker_pool.py maintains consistent
memory footprint across restarts. Technical debt is manageable at 4.2%.""",

    """Dependency audit results touching multiple modules:

The auth.py module depends on PyJWT 2.8.0 and cryptography 41.0.0 — both
up to date. The validate_token function uses PyJWT's decode method internally.
database.py depends on psycopg2 2.9.9 and our custom migration framework.
The migrate_schema function has no external dependencies beyond psycopg2.
api_gateway.py depends on Redis 5.0.0 for the RateLimiter distributed backend.
check_rate() makes approximately 3 Redis calls per invocation. The
deploy_pipeline.py module depends on boto3 for AWS integration. The
provision_access function creates IAM roles using STS assume-role calls.
worker_pool.py depends on multiprocessing (stdlib) and our custom config
parser for config.yaml. spawn_workers reads worker count from config.yaml
at startup. All dependencies are pinned and pass vulnerability scans.
No CVEs found in the current dependency tree.""",
]

MAX_CONTEXT_TOKENS = 8000
SYSTEM_PROMPT = "You are a helpful assistant with perfect memory. Answer questions based on the conversation history provided."
SYSTEM_PROMPT_TOKENS = _estimate_tokens(SYSTEM_PROMPT)

RECALL_QUESTION = """I need you to recall the SPECIFIC ACTIONABLE items from our conversation. Not general notes — I need the exact details:

1. What is the specific bug in auth.py's validate_token? What line, what's wrong, and what's the fix?
2. What is the data corruption risk in database.py's migrate_schema? What row threshold and what batch_size change?
3. What is the security vulnerability in api_gateway.py's RateLimiter? What header is the problem?
4. Who is the new hire, what's their employee ID, and what must they review?
5. What is the memory leak in worker_pool.py? What server, what's the growth rate, and what's the temp fix?

Only answer with the SPECIFIC details mentioned during our conversation. Do NOT guess or fabricate details."""


def generate_needle_placements(num_sessions: int, num_turns: int = 30, num_needles: int = 5) -> list[list[int]]:
    placements = []
    rng = random.Random(42)
    for _ in range(num_sessions):
        turns = sorted(rng.sample(range(num_turns), num_needles))
        placements.append(turns)
    return placements


def sliding_window_truncate(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    if not messages:
        return messages
    last_msg = messages[-1]
    last_tokens = _estimate_tokens(last_msg["content"])
    budget = max_tokens - SYSTEM_PROMPT_TOKENS - last_tokens
    if budget <= 0:
        return [last_msg]
    kept: list[dict[str, str]] = []
    used = 0
    for msg in reversed(messages[:-1]):
        msg_tokens = _estimate_tokens(msg["content"])
        if used + msg_tokens <= budget:
            kept.append(msg)
            used += msg_tokens
        else:
            break
    kept.reverse()
    kept.append(last_msg)
    return kept


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,  # "auto_priority", "engine", or "naive"
    api_key: str,
    num_turns: int = 30,
) -> dict[str, Any]:
    from cerebras.cloud.sdk import Cerebras

    rng = random.Random(session_id * 1000 + hash(mode))
    compaction_log: list[dict] = []
    total_tokens_added = 0

    if mode == "auto_priority":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9, auto_priority=True,
        )
    elif mode == "engine":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9, auto_priority=False,
        )
    else:
        chunk_log = None
        raw_messages: list[dict[str, str]] = []

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            needle = NEEDLES[needle_idx]
            content = needle["fact"]

            if mode == "auto_priority":
                priority = 0.5  # No hardcoded priority
            elif mode == "engine":
                priority = 2.0  # Hardcoded
            else:
                priority = 0.5
        else:
            filler = rng.choice(ADVERSARIAL_FILLER)
            unique_salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] {filler}\n\nAdditional context for this turn:\n{rng.choice(ADVERSARIAL_FILLER)}{unique_salt}"
            priority = 0.5

        total_tokens_added += _estimate_tokens(content)

        if mode in ("auto_priority", "engine"):
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
    if mode in ("auto_priority", "engine"):
        chunk_log.append("user", RECALL_QUESTION, priority=2.0)
        messages = chunk_log.get_context()
        context_tokens = chunk_log.get_context_tokens()
        compaction_events = chunk_log.compaction_count
    else:
        raw_messages.append({"role": "user", "content": RECALL_QUESTION})
        messages = sliding_window_truncate(raw_messages, MAX_CONTEXT_TOKENS)
        context_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
        compaction_events = 0

    # Track which needles survived in context
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

    # Score: check which needles were recalled (using unique keywords)
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    auto_results = [r for r in results if r["mode"] == "auto_priority"]
    engine_results = [r for r in results if r["mode"] == "engine"]
    naive_results = [r for r in results if r["mode"] == "naive"]

    auto_scores = [r["recall_score"] for r in auto_results]
    engine_scores = [r["recall_score"] for r in engine_results]
    naive_scores = [r["recall_score"] for r in naive_results]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Chart 1: Recall scores per session
    ax = axes[0]
    n_sessions = len(auto_scores)
    x = range(n_sessions)
    width = 0.25
    ax.bar([i - width for i in x], auto_scores, width, label="AutoPriority", color="#3498db")
    ax.bar(list(x), engine_scores, width, label="Hardcoded Priority", color="#2ecc71")
    ax.bar([i + width for i in x], naive_scores, width, label="Naive (sliding window)", color="#e74c3c")
    ax.set_xlabel("Session")
    ax.set_ylabel("Needles Recalled (out of 5)")
    ax.set_title("Needle Recall per Session")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"S{i+1}" for i in x])
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=7)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)

    # Chart 2: Average recall comparison
    ax = axes[1]
    avg_auto = np.mean(auto_scores) if auto_scores else 0
    avg_engine = np.mean(engine_scores) if engine_scores else 0
    avg_naive = np.mean(naive_scores) if naive_scores else 0
    bars = ax.bar(
        ["AutoPriority\n(keyword)", "Hardcoded\n(priority=2)", "Naive\n(sliding window)"],
        [avg_auto, avg_engine, avg_naive],
        color=["#3498db", "#2ecc71", "#e74c3c"],
    )
    ax.set_ylabel("Avg Needles Recalled")
    ax.set_title("Average Recall Score")
    ax.set_ylim(0, 5.5)
    for bar, val in zip(bars, [avg_auto, avg_engine, avg_naive]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}/5", ha="center", fontsize=12)

    # Chart 3: Needles in context survival
    ax = axes[2]
    needle_ids = [n["id"] for n in NEEDLES]

    def count_in_context(mode_results):
        counts = {nid: 0 for nid in needle_ids}
        for r in mode_results:
            for nid in r.get("needles_in_context", []):
                counts[nid] += 1
        return counts

    auto_ctx = count_in_context(auto_results)
    engine_ctx = count_in_context(engine_results)
    naive_ctx = count_in_context(naive_results)

    x_arr = np.arange(len(needle_ids))
    ax.bar(x_arr - width, [auto_ctx[nid] for nid in needle_ids], width, label="AutoPriority", color="#3498db")
    ax.bar(x_arr, [engine_ctx[nid] for nid in needle_ids], width, label="Hardcoded", color="#2ecc71")
    ax.bar(x_arr + width, [naive_ctx[nid] for nid in needle_ids], width, label="Naive", color="#e74c3c")
    ax.set_xlabel("Needle")
    ax.set_ylabel(f"Times in Context (out of {n_sessions})")
    ax.set_title("Needle Survival in Context")
    ax.set_xticks(x_arr)
    ax.set_xticklabels([f"N{i+1}" for i in range(len(needle_ids))])
    ax.legend(fontsize=7)

    fig.suptitle(
        "Adversarial NIAH: High Keyword Overlap Stress Test\n"
        "(30 turns, 5 needles, adversarial filler shares filenames/functions, 8k window)",
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

    # Keyword overlap analysis
    print("Keyword overlap analysis:")
    needle_kws = set()
    for n in NEEDLES:
        kws = extract_keywords(n["fact"])
        needle_kws.update(kws)
        print(f"  {n['id']}: {sorted(kws)[:8]}...")
    print(f"  All needle keywords ({len(needle_kws)}): {sorted(needle_kws)[:12]}...")
    print()

    for i, filler in enumerate(ADVERSARIAL_FILLER[:5]):
        filler_kws = extract_keywords(filler)
        overlap = needle_kws & filler_kws
        print(f"  Filler {i+1} overlap with needles ({len(overlap)}): {sorted(overlap)[:8]}...")
    print()

    # Score analysis: how do needles vs filler score against needle keywords?
    print("Score analysis (needle keywords as scoring set):")
    for n in NEEDLES:
        score = score_chunk(n["fact"], needle_kws)
        print(f"  {n['id']}: score={score:.2f}")
    for i, f in enumerate(ADVERSARIAL_FILLER[:5]):
        score = score_chunk(f, needle_kws)
        print(f"  Filler {i+1}: score={score:.2f}")
    print()

    modes = ["auto_priority", "engine", "naive"]
    total_tests = num_sessions * len(modes)

    print("=" * 60)
    print("Adversarial Dense NIAH Benchmark")
    print("=" * 60)
    print(f"Sessions: {num_sessions}")
    print(f"Turns per session: {num_turns}")
    print(f"Needles per session: {len(NEEDLES)}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Modes: {modes}")
    print()
    print("ADVERSARIAL DESIGN: Filler shares filenames and function names")
    print("with needles. Only needles have actionable details.")
    print()
    print("Needle placements (turn numbers):")
    for i, p in enumerate(placements):
        print(f"  Session {i+1}: turns {p}")
    print("-" * 60)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    results: list[dict] = []
    test_num = 0

    for session_id, needle_turns in enumerate(placements):
        for mode in modes:
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
        "benchmark": "niah_adversarial",
        "description": "Adversarial NIAH: high keyword overlap between needles and filler",
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "needles": NEEDLES,
        "adversarial_filler_count": len(ADVERSARIAL_FILLER),
        "placements": placements,
        "results": results,
    }

    json_path = RESULTS_DIR / f"niah_adversarial_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Generate chart
    png_path = RESULTS_DIR / f"niah_adversarial_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for label, mode_key in [
        ("AUTOPRIORITY (keyword detection)", "auto_priority"),
        ("ENGINE (hardcoded priority=2.0)", "engine"),
        ("NAIVE (sliding window)", "naive"),
    ]:
        mode_results = [r for r in results if r["mode"] == mode_key]
        ok_results = [r for r in mode_results if not r["error"]]
        scores = [r["recall_score"] for r in ok_results]
        errors = len(mode_results) - len(ok_results)
        avg = sum(scores) / len(scores) if scores else 0
        compactions = [r["compaction_events"] for r in ok_results]
        avg_compact = sum(compactions) / len(compactions) if compactions else 0
        avg_ctx = sum(r["context_tokens"] for r in ok_results) / len(ok_results) if ok_results else 0
        in_ctx_counts = [len(r.get("needles_in_context", [])) for r in ok_results]
        avg_in_ctx = sum(in_ctx_counts) / len(in_ctx_counts) if in_ctx_counts else 0
        print(f"\n{label}:")
        print(f"  Avg recall: {avg:.1f}/5 ({100*avg/5:.0f}%)")
        print(f"  Scores: {[r['recall_score'] for r in mode_results]}")
        print(f"  Avg needles in context: {avg_in_ctx:.1f}/5")
        print(f"  API errors: {errors}")
        print(f"  Avg context tokens sent: {avg_ctx:.0f}")
        print(f"  Avg compaction events: {avg_compact:.1f}")

        lost_counts: dict[str, int] = {}
        for r in ok_results:
            for nid in r.get("needles_lost", []):
                lost_counts[nid] = lost_counts.get(nid, 0) + 1
        if lost_counts:
            print(f"  Needles lost: {dict(sorted(lost_counts.items(), key=lambda x: -x[1]))}")
        else:
            print(f"  Needles lost: none!")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
