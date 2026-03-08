#!/usr/bin/env python3
"""Dense NIAH AutoPriority benchmark — three-way comparison.

Compares:
1. AutoPriority engine (keyword-based dynamic scoring, no hardcoded priorities)
2. Hardcoded engine (manual priority=2.0 on needles)
3. Naive sliding window (FIFO truncation)

Needles contain identifiable keywords (filenames, function names, error indicators)
so AutoPriority CAN detect them. Filler is generic without those keywords.
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

# --- 5 Needles with keyword-rich content ---
# Each needle contains filenames, function names, or error indicators
# that extract_keywords() can detect.
NEEDLES = [
    {
        "id": "needle_1",
        "fact": "IMPORTANT UPDATE: The critical bug fix is in `auth_handler.py` — function `validate_token()` was returning None on expired sessions. Deploy fix ASAP.",
        "keyword": "validate_token",
    },
    {
        "id": "needle_2",
        "fact": "IMPORTANT UPDATE: The database migration script `migrate_schema.sql` must run on March 15th at 3am UTC. Error in `alter_tenants()` was fixed.",
        "keyword": "march 15",
    },
    {
        "id": "needle_3",
        "fact": "IMPORTANT UPDATE: Rate limit Exception in `rate_limiter.rs` — changed from 100 to 250 requests per minute. See `config.toml` for details.",
        "keyword": "250",
    },
    {
        "id": "needle_4",
        "fact": "IMPORTANT UPDATE: New hire Elena Vasquez starts Monday. Her onboarding issue is tracked in `onboarding.py` class `EmployeeSetup`. Fix the bug in `provision_access()`.",
        "keyword": "elena",
    },
    {
        "id": "needle_5",
        "fact": "IMPORTANT UPDATE: Production server IP 10.42.88.7 — update `deploy_config.yaml` and function `get_server_endpoint()`. Critical fail if wrong IP used.",
        "keyword": "10.42.88.7",
    },
]

# Generic filler WITHOUT any needle keywords.
# Carefully avoids: filenames (.py, .rs, etc.), function/class names,
# error indicators (error, exception, bug, fix, fail, issue, critical, important, update),
# IP addresses, and quoted strings matching needle content.
FILLER_TEMPLATES = [
    """Here are the latest performance metrics from our monitoring dashboard:

The system has been running smoothly for the past week. Average response
times are within acceptable ranges. Memory utilization is at 65% across
all pods. CPU usage peaks at 45% during business hours. The load balancer
is distributing traffic evenly. No alerts have been triggered. The cache
hit ratio remains at 92% which is above our target of 85%. Queue depths
are nominal with average processing times under 200ms. Disk IOPS are
well within provisioned capacity. Network throughput shows normal patterns
with no anomalies detected in the last 24 hours. All health checks are
passing. The monitoring system itself has 99.99% uptime this quarter.""",

    """Weekly standup notes from the development team:

The team has been making good progress on the current sprint. Several
user stories were completed ahead of schedule. Code reviews are being
turned around within 24 hours. The design team provided revised mockups
for the dashboard redesign. QA has completed regression testing for the
latest release candidate. No blocking concerns were reported. The team
velocity has improved by 15% compared to last sprint. Technical debt
reduction items are being addressed incrementally. Documentation changes
are keeping pace with feature development. Cross-team collaboration
sessions were productive. The retrospective action items from last
sprint have all been addressed. Planning for next sprint begins tomorrow.""",

    """Infrastructure cost analysis for the current quarter:

Cloud spending is tracking 8% under budget. The cost optimization
initiatives from last month are showing results. Reserved instance
coverage has been increased to 70% for compute resources. Spot instance
utilization for batch workloads saved approximately twelve thousand this month.
Storage costs are stable with lifecycle policies managing data tiering
effectively. Data transfer costs decreased after implementing regional
caching. The auto-scaling policies are right-sized based on traffic
analysis. Container resource limits were tuned to reduce over-provisioning.
Database costs remain the largest line item at 35% of total spend.
CDN costs are minimal due to high cache hit rates. Monitoring and
observability tooling costs are within allocated budget.""",

    """Summary of the latest architectural decision record:

The team has decided to continue with the current approach for the
time being. Alternative solutions were evaluated but the benefits
did not justify the migration effort. Performance benchmarks showed
the current system meets our requirements. Scalability concerns have
been addressed through horizontal scaling. The team agreed to revisit
this decision in six months. Risk assessment showed low probability of
trouble with the current approach. Compliance requirements are met.
Security review found no concerns. The operational overhead is
manageable with current team size. Knowledge sharing sessions have
ensured no single points of dependency in team expertise. Vendor lock-in
risk is minimal due to abstraction layers. Total cost of ownership
analysis favored the current approach over alternatives.""",

    """Notes from the client feedback session:

Overall satisfaction scores are at 4.2 out of 5. Key feedback areas
include improved navigation, faster load times, and better mobile
experience. Feature requests have been prioritized and added to the
backlog. No major usability concerns were reported. The onboarding
flow received positive feedback after the recent redesign. Support
ticket volume has decreased by 20% since the last release. NPS score
improved by 8 points. Client retention metrics are strong. The feature
adoption rate for new capabilities is tracking above expectations.
Training materials were well received. Integration documentation
was praised for clarity. The feedback loop process itself was
appreciated by stakeholders.""",
]

MAX_CONTEXT_TOKENS = 8000
SYSTEM_PROMPT = "You are a helpful assistant with perfect memory. Answer questions based on the conversation history provided."
SYSTEM_PROMPT_TOKENS = _estimate_tokens(SYSTEM_PROMPT)

RECALL_QUESTION = """Please recall ALL important updates and facts mentioned during our conversation.
Specifically, I need you to tell me:
1. What was the critical bug fix about and which file was it in?
2. When is the database migration scheduled and what script is used?
3. What was the rate limit changed to?
4. What is the new hire's name and when does she start?
5. What is the production server IP address?

Answer each question based ONLY on what was mentioned in our conversation."""


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
        # AutoPriority: all chunks start at default priority, engine re-scores dynamically
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9, auto_priority=True,
        )
    elif mode == "engine":
        # Hardcoded priority mode
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
                priority = 0.5  # NO hardcoded priority — AutoPriority must detect it
            elif mode == "engine":
                priority = 2.0  # Hardcoded priority
            else:
                priority = 0.5
        else:
            filler = rng.choice(FILLER_TEMPLATES)
            unique_salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] {filler}\n\nAdditional context for this turn:\n{rng.choice(FILLER_TEMPLATES)}{unique_salt}"
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
    x = range(len(auto_scores))
    width = 0.25
    ax.bar([i - width for i in x], auto_scores, width, label="AutoPriority", color="#3498db")
    ax.bar([i for i in x], engine_scores, width, label="Hardcoded Priority", color="#2ecc71")
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

    # Chart 3: Needles in context (survived compaction/truncation)
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
    ax.set_ylabel(f"Times in Context (out of {len(auto_results)})")
    ax.set_title("Needle Survival in Context")
    ax.set_xticks(x_arr)
    ax.set_xticklabels([f"N{i+1}" for i in range(len(needle_ids))], rotation=0)
    ax.legend(fontsize=7)

    fig.suptitle(
        "Dense NIAH AutoPriority: Keyword Detection vs Hardcoded vs Sliding Window\n"
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

    # Quick sanity check: verify extract_keywords works on our needles
    print("Keyword extraction sanity check:")
    for needle in NEEDLES:
        kws = extract_keywords(needle["fact"])
        print(f"  {needle['id']}: {sorted(kws)[:8]}...")
    print()

    # Verify filler does NOT match needle keywords
    needle_kws = set()
    for n in NEEDLES:
        needle_kws.update(extract_keywords(n["fact"]))
    for i, filler in enumerate(FILLER_TEMPLATES):
        filler_kws = extract_keywords(filler)
        overlap = needle_kws & filler_kws
        print(f"  Filler {i+1} keyword overlap with needles: {overlap or 'none'}")
    print()

    modes = ["auto_priority", "engine", "naive"]
    total_tests = num_sessions * len(modes)

    print("=" * 60)
    print("Dense NIAH AutoPriority Benchmark")
    print("=" * 60)
    print(f"Sessions: {num_sessions}")
    print(f"Turns per session: {num_turns}")
    print(f"Needles per session: {len(NEEDLES)}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Modes: {modes}")
    print()
    print("AutoPriority: keyword-based dynamic scoring (all chunks start at priority=0.5)")
    print("Engine:       hardcoded priority (needles=2.0, filler=0.5)")
    print("Naive:        sliding window, drop oldest first (FIFO)")
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
        "benchmark": "niah_autopriority",
        "description": "AutoPriority (keyword detection) vs Hardcoded (priority=2.0) vs Naive (sliding window)",
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "needles": NEEDLES,
        "placements": placements,
        "results": results,
    }

    json_path = RESULTS_DIR / f"niah_autopriority_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Generate chart
    png_path = RESULTS_DIR / f"niah_autopriority_{timestamp}.png"
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
        scores = [r["recall_score"] for r in mode_results]
        errors = sum(1 for r in mode_results if r["error"])
        avg = sum(scores) / len(scores) if scores else 0
        compactions = [r["compaction_events"] for r in mode_results]
        avg_compact = sum(compactions) / len(compactions) if compactions else 0
        avg_ctx = sum(r["context_tokens"] for r in mode_results) / len(mode_results) if mode_results else 0
        in_ctx_counts = []
        for r in mode_results:
            in_ctx_counts.append(len(r.get("needles_in_context", [])))
        avg_in_ctx = sum(in_ctx_counts) / len(in_ctx_counts) if in_ctx_counts else 0
        print(f"\n{label}:")
        print(f"  Avg recall: {avg:.1f}/5 ({100*avg/5:.0f}%)")
        print(f"  Scores: {scores}")
        print(f"  Avg needles in context: {avg_in_ctx:.1f}/5")
        print(f"  API errors: {errors}")
        print(f"  Avg context tokens sent: {avg_ctx:.0f}")
        print(f"  Avg compaction events: {avg_compact:.1f}")

        lost_counts: dict[str, int] = {}
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
