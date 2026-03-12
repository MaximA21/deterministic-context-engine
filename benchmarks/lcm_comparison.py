#!/usr/bin/env python3
"""LCM vs BM25 Context Retention Benchmark.

Compares the DAG-based context compression from Voltropy's LCM paper
(Ehrlich & Blackman, 2026) against our BM25 deterministic scorer.

Methodology:
  - NIAH (Needle In A Haystack) adversarial test
  - 5 needles (critical facts) embedded among adversarial filler
  - 30 turns through a 32k token window (~3x compression ratio)
  - Measures: needle retention rate, compression ratio, context utilization
  - No LLM calls — pure context-management comparison

Modes tested:
  1. lcm_dag     — LCM DAG-based compression (extractive summarization proxy)
  2. structural  — Our BM25+Structural priority-scored compaction (default)
  3. bm25        — Our vanilla BM25 priority-scored compaction
  4. naive       — Sliding window truncation (baseline)

Results saved to results/lcm_comparison/
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine import ChunkLog, _estimate_tokens
from deterministic_context_engine.baselines.lcm_dag import LCMContextManager
from benchmarks.niah_adversarial import (
    NEEDLES, ADVERSARIAL_FILLER, generate_needle_placements,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "lcm_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
FILLERS_PER_TURN = 14  # ~96k total through 32k window = ~3x compression
NUM_SESSIONS = 10
NUM_TURNS = 30


def _check_needle_retained(context_text: str, needle: dict) -> bool:
    """Check if a needle's key information is retained in the context."""
    keyword = needle["keyword"].lower()
    return keyword in context_text.lower()


def _check_needle_exact(context_text: str, needle: dict) -> bool:
    """Check if the needle's full fact text is retained verbatim."""
    return needle["fact"] in context_text


def run_lcm_session(
    session_id: int,
    needle_turns: list[int],
    num_turns: int = NUM_TURNS,
) -> dict[str, Any]:
    """Run a session with LCM DAG-based compression."""
    rng = random.Random(session_id * 1000 + hash("lcm"))
    mgr = LCMContextManager(
        max_tokens=MAX_CONTEXT_TOKENS,
        soft_threshold=0.7,
        hard_threshold=0.9,
        summary_block_size=4,
    )

    total_tokens_added = 0
    compaction_events: list[dict] = []

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            content = NEEDLES[needle_idx]["fact"]
        else:
            fillers = [rng.choice(ADVERSARIAL_FILLER) for _ in range(FILLERS_PER_TURN)]
            salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] " + "\n\n".join(fillers) + salt

        total_tokens_added += _estimate_tokens(content)
        compactions_before = mgr.compaction_count
        mgr.append("user", content)
        mgr.next_turn()
        compactions_after = mgr.compaction_count

        if compactions_after > compactions_before:
            compaction_events.append({
                "turn": turn,
                "compactions": compactions_after - compactions_before,
                "active_tokens": mgr.current_tokens(),
            })

    # Check needle retention
    active_text = mgr.get_active_text()
    retention = {}
    for i, needle in enumerate(NEEDLES):
        turn = needle_turns[i] if i < len(needle_turns) else -1
        retained_keyword = _check_needle_retained(active_text, needle)
        retained_exact = _check_needle_exact(active_text, needle)

        # Also check lcm_grep — LCM's retrievability mechanism
        grep_results = mgr.lcm_grep(needle["keyword"])
        retrievable = len(grep_results) > 0

        retention[needle["id"]] = {
            "turn_injected": turn,
            "retained_in_active": retained_keyword,
            "retained_exact": retained_exact,
            "retrievable_via_grep": retrievable,
        }

    mgr.close()

    return {
        "mode": "lcm_dag",
        "session_id": session_id,
        "total_tokens_added": total_tokens_added,
        "final_active_tokens": mgr.current_tokens(),
        "compression_ratio": total_tokens_added / max(1, mgr.current_tokens()),
        "compaction_count": mgr.compaction_count,
        "compaction_events": compaction_events,
        "retention": retention,
        "needles_retained_active": sum(
            1 for r in retention.values() if r["retained_in_active"]
        ),
        "needles_retained_exact": sum(
            1 for r in retention.values() if r["retained_exact"]
        ),
        "needles_retrievable": sum(
            1 for r in retention.values() if r["retrievable_via_grep"]
        ),
    }


def run_bm25_session(
    session_id: int,
    needle_turns: list[int],
    num_turns: int = NUM_TURNS,
) -> dict[str, Any]:
    """Run a session with BM25 priority-scored compaction."""
    rng = random.Random(session_id * 1000 + hash("bm25"))
    chunk_log = ChunkLog(
        db_path=":memory:",
        max_tokens=MAX_CONTEXT_TOKENS,
        soft_threshold=0.7,
        hard_threshold=0.9,
        auto_priority=False,
        scoring_mode="bm25",
    )

    total_tokens_added = 0
    compaction_events: list[dict] = []

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            content = NEEDLES[needle_idx]["fact"]
        else:
            fillers = [rng.choice(ADVERSARIAL_FILLER) for _ in range(FILLERS_PER_TURN)]
            salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] " + "\n\n".join(fillers) + salt

        total_tokens_added += _estimate_tokens(content)
        tokens_before = chunk_log.current_tokens()
        compactions_before = chunk_log.compaction_count
        chunk_log.append("user", content, priority=0.5)
        chunk_log.next_turn()
        tokens_after = chunk_log.current_tokens()
        compactions_after = chunk_log.compaction_count

        if compactions_after > compactions_before:
            compaction_events.append({
                "turn": turn,
                "compactions": compactions_after - compactions_before,
                "tokens_before": tokens_before + _estimate_tokens(content),
                "tokens_after": tokens_after,
            })

    # Check needle retention
    rows = chunk_log._conn.execute(
        "SELECT content FROM chunks ORDER BY turn ASC"
    ).fetchall()
    active_text = "\n\n".join(row[0] for row in rows)

    retention = {}
    for i, needle in enumerate(NEEDLES):
        turn = needle_turns[i] if i < len(needle_turns) else -1
        retained_keyword = _check_needle_retained(active_text, needle)
        retained_exact = _check_needle_exact(active_text, needle)
        retention[needle["id"]] = {
            "turn_injected": turn,
            "retained_in_active": retained_keyword,
            "retained_exact": retained_exact,
            "retrievable_via_grep": retained_exact,  # BM25 retains or evicts
        }

    final_tokens = chunk_log.current_tokens()
    compaction_count = chunk_log.compaction_count
    chunk_log.close()

    return {
        "mode": "bm25",
        "session_id": session_id,
        "total_tokens_added": total_tokens_added,
        "final_active_tokens": final_tokens,
        "compression_ratio": total_tokens_added / max(1, final_tokens),
        "compaction_count": compaction_count,
        "compaction_events": compaction_events,
        "retention": retention,
        "needles_retained_active": sum(
            1 for r in retention.values() if r["retained_in_active"]
        ),
        "needles_retained_exact": sum(
            1 for r in retention.values() if r["retained_exact"]
        ),
        "needles_retrievable": sum(
            1 for r in retention.values() if r["retrievable_via_grep"]
        ),
    }


def run_structural_session(
    session_id: int,
    needle_turns: list[int],
    num_turns: int = NUM_TURNS,
) -> dict[str, Any]:
    """Run a session with BM25+Structural scorer (our default/best)."""
    rng = random.Random(session_id * 1000 + hash("structural"))
    chunk_log = ChunkLog(
        db_path=":memory:",
        max_tokens=MAX_CONTEXT_TOKENS,
        soft_threshold=0.7,
        hard_threshold=0.9,
        auto_priority=False,
        scoring_mode="structural",
    )

    total_tokens_added = 0
    compaction_events: list[dict] = []

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            content = NEEDLES[needle_idx]["fact"]
        else:
            fillers = [rng.choice(ADVERSARIAL_FILLER) for _ in range(FILLERS_PER_TURN)]
            salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] " + "\n\n".join(fillers) + salt

        total_tokens_added += _estimate_tokens(content)
        tokens_before = chunk_log.current_tokens()
        compactions_before = chunk_log.compaction_count
        chunk_log.append("user", content, priority=0.5)
        chunk_log.next_turn()
        tokens_after = chunk_log.current_tokens()
        compactions_after = chunk_log.compaction_count

        if compactions_after > compactions_before:
            compaction_events.append({
                "turn": turn,
                "compactions": compactions_after - compactions_before,
                "tokens_before": tokens_before + _estimate_tokens(content),
                "tokens_after": tokens_after,
            })

    rows = chunk_log._conn.execute(
        "SELECT content FROM chunks ORDER BY turn ASC"
    ).fetchall()
    active_text = "\n\n".join(row[0] for row in rows)

    retention = {}
    for i, needle in enumerate(NEEDLES):
        turn = needle_turns[i] if i < len(needle_turns) else -1
        retained_keyword = _check_needle_retained(active_text, needle)
        retained_exact = _check_needle_exact(active_text, needle)
        retention[needle["id"]] = {
            "turn_injected": turn,
            "retained_in_active": retained_keyword,
            "retained_exact": retained_exact,
            "retrievable_via_grep": retained_exact,
        }

    final_tokens = chunk_log.current_tokens()
    compaction_count = chunk_log.compaction_count
    chunk_log.close()

    return {
        "mode": "structural",
        "session_id": session_id,
        "total_tokens_added": total_tokens_added,
        "final_active_tokens": final_tokens,
        "compression_ratio": total_tokens_added / max(1, final_tokens),
        "compaction_count": compaction_count,
        "compaction_events": compaction_events,
        "retention": retention,
        "needles_retained_active": sum(
            1 for r in retention.values() if r["retained_in_active"]
        ),
        "needles_retained_exact": sum(
            1 for r in retention.values() if r["retained_exact"]
        ),
        "needles_retrievable": sum(
            1 for r in retention.values() if r["retrievable_via_grep"]
        ),
    }


def run_naive_session(
    session_id: int,
    needle_turns: list[int],
    num_turns: int = NUM_TURNS,
) -> dict[str, Any]:
    """Run a session with naive sliding-window truncation."""
    rng = random.Random(session_id * 1000 + hash("naive"))
    messages: list[str] = []
    total_tokens_added = 0

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            content = NEEDLES[needle_idx]["fact"]
        else:
            fillers = [rng.choice(ADVERSARIAL_FILLER) for _ in range(FILLERS_PER_TURN)]
            salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] " + "\n\n".join(fillers) + salt

        total_tokens_added += _estimate_tokens(content)
        messages.append(content)

        # Sliding window: drop oldest messages when over budget
        while sum(_estimate_tokens(m) for m in messages) > MAX_CONTEXT_TOKENS and len(messages) > 1:
            messages.pop(0)

    active_text = "\n\n".join(messages)
    final_tokens = sum(_estimate_tokens(m) for m in messages)

    retention = {}
    for i, needle in enumerate(NEEDLES):
        turn = needle_turns[i] if i < len(needle_turns) else -1
        retained_keyword = _check_needle_retained(active_text, needle)
        retained_exact = _check_needle_exact(active_text, needle)
        retention[needle["id"]] = {
            "turn_injected": turn,
            "retained_in_active": retained_keyword,
            "retained_exact": retained_exact,
            "retrievable_via_grep": retained_exact,
        }

    return {
        "mode": "naive",
        "session_id": session_id,
        "total_tokens_added": total_tokens_added,
        "final_active_tokens": final_tokens,
        "compression_ratio": total_tokens_added / max(1, final_tokens),
        "compaction_count": 0,
        "compaction_events": [],
        "retention": retention,
        "needles_retained_active": sum(
            1 for r in retention.values() if r["retained_in_active"]
        ),
        "needles_retained_exact": sum(
            1 for r in retention.values() if r["retained_exact"]
        ),
        "needles_retrievable": sum(
            1 for r in retention.values() if r["retrievable_via_grep"]
        ),
    }


def aggregate_results(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics over multiple sessions."""
    n = len(sessions)
    if n == 0:
        return {}

    mode = sessions[0]["mode"]
    total_needles = 5 * n

    retained_active = sum(s["needles_retained_active"] for s in sessions)
    retained_exact = sum(s["needles_retained_exact"] for s in sessions)
    retrievable = sum(s["needles_retrievable"] for s in sessions)
    avg_compression = sum(s["compression_ratio"] for s in sessions) / n
    avg_compactions = sum(s["compaction_count"] for s in sessions) / n

    # Per-needle retention rates
    needle_rates = {}
    for needle in NEEDLES:
        nid = needle["id"]
        active_count = sum(
            1 for s in sessions if s["retention"][nid]["retained_in_active"]
        )
        exact_count = sum(
            1 for s in sessions if s["retention"][nid]["retained_exact"]
        )
        needle_rates[nid] = {
            "active_retention": active_count / n,
            "exact_retention": exact_count / n,
        }

    return {
        "mode": mode,
        "sessions": n,
        "total_needles": total_needles,
        "retained_active": retained_active,
        "retained_exact": retained_exact,
        "retrievable": retrievable,
        "active_retention_rate": retained_active / total_needles,
        "exact_retention_rate": retained_exact / total_needles,
        "retrieval_rate": retrievable / total_needles,
        "avg_compression_ratio": avg_compression,
        "avg_compactions": avg_compactions,
        "per_needle": needle_rates,
    }


def format_report(results: dict[str, dict]) -> str:
    """Format a human-readable comparison report."""
    lines = [
        "=" * 80,
        "LCM vs Deterministic Context Engine — Adversarial NIAH Benchmark",
        "=" * 80,
        "",
        f"Setup: {NUM_SESSIONS} sessions, {NUM_TURNS} turns, {MAX_CONTEXT_TOKENS} token window",
        f"       {FILLERS_PER_TURN} adversarial fillers/turn, 5 needles/session",
        f"       Compression ratio: ~{FILLERS_PER_TURN * NUM_TURNS * 400 // MAX_CONTEXT_TOKENS}x",
        "",
        "-" * 80,
        f"{'Metric':<35} {'LCM DAG':>10} {'Struct':>10} {'BM25':>10} {'Naive':>10}",
        "-" * 80,
    ]

    modes = ["lcm_dag", "structural", "bm25", "naive"]
    metrics = [
        ("Active Retention Rate", "active_retention_rate", ".1%"),
        ("Exact Retention Rate", "exact_retention_rate", ".1%"),
        ("Retrieval Rate (grep/store)", "retrieval_rate", ".1%"),
        ("Avg Compression Ratio", "avg_compression_ratio", ".1f"),
        ("Avg Compaction Events", "avg_compactions", ".1f"),
    ]

    for label, key, fmt in metrics:
        vals = []
        for mode in modes:
            if mode in results:
                vals.append(f"{results[mode][key]:{fmt}}")
            else:
                vals.append("N/A")
        lines.append(f"{label:<35} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    lines.extend(["", "-" * 80, "Per-Needle Active Retention:", "-" * 80])
    lines.append(f"{'Needle':<35} {'LCM DAG':>10} {'Struct':>10} {'BM25':>10} {'Naive':>10}")

    for needle in NEEDLES:
        nid = needle["id"]
        vals = []
        for mode in modes:
            if mode in results and nid in results[mode]["per_needle"]:
                rate = results[mode]["per_needle"][nid]["active_retention"]
                vals.append(f"{rate:.0%}")
            else:
                vals.append("N/A")
        lines.append(f"  {nid:<33} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")

    lines.extend(["", "-" * 80, "Analysis:", "-" * 80])

    if "lcm_dag" in results and "structural" in results:
        lcm_rate = results["lcm_dag"]["active_retention_rate"]
        struct_rate = results["structural"]["active_retention_rate"]
        bm25_rate = results.get("bm25", {}).get("active_retention_rate", 0)

        lines.append("")
        lines.append("  APPROACH COMPARISON:")
        lines.append(f"    LCM DAG (summarization):  {lcm_rate:.0%} active retention")
        lines.append(f"    Structural (scoring):     {struct_rate:.0%} active retention")
        lines.append(f"    BM25 vanilla (scoring):   {bm25_rate:.0%} active retention")
        lines.append("")

        if abs(lcm_rate - struct_rate) < 0.05:
            lines.append("  LCM and Structural achieve comparable retention.")
            lines.append("  Both approaches protect high-value content, but via different mechanisms:")
            lines.append("    - LCM: summarizes blocks, preserving unique sentences")
            lines.append("    - Structural: scores chunks by structural density, evicts low-value")
        elif lcm_rate > struct_rate:
            diff = lcm_rate - struct_rate
            lines.append(f"  LCM outperforms Structural by {diff:.1%}.")
            lines.append("  Summarization preserves more info than eviction.")
        else:
            diff = struct_rate - lcm_rate
            lines.append(f"  Structural outperforms LCM by {diff:.1%}.")
            lines.append("  Content-aware scoring beats positional summarization.")

        lines.append("")
        lines.append("  KEY ARCHITECTURAL DIFFERENCES:")
        lines.append("    LCM:        Summarize + store originals. Costs LLM calls.")
        lines.append("    Structural: Score + evict. Zero LLM calls, <50ms decisions.")
        lines.append("")

        if "lcm_dag" in results:
            lcm_retrieval = results["lcm_dag"]["retrieval_rate"]
            lines.append(f"  LCM RETRIEVABILITY (via lcm_grep): {lcm_retrieval:.0%}")
            lines.append("  LCM's immutable store retains ALL originals regardless of compaction.")
            lines.append("  This is LCM's core advantage per the paper — eviction-based systems")
            lines.append("  permanently lose data, while LCM can always drill down to originals.")

    lines.append("")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    print("LCM vs BM25 Context Retention Benchmark")
    print(f"Running {NUM_SESSIONS} sessions x 4 modes...")
    print()

    placements = generate_needle_placements(NUM_SESSIONS, NUM_TURNS, 5)
    all_results: dict[str, list[dict]] = {
        "lcm_dag": [], "structural": [], "bm25": [], "naive": [],
    }

    for i in range(NUM_SESSIONS):
        needle_turns = placements[i]
        print(f"Session {i+1}/{NUM_SESSIONS} (needles at turns {needle_turns})")

        t0 = time.time()
        lcm_result = run_lcm_session(i, needle_turns)
        t_lcm = time.time() - t0

        t0 = time.time()
        struct_result = run_structural_session(i, needle_turns)
        t_struct = time.time() - t0

        t0 = time.time()
        bm25_result = run_bm25_session(i, needle_turns)
        t_bm25 = time.time() - t0

        t0 = time.time()
        naive_result = run_naive_session(i, needle_turns)
        t_naive = time.time() - t0

        print(f"  LCM:      {lcm_result['needles_retained_active']}/5 retained, "
              f"{lcm_result['needles_retrievable']}/5 retrievable ({t_lcm:.2f}s)")
        print(f"  Struct:   {struct_result['needles_retained_active']}/5 retained ({t_struct:.2f}s)")
        print(f"  BM25:     {bm25_result['needles_retained_active']}/5 retained ({t_bm25:.2f}s)")
        print(f"  Naive:    {naive_result['needles_retained_active']}/5 retained ({t_naive:.2f}s)")

        all_results["lcm_dag"].append(lcm_result)
        all_results["structural"].append(struct_result)
        all_results["bm25"].append(bm25_result)
        all_results["naive"].append(naive_result)

    # Aggregate
    aggregated = {}
    for mode, sessions in all_results.items():
        aggregated[mode] = aggregate_results(sessions)

    # Format and print report
    report = format_report(aggregated)
    print()
    print(report)

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON with full session data
    json_path = RESULTS_DIR / f"lcm_comparison_{timestamp}.json"
    output = {
        "metadata": {
            "timestamp": timestamp,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "fillers_per_turn": FILLERS_PER_TURN,
            "num_sessions": NUM_SESSIONS,
            "num_turns": NUM_TURNS,
            "paper": "Ehrlich & Blackman, LCM: Lossless Context Management, Voltropy PBC, 2026",
        },
        "aggregated": aggregated,
        "sessions": {
            mode: sessions for mode, sessions in all_results.items()
        },
    }
    json_path.write_text(json.dumps(output, indent=2))
    print(f"\nJSON results: {json_path}")

    # Text report
    report_path = RESULTS_DIR / f"lcm_comparison_{timestamp}.txt"
    report_path.write_text(report)
    print(f"Text report:  {report_path}")

    return aggregated


if __name__ == "__main__":
    main()
