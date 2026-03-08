#!/usr/bin/env python3
"""Combined Semantic Scorer Benchmark — tests MiniLM against TF-IDF across all NIAH variants.

Runs the semantic scorer (and optionally hybrid) as additional modes alongside
the existing 4 modes (goal_guided, engine, auto_priority, naive) on all benchmark types.

Measures context survival (simulation) for all benchmarks, then runs API evaluation
on the boilerplate benchmark where the TF-IDF weakness is most pronounced.
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

from engine import ChunkLog, _estimate_tokens, GoalGuidedScorer, SemanticScorer

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 8000
SYSTEM_PROMPT = "You are a helpful assistant with perfect memory. Answer questions based on the conversation history provided."


def simulate_context_survival(
    needles: list[dict],
    generate_filler,
    num_sessions: int,
    num_turns: int,
    fillers_per_turn: int,
    mode: str,
) -> dict[str, Any]:
    """Simulate context survival without API calls."""
    from benchmarks.niah_boilerplate import generate_needle_placements

    placements = generate_needle_placements(num_sessions, num_turns)
    session_results = []

    for sid, needle_turns in enumerate(placements):
        if mode == "semantic":
            log = ChunkLog(db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
                          soft_threshold=0.7, hard_threshold=0.9, scoring_mode="semantic")
        elif mode == "hybrid":
            log = ChunkLog(db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
                          soft_threshold=0.7, hard_threshold=0.9, scoring_mode="hybrid")
        elif mode == "goal_guided":
            log = ChunkLog(db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
                          soft_threshold=0.7, hard_threshold=0.9, goal_guided=True)
        elif mode == "auto_priority":
            log = ChunkLog(db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
                          soft_threshold=0.7, hard_threshold=0.9, auto_priority=True)
        elif mode == "engine":
            log = ChunkLog(db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
                          soft_threshold=0.7, hard_threshold=0.9)
        else:  # naive
            log = None
            raw_msgs: list[dict[str, str]] = []

        filler_idx = 0
        for turn in range(num_turns):
            if turn in needle_turns:
                idx = needle_turns.index(turn)
                content = needles[idx]["fact"]
                if mode in ("semantic", "hybrid", "goal_guided", "auto_priority"):
                    priority = 0.5
                elif mode == "engine":
                    priority = 2.0
                else:
                    priority = 0.5

                if log:
                    log.append("user", content, priority=priority)
                else:
                    raw_msgs.append({"role": "user", "content": content})
            else:
                for j in range(fillers_per_turn):
                    fc = generate_filler(sid * 10000 + filler_idx)
                    filler_idx += 1
                    if log:
                        log.append("user", fc, priority=0.5)
                    else:
                        raw_msgs.append({"role": "user", "content": fc})
            if log:
                log.next_turn()

        if log:
            ctx = log.get_context()
            compaction = log.compaction_count
        else:
            # naive: just take last messages that fit
            from benchmarks.niah_boilerplate import sliding_window_truncate
            ctx = sliding_window_truncate(raw_msgs, MAX_CONTEXT_TOKENS)
            compaction = 0

        text = " ".join(m["content"] for m in ctx)
        survived = [n["id"] for n in needles if n["keyword"].lower() in text.lower()]
        lost = [n["id"] for n in needles if n["id"] not in survived]

        session_results.append({
            "session_id": sid,
            "needle_turns": needle_turns,
            "survived": len(survived),
            "survived_ids": survived,
            "lost_ids": lost,
            "compaction": compaction,
        })
        if log:
            log.close()

    scores = [r["survived"] for r in session_results]
    avg = sum(scores) / len(scores) if scores else 0

    lost_counts: dict[str, int] = {}
    for r in session_results:
        for nid in r["lost_ids"]:
            lost_counts[nid] = lost_counts.get(nid, 0) + 1

    return {
        "mode": mode,
        "avg_survival": round(avg, 2),
        "per_session": scores,
        "lost_counts": lost_counts,
        "sessions": session_results,
    }


def run_all_simulations():
    """Run context survival simulations across all benchmarks and modes."""
    # Import all benchmark data
    from benchmarks.niah_boilerplate import (
        NEEDLES as BP_NEEDLES,
        _generate_filler as bp_filler,
    )
    from benchmarks.niah_semantic_gap import (
        NEEDLES as SG_NEEDLES,
        _generate_filler as sg_filler,
    )
    from benchmarks.niah_adversarial import (
        NEEDLES as ADV_NEEDLES,
        ADVERSARIAL_FILLER as adv_fillers,
    )
    from benchmarks.niah_goalguided import (
        NEEDLES as GG_NEEDLES,
        _generate_filler as gg_filler,
    )

    # Adversarial filler adapter
    def adv_filler_gen(seed):
        rng = random.Random(seed)
        f1 = rng.choice(adv_fillers)
        f2 = rng.choice(adv_fillers)
        return f"[Turn {seed}] {f1}\n\n{f2}\n[Ref: {seed}_{rng.randint(0,999999)}]"

    benchmarks = [
        {
            "name": "Boilerplate",
            "needles": BP_NEEDLES,
            "filler_fn": bp_filler,
            "fillers_per_turn": 5,
        },
        {
            "name": "Semantic Gap",
            "needles": SG_NEEDLES,
            "filler_fn": sg_filler,
            "fillers_per_turn": 4,
        },
        {
            "name": "Adversarial",
            "needles": ADV_NEEDLES,
            "filler_fn": adv_filler_gen,
            "fillers_per_turn": 1,
        },
        {
            "name": "Fair Goal-Guided",
            "needles": GG_NEEDLES,
            "filler_fn": gg_filler,
            "fillers_per_turn": 4,
        },
    ]

    modes = ["semantic", "hybrid", "goal_guided", "engine", "auto_priority", "naive"]
    all_results: dict[str, dict[str, Any]] = {}

    for bench in benchmarks:
        print(f"\n{'='*60}")
        print(f"  {bench['name']} Benchmark")
        print(f"{'='*60}")
        bench_results = {}

        for mode in modes:
            print(f"  {mode:16s}...", end=" ", flush=True)
            t0 = time.time()
            result = simulate_context_survival(
                bench["needles"],
                bench["filler_fn"],
                num_sessions=10,
                num_turns=30,
                fillers_per_turn=bench["fillers_per_turn"],
                mode=mode,
            )
            elapsed = time.time() - t0
            bench_results[mode] = result
            print(f"avg={result['avg_survival']:.1f}/5  sessions={result['per_session']}  ({elapsed:.1f}s)")

        all_results[bench["name"]] = bench_results

    return all_results


def print_master_table(all_results: dict):
    """Print comparison table across all benchmarks."""
    modes = ["semantic", "hybrid", "goal_guided", "engine", "auto_priority", "naive"]
    mode_labels = {
        "semantic": "Semantic",
        "hybrid": "Hybrid",
        "goal_guided": "TF-IDF",
        "engine": "Hardcoded",
        "auto_priority": "Keywords",
        "naive": "Naive",
    }

    print(f"\n{'='*80}")
    print("MASTER COMPARISON TABLE — Context Survival (avg needles in context / 5)")
    print(f"{'='*80}")

    header = f"{'Benchmark':<20s}"
    for mode in modes:
        header += f" | {mode_labels[mode]:>9s}"
    print(header)
    print("-" * 80)

    for bench_name, bench_results in all_results.items():
        row = f"{bench_name:<20s}"
        for mode in modes:
            r = bench_results.get(mode, {})
            avg = r.get("avg_survival", 0)
            row += f" | {avg:>8.1f}/5"
        print(row)

    print("-" * 80)
    print(f"{'='*80}")


def generate_combined_chart(all_results: dict, output_path: Path):
    """Generate combined chart comparing all modes across benchmarks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes = ["semantic", "hybrid", "goal_guided", "engine", "auto_priority", "naive"]
    mode_labels = ["Semantic\n(MiniLM)", "Hybrid\n(30/70)", "TF-IDF\n(Goal-Guided)", "Hardcoded\n(priority=2)", "Keywords\nOnly", "Naive\n(sliding)"]
    mode_colors = ["#e67e22", "#f39c12", "#9b59b6", "#2ecc71", "#3498db", "#e74c3c"]

    bench_names = list(all_results.keys())
    n_benchmarks = len(bench_names)
    n_modes = len(modes)

    fig, axes = plt.subplots(1, n_benchmarks, figsize=(7 * n_benchmarks, 6), sharey=True)
    if n_benchmarks == 1:
        axes = [axes]

    for ax_idx, bench_name in enumerate(bench_names):
        ax = axes[ax_idx]
        bench_results = all_results[bench_name]
        x = np.arange(n_modes)
        avgs = [bench_results.get(m, {}).get("avg_survival", 0) for m in modes]
        bars = ax.bar(x, avgs, color=mode_colors)
        ax.set_title(bench_name, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(mode_labels, fontsize=7)
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Avg Needles Survived (out of 5)" if ax_idx == 0 else "")
        ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Semantic Scorer (MiniLM) vs TF-IDF — Context Survival Across All Benchmarks\n"
        "(10 sessions × 30 turns × 5 needles, 8k context window)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


def benchmark_latency():
    """Benchmark scoring latency for MiniLM on 100 chunks."""
    from benchmarks.niah_boilerplate import NEEDLES, _generate_filler, RECALL_QUESTION

    scorer = SemanticScorer()
    chunks = [(f"chunk_{i}", _generate_filler(i)) for i in range(95)]
    chunks += [(f"needle_{n['id']}", n["fact"]) for n in NEEDLES]

    # Warm-up
    scorer.score_chunks(RECALL_QUESTION, chunks[:10])

    # Benchmark
    times = []
    for _ in range(5):
        t0 = time.time()
        scorer.score_chunks(RECALL_QUESTION, chunks)
        times.append((time.time() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)

    print(f"\n{'='*60}")
    print("LATENCY BENCHMARK — MiniLM Scoring (100 chunks)")
    print(f"{'='*60}")
    print(f"  Runs:    5")
    print(f"  Avg:     {avg_ms:.1f}ms")
    print(f"  Min:     {min_ms:.1f}ms")
    print(f"  Max:     {max_ms:.1f}ms")
    if avg_ms < 50:
        print(f"  Status:  PASS (under 50ms)")
    else:
        print(f"  Status:  OVER TARGET (50ms) — {avg_ms:.0f}ms on CPU")
        print(f"  Note:    MiniLM on CPU requires ~{avg_ms:.0f}ms for 100 chunks.")
        print(f"           GPU or ONNX Runtime would reduce to <50ms.")
    print(f"{'='*60}")

    return {"avg_ms": round(avg_ms, 1), "min_ms": round(min_ms, 1), "max_ms": round(max_ms, 1)}


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    print("=" * 60)
    print("SEMANTIC SCORER (MiniLM) — COMBINED BENCHMARK")
    print("=" * 60)

    # Latency
    latency = benchmark_latency()

    # TF-IDF vs Semantic discrimination comparison
    print(f"\n{'='*60}")
    print("DISCRIMINATION ANALYSIS")
    print(f"{'='*60}")

    from benchmarks.niah_boilerplate import NEEDLES as BP_N, _generate_filler as bp_f, RECALL_QUESTION as BP_Q
    from benchmarks.niah_semantic_gap import NEEDLES as SG_N, _generate_filler as sg_f, RECALL_QUESTION as SG_Q

    tfidf = GoalGuidedScorer()
    semantic = SemanticScorer()

    for label, needles, filler_fn, recall_q, n_filler in [
        ("Boilerplate", BP_N, bp_f, BP_Q, 25),
        ("Semantic Gap", SG_N, sg_f, SG_Q, 20),
    ]:
        chunks = [(f"needle_{n['id']}", n["fact"]) for n in needles]
        chunks += [(f"filler_{i}", filler_fn(i)) for i in range(n_filler)]

        ts = tfidf.score_chunks(recall_q, chunks)
        ss = semantic.score_chunks(recall_q, chunks)

        for scorer_name, scores in [("TF-IDF", ts), ("Semantic", ss)]:
            ns = [scores[h] for h, _ in chunks if h.startswith("needle")]
            fs = [scores[h] for h, _ in chunks if h.startswith("filler")]
            gap = min(ns) - max(fs)
            print(f"  {label:15s} {scorer_name:9s}: needles [{min(ns):.3f},{max(ns):.3f}] filler [{min(fs):.3f},{max(fs):.3f}] gap={gap:+.3f}")
        print()

    # Run all simulations
    all_results = run_all_simulations()

    # Master table
    print_master_table(all_results)

    # Generate chart
    png_path = RESULTS_DIR / f"niah_semantic_{timestamp}.png"
    try:
        generate_combined_chart(all_results, png_path)
    except Exception as e:
        print(f"Chart error: {e}")

    # Save results
    output = {
        "timestamp": timestamp,
        "benchmark": "niah_semantic_combined",
        "description": "Combined Semantic Scorer (MiniLM) benchmark across all NIAH variants",
        "latency": latency,
        "results": {},
    }
    for bench_name, bench_results in all_results.items():
        output["results"][bench_name] = {}
        for mode, data in bench_results.items():
            output["results"][bench_name][mode] = {
                "avg_survival": data["avg_survival"],
                "per_session": data["per_session"],
                "lost_counts": data["lost_counts"],
            }

    json_path = RESULTS_DIR / f"niah_semantic_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
