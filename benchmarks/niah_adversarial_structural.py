#!/usr/bin/env python3
"""Adversarial NIAH with Structural Fingerprinting — context survival benchmark.

Tests whether structural fingerprinting (hashing text SHAPE separately from
content) improves needle retention in adversarial scenarios where filler
shares the same vocabulary as needles.

Runs context survival simulation (no LLM API needed) across scoring modes:
- structural: TF-IDF + structural fingerprint density/uniqueness
- goal_guided (tfidf): baseline TF-IDF uniqueness
- entity_aware: TF-IDF + entity extraction
- auto_priority: keyword scoring
- engine: hardcoded priority (ceiling)
- naive: sliding window (floor)
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

from engine import ChunkLog, _estimate_tokens, StructuralFingerprinter, StructuralScorer

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 8000
SYSTEM_PROMPT_TOKENS = _estimate_tokens(
    "You are a helpful assistant with perfect memory."
)

# Import adversarial test data
from benchmarks.niah_adversarial import (
    NEEDLES,
    ADVERSARIAL_FILLER,
    RECALL_QUESTION,
    generate_needle_placements,
    sliding_window_truncate,
)


def simulate_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,
    num_turns: int = 30,
) -> dict[str, Any]:
    """Simulate a session and check which needles survive compaction."""
    rng = random.Random(session_id * 1000 + hash(mode))

    scoring_mode_map = {
        "structural": "structural",
        "goal_guided": None,
        "entity_aware": "entity_aware",
        "auto_priority": None,
        "engine": None,
        "naive": None,
    }

    if mode == "naive":
        log = None
        raw_msgs: list[dict[str, str]] = []
    elif mode == "engine":
        log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
        )
    elif mode == "auto_priority":
        log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9, auto_priority=True,
        )
    elif mode == "goal_guided":
        log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9, goal_guided=True,
        )
    else:
        log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode=scoring_mode_map[mode],
        )

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            content = NEEDLES[needle_idx]["fact"]
            if mode == "engine":
                priority = 2.0
            else:
                priority = 0.5
        else:
            filler = rng.choice(ADVERSARIAL_FILLER)
            unique_salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = (
                f"[Turn {turn+1}] {filler}\n\n"
                f"Additional context for this turn:\n"
                f"{rng.choice(ADVERSARIAL_FILLER)}{unique_salt}"
            )
            priority = 0.5

        if log:
            log.append("user", content, priority=priority)
            log.next_turn()
        else:
            raw_msgs.append({"role": "user", "content": content})

    # Add recall question (triggers final rescoring)
    if log:
        log.append("user", RECALL_QUESTION, priority=2.0)
        ctx = log.get_context()
        compaction = log.compaction_count
    else:
        raw_msgs.append({"role": "user", "content": RECALL_QUESTION})
        ctx = sliding_window_truncate(raw_msgs, MAX_CONTEXT_TOKENS)
        compaction = 0

    # Check which needles survived
    context_text = " ".join(m["content"] for m in ctx)
    survived = []
    lost = []
    for needle in NEEDLES:
        if needle["keyword"].lower() in context_text.lower():
            survived.append(needle["id"])
        else:
            lost.append(needle["id"])

    ctx_tokens = sum(_estimate_tokens(m["content"]) for m in ctx)

    result = {
        "session_id": session_id,
        "mode": mode,
        "needle_turns": needle_turns,
        "survived": survived,
        "lost": lost,
        "survival_count": len(survived),
        "compaction": compaction,
        "context_tokens": ctx_tokens,
    }

    if log:
        log.close()

    return result


def run_benchmark(
    num_sessions: int = 10,
    num_turns: int = 30,
) -> dict[str, Any]:
    """Run full adversarial benchmark across all scoring modes."""
    placements = generate_needle_placements(num_sessions, num_turns)
    modes = ["structural", "goal_guided", "entity_aware", "auto_priority", "engine", "naive"]
    mode_labels = {
        "structural": "Structural (TF-IDF + fingerprint)",
        "goal_guided": "TF-IDF (goal-guided)",
        "entity_aware": "Entity-Aware (TF-IDF + entities)",
        "auto_priority": "Keywords (AutoPriority)",
        "engine": "Hardcoded (priority=2.0)",
        "naive": "Naive (sliding window)",
    }

    all_results: dict[str, list[dict]] = {m: [] for m in modes}

    print("=" * 70)
    print("ADVERSARIAL NIAH — Structural Fingerprinting Benchmark")
    print("=" * 70)
    print(f"Sessions: {num_sessions}, Turns: {num_turns}, Needles: {len(NEEDLES)}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Modes: {list(mode_labels.values())}")
    print()

    for mode in modes:
        label = mode_labels[mode]
        print(f"\n--- {label} ---")
        t0 = time.time()

        for sid, needle_turns in enumerate(placements):
            result = simulate_session(sid, needle_turns, mode, num_turns)
            all_results[mode].append(result)
            survived = result["survival_count"]
            lost = result["lost"]
            compact = result["compaction"]
            print(
                f"  S{sid+1:2d}: {survived}/5 survived "
                f"(lost={lost}) compact={compact}"
            )

        elapsed = time.time() - t0
        scores = [r["survival_count"] for r in all_results[mode]]
        avg = sum(scores) / len(scores)
        print(f"  Avg: {avg:.1f}/5 ({elapsed:.1f}s)")

    return all_results


def print_summary(all_results: dict[str, list[dict]]) -> None:
    """Print comparison summary table."""
    mode_labels = {
        "structural": "Structural",
        "goal_guided": "TF-IDF",
        "entity_aware": "Entity-Aware",
        "auto_priority": "Keywords",
        "engine": "Hardcoded",
        "naive": "Naive",
    }

    print()
    print("=" * 70)
    print("SUMMARY — Adversarial Context Survival (avg needles in context / 5)")
    print("=" * 70)
    print(f"{'Mode':<20s} | {'Avg':>5s} | {'Scores':>40s} | {'Lost':>20s}")
    print("-" * 95)

    for mode in ["structural", "goal_guided", "entity_aware", "auto_priority", "engine", "naive"]:
        results = all_results.get(mode, [])
        scores = [r["survival_count"] for r in results]
        avg = sum(scores) / len(scores) if scores else 0

        lost_counts: dict[str, int] = {}
        for r in results:
            for nid in r["lost"]:
                lost_counts[nid] = lost_counts.get(nid, 0) + 1

        label = mode_labels.get(mode, mode)
        scores_str = str(scores)
        lost_str = str(dict(sorted(lost_counts.items()))) if lost_counts else "none"
        print(f"{label:<20s} | {avg:>4.1f} | {scores_str:>40s} | {lost_str}")

    print("=" * 70)


def discrimination_analysis() -> None:
    """Show per-chunk score discrimination for structural vs TF-IDF."""
    from engine import GoalGuidedScorer

    print()
    print("=" * 70)
    print("DISCRIMINATION ANALYSIS — Structural vs TF-IDF")
    print("=" * 70)

    chunks = [(f'needle_{n["id"]}', n['fact']) for n in NEEDLES]
    chunks += [(f'filler_{i}', f) for i, f in enumerate(ADVERSARIAL_FILLER)]

    structural = StructuralScorer()
    tfidf = GoalGuidedScorer()

    for label, scorer in [("Structural", structural), ("TF-IDF", tfidf)]:
        scores = scorer.score_chunks(RECALL_QUESTION, chunks)
        ns = [scores[h] for h, _ in chunks if h.startswith("needle")]
        fs = [scores[h] for h, _ in chunks if h.startswith("filler")]
        gap = min(ns) - max(fs)
        print(
            f"  {label:13s}: needles [{min(ns):.3f}, {max(ns):.3f}] "
            f"filler [{min(fs):.3f}, {max(fs):.3f}] gap={gap:+.3f}"
        )

    # Show structural fingerprints
    fp = StructuralFingerprinter()
    print()
    print("  Structural fingerprints:")
    for h, c in chunks:
        tokens = fp.extract_structural_tokens(c)
        words = len(c.split())
        density = len(tokens) / max(words, 1)
        print(f"    {h:15s}: {len(tokens):2d} tokens, {words:3d} words, density={density:.4f}")


def generate_chart(all_results: dict[str, list[dict]], output_path: Path) -> None:
    """Generate comparison bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes = ["structural", "goal_guided", "entity_aware", "auto_priority", "engine", "naive"]
    labels = [
        "Structural\n(fingerprint)", "TF-IDF\n(goal-guided)",
        "Entity-Aware\n(TF-IDF+entity)", "Keywords\n(auto)",
        "Hardcoded\n(priority=2)", "Naive\n(sliding)",
    ]
    colors = ["#8e44ad", "#9b59b6", "#1abc9c", "#3498db", "#2ecc71", "#e74c3c"]

    avgs = []
    for mode in modes:
        results = all_results.get(mode, [])
        scores = [r["survival_count"] for r in results]
        avgs.append(sum(scores) / len(scores) if scores else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(modes))
    bars = ax.bar(x, avgs, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Avg Needles in Context (out of 5)")
    ax.set_ylim(0, 5.5)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(
        "Adversarial NIAH: Structural Fingerprinting vs Other Scorers\n"
        "(10 sessions x 30 turns x 5 needles, 8k context window)",
        fontsize=11,
    )

    for bar, val in zip(bars, avgs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f}/5", ha="center", fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Discrimination analysis first
    discrimination_analysis()

    # Run benchmark
    all_results = run_benchmark()

    # Summary
    print_summary(all_results)

    # Save results
    output: dict[str, Any] = {
        "timestamp": timestamp,
        "benchmark": "niah_adversarial_structural",
        "description": "Adversarial NIAH with structural fingerprinting",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "num_sessions": 10,
        "num_turns": 30,
        "num_needles": len(NEEDLES),
        "results": {},
    }

    for mode, results in all_results.items():
        scores = [r["survival_count"] for r in results]
        output["results"][mode] = {
            "avg_survival": round(sum(scores) / len(scores), 2) if scores else 0,
            "per_session": scores,
            "sessions": results,
        }

    json_path = RESULTS_DIR / f"niah_adversarial_structural_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Chart
    png_path = RESULTS_DIR / f"niah_adversarial_structural_{timestamp}.png"
    try:
        generate_chart(all_results, png_path)
    except Exception as e:
        print(f"Chart error: {e}")


if __name__ == "__main__":
    main()
