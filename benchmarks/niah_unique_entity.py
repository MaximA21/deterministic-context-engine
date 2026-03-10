#!/usr/bin/env python3
"""Unique Entity Scorer Benchmark — pure set-difference approach.

Tests the hypothesis: needles contain entities (line numbers, IPs, employee IDs,
error codes) that appear NOWHERE else in context. Filler shares generic entities
(filenames, function names) across many chunks.

Scoring: count how many of a chunk's entities are corpus-unique (freq == 1).
No ML, no TF-IDF, no cosine similarity.

Runs against adversarial benchmark (hardest case) + all other variants.
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine import (
    ChunkLog,
    EntityExtractor,
    UniqueEntityScorer,
    GoalGuidedScorer,
    _estimate_tokens,
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 8000


def generate_needle_placements(num_sessions: int, num_turns: int, num_needles: int = 5) -> list[list[int]]:
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
    budget = max_tokens - last_tokens
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


def entity_discrimination_analysis(needles: list[dict], filler_gen, num_fillers: int = 25):
    """Show which entities are unique to needles vs shared with filler."""
    ext = EntityExtractor()

    needle_texts = [n["fact"] for n in needles]
    filler_texts = [filler_gen(i) for i in range(num_fillers)]

    # Extract entities
    needle_entities = [ext.extract_entities(t) for t in needle_texts]
    filler_entities = [ext.extract_entities(t) for t in filler_texts]

    # Global frequency
    all_entities: Counter[str] = Counter()
    for ents in needle_entities + filler_entities:
        for e in ents:
            all_entities[e] += 1

    print("  Entity Discrimination Analysis:")
    for i, (n, ents) in enumerate(zip(needles, needle_entities)):
        unique = {e for e in ents if all_entities[e] == 1}
        shared = ents - unique
        print(f"    Needle {i+1}: {len(ents)} entities, {len(unique)} unique, {len(shared)} shared")
        if unique:
            print(f"      Unique: {sorted(unique)[:8]}")
        if shared:
            print(f"      Shared: {sorted(shared)[:8]}")

    # Show filler entity overlap
    filler_unique_counts = []
    for ents in filler_entities:
        unique = sum(1 for e in ents if all_entities[e] == 1)
        filler_unique_counts.append(unique)
    avg_filler_unique = sum(filler_unique_counts) / len(filler_unique_counts) if filler_unique_counts else 0
    print(f"    Filler avg unique entities: {avg_filler_unique:.1f}")
    print()


def simulate_context_survival(
    needles: list[dict],
    generate_filler,
    num_sessions: int,
    num_turns: int,
    fillers_per_turn: int,
    mode: str,
    recall_question: str | None = None,
) -> dict[str, Any]:
    """Simulate context survival for a given scoring mode."""
    placements = generate_needle_placements(num_sessions, num_turns)
    session_results = []

    for sid, needle_turns in enumerate(placements):
        if mode == "unique_entity":
            log = ChunkLog(db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
                          soft_threshold=0.7, hard_threshold=0.9, scoring_mode="unique_entity")
        elif mode == "entity_aware":
            log = ChunkLog(db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
                          soft_threshold=0.7, hard_threshold=0.9, scoring_mode="entity_aware")
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
                priority = 2.0 if mode == "engine" else 0.5

                if log:
                    log.append("user", content, priority=priority)
                else:
                    raw_msgs.append({"role": "user", "content": content})
            else:
                for _ in range(fillers_per_turn):
                    fc = generate_filler(sid * 10000 + filler_idx)
                    filler_idx += 1
                    if log:
                        log.append("user", fc, priority=0.5)
                    else:
                        raw_msgs.append({"role": "user", "content": fc})
            if log:
                log.next_turn()

        if recall_question and log:
            log.append("user", recall_question, priority=2.0)

        if log:
            ctx = log.get_context()
            compaction = log.compaction_count
        else:
            if recall_question:
                raw_msgs.append({"role": "user", "content": recall_question})
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


def score_discrimination_test(needles: list[dict], filler_gen, num_fillers: int, label: str):
    """Show raw score distributions for unique_entity vs goal_guided vs entity_aware."""
    scorer_ue = UniqueEntityScorer()
    scorer_gg = GoalGuidedScorer()

    chunks = [(f"needle_{n['id']}", n["fact"]) for n in needles]
    chunks += [(f"filler_{i}", filler_gen(i)) for i in range(num_fillers)]

    # Unique entity scores (goal-independent)
    ue_scores = scorer_ue.score_chunks("", chunks)
    gg_scores = scorer_gg.score_chunks("recall question placeholder", chunks)

    for scorer_name, scores in [("UniqueEntity", ue_scores), ("TF-IDF", gg_scores)]:
        ns = [scores[h] for h, _ in chunks if h.startswith("needle")]
        fs = [scores[h] for h, _ in chunks if h.startswith("filler")]
        gap = min(ns) - max(fs)
        print(f"  {label:20s} {scorer_name:13s}: needles [{min(ns):.3f},{max(ns):.3f}] "
              f"filler [{min(fs):.3f},{max(fs):.3f}] gap={gap:+.3f}")


def main():
    from benchmarks.niah_adversarial import (
        NEEDLES as ADV_NEEDLES,
        ADVERSARIAL_FILLER as adv_fillers,
        RECALL_QUESTION as ADV_RECALL,
    )
    from benchmarks.niah_boilerplate import (
        NEEDLES as BP_NEEDLES,
        _generate_filler as bp_filler,
        RECALL_QUESTION as BP_RECALL,
    )
    from benchmarks.niah_goalguided import (
        NEEDLES as GG_NEEDLES,
        _generate_filler as gg_filler,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")

    print("=" * 70)
    print("UNIQUE ENTITY SCORER — Pure Set-Difference Benchmark")
    print("=" * 70)
    print()

    # --- Adversarial filler generator ---
    def adv_filler_gen(seed):
        rng = random.Random(seed)
        f1 = rng.choice(adv_fillers)
        f2 = rng.choice(adv_fillers)
        return f"[Turn {seed}] {f1}\n\n{f2}\n[Ref: {seed}_{rng.randint(0,999999)}]"

    # --- Entity discrimination analysis ---
    print("ADVERSARIAL ENTITY ANALYSIS:")
    entity_discrimination_analysis(ADV_NEEDLES, adv_filler_gen, num_fillers=25)

    print("BOILERPLATE ENTITY ANALYSIS:")
    entity_discrimination_analysis(BP_NEEDLES, bp_filler, num_fillers=25)

    # --- Score discrimination test ---
    print("SCORE DISCRIMINATION (needle min vs filler max):")
    score_discrimination_test(ADV_NEEDLES, adv_filler_gen, 25, "Adversarial")
    score_discrimination_test(BP_NEEDLES, bp_filler, 25, "Boilerplate")
    score_discrimination_test(GG_NEEDLES, gg_filler, 25, "Fair Goal-Guided")
    print()

    # --- Context survival simulations ---
    benchmarks = [
        {
            "name": "Adversarial",
            "needles": ADV_NEEDLES,
            "filler_fn": adv_filler_gen,
            "fillers_per_turn": 1,
            "recall": ADV_RECALL,
        },
        {
            "name": "Boilerplate",
            "needles": BP_NEEDLES,
            "filler_fn": bp_filler,
            "fillers_per_turn": 5,
            "recall": BP_RECALL,
        },
        {
            "name": "Fair Goal-Guided",
            "needles": GG_NEEDLES,
            "filler_fn": gg_filler,
            "fillers_per_turn": 4,
            "recall": None,
        },
    ]

    modes = ["unique_entity", "goal_guided", "entity_aware", "engine", "auto_priority", "naive"]
    mode_labels = {
        "unique_entity": "UniqueEntity",
        "goal_guided": "TF-IDF",
        "entity_aware": "EntityAware",
        "engine": "Hardcoded",
        "auto_priority": "Keywords",
        "naive": "Naive",
    }

    all_results: dict[str, dict[str, Any]] = {}

    for bench in benchmarks:
        print(f"\n{'='*70}")
        print(f"  {bench['name']} Benchmark (10 sessions × 30 turns × 5 needles)")
        print(f"{'='*70}")
        bench_results = {}

        for mode in modes:
            print(f"  {mode_labels[mode]:16s}...", end=" ", flush=True)
            t0 = time.time()
            result = simulate_context_survival(
                bench["needles"],
                bench["filler_fn"],
                num_sessions=10,
                num_turns=30,
                fillers_per_turn=bench["fillers_per_turn"],
                mode=mode,
                recall_question=bench.get("recall"),
            )
            elapsed = time.time() - t0
            bench_results[mode] = result
            lost_str = f"  lost={result['lost_counts']}" if result['lost_counts'] else ""
            print(f"avg={result['avg_survival']:.1f}/5  sessions={result['per_session']}  ({elapsed:.1f}s){lost_str}")

        all_results[bench["name"]] = bench_results

    # --- Master comparison table ---
    print(f"\n{'='*90}")
    print("MASTER COMPARISON TABLE — Context Survival (avg needles in context / 5)")
    print(f"{'='*90}")

    header = f"{'Benchmark':<20s}"
    for mode in modes:
        header += f" | {mode_labels[mode]:>12s}"
    print(header)
    print("-" * 90)

    for bench_name, bench_results in all_results.items():
        row = f"{bench_name:<20s}"
        for mode in modes:
            r = bench_results.get(mode, {})
            avg = r.get("avg_survival", 0)
            row += f" | {avg:>10.1f}/5"
        print(row)

    print("-" * 90)

    # --- Save results ---
    output = {
        "timestamp": timestamp,
        "benchmark": "niah_unique_entity",
        "description": "UniqueEntityScorer — pure set-difference (no ML) across all NIAH variants",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
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

    json_path = RESULTS_DIR / f"niah_unique_entity_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # --- Generate chart ---
    try:
        generate_chart(all_results, RESULTS_DIR / f"niah_unique_entity_{timestamp}.png", modes, mode_labels)
    except Exception as e:
        print(f"Chart error: {e}")


def generate_chart(all_results: dict, output_path: Path, modes: list[str], mode_labels: dict[str, str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    mode_colors = {
        "unique_entity": "#f39c12",
        "goal_guided": "#9b59b6",
        "entity_aware": "#1abc9c",
        "engine": "#2ecc71",
        "auto_priority": "#3498db",
        "naive": "#e74c3c",
    }

    bench_names = list(all_results.keys())
    n_benchmarks = len(bench_names)

    fig, axes = plt.subplots(1, n_benchmarks, figsize=(7 * n_benchmarks, 6), sharey=True)
    if n_benchmarks == 1:
        axes = [axes]

    for ax_idx, bench_name in enumerate(bench_names):
        ax = axes[ax_idx]
        bench_results = all_results[bench_name]
        x = np.arange(len(modes))
        avgs = [bench_results.get(m, {}).get("avg_survival", 0) for m in modes]
        colors = [mode_colors.get(m, "#95a5a6") for m in modes]
        bars = ax.bar(x, avgs, color=colors)
        ax.set_title(bench_name, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        labels = [mode_labels.get(m, m) for m in modes]
        ax.set_xticklabels(labels, fontsize=7, rotation=15)
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("Avg Needles Survived (out of 5)" if ax_idx == 0 else "")
        ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)
        for bar, val in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(
        "UniqueEntityScorer — Pure Set-Difference vs All Approaches\n"
        "(10 sessions x 30 turns x 5 needles, 8k context window)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


if __name__ == "__main__":
    main()
