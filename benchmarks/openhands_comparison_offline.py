#!/usr/bin/env python3
"""OpenHands vs BM25+Structural — Offline Adversarial NIAH Comparison.

Measures deterministic context quality metrics WITHOUT API calls:
- Needle survival in context (did the scorer retain the needle?)
- Compression ratio (total tokens added / context tokens at recall)
- Compaction events

These metrics directly measure scorer quality. The LLM recall score
(which requires API) is a downstream effect of needle survival.

Modes tested:
- openhands: Positional amortized forgetting (keep first + recent, drop middle)
- structural: BM25 + structural density scoring
- bm25: BM25 goal relevance + corpus uniqueness
- naive: Simple sliding window (most recent only)
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
from benchmarks.niah_adversarial import (
    NEEDLES, ADVERSARIAL_FILLER, SYSTEM_PROMPT_TOKENS,
    RECALL_QUESTION, generate_needle_placements, sliding_window_truncate,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "openhands_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
FILLERS_PER_TURN = 14


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,
    num_turns: int = 30,
) -> dict[str, Any]:
    rng = random.Random(session_id * 1000 + hash(mode))
    compaction_log: list[dict] = []
    total_tokens_added = 0

    if mode in ("openhands", "structural", "bm25"):
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=False, scoring_mode=mode,
        )
    else:
        chunk_log = None
        raw_messages: list[dict[str, str]] = []

    t0 = time.monotonic()

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            needle = NEEDLES[needle_idx]
            content = needle["fact"]
            priority = 0.5
        else:
            fillers = [rng.choice(ADVERSARIAL_FILLER) for _ in range(FILLERS_PER_TURN)]
            unique_salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] " + "\n\n".join(fillers) + unique_salt
            priority = 0.5

        total_tokens_added += _estimate_tokens(content)

        if chunk_log is not None:
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
    if chunk_log is not None:
        chunk_log.append("user", RECALL_QUESTION, priority=2.0)
        messages = chunk_log.get_context()
        context_tokens = chunk_log.get_context_tokens()
        compaction_events = chunk_log.compaction_count
    else:
        raw_messages.append({"role": "user", "content": RECALL_QUESTION})
        messages = sliding_window_truncate(raw_messages, MAX_CONTEXT_TOKENS)
        context_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
        compaction_events = 0

    elapsed_ms = (time.monotonic() - t0) * 1000

    # Check needle survival
    context_text = " ".join(m["content"] for m in messages)
    needles_in_context = []
    needles_lost = []
    for needle in NEEDLES:
        if needle["keyword"].lower() in context_text.lower():
            needles_in_context.append(needle["id"])
        else:
            needles_lost.append(needle["id"])

    if chunk_log:
        chunk_log.close()

    return {
        "session_id": session_id, "mode": mode,
        "needle_turns": needle_turns,
        "needles_in_context": needles_in_context,
        "needles_lost": needles_lost,
        "survival_score": len(needles_in_context),
        "total_needles": len(NEEDLES),
        "context_tokens": context_tokens,
        "total_tokens_added": total_tokens_added,
        "compression_ratio": total_tokens_added / max(1, context_tokens),
        "compaction_events": compaction_events,
        "compaction_log": compaction_log,
        "messages_in_context": len(messages),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def generate_chart(results: list[dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes_config = [
        ("BM25+Structural", "structural", "#2ecc71"),
        ("BM25", "bm25", "#3498db"),
        ("OpenHands\n(amortized forgetting)", "openhands", "#e67e22"),
        ("Naive\n(sliding window)", "naive", "#e74c3c"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Needle survival per session
    ax = axes[0][0]
    n_modes = len(modes_config)
    width = 0.8 / n_modes
    n_sessions = len(set(r["session_id"] for r in results))

    for idx, (label, mode_key, color) in enumerate(modes_config):
        mr = sorted([r for r in results if r["mode"] == mode_key], key=lambda r: r["session_id"])
        scores = [r["survival_score"] for r in mr]
        x = np.arange(len(scores))
        offset = (idx - n_modes / 2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=label, color=color)

    ax.set_xlabel("Session")
    ax.set_ylabel("Needles in Context (out of 5)")
    ax.set_title("Needle Survival per Session")
    ax.set_xticks(np.arange(n_sessions))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_sessions)])
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=7, loc="lower right")
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)

    # Chart 2: Average survival comparison
    ax = axes[0][1]
    labels = []
    avgs = []
    colors = []
    for label, mode_key, color in modes_config:
        mr = [r for r in results if r["mode"] == mode_key]
        scores = [r["survival_score"] for r in mr]
        avg = sum(scores) / len(scores) if scores else 0
        labels.append(label)
        avgs.append(avg)
        colors.append(color)

    bars = ax.bar(labels, avgs, color=colors)
    ax.set_ylabel("Avg Needles in Context")
    ax.set_title("Average Needle Survival")
    ax.set_ylim(0, 5.5)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}/5", ha="center", fontsize=11, fontweight="bold")

    # Chart 3: Per-needle survival rates
    ax = axes[1][0]
    needle_ids = [n["id"] for n in NEEDLES]
    n_needles = len(needle_ids)
    width_n = 0.8 / n_modes

    for idx, (label, mode_key, color) in enumerate(modes_config):
        mr = [r for r in results if r["mode"] == mode_key]
        total = len(mr) if mr else 1
        counts = {nid: 0 for nid in needle_ids}
        for r in mr:
            for nid in r.get("needles_in_context", []):
                counts[nid] += 1
        x = np.arange(n_needles)
        offset = (idx - n_modes / 2 + 0.5) * width_n
        rates = [100 * counts[nid] / total for nid in needle_ids]
        ax.bar(x + offset, rates, width_n, label=label, color=color)

    ax.set_xlabel("Needle")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_title("Per-Needle Survival Rate")
    ax.set_xticks(np.arange(n_needles))
    ax.set_xticklabels([f"N{i+1}" for i in range(n_needles)])
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    # Chart 4: Compression & performance
    ax = axes[1][1]
    mode_names = []
    avg_ratios = []
    avg_times = []
    bar_colors = []
    for label, mode_key, color in modes_config:
        mr = [r for r in results if r["mode"] == mode_key]
        avg_ratio = sum(r["compression_ratio"] for r in mr) / len(mr) if mr else 0
        avg_time = sum(r["elapsed_ms"] for r in mr) / len(mr) if mr else 0
        mode_names.append(label)
        avg_ratios.append(avg_ratio)
        avg_times.append(avg_time)
        bar_colors.append(color)

    x = np.arange(len(mode_names))
    bars = ax.bar(x, avg_ratios, 0.6, color=bar_colors)
    ax.set_ylabel("Compression Ratio (higher = more compression)")
    ax.set_title("Compression Ratio & Processing Time")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, fontsize=8)
    for bar, ratio, t in zip(bars, avg_ratios, avg_times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{ratio:.1f}x\n{t:.0f}ms", ha="center", fontsize=9)

    fig.suptitle(
        "OpenHands vs Deterministic Context Engine — Adversarial NIAH (Offline)\n"
        f"(30 turns, 5 needles, {FILLERS_PER_TURN} adversarial fillers/turn, "
        f"{MAX_CONTEXT_TOKENS // 1024}k context window, no API calls)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


def main():
    num_sessions = 10
    num_turns = 30
    placements = generate_needle_placements(num_sessions, num_turns)
    modes = ["structural", "bm25", "openhands", "naive"]

    print("=" * 70)
    print(f"OpenHands vs DCE — Offline Adversarial NIAH ({MAX_CONTEXT_TOKENS // 1024}k)")
    print("=" * 70)
    print(f"Fillers per turn: {FILLERS_PER_TURN}")
    print(f"Modes: {modes}")
    print(f"Sessions: {num_sessions}, Turns: {num_turns}")
    print()
    print("OpenHands: Amortized forgetting (keep first + tail, drop middle)")
    print("DCE:       Content-aware scoring (BM25 goal relevance + structural density)")
    print()
    print("Needle placements (turn numbers):")
    for i, p in enumerate(placements):
        print(f"  Session {i+1}: turns {p}")
    print("-" * 70)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results: list[dict] = []
    total_tests = num_sessions * len(modes)
    test_num = 0

    for session_id, needle_turns in enumerate(placements):
        for mode in modes:
            test_num += 1
            print(f"[{test_num}/{total_tests}] S{session_id+1} {mode.upper():<18s}", end=" ", flush=True)
            result = run_session(session_id, needle_turns, mode, num_turns)
            results.append(result)
            in_ctx = result["survival_score"]
            ratio = result["compression_ratio"]
            lost = result["needles_lost"]
            t = result["elapsed_ms"]
            print(f"survived={in_ctx}/5  ratio={ratio:.2f}x  compact={result['compaction_events']}  "
                  f"msgs={result['messages_in_context']}  {t:.0f}ms"
                  + (f"  LOST: {lost}" if lost else ""))

    # Save raw results
    output = {
        "timestamp": timestamp,
        "benchmark": "openhands_comparison_adversarial_offline",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "fillers_per_turn": FILLERS_PER_TURN,
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "description": (
            "Offline comparison of OpenHands amortized forgetting vs DCE content-aware scoring. "
            "Measures needle survival in context (deterministic) without LLM API calls."
        ),
        "modes": {
            "openhands": "Positional amortized forgetting — keep first chunk + most recent, drop middle. No content analysis.",
            "structural": "BM25 goal relevance + structural action density boost + frequency demotion.",
            "bm25": "BM25 goal relevance (40%) + corpus uniqueness via Jaccard (60%).",
            "naive": "Simple sliding window — keep most recent messages that fit.",
        },
        "results": results,
    }
    json_path = RESULTS_DIR / f"offline_adversarial_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Generate chart
    png_path = RESULTS_DIR / f"offline_adversarial_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY — OpenHands vs Deterministic Context Engine (Offline)")
    print(f"{'=' * 70}")

    mode_labels = [
        ("BM25+STRUCTURAL (content-aware)", "structural"),
        ("BM25 (content-aware)", "bm25"),
        ("OPENHANDS (positional, amortized forgetting)", "openhands"),
        ("NAIVE (sliding window)", "naive"),
    ]

    summary_rows = []
    for label, mode_key in mode_labels:
        mr = [r for r in results if r["mode"] == mode_key]
        scores = [r["survival_score"] for r in mr]
        avg = sum(scores) / len(scores) if scores else 0
        avg_ctx = sum(r["context_tokens"] for r in mr) / len(mr) if mr else 0
        avg_added = sum(r["total_tokens_added"] for r in mr) / len(mr) if mr else 0
        avg_ratio = sum(r["compression_ratio"] for r in mr) / len(mr) if mr else 0
        avg_compact = sum(r["compaction_events"] for r in mr) / len(mr) if mr else 0
        avg_msgs = sum(r["messages_in_context"] for r in mr) / len(mr) if mr else 0
        avg_time = sum(r["elapsed_ms"] for r in mr) / len(mr) if mr else 0
        perfect = sum(1 for s in scores if s == 5)

        summary_rows.append({
            "label": label, "mode": mode_key,
            "avg_survival": avg, "scores": scores,
            "avg_ctx": avg_ctx, "avg_ratio": avg_ratio,
            "avg_compact": avg_compact, "avg_msgs": avg_msgs,
            "avg_time": avg_time, "perfect": perfect,
        })

        print(f"\n{label}:")
        print(f"  Avg needle survival: {avg:.2f}/5 ({100*avg/5:.0f}%)")
        print(f"  Perfect sessions (5/5): {perfect}/{len(scores)}")
        print(f"  Scores: {scores}")
        print(f"  Avg context tokens: {avg_ctx:.0f} / {MAX_CONTEXT_TOKENS}")
        print(f"  Avg compression: {avg_ratio:.2f}x")
        print(f"  Avg compactions: {avg_compact:.1f}")
        print(f"  Avg messages in context: {avg_msgs:.1f}")
        print(f"  Avg processing time: {avg_time:.1f}ms")

        lost_counts: dict[str, int] = {}
        for r in mr:
            for nid in r.get("needles_lost", []):
                lost_counts[nid] = lost_counts.get(nid, 0) + 1
        if lost_counts:
            print(f"  Needles lost: {dict(sorted(lost_counts.items(), key=lambda x: -x[1]))}")

    # Head-to-head
    oh = next((s for s in summary_rows if s["mode"] == "openhands"), None)
    bs = next((s for s in summary_rows if s["mode"] == "structural"), None)
    bm = next((s for s in summary_rows if s["mode"] == "bm25"), None)
    na = next((s for s in summary_rows if s["mode"] == "naive"), None)

    print(f"\n{'=' * 70}")
    print("HEAD-TO-HEAD COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Mode':<45s} {'Survival':>10s} {'Perfect':>10s} {'Compress':>10s}")
    print("-" * 75)
    for row in summary_rows:
        print(f"{row['label']:<45s} {row['avg_survival']:>7.2f}/5  {row['perfect']:>5d}/10  {row['avg_ratio']:>8.2f}x")

    if oh and bs:
        delta = bs["avg_survival"] - oh["avg_survival"]
        print(f"\nBM25+Structural vs OpenHands delta: {delta:+.2f} needles/session")
        if delta > 0:
            print(f"  -> Content-aware scoring retains {delta:.2f} MORE needles on average")
        elif delta < 0:
            print(f"  -> Positional forgetting retains {-delta:.2f} MORE needles on average")
        else:
            print(f"  -> TIE")

    # Why content-aware beats positional
    if oh and bs and bs["avg_survival"] > oh["avg_survival"]:
        print(f"\nWHY CONTENT-AWARE WINS:")
        print(f"  OpenHands drops middle events regardless of content.")
        print(f"  Needles placed in early-to-mid turns get evicted.")
        print(f"  BM25+Structural identifies needles by structural density")
        print(f"  (line numbers, action markers, specific values) and protects them.")

    # Save summary
    summary_path = RESULTS_DIR / f"offline_summary_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write("OpenHands vs DCE — Offline Adversarial NIAH\n")
        f.write(f"Context: {MAX_CONTEXT_TOKENS // 1024}k tokens\n")
        f.write(f"Fillers/turn: {FILLERS_PER_TURN}\n")
        f.write(f"Sessions: {num_sessions}, Turns: {num_turns}\n\n")
        f.write(f"{'Mode':<45s} {'Survival':>10s} {'Perfect':>10s} {'Compress':>10s}\n")
        f.write("-" * 75 + "\n")
        for row in summary_rows:
            f.write(f"{row['label']:<45s} {row['avg_survival']:>7.2f}/5  {row['perfect']:>5d}/10  {row['avg_ratio']:>8.2f}x\n")
        if oh and bs:
            delta = bs["avg_survival"] - oh["avg_survival"]
            f.write(f"\nBM25+Structural vs OpenHands delta: {delta:+.2f} needles/session\n")
    print(f"\nSummary saved to {summary_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
