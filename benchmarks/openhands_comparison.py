#!/usr/bin/env python3
"""OpenHands vs BM25+Structural — Adversarial NIAH Comparison.

Compares OpenHands amortized forgetting (positional: keep head + tail, drop middle)
against our content-aware scorers on the adversarial needle-in-a-haystack benchmark.

Modes tested:
- openhands: Positional amortized forgetting (keep first + recent, drop middle)
- structural: BM25 + structural density scoring
- bm25: BM25 goal relevance + corpus uniqueness
- naive: Simple sliding window (most recent only)

Reference: github.com/All-Hands-AI/OpenHands
  openhands/memory/condenser/impl/amortized_forgetting_condenser.py
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
from benchmarks.niah_adversarial import (
    NEEDLES, ADVERSARIAL_FILLER, SYSTEM_PROMPT, SYSTEM_PROMPT_TOKENS,
    RECALL_QUESTION, generate_needle_placements, sliding_window_truncate,
)
from benchmarks.gemini_utils import call_gemini, GEMINI_MODEL

RESULTS_DIR = PROJECT_ROOT / "results" / "openhands_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
FILLERS_PER_TURN = 14


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,
    api_key: str,
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

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            needle = NEEDLES[needle_idx]
            content = needle["fact"]
            priority = 0.5  # No hardcoded boost — let scorer decide
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

    needles_in_context = []
    context_text = " ".join(m["content"] for m in messages)
    for needle in NEEDLES:
        if needle["keyword"].lower() in context_text.lower():
            needles_in_context.append(needle["id"])

    result = call_gemini(messages, SYSTEM_PROMPT, api_key, max_output_tokens=512)

    if chunk_log:
        chunk_log.close()

    answer_lower = (result["answer"] or "").lower()
    needles_recalled = []
    needles_lost = []
    for needle in NEEDLES:
        if needle["keyword"].lower() in answer_lower:
            needles_recalled.append(needle["id"])
        else:
            needles_lost.append(needle["id"])

    return {
        "session_id": session_id, "mode": mode, "error": result["error"],
        "needle_turns": needle_turns,
        "needles_in_context": needles_in_context,
        "needles_recalled": needles_recalled, "needles_lost": needles_lost,
        "recall_score": len(needles_recalled), "total_needles": len(NEEDLES),
        "answer": result["answer"], "ttft": result["ttft"],
        "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
        "context_tokens": context_tokens, "total_tokens_added": total_tokens_added,
        "compaction_events": compaction_events, "compaction_log": compaction_log,
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

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Chart 1: Recall scores per session
    ax = axes[0]
    n_modes = len(modes_config)
    width = 0.8 / n_modes

    for idx, (label, mode_key, color) in enumerate(modes_config):
        mr = [r for r in results if r["mode"] == mode_key and not r["error"]]
        scores = [r["recall_score"] for r in mr]
        n_sessions = len(scores)
        x = np.arange(n_sessions)
        offset = (idx - n_modes / 2 + 0.5) * width
        ax.bar(x + offset, scores, width, label=label, color=color)

    ax.set_xlabel("Session")
    ax.set_ylabel("Needles Recalled (out of 5)")
    ax.set_title("Needle Recall per Session")
    ax.set_xticks(np.arange(n_sessions))
    ax.set_xticklabels([f"S{i+1}" for i in range(n_sessions)])
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=7, loc="lower right")
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)

    # Chart 2: Average recall comparison
    ax = axes[1]
    labels = []
    avgs = []
    colors = []
    for label, mode_key, color in modes_config:
        mr = [r for r in results if r["mode"] == mode_key and not r["error"]]
        scores = [r["recall_score"] for r in mr]
        avg = sum(scores) / len(scores) if scores else 0
        labels.append(label)
        avgs.append(avg)
        colors.append(color)

    bars = ax.bar(labels, avgs, color=colors)
    ax.set_ylabel("Avg Needles Recalled")
    ax.set_title("Average Recall Score")
    ax.set_ylim(0, 5.5)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}/5", ha="center", fontsize=11, fontweight="bold")

    # Chart 3: Needles in context survival
    ax = axes[2]
    needle_ids = [n["id"] for n in NEEDLES]
    n_needles = len(needle_ids)
    width_n = 0.8 / n_modes

    for idx, (label, mode_key, color) in enumerate(modes_config):
        mr = [r for r in results if r["mode"] == mode_key and not r["error"]]
        counts = {nid: 0 for nid in needle_ids}
        for r in mr:
            for nid in r.get("needles_in_context", []):
                counts[nid] += 1
        x = np.arange(n_needles)
        offset = (idx - n_modes / 2 + 0.5) * width_n
        ax.bar(x + offset, [counts[nid] for nid in needle_ids], width_n,
               label=label, color=color)

    n_sessions_total = len([r for r in results if r["mode"] == modes_config[0][1] and not r["error"]])
    ax.set_xlabel("Needle")
    ax.set_ylabel(f"Times in Context (out of {n_sessions_total})")
    ax.set_title("Needle Survival in Context")
    ax.set_xticks(np.arange(n_needles))
    ax.set_xticklabels([f"N{i+1}" for i in range(n_needles)])
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(
        "OpenHands vs Deterministic Context Engine — Adversarial NIAH\n"
        f"(30 turns, 5 needles, {FILLERS_PER_TURN} adversarial fillers/turn, "
        f"{MAX_CONTEXT_TOKENS // 1024}k context window)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    num_sessions = 10
    num_turns = 30
    placements = generate_needle_placements(num_sessions, num_turns)
    modes = ["structural", "bm25", "openhands", "naive"]

    print("=" * 70)
    print(f"OpenHands vs DCE — Adversarial NIAH ({MAX_CONTEXT_TOKENS // 1024}k context)")
    print("=" * 70)
    print(f"Model: {GEMINI_MODEL}")
    print(f"Fillers per turn: {FILLERS_PER_TURN}")
    print(f"Modes: {modes}")
    print()
    print("OpenHands strategy: Amortized forgetting (keep first + tail, drop middle)")
    print("DCE strategy: Content-aware scoring (BM25 goal relevance + structural density)")
    print()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results: list[dict] = []
    total_tests = num_sessions * len(modes)
    test_num = 0

    for session_id, needle_turns in enumerate(placements):
        for mode in modes:
            test_num += 1
            print(f"[{test_num}/{total_tests}] S{session_id+1} {mode.upper()}...", end=" ", flush=True)
            result = run_session(session_id, needle_turns, mode, api_key, num_turns)
            results.append(result)
            if result["error"]:
                print(f"ERR: {result['error'][:80]}")
            else:
                ratio = result["total_tokens_added"] / max(1, result["context_tokens"])
                in_ctx = len(result["needles_in_context"])
                print(f"Recalled {result['recall_score']}/5 in_ctx={in_ctx}/5 "
                      f"ctx={result['context_tokens']} ratio={ratio:.2f}x "
                      f"compact={result['compaction_events']}")
            time.sleep(2)

    # Save raw results
    output = {
        "timestamp": timestamp, "model": GEMINI_MODEL,
        "benchmark": "openhands_comparison_adversarial",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "fillers_per_turn": FILLERS_PER_TURN,
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "description": (
            "Comparison of OpenHands amortized forgetting (positional: keep head + tail) "
            "vs DCE content-aware scoring (BM25 + structural density) on adversarial NIAH. "
            "Reference: github.com/All-Hands-AI/OpenHands"
        ),
        "modes": {
            "openhands": "Positional amortized forgetting — keep first chunk + most recent, drop middle. No content analysis.",
            "structural": "BM25 goal relevance + structural action density boost + frequency demotion.",
            "bm25": "BM25 goal relevance (40%) + corpus uniqueness via Jaccard (60%).",
            "naive": "Simple sliding window — keep most recent messages that fit.",
        },
        "results": results,
    }
    json_path = RESULTS_DIR / f"adversarial_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Generate chart
    png_path = RESULTS_DIR / f"adversarial_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY — OpenHands vs Deterministic Context Engine")
    print(f"{'=' * 70}")

    mode_labels = [
        ("BM25+STRUCTURAL (content-aware)", "structural"),
        ("BM25 (content-aware)", "bm25"),
        ("OPENHANDS (positional, amortized forgetting)", "openhands"),
        ("NAIVE (sliding window)", "naive"),
    ]

    summary_rows = []
    for label, mode_key in mode_labels:
        mr = [r for r in results if r["mode"] == mode_key and not r["error"]]
        scores = [r["recall_score"] for r in mr]
        avg = sum(scores) / len(scores) if scores else 0
        avg_in_ctx = sum(len(r["needles_in_context"]) for r in mr) / len(mr) if mr else 0
        avg_compact = sum(r["compaction_events"] for r in mr) / len(mr) if mr else 0
        avg_ctx = sum(r["context_tokens"] for r in mr) / len(mr) if mr else 0
        avg_added = sum(r["total_tokens_added"] for r in mr) / len(mr) if mr else 0
        avg_ratio = avg_added / avg_ctx if avg_ctx > 0 else 0

        summary_rows.append({
            "label": label, "mode": mode_key,
            "avg_recall": avg, "scores": scores,
            "avg_in_ctx": avg_in_ctx, "avg_compact": avg_compact,
            "avg_ctx": avg_ctx, "avg_ratio": avg_ratio,
        })

        print(f"\n{label}:")
        print(f"  Avg recall: {avg:.1f}/5 ({100*avg/5:.0f}%)")
        print(f"  Scores: {scores}")
        print(f"  Avg needles in context: {avg_in_ctx:.1f}/5")
        print(f"  Avg context at recall: {avg_ctx:.0f} tokens")
        print(f"  Compression ratio: {avg_ratio:.2f}x")
        print(f"  Avg compactions: {avg_compact:.1f}")

        lost_counts: dict[str, int] = {}
        for r in mr:
            for nid in r.get("needles_lost", []):
                lost_counts[nid] = lost_counts.get(nid, 0) + 1
        if lost_counts:
            print(f"  Needles lost: {dict(sorted(lost_counts.items(), key=lambda x: -x[1]))}")

    # Head-to-head
    oh = next((s for s in summary_rows if s["mode"] == "openhands"), None)
    bs = next((s for s in summary_rows if s["mode"] == "structural"), None)
    if oh and bs:
        print(f"\n{'=' * 70}")
        print("HEAD-TO-HEAD: BM25+Structural vs OpenHands")
        print(f"{'=' * 70}")
        delta = bs["avg_recall"] - oh["avg_recall"]
        pct = (delta / max(oh["avg_recall"], 0.01)) * 100
        if delta > 0:
            print(f"  BM25+Structural wins by {delta:.1f} needles ({pct:+.0f}%)")
        elif delta < 0:
            print(f"  OpenHands wins by {-delta:.1f} needles ({-pct:+.0f}%)")
        else:
            print(f"  TIE at {bs['avg_recall']:.1f}/5")
        print(f"  BM25+Structural: {bs['avg_recall']:.1f}/5 recall, {bs['avg_in_ctx']:.1f}/5 in context")
        print(f"  OpenHands:       {oh['avg_recall']:.1f}/5 recall, {oh['avg_in_ctx']:.1f}/5 in context")

    # Save summary
    summary_path = RESULTS_DIR / f"summary_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write("OpenHands vs Deterministic Context Engine — Adversarial NIAH\n")
        f.write(f"Model: {GEMINI_MODEL}\n")
        f.write(f"Context: {MAX_CONTEXT_TOKENS // 1024}k tokens\n")
        f.write(f"Fillers/turn: {FILLERS_PER_TURN}\n\n")
        for row in summary_rows:
            f.write(f"{row['label']}:\n")
            f.write(f"  Avg recall: {row['avg_recall']:.1f}/5 ({100*row['avg_recall']/5:.0f}%)\n")
            f.write(f"  Scores: {row['scores']}\n")
            f.write(f"  Avg in context: {row['avg_in_ctx']:.1f}/5\n")
            f.write(f"  Compression: {row['avg_ratio']:.2f}x\n\n")
        if oh and bs:
            delta = bs["avg_recall"] - oh["avg_recall"]
            f.write(f"Delta (BM25+Structural - OpenHands): {delta:+.1f} needles\n")
    print(f"\nSummary saved to {summary_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
