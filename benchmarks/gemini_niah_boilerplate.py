#!/usr/bin/env python3
"""Boilerplate NIAH with Gemini Flash — test if smarter model fixes 3.5/5 score.

Same boilerplate scenario as niah_boilerplate.py but using Gemini 2.5 Flash
with 32k context budget. The hypothesis: a smarter model + engine can overcome
the TF-IDF uniqueness penalty on repetitive critical content.
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
from benchmarks.niah_boilerplate import (
    NEEDLES, SYSTEM_PROMPT, SYSTEM_PROMPT_TOKENS, RECALL_QUESTION,
    _generate_filler, generate_needle_placements, sliding_window_truncate,
)
from benchmarks.gemini_utils import call_gemini

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
MODEL = "gemini-3.1-flash-lite-preview"


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,  # "goal_guided", "auto_priority", "engine", or "naive"
    api_key: str,
    num_turns: int = 30,
    fillers_per_turn: int = 8,  # More fillers to fill 32k
) -> dict[str, Any]:
    rng = random.Random(session_id * 1000 + hash(mode))
    compaction_log: list[dict] = []
    total_tokens_added = 0
    filler_seed_offset = session_id * 10000

    if mode == "goal_guided":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=False, goal_guided=True,
        )
    elif mode == "auto_priority":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=True, goal_guided=False,
        )
    elif mode == "engine":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
        )
    else:
        chunk_log = None
        raw_messages: list[dict[str, str]] = []

    filler_idx = 0
    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            needle = NEEDLES[needle_idx]
            content = needle["fact"]

            if mode in ("goal_guided", "auto_priority"):
                priority = 0.5
            elif mode == "engine":
                priority = 2.0
            else:
                priority = 0.5

            total_tokens_added += _estimate_tokens(content)
            if mode in ("goal_guided", "auto_priority", "engine"):
                tokens_before = chunk_log.current_tokens()
                compactions_before = chunk_log.compaction_count
                chunk_log.append("user", content, priority=priority)
            else:
                raw_messages.append({"role": "user", "content": content})
        else:
            # Generate multiple boilerplate fillers per turn
            filler_parts = []
            for _ in range(fillers_per_turn):
                filler_parts.append(_generate_filler(filler_seed_offset + filler_idx))
                filler_idx += 1
            unique_salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}]\n\n" + "\n\n---\n\n".join(filler_parts) + unique_salt
            priority = 0.5
            total_tokens_added += _estimate_tokens(content)

            if mode in ("goal_guided", "auto_priority", "engine"):
                tokens_before = chunk_log.current_tokens()
                compactions_before = chunk_log.compaction_count
                chunk_log.append("user", content, priority=priority)
            else:
                raw_messages.append({"role": "user", "content": content})

        if mode in ("goal_guided", "auto_priority", "engine"):
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

    # Add recall question
    if mode in ("goal_guided", "auto_priority", "engine"):
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

    result = call_gemini(messages, SYSTEM_PROMPT, api_key, model=MODEL, max_output_tokens=1024)

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
        "session_id": session_id,
        "mode": mode,
        "error": result["error"],
        "needle_turns": needle_turns,
        "needles_in_context": needles_in_context,
        "needles_recalled": needles_recalled,
        "needles_lost": needles_lost,
        "recall_score": len(needles_recalled),
        "total_needles": len(NEEDLES),
        "answer": result["answer"],
        "ttft": result["ttft"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
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

    modes = ["goal_guided", "auto_priority", "engine", "naive"]
    labels = ["Goal-Guided\n(TF-IDF)", "Keywords\n(AutoPriority)", "Hardcoded\n(priority=2)", "Naive\n(sliding window)"]
    colors = ["#9b59b6", "#3498db", "#2ecc71", "#e74c3c"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    avgs = []
    for m in modes:
        scores = [r["recall_score"] for r in results if r["mode"] == m]
        avgs.append(sum(scores) / len(scores) if scores else 0)
    bars = ax.bar(labels, avgs, color=colors)
    ax.set_ylabel("Avg Needles Recalled")
    ax.set_title("Average Recall Score")
    ax.set_ylim(0, 5.5)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}/5", ha="center", fontsize=12)

    ax = axes[1]
    width = 0.2
    for i, (m, label, color) in enumerate(zip(modes, labels, colors)):
        scores = [r["recall_score"] for r in results if r["mode"] == m]
        x = np.arange(len(scores))
        ax.bar(x + i * width, scores, width, label=label.replace("\n", " "), color=color)
    ax.set_xlabel("Session")
    ax.set_ylabel("Needles Recalled (out of 5)")
    ax.set_title("Recall per Session")
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=7)

    fig.suptitle(
        f"Boilerplate NIAH — Gemini Flash ({MAX_CONTEXT_TOKENS // 1024}k context)\n"
        "Repetitive critical content with similar-looking filler",
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

    modes = ["goal_guided", "auto_priority", "engine", "naive"]

    print("=" * 60)
    print(f"Boilerplate NIAH — Gemini Flash ({MAX_CONTEXT_TOKENS // 1024}k context)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Sessions: {num_sessions}")
    print(f"Modes: {modes}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Hypothesis: smarter model + engine fixes the 3.5/5 TF-IDF score")
    print()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    results: list[dict] = []
    total_tests = num_sessions * len(modes)
    test_num = 0

    for session_id, needle_turns in enumerate(placements):
        for mode in modes:
            test_num += 1
            label = f"[{test_num}/{total_tests}] Session {session_id+1} {mode.upper()}"
            print(f"{label} (needles at turns {needle_turns})...", end=" ", flush=True)

            result = run_session(session_id, needle_turns, mode, api_key, num_turns)
            results.append(result)

            if result["error"]:
                print(f"ERR: {result['error'][:80]}")
            else:
                recalled = result["recall_score"]
                in_ctx = len(result.get("needles_in_context", []))
                lost = result.get("needles_lost", [])
                compact = result["compaction_events"]
                print(f"Recalled {recalled}/5 (in_context={in_ctx}/5, lost={lost}) ttft={result['ttft']:.2f}s compact={compact}")

            time.sleep(2)

    # Save results
    output = {
        "timestamp": timestamp,
        "model": MODEL,
        "benchmark": "gemini_niah_boilerplate",
        "description": f"Boilerplate NIAH 4-mode — Gemini Flash {MAX_CONTEXT_TOKENS // 1024}k",
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "modes": modes,
        "results": results,
    }

    json_path = RESULTS_DIR / f"gemini_boilerplate_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    png_path = RESULTS_DIR / f"gemini_boilerplate_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for label, mode_key in [
        ("GOAL-GUIDED (TF-IDF)", "goal_guided"),
        ("KEYWORDS (AutoPriority)", "auto_priority"),
        ("HARDCODED (priority=2.0)", "engine"),
        ("NAIVE (sliding window)", "naive"),
    ]:
        mode_results = [r for r in results if r["mode"] == mode_key]
        ok_results = [r for r in mode_results if not r["error"]]
        scores = [r["recall_score"] for r in ok_results]
        avg = sum(scores) / len(scores) if scores else 0
        errors = len(mode_results) - len(ok_results)
        print(f"\n{label}:")
        print(f"  Avg recall: {avg:.1f}/5 ({100*avg/5:.0f}%)")
        print(f"  Scores: {[r['recall_score'] for r in mode_results]}")
        print(f"  API errors: {errors}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
