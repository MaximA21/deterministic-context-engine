#!/usr/bin/env python3
"""50-turn extended session with Gemini Flash — stress test compaction over time.

Longer session to stress-test the context engine with Gemini 2.5 Flash.
Measures recall at turn 25 (mid-session) AND turn 50 (end).
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
from benchmarks.niah_dense import (
    NEEDLES, FILLER_TEMPLATES, SYSTEM_PROMPT, SYSTEM_PROMPT_TOKENS,
    sliding_window_truncate,
)
from benchmarks.gemini_utils import call_gemini

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
MODEL = "gemini-3.1-flash-lite-preview"
NUM_TURNS = 50

RECALL_QUESTION = """Please recall ALL important updates and facts mentioned during our conversation.
Specifically, I need you to tell me:
1. What is the secret project codename?
2. When is the database migration scheduled?
3. What was the API rate limit changed to?
4. What is the new hire's name and when does she start?
5. What is the production server IP address?

Answer each question based ONLY on what was mentioned in our conversation."""


def generate_needle_placements_50(num_sessions: int) -> list[list[int]]:
    """Generate needle placements spread across 50 turns."""
    placements = []
    rng = random.Random(42)
    for _ in range(num_sessions):
        # Spread needles across the full 50-turn range
        turns = sorted(rng.sample(range(NUM_TURNS), len(NEEDLES)))
        placements.append(turns)
    return placements


def _do_recall(
    messages: list[dict[str, str]],
    api_key: str,
) -> dict:
    """Call Gemini with messages and score needle recall."""
    result = call_gemini(messages, SYSTEM_PROMPT, api_key, model=MODEL, max_output_tokens=512)
    answer_lower = (result["answer"] or "").lower()
    needles_recalled = []
    needles_lost = []
    for needle in NEEDLES:
        if needle["keyword"].lower() in answer_lower:
            needles_recalled.append(needle["id"])
        else:
            needles_lost.append(needle["id"])
    return {
        **result,
        "needles_recalled": needles_recalled,
        "needles_lost": needles_lost,
        "recall_score": len(needles_recalled),
    }


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,
    api_key: str,
) -> dict[str, Any]:
    rng = random.Random(session_id * 1000 + hash(mode))
    compaction_log: list[dict] = []
    total_tokens_added = 0

    if mode == "engine":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
        )
    else:
        chunk_log = None
        raw_messages: list[dict[str, str]] = []

    mid_recall = None
    MID_TURN = 25

    for turn in range(NUM_TURNS):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            needle = NEEDLES[needle_idx]
            content = f"IMPORTANT UPDATE: {needle['fact']}"
            priority = 2.0
        else:
            # 3 fillers per turn (enough for 50 turns to stress 32k)
            fillers = [rng.choice(FILLER_TEMPLATES) for _ in range(3)]
            unique_salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] " + "\n\n".join(fillers) + unique_salt
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

        # Mid-session recall at turn 25
        if turn == MID_TURN - 1:
            if mode == "engine":
                chunk_log.append("user", RECALL_QUESTION, priority=2.0)
                mid_messages = chunk_log.get_context()
            else:
                mid_messages_raw = list(raw_messages) + [{"role": "user", "content": RECALL_QUESTION}]
                mid_messages = sliding_window_truncate(mid_messages_raw, MAX_CONTEXT_TOKENS)

            mid_recall = _do_recall(mid_messages, api_key)

            # Remove the recall question from context (don't pollute rest of session)
            if mode == "engine":
                # Delete the recall question chunk
                recall_hash = chunk_log._conn.execute(
                    "SELECT chunk_hash FROM chunks ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if recall_hash:
                    chunk_log._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (recall_hash[0],))
                    chunk_log._conn.commit()

            time.sleep(2)

    # Final recall at turn 50
    if mode == "engine":
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

    final_recall = _do_recall(messages, api_key)

    if chunk_log:
        chunk_log.close()

    return {
        "session_id": session_id,
        "mode": mode,
        "num_turns": NUM_TURNS,
        "needle_turns": needle_turns,
        "needles_in_context": needles_in_context,
        "mid_recall_turn25": {
            "recall_score": mid_recall["recall_score"] if mid_recall else 0,
            "needles_recalled": mid_recall["needles_recalled"] if mid_recall else [],
            "needles_lost": mid_recall["needles_lost"] if mid_recall else [],
            "ttft": mid_recall["ttft"] if mid_recall else 0,
            "error": mid_recall["error"] if mid_recall else None,
        },
        "final_recall_turn50": {
            "recall_score": final_recall["recall_score"],
            "needles_recalled": final_recall["needles_recalled"],
            "needles_lost": final_recall["needles_lost"],
            "ttft": final_recall["ttft"],
            "error": final_recall["error"],
            "answer": final_recall["answer"],
        },
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    engine_results = [r for r in results if r["mode"] == "engine"]
    naive_results = [r for r in results if r["mode"] == "naive"]

    # Chart 1: Mid vs Final recall comparison
    ax = axes[0]
    categories = ["Engine\nTurn 25", "Engine\nTurn 50", "Naive\nTurn 25", "Naive\nTurn 50"]
    avgs = [
        np.mean([r["mid_recall_turn25"]["recall_score"] for r in engine_results]),
        np.mean([r["final_recall_turn50"]["recall_score"] for r in engine_results]),
        np.mean([r["mid_recall_turn25"]["recall_score"] for r in naive_results]),
        np.mean([r["final_recall_turn50"]["recall_score"] for r in naive_results]),
    ]
    colors = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]
    bars = ax.bar(categories, avgs, color=colors)
    ax.set_ylabel("Avg Needles Recalled")
    ax.set_title("Recall at Turn 25 vs Turn 50")
    ax.set_ylim(0, 5.5)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}/5", ha="center", fontsize=12)

    # Chart 2: Per-session final recall
    ax = axes[1]
    x = np.arange(len(engine_results))
    width = 0.35
    e_scores = [r["final_recall_turn50"]["recall_score"] for r in engine_results]
    n_scores = [r["final_recall_turn50"]["recall_score"] for r in naive_results]
    ax.bar(x - width/2, e_scores, width, label="Engine", color="#2ecc71")
    ax.bar(x + width/2, n_scores, width, label="Naive", color="#e74c3c")
    ax.set_xlabel("Session")
    ax.set_ylabel("Needles Recalled (out of 5)")
    ax.set_title("Final Recall (Turn 50) per Session")
    ax.set_ylim(0, 5.5)
    ax.legend()

    fig.suptitle(
        f"50-Turn Extended Session — Gemini Flash ({MAX_CONTEXT_TOKENS // 1024}k context)\n"
        "Stress testing compaction over time",
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
    placements = generate_needle_placements_50(num_sessions)

    print("=" * 60)
    print(f"50-Turn Extended Session — Gemini Flash ({MAX_CONTEXT_TOKENS // 1024}k context)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Sessions: {num_sessions}")
    print(f"Turns: {NUM_TURNS}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Recall checkpoints: turn 25 and turn 50")
    print()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    results: list[dict] = []
    total_tests = num_sessions * 2
    test_num = 0

    for session_id, needle_turns in enumerate(placements):
        for mode in ["engine", "naive"]:
            test_num += 1
            label = f"[{test_num}/{total_tests}] Session {session_id+1} {mode.upper()}"
            print(f"{label} (needles at turns {needle_turns})...", end=" ", flush=True)

            result = run_session(session_id, needle_turns, mode, api_key)
            results.append(result)

            mid = result["mid_recall_turn25"]
            final = result["final_recall_turn50"]
            err = mid.get("error") or final.get("error")
            if err:
                print(f"ERR: {err[:80]}")
            else:
                print(
                    f"Turn25={mid['recall_score']}/5 Turn50={final['recall_score']}/5 "
                    f"compact={result['compaction_events']}"
                )

            time.sleep(2)

    # Save results
    output = {
        "timestamp": timestamp,
        "model": MODEL,
        "benchmark": "gemini_50turn",
        "description": f"50-turn extended session — Gemini Flash {MAX_CONTEXT_TOKENS // 1024}k",
        "num_sessions": num_sessions,
        "num_turns": NUM_TURNS,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "needles": NEEDLES,
        "placements": placements,
        "results": results,
    }

    json_path = RESULTS_DIR / f"gemini_50turn_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    png_path = RESULTS_DIR / f"gemini_50turn_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    for label, mode_key in [("ENGINE (priority compaction)", "engine"), ("NAIVE (sliding window)", "naive")]:
        mode_results = [r for r in results if r["mode"] == mode_key]
        mid_scores = [r["mid_recall_turn25"]["recall_score"] for r in mode_results]
        final_scores = [r["final_recall_turn50"]["recall_score"] for r in mode_results]
        avg_mid = sum(mid_scores) / len(mid_scores) if mid_scores else 0
        avg_final = sum(final_scores) / len(final_scores) if final_scores else 0
        compactions = [r["compaction_events"] for r in mode_results]
        avg_compact = sum(compactions) / len(compactions) if compactions else 0
        print(f"\n{label}:")
        print(f"  Turn 25 avg recall: {avg_mid:.1f}/5 ({100*avg_mid/5:.0f}%)")
        print(f"  Turn 50 avg recall: {avg_final:.1f}/5 ({100*avg_final/5:.0f}%)")
        print(f"  Turn 25 scores: {mid_scores}")
        print(f"  Turn 50 scores: {final_scores}")
        print(f"  Avg compaction events: {avg_compact:.1f}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
