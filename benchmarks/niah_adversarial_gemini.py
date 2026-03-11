#!/usr/bin/env python3
"""Adversarial NIAH with Gemini LLM recall — full end-to-end benchmark.

Tests structural fingerprinting vs other scorers by actually asking
Gemini to recall needle facts from the managed context window.

Uses: gemini-2.5-flash-lite-preview-09-2025
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine import ChunkLog, _estimate_tokens

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GEMINI_MODEL = "gemini-2.5-flash-lite-preview-09-2025"
MAX_CONTEXT_TOKENS = 8000
SYSTEM_PROMPT = (
    "You are a helpful assistant with perfect memory. "
    "Answer questions based on the conversation history provided."
)

from benchmarks.niah_adversarial import (
    NEEDLES,
    ADVERSARIAL_FILLER,
    RECALL_QUESTION,
    generate_needle_placements,
    sliding_window_truncate,
)


def gemini_chat(messages: list[dict[str, str]], api_key: str) -> tuple[str, float]:
    """Send messages to Gemini and return (response_text, latency)."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)

    # Convert messages to Gemini format (alternating user/model)
    history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)
    last_msg = messages[-1]["content"]

    t0 = time.time()
    response = chat.send_message(last_msg)
    latency = time.time() - t0

    return response.text or "", latency


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,
    api_key: str,
    num_turns: int = 30,
) -> dict[str, Any]:
    """Run one session: fill context, ask recall question via Gemini."""
    rng = random.Random(session_id * 1000 + hash(mode))

    if mode == "structural":
        log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9, scoring_mode="structural",
        )
    elif mode == "goal_guided":
        log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9, goal_guided=True,
        )
    elif mode == "engine":
        log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
        )
    elif mode == "naive":
        log = None
        raw_msgs: list[dict[str, str]] = []
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            content = NEEDLES[needle_idx]["fact"]
            priority = 2.0 if mode == "engine" else 0.5
        else:
            filler = rng.choice(ADVERSARIAL_FILLER)
            salt = f"\n[Ref: turn_{turn}_s{session_id}_id_{rng.randint(0, 999999)}]"
            content = (
                f"[Turn {turn+1}] {filler}\n\n"
                f"Additional context for this turn:\n"
                f"{rng.choice(ADVERSARIAL_FILLER)}{salt}"
            )
            priority = 0.5

        if log:
            log.append("user", content, priority=priority)
            log.next_turn()
        else:
            raw_msgs.append({"role": "user", "content": content})

    # Add recall question
    if log:
        log.append("user", RECALL_QUESTION, priority=2.0)
        messages = log.get_context()
        ctx_tokens = log.get_context_tokens()
        compaction = log.compaction_count
    else:
        raw_msgs.append({"role": "user", "content": RECALL_QUESTION})
        messages = sliding_window_truncate(raw_msgs, MAX_CONTEXT_TOKENS)
        ctx_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
        compaction = 0

    # Check needle survival in context
    ctx_text = " ".join(m["content"] for m in messages)
    needles_in_ctx = [n["id"] for n in NEEDLES if n["keyword"].lower() in ctx_text.lower()]

    # Call Gemini
    try:
        answer, latency = gemini_chat(messages, api_key)
        error = None
    except Exception as e:
        answer, latency, error = "", 0.0, str(e)

    if log:
        log.close()

    # Score recall
    answer_lower = answer.lower()
    recalled = [n["id"] for n in NEEDLES if n["keyword"].lower() in answer_lower]
    lost = [n["id"] for n in NEEDLES if n["id"] not in recalled]

    return {
        "session_id": session_id,
        "mode": mode,
        "error": error,
        "needle_turns": needle_turns,
        "needles_in_context": needles_in_ctx,
        "needles_recalled": recalled,
        "needles_lost": lost,
        "recall_score": len(recalled),
        "context_tokens": ctx_tokens,
        "compaction": compaction,
        "latency": round(latency, 2),
        "answer": answer[:500],
    }


def generate_chart(results: list[dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes = ["structural", "goal_guided", "engine", "naive"]
    labels = ["Structural\n(fingerprint)", "TF-IDF\n(goal-guided)", "Hardcoded\n(priority=2)", "Naive\n(sliding)"]
    colors = ["#8e44ad", "#9b59b6", "#2ecc71", "#e74c3c"]

    avgs = []
    for mode in modes:
        scores = [r["recall_score"] for r in results if r["mode"] == mode and not r["error"]]
        avgs.append(sum(scores) / len(scores) if scores else 0)

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(modes))
    bars = ax.bar(x, avgs, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Avg Needles Recalled (out of 5)")
    ax.set_ylim(0, 5.5)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(
        f"Adversarial NIAH — Gemini LLM Recall\n"
        f"({GEMINI_MODEL}, 10 sessions x 30 turns x 5 needles, 8k window)",
        fontsize=11,
    )
    for bar, val in zip(bars, avgs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.1f}/5", ha="center", fontsize=11, fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    num_sessions = 10
    num_turns = 30
    placements = generate_needle_placements(num_sessions, num_turns)
    modes = ["structural", "goal_guided", "engine", "naive"]
    total = num_sessions * len(modes)

    print("=" * 70)
    print(f"Adversarial NIAH — Gemini LLM Recall ({GEMINI_MODEL})")
    print("=" * 70)
    print(f"Sessions: {num_sessions}, Turns: {num_turns}, Needles: {len(NEEDLES)}")
    print(f"Context: {MAX_CONTEXT_TOKENS:,} tokens, Modes: {modes}")
    print()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    results: list[dict] = []
    test_num = 0

    for sid, needle_turns in enumerate(placements):
        for mode in modes:
            test_num += 1
            tag = f"[{test_num}/{total}] S{sid+1} {mode.upper()}"
            print(f"{tag} (needles@{needle_turns})...", end=" ", flush=True)

            result = run_session(sid, needle_turns, mode, api_key, num_turns)
            results.append(result)

            if result["error"]:
                print(f"ERR: {result['error'][:60]}")
            else:
                r = result["recall_score"]
                ic = len(result["needles_in_context"])
                lost = result["needles_lost"]
                print(f"recalled={r}/5 in_ctx={ic}/5 lost={lost} {result['latency']:.1f}s")

            time.sleep(0.5)  # rate limit courtesy

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for mode in modes:
        mr = [r for r in results if r["mode"] == mode and not r["error"]]
        scores = [r["recall_score"] for r in mr]
        in_ctx = [len(r["needles_in_context"]) for r in mr]
        avg_recall = sum(scores) / len(scores) if scores else 0
        avg_ctx = sum(in_ctx) / len(in_ctx) if in_ctx else 0

        lost_counts: dict[str, int] = {}
        for r in mr:
            for nid in r["needles_lost"]:
                lost_counts[nid] = lost_counts.get(nid, 0) + 1

        label = {"structural": "Structural", "goal_guided": "TF-IDF",
                 "engine": "Hardcoded", "naive": "Naive"}[mode]
        print(f"\n{label}:")
        print(f"  Avg recall:     {avg_recall:.1f}/5 ({100*avg_recall/5:.0f}%)")
        print(f"  Avg in context: {avg_ctx:.1f}/5")
        print(f"  Scores:         {scores}")
        print(f"  Lost:           {dict(sorted(lost_counts.items())) if lost_counts else 'none'}")

    # Save
    output = {
        "timestamp": timestamp,
        "model": GEMINI_MODEL,
        "benchmark": "niah_adversarial_gemini",
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "results": results,
    }
    json_path = RESULTS_DIR / f"niah_adversarial_gemini_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    png_path = RESULTS_DIR / f"niah_adversarial_gemini_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart error: {e}")


if __name__ == "__main__":
    main()
