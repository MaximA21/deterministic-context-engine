#!/usr/bin/env python3
"""Dense NIAH v2 with Gemini Flash — Engine vs Naive sliding window.

Same test as niah_dense.py but using Gemini 2.5 Flash with 32k artificial
context budget. Uses more filler per turn to ensure compaction fires at 32k.
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
    RECALL_QUESTION, generate_needle_placements, sliding_window_truncate,
    generate_chart,
)
from benchmarks.gemini_utils import call_gemini

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
MODEL = "gemini-3.1-flash-lite-preview"


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

    if mode == "engine":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
        )
    else:
        chunk_log = None
        raw_messages: list[dict[str, str]] = []

    for turn in range(num_turns):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            needle = NEEDLES[needle_idx]
            content = f"IMPORTANT UPDATE: {needle['fact']}"
            priority = 2.0
        else:
            # Use 4 filler templates per turn to fill 32k context
            fillers = [rng.choice(FILLER_TEMPLATES) for _ in range(4)]
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

    # Add recall question
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

    # Track which needles survived in context
    needles_in_context = []
    context_text = " ".join(m["content"] for m in messages)
    for needle in NEEDLES:
        if needle["keyword"].lower() in context_text.lower():
            needles_in_context.append(needle["id"])

    # Call Gemini API
    result = call_gemini(messages, SYSTEM_PROMPT, api_key, model=MODEL, max_output_tokens=512)

    if chunk_log:
        chunk_log.close()

    # Score: check which needles were recalled
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


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    num_sessions = 10
    num_turns = 30
    placements = generate_needle_placements(num_sessions, num_turns)

    print("=" * 60)
    print(f"Dense NIAH v2 — Gemini Flash ({MAX_CONTEXT_TOKENS // 1024}k context)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Sessions: {num_sessions}")
    print(f"Turns per session: {num_turns}")
    print(f"Needles per session: {len(NEEDLES)}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Filler per turn: 4 templates (to fill 32k)")
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

            time.sleep(2)  # Rate limit buffer for Gemini

    # Save results
    output = {
        "timestamp": timestamp,
        "model": MODEL,
        "benchmark": "gemini_dense_niah_v2",
        "description": f"Engine (priority compaction) vs Naive (sliding window) — Gemini Flash {MAX_CONTEXT_TOKENS // 1024}k",
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "needles": NEEDLES,
        "placements": placements,
        "results": results,
    }

    json_path = RESULTS_DIR / f"gemini_dense_niah_v2_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Generate chart
    png_path = RESULTS_DIR / f"gemini_dense_niah_v2_{timestamp}.png"
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
        scores = [r["recall_score"] for r in mode_results]
        errors = sum(1 for r in mode_results if r["error"])
        avg = sum(scores) / len(scores) if scores else 0
        compactions = [r["compaction_events"] for r in mode_results]
        avg_compact = sum(compactions) / len(compactions) if compactions else 0
        print(f"\n{label}:")
        print(f"  Avg recall: {avg:.1f}/5 ({100*avg/5:.0f}%)")
        print(f"  Scores: {scores}")
        print(f"  API errors: {errors}")
        print(f"  Avg compaction events: {avg_compact:.1f}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
