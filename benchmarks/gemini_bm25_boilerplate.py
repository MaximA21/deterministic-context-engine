#!/usr/bin/env python3
"""BM25 Boilerplate NIAH — BM25 vs TF-IDF vs Hardcoded vs Naive.

Same boilerplate setup as gemini_fixed_boilerplate (27 fillers/turn, ~3x compression)
but adding BM25 scoring mode for comparison.
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
from benchmarks.gemini_utils import call_gemini, GEMINI_MODEL

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
FILLERS_PER_TURN = 27  # → ~95k total through 32k window = 2.89x


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,  # "bm25", "goal_guided", "engine", or "naive"
    api_key: str,
    num_turns: int = 30,
) -> dict[str, Any]:
    rng = random.Random(session_id * 1000 + hash(mode))
    compaction_log: list[dict] = []
    total_tokens_added = 0
    filler_seed_offset = session_id * 10000

    if mode == "bm25":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=False, scoring_mode="bm25",
        )
    elif mode == "goal_guided":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=False, goal_guided=True,
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
            if mode in ("bm25", "goal_guided"):
                priority = 0.5
            elif mode == "engine":
                priority = 2.0
            else:
                priority = 0.5
        else:
            filler_parts = []
            for _ in range(FILLERS_PER_TURN):
                filler_parts.append(_generate_filler(filler_seed_offset + filler_idx))
                filler_idx += 1
            unique_salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}]\n\n" + "\n\n---\n\n".join(filler_parts) + unique_salt
            priority = 0.5

        total_tokens_added += _estimate_tokens(content)

        if mode in ("bm25", "goal_guided", "engine"):
            tokens_before = chunk_log.current_tokens()
            compactions_before = chunk_log.compaction_count
            chunk_log.append("user", content, priority=priority)
            chunk_log.next_turn()
            tokens_after = chunk_log.current_tokens()
            compactions_after = chunk_log.compaction_count
            if compactions_after > compactions_before:
                compaction_log.append({
                    "turn": turn, "tokens_before": tokens_before + _estimate_tokens(content),
                    "tokens_after": tokens_after, "events": compactions_after - compactions_before,
                })
        else:
            raw_messages.append({"role": "user", "content": content})

    # Add recall question
    if mode in ("bm25", "goal_guided", "engine"):
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

    result = call_gemini(messages, SYSTEM_PROMPT, api_key, max_output_tokens=1024)

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


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    num_sessions = 10
    num_turns = 30
    placements = generate_needle_placements(num_sessions, num_turns)
    modes = ["bm25", "goal_guided", "engine", "naive"]

    print("=" * 60)
    print(f"BM25 Boilerplate NIAH — Gemini ({MAX_CONTEXT_TOKENS // 1024}k context)")
    print("=" * 60)
    print(f"Model: {GEMINI_MODEL}")
    print(f"Fillers per turn: {FILLERS_PER_TURN} (target ~3x compression)")
    print(f"Modes: {modes}")
    print()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
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
                print(f"Recalled {result['recall_score']}/5 ctx={result['context_tokens']} ratio={ratio:.2f}x compact={result['compaction_events']}")
            time.sleep(2)

    # Save
    output = {
        "timestamp": timestamp, "model": GEMINI_MODEL,
        "benchmark": "gemini_bm25_boilerplate",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "fillers_per_turn": FILLERS_PER_TURN,
        "results": results,
    }
    json_path = RESULTS_DIR / f"gemini_bm25_boilerplate_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Audit summary
    print(f"\n{'=' * 60}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 60}")
    for label, mode_key in [("BM25", "bm25"), ("TF-IDF (Goal-Guided)", "goal_guided"), ("HARDCODED (priority=2.0)", "engine"), ("NAIVE (sliding window)", "naive")]:
        mr = [r for r in results if r["mode"] == mode_key and not r["error"]]
        scores = [r["recall_score"] for r in mr]
        avg = sum(scores) / len(scores) if scores else 0
        avg_ctx = sum(r["context_tokens"] for r in mr) / len(mr) if mr else 0
        avg_added = sum(r["total_tokens_added"] for r in mr) / len(mr) if mr else 0
        avg_ratio = avg_added / avg_ctx if avg_ctx > 0 else 0
        avg_compact = sum(r["compaction_events"] for r in mr) / len(mr) if mr else 0
        avg_in_ctx = sum(len(r["needles_in_context"]) for r in mr) / len(mr) if mr else 0
        print(f"\n{label}:")
        print(f"  Avg recall: {avg:.1f}/5  Scores: {scores}")
        print(f"  Avg context at recall: {avg_ctx:.0f} / {MAX_CONTEXT_TOKENS} ({100*avg_ctx/MAX_CONTEXT_TOKENS:.0f}%)")
        print(f"  Avg total added: {avg_added:.0f}  Compression ratio: {avg_ratio:.2f}x")
        print(f"  Avg compactions: {avg_compact:.1f}  Avg needles in context: {avg_in_ctx:.1f}/5")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
