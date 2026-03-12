#!/usr/bin/env python3
"""ACC 50-Turn Extended — Active Context Compression vs BM25 vs Naive.

50 turns with all 5 needles planted before turn 20. Recall at turn 25
AND turn 50. Tests long-horizon retention with ACC's sawtooth pattern.

This is the hardest test for ACC: needles planted early must survive
multiple consolidation cycles (turns 10, 20, 30, 40).
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
from benchmarks.gemini_utils import call_gemini, GEMINI_MODEL

RESULTS_DIR = PROJECT_ROOT / "results" / "acc_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
NUM_TURNS = 50
ACC_INTERVAL = 10
ACC_KEEP_RECENT = 3

RECALL_QUESTION = """Please recall ALL important updates and facts mentioned during our conversation.
Specifically, I need you to tell me:
1. What is the secret project codename?
2. When is the database migration scheduled?
3. What was the API rate limit changed to?
4. What is the new hire's name and when does she start?
5. What is the production server IP address?

Answer each question based ONLY on what was mentioned in our conversation."""


def generate_early_placements(num_sessions: int) -> list[list[int]]:
    """All 5 needles placed in turns 0-19."""
    placements = []
    rng = random.Random(42)
    for _ in range(num_sessions):
        turns = sorted(rng.sample(range(20), 5))
        placements.append(turns)
    return placements


def _do_recall(messages: list[dict[str, str]], api_key: str) -> dict:
    result = call_gemini(messages, SYSTEM_PROMPT, api_key, max_output_tokens=512)
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


def run_session(session_id: int, needle_turns: list[int], mode: str, api_key: str) -> dict[str, Any]:
    rng = random.Random(session_id * 1000 + hash(mode))
    compaction_log: list[dict] = []
    total_tokens_added = 0

    if mode == "acc":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=2.0, hard_threshold=2.0,
            auto_priority=False, scoring_mode="acc",
            acc_interval=ACC_INTERVAL, acc_keep_recent=ACC_KEEP_RECENT,
            acc_api_key=api_key,
        )
    elif mode == "bm25":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=False, scoring_mode="bm25",
        )
    else:
        chunk_log = None
        raw_messages: list[dict[str, str]] = []

    mid_recall = None

    for turn in range(NUM_TURNS):
        if turn in needle_turns:
            needle_idx = needle_turns.index(turn)
            needle = NEEDLES[needle_idx]
            content = f"IMPORTANT UPDATE: {needle['fact']}"
            priority = 0.5
        else:
            fillers = [rng.choice(FILLER_TEMPLATES) for _ in range(3)]
            unique_salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] " + "\n\n".join(fillers) + unique_salt
            priority = 0.5

        total_tokens_added += _estimate_tokens(content)

        if mode in ("acc", "bm25"):
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

        # Mid-session recall at turn 25
        if turn == 24:
            if mode in ("acc", "bm25"):
                chunk_log.append("user", RECALL_QUESTION, priority=2.0)
                mid_messages = chunk_log.get_context()
                mid_ctx_tokens = chunk_log.get_context_tokens()
            else:
                mid_raw = list(raw_messages) + [{"role": "user", "content": RECALL_QUESTION}]
                mid_messages = sliding_window_truncate(mid_raw, MAX_CONTEXT_TOKENS)
                mid_ctx_tokens = sum(_estimate_tokens(m["content"]) for m in mid_messages)

            mid_recall = _do_recall(mid_messages, api_key)
            mid_recall["context_tokens"] = mid_ctx_tokens

            # Remove recall question from context
            if mode in ("acc", "bm25"):
                recall_hash = chunk_log._conn.execute(
                    "SELECT chunk_hash FROM chunks ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if recall_hash:
                    chunk_log._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (recall_hash[0],))
                    chunk_log._conn.commit()
            time.sleep(2)

    # Final recall at turn 50
    if mode in ("acc", "bm25"):
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

    acc_metrics = chunk_log.acc_metrics if mode in ("acc",) else {}

    if chunk_log:
        chunk_log.close()

    return {
        "session_id": session_id, "mode": mode, "num_turns": NUM_TURNS,
        "needle_turns": needle_turns,
        "needles_in_context": needles_in_context,
        "mid_recall_turn25": {
            "recall_score": mid_recall["recall_score"] if mid_recall else 0,
            "needles_recalled": mid_recall["needles_recalled"] if mid_recall else [],
            "needles_lost": mid_recall["needles_lost"] if mid_recall else [],
            "context_tokens": mid_recall.get("context_tokens", 0) if mid_recall else 0,
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
        "acc_metrics": acc_metrics,
    }


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    num_sessions = 10
    placements = generate_early_placements(num_sessions)
    modes = ["acc", "bm25", "naive"]

    print("=" * 60)
    print(f"ACC 50-Turn Extended — Gemini ({MAX_CONTEXT_TOKENS // 1024}k context)")
    print("=" * 60)
    print(f"Model: {GEMINI_MODEL}")
    print(f"ACC interval: every {ACC_INTERVAL} turns, keep last {ACC_KEEP_RECENT} raw")
    print(f"All 5 needles planted before turn 20")
    print(f"Recall at turn 25 AND turn 50")
    print(f"Modes: {modes}")
    print()
    for i, p in enumerate(placements):
        print(f"  Session {i+1}: needles at turns {p} (all < 20)")
    print()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    results: list[dict] = []
    total_tests = num_sessions * len(modes)
    test_num = 0

    for session_id, needle_turns in enumerate(placements):
        for mode in modes:
            test_num += 1
            print(f"[{test_num}/{total_tests}] S{session_id+1} {mode.upper()}...", end=" ", flush=True)
            result = run_session(session_id, needle_turns, mode, api_key)
            results.append(result)
            mid = result["mid_recall_turn25"]
            final = result["final_recall_turn50"]
            err = mid.get("error") or final.get("error")
            if err:
                print(f"ERR: {err[:80]}")
            else:
                ratio = result["total_tokens_added"] / max(1, result["context_tokens"])
                acc_lat = ""
                if result.get("acc_metrics"):
                    avg_lat = result["acc_metrics"].get("avg_consolidation_latency", 0)
                    acc_lat = f" acc_lat={avg_lat:.2f}s"
                print(f"T25={mid['recall_score']}/5 T50={final['recall_score']}/5 ratio={ratio:.2f}x compact={result['compaction_events']}{acc_lat}")
            time.sleep(2)

    # Save
    output = {
        "timestamp": timestamp, "model": GEMINI_MODEL,
        "benchmark": "acc_50turn",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "acc_interval": ACC_INTERVAL,
        "acc_keep_recent": ACC_KEEP_RECENT,
        "needle_placement": "all 5 needles before turn 20",
        "results": results,
    }
    json_path = RESULTS_DIR / f"acc_50turn_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Audit summary
    print(f"\n{'=' * 60}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 60}")
    for label, mode_key in [("ACC (sawtooth)", "acc"), ("BM25", "bm25"), ("NAIVE (sliding window)", "naive")]:
        mr = [r for r in results if r["mode"] == mode_key]
        mid_scores = [r["mid_recall_turn25"]["recall_score"] for r in mr]
        final_scores = [r["final_recall_turn50"]["recall_score"] for r in mr]
        avg_mid = sum(mid_scores) / len(mid_scores) if mid_scores else 0
        avg_final = sum(final_scores) / len(final_scores) if final_scores else 0
        avg_ratio = sum(r["total_tokens_added"] / max(1, r["context_tokens"]) for r in mr) / len(mr) if mr else 0
        avg_compact = sum(r["compaction_events"] for r in mr) / len(mr) if mr else 0
        print(f"\n{label}:")
        print(f"  Turn 25: {avg_mid:.1f}/5  Scores: {mid_scores}")
        print(f"  Turn 50: {avg_final:.1f}/5  Scores: {final_scores}")
        print(f"  Compression ratio: {avg_ratio:.2f}x  Compactions: {avg_compact:.1f}")
        if mode_key == "acc":
            acc_results = [r for r in mr if r.get("acc_metrics")]
            if acc_results:
                all_lats = [lat for r in acc_results for lat in r["acc_metrics"].get("consolidation_latencies", [])]
                avg_lat = sum(all_lats) / len(all_lats) if all_lats else 0
                total_lat = sum(r["acc_metrics"].get("total_consolidation_latency", 0) for r in acc_results)
                avg_total_lat = total_lat / len(acc_results)
                llm_in = sum(r["acc_metrics"].get("llm_input_tokens", 0) for r in acc_results)
                llm_out = sum(r["acc_metrics"].get("llm_output_tokens", 0) for r in acc_results)
                print(f"  Avg consolidation latency: {avg_lat:.2f}s  Avg total latency per session: {avg_total_lat:.2f}s")
                print(f"  Total LLM tokens (summarization): {llm_in} in / {llm_out} out")
    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
