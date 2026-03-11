#!/usr/bin/env python3
"""BM25+Structural regression check: boilerplate + 50-turn benchmarks.

Modes: bm25_structural, bm25, hybrid_v2, naive
Outputs: results/bm25_structural_regression.json + summary table
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
    NEEDLES as BP_NEEDLES, SYSTEM_PROMPT as BP_SYSTEM_PROMPT,
    RECALL_QUESTION as BP_RECALL_QUESTION,
    _generate_filler, generate_needle_placements as bp_placements,
    sliding_window_truncate,
)
from benchmarks.niah_dense import (
    NEEDLES as DENSE_NEEDLES, FILLER_TEMPLATES,
    SYSTEM_PROMPT as DENSE_SYSTEM_PROMPT,
)
from benchmarks.gemini_utils import call_gemini, GEMINI_MODEL

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
MODES = ["bm25_structural", "bm25", "hybrid_v2", "naive"]

# 50-turn recall question (from gemini_bm25_50turn.py)
RECALL_50 = """Please recall ALL important updates and facts mentioned during our conversation.
Specifically, I need you to tell me:
1. What is the secret project codename?
2. When is the database migration scheduled?
3. What was the API rate limit changed to?
4. What is the new hire's name and when does she start?
5. What is the production server IP address?

Answer each question based ONLY on what was mentioned in our conversation."""


def _make_chunk_log(mode: str) -> ChunkLog | None:
    if mode == "naive":
        return None
    scoring = {"bm25_structural": "bm25_structural", "bm25": "bm25", "hybrid_v2": "hybrid_v2"}
    return ChunkLog(
        db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
        soft_threshold=0.7, hard_threshold=0.9,
        auto_priority=False, scoring_mode=scoring[mode],
    )


def _recall(messages, system_prompt, api_key, needles):
    result = call_gemini(messages, system_prompt, api_key, max_output_tokens=1024)
    answer_lower = (result["answer"] or "").lower()
    recalled = [n["id"] for n in needles if n["keyword"].lower() in answer_lower]
    return {**result, "recall_score": len(recalled), "needles_recalled": recalled}


# --- Boilerplate benchmark ---

def run_boilerplate(session_id, needle_turns, mode, api_key, num_turns=30):
    rng = random.Random(session_id * 1000 + hash(mode))
    chunk_log = _make_chunk_log(mode)
    raw_messages: list[dict[str, str]] = [] if chunk_log is None else []
    filler_idx = 0

    for turn in range(num_turns):
        if turn in needle_turns:
            idx = needle_turns.index(turn)
            content = BP_NEEDLES[idx]["fact"]
            priority = 0.5
        else:
            parts = [_generate_filler(session_id * 10000 + filler_idx + i) for i in range(27)]
            filler_idx += 27
            salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}]\n\n" + "\n\n---\n\n".join(parts) + salt
            priority = 0.5

        if chunk_log:
            chunk_log.append("user", content, priority=priority)
            chunk_log.next_turn()
        else:
            raw_messages.append({"role": "user", "content": content})

    if chunk_log:
        chunk_log.append("user", BP_RECALL_QUESTION, priority=2.0)
        messages = chunk_log.get_context()
    else:
        raw_messages.append({"role": "user", "content": BP_RECALL_QUESTION})
        messages = sliding_window_truncate(raw_messages, MAX_CONTEXT_TOKENS)

    result = _recall(messages, BP_SYSTEM_PROMPT, api_key, BP_NEEDLES)
    if chunk_log:
        chunk_log.close()
    return result


# --- 50-turn benchmark ---

def run_50turn(session_id, needle_turns, mode, api_key):
    rng = random.Random(session_id * 1000 + hash(mode))
    chunk_log = _make_chunk_log(mode)
    raw_messages: list[dict[str, str]] = [] if chunk_log is None else []
    mid_recall = None

    for turn in range(50):
        if turn in needle_turns:
            idx = needle_turns.index(turn)
            content = f"IMPORTANT UPDATE: {DENSE_NEEDLES[idx]['fact']}"
            priority = 0.5
        else:
            fillers = [rng.choice(FILLER_TEMPLATES) for _ in range(3)]
            salt = f"\n[Ref: turn_{turn}_session_{session_id}_id_{rng.randint(0, 999999)}]"
            content = f"[Turn {turn+1}] " + "\n\n".join(fillers) + salt
            priority = 0.5

        if chunk_log:
            chunk_log.append("user", content, priority=priority)
            chunk_log.next_turn()
        else:
            raw_messages.append({"role": "user", "content": content})

        # Mid-recall at turn 25
        if turn == 24:
            if chunk_log:
                chunk_log.append("user", RECALL_50, priority=2.0)
                mid_messages = chunk_log.get_context()
            else:
                mid_messages = sliding_window_truncate(
                    list(raw_messages) + [{"role": "user", "content": RECALL_50}],
                    MAX_CONTEXT_TOKENS,
                )
            mid_recall = _recall(mid_messages, DENSE_SYSTEM_PROMPT, api_key, DENSE_NEEDLES)
            # Remove recall question from context
            if chunk_log:
                row = chunk_log._conn.execute(
                    "SELECT chunk_hash FROM chunks ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if row:
                    chunk_log._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (row[0],))
                    chunk_log._conn.commit()
            time.sleep(2)

    # Final recall at turn 50
    if chunk_log:
        chunk_log.append("user", RECALL_50, priority=2.0)
        messages = chunk_log.get_context()
    else:
        raw_messages.append({"role": "user", "content": RECALL_50})
        messages = sliding_window_truncate(raw_messages, MAX_CONTEXT_TOKENS)

    final_recall = _recall(messages, DENSE_SYSTEM_PROMPT, api_key, DENSE_NEEDLES)
    if chunk_log:
        chunk_log.close()

    return {
        "t25": mid_recall["recall_score"] if mid_recall else 0,
        "t50": final_recall["recall_score"],
        "t25_error": mid_recall["error"] if mid_recall else None,
        "t50_error": final_recall["error"],
    }


def _50turn_placements(n):
    placements = []
    rng = random.Random(42)
    for _ in range(n):
        placements.append(sorted(rng.sample(range(20), 5)))
    return placements


def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        sys.exit(1)

    num_sessions = 10
    bp_turns = bp_placements(num_sessions, 30)
    ft_turns = _50turn_placements(num_sessions)

    print("=" * 70)
    print(f"BM25+Structural Regression — Gemini {GEMINI_MODEL}")
    print("=" * 70)
    print(f"Modes: {MODES}")
    print(f"Benchmarks: Boilerplate (30T), 50-Turn (T25 + T50)")
    print()

    all_results: dict[str, Any] = {"model": GEMINI_MODEL, "timestamp": datetime.now(timezone.utc).isoformat()}
    bp_results: dict[str, list] = {m: [] for m in MODES}
    ft_t25: dict[str, list] = {m: [] for m in MODES}
    ft_t50: dict[str, list] = {m: [] for m in MODES}

    # --- Boilerplate ---
    total = num_sessions * len(MODES)
    n = 0
    print("--- BOILERPLATE ---")
    for sid, turns in enumerate(bp_turns):
        for mode in MODES:
            n += 1
            print(f"[{n}/{total}] S{sid+1} {mode}...", end=" ", flush=True)
            r = run_boilerplate(sid, turns, mode, api_key)
            bp_results[mode].append(r["recall_score"])
            if r["error"]:
                print(f"ERR: {r['error'][:60]}")
            else:
                print(f"{r['recall_score']}/5")
            time.sleep(2)

    # --- 50-turn ---
    total = num_sessions * len(MODES)
    n = 0
    print("\n--- 50-TURN ---")
    for sid, turns in enumerate(ft_turns):
        for mode in MODES:
            n += 1
            print(f"[{n}/{total}] S{sid+1} {mode}...", end=" ", flush=True)
            r = run_50turn(sid, turns, mode, api_key)
            ft_t25[mode].append(r["t25"])
            ft_t50[mode].append(r["t50"])
            err = r["t25_error"] or r["t50_error"]
            if err:
                print(f"ERR: {err[:60]}")
            else:
                print(f"T25={r['t25']}/5 T50={r['t50']}/5")
            time.sleep(2)

    # --- Summary table ---
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\n{'=' * 70}")
    print("REGRESSION SUMMARY")
    print(f"{'=' * 70}")
    header = f"{'Scorer':<20} {'Boilerplate':>12} {'50T-T25':>10} {'50T-T50':>10}"
    print(header)
    print("-" * len(header))
    for mode in MODES:
        label = {"bm25_structural": "BM25+Structural", "bm25": "BM25", "hybrid_v2": "HybridV2", "naive": "Naive"}[mode]
        bp_avg = avg(bp_results[mode])
        t25_avg = avg(ft_t25[mode])
        t50_avg = avg(ft_t50[mode])
        print(f"{label:<20} {bp_avg:>10.1f}/5 {t25_avg:>8.1f}/5 {t50_avg:>8.1f}/5")
    print()
    for mode in MODES:
        label = {"bm25_structural": "BM25+Structural", "bm25": "BM25", "hybrid_v2": "HybridV2", "naive": "Naive"}[mode]
        print(f"  {label} BP scores: {bp_results[mode]}")
        print(f"  {label} T25 scores: {ft_t25[mode]}")
        print(f"  {label} T50 scores: {ft_t50[mode]}")
    print(f"{'=' * 70}")

    # Save
    all_results["boilerplate"] = {m: bp_results[m] for m in MODES}
    all_results["50turn_t25"] = {m: ft_t25[m] for m in MODES}
    all_results["50turn_t50"] = {m: ft_t50[m] for m in MODES}
    all_results["summary"] = {
        mode: {
            "boilerplate": avg(bp_results[mode]),
            "50turn_t25": avg(ft_t25[mode]),
            "50turn_t50": avg(ft_t50[mode]),
        }
        for mode in MODES
    }
    json_path = RESULTS_DIR / "bm25_structural_regression.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    main()
