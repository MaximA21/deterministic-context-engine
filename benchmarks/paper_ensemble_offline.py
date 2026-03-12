#!/usr/bin/env python3
"""Paper Ensemble Offline Benchmark — needle retention test (no LLM calls).

Tests needle-in-context retention across all 4 NIAH scenarios:
- Dense: standard filler, 3 per turn
- Adversarial: keyword-sharing filler, 14 per turn
- Boilerplate: structurally similar filler, 27 per turn
- 50-Turn: extended session, needles in first 20 turns

Measures: needles retained in context after compaction (0-5 per session).
This is the lower bound — Gemini recall may be lower than retention.
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine import ChunkLog, _estimate_tokens
from benchmarks.niah_dense import NEEDLES as DENSE_NEEDLES, FILLER_TEMPLATES
from benchmarks.niah_adversarial import (
    NEEDLES as ADV_NEEDLES, ADVERSARIAL_FILLER, RECALL_QUESTION as ADV_RECALL,
)
from benchmarks.niah_boilerplate import (
    NEEDLES as BOILERPLATE_NEEDLES, RECALL_QUESTION as BOILERPLATE_RECALL,
    _generate_filler,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "paper_ensemble"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT_TOKENS = 32768
NUM_SESSIONS = 10
MODES = ["paper_ensemble", "bm25", "structural"]


DENSE_RECALL = (
    "What is the secret project codename? When is the database migration? "
    "What was the API rate limit changed to? New hire name? Production server IP?"
)


def run_scenario(
    scenario: str,
    needles: list[dict],
    recall_question: str,
    filler_fn,
    needle_prefix: str,
    num_turns: int,
    num_sessions: int,
) -> dict:
    """Run one scenario across all modes and sessions."""
    scenario_results = {}

    for mode in MODES:
        mode_results = []
        for sess in range(num_sessions):
            rng = random.Random(42 + sess)
            if scenario == "50turn":
                needle_turns = sorted(rng.sample(range(20), 5))
            else:
                needle_turns = sorted(rng.sample(range(num_turns), 5))
            rng2 = random.Random(sess * 1000)

            log = ChunkLog(
                db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
                soft_threshold=0.7, hard_threshold=0.9,
                scoring_mode=mode,
            )

            t0 = time.time()
            total_tokens_added = 0

            for t in range(num_turns):
                if t in needle_turns:
                    idx = needle_turns.index(t)
                    content = needle_prefix + needles[idx]["fact"]
                else:
                    content = filler_fn(rng2, t, sess)

                total_tokens_added += _estimate_tokens(content)
                log.append("user", content, priority=0.5)
                log.next_turn()

            log.append("user", recall_question, priority=2.0)
            elapsed = time.time() - t0

            context_text = " ".join(m["content"] for m in log.get_context())
            retained = [
                n["id"] for n in needles
                if n["keyword"].lower() in context_text.lower()
            ]

            mode_results.append({
                "session_id": sess,
                "needle_turns": needle_turns,
                "needles_retained": retained,
                "retention_score": len(retained),
                "context_tokens": log.current_tokens(),
                "total_tokens_added": total_tokens_added,
                "compaction_events": log.compaction_count,
                "scoring_time_ms": round(elapsed * 1000, 1),
            })
            log.close()

        scores = [r["retention_score"] for r in mode_results]
        scenario_results[mode] = {
            "results": mode_results,
            "avg_retention": sum(scores) / len(scores),
            "scores": scores,
            "perfect_sessions": sum(1 for s in scores if s == 5),
        }

    return scenario_results


def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    scenarios = {
        "dense": {
            "needles": DENSE_NEEDLES,
            "recall_question": DENSE_RECALL,
            "filler_fn": lambda rng, t, s: (
                f"[Turn {t+1}] "
                + "\n\n".join(rng.choice(FILLER_TEMPLATES) for _ in range(3))
            ),
            "needle_prefix": "IMPORTANT UPDATE: ",
            "num_turns": 30,
        },
        "adversarial": {
            "needles": ADV_NEEDLES,
            "recall_question": ADV_RECALL,
            "filler_fn": lambda rng, t, s: (
                f"[Turn {t+1}] "
                + "\n\n".join(rng.choice(ADVERSARIAL_FILLER) for _ in range(14))
            ),
            "needle_prefix": "",
            "num_turns": 30,
        },
        "boilerplate": {
            "needles": BOILERPLATE_NEEDLES,
            "recall_question": BOILERPLATE_RECALL,
            "filler_fn": lambda rng, t, s: (
                f"[Turn {t+1}] "
                + "\n\n".join(_generate_filler(s * 10000 + t * 100 + f) for f in range(27))
            ),
            "needle_prefix": "",
            "num_turns": 30,
        },
        "50turn": {
            "needles": DENSE_NEEDLES,
            "recall_question": DENSE_RECALL,
            "filler_fn": lambda rng, t, s: (
                f"[Turn {t+1}] "
                + "\n\n".join(rng.choice(FILLER_TEMPLATES) for _ in range(3))
            ),
            "needle_prefix": "IMPORTANT UPDATE: ",
            "num_turns": 50,
        },
    }

    all_results = {}

    for scenario_name, config in scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"  {scenario_name.upper()} ({NUM_SESSIONS} sessions)")
        print(f"{'=' * 60}")

        results = run_scenario(
            scenario=scenario_name,
            num_sessions=NUM_SESSIONS,
            **config,
        )
        all_results[scenario_name] = results

        for mode in MODES:
            r = results[mode]
            print(f"  {mode.upper():20s}: avg {r['avg_retention']:.1f}/5  "
                  f"perfect={r['perfect_sessions']}/{NUM_SESSIONS}  "
                  f"scores={r['scores']}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY: Average Needle Retention (out of 5)")
    print(f"{'=' * 60}")
    print(f"{'Scenario':<15s}", end="")
    for mode in MODES:
        print(f"  {mode.upper():<20s}", end="")
    print()
    print("-" * 75)
    for scenario_name in scenarios:
        print(f"{scenario_name:<15s}", end="")
        for mode in MODES:
            avg = all_results[scenario_name][mode]["avg_retention"]
            print(f"  {avg:<20.1f}", end="")
        print()
    print(f"{'=' * 60}")

    # Save results
    output = {
        "timestamp": timestamp,
        "benchmark": "paper_ensemble_offline",
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "num_sessions": NUM_SESSIONS,
        "modes": MODES,
        "results": {
            scenario: {
                mode: {
                    "avg_retention": data["avg_retention"],
                    "scores": data["scores"],
                    "perfect_sessions": data["perfect_sessions"],
                }
                for mode, data in scenario_results.items()
            }
            for scenario, scenario_results in all_results.items()
        },
    }
    json_path = RESULTS_DIR / f"offline_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
