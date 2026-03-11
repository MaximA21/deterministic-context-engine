#!/usr/bin/env python3
"""Master comparison: Cerebras llama3.1-8b vs Gemini Flash for all benchmarks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_latest(pattern: str) -> dict | None:
    """Load the most recent JSON result file matching pattern."""
    files = sorted(RESULTS_DIR.glob(pattern), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def avg_score(results: list[dict], mode: str, key: str = "recall_score") -> float:
    scores = [r[key] for r in results if r.get("mode") == mode and not r.get("error")]
    return sum(scores) / len(scores) if scores else 0.0


def print_comparison_table(title: str, rows: list[tuple[str, float, float]]) -> None:
    """Print a formatted comparison table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Mode':<35} {'Cerebras 8b':>12} {'Gemini Flash':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    for mode, cerebras, gemini in rows:
        c_str = f"{cerebras:.1f}/5" if cerebras >= 0 else "N/A"
        g_str = f"{gemini:.1f}/5" if gemini >= 0 else "N/A"
        print(f"  {mode:<35} {c_str:>12} {g_str:>12}")
    print(f"{'=' * 70}")


def main():
    print("\n" + "#" * 70)
    print("#  MASTER COMPARISON: Cerebras llama3.1-8b vs Gemini 3.1 Flash Lite")
    print("#" * 70)

    # --- Dense NIAH ---
    cerebras_dense = load_latest("niah_dense_v2_*.json")
    gemini_dense = load_latest("gemini_dense_niah_v2_*.json")

    rows = []
    if cerebras_dense and gemini_dense:
        for mode in ["engine", "naive"]:
            c = avg_score(cerebras_dense["results"], mode)
            g = avg_score(gemini_dense["results"], mode)
            label = "Engine (priority compaction)" if mode == "engine" else "Naive (sliding window)"
            rows.append((label, c, g))
        print_comparison_table("Dense NIAH v2 (30 turns, 5 needles)", rows)
    else:
        missing = []
        if not cerebras_dense:
            missing.append("Cerebras dense")
        if not gemini_dense:
            missing.append("Gemini dense")
        print(f"\n[Dense NIAH] Missing results: {', '.join(missing)}")

    # --- Adversarial NIAH ---
    cerebras_adv = load_latest("niah_adversarial_*.json")
    gemini_adv = load_latest("gemini_adversarial_*.json")

    rows = []
    if cerebras_adv and gemini_adv:
        mode_pairs = [
            ("goal_guided", "Goal-Guided (TF-IDF)"),
            ("engine", "Hardcoded (priority=2.0)"),
            ("auto_priority", "Keywords (AutoPriority)"),
            ("naive", "Naive (sliding window)"),
        ]
        for mode_key, label in mode_pairs:
            c = avg_score(cerebras_adv["results"], mode_key)
            g = avg_score(gemini_adv["results"], mode_key)
            if c < 0 and g < 0:
                continue  # Skip modes not present in both
            rows.append((label, c if c >= 0 else -1, g if g >= 0 else -1))
        # Cerebras adversarial may not have goal_guided mode — check
        cerebras_modes = set(r["mode"] for r in cerebras_adv["results"])
        if "goal_guided" not in cerebras_modes:
            rows = [(l, c, g) if "Goal-Guided" not in l else (l, -1, g) for l, c, g in rows]
        print_comparison_table("Adversarial NIAH (30 turns, keyword overlap)", rows)
    else:
        missing = []
        if not cerebras_adv:
            missing.append("Cerebras adversarial")
        if not gemini_adv:
            missing.append("Gemini adversarial")
        print(f"\n[Adversarial NIAH] Missing results: {', '.join(missing)}")

    # --- Boilerplate NIAH ---
    cerebras_bp = load_latest("niah_boilerplate_*.json")
    gemini_bp = load_latest("gemini_boilerplate_*.json")

    rows = []
    if cerebras_bp and gemini_bp:
        for mode_key, label in [
            ("goal_guided", "Goal-Guided (TF-IDF)"),
            ("engine", "Hardcoded (priority=2.0)"),
            ("auto_priority", "Keywords (AutoPriority)"),
            ("naive", "Naive (sliding window)"),
        ]:
            c = avg_score(cerebras_bp["results"], mode_key)
            g = avg_score(gemini_bp["results"], mode_key)
            rows.append((label, c, g))
        print_comparison_table("Boilerplate NIAH (repetitive critical content)", rows)
    else:
        missing = []
        if not cerebras_bp:
            missing.append("Cerebras boilerplate")
        if not gemini_bp:
            missing.append("Gemini boilerplate")
        print(f"\n[Boilerplate NIAH] Missing results: {', '.join(missing)}")

    # --- 50-Turn Extended ---
    gemini_50 = load_latest("gemini_50turn_*.json")

    if gemini_50:
        print(f"\n{'=' * 70}")
        print(f"  50-Turn Extended Session (Gemini Flash only)")
        print(f"{'=' * 70}")
        print(f"  {'Mode':<35} {'Turn 25':>12} {'Turn 50':>12}")
        print(f"  {'-'*35} {'-'*12} {'-'*12}")
        for mode in ["engine", "naive"]:
            mode_results = [r for r in gemini_50["results"] if r["mode"] == mode]
            mid = [r["mid_recall_turn25"]["recall_score"] for r in mode_results]
            final = [r["final_recall_turn50"]["recall_score"] for r in mode_results]
            avg_mid = sum(mid) / len(mid) if mid else 0
            avg_final = sum(final) / len(final) if final else 0
            label = "Engine (priority compaction)" if mode == "engine" else "Naive (sliding window)"
            print(f"  {label:<35} {avg_mid:.1f}/5{' ':>6} {avg_final:.1f}/5{' ':>6}")
        print(f"{'=' * 70}")
    else:
        print(f"\n[50-Turn] Missing Gemini 50-turn results")

    # --- Overall Summary ---
    print(f"\n{'#' * 70}")
    print(f"#  KEY FINDINGS")
    print(f"{'#' * 70}")

    if gemini_dense and cerebras_dense:
        c_eng = avg_score(cerebras_dense["results"], "engine")
        g_eng = avg_score(gemini_dense["results"], "engine")
        delta = g_eng - c_eng
        direction = "better" if delta > 0 else "worse" if delta < 0 else "same"
        print(f"\n  Dense NIAH: Gemini engine is {abs(delta):.1f} points {direction} than Cerebras")

    if gemini_bp and cerebras_bp:
        c_gg = avg_score(cerebras_bp["results"], "goal_guided")
        g_gg = avg_score(gemini_bp["results"], "goal_guided")
        print(f"  Boilerplate TF-IDF: Cerebras {c_gg:.1f}/5 → Gemini {g_gg:.1f}/5")
        if g_gg > c_gg:
            print(f"  → Smarter model DOES improve boilerplate recall (+{g_gg - c_gg:.1f})")
        else:
            print(f"  → Smarter model does NOT improve boilerplate recall")

    if gemini_50:
        eng_50 = [r for r in gemini_50["results"] if r["mode"] == "engine"]
        mid_avg = sum(r["mid_recall_turn25"]["recall_score"] for r in eng_50) / len(eng_50)
        fin_avg = sum(r["final_recall_turn50"]["recall_score"] for r in eng_50) / len(eng_50)
        print(f"  50-turn engine: Turn 25 = {mid_avg:.1f}/5, Turn 50 = {fin_avg:.1f}/5")
        if fin_avg >= mid_avg - 0.5:
            print(f"  → Engine maintains recall over extended sessions")
        else:
            print(f"  → Engine loses {mid_avg - fin_avg:.1f} points over extended sessions")

    print()


if __name__ == "__main__":
    main()
