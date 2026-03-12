#!/usr/bin/env python3
"""Analyze AGENTbench results and compare with published baselines.

Published results from "Evaluating AGENTS.md" (arXiv:2602.11988):

| Setting         | Claude Sonnet 4.5 | Codex (o4-mini) |
|-----------------|-------------------|-----------------|
| NONE (no ctx)   | 38.4%             | 31.9%           |
| LLM (auto-gen)  | 37.0%             | 31.2%           |
| HUMAN (written) | 42.0%             | 33.3%           |

Our engine should beat LLM-generated context (37.0%) and approach
or exceed HUMAN-written context (42.0%).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Published baselines from the paper
BASELINES = {
    "claude_sonnet_4.5": {
        "NONE": 38.4,
        "LLM": 37.0,
        "HUMAN": 42.0,
    },
    "codex_o4_mini": {
        "NONE": 31.9,
        "LLM": 31.2,
        "HUMAN": 33.3,
    },
}


def analyze_report(report_path: str | Path) -> dict[str, Any]:
    """Analyze a report.json from the Docker evaluation."""
    with open(report_path) as f:
        data = json.load(f)

    report = data.get("report", data)
    total = len(report)
    resolved = sum(1 for r in report.values() if r.get("resolved", False))
    inst_passed = sum(1 for r in report.values() if r.get("instance_test_passed", False))
    repo_passed = sum(1 for r in report.values() if r.get("repo_test_passed", False))
    errors = sum(1 for r in report.values() if r.get("error"))

    resolve_rate = 100 * resolved / total if total else 0

    # Per-repo breakdown
    per_repo: dict[str, dict[str, int]] = {}
    for iid, result in report.items():
        repo = iid.rsplit("-", 1)[0].replace("_", "/", 1) if "_" in iid else "unknown"
        if repo not in per_repo:
            per_repo[repo] = {"total": 0, "resolved": 0}
        per_repo[repo]["total"] += 1
        if result.get("resolved", False):
            per_repo[repo]["resolved"] += 1

    analysis = {
        "total": total,
        "resolved": resolved,
        "resolve_rate": resolve_rate,
        "instance_test_passed": inst_passed,
        "repo_test_passed": repo_passed,
        "errors": errors,
        "per_repo": per_repo,
    }

    print(f"{'=' * 60}")
    print(f"AGENTbench Results Analysis")
    print(f"{'=' * 60}")
    print(f"Total instances:       {total}")
    print(f"Resolved:              {resolved} ({resolve_rate:.1f}%)")
    print(f"Instance tests passed: {inst_passed}")
    print(f"Repo tests passed:     {repo_passed}")
    print(f"Errors:                {errors}")

    print(f"\n{'=' * 60}")
    print("Comparison with Published Baselines")
    print(f"{'=' * 60}")
    print(f"{'Setting':<25} {'Resolve Rate':>12}")
    print(f"{'-' * 37}")
    print(f"{'OUR ENGINE':<25} {resolve_rate:>11.1f}%")
    print()
    for model, settings in BASELINES.items():
        print(f"  {model}:")
        for setting, rate in settings.items():
            delta = resolve_rate - rate
            marker = ">>>" if delta > 0 else "   "
            print(f"    {marker} {setting:<19} {rate:>11.1f}%  (delta: {delta:+.1f}%)")

    if per_repo:
        print(f"\n{'=' * 60}")
        print("Per-Repository Breakdown")
        print(f"{'=' * 60}")
        for repo, stats in sorted(per_repo.items(), key=lambda x: -x[1]["resolved"]):
            rate = 100 * stats["resolved"] / stats["total"] if stats["total"] else 0
            print(f"  {repo:<40} {stats['resolved']}/{stats['total']} ({rate:.0f}%)")

    return analysis


def analyze_context_eval(results_path: str | Path) -> dict[str, Any]:
    """Analyze results from the offline context quality evaluation."""
    with open(results_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    per_instance = data.get("per_instance", {})

    print(f"{'=' * 60}")
    print("AGENTbench Context Quality Analysis")
    print(f"{'=' * 60}")
    print(f"Instances: {data.get('n_instances', '?')}")
    print(f"Max tokens: {data.get('max_tokens', '?')}")

    for mode, stats in summary.items():
        print(f"\n--- {mode.upper()} ---")
        print(f"  Key file recall:    {stats['avg_key_file_recall']:.3f}")
        print(f"  Key file precision: {stats['avg_key_file_precision']:.3f}")
        print(f"  Token compression:  {stats['avg_token_compression']:.1f}x")
        print(f"  Priority gap:       {stats['avg_priority_gap']:+.3f}")
        print(f"  Avg scoring time:   {stats['avg_scoring_time_ms']:.1f}ms")
        print(f"  Perfect recall:     {stats['perfect_recall_count']}/{stats['n_instances']}")

        # Comparison with SWE-Pruner's compression
        compression = stats["avg_token_compression"]
        print(f"\n  vs SWE-Pruner (Qwen3-Reranker-0.6B):")
        print(f"    Their compression: ~1.3x (23% reduction on SWE-Bench)")
        print(f"    Our compression:   {compression:.1f}x ({100*(1-1/compression):.0f}% reduction)")
        print(f"    Our latency:       {stats['avg_scoring_time_ms']:.1f}ms (theirs: ~100ms+ for 0.6B model)")
        print(f"    Our model size:    0 params (pure BM25, no GPU)")

    # Per-repo aggregation
    for mode, instances in per_instance.items():
        repos: dict[str, list[float]] = {}
        for inst in instances:
            repo = inst["repo"]
            if repo not in repos:
                repos[repo] = []
            repos[repo].append(inst["key_file_recall"])

        print(f"\n--- {mode.upper()} per-repo recall ---")
        for repo, recalls in sorted(repos.items(), key=lambda x: -sum(x[1]) / len(x[1])):
            avg = sum(recalls) / len(recalls)
            print(f"  {repo:<40} {avg:.3f} ({len(recalls)} instances)")

    return {"summary": summary}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze AGENTbench results")
    parser.add_argument("results", type=str, help="Path to results JSON")
    parser.add_argument("--type", choices=["report", "context"],
                        default="context",
                        help="Result type: 'report' (Docker eval) or 'context' (offline eval)")
    args = parser.parse_args()

    if args.type == "report":
        analyze_report(args.results)
    else:
        analyze_context_eval(args.results)


if __name__ == "__main__":
    main()
