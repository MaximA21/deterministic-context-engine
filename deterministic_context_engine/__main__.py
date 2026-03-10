"""CLI entry point for the Deterministic Context Engine.

Usage:
    python -m deterministic_context_engine serve       # Start MCP server (stdio)
    python -m deterministic_context_engine benchmark   # Run all 4 NIAH benchmarks
    python -m deterministic_context_engine demo        # Run terminal demo session
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable (for engine, agent, benchmarks)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server over stdio."""
    from deterministic_context_engine.mcp_server import run_server

    run_server(
        max_tokens=args.max_tokens,
        soft_threshold=args.soft,
        hard_threshold=args.hard,
        scoring_mode=args.scoring,
    )


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run all 4 core NIAH benchmarks."""
    from benchmarks import niah_dense, niah_adversarial, niah_goalguided, benchmark_50turn

    suites = [
        ("niah_dense", niah_dense),
        ("niah_adversarial", niah_adversarial),
        ("niah_goalguided", niah_goalguided),
        ("benchmark_50turn", benchmark_50turn),
    ]

    failed = []
    for name, module in suites:
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}\n")
        try:
            module.main()
        except SystemExit:
            pass  # benchmarks may call sys.exit(0)
        except Exception as exc:
            print(f"\n  FAILED: {name} — {exc}")
            failed.append(name)

    print(f"\n{'='*60}")
    if failed:
        print(f"  Finished with errors: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("  All 4 benchmarks completed successfully.")
    print(f"{'='*60}")


def cmd_demo(args: argparse.Namespace) -> None:
    """Run the terminal demo session."""
    # Inject --mock flag into sys.argv so demo_session's argparse picks it up
    demo_argv = []
    if args.mock:
        demo_argv.append("--mock")

    sys.argv = ["demo_session"] + demo_argv

    from demo_session import main as demo_main

    demo_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m deterministic_context_engine",
        description="Deterministic Context Engine CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- serve ---
    serve_p = sub.add_parser("serve", help="Start MCP server (stdio)")
    serve_p.add_argument(
        "--max-tokens", type=int, default=128_000,
        help="Maximum context window tokens (default: 128000)",
    )
    serve_p.add_argument(
        "--soft", type=float, default=0.7,
        help="Soft compaction threshold (default: 0.7)",
    )
    serve_p.add_argument(
        "--hard", type=float, default=0.9,
        help="Hard compaction threshold (default: 0.9)",
    )
    serve_p.add_argument(
        "--scoring", choices=["tfidf", "semantic", "hybrid", "entity_aware"],
        default="tfidf",
        help="Scoring mode (default: tfidf)",
    )
    serve_p.set_defaults(func=cmd_serve)

    # --- benchmark ---
    bench_p = sub.add_parser("benchmark", help="Run all 4 NIAH benchmarks")
    bench_p.set_defaults(func=cmd_benchmark)

    # --- demo ---
    demo_p = sub.add_parser("demo", help="Run terminal demo session")
    demo_p.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM responses (no API key needed)",
    )
    demo_p.set_defaults(func=cmd_demo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
