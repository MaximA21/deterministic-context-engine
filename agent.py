#!/usr/bin/env python3
"""Interactive coding agent powered by the deterministic context engine.

Uses ChunkLog with goal_guided TF-IDF scoring to manage context while
talking to Cerebras llama3.1-8b about a codebase.

Usage:
    python agent.py /path/to/repo              # interactive mode
    python agent.py /path/to/repo --replay     # dump DecisionRecord history
    python agent.py /path/to/repo --budget 16000  # custom token budget
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from engine import ChunkLog, CerebrasSession, DecisionRecord, _estimate_tokens


# --- Repo indexing ---

# Extensions worth reading for code understanding
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".rb", ".sh", ".sql", ".toml", ".yaml", ".yml", ".json", ".md",
    ".cfg", ".ini", ".env.example", ".html", ".css", ".proto", ".graphql",
}

# Files to skip
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".tox",
    "dist", "build", ".eggs", ".mypy_cache", ".pytest_cache",
    ".next", ".nuxt", "target", "vendor",
}

_MAX_FILE_TOKENS = 2000  # Don't ingest files larger than this
_MAX_FILES = 50          # Cap on number of files to read


def index_repo(repo_path: Path) -> list[dict[str, str]]:
    """Walk a repo and return a list of {path, content} for key files."""
    files: list[dict[str, str]] = []

    # Collect candidate files
    candidates: list[Path] = []
    for root, dirs, filenames in os.walk(repo_path):
        # Prune skip dirs
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in filenames:
            fpath = Path(root) / fname
            if fpath.suffix in _CODE_EXTENSIONS or fname in (
                "Makefile", "Dockerfile", "Cargo.toml", "go.mod",
                "requirements.txt", "package.json", "pyproject.toml",
                "CLAUDE.md", "AGENTS.md",
            ):
                candidates.append(fpath)

    # Sort: config/meta files first, then by size (smaller = more likely important)
    priority_names = {
        "README.md", "CLAUDE.md", "AGENTS.md", "pyproject.toml",
        "package.json", "Cargo.toml", "go.mod", "requirements.txt",
        "Makefile", "Dockerfile",
    }

    def sort_key(p: Path) -> tuple[int, int]:
        is_priority = 0 if p.name in priority_names else 1
        try:
            size = p.stat().st_size
        except OSError:
            size = 999999
        return (is_priority, size)

    candidates.sort(key=sort_key)

    for fpath in candidates[:_MAX_FILES]:
        try:
            content = fpath.read_text(errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        tokens = _estimate_tokens(content)
        if tokens > _MAX_FILE_TOKENS:
            # Truncate large files
            content = content[: _MAX_FILE_TOKENS * 4] + f"\n\n[... truncated, {tokens} tokens total]"
        rel = fpath.relative_to(repo_path)
        files.append({"path": str(rel), "content": content})

    return files


def build_file_tree(repo_path: Path) -> str:
    """Build a compact file tree string for the system prompt."""
    lines: list[str] = []
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        level = len(Path(root).relative_to(repo_path).parts)
        indent = "  " * level
        if level > 0:
            lines.append(f"{indent}{Path(root).name}/")
        for fname in sorted(filenames):
            fpath = Path(root) / fname
            if fpath.suffix in _CODE_EXTENSIONS or fname in (
                "Makefile", "Dockerfile", "Cargo.toml", "go.mod",
                "requirements.txt", "package.json", "pyproject.toml",
            ):
                try:
                    size = fpath.stat().st_size
                except OSError:
                    size = 0
                lines.append(f"{indent}  {fname} ({size}b)")
        if len(lines) > 200:
            lines.append("  ... (tree truncated)")
            break
    return "\n".join(lines)


# --- Status display ---

def format_status(log: ChunkLog, session: CerebrasSession, max_tokens: int) -> str:
    """One-line status bar."""
    ctx_tokens = log.current_tokens()
    chunk_count = log._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    evicted = sum(1 for d in log.decisions if d.action.startswith("compact"))
    turn = log.turn()
    ttft = f"{session.avg_ttft:.2f}s" if session.avg_ttft > 0 else "-"

    return (
        f"[Turn {turn}] Context: {ctx_tokens:,}/{max_tokens:,} tok "
        f"| Chunks: {chunk_count} active, {evicted} evicted "
        f"| Compactions: {log.compaction_count} "
        f"| TTFT: {ttft}"
    )


def print_compaction_detail(log: ChunkLog, prev_decision_count: int) -> None:
    """Print details of any new compaction events since last check."""
    new_decisions = log.decisions[prev_decision_count:]
    compactions = [d for d in new_decisions if d.action.startswith("compact")]
    if not compactions:
        return

    print(f"\n  --- Compaction: {len(compactions)} chunk(s) evicted ---")
    for d in compactions[:5]:  # Show at most 5
        print(f"  {d.action}: {d.chunk_hash[:12]}... "
              f"({d.reason}) "
              f"[{d.context_size_before:,} -> {d.context_size_after:,} tok]")
    if len(compactions) > 5:
        print(f"  ... and {len(compactions) - 5} more")


def replay_decisions(log: ChunkLog) -> None:
    """Dump full DecisionRecord history."""
    decisions = log.decisions
    if not decisions:
        print("No decisions recorded.")
        return

    print(f"\n{'='*70}")
    print(f"DECISION HISTORY — {len(decisions)} records")
    print(f"{'='*70}")

    for i, d in enumerate(decisions):
        ts = time.strftime("%H:%M:%S", time.localtime(d.timestamp))
        print(f"  [{i+1:3d}] {ts} {d.action:<14s} {d.chunk_hash[:16]}... "
              f"{d.reason:<40s} {d.context_size_before:>6,} -> {d.context_size_after:>6,} tok")

    # Summary
    appends = sum(1 for d in decisions if d.action == "append")
    evictions = sum(1 for d in decisions if d.action.startswith("compact"))
    print(f"\n  Total: {appends} appends, {evictions} evictions")
    print(f"{'='*70}")


# --- Main agent loop ---

SYSTEM_PROMPT_TEMPLATE = """You are a coding assistant analyzing the repository at {repo_path}.

## File tree
{file_tree}

## Instructions
- Answer questions about the codebase based on the files loaded into context
- Be specific: reference file paths, line numbers, function names
- If you don't have enough context to answer, say so
- Keep responses concise and actionable"""


def main():
    parser = argparse.ArgumentParser(description="Interactive coding agent with context engine")
    parser.add_argument("repo", help="Path to the repository")
    parser.add_argument("--budget", type=int, default=8192,
                        help="Token budget for context window (default: 8192)")
    parser.add_argument("--replay", action="store_true",
                        help="After session, dump full DecisionRecord history")
    parser.add_argument("--soft", type=float, default=0.7,
                        help="Soft compaction threshold (default: 0.7)")
    parser.add_argument("--hard", type=float, default=0.9,
                        help="Hard compaction threshold (default: 0.9)")
    args = parser.parse_args()

    repo_path = Path(args.repo).resolve()
    if not repo_path.is_dir():
        print(f"Error: {repo_path} is not a directory")
        sys.exit(1)

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("Error: CEREBRAS_API_KEY not set")
        sys.exit(1)

    max_tokens = args.budget

    # Initialize engine
    log = ChunkLog(
        db_path=":memory:",
        max_tokens=max_tokens,
        soft_threshold=args.soft,
        hard_threshold=args.hard,
        goal_guided=True,
    )
    session = CerebrasSession(log, api_key=api_key)

    # Build file tree and system prompt
    print(f"Indexing {repo_path}...")
    file_tree = build_file_tree(repo_path)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        repo_path=repo_path, file_tree=file_tree,
    )
    sys_tokens = _estimate_tokens(system_prompt)
    print(f"  System prompt: {sys_tokens:,} tokens")

    # Index key files into context
    files = index_repo(repo_path)
    print(f"  Indexing {len(files)} files into context...")

    for f in files:
        content = f"[FILE: {f['path']}]\n{f['content']}"
        log.append("user", content, priority=1.0)

    ctx_after_index = log.current_tokens()
    chunks_after_index = log._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"  Context after indexing: {ctx_after_index:,}/{max_tokens:,} tokens ({chunks_after_index} chunks)")

    if log.compaction_count > 0:
        print(f"  Compaction during indexing: {log.compaction_count} events")
        evicted = sum(1 for d in log.decisions if d.action.startswith("compact"))
        print(f"  Evicted {evicted} chunks to fit budget")

    print(f"\nReady. Type your coding questions (Ctrl+D or 'quit' to exit).\n")
    print(f"{'='*60}")

    prev_decisions = len(log.decisions)

    try:
        while True:
            # Prompt
            try:
                user_input = input("\nyou> ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "/replay":
                replay_decisions(log)
                continue
            if user_input.lower() == "/status":
                print(format_status(log, session, max_tokens))
                continue

            # Record previous state
            prev_decisions = len(log.decisions)
            prev_compactions = log.compaction_count

            # Append user message
            log.append("user", user_input, priority=1.5)
            log.next_turn()

            # Show any compaction from user message
            print_compaction_detail(log, prev_decisions)
            prev_decisions = len(log.decisions)

            # Call Cerebras
            print("\nassistant> ", end="", flush=True)
            try:
                t0 = time.time()
                response = session.chat(system_prompt=system_prompt)
                elapsed = time.time() - t0
            except Exception as e:
                print(f"\n[API Error: {e}]")
                continue

            print(response)

            # Show compaction from assistant response
            print_compaction_detail(log, prev_decisions)

            # Status line
            print(f"\n{format_status(log, session, max_tokens)}")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")

    # Session summary
    metrics = session.get_metrics()
    print(f"\n{'='*60}")
    print("SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Turns:           {metrics['total_turns']}")
    print(f"  Total tokens:    {metrics['total_tokens']:,} ({metrics['total_input_tokens']:,} in, {metrics['total_output_tokens']:,} out)")
    print(f"  Avg TTFT:        {metrics['avg_ttft']:.3f}s")
    print(f"  Compactions:     {metrics['compaction_events']}")
    print(f"  Context size:    {metrics['context_size_tokens']:,}/{max_tokens:,} tokens")
    total_evicted = sum(1 for d in log.decisions if d.action.startswith("compact"))
    print(f"  Chunks evicted:  {total_evicted}")
    print(f"{'='*60}")

    # Replay if requested
    if args.replay:
        replay_decisions(log)

    log.close()


if __name__ == "__main__":
    main()
