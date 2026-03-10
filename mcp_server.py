"""MCP server exposing the Context Engine for Claude Code.

Tools:
    store_chunk  — Append a chunk to the context log
    get_context  — Retrieve current context window
    compact_now  — Force compaction immediately
    get_decisions — Audit trail of compaction decisions
    set_goal     — Set the scoring goal for chunk prioritization

Run:
    python mcp_server.py                              # stdio (default for Claude Code)
    python mcp_server.py --transport sse --port 8000  # SSE for debugging

Configure in Claude Code:
    See README or add to ~/.claude.json mcpServers.
"""

from __future__ import annotations

import os
from typing import Annotated

from fastmcp import FastMCP

from engine import ChunkLog, extract_keywords

# --- Server setup ---

mcp = FastMCP(
    "Context Engine",
    instructions=(
        "Context engine with append-only logging, goal-guided scoring, "
        "and automatic compaction. Store conversation chunks, retrieve "
        "prioritized context, and inspect compaction decisions."
    ),
)

# --- Singleton ChunkLog ---

_chunk_log: ChunkLog | None = None


def _get_log() -> ChunkLog:
    """Lazy-init a ChunkLog from environment variables."""
    global _chunk_log
    if _chunk_log is not None:
        return _chunk_log

    db_path = os.environ.get("CONTEXT_DB_PATH", "context.db")
    max_tokens = int(os.environ.get("CONTEXT_MAX_TOKENS", "128000"))
    soft_threshold = float(os.environ.get("CONTEXT_SOFT_THRESHOLD", "0.7"))
    hard_threshold = float(os.environ.get("CONTEXT_HARD_THRESHOLD", "0.9"))
    scoring_mode = os.environ.get("CONTEXT_SCORING_MODE", "tfidf")

    _chunk_log = ChunkLog(
        db_path=db_path,
        max_tokens=max_tokens,
        soft_threshold=soft_threshold,
        hard_threshold=hard_threshold,
        scoring_mode=scoring_mode,
    )
    return _chunk_log


def reset_log(log: ChunkLog | None = None) -> None:
    """Replace the singleton (for testing)."""
    global _chunk_log
    _chunk_log = log


# --- Core logic (plain functions, testable) ---

_VALID_ROLES = ("user", "assistant", "system")


def do_store_chunk(role: str, content: str, priority: float = 1.0) -> dict:
    if role not in _VALID_ROLES:
        return {"error": f"Invalid role '{role}'. Must be 'user', 'assistant', or 'system'."}
    if not content:
        return {"error": "Content must not be empty."}
    priority = max(0.5, min(2.0, priority))

    log = _get_log()
    chunk_hash = log.append(role, content, priority=priority)
    return {
        "chunk_hash": chunk_hash,
        "tokens": log.current_tokens(),
        "turn": log.turn(),
    }


def do_get_context(max_chunks: int = 0) -> dict:
    log = _get_log()
    messages = log.get_context()
    total = len(messages)
    if max_chunks > 0:
        messages = messages[-max_chunks:]
    return {
        "messages": messages,
        "total_chunks": total,
        "tokens": log.current_tokens(),
        "turn": log.turn(),
    }


def do_compact_now() -> dict:
    log = _get_log()
    tokens_before = log.current_tokens()
    count_before = log.compaction_count

    orig_soft = log.soft_threshold
    orig_hard = log.hard_threshold
    log.soft_threshold = 0.0
    log.hard_threshold = 0.0
    log._maybe_compact()
    log.soft_threshold = orig_soft
    log.hard_threshold = orig_hard

    tokens_after = log.current_tokens()
    return {
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_freed": tokens_before - tokens_after,
        "compaction_events": log.compaction_count - count_before,
    }


def do_get_decisions(limit: int = 20, action_filter: str = "all") -> dict:
    log = _get_log()
    decisions = log.decisions
    if action_filter != "all":
        decisions = [d for d in decisions if d.action == action_filter]
    recent = decisions[-limit:] if limit > 0 else decisions
    return {
        "decisions": [
            {
                "timestamp": d.timestamp,
                "action": d.action,
                "chunk_hash": d.chunk_hash[:12],
                "reason": d.reason,
                "context_size_before": d.context_size_before,
                "context_size_after": d.context_size_after,
            }
            for d in recent
        ],
        "total_decisions": len(log.decisions),
    }


def do_set_goal(goal: str) -> dict:
    if not goal:
        return {"error": "Goal must not be empty."}
    log = _get_log()
    log._last_user_message = goal
    log._accumulated_keywords.update(extract_keywords(goal))
    return {
        "goal": goal,
        "keywords_extracted": len(log._accumulated_keywords),
        "scoring_mode": log.scoring_mode or (
            "goal_guided" if log.goal_guided
            else "auto" if log.auto_priority
            else "none"
        ),
    }


# --- MCP tool wrappers ---


@mcp.tool()
def store_chunk(
    role: Annotated[str, "Message role: 'user', 'assistant', or 'system'"],
    content: Annotated[str, "The text content to store"],
    priority: Annotated[float, "Priority weight 0.5-2.0 (default 1.0)"] = 1.0,
) -> dict:
    """Store a chunk in the context log. Returns the chunk hash and current token count."""
    return do_store_chunk(role, content, priority)


@mcp.tool()
def get_context(
    max_chunks: Annotated[int, "Maximum number of chunks to return (0 = all)"] = 0,
) -> dict:
    """Retrieve the current context window as ordered messages."""
    return do_get_context(max_chunks)


@mcp.tool()
def compact_now() -> dict:
    """Force compaction immediately, regardless of threshold."""
    return do_compact_now()


@mcp.tool()
def get_decisions(
    limit: Annotated[int, "Number of recent decisions to return (default 20)"] = 20,
    action_filter: Annotated[str, "Filter by action type: 'append', 'compact_soft', 'compact_hard', or 'all'"] = "all",
) -> dict:
    """Get the audit trail of context engine decisions."""
    return do_get_decisions(limit, action_filter)


@mcp.tool()
def set_goal(
    goal: Annotated[str, "The goal or query to prioritize chunks against"],
) -> dict:
    """Set the scoring goal. Chunks are re-scored on next compaction based on this goal."""
    return do_set_goal(goal)


# --- Entrypoint ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Context Engine MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run(transport="stdio")
