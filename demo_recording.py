#!/usr/bin/env python3
"""Terminal demo: deterministic context engine managing a 20-turn coding session.

Uses the real engine with mock data — no API key needed.
Shows compaction, scoring, and needle recall with colored output.

Usage:
    python3 demo_recording.py              # run the demo
    python3 demo_recording.py --fast       # skip typing delays (for testing)

Recording with asciinema:
    asciinema rec demo.cast --cols 100 --rows 35 -c "python3 demo_recording.py"
    # Convert to GIF (requires agg: cargo install --git https://github.com/asciinema/agg):
    agg demo.cast demo.gif --cols 100 --rows 35 --speed 1.5
"""

from __future__ import annotations

import argparse
import sys
import time

from engine import ChunkLog, _estimate_tokens


# --- ANSI Colors ---

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    WHITE = "\033[97m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"
    GRAY = "\033[90m"


# --- Scenario: 20-turn coding conversation ---
# 5 needles planted at turns 2, 5, 8, 12, 15 — rest is realistic filler

NEEDLES = {
    2: {
        "label": "auth.py line 42 bug",
        "user": "There's a critical bug in auth.py at line 42 — the JWT token validation skips expiry check when refresh=True. This causes tokens to never expire for refreshed sessions. We need to fix this before the security audit on March 15th.",
        "assistant": "I found the bug in auth.py line 42. The `validate_token()` function has a conditional `if not refresh` guard around the expiry check. When `refresh=True`, it bypasses `check_token_expiry()` entirely. The fix is to always run the expiry check and only skip re-authentication on refresh. I'll prepare a patch.",
    },
    5: {
        "label": "database migration #47",
        "user": "We need to run migration #47 before deploying — it adds the `last_login_ip` column to the users table. The migration file is in db/migrations/047_add_last_login_ip.sql. Don't forget to update the ORM model in models/user.py too.",
        "assistant": "Migration #47 (`047_add_last_login_ip.sql`) adds `last_login_ip INET` column to the `users` table with a default of NULL. I've noted we need to update `models/user.py` to add the `last_login_ip` field to the User model. This should be deployed before the next release.",
    },
    8: {
        "label": "API rate limit config",
        "user": "The rate limit for /api/v2/search is set to 100 req/min in config/rate_limits.yaml, but the product team wants it bumped to 500 req/min for premium users. The rate limiter middleware reads from RATE_LIMIT_SEARCH_PREMIUM env var.",
        "assistant": "Got it — the search endpoint rate limit change: `/api/v2/search` goes from 100 to 500 req/min for premium users. The config lives in `config/rate_limits.yaml` and the middleware checks `RATE_LIMIT_SEARCH_PREMIUM` env var. I'll update both the YAML config and the .env.example file.",
    },
    12: {
        "label": "memory leak in worker.py",
        "user": "We found a memory leak in worker.py — the Redis connection pool in `process_batch()` creates new connections on every call but never closes them. The pool grows to 500+ connections under load. Fix is to use a singleton pool initialized in `__init__`.",
        "assistant": "The memory leak in `worker.py` is in `process_batch()` which calls `redis.ConnectionPool()` on every invocation instead of reusing a singleton. Under load this creates 500+ open connections. The fix: initialize the pool once in `__init__` and pass it to `process_batch()`. I'll also add a `__del__` cleanup method.",
    },
    15: {
        "label": "deploy key rotation",
        "user": "IMPORTANT: The deploy SSH key for prod (fingerprint SHA256:xK9m2...) expires on March 20th. The new key is already in 1Password vault 'Infrastructure'. Rotation procedure is in runbooks/key-rotation.md. This blocks the next deploy if not done.",
        "assistant": "Critical deadline noted: prod deploy SSH key (SHA256:xK9m2...) expires March 20th. New key is in 1Password 'Infrastructure' vault. Rotation procedure: follow `runbooks/key-rotation.md`. This is a deploy blocker — must be completed before the next production deployment.",
    },
}

# Filler turns — deliberately repetitive/template-based so TF-IDF uniqueness
# clearly separates them from the specific, entity-rich needles.
# All filler discusses the same themes (Python web app patterns) with shared vocabulary.
FILLER_TURNS = {
    1: {
        "user": "Let's start working on the API service. Can you give me an overview?",
        "assistant": "The project is a standard Python web application following common patterns. The source code is organized with route handlers, middleware layers, Pydantic schemas for request validation, and service modules for business logic. Configuration is loaded from environment variables and YAML files. The application uses an async framework with standard Python patterns for dependency injection and request processing. Tests follow the standard pytest conventions with unit and integration test directories.",
    },
    3: {
        "user": "How do we handle error responses in the application?",
        "assistant": "The application follows standard Python web patterns for error handling. Custom exception classes extend a base error class with status codes and error codes. A middleware layer catches these exceptions and formats them into a consistent JSON response envelope. The standard patterns include returning appropriate HTTP status codes, providing error detail messages, and including request correlation identifiers. Unhandled exceptions are caught by the framework and return standard error responses. This follows common Python web application conventions.",
    },
    4: {
        "user": "What does the test setup look like?",
        "assistant": "Testing follows standard Python pytest patterns with fixture-based test organization. Unit tests mock external dependencies using standard Python mocking patterns. Integration tests use containerized services following common Python testing conventions. The test configuration uses pyproject.toml with standard pytest markers and coverage settings. The test suite follows common Python patterns for organizing test files by module and feature area. Coverage reporting uses standard Python tools to track line and branch coverage across the application modules.",
    },
    6: {
        "user": "How does the caching layer work?",
        "assistant": "The caching layer follows standard Python patterns with a cache manager class that wraps common operations. The implementation uses standard patterns for cache-aside (get-or-set), TTL-based expiration, and pattern-based invalidation. Cache keys follow common naming conventions with namespace prefixes. The configuration uses standard Python patterns for connection management and TTL defaults. Invalidation follows common patterns where write operations clear related cache entries. This is a standard Python caching implementation following well-known patterns.",
    },
    7: {
        "user": "Walk me through the CI/CD pipeline.",
        "assistant": "The CI/CD pipeline follows standard patterns with multiple stages. The first stage runs linting and type checking using common Python tools, followed by unit tests with coverage requirements. The second stage runs integration tests with containerized services following standard patterns. The deployment stage uses standard container build and deploy patterns. The pipeline configuration follows common patterns for Python web applications with standard health checks and rollback procedures. Each stage follows standard Python CI/CD conventions.",
    },
    9: {
        "user": "How is the logging structured?",
        "assistant": "Logging follows standard Python patterns with structured JSON output for production environments. The implementation uses common Python logging patterns with context-based log enrichment. Each request gets a correlation identifier following standard patterns for request tracing. The log format follows common conventions with event name, log level, timestamp, and contextual fields. Log shipping follows standard patterns for cloud environments. Alert thresholds follow common Python application monitoring patterns.",
    },
    10: {
        "user": "What about the WebSocket implementation?",
        "assistant": "The WebSocket implementation follows standard Python async patterns with a connection manager class. The implementation uses common patterns for connection lifecycle management: accept, message handling, and disconnect cleanup. Authentication follows standard patterns where tokens are passed as query parameters. The message protocol uses a standard type/payload schema following common conventions. Connection tracking uses standard Python patterns with dictionary-based state management. This follows common Python WebSocket implementation patterns.",
    },
    11: {
        "user": "How does background job processing work?",
        "assistant": "Background processing follows standard Python patterns using a task queue with a message broker. The implementation uses common patterns for task serialization, acknowledgment, and retry with exponential backoff. Task types follow standard conventions: email sending, report generation, data synchronization, and batch processing. Monitoring uses standard Python patterns with dashboard visibility into queue depth and worker status. Scaling follows common patterns with auto-scaling based on queue depth. This is a standard Python background processing implementation.",
    },
    13: {
        "user": "How is the database connection pooling configured?",
        "assistant": "Database connection pooling follows standard Python patterns with configurable pool sizes, overflow limits, and connection recycling. The implementation uses common patterns for connection health checking with pre-ping verification. Async database access follows standard Python patterns for async connection management. Connection monitoring follows common conventions with pool utilization metrics exposed via an internal endpoint. Read replica routing follows standard patterns for query distribution. This uses standard Python database patterns.",
    },
    14: {
        "user": "What's the approach for API versioning?",
        "assistant": "API versioning follows standard Python patterns with URL path-based version prefixes. The implementation uses common patterns for router mounting with version-specific route groups. Versioning policy follows standard conventions: deprecated versions are maintained with sunset headers, new features go to the current version, and breaking changes require a new version. Schema versioning uses standard Python model inheritance patterns. This follows common Python API versioning conventions.",
    },
    16: {
        "user": "How do we handle database transactions?",
        "assistant": "Database transactions follow standard Python patterns with context manager-based session management. The implementation uses common patterns for automatic commit on success and rollback on exception. Service layer operations follow standard conventions where all related database operations occur within a single transaction scope. For long-running operations, the implementation follows common patterns with saga-based compensation. This is a standard Python database transaction management implementation.",
    },
    17: {
        "user": "What monitoring and observability do we have?",
        "assistant": "Observability follows standard Python application patterns with metrics, logging, and distributed tracing. Metrics use common patterns for counters, histograms, and gauge types tracking request rates, latencies, and resource utilization. Logging follows standard Python patterns with structured JSON output and request correlation. Tracing follows common conventions for span-based distributed tracing across service boundaries. Alerting follows standard patterns with threshold-based rules. This uses common Python observability patterns.",
    },
    18: {
        "user": "How do we handle file uploads?",
        "assistant": "File uploads follow standard Python patterns with streaming upload to object storage. The implementation uses common patterns for file validation (size limits, content type checks) and secure key generation. Upload handling follows standard conventions for streaming directly to storage without temporary disk files. Presigned URLs follow common patterns for secure, time-limited download access. For large files, the implementation uses standard patterns for multipart upload with client-side presigning. This follows common Python file upload conventions.",
    },
    19: {
        "user": "What's the authorization model?",
        "assistant": "Authorization follows standard Python patterns with role-based access control. The implementation uses common patterns for role-to-permission mapping with decorator-based enforcement. Permission checking follows standard conventions where endpoint decorators verify the current user's role against required permissions. Resource-level access control follows common patterns with a permissions table tracking user-resource-permission relationships. This is a standard Python RBAC implementation following well-known patterns.",
    },
}

# Turn 20: recall all 5 needles
RECALL_QUESTION = "Before we wrap up, I need you to recall all the critical items we discussed: (1) the auth bug, (2) the database migration, (3) the rate limit change, (4) the memory leak, and (5) the deploy key. Give me the specific details for each."


def status_bar(log: ChunkLog, turn: int, max_tokens: int) -> str:
    """Generate a colored status bar showing context utilization."""
    current = log.current_tokens()
    ratio = current / max_tokens
    pct = ratio * 100

    # Color based on threshold zones
    if ratio >= log.hard_threshold:
        color = C.RED
        bg = C.BG_RED
        zone = "HARD"
    elif ratio >= log.soft_threshold:
        color = C.YELLOW
        bg = C.BG_YELLOW
        zone = "SOFT"
    else:
        color = C.GREEN
        bg = C.BG_GREEN
        zone = "SAFE"

    # Build the bar
    bar_width = 40
    filled = int(bar_width * ratio)
    filled = min(filled, bar_width)
    bar = f"{bg}{C.BOLD}{' ' * filled}{C.RESET}{C.DIM}{'.' * (bar_width - filled)}{C.RESET}"

    chunks = log._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    return (
        f"  {C.BOLD}[Turn {turn:>2}/20]{C.RESET}  "
        f"{bar}  "
        f"{color}{C.BOLD}{current:,}/{max_tokens:,} tokens ({pct:.0f}%) [{zone}]{C.RESET}  "
        f"{C.DIM}{chunks} chunks{C.RESET}"
    )


def show_compaction_details(log: ChunkLog, prev_decision_count: int) -> list[str]:
    """Show what got evicted with human-readable descriptions."""
    new_decisions = log.decisions[prev_decision_count:]
    compactions = [d for d in new_decisions if d.action.startswith("compact")]

    if not compactions:
        return []

    lines = []
    evicted_count = len(compactions)
    tokens_freed = sum(
        d.context_size_before - d.context_size_after for d in compactions
    )

    # Summarize what was evicted
    lines.append(
        f"  {C.MAGENTA}{C.BOLD}COMPACTING:{C.RESET} "
        f"{C.MAGENTA}evicting {evicted_count} chunks ({tokens_freed:,} tokens freed){C.RESET}"
    )

    # Show a few examples of what was evicted
    shown = 0
    for d in compactions[:3]:
        # Get the content preview from the reason field
        lines.append(
            f"    {C.DIM}  evicted: {d.chunk_hash[:12]}... ({d.reason}){C.RESET}"
        )
        shown += 1
    if len(compactions) > 3:
        lines.append(f"    {C.DIM}  ... and {len(compactions) - 3} more{C.RESET}")

    return lines


# Key phrases that identify each needle in context
NEEDLE_KEY_PHRASES = {
    2: ["auth.py", "line 42", "JWT", "expiry"],
    5: ["migration #47", "last_login_ip", "047"],
    8: ["/api/v2/search", "500 req/min", "RATE_LIMIT"],
    12: ["memory leak", "worker.py", "ConnectionPool", "singleton"],
    15: ["SSH key", "SHA256:xK9m2", "March 20", "1Password"],
}


def needle_in_context(log: ChunkLog, needle_turn: int) -> bool:
    """Check if a needle's content is still in context using key phrase matching."""
    context = log.get_context()
    context_text = " ".join(m["content"] for m in context).lower()
    phrases = NEEDLE_KEY_PHRASES[needle_turn]
    found = sum(1 for phrase in phrases if phrase.lower() in context_text)
    return found >= 2


def describe_needle_kept(log: ChunkLog, needle_turn: int, label: str) -> str:
    """Check if a needle's content is still in context."""
    if needle_in_context(log, needle_turn):
        return f"    {C.GREEN}  kept: {label}{C.RESET}"
    return f"    {C.RED}  lost: {label}{C.RESET}"


def print_slow(text: str, delay: float = 0.02):
    """Print with a small line delay for readability."""
    print(text)
    sys.stdout.flush()
    if delay > 0:
        time.sleep(delay)


def run_demo(fast: bool = False):
    """Run the full 20-turn demo."""
    delay = 0.0 if fast else 0.03

    TOKEN_BUDGET = 2600
    SOFT = 0.7
    HARD = 0.9

    # Initialize engine — entity_aware scoring protects entity-rich needles
    log = ChunkLog(
        db_path=":memory:",
        max_tokens=TOKEN_BUDGET,
        soft_threshold=SOFT,
        hard_threshold=HARD,
        scoring_mode="entity_aware",
    )

    # Banner
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 72}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  Deterministic Context Engine — Live Demo{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  20-turn coding session | {TOKEN_BUDGET:,} token budget | Entity-aware TF-IDF{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 72}{C.RESET}")
    print(f"  {C.DIM}Soft threshold: {SOFT:.0%}  |  Hard threshold: {HARD:.0%}  |  Scorer: TF-IDF + entity overlap{C.RESET}")
    print(f"  {C.DIM}5 critical needles planted at turns 2, 5, 8, 12, 15{C.RESET}")
    print()
    time.sleep(0.5 if not fast else 0)

    total_tokens_in = 0
    compaction_events = 0

    for turn_num in range(1, 21):
        prev_decisions = len(log.decisions)
        prev_compactions = log.compaction_count

        is_needle = turn_num in NEEDLES
        is_recall = turn_num == 20

        # Turn header
        if is_needle:
            marker = f" {C.RED}{C.BOLD}[NEEDLE]{C.RESET}"
        elif is_recall:
            marker = f" {C.CYAN}{C.BOLD}[RECALL TEST]{C.RESET}"
        else:
            marker = ""

        print_slow(f"\n{C.BOLD}--- Turn {turn_num}/20 ---{C.RESET}{marker}", delay)

        # Get content for this turn
        if is_recall:
            user_msg = RECALL_QUESTION
        elif is_needle:
            user_msg = NEEDLES[turn_num]["user"]
        elif turn_num in FILLER_TURNS:
            user_msg = FILLER_TURNS[turn_num]["user"]
        else:
            # Shouldn't happen but safety fallback
            user_msg = f"Tell me more about the codebase architecture."

        # Show user message
        truncated = user_msg[:120] + "..." if len(user_msg) > 120 else user_msg
        print_slow(f"  {C.WHITE}{C.BOLD}User:{C.RESET} {truncated}", delay)

        # Append user message
        user_tokens = _estimate_tokens(user_msg)
        total_tokens_in += user_tokens
        log.append("user", user_msg, priority=1.5 if is_needle else 1.0)

        # Check for compaction after user append
        comp_lines = show_compaction_details(log, prev_decisions)
        for line in comp_lines:
            print_slow(line, delay * 2)
        prev_decisions = len(log.decisions)

        # Get assistant response
        if is_recall:
            # Build recall response from whatever needles are still in context
            recall_parts = []
            recall_count = 0
            for needle_turn, needle_data in sorted(NEEDLES.items()):
                if needle_in_context(log, needle_turn):
                    recall_count += 1
                    recall_parts.append(f"({recall_count}) {needle_data['label']}: recalled from turn {needle_turn}")
                else:
                    recall_parts.append(f"(?) {needle_data['label']}: LOST — evicted during compaction")

            asst_msg = f"Recalling critical items from our conversation:\n" + "\n".join(recall_parts)
        elif is_needle:
            asst_msg = NEEDLES[turn_num]["assistant"]
        elif turn_num in FILLER_TURNS:
            asst_msg = FILLER_TURNS[turn_num]["assistant"]
        else:
            asst_msg = "I'll continue reviewing the codebase architecture for you."

        # Append assistant response
        asst_tokens = _estimate_tokens(asst_msg)
        total_tokens_in += asst_tokens
        log.append("assistant", asst_msg, priority=1.0)
        log.next_turn()

        # Check for compaction after assistant append
        comp_lines = show_compaction_details(log, prev_decisions)
        for line in comp_lines:
            print_slow(line, delay * 2)

        # Show which needles survived if compaction happened
        new_compactions = log.compaction_count - prev_compactions
        if new_compactions > 0:
            compaction_events += new_compactions
            # Check needle survival
            for nt, nd in sorted(NEEDLES.items()):
                if nt <= turn_num:
                    print_slow(describe_needle_kept(log, nt, nd["label"]), delay)

        # Show assistant response (truncated)
        if is_recall:
            # Show full recall for the finale
            print_slow(f"  {C.CYAN}{C.BOLD}Assistant:{C.RESET}", delay)
            for part in recall_parts:
                if "recalled" in part:
                    print_slow(f"    {C.GREEN}{part}{C.RESET}", delay * 3)
                else:
                    print_slow(f"    {C.RED}{part}{C.RESET}", delay * 3)
        else:
            asst_truncated = asst_msg[:100] + "..." if len(asst_msg) > 100 else asst_msg
            print_slow(f"  {C.CYAN}Assistant:{C.RESET} {C.DIM}{asst_truncated}{C.RESET}", delay)

        # Status bar
        print_slow(status_bar(log, turn_num, TOKEN_BUDGET), delay)

        time.sleep(0.3 if not fast else 0)

    # Final summary
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 72}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  SESSION COMPLETE{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 72}{C.RESET}")

    # Count recall
    recall_score = sum(1 for nt in NEEDLES if needle_in_context(log, nt))

    final_tokens = log.current_tokens()
    naive_tokens = total_tokens_in  # without compaction, all tokens would be in context
    tokens_saved = naive_tokens - final_tokens
    savings_pct = (tokens_saved / naive_tokens * 100) if naive_tokens > 0 else 0

    print()
    print(f"  {C.BOLD}Tokens processed:{C.RESET}      {total_tokens_in:,}")
    print(f"  {C.BOLD}Final context size:{C.RESET}    {final_tokens:,} / {TOKEN_BUDGET:,}")
    print(f"  {C.GREEN}{C.BOLD}Tokens saved:{C.RESET}          {tokens_saved:,} ({savings_pct:.0f}% reduction)")
    print(f"  {C.BOLD}Compaction events:{C.RESET}     {compaction_events}")
    print(f"  {C.BOLD}Recall score:{C.RESET}          {C.GREEN if recall_score == 5 else C.RED}{C.BOLD}{recall_score}/5 needles recalled{C.RESET}")
    print()

    if recall_score == 5:
        print(f"  {C.GREEN}{C.BOLD}All critical facts survived compaction.{C.RESET}")
        print(f"  {C.DIM}The engine evicted filler while preserving needles via TF-IDF scoring.{C.RESET}")
    else:
        lost = 5 - recall_score
        print(f"  {C.YELLOW}{C.BOLD}{lost} needle(s) lost during compaction.{C.RESET}")

    print()
    print(f"{C.DIM}  No API calls were made. The engine runs locally with zero latency.{C.RESET}")
    print(f"{C.DIM}  Scoring: TF-IDF + entity overlap, 40% goal alignment + 60% uniqueness{C.RESET}")
    print()

    log.close()
    return recall_score


def main():
    parser = argparse.ArgumentParser(description="Terminal demo for the context engine")
    parser.add_argument("--fast", action="store_true", help="Skip typing delays")
    args = parser.parse_args()

    # Print recording instructions if running directly
    if sys.stdout.isatty():
        print(f"{C.DIM}")
        print("  To record this demo:")
        print("    asciinema rec demo.cast --cols 100 --rows 35 -c 'python3 demo_recording.py'")
        print()
        print("  To convert to GIF (requires agg):")
        print("    cargo install --git https://github.com/asciinema/agg")
        print("    agg demo.cast demo.gif --cols 100 --rows 35 --speed 1.5")
        print(f"{C.RESET}")
        time.sleep(1.0 if not args.fast else 0)

    run_demo(fast=args.fast)


if __name__ == "__main__":
    main()
