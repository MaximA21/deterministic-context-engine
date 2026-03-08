#!/usr/bin/env python3
"""Fair Goal-Guided NIAH benchmark — removes length confound from original.

The original benchmark had needles ~244 chars and filler ~1801 chars (7.4x longer).
TF-IDF uniqueness naturally favored shorter documents. This benchmark fixes that:

1. All chunks ~450-550 chars (needles padded, filler shortened)
2. 100 unique hard-negative fillers (no template recycling)
3. Each filler has its own specific details (line numbers, names, metrics)
4. Same keyword overlap (same filenames/functions)

If Goal-Guided still scores 5.0/5 here, it's real.
If it drops to naive-level, the original result was a length artifact.
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

from engine import ChunkLog, _estimate_tokens, GoalGuidedScorer

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_CONTEXT_TOKENS = 8000
SYSTEM_PROMPT = "You are a helpful assistant with perfect memory. Answer questions based on the conversation history provided."
SYSTEM_PROMPT_TOKENS = _estimate_tokens(SYSTEM_PROMPT)

# --- Padded needles (~500 chars each to match filler length) ---
NEEDLES = [
    {
        "id": "needle_1",
        "fact": (
            "CRITICAL BUG: auth.py line 42, function validate_token() has an off-by-one error. "
            "Token expiry check uses `<=` instead of `<`, causing tokens to be valid for one extra second. "
            "Fix: change line 42 from `if expires <= now` to `if expires < now`. "
            "This was discovered during Tuesday's security audit by the platform reliability team. "
            "Impact assessment: low user impact currently but it weakens token lifecycle guarantees. "
            "Priority P1, assign to the auth team for the next sprint. Tracking in JIRA-4821."
        ),
        "keyword": "off-by-one",
    },
    {
        "id": "needle_2",
        "fact": (
            "URGENT: database.py function migrate_schema() will corrupt data if run on tables with more than "
            "10 million rows. The batch_size on line 187 must be changed from 100000 to 10000. "
            "Schedule the fix for March 15th 3am UTC maintenance window. "
            "The corruption manifests as truncated varchar columns in the migrated table. "
            "QA discovered this during load testing with synthetic datasets last Wednesday. "
            "Rollback procedure documented in runbook section 7.3. Contact the DBA team before deploying."
        ),
        "keyword": "10 million",
    },
    {
        "id": "needle_3",
        "fact": (
            "SECURITY ALERT: api_gateway.py class RateLimiter has a bypass vulnerability. "
            "Function check_rate() on line 93 doesn't validate the X-Forwarded-For header, "
            "allowing attackers to spoof IPs. Rate limit must change from 100 to 250 req/min after patching. "
            "The vulnerability was reported through our bug bounty program by researcher handle @netguard. "
            "CVE assignment pending. Temporary mitigation: enable strict proxy trust in nginx config. "
            "Full patch requires updating the IP extraction logic in the middleware chain."
        ),
        "keyword": "x-forwarded-for",
    },
    {
        "id": "needle_4",
        "fact": (
            "ACTION REQUIRED: New hire Elena Vasquez (senior engineer) starts Monday. She needs admin access "
            "to deploy_pipeline.py and must review the Exception handling in provision_access() before her "
            "first deploy. Her employee ID is EV-2847. "
            "She is transferring from the Barcelona office where she led the payments platform migration. "
            "Her manager is Carlos in the infrastructure team. Onboarding buddy assigned: Priya from SRE. "
            "She should complete the security training module before getting production access."
        ),
        "keyword": "ev-2847",
    },
    {
        "id": "needle_5",
        "fact": (
            "PRODUCTION INCIDENT: Server 10.42.88.7 in config.yaml has a memory leak in worker_pool.py "
            "function spawn_workers() line 156. RSS grows 50MB/hour. "
            "Temporary fix: restart cron every 4 hours. Permanent fix requires refactoring the connection "
            "pool in lines 156-180. "
            "The leak was traced to unclosed database connections in the worker initialization path. "
            "Monitoring alert fires when RSS exceeds 2GB. Three incidents in the past week. "
            "The infrastructure team has allocated sprint capacity for the permanent fix next week."
        ),
        "keyword": "50mb/hour",
    },
]

RECALL_QUESTION = (
    "I need you to recall the SPECIFIC ACTIONABLE items from our conversation. "
    "Not general notes — I need the exact details:\n\n"
    "1. What is the specific bug in auth.py's validate_token? What line, what's wrong, and what's the fix?\n"
    "2. What is the data corruption risk in database.py's migrate_schema? What row threshold and what batch_size change?\n"
    "3. What is the security vulnerability in api_gateway.py's RateLimiter? What header is the problem?\n"
    "4. Who is the new hire, what's their employee ID, and what must they review?\n"
    "5. What is the memory leak in worker_pool.py? What server, what's the growth rate, and what's the temp fix?\n\n"
    "Only answer with the SPECIFIC details mentioned during our conversation. Do NOT guess or fabricate details."
)

# --- Hard-negative filler generation ---
# Each filler is ~450-550 chars, mentions same files/functions as needles,
# contains its own SPECIFIC details, but NEVER contains the needle keywords:
# "off-by-one", "10 million", "x-forwarded-for", "ev-2847", "50mb/hour"

_PEOPLE = [
    "Sarah Chen", "Marcus Rodriguez", "Priya Patel", "James Kim", "Lisa Wang",
    "Alex Thompson", "David Lee", "Nina Morales", "Ryan O'Brien", "Emily Fischer",
    "Tom Nguyen", "Aisha Hassan", "Chris Park", "Rachel Green", "Omar Diaz",
]
_ACTIONS = [
    "refactored the error paths", "added structured logging", "improved type annotations",
    "cleaned up dead code", "standardized exception handling", "added retry logic",
    "optimized the hot path", "rewrote the cache layer", "simplified the control flow",
    "extracted helper methods", "added input validation", "improved error messages",
]
_OUTCOMES = [
    "complexity score dropped from 18 to 12", "test pass rate improved to 99.8%",
    "p99 latency reduced by 15%", "memory allocation decreased by 22%",
    "startup time improved by 340ms", "error rate dropped below 0.005%",
    "code coverage increased to 96%", "lint warnings reduced from 14 to 2",
    "binary size reduced by 8KB", "cold start improved by 200ms",
]
_CONTEXTS = [
    "during the quarterly architecture review", "as part of the tech debt sprint",
    "following the post-incident retrospective", "before the compliance audit deadline",
    "during the platform modernization effort", "after the performance regression last week",
    "as part of the observability initiative", "during the security hardening phase",
    "following customer feedback from the enterprise tier", "before the SOC2 certification review",
]

# 20 template functions — 4 per file topic
_FILLER_GENERATORS = [
    # auth.py templates (4)
    lambda rng: (
        f"AUTH REVIEW: auth.py validate_token() line {rng.randint(30, 120)} was {rng.choice(_ACTIONS)} "
        f"by {rng.choice(_PEOPLE)} {rng.choice(_CONTEXTS)}. The token parsing path now separates header "
        f"validation from payload decoding for better error isolation. "
        f"PR #{rng.randint(1200, 1999)} approved with {rng.randint(2, 4)} reviewers after {rng.randint(1, 5)} rounds. "
        f"Result: {rng.choice(_OUTCOMES)}. The auth module maintains its position as one of the most "
        f"well-tested components in the codebase with {rng.randint(85, 98)}% branch coverage."
    ),
    lambda rng: (
        f"AUTH METRICS: auth.py validate_token() benchmarked at {rng.randint(2, 18)}ms p99 latency "
        f"across {rng.randint(50, 200)}K requests in the staging environment. {rng.choice(_PEOPLE)} ran "
        f"the benchmark {rng.choice(_CONTEXTS)} using the updated load testing framework. Token validation "
        f"throughput measured at {rng.randint(8, 45)}K validations per second on a single core. Memory "
        f"footprint stable at {rng.randint(12, 48)}MB RSS during sustained load. The auth team plans "
        f"to add flame graph profiling to identify remaining optimization opportunities in the decode path."
    ),
    lambda rng: (
        f"AUTH TESTING: New integration test suite for auth.py validate_token() written by {rng.choice(_PEOPLE)}. "
        f"Covers {rng.randint(12, 30)} edge cases including expired tokens, malformed headers, "
        f"encoding mismatches, and clock skew scenarios. Test execution time: {rng.randint(2, 8)}s on CI. "
        f"Previously only {rng.randint(5, 11)} test cases existed. The new suite caught {rng.randint(0, 3)} "
        f"minor regressions in the nightly build. Coverage for lines {rng.randint(30, 50)}-{rng.randint(80, 130)} "
        f"increased from {rng.randint(72, 85)}% to {rng.randint(90, 99)}%. Approved in PR #{rng.randint(1200, 1999)}."
    ),
    lambda rng: (
        f"AUTH DOCS: auth.py module documentation updated by {rng.choice(_PEOPLE)} {rng.choice(_CONTEXTS)}. "
        f"The validate_token() function now has comprehensive docstrings following Google style. "
        f"Added sequence diagrams for the token refresh flow and the session revocation process. "
        f"Internal wiki page updated with troubleshooting guide for common auth failures. "
        f"The team also added runbook entries for {rng.randint(3, 7)} operational scenarios including "
        f"token store failover and certificate rotation. Documentation review completed in {rng.randint(2, 5)} days."
    ),
    # database.py templates (4)
    lambda rng: (
        f"DB PERF: database.py migrate_schema() benchmarked with {rng.randint(1, 8)} million test rows "
        f"by {rng.choice(_PEOPLE)} {rng.choice(_CONTEXTS)}. Average migration time: {rng.randint(15, 90)}s "
        f"with index-aware batching enabled. Lock contention measured at {rng.randint(1, 12)}ms per batch. "
        f"The team confirmed no deadlocks in the multi-tenant configuration. Transaction savepoints "
        f"add approximately {rng.randint(3, 15)}% overhead but provide essential rollback capability. "
        f"Next optimization target: reduce WAL file growth during large schema changes."
    ),
    lambda rng: (
        f"DB REFACTOR: database.py migrate_schema() lines {rng.randint(150, 190)}-{rng.randint(200, 240)} "
        f"restructured by {rng.choice(_PEOPLE)}. Separated the DDL generation from the execution phase "
        f"for better testability. Added dry-run mode that validates migrations without applying them. "
        f"The refactoring took {rng.randint(3, 8)} days and touched {rng.randint(12, 30)} functions. "
        f"Result: {rng.choice(_OUTCOMES)}. Integration tests pass against PostgreSQL {rng.choice(['14', '15', '16'])} "
        f"and MySQL {rng.choice(['8.0', '8.1'])}. PR #{rng.randint(1200, 1999)} merged after review."
    ),
    lambda rng: (
        f"DB MONITORING: database.py connection pool metrics reviewed by {rng.choice(_PEOPLE)} "
        f"{rng.choice(_CONTEXTS)}. Pool size: {rng.randint(10, 50)} connections with "
        f"{rng.randint(60, 180)}s idle timeout. migrate_schema() acquires {rng.randint(1, 5)} connections "
        f"during execution. Prometheus dashboards show {rng.randint(95, 99)}.{rng.randint(1, 9)}% connection "
        f"reuse rate. Alert threshold for pool exhaustion set at {rng.randint(80, 95)}% utilization. "
        f"The team added {rng.randint(3, 8)} new Grafana panels for migration-specific metrics. "
        f"No anomalies detected in the past {rng.randint(14, 60)} days of production data."
    ),
    lambda rng: (
        f"DB REVIEW: database.py code quality audit by {rng.choice(_PEOPLE)} {rng.choice(_CONTEXTS)}. "
        f"The migrate_schema() function scored {rng.randint(15, 28)} on cyclomatic complexity — within acceptable "
        f"bounds but flagged for potential simplification. {rng.randint(3, 8)} TODO comments identified "
        f"for cleanup. The function's dependency on psycopg2 {rng.choice(['2.9.7', '2.9.8', '2.9.9'])} is "
        f"pinned and passes vulnerability scans. Type hints added to {rng.randint(8, 20)} function signatures. "
        f"The module's overall technical debt score: {rng.randint(2, 6)}.{rng.randint(0, 9)}%."
    ),
    # api_gateway.py templates (4)
    lambda rng: (
        f"API CONFIG: api_gateway.py RateLimiter configuration updated by {rng.choice(_PEOPLE)} "
        f"{rng.choice(_CONTEXTS)}. Default rate limit adjusted to {rng.randint(50, 300)} requests per minute "
        f"for standard tier. Premium tier set to {rng.randint(500, 2000)} req/min. The check_rate() function "
        f"now uses a {rng.choice(['sliding', 'fixed', 'token bucket'])} window algorithm with "
        f"{rng.randint(1, 5)}s granularity. Redis backend latency: {rng.randint(1, 8)}ms per lookup. "
        f"Config stored in rate_limits.yaml, version-controlled with schema validation."
    ),
    lambda rng: (
        f"API LOADTEST: api_gateway.py RateLimiter load-tested by {rng.choice(_PEOPLE)} with "
        f"{rng.randint(10, 100)}K concurrent connections. check_rate() sustained {rng.randint(20, 80)}K "
        f"lookups per second on the {rng.choice(['staging', 'perf-test', 'pre-prod'])} cluster. "
        f"Memory usage stable at {rng.randint(120, 500)}MB under sustained load. "
        f"Circuit breaker triggered {rng.randint(0, 3)} times during the {rng.randint(2, 8)} hour test window. "
        f"Results documented in the capacity planning spreadsheet. Next benchmark scheduled for "
        f"{rng.choice(['next month', 'Q2', 'after the infrastructure upgrade', 'post-migration'])}."
    ),
    lambda rng: (
        f"API SECURITY: api_gateway.py security audit completed by {rng.choice(_PEOPLE)} "
        f"{rng.choice(_CONTEXTS)}. The RateLimiter class passed {rng.randint(8, 20)} security test cases. "
        f"check_rate() input sanitization verified against OWASP top-{rng.randint(5, 10)} patterns. "
        f"Header validation follows RFC {rng.randint(7230, 7235)} specifications. "
        f"No injection vectors found in the request parsing pipeline. "
        f"The team added {rng.randint(2, 6)} new security regression tests. "
        f"Audit report filed under compliance ticket COMP-{rng.randint(300, 500)}."
    ),
    lambda rng: (
        f"API DOCS: api_gateway.py RateLimiter documentation refreshed by {rng.choice(_PEOPLE)}. "
        f"Added architecture decision records for the rate limiting strategy. The check_rate() function "
        f"has {rng.randint(3, 8)} usage examples in the developer guide. API versioning policy documented "
        f"for clients on v{rng.randint(1, 3)}.{rng.randint(0, 9)} and above. "
        f"Response code semantics clarified: 429 includes Retry-After header with {rng.randint(5, 60)}s backoff. "
        f"The gateway handles {rng.randint(2, 15)}M requests daily with {rng.randint(99, 100)}.{rng.randint(90, 99)}% uptime."
    ),
    # deploy_pipeline.py templates (4)
    lambda rng: (
        f"DEPLOY METRICS: deploy_pipeline.py processed {rng.randint(20, 60)} deployments "
        f"last {rng.choice(['week', 'sprint', 'month'])}. provision_access() handled {rng.randint(50, 300)} "
        f"IAM role assignments with {rng.randint(0, 2)} failures. Average deployment time: "
        f"{rng.randint(4, 18)} minutes. Canary release success rate: {rng.randint(95, 100)}%. "
        f"{rng.choice(_PEOPLE)} reviewed the pipeline {rng.choice(_CONTEXTS)} and recommended "
        f"adding parallel stage execution. Estimated improvement: {rng.randint(15, 40)}% faster deploys. "
        f"Feature flag evaluation adds {rng.randint(50, 200)}ms per deployment stage."
    ),
    lambda rng: (
        f"DEPLOY REFACTOR: deploy_pipeline.py provision_access() rewritten by {rng.choice(_PEOPLE)} "
        f"to support role-based access control with {rng.randint(5, 15)} predefined role templates. "
        f"The new implementation uses STS assume-role with session duration of {rng.randint(1, 8)} hours. "
        f"Exception handling standardized with {rng.randint(3, 7)} custom exception types. "
        f"Rollback capability tested across {rng.randint(3, 6)} AWS regions. "
        f"PR #{rng.randint(1200, 1999)} merged {rng.choice(_CONTEXTS)}. {rng.choice(_OUTCOMES)}."
    ),
    lambda rng: (
        f"DEPLOY RUNBOOK: deploy_pipeline.py operational procedures updated by {rng.choice(_PEOPLE)}. "
        f"Added {rng.randint(5, 12)} new runbook entries covering failure scenarios for provision_access(). "
        f"Blue-green deployment rollback procedure tested in {rng.randint(2, 5)} disaster recovery drills. "
        f"Mean time to recovery: {rng.randint(3, 15)} minutes for standard rollbacks. "
        f"Emergency contact escalation path documented with {rng.randint(3, 6)} tiers. "
        f"The team conducts deployment reviews every {rng.choice(['Monday', 'Wednesday', 'Friday'])} morning."
    ),
    lambda rng: (
        f"DEPLOY AUDIT: deploy_pipeline.py access logs reviewed by {rng.choice(_PEOPLE)} "
        f"{rng.choice(_CONTEXTS)}. provision_access() granted {rng.randint(100, 500)} unique user sessions "
        f"in the past quarter. {rng.randint(0, 5)} access anomalies flagged for investigation — all resolved "
        f"as legitimate. Least-privilege analysis shows {rng.randint(70, 95)}% of roles use fewer than "
        f"{rng.randint(5, 12)} permissions. Token rotation policy enforced every {rng.randint(24, 72)} hours. "
        f"Compliance score: {rng.randint(88, 99)}/100. Next audit scheduled for {rng.choice(['Q2', 'Q3', 'next month'])}."
    ),
    # worker_pool.py / config.yaml templates (4)
    lambda rng: (
        f"WORKER METRICS: worker_pool.py spawn_workers() managing {rng.randint(4, 32)} workers "
        f"on server {rng.randint(1, 50)}.{rng.randint(1, 255)}.{rng.randint(1, 255)}.{rng.randint(1, 255)}. "
        f"Queue depth: {rng.randint(100, 5000)} jobs. Processing rate: {rng.randint(50, 500)} jobs/minute. "
        f"config.yaml worker_count set to {rng.randint(4, 16)} with auto-scaling threshold at "
        f"{rng.randint(60, 90)}% CPU utilization. {rng.choice(_PEOPLE)} reviewed metrics {rng.choice(_CONTEXTS)}. "
        f"Job success rate: {rng.randint(99, 100)}.{rng.randint(1, 9)}%. Average job duration: "
        f"{rng.randint(100, 5000)}ms. No timeout events in the past {rng.randint(7, 30)} days."
    ),
    lambda rng: (
        f"WORKER REFACTOR: worker_pool.py spawn_workers() line {rng.randint(130, 180)} through "
        f"{rng.randint(185, 220)} refactored by {rng.choice(_PEOPLE)}. Extracted the worker initialization "
        f"into a separate factory method for better unit testing. Connection pool setup now uses "
        f"context managers for automatic cleanup. config.yaml schema validated with jsonschema v{rng.randint(3, 4)}. "
        f"The refactoring reduced module complexity by {rng.randint(10, 30)}%. "
        f"PR #{rng.randint(1200, 1999)}: {rng.choice(_OUTCOMES)}."
    ),
    lambda rng: (
        f"WORKER CONFIG: config.yaml updated by {rng.choice(_PEOPLE)} {rng.choice(_CONTEXTS)}. "
        f"Worker pool size changed from {rng.randint(4, 8)} to {rng.randint(8, 16)} based on traffic analysis. "
        f"Queue priority weights rebalanced: critical={rng.randint(5, 10)}, high={rng.randint(3, 5)}, "
        f"normal={rng.randint(1, 2)}. Timeout values adjusted from {rng.randint(30, 60)}s to {rng.randint(60, 120)}s "
        f"for long-running jobs. spawn_workers() restart grace period set to {rng.randint(10, 30)}s. "
        f"Changes deployed to staging, production rollout scheduled for {rng.choice(['tomorrow', 'next week', 'after QA sign-off'])}."
    ),
    lambda rng: (
        f"WORKER MONITOR: worker_pool.py Prometheus metrics expanded by {rng.choice(_PEOPLE)}. "
        f"Added {rng.randint(4, 12)} new metrics for spawn_workers() including worker lifecycle events, "
        f"connection pool utilization, and queue backpressure indicators. Grafana dashboard updated with "
        f"{rng.randint(3, 8)} new panels. Alert rules: worker crash rate > {rng.randint(1, 5)}% triggers PagerDuty. "
        f"config.yaml monitoring section now includes retention policies for {rng.randint(7, 30)} days of data. "
        f"The infrastructure team reviews these dashboards {rng.choice(['daily', 'weekly', 'at each standup'])}."
    ),
]


def _generate_filler(seed: int) -> str:
    """Generate one unique hard-negative filler chunk (~450-550 chars)."""
    rng = random.Random(seed)
    template_fn = _FILLER_GENERATORS[seed % len(_FILLER_GENERATORS)]
    return template_fn(rng)


def generate_needle_placements(num_sessions: int, num_turns: int = 30, num_needles: int = 5) -> list[list[int]]:
    placements = []
    rng = random.Random(42)
    for _ in range(num_sessions):
        turns = sorted(rng.sample(range(num_turns), num_needles))
        placements.append(turns)
    return placements


def sliding_window_truncate(messages: list[dict[str, str]], max_tokens: int) -> list[dict[str, str]]:
    if not messages:
        return messages
    last_msg = messages[-1]
    last_tokens = _estimate_tokens(last_msg["content"])
    budget = max_tokens - SYSTEM_PROMPT_TOKENS - last_tokens
    if budget <= 0:
        return [last_msg]
    kept: list[dict[str, str]] = []
    used = 0
    for msg in reversed(messages[:-1]):
        msg_tokens = _estimate_tokens(msg["content"])
        if used + msg_tokens <= budget:
            kept.append(msg)
            used += msg_tokens
        else:
            break
    kept.reverse()
    kept.append(last_msg)
    return kept


def run_session(
    session_id: int,
    needle_turns: list[int],
    mode: str,  # "goal_guided", "auto_priority", "engine", or "naive"
    api_key: str,
    num_turns: int = 30,
    fillers_per_turn: int = 4,
) -> dict[str, Any]:
    from cerebras.cloud.sdk import Cerebras

    rng = random.Random(session_id * 1000 + hash(mode))
    compaction_log: list[dict] = []
    total_tokens_added = 0
    filler_seed_offset = session_id * 10000

    if mode == "goal_guided":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=False, goal_guided=True,
        )
    elif mode == "auto_priority":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=True, goal_guided=False,
        )
    elif mode == "engine":
        chunk_log = ChunkLog(
            db_path=":memory:", max_tokens=MAX_CONTEXT_TOKENS,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=False, goal_guided=False,
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

            if mode in ("goal_guided", "auto_priority"):
                priority = 0.5  # No hardcoded priority
            elif mode == "engine":
                priority = 2.0
            else:
                priority = 0.5

            total_tokens_added += _estimate_tokens(content)
            if mode in ("goal_guided", "auto_priority", "engine"):
                tokens_before = chunk_log.current_tokens()
                compactions_before = chunk_log.compaction_count
                chunk_log.append("user", content, priority=priority)
            else:
                raw_messages.append({"role": "user", "content": content})
        else:
            # Add multiple unique filler chunks as separate messages
            for j in range(fillers_per_turn):
                seed = filler_seed_offset + filler_idx
                filler_content = _generate_filler(seed)
                filler_idx += 1
                total_tokens_added += _estimate_tokens(filler_content)

                if mode in ("goal_guided", "auto_priority", "engine"):
                    if j == 0:
                        tokens_before = chunk_log.current_tokens()
                        compactions_before = chunk_log.compaction_count
                    chunk_log.append("user", filler_content, priority=0.5)
                else:
                    raw_messages.append({"role": "user", "content": filler_content})

        if mode in ("goal_guided", "auto_priority", "engine"):
            chunk_log.next_turn()
            tokens_after = chunk_log.current_tokens()
            compactions_after = chunk_log.compaction_count
            if compactions_after > compactions_before:
                compaction_log.append({
                    "turn": turn,
                    "tokens_before": tokens_before,
                    "tokens_after": tokens_after,
                    "events": compactions_after - compactions_before,
                })
        # Naive mode doesn't track compaction

    # Add recall question
    if mode in ("goal_guided", "auto_priority", "engine"):
        chunk_log.append("user", RECALL_QUESTION, priority=2.0)
        messages = chunk_log.get_context()
        context_tokens = chunk_log.get_context_tokens()
        compaction_events = chunk_log.compaction_count
    else:
        raw_messages.append({"role": "user", "content": RECALL_QUESTION})
        messages = sliding_window_truncate(raw_messages, MAX_CONTEXT_TOKENS)
        context_tokens = sum(_estimate_tokens(m["content"]) for m in messages)
        compaction_events = 0

    # Track which needles survived in context
    needles_in_context = []
    context_text = " ".join(m["content"] for m in messages)
    for needle in NEEDLES:
        if needle["keyword"].lower() in context_text.lower():
            needles_in_context.append(needle["id"])

    # Call Cerebras API
    client = Cerebras(api_key=api_key)

    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *messages,
            ],
            max_tokens=512,
            temperature=0.0,
        )
        ttft = time.time() - t0
        answer = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        error = None
    except Exception as e:
        answer = ""
        ttft = 0
        input_tokens = 0
        output_tokens = 0
        error = str(e)
    finally:
        if chunk_log:
            chunk_log.close()

    # Score: check which needles were recalled
    answer_lower = answer.lower()
    needles_recalled = []
    needles_lost = []
    for needle in NEEDLES:
        if needle["keyword"].lower() in answer_lower:
            needles_recalled.append(needle["id"])
        else:
            needles_lost.append(needle["id"])

    return {
        "session_id": session_id,
        "mode": mode,
        "error": error,
        "needle_turns": needle_turns,
        "needles_in_context": needles_in_context,
        "needles_recalled": needles_recalled,
        "needles_lost": needles_lost,
        "recall_score": len(needles_recalled),
        "total_needles": len(NEEDLES),
        "answer": answer,
        "ttft": round(ttft, 4),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_tokens": context_tokens,
        "total_tokens_added": total_tokens_added,
        "compaction_events": compaction_events,
        "compaction_log": compaction_log,
    }


def generate_chart(results: list[dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    modes_config = [
        ("goal_guided", "Goal-Guided\n(TF-IDF)", "#9b59b6"),
        ("engine", "Hardcoded\n(priority=2)", "#2ecc71"),
        ("auto_priority", "Keywords Only", "#3498db"),
        ("naive", "Naive\n(sliding window)", "#e74c3c"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    mode_scores = {}
    for mode_key, _, _ in modes_config:
        mode_results = [r for r in results if r["mode"] == mode_key]
        mode_scores[mode_key] = [r["recall_score"] for r in mode_results]

    n_sessions = len(mode_scores[modes_config[0][0]])

    # Chart 1: Per-session recall
    ax = axes[0]
    x = range(n_sessions)
    n_modes = len(modes_config)
    width = 0.8 / n_modes
    for i, (mode_key, label, color) in enumerate(modes_config):
        offset = (i - n_modes / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], mode_scores[mode_key], width, label=label, color=color)
    ax.set_xlabel("Session")
    ax.set_ylabel("Needles Recalled (out of 5)")
    ax.set_title("Needle Recall per Session")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"S{i+1}" for i in x])
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=6, loc="lower left")
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.3)

    # Chart 2: Average recall
    ax = axes[1]
    avgs = []
    labels = []
    colors = []
    for mode_key, label, color in modes_config:
        scores = mode_scores[mode_key]
        avgs.append(np.mean(scores) if scores else 0)
        labels.append(label)
        colors.append(color)
    bars = ax.bar(labels, avgs, color=colors)
    ax.set_ylabel("Avg Needles Recalled")
    ax.set_title("Average Recall Score (FAIR benchmark)")
    ax.set_ylim(0, 5.5)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}/5", ha="center", fontsize=11)

    # Chart 3: Needles in context
    ax = axes[2]
    needle_ids = [n["id"] for n in NEEDLES]
    x_arr = np.arange(len(needle_ids))
    for i, (mode_key, label, color) in enumerate(modes_config):
        mode_results = [r for r in results if r["mode"] == mode_key]
        counts = {nid: 0 for nid in needle_ids}
        for r in mode_results:
            for nid in r.get("needles_in_context", []):
                counts[nid] += 1
        offset = (i - n_modes / 2 + 0.5) * width
        ax.bar(x_arr + offset, [counts[nid] for nid in needle_ids], width, label=label, color=color)
    ax.set_xlabel("Needle")
    ax.set_ylabel(f"Times in Context (out of {n_sessions})")
    ax.set_title("Needle Survival in Context")
    ax.set_xticks(x_arr)
    ax.set_xticklabels([f"N{i+1}" for i in range(len(needle_ids))])
    ax.legend(fontsize=6)

    fig.suptitle(
        "FAIR Goal-Guided Benchmark: Length-Matched Chunks\n"
        "(All chunks ~500 chars, unique filler, adversarial keywords, 30 turns, 5 needles, 8k window)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to {output_path}")


def main():
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("ERROR: CEREBRAS_API_KEY not set")
        sys.exit(1)

    num_sessions = 10
    num_turns = 30
    fillers_per_turn = 4
    placements = generate_needle_placements(num_sessions, num_turns)

    # Length analysis
    print("LENGTH ANALYSIS (fairness check):")
    needle_lens = [len(n["fact"]) for n in NEEDLES]
    print(f"  Needle lengths: {needle_lens}")
    print(f"  Needle avg: {sum(needle_lens)/len(needle_lens):.0f} chars")

    sample_fillers = [_generate_filler(i) for i in range(20)]
    filler_lens = [len(f) for f in sample_fillers]
    print(f"  Filler lengths (sample 20): min={min(filler_lens)}, max={max(filler_lens)}, avg={sum(filler_lens)/len(filler_lens):.0f} chars")
    print(f"  Length ratio (filler/needle): {(sum(filler_lens)/len(filler_lens)) / (sum(needle_lens)/len(needle_lens)):.2f}x")
    print()

    # Token throughput estimate
    tokens_per_filler_turn = fillers_per_turn * (sum(filler_lens) / len(filler_lens)) / 4
    tokens_per_needle_turn = (sum(needle_lens) / len(needle_lens)) / 4
    filler_turns = num_turns - len(NEEDLES)
    total_throughput = filler_turns * tokens_per_filler_turn + len(NEEDLES) * tokens_per_needle_turn
    print(f"THROUGHPUT ESTIMATE:")
    print(f"  Filler turns: {filler_turns} x {fillers_per_turn} fillers x ~{tokens_per_filler_turn/fillers_per_turn:.0f} tokens = ~{filler_turns * tokens_per_filler_turn:.0f} tokens")
    print(f"  Needle turns: {len(NEEDLES)} x ~{tokens_per_needle_turn:.0f} tokens = ~{len(NEEDLES) * tokens_per_needle_turn:.0f} tokens")
    print(f"  Total throughput: ~{total_throughput:.0f} tokens through {MAX_CONTEXT_TOKENS} token window")
    print(f"  Soft threshold: {int(MAX_CONTEXT_TOKENS * 0.7)} tokens")
    print(f"  Hard threshold: {int(MAX_CONTEXT_TOKENS * 0.9)} tokens")
    print()

    # TF-IDF discrimination test with length-matched chunks
    print("TF-IDF DISCRIMINATION TEST (fair/length-matched):")
    scorer = GoalGuidedScorer()
    chunks = [(f"needle_{n['id']}", n["fact"]) for n in NEEDLES]
    chunks += [(f"filler_{i}", _generate_filler(i)) for i in range(20)]
    scores = scorer.score_chunks(RECALL_QUESTION, chunks)
    needle_s = [scores[h] for h, _ in chunks if h.startswith("needle")]
    filler_s = [scores[h] for h, _ in chunks if h.startswith("filler")]
    print(f"  Needle scores: [{min(needle_s):.3f}, {max(needle_s):.3f}] avg={sum(needle_s)/len(needle_s):.3f}")
    print(f"  Filler scores: [{min(filler_s):.3f}, {max(filler_s):.3f}] avg={sum(filler_s)/len(filler_s):.3f}")
    gap = min(needle_s) - max(filler_s)
    print(f"  Gap (min needle - max filler): {gap:.3f} {'SEPARABLE' if gap > 0 else 'OVERLAPPING'}")
    print()

    modes = ["goal_guided", "engine", "auto_priority", "naive"]
    total_tests = num_sessions * len(modes)

    print("=" * 60)
    print("FAIR Goal-Guided NIAH Benchmark")
    print("=" * 60)
    print(f"Sessions: {num_sessions}")
    print(f"Turns per session: {num_turns}")
    print(f"Fillers per filler turn: {fillers_per_turn}")
    print(f"Needles per session: {len(NEEDLES)}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Modes: {modes}")
    print()
    print("FAIR DESIGN: All chunks ~500 chars (needles padded, filler shortened)")
    print("Unique filler per session (no template recycling)")
    print("Same adversarial keyword overlap (same filenames/functions)")
    print()
    for i, p in enumerate(placements):
        print(f"  Session {i+1}: turns {p}")
    print("-" * 60)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    results: list[dict] = []
    test_num = 0

    for session_id, needle_turns in enumerate(placements):
        for mode in modes:
            test_num += 1
            label = f"[{test_num}/{total_tests}] Session {session_id+1} {mode.upper()}"
            print(f"{label} (needles at turns {needle_turns})...", end=" ", flush=True)

            result = run_session(session_id, needle_turns, mode, api_key, num_turns, fillers_per_turn)
            results.append(result)

            if result["error"]:
                print(f"ERR: {result['error'][:80]}")
            else:
                recalled = result["recall_score"]
                in_ctx = len(result.get("needles_in_context", []))
                lost = result.get("needles_lost", [])
                compact = result["compaction_events"]
                print(f"Recalled {recalled}/5 (in_context={in_ctx}/5, lost={lost}) ttft={result['ttft']:.2f}s compact={compact}")

            time.sleep(1)

    # Save results
    output = {
        "timestamp": timestamp,
        "model": "llama3.1-8b",
        "benchmark": "niah_goalguided_fair",
        "description": "FAIR Goal-Guided benchmark: length-matched chunks, unique filler, adversarial keywords",
        "design_notes": {
            "needle_avg_chars": sum(needle_lens) / len(needle_lens),
            "filler_avg_chars": sum(filler_lens) / len(filler_lens),
            "length_ratio": (sum(filler_lens) / len(filler_lens)) / (sum(needle_lens) / len(needle_lens)),
            "fillers_per_turn": fillers_per_turn,
            "unique_fillers": True,
            "adversarial_keywords": True,
        },
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "needles": NEEDLES,
        "placements": placements,
        "results": results,
    }

    json_path = RESULTS_DIR / f"niah_goalguided_fair_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    png_path = RESULTS_DIR / f"niah_goalguided_fair_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY (FAIR BENCHMARK)")
    print(f"{'=' * 60}")

    for label, mode_key in [
        ("GOAL-GUIDED (TF-IDF scoring)", "goal_guided"),
        ("ENGINE (hardcoded priority=2.0)", "engine"),
        ("KEYWORDS ONLY (auto_priority)", "auto_priority"),
        ("NAIVE (sliding window)", "naive"),
    ]:
        mode_results = [r for r in results if r["mode"] == mode_key]
        ok_results = [r for r in mode_results if not r["error"]]
        scores = [r["recall_score"] for r in ok_results]
        errors = len(mode_results) - len(ok_results)
        avg = sum(scores) / len(scores) if scores else 0
        compactions = [r["compaction_events"] for r in ok_results]
        avg_compact = sum(compactions) / len(compactions) if compactions else 0
        avg_ctx = sum(r["context_tokens"] for r in ok_results) / len(ok_results) if ok_results else 0
        in_ctx = [len(r.get("needles_in_context", [])) for r in ok_results]
        avg_in_ctx = sum(in_ctx) / len(in_ctx) if in_ctx else 0
        print(f"\n{label}:")
        print(f"  Avg recall: {avg:.1f}/5 ({100*avg/5:.0f}%)")
        print(f"  Scores: {[r['recall_score'] for r in mode_results]}")
        print(f"  Avg needles in context: {avg_in_ctx:.1f}/5")
        print(f"  API errors: {errors}")
        print(f"  Avg context tokens: {avg_ctx:.0f}")
        print(f"  Avg compaction events: {avg_compact:.1f}")

        lost_counts: dict[str, int] = {}
        for r in ok_results:
            for nid in r.get("needles_lost", []):
                lost_counts[nid] = lost_counts.get(nid, 0) + 1
        if lost_counts:
            print(f"  Needles lost: {dict(sorted(lost_counts.items(), key=lambda x: -x[1]))}")
        else:
            print(f"  Needles lost: none!")

    print(f"\n{'=' * 60}")
    print("\nCOMPARISON WITH ORIGINAL (UNFAIR) BENCHMARK:")
    print("Original: Needle=244 chars, Filler=1801 chars (7.4x ratio)")
    print("Fair:     All chunks ~500 chars (1.0x ratio)")
    print("If Goal-Guided drops significantly, the original 5.0/5 was a length artifact.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
