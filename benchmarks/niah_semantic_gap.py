#!/usr/bin/env python3
"""Semantic Gap NIAH benchmark — tests TF-IDF weakness with synonym/paraphrase mismatches.

PROBLEM: TF-IDF matches exact words. If a needle says 'jwt_validation_error on line 58'
but the recall question asks about 'authentication bug', TF-IDF can't connect them.
This is a fundamental limitation of bag-of-words approaches.

DESIGN:
- 5 needles using TECHNICAL terminology (jwt_validation_error, CORS preflight failure,
  race condition in db_pool, memory leak in cache_eviction, segfault in protobuf deserializer)
- Recall questions use NATURAL LANGUAGE equivalents (authentication bug, cross-origin
  request issue, database connection problem, cache performance issue, data parsing crash)
- Filler mentions the natural language terms casually so keyword overlap is with
  FILLER not needles
- 30 turns, 8k window, 10 sessions

Expected results:
- Goal-Guided TF-IDF: STRUGGLE — uniqueness signal may not save it when recall
  question vocabulary doesn't overlap with needle vocabulary
- Hardcoded priority: CEILING — bypasses scoring entirely
- Keyword-only: FAIL — keywords from recall question match filler, not needles
- Naive sliding window: FAIL — recent filler pushes out older needles
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

# --- Needles: TECHNICAL terminology, ~500 chars each ---
# Each needle uses jargon/identifiers that DON'T appear in the recall questions.
# The recall questions use everyday language equivalents.

NEEDLES = [
    {
        "id": "needle_1",
        "fact": (
            "INCIDENT REPORT: jwt_validation_error thrown on line 58 of token_middleware.py "
            "in the verify_claims() function. The RS256 signature check passes but the 'aud' "
            "claim fails when the issuer rotates signing keys. The JWKS endpoint cache has a "
            "stale TTL of 3600s — must be reduced to 300s. Affected users see HTTP 401 with "
            "error code TVE-4012. The incident correlates with the key rotation at 02:15 UTC. "
            "Fix deployed in hotfix branch hotfix/jwt-aud-cache. Rollout to prod cluster-east "
            "scheduled for Thursday. Tracked in incident INC-7734."
        ),
        "keyword": "tve-4012",  # unique verification keyword
    },
    {
        "id": "needle_2",
        "fact": (
            "POSTMORTEM: CORS preflight OPTIONS request returns 403 from the nginx reverse proxy "
            "when the Origin header contains port 8443. The proxy_pass directive in site.conf "
            "line 127 strips the Access-Control-Allow-Origin response header for non-standard "
            "ports. Workaround: add 'proxy_hide_header Access-Control-Allow-Origin' and "
            "'add_header Access-Control-Allow-Origin $http_origin always' to the location block. "
            "This affects the partner integration with Acme Corp on subdomain partner.acme.io. "
            "SRE ticket SRE-2291 tracks the permanent fix. Reproduces on nginx 1.24 but not 1.25."
        ),
        "keyword": "sre-2291",
    },
    {
        "id": "needle_3",
        "fact": (
            "RACE CONDITION: db_pool.acquire() in pool_manager.py line 203 has a TOCTOU bug. "
            "Two threads can pass the availability check simultaneously when pool_size=max_size, "
            "causing a ConnectionExhaustedError after 30s timeout. The fix requires replacing the "
            "if-check with an atomic compare_and_swap on the pool counter using threading.Lock. "
            "Reproduces under 500+ concurrent requests in the load test suite. Stack trace shows "
            "the deadlock originates in _checkout_connection() at line 215. Monitoring alert "
            "DBPOOL-EXHAUST fires 3-5 times daily on the payments service. Fix PR #4417."
        ),
        "keyword": "pr #4417",
    },
    {
        "id": "needle_4",
        "fact": (
            "MEMORY PROFILE: cache_eviction.py LRUCache.evict() leaks 2.4KB per eviction cycle "
            "due to __del__ weak reference prevent preventing destructor call chain in weakref callback. "
            "The WeakValueDictionary in _ref_table retains stale entries when the referent is "
            "collected before the callback fires. Over 24h the process RSS grows from 512MB to "
            "1.8GB. Heap dump shows 847K orphaned CacheEntry objects. The fix: replace weakref "
            "callbacks with explicit cleanup in evict() using _ref_table.pop(key, None). "
            "Profiled with memray on Python 3.12.1. Tracked in PERF-0553."
        ),
        "keyword": "perf-0553",
    },
    {
        "id": "needle_5",
        "fact": (
            "SEGFAULT: protobuf_deserializer.cpp ParseFromArray() crashes with SIGSEGV at offset "
            "0x7f3a when deserializing MessageType.TRADE_EXECUTION with field tag 17 (repeated "
            "bytes). The proto definition trade_exec.proto has a packed=true annotation that "
            "conflicts with the wire format from the Java producer using protobuf 3.21.x. "
            "The C++ consumer runs protobuf 3.25.1 which handles packed encoding differently. "
            "Core dump at /var/crash/core.parser.14552. Workaround: set "
            "PROTOBUF_FORCE_COPY_DEFAULT_STRING=1. Fix: align proto versions. Ticket PROTO-8891."
        ),
        "keyword": "proto-8891",
    },
]

# Recall questions use NATURAL LANGUAGE — no technical jargon from needles
RECALL_QUESTION = (
    "I need you to recall specific details from our earlier conversation about these issues:\n\n"
    "1. We discussed an authentication bug where users were getting logged out unexpectedly. "
    "What was the root cause, and what's the fix? What's the incident ticket number?\n"
    "2. There was a cross-origin request issue affecting a partner integration. "
    "Which partner was affected, and what's the tracking ticket?\n"
    "3. We talked about a database connection problem causing timeouts under load. "
    "What's the underlying cause and which PR has the fix?\n"
    "4. There was a cache performance issue causing memory growth over time. "
    "How much memory does it leak, and what's the tracking ticket?\n"
    "5. We discussed a data parsing crash in production. "
    "What format was being parsed, and what's the ticket number?\n\n"
    "Only answer with SPECIFIC details from our conversation. Do NOT guess or fabricate."
)

# --- Filler: mentions the NATURAL LANGUAGE terms casually ---
# This is the adversarial part: filler uses the same vocabulary as the recall questions
# (authentication, cross-origin, database connection, cache performance, data parsing)
# so TF-IDF goal similarity will favor filler over needles.

_PEOPLE = [
    "Sarah Chen", "Marcus Rodriguez", "Priya Patel", "James Kim", "Lisa Wang",
    "Alex Thompson", "David Lee", "Nina Morales", "Ryan O'Brien", "Emily Fischer",
    "Tom Nguyen", "Aisha Hassan", "Chris Park", "Rachel Green", "Omar Diaz",
]
_CONTEXTS = [
    "during the quarterly review", "in the sprint retrospective",
    "at the architecture meeting", "during code review",
    "in the post-incident debrief", "at the team standup",
    "during the design review", "in the planning session",
    "at the all-hands meeting", "during pair programming",
]

# 20 filler templates — each casually mentions natural language terms from recall questions
_FILLER_GENERATORS = [
    # Authentication bug fillers (4) — mentions "authentication bug", "logged out", etc.
    lambda rng: (
        f"TEAM UPDATE: {rng.choice(_PEOPLE)} presented {rng.choice(_CONTEXTS)} about our "
        f"authentication bug tracking dashboard. We now monitor {rng.randint(12, 40)} auth-related "
        f"metrics. Users getting logged out unexpectedly has been a recurring theme in support "
        f"tickets — most turn out to be browser cookie issues, not backend problems. The "
        f"authentication team has reduced false-positive bug reports by {rng.randint(30, 70)}% "
        f"through better client-side session handling. Our auth error rate is {rng.randint(1, 5)}.{rng.randint(0, 9)}% "
        f"which is within SLA. The fix for most login issues is simply clearing the browser cache."
    ),
    lambda rng: (
        f"KNOWLEDGE BASE: {rng.choice(_PEOPLE)} updated the authentication bug troubleshooting guide "
        f"{rng.choice(_CONTEXTS)}. Common authentication bug patterns include expired sessions, "
        f"clock skew between client and server, and SSO federation misconfigs. Users logged out "
        f"unexpectedly should first check their network connectivity and VPN status. The guide "
        f"now covers {rng.randint(15, 30)} authentication bug scenarios with step-by-step "
        f"resolution procedures. Root cause for most authentication issues: the user's "
        f"corporate proxy strips auth headers. Updated FAQ has {rng.randint(50, 200)} views this week."
    ),
    lambda rng: (
        f"METRICS REVIEW: {rng.choice(_PEOPLE)} analyzed authentication bug trends {rng.choice(_CONTEXTS)}. "
        f"Authentication failures are down {rng.randint(10, 40)}% quarter over quarter. The most "
        f"common authentication bug category is password policy violations at {rng.randint(30, 60)}% "
        f"of total incidents. Users being logged out was investigated — {rng.randint(80, 95)}% of cases "
        f"were caused by session timeout settings, not actual bugs. Fix: adjusted default "
        f"session duration from {rng.randint(15, 30)} to {rng.randint(45, 90)} minutes. Auth team "
        f"velocity: {rng.randint(20, 40)} story points per sprint."
    ),
    lambda rng: (
        f"TRAINING NOTES: {rng.choice(_PEOPLE)} led a workshop on diagnosing authentication bugs "
        f"{rng.choice(_CONTEXTS)}. Key takeaway: most authentication bug reports from users getting "
        f"logged out are not real bugs — they're configuration issues. The root cause analysis "
        f"framework now includes {rng.randint(5, 12)} checkpoints for auth issues. Workshop "
        f"attendance: {rng.randint(15, 35)} engineers. Feedback score: {rng.randint(4, 5)}.{rng.randint(0, 9)}/5. "
        f"Materials published to the internal wiki. Next authentication security workshop "
        f"scheduled for {rng.choice(['next month', 'Q3', 'after the offsite'])}."
    ),
    # Cross-origin / partner integration fillers (4)
    lambda rng: (
        f"API REVIEW: {rng.choice(_PEOPLE)} reviewed cross-origin request handling {rng.choice(_CONTEXTS)}. "
        f"Our cross-origin request policy allows {rng.randint(5, 20)} approved partner domains. "
        f"Partner integration health: {rng.randint(95, 100)}.{rng.randint(0, 9)}% uptime across "
        f"all integrations. The cross-origin request middleware processes {rng.randint(100, 500)}K "
        f"preflight requests daily. No partner has reported cross-origin issues in {rng.randint(30, 90)} "
        f"days. The cross-origin configuration is version-controlled and requires {rng.randint(2, 4)} "
        f"approvals to modify. CORS policy documentation was last updated {rng.randint(1, 4)} weeks ago."
    ),
    lambda rng: (
        f"PARTNER STATUS: {rng.choice(_PEOPLE)} reported on partner integration health {rng.choice(_CONTEXTS)}. "
        f"Cross-origin request issues from partners have decreased by {rng.randint(20, 60)}% since "
        f"we deployed the new CORS middleware. Partner integration onboarding now takes "
        f"{rng.randint(2, 5)} days instead of {rng.randint(10, 20)}. The most common partner "
        f"integration complaint is still about cross-origin request timeouts on slow networks. "
        f"We maintain {rng.randint(10, 30)} active partner integrations. Partner satisfaction "
        f"score: {rng.randint(4, 5)}.{rng.randint(0, 9)}/5. Next partner review: {rng.choice(['Friday', 'next week', 'end of month'])}."
    ),
    lambda rng: (
        f"SECURITY AUDIT: {rng.choice(_PEOPLE)} completed the cross-origin request security review "
        f"{rng.choice(_CONTEXTS)}. All partner integration endpoints validated against OWASP "
        f"guidelines. Cross-origin request headers properly sanitized in {rng.randint(95, 100)}% "
        f"of endpoints. The partner integration testing suite covers {rng.randint(40, 80)} "
        f"cross-origin request scenarios. No cross-origin vulnerabilities found. Audit report "
        f"submitted to compliance team. Next cross-origin security review in {rng.randint(3, 6)} months. "
        f"Total endpoints audited: {rng.randint(50, 150)}."
    ),
    lambda rng: (
        f"DOCS UPDATE: {rng.choice(_PEOPLE)} refreshed the cross-origin request configuration guide "
        f"{rng.choice(_CONTEXTS)}. New partner integration setup instructions include cross-origin "
        f"request debugging steps. Added {rng.randint(3, 8)} troubleshooting scenarios for common "
        f"cross-origin request errors. The partner integration SDK now handles cross-origin "
        f"preflight automatically. Documentation covers {rng.choice(['Chrome', 'Firefox', 'Safari'])} "
        f"specific cross-origin request behaviors. Guide length: {rng.randint(15, 30)} pages. "
        f"Reader feedback: {rng.randint(4, 5)}.{rng.randint(0, 9)}/5 helpfulness rating."
    ),
    # Database connection / timeout fillers (4)
    lambda rng: (
        f"DB STATUS: {rng.choice(_PEOPLE)} reviewed database connection pool health {rng.choice(_CONTEXTS)}. "
        f"Database connection timeouts are at historic lows — {rng.randint(1, 8)} per million requests. "
        f"The database connection problem we saw last quarter was resolved by upgrading the "
        f"connection pooler. Pool utilization: {rng.randint(40, 75)}% average, {rng.randint(80, 95)}% peak. "
        f"Database connection monitoring covers {rng.randint(5, 15)} clusters. No database connection "
        f"issues reported under load this sprint. Connection string rotation completed for "
        f"{rng.randint(3, 8)} services. Health check interval: {rng.randint(10, 30)}s."
    ),
    lambda rng: (
        f"CAPACITY PLAN: {rng.choice(_PEOPLE)} presented database connection capacity analysis "
        f"{rng.choice(_CONTEXTS)}. Current database connection limit: {rng.randint(100, 500)} "
        f"per service instance. Database connection problem threshold: alert fires at "
        f"{rng.randint(80, 95)}% utilization. Under load testing, database connection timeouts "
        f"only occur above {rng.randint(2000, 5000)} concurrent requests. The database connection "
        f"pool auto-scales between {rng.randint(10, 30)} and {rng.randint(50, 200)} connections. "
        f"Projected growth: {rng.randint(10, 25)}% more database connections needed by Q4."
    ),
    lambda rng: (
        f"RUNBOOK UPDATE: {rng.choice(_PEOPLE)} updated the database connection troubleshooting "
        f"runbook {rng.choice(_CONTEXTS)}. When users report database connection problems under "
        f"load, the first step is checking connection pool saturation. The database connection "
        f"timeout default is {rng.randint(15, 45)}s. Most database connection issues resolve "
        f"themselves within {rng.randint(1, 5)} minutes. The runbook now covers {rng.randint(8, 15)} "
        f"database connection problem scenarios. Load testing showed our database connection "
        f"handling is robust up to {rng.randint(3, 10)}x normal traffic. "
        f"Runbook last tested: {rng.randint(1, 4)} weeks ago."
    ),
    lambda rng: (
        f"PERFORMANCE: {rng.choice(_PEOPLE)} benchmarked database connection establishment time "
        f"{rng.choice(_CONTEXTS)}. Average database connection setup: {rng.randint(2, 15)}ms. "
        f"Database connection reuse ratio: {rng.randint(92, 99)}%. Under load, database connection "
        f"wait time stays below {rng.randint(50, 200)}ms for p99. The database connection problem "
        f"detection pipeline has {rng.randint(3, 8)} stages. Connection pool warm-up takes "
        f"{rng.randint(5, 20)}s on cold start. Database connection timeout monitoring has "
        f"{rng.randint(99, 100)}.{rng.randint(0, 9)}% alert coverage."
    ),
    # Cache performance / memory growth fillers (4)
    lambda rng: (
        f"CACHE REVIEW: {rng.choice(_PEOPLE)} analyzed cache performance metrics {rng.choice(_CONTEXTS)}. "
        f"Cache hit ratio: {rng.randint(85, 98)}%. Cache performance issue reports have dropped "
        f"{rng.randint(20, 50)}% since the last tuning session. Memory growth from cache is "
        f"normal — {rng.randint(10, 50)}MB per day is expected and handled by periodic GC. "
        f"Cache performance monitoring covers {rng.randint(5, 15)} cache instances. The cache "
        f"performance team runs weekly reviews. TTL distribution: {rng.randint(60, 300)}s avg "
        f"across all cache keys. Eviction rate: {rng.randint(100, 1000)} keys/minute."
    ),
    lambda rng: (
        f"CACHE TUNING: {rng.choice(_PEOPLE)} completed cache performance optimization "
        f"{rng.choice(_CONTEXTS)}. Memory growth over time was investigated — current rate is "
        f"healthy at {rng.randint(5, 30)}MB/hour during peak. Cache performance issues typically "
        f"come from hot keys, not memory leaks. The cache performance dashboard shows "
        f"{rng.randint(3, 10)} key metrics. Memory growth tracking ticket resolved — no action "
        f"needed. Cache size: {rng.randint(2, 16)}GB across {rng.randint(3, 12)} nodes. "
        f"Cache performance SLA: p99 read latency < {rng.randint(1, 10)}ms."
    ),
    lambda rng: (
        f"CACHE OPS: {rng.choice(_PEOPLE)} reviewed cache performance alerts {rng.choice(_CONTEXTS)}. "
        f"Cache performance issue false alarms reduced from {rng.randint(10, 30)} to {rng.randint(1, 5)} "
        f"per week. Memory growth alerts correctly fire when RSS exceeds {rng.randint(4, 16)}GB. "
        f"The cache performance team processes {rng.randint(5, 20)} tickets per sprint. "
        f"Cache eviction strategy: LRU with {rng.randint(1, 5)} minute grace period. "
        f"No cache performance issues in production for {rng.randint(14, 60)} days. "
        f"Memory growth patterns are seasonal — higher during {rng.choice(['marketing campaigns', 'holiday traffic', 'end of quarter'])}."
    ),
    lambda rng: (
        f"CACHE DOCS: {rng.choice(_PEOPLE)} updated cache performance best practices "
        f"{rng.choice(_CONTEXTS)}. Guidelines for avoiding cache performance issues include "
        f"setting appropriate TTLs, monitoring memory growth, and using cache-aside pattern. "
        f"The cache performance guide now has {rng.randint(10, 25)} pages. Memory growth "
        f"troubleshooting section expanded with {rng.randint(3, 8)} diagnostic steps. Cache "
        f"performance benchmarking procedure documented for {rng.randint(3, 6)} cache backends. "
        f"Documentation read by {rng.randint(20, 50)} engineers this month."
    ),
    # Data parsing / crash fillers (4)
    lambda rng: (
        f"PARSING REVIEW: {rng.choice(_PEOPLE)} reviewed data parsing crash reports {rng.choice(_CONTEXTS)}. "
        f"Data parsing crash rate: {rng.randint(1, 10)} per million messages. Most data parsing "
        f"crashes are caused by malformed input from external feeds, not code bugs. The data "
        f"parsing pipeline handles {rng.randint(5, 50)}M messages daily. Crash recovery is "
        f"automatic with {rng.randint(1, 3)} retry attempts. Data parsing format validation "
        f"catches {rng.randint(95, 99)}% of bad input before it reaches the parser. The data "
        f"parsing crash dashboard was updated with {rng.randint(3, 8)} new panels."
    ),
    lambda rng: (
        f"PARSING METRICS: {rng.choice(_PEOPLE)} analyzed data parsing performance {rng.choice(_CONTEXTS)}. "
        f"Data parsing throughput: {rng.randint(10, 100)}K messages/second. Data parsing crash "
        f"frequency is stable month over month. The data parsing format we use supports "
        f"{rng.randint(5, 15)} message types. Production data parsing crash investigation found "
        f"no systemic issues — all were edge cases in malformed input. Data parsing latency: "
        f"{rng.randint(1, 10)}ms p99. The data parsing team maintains {rng.randint(50, 150)} "
        f"integration tests. Crash dump retention: {rng.randint(7, 30)} days."
    ),
    lambda rng: (
        f"PARSING OPS: {rng.choice(_PEOPLE)} updated data parsing crash response procedures "
        f"{rng.choice(_CONTEXTS)}. When a data parsing crash occurs in production, the on-call "
        f"engineer checks the format validation stage first. Data parsing crash alerts include "
        f"sample payload for debugging. The data parsing format schema is versioned and backward "
        f"compatible for {rng.randint(3, 8)} major versions. Data parsing crash rate SLA: "
        f"< {rng.randint(5, 20)} per million. Format migration guide covers "
        f"{rng.randint(4, 10)} data parsing formats. Crash auto-recovery: {rng.randint(95, 99)}% success rate."
    ),
    lambda rng: (
        f"PARSING IMPROVEMENTS: {rng.choice(_PEOPLE)} proposed data parsing crash prevention "
        f"measures {rng.choice(_CONTEXTS)}. Suggested adding format pre-validation, schema "
        f"registry, and data parsing crash circuit breaker. Data parsing crash budget: "
        f"{rng.randint(10, 50)} allowed per day before alert. The data parsing format team "
        f"reviewed {rng.randint(5, 15)} crash scenarios. Production data parsing crash "
        f"analysis shows {rng.randint(60, 90)}% are from a single external data source. "
        f"Proposed fix: add format negotiation in the ingestion layer. "
        f"Estimated effort: {rng.randint(3, 8)} story points."
    ),
]


def _generate_filler(seed: int) -> str:
    """Generate one unique filler chunk (~450-550 chars) using natural language terms."""
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

    # Score: check which needles were recalled via unique verification keywords
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
    ax.set_title("Average Recall Score (SEMANTIC GAP)")
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
        "SEMANTIC GAP Benchmark: TF-IDF vs Synonym/Paraphrase Mismatch\n"
        "(Technical needles, natural-language recall, filler uses recall vocabulary, 30 turns, 5 needles, 8k window)",
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
    print("LENGTH ANALYSIS:")
    needle_lens = [len(n["fact"]) for n in NEEDLES]
    print(f"  Needle lengths: {needle_lens}")
    print(f"  Needle avg: {sum(needle_lens)/len(needle_lens):.0f} chars")

    sample_fillers = [_generate_filler(i) for i in range(20)]
    filler_lens = [len(f) for f in sample_fillers]
    print(f"  Filler lengths (sample 20): min={min(filler_lens)}, max={max(filler_lens)}, avg={sum(filler_lens)/len(filler_lens):.0f} chars")
    print(f"  Length ratio (filler/needle): {(sum(filler_lens)/len(filler_lens)) / (sum(needle_lens)/len(needle_lens)):.2f}x")
    print()

    # Semantic gap analysis — show that TF-IDF goal similarity favors filler
    print("SEMANTIC GAP ANALYSIS:")
    print("  Recall question vocabulary: authentication bug, cross-origin request, database connection,")
    print("                              cache performance, data parsing crash")
    print("  Needle vocabulary:          jwt_validation_error, CORS preflight, db_pool.acquire(),")
    print("                              cache_eviction LRUCache, protobuf_deserializer SIGSEGV")
    print()

    scorer = GoalGuidedScorer()
    chunks = [(f"needle_{n['id']}", n["fact"]) for n in NEEDLES]
    chunks += [(f"filler_{i}", _generate_filler(i)) for i in range(20)]
    scores = scorer.score_chunks(RECALL_QUESTION, chunks)
    needle_s = [scores[h] for h, _ in chunks if h.startswith("needle")]
    filler_s = [scores[h] for h, _ in chunks if h.startswith("filler")]
    print(f"  TF-IDF GOAL SIMILARITY (higher = more similar to recall question):")
    print(f"    Needle scores: [{min(needle_s):.3f}, {max(needle_s):.3f}] avg={sum(needle_s)/len(needle_s):.3f}")
    print(f"    Filler scores: [{min(filler_s):.3f}, {max(filler_s):.3f}] avg={sum(filler_s)/len(filler_s):.3f}")
    gap = min(needle_s) - max(filler_s)
    print(f"    Gap (min needle - max filler): {gap:.3f} {'SEPARABLE' if gap > 0 else 'OVERLAPPING — SEMANTIC GAP EXPLOITED'}")
    print()

    # Per-needle TF-IDF scores
    print("  Per-needle TF-IDF scores:")
    for n in NEEDLES:
        h = f"needle_{n['id']}"
        print(f"    {n['id']}: {scores[h]:.3f}")
    print("  Top 5 filler TF-IDF scores:")
    filler_scores_sorted = sorted(
        [(h, scores[h]) for h, _ in chunks if h.startswith("filler")],
        key=lambda x: -x[1]
    )
    for h, s in filler_scores_sorted[:5]:
        print(f"    {h}: {s:.3f}")
    print()

    # Token throughput estimate
    tokens_per_filler_turn = fillers_per_turn * (sum(filler_lens) / len(filler_lens)) / 4
    tokens_per_needle_turn = (sum(needle_lens) / len(needle_lens)) / 4
    filler_turns = num_turns - len(NEEDLES)
    total_throughput = filler_turns * tokens_per_filler_turn + len(NEEDLES) * tokens_per_needle_turn
    print(f"THROUGHPUT ESTIMATE:")
    print(f"  Total throughput: ~{total_throughput:.0f} tokens through {MAX_CONTEXT_TOKENS} token window")
    print(f"  Soft threshold: {int(MAX_CONTEXT_TOKENS * 0.7)} tokens")
    print(f"  Hard threshold: {int(MAX_CONTEXT_TOKENS * 0.9)} tokens")
    print()

    modes = ["goal_guided", "engine", "auto_priority", "naive"]
    total_tests = num_sessions * len(modes)

    print("=" * 60)
    print("SEMANTIC GAP NIAH Benchmark")
    print("=" * 60)
    print(f"Sessions: {num_sessions}")
    print(f"Turns per session: {num_turns}")
    print(f"Fillers per filler turn: {fillers_per_turn}")
    print(f"Needles per session: {len(NEEDLES)}")
    print(f"Context window: {MAX_CONTEXT_TOKENS:,} tokens")
    print(f"Modes: {modes}")
    print()
    print("SEMANTIC GAP DESIGN: Needles use technical jargon, recall uses natural language.")
    print("Filler uses the SAME natural language as recall questions.")
    print("TF-IDF should favor filler over needles for goal similarity.")
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
        "benchmark": "niah_semantic_gap",
        "description": "Semantic Gap NIAH: technical needles vs natural-language recall with adversarial filler",
        "design_notes": {
            "needle_avg_chars": sum(needle_lens) / len(needle_lens),
            "filler_avg_chars": sum(filler_lens) / len(filler_lens),
            "length_ratio": (sum(filler_lens) / len(filler_lens)) / (sum(needle_lens) / len(needle_lens)),
            "fillers_per_turn": fillers_per_turn,
            "semantic_gap": True,
            "filler_uses_recall_vocabulary": True,
        },
        "num_sessions": num_sessions,
        "num_turns": num_turns,
        "num_needles": len(NEEDLES),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "needles": NEEDLES,
        "placements": placements,
        "results": results,
    }

    json_path = RESULTS_DIR / f"niah_semantic_gap_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    png_path = RESULTS_DIR / f"niah_semantic_gap_{timestamp}.png"
    try:
        generate_chart(results, png_path)
    except Exception as e:
        print(f"Chart generation failed: {e}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY (SEMANTIC GAP BENCHMARK)")
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
    print("\nSEMANTIC GAP HYPOTHESIS:")
    print("If Goal-Guided drops significantly vs hardcoded, TF-IDF can't bridge")
    print("the vocabulary gap between technical needles and natural-language recall.")
    print("This would confirm TF-IDF's synonym/paraphrase blindness.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
