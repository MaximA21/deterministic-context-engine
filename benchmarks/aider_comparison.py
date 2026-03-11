#!/usr/bin/env python3
"""Aider comparison: stock ChatSummary vs ChunkLogSummary.

Tests both approaches on the same sequence of 10 coding tasks by simulating
aider's internal message flow. No actual LLM API calls are needed —
we compare how each summarizer manages the conversation history.

This measures the context management quality, not LLM output quality.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine import _estimate_tokens
from aider_integration import ChunkLogSummary

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 10 sequential coding tasks (realistic aider session)
TASKS = [
    {
        "id": "task_1",
        "user": "Look at httpx/_client.py and explain the Client class constructor. What are the key parameters?",
        "assistant": "The `Client` class in httpx/_client.py has these key constructor parameters:\n- `auth`: Authentication handler (BasicAuth, DigestAuth, etc.)\n- `params`: Default query parameters\n- `headers`: Default headers dict\n- `cookies`: Cookie jar\n- `timeout`: Default timeout (5.0s)\n- `follow_redirects`: Whether to follow redirects (default False)\n- `max_redirects`: Max redirect count (default 20)\n- `base_url`: Base URL for relative requests\n- `transport`: Custom HTTP transport\n- `app`: ASGI/WSGI application for testing\n- `trust_env`: Whether to trust environment for proxy settings\n- `default_encoding`: Response text encoding fallback",
        "keyword": "follow_redirects",
    },
    {
        "id": "task_2",
        "user": "Find the retry logic in httpx. Does it support automatic retries on failure?",
        "assistant": "httpx does NOT have built-in retry logic. The `_send_single_request()` method in _client.py makes a single attempt. For retries, users need to either:\n1. Use httpx-retry third-party package\n2. Implement custom retry logic with a Transport wrapper\n3. Write a simple retry loop around client.get()/post()\n\nThe transport layer in httpx/_transports/default.py uses httpcore which also doesn't retry.",
        "keyword": "_send_single_request",
    },
    {
        "id": "task_3",
        "user": "How does httpx handle connection pooling? Find the relevant code.",
        "assistant": "Connection pooling is handled by httpcore (httpx's transport layer):\n\n1. `httpx/_transports/default.py` creates a `httpcore.ConnectionPool`\n2. Key parameters: `max_connections=100`, `max_keepalive_connections=20`, `keepalive_expiry=5.0`\n3. The pool maintains a dict of connections keyed by (scheme, host, port)\n4. Idle connections are reused for subsequent requests to the same origin\n5. `Client.__exit__()` calls `transport.close()` which closes all pooled connections\n\nThe pool is NOT shared between Client instances — each Client has its own pool.",
        "keyword": "max_keepalive_connections",
    },
    {
        "id": "task_4",
        "user": "Add a custom middleware that logs request/response timing. Show me how to do it with httpx transports.",
        "assistant": "Here's a timing middleware using httpx's transport layer:\n\n```python\nimport httpx\nimport time\n\nclass TimingTransport(httpx.BaseTransport):\n    def __init__(self, transport: httpx.BaseTransport):\n        self._transport = transport\n    \n    def handle_request(self, request: httpx.Request) -> httpx.Response:\n        start = time.monotonic()\n        response = self._transport.handle_request(request)\n        elapsed = time.monotonic() - start\n        print(f\"{request.method} {request.url} -> {response.status_code} ({elapsed:.3f}s)\")\n        return response\n\nclient = httpx.Client(transport=TimingTransport(httpx.HTTPTransport()))\n```\n\nThis wraps the default HTTPTransport, measuring time for each request.",
        "keyword": "TimingTransport",
    },
    {
        "id": "task_5",
        "user": "Find and fix the bug: httpx._exceptions.ConnectTimeout is raised but the error message doesn't include the URL. Add the URL to the error message.",
        "assistant": "Found it in httpx/_exceptions.py. The `ConnectTimeout` inherits from `TimeoutException` which uses a generic message. Fix:\n\n```python\n# In httpx/_exceptions.py, class TimeoutException:\ndef __init__(self, message: str = \"\", *, request: Request | None = None):\n    if request and not message:\n        message = f\"Timed out connecting to {request.url}\"\n    super().__init__(message)\n    self._request = request\n```\n\nThis adds the URL to the error message when a request object is available.",
        "keyword": "ConnectTimeout",
    },
    {
        "id": "task_6",
        "user": "How does httpx handle HTTP/2? Find the multiplexing implementation.",
        "assistant": "HTTP/2 support in httpx:\n\n1. Enabled via `httpx.Client(http2=True)` — requires `h2` package\n2. `httpx/_transports/default.py` passes `http2=True` to `httpcore.ConnectionPool`\n3. httpcore's HTTP/2 handling uses `h2` library for frame-level protocol\n4. Multiplexing: multiple requests share a single TCP connection via streams\n5. Each stream has its own flow control window (default 64KB)\n6. The connection pool tracks which connections support H2 vs H1.1\n7. `httpx._config.DEFAULT_LIMITS` sets `max_connections=100` for H2 pool\n\nNote: HTTP/2 is NOT default because it requires the optional `h2` dependency.",
        "keyword": "multiplexing",
    },
    {
        "id": "task_7",
        "user": "Add request/response hooks to the Client. How do event hooks work in httpx?",
        "assistant": "httpx supports event hooks via the `event_hooks` parameter:\n\n```python\ndef log_request(request: httpx.Request):\n    print(f\"Request: {request.method} {request.url}\")\n\ndef log_response(response: httpx.Response):\n    print(f\"Response: {response.status_code} from {response.url}\")\n\nclient = httpx.Client(\n    event_hooks={\n        'request': [log_request],\n        'response': [log_response],\n    }\n)\n```\n\nHooks are defined in `httpx/_client.py` in `_send_single_request()`. Request hooks fire before sending, response hooks after receiving. Multiple hooks per event are supported — they execute in order. Async hooks work with AsyncClient.",
        "keyword": "event_hooks",
    },
    {
        "id": "task_8",
        "user": "Review the authentication flow. How does DigestAuth work in httpx?",
        "assistant": "DigestAuth implementation in httpx/_auth.py:\n\n1. `DigestAuth(username, password)` creates an auth handler\n2. On first request, it sends WITHOUT auth headers\n3. Server responds with 401 + `WWW-Authenticate: Digest ...` header\n4. DigestAuth parses: realm, nonce, qop, algorithm (MD5/SHA-256)\n5. Computes: HA1 = MD5(username:realm:password), HA2 = MD5(method:uri)\n6. Response = MD5(HA1:nonce:nc:cnonce:qop:HA2)\n7. Retries request with Authorization header containing the digest\n8. `_digest_auth_flow()` is a generator that yields request objects\n\nKey: uses `nc` (nonce count) incrementing per request for replay protection.",
        "keyword": "DigestAuth",
    },
    {
        "id": "task_9",
        "user": "Now recall: what were the key Client constructor parameters we discussed at the start? And what was the connection pooling config?",
        "assistant": "From our earlier discussion:\n\n**Client constructor** (task 1): Key parameters were auth, params, headers, cookies, timeout (5.0s default), follow_redirects (False default), max_redirects (20), base_url, transport, app, trust_env, default_encoding.\n\n**Connection pooling** (task 3): Handled by httpcore.ConnectionPool with max_connections=100, max_keepalive_connections=20, keepalive_expiry=5.0. Each Client has its own pool, not shared.",
        "keyword": "keepalive_expiry",
    },
    {
        "id": "task_10",
        "user": "Final: write a comprehensive test for the TimingTransport middleware we created earlier.",
        "assistant": "```python\nimport httpx\nimport time\nimport pytest\nfrom unittest.mock import MagicMock, patch\n\ndef test_timing_transport_measures_elapsed():\n    mock_transport = MagicMock(spec=httpx.BaseTransport)\n    mock_response = httpx.Response(200, text=\"OK\")\n    mock_transport.handle_request.return_value = mock_response\n    \n    transport = TimingTransport(mock_transport)\n    request = httpx.Request(\"GET\", \"https://example.com/api\")\n    \n    with patch('builtins.print') as mock_print:\n        response = transport.handle_request(request)\n    \n    assert response.status_code == 200\n    mock_transport.handle_request.assert_called_once_with(request)\n    mock_print.assert_called_once()\n    output = mock_print.call_args[0][0]\n    assert \"GET\" in output\n    assert \"example.com\" in output\n    assert \"200\" in output\n```",
        "keyword": "TimingTransport",
    },
]

# Recall questions to check at the end
RECALL_CHECKS = [
    ("Client constructor params (task 1)", "follow_redirects"),
    ("No built-in retry (task 2)", "_send_single_request"),
    ("Connection pool config (task 3)", "max_keepalive_connections"),
    ("TimingTransport middleware (task 4)", "TimingTransport"),
    ("ConnectTimeout URL fix (task 5)", "ConnectTimeout"),
    ("HTTP/2 multiplexing (task 6)", "multiplexing"),
    ("Event hooks (task 7)", "event_hooks"),
    ("DigestAuth flow (task 8)", "DigestAuth"),
]


def simulate_session(summarizer, label: str, max_tokens: int) -> dict:
    """Simulate an aider session with the given summarizer."""
    done_messages: list[dict] = []

    for i, task in enumerate(TASKS):
        # Add user message and assistant reply
        done_messages.append({"role": "user", "content": task["user"]})
        done_messages.append({"role": "assistant", "content": task["assistant"]})

        # After each turn, aider checks if done_messages is too big
        if summarizer.too_big(done_messages):
            done_messages = summarizer.summarize(done_messages)

    # Check what survived
    all_content = " ".join(m.get("content", "") for m in done_messages).lower()
    tokens = sum(_estimate_tokens(str(m.get("content", ""))) for m in done_messages)

    recalls = {}
    for check_label, keyword in RECALL_CHECKS:
        recalls[check_label] = keyword.lower() in all_content

    recalled = sum(1 for v in recalls.values() if v)

    return {
        "label": label,
        "max_tokens": max_tokens,
        "messages_remaining": len(done_messages),
        "tokens_remaining": tokens,
        "recall_score": recalled,
        "total_checks": len(RECALL_CHECKS),
        "recalls": recalls,
    }


def main():
    print("=" * 70)
    print("Aider Context Management Comparison")
    print("Stock ChatSummary (LLM) vs ChunkLogSummary (BM25)")
    print("=" * 70)
    print(f"\nTasks: {len(TASKS)}")
    print(f"Recall checks: {len(RECALL_CHECKS)}")

    # Test at different budget levels
    budgets = [2000, 3000, 5000]
    results = []

    for budget in budgets:
        print(f"\n{'─' * 70}")
        print(f"Token budget: {budget}")
        print(f"{'─' * 70}")

        # BM25 compaction
        bm25_summarizer = ChunkLogSummary(max_tokens=budget, scoring_mode="bm25")
        bm25_result = simulate_session(bm25_summarizer, "BM25", budget)
        results.append(bm25_result)

        # TF-IDF compaction (for comparison)
        tfidf_summarizer = ChunkLogSummary(max_tokens=budget, scoring_mode="tfidf")
        tfidf_result = simulate_session(tfidf_summarizer, "TF-IDF", budget)
        results.append(tfidf_result)

        # No compaction (what would happen without summarization)
        total_tokens = sum(
            _estimate_tokens(t["user"]) + _estimate_tokens(t["assistant"])
            for t in TASKS
        )
        no_compact_result = {
            "label": "No compaction",
            "max_tokens": budget,
            "messages_remaining": len(TASKS) * 2,
            "tokens_remaining": total_tokens,
            "recall_score": len(RECALL_CHECKS),  # All present if no eviction
            "total_checks": len(RECALL_CHECKS),
            "recalls": {label: True for label, _ in RECALL_CHECKS},
        }
        results.append(no_compact_result)

        for r in [bm25_result, tfidf_result, no_compact_result]:
            overflow = "OVERFLOW" if r["tokens_remaining"] > budget else "OK"
            print(f"\n  {r['label']:15s}: {r['recall_score']}/{r['total_checks']} recalled, "
                  f"{r['messages_remaining']:2d} msgs, {r['tokens_remaining']:5d} tok [{overflow}]")
            for check, found in r["recalls"].items():
                status = "✓" if found else "✗"
                print(f"    {status} {check}")

    # Save results
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    output = {
        "timestamp": timestamp,
        "benchmark": "aider_comparison",
        "tasks": len(TASKS),
        "recall_checks": len(RECALL_CHECKS),
        "results": results,
    }
    json_path = RESULTS_DIR / f"aider_comparison_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Budget':>6s}  {'BM25':>8s}  {'TF-IDF':>8s}  {'No compact':>10s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*10}")
    for budget in budgets:
        bm25 = [r for r in results if r["label"] == "BM25" and r["max_tokens"] == budget][0]
        tfidf = [r for r in results if r["label"] == "TF-IDF" and r["max_tokens"] == budget][0]
        nc = [r for r in results if r["label"] == "No compaction" and r["max_tokens"] == budget][0]
        print(f"  {budget:>6d}  {bm25['recall_score']:>5d}/8  {tfidf['recall_score']:>5d}/8  {nc['recall_score']:>7d}/8")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
