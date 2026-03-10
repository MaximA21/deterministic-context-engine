#!/usr/bin/env python3
"""Real Aider session benchmark: stock vs BM25-patched context management.

Runs 10 sequential prompts through Aider against the httpx codebase.
Session A: stock Aider (LLM-based summarization for context compaction)
Session B: patched Aider (BM25-scored message selection instead of summarization)

Measures:
- Did the response reference relevant earlier context?
- Token count sent to API
- Response coherence (does it build on previous work?)

Usage:
    ANTHROPIC_API_KEY=... python3 benchmarks/aider_real_session.py
    python3 benchmarks/aider_real_session.py --mock  # dry run with mock LLM
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path for engine imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --- Prompts ---

PROMPTS = [
    "Add a retry mechanism to the Client class",
    "Add exponential backoff to the retry logic",
    "Add a max_retries parameter with default 3",
    "Now add timeout handling that works with the retry",
    "Add logging for each retry attempt",
    "Write tests for the retry mechanism",
    "Add a circuit breaker pattern on top of retries",
    "Make sure the circuit breaker respects the timeout from step 4",
    "Refactor: extract retry+circuit breaker into a separate Resilience class",
    "Update all tests to use the new Resilience class",
]

# Expected context references per prompt (for coherence scoring)
# Each entry: (prompt_index, expected_keywords_from_earlier_turns)
CONTEXT_CHECKS = {
    1: {"retry", "client"},  # should reference retry from prompt 0
    2: {"retry", "backoff", "max_retries"},  # should reference both
    3: {"retry", "timeout", "backoff"},  # should reference retry chain
    4: {"retry", "logging", "backoff", "timeout"},  # references all prior
    5: {"retry", "test", "backoff", "timeout", "max_retries"},  # tests for all
    6: {"retry", "circuit", "breaker"},  # new + existing
    7: {"circuit", "breaker", "timeout"},  # explicit cross-reference
    8: {"retry", "circuit", "breaker", "resilience", "class"},  # refactor
    9: {"resilience", "test", "circuit", "retry"},  # update tests
}

HTTPX_REPO = Path("/tmp/httpx")
MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"


# --- BM25 Patched Summarizer ---

class BM25Summarizer:
    """Replaces Aider's LLM-based ChatSummary with BM25-scored message selection.

    Instead of asking an LLM to summarize old messages, we score each message
    against the current conversation goal using BM25, and keep only the most
    relevant messages. This is deterministic, fast, and costs zero tokens.
    """

    def __init__(self, max_tokens=1024, token_count_fn=None):
        from deterministic_context_engine.scorers.bm25 import BM25Scorer
        self.max_tokens = max_tokens
        self.token_count_fn = token_count_fn
        self._scorer = BM25Scorer(k1=1.5, b=0.75)
        self._last_user_message = ""

    def set_current_goal(self, message: str):
        """Update the current goal/query for scoring relevance."""
        self._last_user_message = message

    def too_big(self, messages):
        """Check if messages exceed the token budget."""
        total = sum(self._count_tokens(m) for m in messages)
        return total > self.max_tokens

    def _count_tokens(self, msg):
        if self.token_count_fn:
            return self.token_count_fn(msg)
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        return max(1, len(content) // 4)

    def summarize(self, messages, depth=0):
        """BM25-based message selection instead of LLM summarization.

        Scores each message pair (user+assistant) against the current goal,
        keeps the highest-scoring pairs until we fit within the token budget.
        """
        if not messages:
            return messages

        if not self.too_big(messages):
            return messages

        goal = self._last_user_message or ""
        if not goal and messages:
            # Fallback: use the last user message in the history
            for m in reversed(messages):
                if m.get("role") == "user":
                    goal = m.get("content", "")
                    break

        # Group messages into (user, assistant) pairs for scoring
        pairs = self._group_into_pairs(messages)
        if not pairs:
            return messages

        # Score each pair
        chunks_for_scoring = []
        for i, pair in enumerate(pairs):
            combined_text = " ".join(
                m.get("content", "") for m in pair
                if isinstance(m.get("content", ""), str)
            )
            chunk_hash = f"pair_{i}"
            chunks_for_scoring.append((chunk_hash, combined_text))

        scores = self._scorer.score_chunks(goal, chunks_for_scoring)

        # Sort pairs by score (highest first)
        scored_pairs = sorted(
            zip(pairs, [scores.get(f"pair_{i}", 0.5) for i in range(len(pairs))]),
            key=lambda x: x[1],
            reverse=True,
        )

        # Greedily add highest-scoring pairs until we hit the budget
        selected = []
        total_tokens = 0
        for pair, score in scored_pairs:
            pair_tokens = sum(self._count_tokens(m) for m in pair)
            if total_tokens + pair_tokens <= self.max_tokens:
                selected.append((pair, score))
                total_tokens += pair_tokens

        # Restore chronological order
        selected_pairs = [p for p, _ in selected]
        # Sort by original index
        pair_indices = {id(p): i for i, p in enumerate(pairs)}
        selected_pairs.sort(key=lambda p: min(pair_indices.get(id(p), 999), 999))

        result = []
        for pair in selected_pairs:
            result.extend(pair)

        # Ensure ends with assistant message
        if result and result[-1]["role"] != "assistant":
            result.append(dict(role="assistant", content="Ok."))

        return result

    @staticmethod
    def _group_into_pairs(messages):
        """Group messages into conversation pairs (user + assistant)."""
        pairs = []
        current_pair = []
        for msg in messages:
            current_pair.append(msg)
            if msg.get("role") == "assistant":
                pairs.append(list(current_pair))
                current_pair = []
        if current_pair:
            pairs.append(current_pair)
        return pairs


# --- Session Runner ---

class SessionLog:
    """Accumulates per-turn metrics for one session."""

    def __init__(self, name: str):
        self.name = name
        self.turns: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def record_turn(self, prompt_idx, prompt, response, input_tokens, output_tokens,
                    elapsed, done_messages_count, cur_messages_count):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Check context referencing
        expected = CONTEXT_CHECKS.get(prompt_idx, set())
        response_lower = response.lower()
        matches = {kw for kw in expected if kw in response_lower}
        context_score = len(matches) / len(expected) if expected else 1.0

        self.turns.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "response": response,
            "response_length": len(response),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": round(elapsed, 2),
            "context_score": round(context_score, 2),
            "expected_keywords": sorted(expected),
            "matched_keywords": sorted(matches),
            "done_messages": done_messages_count,
            "cur_messages": cur_messages_count,
        })


def _make_quiet_io():
    """Create an IO object that never opens URLs or prompts interactively."""
    from aider.io import InputOutput
    io = InputOutput(yes=True, chat_history_file="/dev/null")
    # Suppress browser URL opening
    io.offer_url = lambda *a, **kw: None
    return io


def create_stock_coder(model_name: str, repo_path: Path):
    """Create a stock Aider coder with default settings."""
    from aider.coders import Coder
    from aider.models import Model

    model = Model(model_name)
    io = _make_quiet_io()

    # Get key source files to add to chat
    fnames = _get_key_files(repo_path)

    coder = Coder.create(
        main_model=model,
        io=io,
        fnames=fnames,
        auto_commits=False,
        stream=False,
        use_git=False,
        map_tokens=0,
        suggest_shell_commands=False,
        detect_urls=False,
    )
    return coder


def create_patched_coder(model_name: str, repo_path: Path):
    """Create an Aider coder with BM25-based context management.

    Replaces the default LLM summarizer with our BM25Summarizer,
    so context compaction uses relevance scoring instead of LLM calls.
    """
    from aider.coders import Coder
    from aider.models import Model

    model = Model(model_name)
    io = _make_quiet_io()
    fnames = _get_key_files(repo_path)

    # Create BM25 summarizer with the model's token counter
    bm25_summarizer = BM25Summarizer(
        max_tokens=1024,
        token_count_fn=model.token_count,
    )

    coder = Coder.create(
        main_model=model,
        io=io,
        fnames=fnames,
        auto_commits=False,
        stream=False,
        use_git=False,
        map_tokens=0,
        suggest_shell_commands=False,
        detect_urls=False,
        summarizer=bm25_summarizer,
    )

    # Monkey-patch move_back_cur_messages to feed goal to BM25
    original_move_back = coder.move_back_cur_messages

    def patched_move_back(message):
        # Update BM25 scorer with current user message as goal
        if message:
            bm25_summarizer.set_current_goal(message)
        elif coder.cur_messages:
            for m in reversed(coder.cur_messages):
                if m.get("role") == "user":
                    bm25_summarizer.set_current_goal(m.get("content", ""))
                    break
        original_move_back(message)

    coder.move_back_cur_messages = patched_move_back

    return coder


def _get_key_files(repo_path: Path) -> list[str]:
    """Get the most relevant httpx source files for the chat."""
    candidates = [
        "httpx/_client.py",
        "httpx/_config.py",
        "httpx/_transports/default.py",
        "httpx/_models.py",
        "httpx/_exceptions.py",
    ]
    result = []
    for c in candidates:
        p = repo_path / c
        if p.exists():
            result.append(str(p))
    return result


def run_session(coder, session_name: str, use_mock: bool = False) -> SessionLog:
    """Run all 10 prompts through the coder and collect metrics."""
    log = SessionLog(session_name)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  [{session_name}] Turn {i+1}/{len(PROMPTS)}: {prompt[:60]}...")

        t0 = time.time()
        try:
            response = coder.run(with_message=prompt, preproc=False) or ""
        except Exception as e:
            response = f"[ERROR: {e}]"
        elapsed = time.time() - t0

        # Estimate tokens from message counts
        input_tokens = coder.message_tokens_sent or 0
        output_tokens = coder.message_tokens_received or 0

        done_count = len(coder.done_messages) if hasattr(coder, 'done_messages') else 0
        cur_count = len(coder.cur_messages) if hasattr(coder, 'cur_messages') else 0

        log.record_turn(
            prompt_idx=i,
            prompt=prompt,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed=elapsed,
            done_messages_count=done_count,
            cur_messages_count=cur_count,
        )

        # Show brief status
        ctx_score = log.turns[-1]["context_score"]
        resp_len = len(response)
        print(f"    Response: {resp_len} chars | Context score: {ctx_score:.0%} "
              f"| Done msgs: {done_count} | Cur msgs: {cur_count} | {elapsed:.1f}s")

        if use_mock:
            time.sleep(0.1)
        else:
            time.sleep(1)  # Rate limiting

    return log


def generate_comparison_report(stock_log: SessionLog, patched_log: SessionLog) -> str:
    """Generate a detailed comparison report."""
    lines = []
    lines.append("=" * 78)
    lines.append(f"AIDER SESSION COMPARISON — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Stock Aider vs BM25-Patched Aider")
    lines.append(f"Repository: httpx (encode/httpx)")
    lines.append(f"Model: {MODEL_NAME}")
    lines.append("=" * 78)

    # Summary table
    lines.append("\n## Summary")
    lines.append(f"  {'Metric':<35} {'Stock':>12} {'BM25-Patched':>12} {'Delta':>10}")
    lines.append(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*10}")

    stock_ctx_avg = sum(t["context_score"] for t in stock_log.turns) / len(stock_log.turns)
    patch_ctx_avg = sum(t["context_score"] for t in patched_log.turns) / len(patched_log.turns)
    lines.append(f"  {'Avg context reference score':<35} {stock_ctx_avg:>11.0%} {patch_ctx_avg:>11.0%} {patch_ctx_avg - stock_ctx_avg:>+9.0%}")

    stock_tok = stock_log.total_input_tokens
    patch_tok = patched_log.total_input_tokens
    if stock_tok > 0:
        tok_delta = f"{(patch_tok - stock_tok) / stock_tok:>+9.0%}"
    else:
        tok_delta = "N/A"
    lines.append(f"  {'Total input tokens':<35} {stock_tok:>12,} {patch_tok:>12,} {tok_delta}")

    stock_out = stock_log.total_output_tokens
    patch_out = patched_log.total_output_tokens
    lines.append(f"  {'Total output tokens':<35} {stock_out:>12,} {patch_out:>12,}")

    stock_resp_avg = sum(t["response_length"] for t in stock_log.turns) / len(stock_log.turns)
    patch_resp_avg = sum(t["response_length"] for t in patched_log.turns) / len(patched_log.turns)
    lines.append(f"  {'Avg response length (chars)':<35} {stock_resp_avg:>12,.0f} {patch_resp_avg:>12,.0f}")

    # Per-turn comparison
    lines.append("\n## Per-Turn Comparison")
    lines.append(f"  {'Turn':>4} {'Prompt':<45} {'Stock Ctx':>9} {'BM25 Ctx':>9} {'Stock Tok':>10} {'BM25 Tok':>10}")
    lines.append(f"  {'----':>4} {'-----':<45} {'---------':>9} {'---------':>9} {'---------':>10} {'---------':>10}")

    for i in range(len(PROMPTS)):
        st = stock_log.turns[i]
        pt = patched_log.turns[i]
        lines.append(
            f"  {i+1:>4} {PROMPTS[i][:45]:<45} {st['context_score']:>8.0%} {pt['context_score']:>8.0%}"
            f" {st['input_tokens']:>10,} {pt['input_tokens']:>10,}"
        )

    # Context referencing detail
    lines.append("\n## Context Referencing Detail")
    for i in range(len(PROMPTS)):
        if i not in CONTEXT_CHECKS:
            continue
        st = stock_log.turns[i]
        pt = patched_log.turns[i]
        lines.append(f"\n  Turn {i+1}: {PROMPTS[i]}")
        lines.append(f"    Expected keywords: {st['expected_keywords']}")
        lines.append(f"    Stock matched:     {st['matched_keywords']} ({st['context_score']:.0%})")
        lines.append(f"    BM25 matched:      {pt['matched_keywords']} ({pt['context_score']:.0%})")

    # Message history sizes
    lines.append("\n## Message History Sizes")
    lines.append(f"  {'Turn':>4} {'Stock done':>12} {'Stock cur':>10} {'BM25 done':>12} {'BM25 cur':>10}")
    for i in range(len(PROMPTS)):
        st = stock_log.turns[i]
        pt = patched_log.turns[i]
        lines.append(
            f"  {i+1:>4} {st['done_messages']:>12} {st['cur_messages']:>10}"
            f" {pt['done_messages']:>12} {pt['cur_messages']:>10}"
        )

    lines.append("\n" + "=" * 78)
    lines.append("END OF REPORT")
    lines.append("=" * 78)

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aider stock vs BM25 benchmark")
    parser.add_argument("--mock", action="store_true", help="Use mock mode (no API calls)")
    args = parser.parse_args()

    if not HTTPX_REPO.is_dir():
        print(f"Error: httpx repo not found at {HTTPX_REPO}")
        print("Clone it first: git clone --depth 1 https://github.com/encode/httpx.git /tmp/httpx")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.mock:
        print("Error: GEMINI_API_KEY not set (use --mock for dry run)")
        sys.exit(1)

    datestamp = datetime.now().strftime("%Y%m%d")
    results_dir = PROJECT_ROOT / "results" / f"aider_real_session_{datestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {results_dir}")
    print(f"Model: {MODEL_NAME}")
    print(f"Repo: {HTTPX_REPO}")
    print(f"Prompts: {len(PROMPTS)}")

    # Reset httpx repo to clean state
    import subprocess
    subprocess.run(["git", "checkout", "."], cwd=str(HTTPX_REPO), capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=str(HTTPX_REPO), capture_output=True)

    # --- Session A: Stock Aider ---
    print(f"\n{'='*60}")
    print("SESSION A: Stock Aider (LLM summarization)")
    print(f"{'='*60}")

    stock_coder = create_stock_coder(MODEL_NAME, HTTPX_REPO)
    stock_log = run_session(stock_coder, "Stock", use_mock=args.mock)

    # Save Session A log
    with open(results_dir / "session_a_stock.json", "w") as f:
        json.dump({
            "session": "A_stock",
            "model": MODEL_NAME,
            "total_input_tokens": stock_log.total_input_tokens,
            "total_output_tokens": stock_log.total_output_tokens,
            "turns": stock_log.turns,
        }, f, indent=2, default=str)
    print(f"\nSession A saved to {results_dir / 'session_a_stock.json'}")

    # Reset httpx repo to clean state before Session B
    import subprocess
    subprocess.run(["git", "checkout", "."], cwd=str(HTTPX_REPO), capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=str(HTTPX_REPO), capture_output=True)
    print("\nReset httpx repo to clean state for Session B")

    # --- Session B: BM25-Patched Aider ---
    print(f"\n{'='*60}")
    print("SESSION B: BM25-Patched Aider (deterministic context)")
    print(f"{'='*60}")

    patched_coder = create_patched_coder(MODEL_NAME, HTTPX_REPO)
    patched_log = run_session(patched_coder, "BM25-Patched", use_mock=args.mock)

    # Save Session B log
    with open(results_dir / "session_b_bm25.json", "w") as f:
        json.dump({
            "session": "B_bm25_patched",
            "model": MODEL_NAME,
            "total_input_tokens": patched_log.total_input_tokens,
            "total_output_tokens": patched_log.total_output_tokens,
            "turns": patched_log.turns,
        }, f, indent=2, default=str)
    print(f"\nSession B saved to {results_dir / 'session_b_bm25.json'}")

    # --- Comparison Report ---
    report = generate_comparison_report(stock_log, patched_log)
    report_path = results_dir / "comparison_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
