#!/usr/bin/env python3
"""Forced-budget Aider benchmark: stock vs BM25-patched under context pressure.

Forces a 16K token chat history budget so compaction actually triggers,
even on models with 1M+ context windows. This is where stock LLM-summarization
and BM25-scored message selection diverge.

Usage:
    GEMINI_API_KEY=... python3 benchmarks/aider_forced_budget.py
    python3 benchmarks/aider_forced_budget.py --budget 8192  # tighter budget
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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

CONTEXT_CHECKS = {
    1: {"retry", "client"},
    2: {"retry", "backoff", "max_retries"},
    3: {"retry", "timeout", "backoff"},
    4: {"retry", "logging", "backoff", "timeout"},
    5: {"retry", "test", "backoff", "timeout", "max_retries"},
    6: {"retry", "circuit", "breaker"},
    7: {"circuit", "breaker", "timeout"},
    8: {"retry", "circuit", "breaker", "resilience", "class"},
    9: {"resilience", "test", "circuit", "retry"},
}

HTTPX_REPO = Path("/tmp/httpx")
MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"
DEFAULT_FORCE_BUDGET = 16384


# --- BM25 Summarizer with force_budget ---

class BM25Summarizer:
    """BM25-scored message selection replacing Aider's LLM-based ChatSummary.

    Args:
        max_tokens: Token budget for chat history. When done_messages exceed
            this, BM25 scoring selects the most relevant pairs to keep.
        force_budget: If set, overrides the model's max_chat_history_tokens.
            This forces compaction at a specific threshold regardless of the
            model's actual context window size.
            Default: None (use max_tokens as-is).
            Example: force_budget=16384 forces compaction at 16K even on 1M models.
        token_count_fn: Callable to count tokens for a message dict.
    """

    def __init__(self, max_tokens=1024, force_budget=None, token_count_fn=None):
        from deterministic_context_engine.scorers.bm25 import BM25Scorer
        self.max_tokens = force_budget if force_budget is not None else max_tokens
        self.force_budget = force_budget
        self.token_count_fn = token_count_fn
        self._scorer = BM25Scorer(k1=1.5, b=0.75)
        self._last_user_message = ""
        self._compaction_count = 0
        self._evicted_pairs = 0

    def set_current_goal(self, message: str):
        self._last_user_message = message

    def too_big(self, messages):
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
        """BM25-based message selection instead of LLM summarization."""
        if not messages:
            return messages
        if not self.too_big(messages):
            return messages

        self._compaction_count += 1
        total_before = sum(self._count_tokens(m) for m in messages)

        goal = self._last_user_message or ""
        if not goal:
            for m in reversed(messages):
                if m.get("role") == "user":
                    goal = m.get("content", "")
                    break

        pairs = self._group_into_pairs(messages)
        if not pairs:
            return messages

        chunks_for_scoring = []
        for i, pair in enumerate(pairs):
            combined_text = " ".join(
                m.get("content", "") for m in pair
                if isinstance(m.get("content", ""), str)
            )
            chunks_for_scoring.append((f"pair_{i}", combined_text))

        scores = self._scorer.score_chunks(goal, chunks_for_scoring)

        scored_pairs = sorted(
            zip(pairs, [scores.get(f"pair_{i}", 0.5) for i in range(len(pairs))]),
            key=lambda x: x[1],
            reverse=True,
        )

        selected = []
        total_tokens = 0
        for pair, score in scored_pairs:
            pair_tokens = sum(self._count_tokens(m) for m in pair)
            if total_tokens + pair_tokens <= self.max_tokens:
                selected.append((pair, score))
                total_tokens += pair_tokens

        self._evicted_pairs += len(pairs) - len(selected)

        # Restore chronological order
        pair_order = {id(p): i for i, p in enumerate(pairs)}
        selected.sort(key=lambda x: pair_order.get(id(x[0]), 999))

        result = []
        for pair, _ in selected:
            result.extend(pair)

        if result and result[-1]["role"] != "assistant":
            result.append(dict(role="assistant", content="Ok."))

        total_after = sum(self._count_tokens(m) for m in result)
        print(f"      [BM25] Compaction #{self._compaction_count}: "
              f"{len(messages)} msgs ({total_before:,} tok) -> "
              f"{len(result)} msgs ({total_after:,} tok), "
              f"evicted {len(pairs) - len(selected)} pairs, "
              f"goal: '{goal[:60]}...'")

        return result

    @staticmethod
    def _group_into_pairs(messages):
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


# --- Helpers ---

class SessionLog:
    def __init__(self, name: str):
        self.name = name
        self.turns: list[dict] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def record_turn(self, prompt_idx, prompt, response, input_tokens, output_tokens,
                    elapsed, done_messages_count, cur_messages_count,
                    done_messages_tokens=0):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

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
            "done_messages_tokens": done_messages_tokens,
        })


def _make_quiet_io():
    from aider.io import InputOutput
    io = InputOutput(yes=True, chat_history_file="/dev/null")
    io.offer_url = lambda *a, **kw: None
    return io


def _get_key_files(repo_path: Path) -> list[str]:
    candidates = [
        "httpx/_client.py",
        "httpx/_config.py",
        "httpx/_transports/default.py",
        "httpx/_models.py",
        "httpx/_exceptions.py",
    ]
    return [str(repo_path / c) for c in candidates if (repo_path / c).exists()]


def _estimate_done_tokens(coder):
    """Estimate total tokens in done_messages."""
    total = 0
    for m in coder.done_messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        total += max(1, len(content) // 4)
    return total


def create_stock_coder(model_name: str, repo_path: Path, force_budget: int | None = None):
    """Stock Aider coder. If force_budget is set, override the summarizer's max_tokens."""
    from aider.coders import Coder
    from aider.models import Model
    from aider.history import ChatSummary

    model = Model(model_name)
    io = _make_quiet_io()
    fnames = _get_key_files(repo_path)

    summarizer = None
    if force_budget is not None:
        # Override the model's default chat history budget
        summarizer = ChatSummary(
            [model.weak_model, model],
            force_budget,
        )
        print(f"  Stock: force_budget={force_budget:,} "
              f"(model default was {model.max_chat_history_tokens:,})")

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
        summarizer=summarizer,
    )
    return coder


def create_patched_coder(model_name: str, repo_path: Path, force_budget: int | None = None):
    """BM25-patched Aider coder with optional forced budget."""
    from aider.coders import Coder
    from aider.models import Model

    model = Model(model_name)
    io = _make_quiet_io()
    fnames = _get_key_files(repo_path)

    bm25_summarizer = BM25Summarizer(
        max_tokens=model.max_chat_history_tokens,
        force_budget=force_budget,
        token_count_fn=model.token_count,
    )
    effective = force_budget if force_budget is not None else model.max_chat_history_tokens
    print(f"  BM25:  force_budget={effective:,} "
          f"(model default was {model.max_chat_history_tokens:,})")

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

    original_move_back = coder.move_back_cur_messages

    def patched_move_back(message):
        if message:
            bm25_summarizer.set_current_goal(message)
        elif coder.cur_messages:
            for m in reversed(coder.cur_messages):
                if m.get("role") == "user":
                    bm25_summarizer.set_current_goal(m.get("content", ""))
                    break
        original_move_back(message)

    coder.move_back_cur_messages = patched_move_back
    return coder, bm25_summarizer


def run_session(coder, session_name: str) -> SessionLog:
    log = SessionLog(session_name)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  [{session_name}] Turn {i+1}/{len(PROMPTS)}: {prompt[:60]}...")

        t0 = time.time()
        try:
            response = coder.run(with_message=prompt, preproc=False) or ""
        except Exception as e:
            response = f"[ERROR: {e}]"
        elapsed = time.time() - t0

        input_tokens = coder.message_tokens_sent or 0
        output_tokens = coder.message_tokens_received or 0
        done_count = len(coder.done_messages)
        cur_count = len(coder.cur_messages)
        done_tokens = _estimate_done_tokens(coder)

        log.record_turn(
            prompt_idx=i,
            prompt=prompt,
            response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed=elapsed,
            done_messages_count=done_count,
            cur_messages_count=cur_count,
            done_messages_tokens=done_tokens,
        )

        ctx_score = log.turns[-1]["context_score"]
        print(f"    Response: {len(response):,} chars | Ctx score: {ctx_score:.0%} "
              f"| Done: {done_count} msgs ({done_tokens:,} tok) "
              f"| Cur: {cur_count} | {elapsed:.1f}s")

        time.sleep(1)

    return log


def generate_report(stock_log: SessionLog, patched_log: SessionLog,
                    force_budget: int, bm25_stats: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"FORCED BUDGET BENCHMARK — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Stock Aider (LLM summarization) vs BM25-Patched Aider")
    lines.append(f"Model: {MODEL_NAME}  |  force_budget: {force_budget:,} tokens")
    lines.append(f"Repository: httpx (encode/httpx)  |  Prompts: {len(PROMPTS)}")
    lines.append("=" * 78)

    # Summary
    lines.append("\n## Summary")
    stock_ctx = sum(t["context_score"] for t in stock_log.turns) / len(stock_log.turns)
    patch_ctx = sum(t["context_score"] for t in patched_log.turns) / len(patched_log.turns)
    delta_ctx = patch_ctx - stock_ctx

    lines.append(f"  {'Metric':<40} {'Stock':>10} {'BM25':>10} {'Delta':>10}")
    lines.append(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")
    lines.append(f"  {'Avg context reference score':<40} {stock_ctx:>9.0%} {patch_ctx:>9.0%} {delta_ctx:>+9.0%}")

    stock_chars = sum(t["response_length"] for t in stock_log.turns)
    patch_chars = sum(t["response_length"] for t in patched_log.turns)
    lines.append(f"  {'Total response chars':<40} {stock_chars:>10,} {patch_chars:>10,}")

    stock_time = sum(t["elapsed_s"] for t in stock_log.turns)
    patch_time = sum(t["elapsed_s"] for t in patched_log.turns)
    lines.append(f"  {'Total wall time (s)':<40} {stock_time:>10.0f} {patch_time:>10.0f}")

    lines.append(f"  {'BM25 compaction events':<40} {'N/A':>10} {bm25_stats['compactions']:>10}")
    lines.append(f"  {'BM25 evicted pairs':<40} {'N/A':>10} {bm25_stats['evicted_pairs']:>10}")

    # Identical check
    identical = sum(1 for i in range(len(PROMPTS))
                    if stock_log.turns[i]["response"] == patched_log.turns[i]["response"])
    lines.append(f"  {'Identical responses':<40} {identical:>10}/{len(PROMPTS)}")

    # Per-turn
    lines.append(f"\n## Per-Turn Detail")
    lines.append(f"  {'T':>2} {'Prompt':<42} {'S-Ctx':>5} {'B-Ctx':>5} "
                 f"{'S-Done':>7} {'B-Done':>7} {'S-DoneTok':>9} {'B-DoneTok':>9} {'Same?':>5}")
    lines.append(f"  {'--':>2} {'-----':<42} {'-----':>5} {'-----':>5} "
                 f"{'------':>7} {'------':>7} {'---------':>9} {'---------':>9} {'-----':>5}")

    for i in range(len(PROMPTS)):
        st = stock_log.turns[i]
        pt = patched_log.turns[i]
        same = "Y" if st["response"] == pt["response"] else "N"
        lines.append(
            f"  {i+1:>2} {PROMPTS[i][:42]:<42} "
            f"{st['context_score']:>4.0%} {pt['context_score']:>4.0%} "
            f"{st['done_messages']:>7} {pt['done_messages']:>7} "
            f"{st['done_messages_tokens']:>9,} {pt['done_messages_tokens']:>9,} "
            f"{same:>5}"
        )

    # Context detail
    lines.append(f"\n## Context Referencing Detail")
    for i in range(len(PROMPTS)):
        if i not in CONTEXT_CHECKS:
            continue
        st = stock_log.turns[i]
        pt = patched_log.turns[i]
        lines.append(f"  Turn {i+1}: {PROMPTS[i]}")
        lines.append(f"    Expected:  {st['expected_keywords']}")
        lines.append(f"    Stock:     {st['matched_keywords']} ({st['context_score']:.0%})")
        lines.append(f"    BM25:      {pt['matched_keywords']} ({pt['context_score']:.0%})")

    # Divergence analysis
    lines.append(f"\n## Divergence Analysis")
    for i in range(len(PROMPTS)):
        st = stock_log.turns[i]
        pt = patched_log.turns[i]
        if st["response"] != pt["response"]:
            lines.append(f"\n  Turn {i+1}: DIVERGED")
            lines.append(f"    Stock:  {st['done_messages']} done msgs, "
                         f"{st['done_messages_tokens']:,} tok, "
                         f"{st['response_length']:,} chars response")
            lines.append(f"    BM25:   {pt['done_messages']} done msgs, "
                         f"{pt['done_messages_tokens']:,} tok, "
                         f"{pt['response_length']:,} chars response")
            # Show first 200 chars of each
            lines.append(f"    Stock response[:200]: {st['response'][:200]}...")
            lines.append(f"    BM25  response[:200]: {pt['response'][:200]}...")

    if identical == len(PROMPTS):
        lines.append("  No divergence — all responses identical.")

    lines.append("\n" + "=" * 78)
    lines.append("END OF REPORT")
    lines.append("=" * 78)
    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Forced-budget Aider benchmark")
    parser.add_argument("--budget", type=int, default=DEFAULT_FORCE_BUDGET,
                        help=f"Force chat history budget in tokens (default: {DEFAULT_FORCE_BUDGET})")
    args = parser.parse_args()

    force_budget = args.budget

    if not HTTPX_REPO.is_dir():
        print(f"Error: httpx repo not found at {HTTPX_REPO}")
        sys.exit(1)

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)

    datestamp = datetime.now().strftime("%Y%m%d")
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"Model: {MODEL_NAME}")
    print(f"force_budget: {force_budget:,} tokens")
    print(f"Repo: {HTTPX_REPO}")
    print(f"Prompts: {len(PROMPTS)}")

    # --- Session A: Stock ---
    subprocess.run(["git", "checkout", "."], cwd=str(HTTPX_REPO), capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=str(HTTPX_REPO), capture_output=True)

    print(f"\n{'='*60}")
    print(f"SESSION A: Stock Aider (LLM summarization, budget={force_budget:,})")
    print(f"{'='*60}")

    stock_coder = create_stock_coder(MODEL_NAME, HTTPX_REPO, force_budget=force_budget)
    stock_log = run_session(stock_coder, "Stock")

    # --- Session B: BM25 ---
    subprocess.run(["git", "checkout", "."], cwd=str(HTTPX_REPO), capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=str(HTTPX_REPO), capture_output=True)
    print(f"\nReset httpx repo for Session B")

    print(f"\n{'='*60}")
    print(f"SESSION B: BM25-Patched (force_budget={force_budget:,})")
    print(f"{'='*60}")

    patched_coder, bm25_summarizer = create_patched_coder(
        MODEL_NAME, HTTPX_REPO, force_budget=force_budget
    )
    patched_log = run_session(patched_coder, "BM25")

    bm25_stats = {
        "compactions": bm25_summarizer._compaction_count,
        "evicted_pairs": bm25_summarizer._evicted_pairs,
    }

    # --- Save results ---
    output_path = results_dir / f"aider_forced_budget_{datestamp}.json"
    result_data = {
        "benchmark": "aider_forced_budget",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "force_budget": force_budget,
        "repo": "encode/httpx",
        "prompts": PROMPTS,
        "stock": {
            "total_input_tokens": stock_log.total_input_tokens,
            "total_output_tokens": stock_log.total_output_tokens,
            "turns": stock_log.turns,
        },
        "bm25": {
            "total_input_tokens": patched_log.total_input_tokens,
            "total_output_tokens": patched_log.total_output_tokens,
            "compaction_events": bm25_stats["compactions"],
            "evicted_pairs": bm25_stats["evicted_pairs"],
            "turns": patched_log.turns,
        },
    }

    with open(output_path, "w") as f:
        json.dump(result_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # --- Report ---
    report = generate_report(stock_log, patched_log, force_budget, bm25_stats)
    report_path = results_dir / f"aider_forced_budget_{datestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
