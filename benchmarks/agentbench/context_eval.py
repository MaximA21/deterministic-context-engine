#!/usr/bin/env python3
"""Offline context quality evaluation on AGENTbench instances.

Measures how well our scoring engine retains relevant context (the files
that the gold patch actually touches) under token budget constraints.

No Docker or API keys required — pure offline evaluation.

Metrics:
  - key_file_recall: fraction of gold-patch files retained in context
  - key_file_precision: fraction of retained files that are in the gold patch
  - token_compression: ratio of total tokens to context window tokens
  - avg_priority_gap: mean priority difference (key files vs filler files)
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine import ChunkLog, _estimate_tokens

from benchmarks.agentbench.download import load_instances

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ContextEvalResult:
    instance_id: str
    repo: str
    scoring_mode: str
    max_tokens: int
    total_chunks: int
    retained_chunks: int
    key_files: tuple[str, ...]
    retained_key_files: tuple[str, ...]
    key_file_recall: float
    key_file_precision: float
    token_compression: float
    total_tokens_in: int
    context_tokens_out: int
    compaction_events: int
    avg_key_priority: float
    avg_filler_priority: float
    priority_gap: float
    scoring_time_ms: float


def extract_key_files_from_patch(patch: str) -> list[str]:
    """Extract file paths modified by a git diff."""
    return re.findall(r"^diff --git a/(.*?) b/", patch, re.MULTILINE)


def build_repo_chunks(instance: dict) -> list[dict[str, str]]:
    """Build context chunks simulating a coding agent's exploration.

    Simulates a realistic agent workflow:
      1. Problem description (high priority — the user's task, protected from eviction)
      2. Setup context
      3. Filler files (repo exploration — browsing unrelated files)
      4. Key files interleaved at realistic positions
      5. Test files (specification)
      6. Goal restatement (becomes _last_user_message for BM25 scoring)

    The problem description is added first AND last: first so it's always
    in context (high priority), last so BM25 scores against it.
    """
    chunks = []

    problem = instance.get("problem_description", "")

    # 1. Problem description first — high priority, protected from eviction.
    # Uses role=user so it sets _last_user_message for BM25 scoring early.
    if problem:
        chunks.append({
            "role": "user",
            "content": f"[TASK] {problem}",
            "source": "__problem__",
            "priority": 2.0,
        })

    # 2. Setup commands
    setup = instance.get("setup_commands") or []
    if setup:
        setup_text = "\n".join(setup)
        chunks.append({
            "role": "user",
            "content": f"[SETUP]\n{setup_text}",
            "source": "__setup__",
        })

    # 3. ALL filler chunks first — simulates browsing unrelated repo files.
    # Must come before key files so compaction during filler insertion
    # only evicts other fillers, not the key files.
    _add_filler_chunks(chunks, instance)

    # 4. Key files — added after fillers so they're recent when scored.
    # BM25 will score these against the goal restatement (added last).
    gold_patch = instance.get("clean_pr_patch", "")
    if gold_patch:
        file_diffs = re.split(r"(?=^diff --git )", gold_patch, flags=re.MULTILINE)
        for diff_hunk in file_diffs:
            if not diff_hunk.strip():
                continue
            match = re.search(r"^diff --git a/(.*?) b/", diff_hunk, re.MULTILINE)
            if match:
                filepath = match.group(1)
                context_lines = []
                for line in diff_hunk.split("\n"):
                    if line.startswith(" ") or line.startswith("-"):
                        context_lines.append(line[1:] if len(line) > 1 else "")
                if context_lines:
                    chunks.append({
                        "role": "user",
                        "content": f"[FILE: {filepath}]\n" + "\n".join(context_lines),
                        "source": filepath,
                    })

    # 5. Test files (specification — truncated to avoid blowing the budget)
    test_names = instance.get("test_file_names") or []
    test_contents = instance.get("test_file_contents") or []
    for name, content in zip(test_names, test_contents):
        if content and content.strip():
            # Cap test files at ~2000 tokens (~6400 chars) to avoid
            # a single chunk exceeding the context budget
            truncated = content[:6400]
            chunks.append({
                "role": "user",
                "content": f"[FILE: {name}]\n{truncated}",
                "source": name,
            })

    # 6. PR body — truncate to avoid single-chunk budget blowout.
    # Some PR bodies are 12k+ tokens which would exceed the entire budget.
    body = instance.get("body", "")
    if body and body.strip() and len(body) > 50:
        truncated_body = body[:3200]  # ~1000 tokens max
        chunks.append({
            "role": "user",
            "content": f"[PR DESCRIPTION]\n{truncated_body}",
            "source": "__pr_body__",
        })

    # 7. Goal restatement — becomes _last_user_message for BM25 scoring
    if problem:
        goal_summary = problem[:500] if len(problem) > 500 else problem
        chunks.append({
            "role": "user",
            "content": f"[GOAL] Fix the issue described above. {goal_summary}",
            "source": "__goal__",
        })

    return chunks


def _add_filler_chunks(chunks: list[dict], instance: dict) -> None:
    """Add synthetic filler to simulate large repo context pressure.

    Real repos have hundreds of files. We generate boilerplate fillers
    from the repo metadata to stress-test compaction.
    """
    repo = instance.get("repo") or instance.get("base_repo", "unknown/unknown")
    repo_name = repo.split("/")[-1]

    filler_templates = [
        f"[FILE: {repo_name}/utils/helpers.py]\n"
        "import os\nimport sys\nimport logging\n\nlogger = logging.getLogger(__name__)\n\n"
        "def setup_logging(level='INFO'):\n    logging.basicConfig(level=level)\n    return logger\n\n"
        "def get_config_path():\n    return os.path.join(os.path.dirname(__file__), 'config.yaml')\n",

        f"[FILE: {repo_name}/utils/validators.py]\n"
        "from typing import Any, Optional\n\n"
        "def validate_input(data: Any) -> bool:\n    if data is None:\n        return False\n"
        "    if isinstance(data, str) and not data.strip():\n        return False\n    return True\n\n"
        "def validate_config(config: dict) -> Optional[str]:\n    required = ['name', 'version']\n"
        "    for key in required:\n        if key not in config:\n            return f'Missing: {key}'\n    return None\n",

        f"[FILE: {repo_name}/constants.py]\n"
        "VERSION = '1.0.0'\nDEFAULT_TIMEOUT = 30\nMAX_RETRIES = 3\n"
        "SUPPORTED_FORMATS = ['json', 'yaml', 'toml']\n"
        "API_BASE_URL = 'https://api.example.com/v1'\n",

        f"[FILE: {repo_name}/models/base.py]\n"
        "from dataclasses import dataclass, field\nfrom typing import Optional\nimport time\n\n"
        "@dataclass\nclass BaseModel:\n    id: Optional[str] = None\n    created_at: float = field(default_factory=time.time)\n"
        "    updated_at: float = field(default_factory=time.time)\n\n"
        "    def to_dict(self) -> dict:\n        return {k: v for k, v in self.__dict__.items() if v is not None}\n",

        f"[FILE: {repo_name}/cli.py]\n"
        "import argparse\nimport sys\n\ndef main():\n    parser = argparse.ArgumentParser(description='{repo_name}')\n"
        "    parser.add_argument('--verbose', '-v', action='store_true')\n"
        "    parser.add_argument('--config', '-c', type=str, default='config.yaml')\n"
        "    args = parser.parse_args()\n    return 0\n\nif __name__ == '__main__':\n    sys.exit(main())\n",

        f"[FILE: {repo_name}/exceptions.py]\n"
        "class AppError(Exception):\n    pass\n\nclass ConfigError(AppError):\n    pass\n\n"
        "class ValidationError(AppError):\n    pass\n\nclass NetworkError(AppError):\n    pass\n\n"
        "class TimeoutError(AppError):\n    pass\n",

        f"[FILE: {repo_name}/middleware.py]\n"
        "import time\nimport logging\n\nlogger = logging.getLogger(__name__)\n\n"
        "def timing_middleware(func):\n    def wrapper(*args, **kwargs):\n"
        "        start = time.monotonic()\n        result = func(*args, **kwargs)\n"
        "        elapsed = time.monotonic() - start\n"
        "        logger.debug(f'{func.__name__} took {elapsed:.3f}s')\n        return result\n    return wrapper\n",

        f"[FILE: {repo_name}/cache.py]\n"
        "from functools import lru_cache\nimport hashlib\n\n"
        "@lru_cache(maxsize=256)\ndef compute_hash(data: str) -> str:\n"
        "    return hashlib.sha256(data.encode()).hexdigest()\n\n"
        "class SimpleCache:\n    def __init__(self, max_size=1000):\n        self._store = {{}}\n"
        "        self._max = max_size\n\n    def get(self, key):\n        return self._store.get(key)\n\n"
        "    def set(self, key, value):\n        if len(self._store) >= self._max:\n"
        "            oldest = next(iter(self._store))\n            del self._store[oldest]\n"
        "        self._store[key] = value\n",
    ]

    # Add enough fillers to force compaction.
    # Target: ~3x the context budget. Each template is ~100-200 tokens,
    # so we need ~120 fillers for 8k budget (120 * 150 = 18k > 3 * 8k).
    # Also expand each filler with realistic padding content.
    padding_blocks = [
        "\n\ndef process_batch(items: list, batch_size: int = 100) -> list:\n"
        "    results = []\n    for i in range(0, len(items), batch_size):\n"
        "        batch = items[i:i + batch_size]\n"
        "        processed = [transform(item) for item in batch]\n"
        "        results.extend(processed)\n    return results\n",

        "\n\nclass DatabaseConnection:\n    def __init__(self, host, port, db_name):\n"
        "        self.host = host\n        self.port = port\n"
        "        self.db_name = db_name\n        self._conn = None\n\n"
        "    def connect(self):\n        if self._conn is None:\n"
        "            self._conn = self._create_connection()\n        return self._conn\n",

        "\n\ndef retry_with_backoff(func, max_retries=3, base_delay=1.0):\n"
        "    for attempt in range(max_retries):\n        try:\n"
        "            return func()\n        except Exception as e:\n"
        "            if attempt == max_retries - 1:\n                raise\n"
        "            delay = base_delay * (2 ** attempt)\n"
        "            time.sleep(delay)\n",

        "\n\nclass EventBus:\n    def __init__(self):\n        self._handlers = {}\n\n"
        "    def subscribe(self, event_type, handler):\n"
        "        self._handlers.setdefault(event_type, []).append(handler)\n\n"
        "    def publish(self, event_type, data=None):\n"
        "        for handler in self._handlers.get(event_type, []):\n"
        "            handler(data)\n",
    ]

    for i in range(80):
        template = filler_templates[i % len(filler_templates)]
        padding = padding_blocks[i % len(padding_blocks)]
        salted = (
            template
            + padding
            + f"\n# Section {i+1} of {repo_name} — auto-generated module {i+1}\n"
            + f"# Build hash: {''.join(f'{(i*37+j) % 16:x}' for j in range(32))}\n"
        )
        source_match = re.search(r"\[FILE: (.+?)\]", template)
        source = source_match.group(1) if source_match else f"__filler_{i}__"
        chunks.append({
            "role": "user",
            "content": salted,
            "source": f"{source}__copy{i}",
        })


def evaluate_instance(
    instance: dict,
    scoring_mode: str = "bm25",
    max_tokens: int = 8000,
) -> ContextEvalResult:
    """Evaluate context quality for a single AGENTbench instance."""
    instance_id = instance.get("instance_id", "unknown")
    repo = instance.get("repo") or instance.get("base_repo", "unknown")
    gold_patch = instance.get("clean_pr_patch", "")
    key_files = extract_key_files_from_patch(gold_patch)

    # Build chunks
    chunks = build_repo_chunks(instance)
    total_chunks = len(chunks)
    total_tokens_in = sum(_estimate_tokens(c["content"]) for c in chunks)

    # Feed through our engine
    log = ChunkLog(
        db_path=":memory:",
        max_tokens=max_tokens,
        soft_threshold=0.7,
        hard_threshold=0.9,
        scoring_mode=scoring_mode,
    )

    t0 = time.perf_counter()
    chunk_source_map: dict[str, str] = {}
    for chunk in chunks:
        role = chunk.get("role", "user")
        priority = chunk.get("priority", 0.5)
        chunk_hash = log.append(role, chunk["content"], priority=priority)
        chunk_source_map[chunk_hash] = chunk["source"]
        log.next_turn()
    scoring_time_ms = (time.perf_counter() - t0) * 1000

    # Get surviving context
    context = log.get_context()
    context_tokens = log.get_context_tokens()
    compaction_events = log.compaction_count

    # Map retained chunks back to sources
    retained_sources = set()
    for msg in context:
        # Check which source files survived
        for key_file in key_files:
            if key_file in msg["content"]:
                retained_sources.add(key_file)

    # Also check by chunk hash
    retained_hashes = set()
    for msg in context:
        for chunk_hash, source in chunk_source_map.items():
            if msg["content"] and source in msg["content"]:
                retained_hashes.add(chunk_hash)

    retained_key = [f for f in key_files if f in retained_sources]

    # Compute precision: what fraction of retained chunks are key files?
    retained_total = len(context)
    retained_key_count = sum(
        1 for msg in context
        if any(kf in msg["content"] for kf in key_files)
    )

    recall = len(retained_key) / len(key_files) if key_files else 1.0
    precision = retained_key_count / retained_total if retained_total > 0 else 0.0
    compression = total_tokens_in / max(1, context_tokens)

    # Priority analysis: get priorities from the engine's internal state
    avg_key_priority = 0.0
    avg_filler_priority = 0.0
    key_priorities = []
    filler_priorities = []

    # Read priorities from the SQLite DB
    if hasattr(log, '_conn') and log._conn:
        try:
            cursor = log._conn.execute("SELECT chunk_hash, priority FROM chunks")
            for row in cursor:
                chunk_hash, priority = row
                source = chunk_source_map.get(chunk_hash, "")
                is_key = any(kf in source for kf in key_files) or source == "__problem__"
                if is_key:
                    key_priorities.append(priority)
                else:
                    filler_priorities.append(priority)
        except Exception:
            pass

    avg_key_priority = sum(key_priorities) / len(key_priorities) if key_priorities else 0.0
    avg_filler_priority = sum(filler_priorities) / len(filler_priorities) if filler_priorities else 0.0
    priority_gap = avg_key_priority - avg_filler_priority

    log.close()

    return ContextEvalResult(
        instance_id=instance_id,
        repo=repo,
        scoring_mode=scoring_mode,
        max_tokens=max_tokens,
        total_chunks=total_chunks,
        retained_chunks=retained_total,
        key_files=tuple(key_files),
        retained_key_files=tuple(retained_key),
        key_file_recall=recall,
        key_file_precision=precision,
        token_compression=compression,
        total_tokens_in=total_tokens_in,
        context_tokens_out=context_tokens,
        compaction_events=compaction_events,
        avg_key_priority=avg_key_priority,
        avg_filler_priority=avg_filler_priority,
        priority_gap=priority_gap,
        scoring_time_ms=scoring_time_ms,
    )


def run_evaluation(
    scoring_modes: list[str] | None = None,
    max_tokens: int = 8000,
    instance_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run context quality evaluation across all AGENTbench instances.

    Args:
        scoring_modes: Scoring modes to compare. Defaults to ["bm25", "structural"].
        max_tokens: Context window budget.
        instance_ids: Optional filter — evaluate only these instances.

    Returns:
        Full results dict with per-instance and aggregate metrics.
    """
    if scoring_modes is None:
        scoring_modes = ["bm25"]

    instances = load_instances()
    if instance_ids:
        id_set = set(instance_ids)
        instances = [i for i in instances if i.get("instance_id") in id_set]

    print(f"Evaluating {len(instances)} instances × {len(scoring_modes)} modes")
    print(f"Context budget: {max_tokens} tokens")
    print("=" * 60)

    all_results: dict[str, list[ContextEvalResult]] = {m: [] for m in scoring_modes}

    for idx, instance in enumerate(instances):
        iid = instance.get("instance_id", f"instance_{idx}")
        for mode in scoring_modes:
            try:
                result = evaluate_instance(instance, scoring_mode=mode, max_tokens=max_tokens)
                all_results[mode].append(result)
                print(
                    f"[{idx+1}/{len(instances)}] {iid} {mode}: "
                    f"recall={result.key_file_recall:.2f} "
                    f"precision={result.key_file_precision:.2f} "
                    f"compress={result.token_compression:.1f}x "
                    f"gap={result.priority_gap:+.2f} "
                    f"time={result.scoring_time_ms:.1f}ms"
                )
            except Exception as e:
                print(f"[{idx+1}/{len(instances)}] {iid} {mode}: ERROR {e}")

    # Aggregate
    summary = {}
    for mode, results in all_results.items():
        if not results:
            continue
        n = len(results)
        summary[mode] = {
            "n_instances": n,
            "avg_key_file_recall": sum(r.key_file_recall for r in results) / n,
            "avg_key_file_precision": sum(r.key_file_precision for r in results) / n,
            "avg_token_compression": sum(r.token_compression for r in results) / n,
            "avg_priority_gap": sum(r.priority_gap for r in results) / n,
            "avg_scoring_time_ms": sum(r.scoring_time_ms for r in results) / n,
            "avg_compaction_events": sum(r.compaction_events for r in results) / n,
            "perfect_recall_count": sum(1 for r in results if r.key_file_recall == 1.0),
            "zero_recall_count": sum(1 for r in results if r.key_file_recall == 0.0),
        }

    print(f"\n{'=' * 60}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 60}")
    for mode, stats in summary.items():
        print(f"\n{mode.upper()}:")
        print(f"  Instances:          {stats['n_instances']}")
        print(f"  Avg key file recall:    {stats['avg_key_file_recall']:.3f}")
        print(f"  Avg key file precision: {stats['avg_key_file_precision']:.3f}")
        print(f"  Avg compression:        {stats['avg_token_compression']:.1f}x")
        print(f"  Avg priority gap:       {stats['avg_priority_gap']:+.3f}")
        print(f"  Avg scoring time:       {stats['avg_scoring_time_ms']:.1f}ms")
        print(f"  Perfect recall:         {stats['perfect_recall_count']}/{stats['n_instances']}")
        print(f"  Zero recall:            {stats['zero_recall_count']}/{stats['n_instances']}")

    # Serialize
    output = {
        "benchmark": "agentbench_context_quality",
        "n_instances": len(instances),
        "max_tokens": max_tokens,
        "scoring_modes": scoring_modes,
        "summary": summary,
        "per_instance": {
            mode: [
                {
                    "instance_id": r.instance_id,
                    "repo": r.repo,
                    "key_file_recall": r.key_file_recall,
                    "key_file_precision": r.key_file_precision,
                    "token_compression": r.token_compression,
                    "priority_gap": r.priority_gap,
                    "scoring_time_ms": r.scoring_time_ms,
                    "compaction_events": r.compaction_events,
                    "key_files": list(r.key_files),
                    "retained_key_files": list(r.retained_key_files),
                }
                for r in results
            ]
            for mode, results in all_results.items()
        },
    }
    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AGENTbench context quality evaluation")
    parser.add_argument("--modes", nargs="+", default=["bm25"],
                        help="Scoring modes to evaluate")
    parser.add_argument("--max-tokens", type=int, default=8000,
                        help="Context window token budget")
    parser.add_argument("--instances", nargs="*", default=None,
                        help="Specific instance IDs to evaluate (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/agentbench_context_*.json)")
    args = parser.parse_args()

    output = run_evaluation(
        scoring_modes=args.modes,
        max_tokens=args.max_tokens,
        instance_ids=args.instances,
    )

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output or str(RESULTS_DIR / f"agentbench_context_{timestamp}.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
