"""Context engine with append-only ChunkLog, compaction, and CerebrasSession wrapper.

Manages what stays in LLM context via:
- Append-only logging with SHA-256 content-addressing
- Soft/hard threshold compaction
- DecisionRecords for auditability
- SQLite WAL for persistence
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# --- AutoPriority ---

_RE_FILENAMES = re.compile(r'\b[\w\-]+\.(?:py|js|ts|rs|go|java|c|cpp|h|rb|sh|yaml|yml|json|sql|toml|cfg|md|txt|html|css)\b')
_RE_FUNC_CLASS = re.compile(r'\b(?:def|class|function|fn)\s+(\w+)')
_RE_ERROR_INDICATORS = re.compile(r'\b(?:Error|Exception|bug|fix|fail|issue|traceback|panic|IMPORTANT|CRITICAL|UPDATE)\b', re.IGNORECASE)
_RE_QUOTED = re.compile(r'(?:`([^`]+)`|"([^"]+)"|\'([^\']+)\')')
_RE_IP_ADDR = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
_RE_DATE_PATTERNS = re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?\b', re.IGNORECASE)


def extract_keywords(message: str) -> set[str]:
    """Extract searchable keywords from a message using regex patterns.

    Extracts: filenames, function/class names, error indicators,
    quoted/backtick contents, IP addresses, date patterns.
    """
    keywords: set[str] = set()

    # Filenames
    for m in _RE_FILENAMES.finditer(message):
        keywords.add(m.group(0).lower())

    # Function/class names
    for m in _RE_FUNC_CLASS.finditer(message):
        keywords.add(m.group(1).lower())

    # Error indicators
    for m in _RE_ERROR_INDICATORS.finditer(message):
        keywords.add(m.group(0).lower())

    # Quoted and backtick contents
    for m in _RE_QUOTED.finditer(message):
        content = m.group(1) or m.group(2) or m.group(3)
        if content and len(content) > 2:
            keywords.add(content.lower())

    # IP addresses
    for m in _RE_IP_ADDR.finditer(message):
        keywords.add(m.group(0))

    # Date patterns
    for m in _RE_DATE_PATTERNS.finditer(message):
        keywords.add(m.group(0).lower())

    # Also extract individual significant words from the message (nouns, proper names)
    # Look for capitalized words that aren't at sentence starts
    words = message.split()
    for i, word in enumerate(words):
        clean = re.sub(r'[^\w]', '', word)
        if clean and len(clean) > 2:
            # Numbers that look like specific values (e.g., "250", "3am")
            if re.match(r'^\d+[a-z]*$', clean) and len(clean) <= 8:
                keywords.add(clean.lower())

    return keywords


def score_chunk(chunk_text: str, keywords: set[str]) -> float:
    """Score a chunk based on keyword matches.

    Returns priority: 0.5 (no matches) to 2.0 (3+ matches).
    Linear interpolation between.
    """
    if not keywords:
        return 0.5

    chunk_lower = chunk_text.lower()
    matches = sum(1 for kw in keywords if kw in chunk_lower)

    if matches == 0:
        return 0.5
    elif matches >= 3:
        return 2.0
    else:
        # Linear interpolation: 0 matches -> 0.5, 3 matches -> 2.0
        return 0.5 + (matches / 3.0) * 1.5


# --- Goal-Guided TF-IDF Scorer ---

class GoalGuidedScorer:
    """Scores chunks by TF-IDF goal similarity + corpus uniqueness.

    Two signals combined:
    1. Goal alignment: cosine similarity to the last user message
    2. Uniqueness: how different a chunk is from all other chunks
       (1 - avg similarity to peers). Needles contain rare specific details
       (line numbers, error names) that score high; filler repeats generic
       themes that score low.

    This handles adversarial cases where keyword overlap is high.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    def score_chunks(
        self, goal: str, chunks: list[tuple[str, str]], keyword_scores: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Score chunks by goal similarity + uniqueness.

        Args:
            goal: The goal/query message (last user message).
            chunks: List of (chunk_hash, content) tuples.
            keyword_scores: Optional dict of chunk_hash -> keyword score [0.5, 2.0].

        Returns:
            Dict of chunk_hash -> final_score in [0.5, 2.0].
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        if not chunks:
            return {}

        # Build corpus: goal first, then all chunks
        texts = [goal] + [content for _, content in chunks]

        try:
            tfidf_matrix = self._vectorizer.fit_transform(texts)
        except ValueError:
            return {h: 0.5 for h, _ in chunks}

        goal_vec = tfidf_matrix[0:1]
        chunk_vecs = tfidf_matrix[1:]
        n = chunk_vecs.shape[0]

        # Signal 1: Goal alignment — cosine similarity to goal
        goal_sims = cosine_similarity(goal_vec, chunk_vecs)[0]

        # Signal 2: Uniqueness — 1 - average similarity to all other chunks
        if n > 1:
            pairwise = cosine_similarity(chunk_vecs)
            # Zero out diagonal (self-similarity)
            np.fill_diagonal(pairwise, 0.0)
            avg_peer_sim = pairwise.sum(axis=1) / (n - 1)
            uniqueness = 1.0 - avg_peer_sim
        else:
            uniqueness = np.array([1.0])

        result: dict[str, float] = {}
        for i, (chunk_hash, _) in enumerate(chunks):
            # Blend: 40% goal alignment + 60% uniqueness
            # Uniqueness dominates because during filler turns, goal alignment
            # can't discriminate (goal IS filler), but uniqueness always can.
            blended = 0.4 * goal_sims[i] + 0.6 * uniqueness[i]

            # Map to [0.5, 2.0]
            score = 0.5 + blended * 1.5

            # Clamp
            result[chunk_hash] = max(0.5, min(2.0, score))

        return result


@dataclass(frozen=True)
class DecisionRecord:
    timestamp: float
    action: str  # "append", "compact_soft", "compact_hard", "drop"
    chunk_hash: str
    reason: str
    context_size_before: int
    context_size_after: int


@dataclass
class ChunkEntry:
    chunk_hash: str
    role: str
    content: str
    tokens: int
    turn: int
    priority: float
    timestamp: float


class ChunkLog:
    """Append-only context log with soft/hard threshold compaction.

    Args:
        db_path: SQLite database path (":memory:" for in-memory).
        max_tokens: Maximum context window size in tokens.
        soft_threshold: Ratio of max_tokens that triggers soft compaction (summarize old).
        hard_threshold: Ratio of max_tokens that triggers hard compaction (drop low-priority).
            Set both to 2.0 to effectively disable compaction.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        max_tokens: int = 128_000,
        soft_threshold: float = 0.7,
        hard_threshold: float = 0.9,
        auto_priority: bool = False,
        goal_guided: bool = False,
    ):
        self.db_path = db_path
        self.max_tokens = max_tokens
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.auto_priority = auto_priority
        self.goal_guided = goal_guided
        self._turn = 0
        self._compaction_count = 0
        self._decisions: list[DecisionRecord] = []
        self._last_user_message: str = ""
        self._accumulated_keywords: set[str] = set()
        self._goal_scorer: GoalGuidedScorer | None = GoalGuidedScorer() if goal_guided else None

        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                chunk_hash TEXT PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER NOT NULL,
                turn INTEGER NOT NULL,
                priority REAL NOT NULL DEFAULT 1.0,
                timestamp REAL NOT NULL
            )"""
        )
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                action TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                reason TEXT NOT NULL,
                context_size_before INTEGER NOT NULL,
                context_size_after INTEGER NOT NULL
            )"""
        )
        self._conn.commit()

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    @property
    def decisions(self) -> list[DecisionRecord]:
        return list(self._decisions)

    def current_tokens(self) -> int:
        row = self._conn.execute("SELECT COALESCE(SUM(tokens), 0) FROM chunks").fetchone()
        return row[0]

    def turn(self) -> int:
        return self._turn

    def append(self, role: str, content: str, priority: float = 1.0) -> str:
        """Append content to the log. Returns the chunk hash."""
        chunk_hash = _sha256(f"{role}:{content}")
        tokens = _estimate_tokens(content)
        now = time.time()

        # Track last user message and accumulate keywords for auto-priority / goal-guided
        if role == "user":
            self._last_user_message = content
            if self.auto_priority or self.goal_guided:
                self._accumulated_keywords.update(extract_keywords(content))

        # Content-addressed: skip if already present
        existing = self._conn.execute(
            "SELECT chunk_hash FROM chunks WHERE chunk_hash = ?", (chunk_hash,)
        ).fetchone()
        if existing:
            return chunk_hash

        size_before = self.current_tokens()
        self._conn.execute(
            "INSERT INTO chunks (chunk_hash, role, content, tokens, turn, priority, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chunk_hash, role, content, tokens, self._turn, priority, now),
        )
        self._conn.commit()

        self._record_decision("append", chunk_hash, f"role={role} tokens={tokens}", size_before, size_before + tokens)

        # Check compaction thresholds
        self._maybe_compact()
        return chunk_hash

    def next_turn(self) -> None:
        self._turn += 1

    def get_context(self) -> list[dict[str, str]]:
        """Get current context as a list of messages, ordered by turn then timestamp."""
        rows = self._conn.execute(
            "SELECT role, content FROM chunks ORDER BY turn ASC, timestamp ASC"
        ).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def get_context_tokens(self) -> int:
        return self.current_tokens()

    def _rescore_chunks_auto(self) -> None:
        """Re-score all chunks based on accumulated keywords from user messages."""
        if not self._accumulated_keywords:
            return
        rows = self._conn.execute(
            "SELECT chunk_hash, content FROM chunks"
        ).fetchall()
        for chunk_hash, content in rows:
            new_priority = score_chunk(content, self._accumulated_keywords)
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (new_priority, chunk_hash),
            )
        self._conn.commit()

    def _rescore_chunks_goal_guided(self) -> None:
        """Re-score all chunks using TF-IDF similarity to last user message."""
        if not self._last_user_message or not self._goal_scorer:
            return
        rows = self._conn.execute(
            "SELECT chunk_hash, content FROM chunks"
        ).fetchall()
        if not rows:
            return

        # Pure TF-IDF scoring — subsumes keyword matching and handles
        # adversarial cases where keyword overlap makes keyword scores useless
        scores = self._goal_scorer.score_chunks(
            self._last_user_message, rows, keyword_scores=None
        )

        for chunk_hash, new_priority in scores.items():
            self._conn.execute(
                "UPDATE chunks SET priority = ? WHERE chunk_hash = ?",
                (new_priority, chunk_hash),
            )
        self._conn.commit()

    def _maybe_compact(self) -> None:
        # Thresholds > 1.0 mean "never compact"
        if self.soft_threshold > 1.0 and self.hard_threshold > 1.0:
            return
        current = self.current_tokens()
        hard_limit = int(self.max_tokens * self.hard_threshold)
        soft_limit = int(self.max_tokens * self.soft_threshold)

        if current > hard_limit or current > soft_limit:
            # Re-score chunks before compaction
            if self.goal_guided:
                self._rescore_chunks_goal_guided()
            elif self.auto_priority:
                self._rescore_chunks_auto()

        if current > hard_limit:
            self._compact_hard(current)
        elif current > soft_limit:
            self._compact_soft(current)

    def _compact_soft(self, current_tokens: int) -> None:
        """Soft compaction: drop oldest low-priority chunks."""
        self._compaction_count += 1
        target = int(self.max_tokens * self.soft_threshold * 0.8)
        rows = self._conn.execute(
            "SELECT chunk_hash, tokens, priority, turn FROM chunks ORDER BY priority ASC, turn ASC"
        ).fetchall()

        removed = 0
        use_scoring = self.auto_priority or self.goal_guided
        for chunk_hash, tokens, priority, turn in rows:
            if current_tokens - removed <= target:
                break
            # With auto_priority/goal_guided, trust the scoring — evict any low-priority chunk
            if use_scoring:
                if priority < 1.5:
                    self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                    removed += tokens
                    self._record_decision(
                        "compact_soft", chunk_hash,
                        f"priority={priority:.2f} turn={turn}",
                        current_tokens - removed + tokens, current_tokens - removed,
                    )
            else:
                if priority < 1.5 and turn < self._turn - 1:
                    self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                    removed += tokens
                    self._record_decision(
                        "compact_soft", chunk_hash,
                        f"priority={priority} turn={turn}",
                        current_tokens - removed + tokens, current_tokens - removed,
                    )
        self._conn.commit()

    def _compact_hard(self, current_tokens: int) -> None:
        """Hard compaction: aggressively drop to get below threshold."""
        self._compaction_count += 1
        target = int(self.max_tokens * self.soft_threshold * 0.6)
        rows = self._conn.execute(
            "SELECT chunk_hash, tokens, priority, turn FROM chunks ORDER BY priority ASC, turn ASC"
        ).fetchall()

        removed = 0
        use_scoring = self.auto_priority or self.goal_guided
        # First pass: try to evict only low-priority chunks (scoring protection)
        if use_scoring:
            for chunk_hash, tokens, priority, turn in rows:
                if current_tokens - removed <= target:
                    break
                if priority >= 1.5:
                    continue
                if turn < self._turn:
                    self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                    removed += tokens
                    self._record_decision(
                        "compact_hard", chunk_hash,
                        f"priority={priority:.2f} turn={turn}",
                        current_tokens - removed + tokens, current_tokens - removed,
                    )
            # If still over hard threshold, fall back to standard eviction (no priority floor)
            if current_tokens - removed > int(self.max_tokens * self.hard_threshold):
                rows = self._conn.execute(
                    "SELECT chunk_hash, tokens, priority, turn FROM chunks ORDER BY priority ASC, turn ASC"
                ).fetchall()
                for chunk_hash, tokens, priority, turn in rows:
                    if current_tokens - removed <= target:
                        break
                    if turn < self._turn:
                        self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                        removed += tokens
                        self._record_decision(
                            "compact_hard", chunk_hash,
                            f"priority={priority:.2f} turn={turn} (fallback)",
                            current_tokens - removed + tokens, current_tokens - removed,
                        )
        else:
            for chunk_hash, tokens, priority, turn in rows:
                if current_tokens - removed <= target:
                    break
                if turn < self._turn:
                    self._conn.execute("DELETE FROM chunks WHERE chunk_hash = ?", (chunk_hash,))
                    removed += tokens
                    self._record_decision(
                        "compact_hard", chunk_hash,
                        f"priority={priority} turn={turn}",
                        current_tokens - removed + tokens, current_tokens - removed,
                    )
        self._conn.commit()

    def _record_decision(self, action: str, chunk_hash: str, reason: str, before: int, after: int) -> None:
        now = time.time()
        rec = DecisionRecord(
            timestamp=now, action=action, chunk_hash=chunk_hash,
            reason=reason, context_size_before=before, context_size_after=after,
        )
        self._decisions.append(rec)
        self._conn.execute(
            "INSERT INTO decisions (timestamp, action, chunk_hash, reason, context_size_before, context_size_after) VALUES (?, ?, ?, ?, ?, ?)",
            (now, action, chunk_hash, reason, before, after),
        )

    def close(self) -> None:
        self._conn.close()


class CerebrasSession:
    """Wrapper around Cerebras API with context management via ChunkLog.

    Tracks TTFT, token usage, and handles rate limiting with exponential backoff.
    """

    def __init__(
        self,
        chunk_log: ChunkLog,
        model: str = "llama3.1-8b",
        api_key: str | None = None,
        max_retries: int = 5,
    ):
        from cerebras.cloud.sdk import Cerebras

        self.chunk_log = chunk_log
        self.model = model
        self.max_retries = max_retries
        self._client = Cerebras(api_key=api_key or os.environ.get("CEREBRAS_API_KEY"))
        self._ttft_samples: list[float] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_turns = 0

    @property
    def avg_ttft(self) -> float:
        if not self._ttft_samples:
            return 0.0
        return sum(self._ttft_samples) / len(self._ttft_samples)

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_turns(self) -> int:
        return self._total_turns

    def chat(self, system_prompt: str | None = None) -> str:
        """Send current context to Cerebras and return the response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.chunk_log.get_context())

        response_text = ""
        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=16384,
                    temperature=0.0,
                )
                ttft = time.time() - t0
                self._ttft_samples.append(ttft)

                choice = response.choices[0]
                response_text = choice.message.content or ""
                # For reasoning models: if content is empty, include reasoning as fallback
                if not response_text and hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                    response_text = choice.message.reasoning

                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens
                    self._total_output_tokens += response.usage.completion_tokens

                self._total_turns += 1
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Cerebras API failed after {self.max_retries} retries: {e}") from e

        # Append assistant response to context
        self.chunk_log.append("assistant", response_text, priority=1.0)
        self.chunk_log.next_turn()
        return response_text

    def get_metrics(self) -> dict[str, Any]:
        return {
            "avg_ttft": round(self.avg_ttft, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_turns": self.total_turns,
            "context_size_tokens": self.chunk_log.get_context_tokens(),
            "compaction_events": self.chunk_log.compaction_count,
        }
