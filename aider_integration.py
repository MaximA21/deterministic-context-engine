"""Drop-in context engine integration for Aider.

Replaces Aider's LLM-based ChatSummary with deterministic BM25-scored
compaction from our context engine. Zero model calls for context management.

Usage:
    from aider_integration import patch_aider_coder

    # After creating an aider Coder instance:
    coder = Coder.create(...)
    patch_aider_coder(coder)
    # Now coder uses BM25 compaction instead of LLM summarization

Or as a standalone wrapper:
    from aider_integration import AiderContextEngine

    engine = AiderContextEngine(max_tokens=8000, scoring_mode="bm25")
    engine.add_message("user", "fix the bug in auth.py line 42")
    engine.add_message("assistant", "I'll fix the off-by-one error.")
    managed = engine.get_managed_messages()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from engine import ChunkLog, _estimate_tokens, BM25Scorer

logger = logging.getLogger(__name__)


class ChunkLogSummary:
    """Drop-in replacement for aider.history.ChatSummary.

    Same interface: too_big(), summarize(). But instead of making LLM calls
    to summarize old messages, uses deterministic BM25-scored compaction.

    Aider calls:
        summarizer.too_big(done_messages) -> bool
        summarizer.summarize(done_messages) -> list[dict]

    Our implementation:
        1. Ingests all messages into a ChunkLog
        2. ChunkLog's threshold compaction evicts low-scored chunks
        3. Returns the surviving messages
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        scoring_mode: str = "bm25",
        soft_threshold: float = 0.7,
        hard_threshold: float = 0.9,
        db_path: str = ":memory:",
        decision_log_path: str | None = None,
    ):
        self.max_tokens = max_tokens
        self.scoring_mode = scoring_mode
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.db_path = db_path
        self.decision_log_path = decision_log_path
        self._decision_records: list[dict] = []

    def token_count(self, msg: dict | list) -> int:
        """Estimate tokens for a message or list of messages."""
        if isinstance(msg, list):
            return sum(self.token_count(m) for m in msg)
        content = msg.get("content", "")
        if isinstance(content, list):
            # Vision/multimodal messages
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            content = " ".join(text_parts)
        return _estimate_tokens(str(content))

    def too_big(self, messages: list[dict]) -> bool:
        """Check if messages exceed the token budget."""
        total = sum(self.token_count(m) for m in messages)
        return total > self.max_tokens

    def summarize(self, messages: list[dict], depth: int = 0) -> list[dict]:
        """Replace LLM summarization with deterministic BM25 compaction.

        Ingests all messages into a ChunkLog, which compacts based on
        BM25 scores. Returns the surviving messages in original order.
        """
        total_before = sum(self.token_count(m) for m in messages)

        if total_before <= self.max_tokens:
            return messages

        # Create a fresh ChunkLog for this compaction pass
        chunk_log = ChunkLog(
            db_path=self.db_path,
            max_tokens=self.max_tokens,
            soft_threshold=self.soft_threshold,
            hard_threshold=self.hard_threshold,
            scoring_mode=self.scoring_mode,
        )

        try:
            # Ingest all messages
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Flatten multimodal content to text for scoring
                    text_parts = [
                        p.get("text", "") for p in content if isinstance(p, dict)
                    ]
                    text_content = "\n".join(text_parts)
                else:
                    text_content = str(content)

                if not text_content.strip():
                    continue

                # All messages start at priority 0.5 — BM25 rescoring
                # during compaction will assign real priorities
                chunk_log.append(role, text_content, priority=0.5)
                chunk_log.next_turn()

            # Get surviving messages after compaction
            managed = chunk_log.get_context()
            total_after = chunk_log.get_context_tokens()

            # Log decisions
            decisions = chunk_log.decisions
            if decisions:
                record = {
                    "timestamp": time.time(),
                    "messages_before": len(messages),
                    "messages_after": len(managed),
                    "tokens_before": total_before,
                    "tokens_after": total_after,
                    "compaction_events": chunk_log.compaction_count,
                    "decisions": [
                        {
                            "action": d.action,
                            "reason": d.reason,
                            "size_before": d.context_size_before,
                            "size_after": d.context_size_after,
                        }
                        for d in decisions
                        if d.action.startswith("compact")
                    ],
                }
                self._decision_records.append(record)
                logger.info(
                    f"Context compaction: {len(messages)} msgs ({total_before} tok) "
                    f"→ {len(managed)} msgs ({total_after} tok), "
                    f"{chunk_log.compaction_count} compactions"
                )

                if self.decision_log_path:
                    self._flush_decisions()

        finally:
            chunk_log.close()

        # Ensure trailing assistant message (aider convention)
        if managed and managed[-1]["role"] != "assistant":
            managed.append({"role": "assistant", "content": "Ok."})

        return managed

    def _flush_decisions(self) -> None:
        """Write decision records to a JSON file."""
        if not self.decision_log_path or not self._decision_records:
            return
        path = Path(self.decision_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._decision_records, f, indent=2, default=str)

    @property
    def decision_records(self) -> list[dict]:
        return list(self._decision_records)


class AiderContextEngine:
    """Standalone wrapper for using the context engine with Aider-style messages.

    Useful for testing or for non-monkey-patch integrations.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        scoring_mode: str = "bm25",
        soft_threshold: float = 0.7,
        hard_threshold: float = 0.9,
        db_path: str = ":memory:",
    ):
        self._chunk_log = ChunkLog(
            db_path=db_path,
            max_tokens=max_tokens,
            soft_threshold=soft_threshold,
            hard_threshold=hard_threshold,
            scoring_mode=scoring_mode,
        )

    def add_message(self, role: str, content: str, priority: float = 0.5) -> str:
        """Add a message. Returns the chunk hash."""
        chunk_hash = self._chunk_log.append(role, content, priority=priority)
        self._chunk_log.next_turn()
        return chunk_hash

    def get_managed_messages(self) -> list[dict[str, str]]:
        """Get the current compacted message list."""
        return self._chunk_log.get_context()

    def get_tokens(self) -> int:
        return self._chunk_log.get_context_tokens()

    @property
    def compaction_count(self) -> int:
        return self._chunk_log.compaction_count

    @property
    def decisions(self):
        return self._chunk_log.decisions

    def close(self) -> None:
        self._chunk_log.close()


def patch_aider_coder(coder, scoring_mode: str = "bm25", decision_log_path: str | None = None) -> None:
    """Monkey-patch an aider Coder instance to use our context engine.

    Replaces coder.summarizer with a ChunkLogSummary that uses BM25
    compaction instead of LLM summarization.

    Args:
        coder: An aider Coder instance (from aider.coders.base_coder.Coder)
        scoring_mode: Scoring mode for compaction ("bm25", "tfidf", etc.)
        decision_log_path: Optional path to write decision logs
    """
    # Get the token budget from the model
    max_chat_history_tokens = getattr(
        coder.main_model, "max_chat_history_tokens", 4096
    )

    # Replace the summarizer
    coder.summarizer = ChunkLogSummary(
        max_tokens=max_chat_history_tokens,
        scoring_mode=scoring_mode,
        decision_log_path=decision_log_path,
    )

    logger.info(
        f"Patched aider coder with ChunkLogSummary "
        f"(max_tokens={max_chat_history_tokens}, scoring={scoring_mode})"
    )


def create_patched_coder(**aider_kwargs):
    """Create an aider Coder with our context engine pre-installed.

    Accepts all the same kwargs as aider.coders.Coder.create().
    Additionally accepts:
        context_scoring_mode: str = "bm25"
        context_decision_log: str | None = None

    Returns:
        A Coder instance with ChunkLogSummary installed.
    """
    scoring_mode = aider_kwargs.pop("context_scoring_mode", "bm25")
    decision_log = aider_kwargs.pop("context_decision_log", None)

    from aider.coders import Coder

    coder = Coder.create(**aider_kwargs)
    patch_aider_coder(coder, scoring_mode=scoring_mode, decision_log_path=decision_log)
    return coder
