"""LCM (Lossless Context Management) baseline — DAG-based context compression.

Implements the core architecture from:
  Ehrlich & Blackman, "LCM: Lossless Context Management", Voltropy PBC, 2026.
  https://papers.voltropy.com/LCM

Key concepts:
  - Immutable Store: every message persisted verbatim, never modified.
  - Active Context: the window actually sent to the LLM, with pointers to originals.
  - Summary DAG: hierarchical summary nodes that compact older messages while
    retaining lossless pointers to every original.
  - Three-Level Escalation: guaranteed convergence for compaction.
  - Soft/Hard thresholds (τ_soft, τ_hard) for async/blocking compaction.

Since LCM's L1/L2 escalation levels use LLM calls and we need reproducibility
without API keys, this implementation uses extractive summarization as a proxy:
  - L1 (Normal): Extractive sentence ranking (top sentences by TF-IDF-like score)
  - L2 (Aggressive): Key sentence extraction (first + last + highest-scored)
  - L3 (Deterministic): Truncation to 512 tokens — exactly per the paper
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.2))


# ---------------------------------------------------------------------------
# Extractive summarization (proxy for LLM summarization in L1/L2)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _sentence_scores(sentences: list[str]) -> list[float]:
    """Score sentences by a simple TF-IDF-like measure against the document."""
    if not sentences:
        return []
    # Build document frequency
    doc_tokens = Counter()
    sent_tokens = []
    for sent in sentences:
        tokens = set(re.findall(r'\b\w+\b', sent.lower()))
        sent_tokens.append(tokens)
        doc_tokens.update(tokens)

    n = len(sentences)
    scores = []
    for tokens in sent_tokens:
        if not tokens:
            scores.append(0.0)
            continue
        score = sum(
            (1.0 / max(1, doc_tokens[t])) * math.log(n + 1)
            for t in tokens
        ) / len(tokens)
        scores.append(score)
    return scores


def _extractive_summary_l1(text: str, target_tokens: int) -> str:
    """L1 Normal: preserve details via extractive sentence ranking."""
    sentences = _split_sentences(text)
    if not sentences:
        return text[:int(target_tokens * 3.2)]

    scores = _sentence_scores(sentences)
    # Rank by score, keep top sentences in original order
    indexed = sorted(enumerate(scores), key=lambda x: -x[1])
    selected_indices: set[int] = set()
    current_tokens = 0
    for idx, _score in indexed:
        sent_tokens = _estimate_tokens(sentences[idx])
        if current_tokens + sent_tokens > target_tokens:
            break
        selected_indices.add(idx)
        current_tokens += sent_tokens

    if not selected_indices:
        selected_indices.add(indexed[0][0])

    result = " ".join(sentences[i] for i in sorted(selected_indices))
    return result


def _extractive_summary_l2(text: str, target_tokens: int) -> str:
    """L2 Aggressive: bullet-point style — first, last, and top-scored sentences."""
    sentences = _split_sentences(text)
    if not sentences:
        return text[:int(target_tokens * 3.2)]

    scores = _sentence_scores(sentences)
    # Always include first and last sentence
    selected_indices = {0, len(sentences) - 1}
    current_tokens = _estimate_tokens(sentences[0])
    if len(sentences) > 1:
        current_tokens += _estimate_tokens(sentences[-1])

    # Fill remaining budget with top-scored sentences
    indexed = sorted(enumerate(scores), key=lambda x: -x[1])
    for idx, _score in indexed:
        if idx in selected_indices:
            continue
        sent_tokens = _estimate_tokens(sentences[idx])
        if current_tokens + sent_tokens > target_tokens:
            break
        selected_indices.add(idx)
        current_tokens += sent_tokens

    result = " ".join(sentences[i] for i in sorted(selected_indices))
    return result


def _deterministic_truncate(text: str, max_tokens: int = 512) -> str:
    """L3 Deterministic: truncate to max_tokens. No LLM involved."""
    char_limit = int(max_tokens * 3.2)
    if len(text) <= char_limit:
        return text
    return text[:char_limit]


# ---------------------------------------------------------------------------
# DAG Node
# ---------------------------------------------------------------------------

@dataclass
class SummaryNode:
    """A node in the hierarchical summary DAG.

    Leaf nodes point to original messages in the immutable store.
    Interior nodes are summaries that cover a block of children.
    """
    node_id: str
    summary_text: str
    tokens: int
    children: list[str] = field(default_factory=list)  # child node_ids
    parent_id: str | None = None
    level: int = 0  # 0 = leaf (original message), 1+ = summary
    timestamp: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.level == 0


# ---------------------------------------------------------------------------
# Immutable Store
# ---------------------------------------------------------------------------

@dataclass
class ImmutableMessage:
    """A verbatim message in the immutable store."""
    msg_id: str
    role: str
    content: str
    tokens: int
    turn: int
    timestamp: float
    node_id: str  # corresponding leaf node in the DAG


# ---------------------------------------------------------------------------
# LCM Context Manager
# ---------------------------------------------------------------------------

class LCMContextManager:
    """DAG-based context compression following the LCM architecture.

    Args:
        max_tokens: Maximum active context window size.
        soft_threshold: Ratio triggering async compaction (τ_soft).
        hard_threshold: Ratio triggering blocking compaction (τ_hard).
        summary_block_size: Number of consecutive items to compact at once.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        soft_threshold: float = 0.7,
        hard_threshold: float = 0.9,
        summary_block_size: int = 4,
    ):
        self.max_tokens = max_tokens
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.summary_block_size = summary_block_size

        # Immutable store: msg_id -> ImmutableMessage
        self._store: dict[str, ImmutableMessage] = {}
        # DAG nodes: node_id -> SummaryNode
        self._dag: dict[str, SummaryNode] = {}
        # Active context: ordered list of node_ids currently in the window
        self._active: list[str] = []

        self._turn = 0
        self._compaction_count = 0
        self._summary_counter = 0

    @property
    def compaction_count(self) -> int:
        return self._compaction_count

    def current_tokens(self) -> int:
        """Total tokens in the active context window."""
        return sum(self._dag[nid].tokens for nid in self._active if nid in self._dag)

    def append(self, role: str, content: str, priority: float = 1.0) -> str:
        """Append a message. Returns the message ID.

        Priority is accepted for API compatibility but ignored — LCM uses
        structural DAG compaction rather than priority scoring.
        """
        msg_id = _sha256(f"{role}:{content}:{self._turn}")
        tokens = _estimate_tokens(content)
        now = time.time()

        # Create leaf node in DAG
        node_id = f"leaf_{msg_id[:16]}"
        node = SummaryNode(
            node_id=node_id,
            summary_text=content,
            tokens=tokens,
            level=0,
            timestamp=now,
        )
        self._dag[node_id] = node

        # Persist in immutable store
        msg = ImmutableMessage(
            msg_id=msg_id,
            role=role,
            content=content,
            tokens=tokens,
            turn=self._turn,
            timestamp=now,
            node_id=node_id,
        )
        self._store[msg_id] = msg

        # Add to active context
        self._active.append(node_id)

        # Run compaction control loop (Figure 2 from paper)
        self._control_loop()

        return msg_id

    def next_turn(self) -> None:
        self._turn += 1

    def get_active_content(self) -> list[dict[str, str]]:
        """Return the active context as a list of {role, content} dicts."""
        result = []
        for node_id in self._active:
            node = self._dag.get(node_id)
            if node is None:
                continue
            # For summary nodes, prefix with [Summary L{level}]
            if node.level > 0:
                result.append({
                    "role": "system",
                    "content": f"[Summary L{node.level}] {node.summary_text}",
                })
            else:
                # Look up the original message for role info
                msg = self._find_message_for_node(node_id)
                role = msg.role if msg else "user"
                result.append({"role": role, "content": node.summary_text})
        return result

    def get_active_text(self) -> str:
        """Return active context as concatenated text."""
        parts = []
        for item in self.get_active_content():
            parts.append(item["content"])
        return "\n\n".join(parts)

    def lcm_grep(self, pattern: str, summary_id: str | None = None) -> list[dict[str, Any]]:
        """Regex search across the full immutable store.

        Returns matching messages grouped by the summary node that covers them.
        """
        regex = re.compile(pattern, re.IGNORECASE)
        results = []
        for msg in self._store.values():
            if summary_id and not self._is_under_summary(msg.node_id, summary_id):
                continue
            if regex.search(msg.content):
                covering = self._find_covering_summary(msg.node_id)
                results.append({
                    "msg_id": msg.msg_id,
                    "role": msg.role,
                    "turn": msg.turn,
                    "content_preview": msg.content[:200],
                    "covering_summary": covering,
                })
        return results

    def lcm_expand(self, summary_id: str) -> list[str]:
        """Expand a summary node into its constituent original messages."""
        node = self._dag.get(summary_id)
        if node is None:
            return []
        if node.is_leaf:
            return [node.summary_text]
        # Recursively expand children
        result = []
        for child_id in node.children:
            result.extend(self.lcm_expand(child_id))
        return result

    # -------------------------------------------------------------------
    # Control Loop (Figure 2 from paper)
    # -------------------------------------------------------------------

    def _control_loop(self) -> None:
        """LCM Context Control Loop per Figure 2."""
        current = self.current_tokens()
        soft_limit = int(self.max_tokens * self.soft_threshold)
        hard_limit = int(self.max_tokens * self.hard_threshold)

        # Soft threshold: trigger compaction (async in real LCM, sync here)
        if current > soft_limit:
            self._compact_oldest_block()

        # Hard threshold: block and compact until under limit
        while self.current_tokens() > hard_limit:
            compacted = self._compact_oldest_block()
            if not compacted:
                break  # Nothing left to compact

    def _compact_oldest_block(self) -> bool:
        """Identify the oldest block in active context and compact it.

        Returns True if compaction occurred.
        """
        if len(self._active) < 2:
            return False

        self._compaction_count += 1

        # Take the oldest `summary_block_size` items
        block_size = min(self.summary_block_size, len(self._active) - 1)
        # Never compact the most recent item
        block_ids = self._active[:block_size]

        # Collect text from the block
        block_texts = []
        block_tokens = 0
        for nid in block_ids:
            node = self._dag.get(nid)
            if node:
                block_texts.append(node.summary_text)
                block_tokens += node.tokens

        combined = "\n\n".join(block_texts)

        # Three-Level Escalation (Figure 3 from paper)
        summary = self._escalated_summary(combined, block_tokens)

        # Create summary node
        self._summary_counter += 1
        summary_id = f"summary_{self._summary_counter}"
        max_level = max(
            (self._dag[nid].level for nid in block_ids if nid in self._dag),
            default=0,
        )
        summary_node = SummaryNode(
            node_id=summary_id,
            summary_text=summary,
            tokens=_estimate_tokens(summary),
            children=list(block_ids),
            level=max_level + 1,
            timestamp=time.time(),
        )
        self._dag[summary_id] = summary_node

        # Update parent pointers
        for nid in block_ids:
            if nid in self._dag:
                self._dag[nid].parent_id = summary_id

        # Replace block in active context with the summary pointer
        insert_pos = self._active.index(block_ids[0])
        for nid in block_ids:
            self._active.remove(nid)
        self._active.insert(insert_pos, summary_id)

        return True

    def _escalated_summary(self, text: str, original_tokens: int) -> str:
        """Three-Level Summarization Escalation (Figure 3).

        Guaranteed convergence: L3 always produces fewer tokens.
        """
        # Target: reduce to ~50% of original tokens
        target = max(original_tokens // 2, 50)

        # Level 1: Normal — extractive, preserve details
        s1 = _extractive_summary_l1(text, target)
        if _estimate_tokens(s1) < original_tokens:
            return s1

        # Level 2: Aggressive — bullet points
        s2 = _extractive_summary_l2(text, target // 2)
        if _estimate_tokens(s2) < original_tokens:
            return s2

        # Level 3: Deterministic truncate — guaranteed convergence
        return _deterministic_truncate(text, 512)

    # -------------------------------------------------------------------
    # DAG navigation helpers
    # -------------------------------------------------------------------

    def _find_message_for_node(self, node_id: str) -> ImmutableMessage | None:
        for msg in self._store.values():
            if msg.node_id == node_id:
                return msg
        return None

    def _find_covering_summary(self, node_id: str) -> str | None:
        """Find the summary node that currently covers this leaf."""
        node = self._dag.get(node_id)
        if node is None:
            return None
        if node.parent_id:
            return node.parent_id
        return None

    def _is_under_summary(self, node_id: str, summary_id: str) -> bool:
        """Check if node_id is a descendant of summary_id in the DAG."""
        summary = self._dag.get(summary_id)
        if summary is None:
            return False
        if node_id in summary.children:
            return True
        return any(
            self._is_under_summary(node_id, child_id)
            for child_id in summary.children
        )

    def close(self) -> None:
        """Cleanup (no-op for in-memory implementation)."""
        pass
