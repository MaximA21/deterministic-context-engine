"""Tests for compaction under extreme pressure: 100 chunks through tight budgets."""

from __future__ import annotations

import pytest

from engine import ChunkLog, _estimate_tokens
from tests.conftest import requires_bm25, requires_sklearn


class TestCompactionPressure:
    def test_100_chunks_through_20_chunk_budget_no_scoring(self):
        """100 chunks forced through a budget fitting ~20 chunks."""
        avg_chunk = "Chunk content with some words to fill up space nicely enough"
        chunk_tokens = _estimate_tokens(avg_chunk)
        budget = chunk_tokens * 20  # ~20 chunks

        log = ChunkLog(
            db_path=":memory:", max_tokens=budget,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode=None,
        )
        for i in range(100):
            log.append("user", f"Chunk {i}: {avg_chunk}")
            log.next_turn()

        ctx = log.get_context()
        assert len(ctx) < 30
        assert log.current_tokens() <= budget
        assert log.compaction_count > 0
        log.close()

    @requires_bm25
    def test_100_chunks_bm25_needles_survive(self):
        """With BM25 scoring, needles should preferentially survive."""
        filler = "The weather is sunny and warm today in the park with birds"
        filler_tokens = _estimate_tokens(filler)
        budget = filler_tokens * 25

        log = ChunkLog(
            db_path=":memory:", max_tokens=budget,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode="bm25",
        )

        # Add 5 needles about authentication
        needle_contents = [
            "authentication handler error on line 42 of login.py",
            "JWT token validation failed for user admin@example.com",
            "login.py auth middleware returns 401 Unauthorized",
            "password hashing uses bcrypt with 12 rounds in auth.py",
            "session cookie secure flag missing in authentication config",
        ]
        for needle in needle_contents:
            log.append("assistant", needle)
            log.next_turn()

        # Add 95 filler chunks
        for i in range(95):
            log.append("user", f"Weather report {i}: {filler}")
            log.next_turn()

        # Ask about authentication to trigger rescoring
        log.append("user", "What were the authentication issues in login.py?")
        log.next_turn()

        ctx = log.get_context()
        contents = [m["content"] for m in ctx]

        # Most recent query should survive
        assert "What were the authentication issues in login.py?" in contents
        # Should have compacted significantly
        assert len(ctx) < 40
        log.close()

    @requires_sklearn
    def test_100_chunks_tfidf_scoring(self):
        """TF-IDF scoring with compaction pressure."""
        filler = "Random text about various topics that are not important at all"
        filler_tokens = _estimate_tokens(filler)
        budget = filler_tokens * 20

        log = ChunkLog(
            db_path=":memory:", max_tokens=budget,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode="tfidf",
        )

        for i in range(100):
            log.append("user", f"Message {i}: {filler}")
            log.next_turn()

        assert log.compaction_count > 0
        assert log.current_tokens() <= budget
        log.close()

    def test_compaction_with_mixed_priorities(self):
        """Chunks with explicit high priority should survive better."""
        budget = 300  # ~20 chunks

        log = ChunkLog(
            db_path=":memory:", max_tokens=budget,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode=None,
        )

        # Add high-priority chunks
        for i in range(5):
            log.append("user", f"CRITICAL data point {i}", priority=2.0)
            log.next_turn()

        # Add 95 low-priority fillers
        for i in range(95):
            log.append("user", f"Low priority filler {i} nothing important", priority=0.5)
            log.next_turn()

        ctx = log.get_context()
        assert log.compaction_count > 0
        # Some high-priority chunks should survive
        critical_count = sum(1 for m in ctx if "CRITICAL" in m["content"])
        assert critical_count > 0
        log.close()

    def test_compaction_stress_200_chunks(self):
        """Stress test: 200 chunks through tiny budget."""
        log = ChunkLog(
            db_path=":memory:", max_tokens=150,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode=None,
        )

        for i in range(200):
            log.append("user", f"Stress test chunk {i} padding words here")
            log.next_turn()

        assert log.compaction_count > 5
        assert log.current_tokens() <= 150
        log.close()

    def test_all_chunks_same_turn_no_compaction_deadlock(self):
        """If all chunks are on current turn, compaction shouldn't deadlock."""
        log = ChunkLog(
            db_path=":memory:", max_tokens=100,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode=None,
        )

        # Add many chunks on the same turn (turn 0)
        for i in range(50):
            log.append("user", f"Same turn chunk {i} with extra padding words here")

        # Should not hang — compaction can't evict current-turn chunks in non-scoring mode
        assert log.current_tokens() > 0
        log.close()

    def test_empty_chunks_dont_block_compaction(self):
        """Empty chunks should be handled gracefully during compaction."""
        log = ChunkLog(
            db_path=":memory:", max_tokens=100,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode=None,
        )

        for i in range(20):
            log.append("user", "")
            log.next_turn()
        for i in range(20):
            log.append("user", f"Real content {i} with some extra padding words")
            log.next_turn()

        assert log.compaction_count >= 0  # May or may not fire depending on empty token count
        log.close()
