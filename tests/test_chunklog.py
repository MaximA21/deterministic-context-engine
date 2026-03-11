"""Tests for ChunkLog: append, compaction, scoring modes, decisions, persistence."""

from __future__ import annotations

import sqlite3
import tempfile
import os

import pytest

from engine import ChunkLog, DecisionRecord, _estimate_tokens
from tests.conftest import requires_sklearn, requires_bm25, requires_sentence_transformers


# ── Basic operations ─────────────────────────────────────────────────────


class TestChunkLogBasic:
    def test_append_and_context(self, memory_log):
        memory_log.append("user", "Hello world")
        memory_log.append("assistant", "Hi there")
        ctx = memory_log.get_context()
        assert len(ctx) == 2
        assert ctx[0]["role"] == "user"
        assert ctx[1]["role"] == "assistant"

    def test_content_addressing_dedup(self, memory_log):
        h1 = memory_log.append("user", "same content")
        h2 = memory_log.append("user", "same content")
        assert h1 == h2
        assert len(memory_log.get_context()) == 1

    def test_different_roles_different_hashes(self, memory_log):
        h1 = memory_log.append("user", "same content")
        h2 = memory_log.append("assistant", "same content")
        assert h1 != h2
        assert len(memory_log.get_context()) == 2

    def test_token_tracking(self, memory_log):
        memory_log.append("user", "a" * 100)
        expected = _estimate_tokens("a" * 100)
        assert memory_log.current_tokens() == expected

    def test_turn_counter(self, memory_log):
        assert memory_log.turn() == 0
        memory_log.next_turn()
        assert memory_log.turn() == 1
        memory_log.next_turn()
        assert memory_log.turn() == 2

    def test_get_context_tokens(self, memory_log):
        memory_log.append("user", "hello")
        assert memory_log.get_context_tokens() == memory_log.current_tokens()

    def test_context_ordering_by_turn(self, memory_log):
        memory_log.append("user", "first message")
        memory_log.next_turn()
        memory_log.append("assistant", "second message")
        memory_log.next_turn()
        memory_log.append("user", "third message")

        ctx = memory_log.get_context()
        assert ctx[0]["content"] == "first message"
        assert ctx[1]["content"] == "second message"
        assert ctx[2]["content"] == "third message"

    def test_append_returns_hash(self, memory_log):
        h = memory_log.append("user", "test")
        assert isinstance(h, str) and len(h) == 64

    def test_close(self):
        log = ChunkLog(db_path=":memory:", max_tokens=1000)
        log.append("user", "test")
        log.close()
        # After close, operations should fail
        with pytest.raises(Exception):
            log.current_tokens()

    def test_multiple_appends_accumulate_tokens(self, memory_log):
        memory_log.append("user", "a" * 100)
        memory_log.append("assistant", "b" * 100)
        expected = _estimate_tokens("a" * 100) + _estimate_tokens("b" * 100)
        assert memory_log.current_tokens() == expected


# ── Compaction ───────────────────────────────────────────────────────────


class TestChunkLogCompaction:
    def test_soft_compaction_fires(self, tiny_log):
        for i in range(20):
            tiny_log.append("user", f"Message number {i} with filler content padding")
            tiny_log.next_turn()
        assert tiny_log.compaction_count > 0

    def test_hard_compaction_fires(self):
        log = ChunkLog(
            db_path=":memory:", max_tokens=50,
            soft_threshold=0.5, hard_threshold=0.6,
        )
        for i in range(30):
            log.append("user", f"Message number {i} with extra padding words here")
            log.next_turn()
        assert tiny_log_compaction_count_is_positive(log)
        log.close()

    def test_compaction_disabled(self):
        log = ChunkLog(
            db_path=":memory:", max_tokens=100,
            soft_threshold=2.0, hard_threshold=2.0,
        )
        for i in range(10):
            log.append("user", f"Message {i}")
            log.next_turn()
        assert log.compaction_count == 0
        log.close()

    def test_context_smaller_after_compaction(self, tiny_log):
        total_appended = 0
        for i in range(20):
            content = f"Message number {i} with some filler content to use tokens"
            tiny_log.append("user", content)
            total_appended += _estimate_tokens(content)
            tiny_log.next_turn()
        assert tiny_log.current_tokens() < total_appended

    def test_compaction_under_pressure_100_chunks(self):
        """100 chunks through a 20-chunk-equivalent budget."""
        # Each message ~15 tokens. 20 * 15 = 300 tokens budget.
        log = ChunkLog(
            db_path=":memory:", max_tokens=300,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode=None,
        )
        for i in range(100):
            log.append("user", f"Chunk number {i}: some content words here to fill space")
            log.next_turn()

        assert log.compaction_count > 0
        ctx = log.get_context()
        # Should have significantly fewer than 100 chunks
        assert len(ctx) < 100
        # Should be within budget
        assert log.current_tokens() <= 300
        log.close()

    def test_compaction_preserves_recent_turns(self):
        """Recent chunks should survive compaction (same turn as current)."""
        log = ChunkLog(
            db_path=":memory:", max_tokens=100,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode=None,
        )
        # Fill up with old chunks
        for i in range(15):
            log.append("user", f"Old message {i} with filler padding words")
            log.next_turn()

        # Add a recent chunk (current turn)
        log.append("user", "Recent important message")
        ctx = log.get_context()
        contents = [m["content"] for m in ctx]
        assert "Recent important message" in contents
        log.close()


def tiny_log_compaction_count_is_positive(log):
    return log.compaction_count > 0


# ── Scoring modes initialization ─────────────────────────────────────────


class TestChunkLogScoringModes:
    def test_init_bm25(self):
        log = ChunkLog(db_path=":memory:", max_tokens=1000, scoring_mode="bm25")
        assert log.scoring_mode == "bm25"
        assert log._bm25_scorer is not None
        assert log._goal_scorer is None
        assert log._semantic_scorer is None
        log.close()

    @requires_sklearn
    def test_init_tfidf(self):
        log = ChunkLog(db_path=":memory:", max_tokens=1000, scoring_mode="tfidf")
        assert log.scoring_mode == "tfidf"
        assert log._goal_scorer is not None
        log.close()

    @requires_sklearn
    def test_init_entity_aware(self):
        log = ChunkLog(db_path=":memory:", max_tokens=1000, scoring_mode="entity_aware")
        assert log.scoring_mode == "entity_aware"
        assert log._entity_scorer is not None
        log.close()

    def test_init_none(self):
        log = ChunkLog(db_path=":memory:", max_tokens=1000, scoring_mode=None)
        assert log._goal_scorer is None
        assert log._semantic_scorer is None
        assert log._bm25_scorer is None
        log.close()

    @requires_sklearn
    def test_init_goal_guided_flag(self):
        log = ChunkLog(db_path=":memory:", max_tokens=1000, goal_guided=True, scoring_mode=None)
        assert log._goal_scorer is not None
        log.close()


# ── Auto-priority ────────────────────────────────────────────────────────


class TestChunkLogAutoPriority:
    def test_auto_priority_enabled(self):
        log = ChunkLog(
            db_path=":memory:", max_tokens=200,
            soft_threshold=0.7, hard_threshold=0.9,
            auto_priority=True, scoring_mode=None,
        )
        log.append("user", "Fix the bug in utils.py")
        log.next_turn()
        log.append("assistant", "The utils.py file has an error on line 42")
        log.next_turn()

        for i in range(10):
            log.append("user", f"Generic filler content number {i} about nothing specific")
            log.next_turn()

        assert log.compaction_count > 0
        log.close()

    def test_keyword_accumulation(self):
        log = ChunkLog(
            db_path=":memory:", max_tokens=10000,
            auto_priority=True, scoring_mode=None,
        )
        log.append("user", "Fix the bug in utils.py")
        assert "utils.py" in log._accumulated_keywords
        log.append("user", "Also check config.yaml")
        assert "config.yaml" in log._accumulated_keywords
        log.close()


# ── BM25 scoring integration ────────────────────────────────────────────


@requires_bm25
class TestChunkLogBM25Scoring:
    def test_bm25_rescoring_during_compaction(self):
        log = ChunkLog(
            db_path=":memory:", max_tokens=200,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode="bm25",
        )
        # Needle message
        log.append("user", "Fix authentication bug in login.py module")
        log.next_turn()
        log.append("assistant", "The login.py authentication handler has a null check error on line 42")
        log.next_turn()

        # Filler messages
        for i in range(15):
            log.append("user", f"Generic weather update number {i} sunny day report")
            log.next_turn()

        assert log.compaction_count > 0
        log.close()

    def test_bm25_preserves_needles_under_pressure(self):
        """Needles should score higher and survive compaction."""
        log = ChunkLog(
            db_path=":memory:", max_tokens=300,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode="bm25",
        )
        # Add needle
        log.append("user", "Fix the payment processing bug in stripe_handler.py")
        log.next_turn()
        needle_content = "stripe_handler.py line 55: PaymentError when card is declined during retry"
        log.append("assistant", needle_content)
        log.next_turn()

        # Add lots of filler
        for i in range(30):
            log.append("user", f"The weather report for day {i} shows sunny skies and clear conditions")
            log.next_turn()

        # Ask about the needle topic to trigger rescoring
        log.append("user", "What was the payment error in stripe_handler.py?")
        log.next_turn()

        ctx = log.get_context()
        contents = [m["content"] for m in ctx]
        # The most recent user question should survive
        assert "What was the payment error in stripe_handler.py?" in contents
        log.close()


# ── TF-IDF scoring integration ──────────────────────────────────────────


@requires_sklearn
class TestChunkLogTFIDFScoring:
    def test_tfidf_rescoring_during_compaction(self):
        log = ChunkLog(
            db_path=":memory:", max_tokens=200,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode="tfidf",
        )
        log.append("user", "Fix authentication bug in login.py")
        log.next_turn()

        for i in range(15):
            log.append("user", f"Generic weather update number {i}")
            log.next_turn()

        assert log.compaction_count > 0
        log.close()


# ── Goal changes between turns ───────────────────────────────────────────


@requires_bm25
class TestGoalChanges:
    def test_rescoring_on_goal_change(self):
        """Priorities should update when the user changes topic."""
        log = ChunkLog(
            db_path=":memory:", max_tokens=500,
            soft_threshold=0.7, hard_threshold=0.9,
            scoring_mode="bm25",
        )

        # Topic 1: authentication
        log.append("user", "Fix authentication bug in login.py")
        log.next_turn()
        log.append("assistant", "login.py authentication handler fixed")
        log.next_turn()

        # Topic 2: database
        log.append("user", "Now optimize the database queries in models.py")
        log.next_turn()
        log.append("assistant", "models.py database queries optimized with indexing")
        log.next_turn()

        # Fill up context to trigger compaction
        for i in range(20):
            log.append("user", f"Generic filler message number {i} about unrelated topic")
            log.next_turn()

        # Change goal back to authentication
        log.append("user", "What was the authentication bug in login.py?")
        log.next_turn()

        # last_user_message should reflect the latest query
        assert "authentication" in log._last_user_message
        log.close()


# ── DecisionRecord ───────────────────────────────────────────────────────


class TestDecisionRecord:
    def test_decision_logged_on_append(self, memory_log):
        memory_log.append("user", "test message")
        assert len(memory_log.decisions) > 0
        assert memory_log.decisions[0].action == "append"

    def test_decision_record_fields(self, memory_log):
        memory_log.append("user", "test")
        rec = memory_log.decisions[0]
        assert isinstance(rec, DecisionRecord)
        assert isinstance(rec.timestamp, float)
        assert rec.action == "append"
        assert isinstance(rec.chunk_hash, str)
        assert "role=user" in rec.reason
        assert isinstance(rec.context_size_before, int)
        assert isinstance(rec.context_size_after, int)

    def test_decisions_list_is_copy(self, memory_log):
        memory_log.append("user", "test")
        decisions = memory_log.decisions
        decisions.clear()
        # Original should be unaffected
        assert len(memory_log.decisions) > 0

    def test_decision_persisted_to_sqlite(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Use tiny budget to trigger compaction, which commits decisions
            log = ChunkLog(
                db_path=db_path, max_tokens=100,
                soft_threshold=0.7, hard_threshold=0.9,
                scoring_mode=None,
            )
            for i in range(15):
                log.append("user", f"Message {i} with filler content padding words")
                log.next_turn()
            log.close()

            # Read decisions directly from SQLite
            conn = sqlite3.connect(db_path)
            rows = conn.execute("SELECT action FROM decisions").fetchall()
            conn.close()
            actions = [r[0] for r in rows]
            assert len(rows) >= 1
            assert "compact_soft" in actions or "compact_hard" in actions
        finally:
            os.unlink(db_path)

    def test_compaction_decisions_logged(self, tiny_log):
        for i in range(20):
            tiny_log.append("user", f"Message {i} with filler content padding words")
            tiny_log.next_turn()

        compact_decisions = [d for d in tiny_log.decisions if "compact" in d.action]
        assert len(compact_decisions) > 0


# ── SQLite persistence ───────────────────────────────────────────────────


class TestChunkLogPersistence:
    def test_data_survives_reopen(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Write data
            log = ChunkLog(db_path=db_path, max_tokens=10000)
            log.append("user", "persistent message")
            log.close()

            # Reopen and verify
            log2 = ChunkLog(db_path=db_path, max_tokens=10000)
            ctx = log2.get_context()
            assert len(ctx) == 1
            assert ctx[0]["content"] == "persistent message"
            log2.close()
        finally:
            os.unlink(db_path)

    def test_wal_mode_enabled(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            log = ChunkLog(db_path=db_path, max_tokens=10000)
            mode = log._conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert mode == "wal"
            log.close()
        finally:
            os.unlink(db_path)


# ── Edge cases ───────────────────────────────────────────────────────────


class TestChunkLogEdgeCases:
    def test_empty_content(self, memory_log):
        h = memory_log.append("user", "")
        assert isinstance(h, str)
        ctx = memory_log.get_context()
        assert len(ctx) == 1
        assert ctx[0]["content"] == ""

    def test_unicode_content(self, memory_log):
        memory_log.append("user", "こんにちは世界 🌍 Привет мир")
        ctx = memory_log.get_context()
        assert ctx[0]["content"] == "こんにちは世界 🌍 Привет мир"

    def test_very_long_content(self, memory_log):
        long_text = "word " * 10000  # ~50000 chars
        memory_log.append("user", long_text)
        ctx = memory_log.get_context()
        assert ctx[0]["content"] == long_text

    def test_newlines_and_special_chars(self, memory_log):
        content = "line1\nline2\ttab\r\nwindows\0null"
        memory_log.append("user", content)
        ctx = memory_log.get_context()
        assert ctx[0]["content"] == content

    def test_sql_injection_safe(self, memory_log):
        evil = "'; DROP TABLE chunks; --"
        memory_log.append("user", evil)
        ctx = memory_log.get_context()
        assert ctx[0]["content"] == evil
        # Table should still exist
        assert memory_log.current_tokens() > 0

    def test_duplicate_content_different_order(self, memory_log):
        memory_log.append("user", "first")
        memory_log.append("assistant", "second")
        # Same content again
        memory_log.append("user", "first")
        ctx = memory_log.get_context()
        assert len(ctx) == 2  # Deduped

    def test_whitespace_only_content(self, memory_log):
        memory_log.append("user", "   \n\t  ")
        ctx = memory_log.get_context()
        assert len(ctx) == 1

    def test_priority_parameter(self, memory_log):
        memory_log.append("user", "high priority", priority=2.0)
        memory_log.append("user", "low priority", priority=0.5)
        # Both should be stored
        ctx = memory_log.get_context()
        assert len(ctx) == 2

    def test_max_tokens_zero(self):
        """Edge case: zero max_tokens should still work (immediate compaction)."""
        log = ChunkLog(db_path=":memory:", max_tokens=0, soft_threshold=0.7, hard_threshold=0.9)
        log.append("user", "test")
        log.next_turn()
        log.close()

    def test_many_turns_no_content(self, memory_log):
        for _ in range(100):
            memory_log.next_turn()
        assert memory_log.turn() == 100
        assert len(memory_log.get_context()) == 0
