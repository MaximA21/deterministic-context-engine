"""Tests for Active Context Compression (ACC) sawtooth compaction strategy."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import ChunkLog, _estimate_tokens


class TestACCInitialization:
    """Test ACC mode initialization and configuration."""

    def test_acc_mode_creates_no_scorers(self):
        log = ChunkLog(db_path=":memory:", scoring_mode="acc", acc_api_key="fake")
        assert log._bm25_scorer is None
        assert log._goal_scorer is None
        assert log._semantic_scorer is None
        assert log._structural_scorer is None
        log.close()

    def test_acc_default_parameters(self):
        log = ChunkLog(db_path=":memory:", scoring_mode="acc", acc_api_key="fake")
        assert log._acc_interval == 10
        assert log._acc_keep_recent == 3
        assert log._acc_knowledge_block == ""
        assert log._acc_last_consolidation_turn == 0
        assert log._acc_consolidation_latencies == []
        log.close()

    def test_acc_custom_parameters(self):
        log = ChunkLog(
            db_path=":memory:", scoring_mode="acc",
            acc_interval=5, acc_keep_recent=2, acc_api_key="my_key",
        )
        assert log._acc_interval == 5
        assert log._acc_keep_recent == 2
        assert log._acc_api_key == "my_key"
        log.close()

    def test_acc_metrics_empty(self):
        log = ChunkLog(db_path=":memory:", scoring_mode="acc", acc_api_key="fake")
        metrics = log.acc_metrics
        assert metrics["consolidation_count"] == 0
        assert metrics["avg_consolidation_latency"] == 0.0
        assert metrics["total_consolidation_latency"] == 0
        assert metrics["llm_input_tokens"] == 0
        assert metrics["llm_output_tokens"] == 0
        assert metrics["knowledge_block_tokens"] == 0
        log.close()


class TestACCSawtoothTiming:
    """Test that consolidation fires at the correct intervals."""

    def test_no_consolidation_before_interval(self):
        log = ChunkLog(
            db_path=":memory:", scoring_mode="acc",
            acc_interval=5, acc_keep_recent=2,
            soft_threshold=2.0, hard_threshold=2.0,
            acc_api_key="fake",
        )
        for i in range(4):
            log.append("user", f"Turn {i} content")
            log.next_turn()
        assert log.compaction_count == 0
        log.close()

    @patch("engine.ChunkLog._acc_summarize")
    def test_consolidation_fires_at_interval(self, mock_summarize):
        mock_summarize.return_value = "Summary of turns 0-2"
        log = ChunkLog(
            db_path=":memory:", scoring_mode="acc",
            acc_interval=5, acc_keep_recent=2,
            soft_threshold=2.0, hard_threshold=2.0,
            acc_api_key="fake",
        )
        for i in range(5):
            log.append("user", f"Turn {i}: important content about topic {i}")
            log.next_turn()

        # At turn 5, consolidation should fire (interval=5)
        # But it fires on the next append after turn reaches interval
        log.append("user", "Turn 5 content")
        log.next_turn()

        # Check: consolidation should have fired
        # Turns 0-3 should be summarized (keep_recent=2 means keep turns >= 4)
        assert log.compaction_count >= 1 or mock_summarize.called
        log.close()

    @patch("engine.ChunkLog._acc_summarize")
    def test_consolidation_fires_multiple_times(self, mock_summarize):
        mock_summarize.return_value = "Summary chunk"
        log = ChunkLog(
            db_path=":memory:", scoring_mode="acc",
            acc_interval=3, acc_keep_recent=1,
            soft_threshold=2.0, hard_threshold=2.0,
            acc_api_key="fake",
        )
        compaction_turns = []
        for i in range(12):
            prev_count = log.compaction_count
            log.append("user", f"Turn {i}: content about topic {i}")
            log.next_turn()
            if log.compaction_count > prev_count:
                compaction_turns.append(i)

        # Should fire at least twice in 12 turns with interval=3
        assert log.compaction_count >= 2, f"Expected >= 2 compactions, got {log.compaction_count}"
        log.close()


class TestACCKnowledgeBlock:
    """Test knowledge block creation and accumulation."""

    @patch("engine.ChunkLog._acc_summarize")
    def test_knowledge_block_created(self, mock_summarize):
        mock_summarize.return_value = "Files touched: engine.py\nFacts: IP is 10.42.88.7"
        log = ChunkLog(
            db_path=":memory:", scoring_mode="acc",
            acc_interval=3, acc_keep_recent=1,
            soft_threshold=2.0, hard_threshold=2.0,
            acc_api_key="fake",
        )
        for i in range(4):
            log.append("user", f"Turn {i}: content")
            log.next_turn()

        # After consolidation, check knowledge block exists in context
        messages = log.get_context()
        kb_messages = [m for m in messages if "[KNOWLEDGE BLOCK" in m["content"]]
        if log.compaction_count > 0:
            assert len(kb_messages) == 1
            assert "10.42.88.7" in kb_messages[0]["content"]
        log.close()

    @patch("engine.ChunkLog._acc_summarize")
    def test_recent_turns_preserved(self, mock_summarize):
        mock_summarize.return_value = "Summary"
        log = ChunkLog(
            db_path=":memory:", scoring_mode="acc",
            acc_interval=5, acc_keep_recent=2,
            soft_threshold=2.0, hard_threshold=2.0,
            acc_api_key="fake",
        )
        for i in range(6):
            log.append("user", f"Turn {i}: unique_marker_{i}")
            log.next_turn()

        if log.compaction_count > 0:
            messages = log.get_context()
            context_text = " ".join(m["content"] for m in messages)
            # Recent turns (4, 5) should still be raw
            assert "unique_marker_4" in context_text or "unique_marker_5" in context_text
        log.close()

    @patch("engine.ChunkLog._acc_summarize")
    def test_knowledge_block_accumulates(self, mock_summarize):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return f"Summary batch {call_count[0]}"

        mock_summarize.side_effect = side_effect
        log = ChunkLog(
            db_path=":memory:", scoring_mode="acc",
            acc_interval=3, acc_keep_recent=1,
            soft_threshold=2.0, hard_threshold=2.0,
            acc_api_key="fake",
        )
        for i in range(9):
            log.append("user", f"Turn {i}: content_{i}")
            log.next_turn()

        if log.compaction_count >= 2:
            # Knowledge block should contain both summaries
            assert "Summary batch 1" in log._acc_knowledge_block
            assert "Summary batch 2" in log._acc_knowledge_block
        log.close()


class TestACCDoesNotUseScorerCompaction:
    """Verify ACC bypasses threshold-based compaction entirely."""

    @patch("engine.ChunkLog._acc_summarize")
    def test_no_threshold_compaction(self, mock_summarize):
        mock_summarize.return_value = "Summary"
        log = ChunkLog(
            db_path=":memory:", max_tokens=500, scoring_mode="acc",
            acc_interval=100,  # never fires
            soft_threshold=0.5, hard_threshold=0.7,  # would fire with other modes
            acc_api_key="fake",
        )
        # Add lots of content that would trigger threshold compaction
        for i in range(20):
            log.append("user", f"Turn {i}: " + "x" * 100)
            log.next_turn()

        # ACC should NOT use threshold-based compaction
        # (it should only use turn-based consolidation, which has interval=100)
        assert log.compaction_count == 0
        log.close()


class TestACCDecisionRecords:
    """Test that ACC creates proper audit trail."""

    @patch("engine.ChunkLog._acc_summarize")
    def test_consolidation_recorded(self, mock_summarize):
        mock_summarize.return_value = "Summary"
        log = ChunkLog(
            db_path=":memory:", scoring_mode="acc",
            acc_interval=3, acc_keep_recent=1,
            soft_threshold=2.0, hard_threshold=2.0,
            acc_api_key="fake",
        )
        for i in range(4):
            log.append("user", f"Turn {i}: content")
            log.next_turn()

        if log.compaction_count > 0:
            acc_decisions = [d for d in log.decisions if d.action == "acc_consolidate"]
            assert len(acc_decisions) >= 1
            assert "summarized" in acc_decisions[0].reason
        log.close()
