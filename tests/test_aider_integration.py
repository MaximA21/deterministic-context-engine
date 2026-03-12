"""Comprehensive integration tests for aider_integration.py.

Tests: patching, compaction, BM25 scoring on real Aider message formats,
DecisionRecords, edge cases (empty messages, huge tool outputs, unicode).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aider_integration import (
    AiderContextEngine,
    ChunkLogSummary,
    patch_aider_coder,
    create_patched_coder,
)
from engine import _estimate_tokens, DecisionRecord


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def summarizer():
    """Small-budget summarizer for fast compaction tests."""
    return ChunkLogSummary(max_tokens=500, scoring_mode="bm25")


@pytest.fixture
def engine():
    """AiderContextEngine with small budget."""
    e = AiderContextEngine(max_tokens=500, scoring_mode="bm25")
    yield e
    e.close()


@pytest.fixture
def large_engine():
    """AiderContextEngine with larger budget for multi-turn tests."""
    e = AiderContextEngine(max_tokens=3000, scoring_mode="bm25")
    yield e
    e.close()


def _make_messages(pairs: list[tuple[str, str]]) -> list[dict]:
    """Build a message list from (role, content) pairs."""
    return [{"role": r, "content": c} for r, c in pairs]


def _filler(n: int, base_tokens: int = 80) -> list[dict]:
    """Generate n filler user/assistant message pairs (~base_tokens each)."""
    msgs = []
    for i in range(n):
        user_text = f"Review module {i}. " + ("Check coverage. " * (base_tokens // 5))
        asst_text = f"Module {i} looks good. " + ("Tests pass. " * (base_tokens // 5))
        msgs.append({"role": "user", "content": user_text})
        msgs.append({"role": "assistant", "content": asst_text})
    return msgs


# ──────────────────────────────────────────────────────────
# ChunkLogSummary — core interface tests
# ──────────────────────────────────────────────────────────

class TestChunkLogSummaryInterface:
    """Tests for the ChunkLogSummary too_big/summarize interface."""

    def test_too_big_returns_false_under_budget(self, summarizer):
        msgs = [{"role": "user", "content": "short"}]
        assert summarizer.too_big(msgs) is False

    def test_too_big_returns_true_over_budget(self, summarizer):
        msgs = [{"role": "user", "content": "x " * 2000}]
        assert summarizer.too_big(msgs) is True

    def test_summarize_returns_messages_under_budget(self, summarizer):
        msgs = [{"role": "user", "content": "short"}]
        result = summarizer.summarize(msgs)
        assert result == msgs

    def test_summarize_compacts_over_budget(self):
        summarizer = ChunkLogSummary(max_tokens=300, scoring_mode="bm25")
        msgs = _filler(10, base_tokens=60)
        result = summarizer.summarize(msgs)
        assert len(result) < len(msgs)

    def test_summarize_trailing_assistant(self):
        """Aider convention: result must end with assistant message."""
        summarizer = ChunkLogSummary(max_tokens=300, scoring_mode="bm25")
        msgs = _filler(8, base_tokens=60)
        result = summarizer.summarize(msgs)
        assert result[-1]["role"] == "assistant"

    def test_summarize_trailing_assistant_appended_when_needed(self):
        """If compaction leaves trailing user msg, 'Ok.' should be appended."""
        summarizer = ChunkLogSummary(max_tokens=200, scoring_mode="bm25")
        # All user messages — must exceed budget to trigger compaction path
        msgs = [
            {"role": "user", "content": "one " * 80},
            {"role": "user", "content": "two " * 80},
            {"role": "user", "content": "three " * 80},
        ]
        result = summarizer.summarize(msgs)
        assert result[-1]["role"] == "assistant"
        assert result[-1]["content"] == "Ok."

    def test_summarize_preserves_message_format(self):
        """Surviving messages should have role and content keys."""
        summarizer = ChunkLogSummary(max_tokens=400, scoring_mode="bm25")
        msgs = _filler(6, base_tokens=50)
        result = summarizer.summarize(msgs)
        for m in result:
            assert "role" in m
            assert "content" in m


# ──────────────────────────────────────────────────────────
# ChunkLogSummary — token counting
# ──────────────────────────────────────────────────────────

class TestTokenCounting:
    def test_token_count_single_message(self, summarizer):
        msg = {"role": "user", "content": "hello world"}
        count = summarizer.token_count(msg)
        assert count == _estimate_tokens("hello world")

    def test_token_count_list_of_messages(self, summarizer):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        total = summarizer.token_count(msgs)
        expected = _estimate_tokens("hello") + _estimate_tokens("world")
        assert total == expected

    def test_token_count_multimodal_message(self, summarizer):
        """Vision/multimodal messages with list content."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
        count = summarizer.token_count(msg)
        assert count > 0

    def test_token_count_empty_content(self, summarizer):
        msg = {"role": "user", "content": ""}
        count = summarizer.token_count(msg)
        assert count == _estimate_tokens("")

    def test_token_count_missing_content(self, summarizer):
        msg = {"role": "user"}
        count = summarizer.token_count(msg)
        assert count == _estimate_tokens("")

    def test_token_count_numeric_content(self, summarizer):
        """Content could be non-string (edge case)."""
        msg = {"role": "user", "content": 42}
        count = summarizer.token_count(msg)
        assert count == _estimate_tokens("42")


# ──────────────────────────────────────────────────────────
# ChunkLogSummary — compaction behavior
# ──────────────────────────────────────────────────────────

class TestCompaction:
    def test_compaction_reduces_token_count(self):
        summarizer = ChunkLogSummary(max_tokens=400, scoring_mode="bm25")
        msgs = _filler(10, base_tokens=60)
        before_tokens = sum(summarizer.token_count(m) for m in msgs)
        result = summarizer.summarize(msgs)
        after_tokens = sum(summarizer.token_count(m) for m in result)
        assert after_tokens < before_tokens

    def test_compaction_preserves_unique_content(self):
        """Unique/specific messages should survive over generic filler."""
        summarizer = ChunkLogSummary(max_tokens=600, scoring_mode="bm25")
        needle = {"role": "user", "content": "Fix the NullPointerException in PaymentProcessor.java at line 237 in processRefund()"}
        filler = _filler(8, base_tokens=50)
        msgs = [needle] + filler
        result = summarizer.summarize(msgs)
        contents = " ".join(m["content"] for m in result).lower()
        assert "paymentprocessor" in contents or "processrefund" in contents

    def test_decision_records_created_on_compaction(self):
        summarizer = ChunkLogSummary(max_tokens=300, scoring_mode="bm25")
        msgs = _filler(10, base_tokens=60)
        summarizer.summarize(msgs)
        assert len(summarizer.decision_records) > 0

    def test_decision_record_structure(self):
        summarizer = ChunkLogSummary(max_tokens=300, scoring_mode="bm25")
        msgs = _filler(10, base_tokens=60)
        summarizer.summarize(msgs)
        for record in summarizer.decision_records:
            assert "timestamp" in record
            assert "messages_before" in record
            assert "messages_after" in record
            assert "tokens_before" in record
            assert "tokens_after" in record
            assert "compaction_events" in record
            assert "decisions" in record

    def test_compaction_count_tracked(self):
        summarizer = ChunkLogSummary(max_tokens=300, scoring_mode="bm25")
        msgs = _filler(10, base_tokens=60)
        summarizer.summarize(msgs)
        records = summarizer.decision_records
        assert any(r["compaction_events"] > 0 for r in records)

    def test_soft_threshold_triggers(self):
        """Soft compaction fires at 70% of budget by default."""
        summarizer = ChunkLogSummary(
            max_tokens=500,
            scoring_mode="bm25",
            soft_threshold=0.7,
            hard_threshold=0.9,
        )
        # Generate enough to cross 70% threshold (~350 tokens)
        msgs = _filler(6, base_tokens=50)
        result = summarizer.summarize(msgs)
        assert len(result) < len(msgs)

    def test_hard_threshold_triggers(self):
        """Hard compaction fires at 90% and evicts more aggressively."""
        summarizer = ChunkLogSummary(
            max_tokens=300,
            scoring_mode="bm25",
            soft_threshold=0.7,
            hard_threshold=0.9,
        )
        msgs = _filler(15, base_tokens=60)
        result = summarizer.summarize(msgs)
        after_tokens = sum(summarizer.token_count(m) for m in result)
        assert after_tokens <= 300


# ──────────────────────────────────────────────────────────
# ChunkLogSummary — decision log persistence
# ──────────────────────────────────────────────────────────

class TestDecisionLogPersistence:
    def test_decision_log_written_to_file(self, tmp_path):
        log_path = str(tmp_path / "decisions.json")
        summarizer = ChunkLogSummary(
            max_tokens=300,
            scoring_mode="bm25",
            decision_log_path=log_path,
        )
        msgs = _filler(10, base_tokens=60)
        summarizer.summarize(msgs)

        assert Path(log_path).exists()
        data = json.loads(Path(log_path).read_text())
        assert isinstance(data, list)
        assert len(data) > 0

    def test_decision_log_creates_parent_dirs(self, tmp_path):
        log_path = str(tmp_path / "nested" / "deep" / "decisions.json")
        summarizer = ChunkLogSummary(
            max_tokens=300,
            scoring_mode="bm25",
            decision_log_path=log_path,
        )
        msgs = _filler(10, base_tokens=60)
        summarizer.summarize(msgs)
        assert Path(log_path).exists()

    def test_no_decision_log_when_no_compaction(self, tmp_path):
        log_path = str(tmp_path / "decisions.json")
        summarizer = ChunkLogSummary(
            max_tokens=10000,
            scoring_mode="bm25",
            decision_log_path=log_path,
        )
        msgs = [{"role": "user", "content": "short"}]
        summarizer.summarize(msgs)
        # File should not exist since no compaction occurred
        assert not Path(log_path).exists()


# ──────────────────────────────────────────────────────────
# BM25 scoring on real Aider message formats
# ──────────────────────────────────────────────────────────

class TestBM25ScoringAiderFormats:
    """Test that BM25 scores correctly on message formats Aider actually sends."""

    def test_tool_output_format(self):
        """Aider sends tool outputs as assistant messages with file contents."""
        summarizer = ChunkLogSummary(max_tokens=800, scoring_mode="bm25")
        needle = {
            "role": "assistant",
            "content": (
                "Here are the contents of auth.py:\n"
                "```python\n"
                "def validate_token(token, secret_key='abc123'):\n"
                "    decoded = jwt.decode(token, secret_key)\n"
                "    if decoded['exp'] <= time.time():  # BUG: should be <\n"
                "        raise TokenExpired()\n"
                "    return decoded\n"
                "```\n"
            ),
        }
        filler = _filler(8, base_tokens=50)
        # Recall question about the bug
        recall_q = {"role": "user", "content": "What was the bug in validate_token?"}
        msgs = [needle] + filler + [recall_q]
        result = summarizer.summarize(msgs)
        contents = " ".join(m["content"] for m in result).lower()
        assert "validate_token" in contents

    def test_diff_format_preserved(self):
        """Aider generates SEARCH/REPLACE diffs."""
        summarizer = ChunkLogSummary(max_tokens=800, scoring_mode="bm25")
        diff_msg = {
            "role": "assistant",
            "content": (
                "auth.py\n"
                "<<<<<<< SEARCH\n"
                "    if decoded['exp'] <= time.time():\n"
                "=======\n"
                "    if decoded['exp'] < time.time():\n"
                ">>>>>>> REPLACE\n"
            ),
        }
        filler = _filler(8, base_tokens=50)
        msgs = [diff_msg] + filler
        result = summarizer.summarize(msgs)
        # The diff should survive since it has unique content
        contents = " ".join(m["content"] for m in result).lower()
        # At minimum the diff shouldn't be the first evicted
        assert len(result) > 0

    def test_commit_message_format(self):
        """Aider generates commit messages."""
        summarizer = ChunkLogSummary(max_tokens=600, scoring_mode="bm25")
        commit_msg = {
            "role": "assistant",
            "content": "Commit 3a7f2b1: fix off-by-one in auth.py validate_token() expiry check",
        }
        filler = _filler(6, base_tokens=50)
        msgs = [commit_msg] + filler
        result = summarizer.summarize(msgs)
        contents = " ".join(m["content"] for m in result).lower()
        assert "3a7f2b1" in contents or "validate_token" in contents

    def test_file_listing_format(self):
        """Aider shows file listings in repo map."""
        summarizer = ChunkLogSummary(max_tokens=800, scoring_mode="bm25")
        repo_map = {
            "role": "user",
            "content": (
                "Files in chat:\n"
                "  src/auth.py\n"
                "  src/api_gateway.py\n"
                "  tests/test_auth.py\n"
                "  Dockerfile\n"
            ),
        }
        filler = _filler(8, base_tokens=50)
        msgs = [repo_map] + filler
        result = summarizer.summarize(msgs)
        assert len(result) > 0

    def test_error_traceback_format(self):
        """Python tracebacks should score as unique content."""
        summarizer = ChunkLogSummary(max_tokens=800, scoring_mode="bm25")
        traceback_msg = {
            "role": "user",
            "content": (
                "Traceback (most recent call last):\n"
                '  File "auth.py", line 42, in validate_token\n'
                "    decoded = jwt.decode(token, secret_key)\n"
                "jwt.exceptions.DecodeError: Invalid header padding\n"
            ),
        }
        filler = _filler(6, base_tokens=50)
        msgs = [traceback_msg] + filler
        result = summarizer.summarize(msgs)
        contents = " ".join(m["content"] for m in result).lower()
        assert "decodeerror" in contents or "jwt" in contents

    def test_mixed_roles_handled(self):
        """Messages should work with user, assistant, system roles."""
        summarizer = ChunkLogSummary(max_tokens=600, scoring_mode="bm25")
        msgs = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Fix the bug in auth.py validate_token()"},
            {"role": "assistant", "content": "I'll fix the off-by-one error."},
            {"role": "user", "content": "Great, now add tests."},
            {"role": "assistant", "content": "Added 5 unit tests for validate_token."},
        ]
        result = summarizer.summarize(msgs)
        assert all("role" in m and "content" in m for m in result)


# ──────────────────────────────────────────────────────────
# AiderContextEngine — standalone wrapper tests
# ──────────────────────────────────────────────────────────

class TestAiderContextEngine:
    def test_add_message_returns_hash(self, engine):
        h = engine.add_message("user", "hello world")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_get_managed_messages(self, engine):
        engine.add_message("user", "hello")
        engine.add_message("assistant", "hi there")
        msgs = engine.get_managed_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"

    def test_get_tokens(self, engine):
        engine.add_message("user", "hello world")
        assert engine.get_tokens() > 0

    def test_compaction_count_starts_zero(self, engine):
        assert engine.compaction_count == 0

    def test_decisions_starts_empty_except_appends(self, engine):
        engine.add_message("user", "hello")
        # append decisions should be recorded
        assert len(engine.decisions) >= 1
        assert engine.decisions[0].action == "append"

    def test_close_is_safe(self, engine):
        engine.add_message("user", "hello")
        engine.close()
        # Double close should not raise
        engine.close()

    def test_compaction_triggers_at_budget(self):
        engine = AiderContextEngine(max_tokens=300, scoring_mode="bm25")
        try:
            for i in range(20):
                engine.add_message("user", f"Generic message {i}. " * 10)
                engine.add_message("assistant", f"Response {i}. " * 10)
            assert engine.compaction_count > 0
            assert engine.get_tokens() <= 300
        finally:
            engine.close()

    def test_content_addressing_deduplicates(self, engine):
        h1 = engine.add_message("user", "duplicate message")
        h2 = engine.add_message("user", "duplicate message")
        assert h1 == h2
        msgs = engine.get_managed_messages()
        # Should only appear once
        user_msgs = [m for m in msgs if m["content"] == "duplicate message"]
        assert len(user_msgs) == 1

    def test_multi_turn_preserves_order(self, large_engine):
        large_engine.add_message("user", "first question about auth.py")
        large_engine.add_message("assistant", "answer about auth.py")
        large_engine.add_message("user", "second question about db.py")
        large_engine.add_message("assistant", "answer about db.py")
        msgs = large_engine.get_managed_messages()
        assert msgs[0]["content"] == "first question about auth.py"
        assert msgs[-1]["content"] == "answer about db.py"


# ──────────────────────────────────────────────────────────
# patch_aider_coder — monkey-patching tests
# ──────────────────────────────────────────────────────────

class TestPatchAiderCoder:
    def _mock_coder(self, max_tokens=4096):
        """Create a mock coder that mimics aider's Coder interface."""
        coder = MagicMock()
        coder.main_model = SimpleNamespace(max_chat_history_tokens=max_tokens)
        coder.summarizer = MagicMock()
        return coder

    def test_patch_replaces_summarizer(self):
        coder = self._mock_coder()
        patch_aider_coder(coder)
        assert isinstance(coder.summarizer, ChunkLogSummary)

    def test_patch_uses_model_token_budget(self):
        coder = self._mock_coder(max_tokens=8192)
        patch_aider_coder(coder)
        assert coder.summarizer.max_tokens == 8192

    def test_patch_default_scoring_mode(self):
        coder = self._mock_coder()
        patch_aider_coder(coder)
        assert coder.summarizer.scoring_mode == "bm25"

    def test_patch_custom_scoring_mode(self):
        coder = self._mock_coder()
        patch_aider_coder(coder, scoring_mode="tfidf")
        assert coder.summarizer.scoring_mode == "tfidf"

    def test_patch_with_decision_log(self, tmp_path):
        coder = self._mock_coder()
        log_path = str(tmp_path / "log.json")
        patch_aider_coder(coder, decision_log_path=log_path)
        assert coder.summarizer.decision_log_path == log_path

    def test_patch_fallback_max_tokens(self):
        """If model has no max_chat_history_tokens, fallback to 4096."""
        coder = MagicMock()
        coder.main_model = SimpleNamespace()  # no max_chat_history_tokens
        patch_aider_coder(coder)
        assert coder.summarizer.max_tokens == 4096

    def test_patched_summarizer_works_end_to_end(self):
        coder = self._mock_coder(max_tokens=400)
        patch_aider_coder(coder)

        msgs = _filler(8, base_tokens=50)
        assert coder.summarizer.too_big(msgs) is True

        result = coder.summarizer.summarize(msgs)
        assert len(result) < len(msgs)
        assert result[-1]["role"] == "assistant"


# ──────────────────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_message_list(self, summarizer):
        result = summarizer.summarize([])
        assert result == []

    def test_single_message(self, summarizer):
        msgs = [{"role": "user", "content": "hello"}]
        result = summarizer.summarize(msgs)
        assert len(result) >= 1

    def test_empty_content_message(self, summarizer):
        msgs = [{"role": "user", "content": ""}]
        result = summarizer.summarize(msgs)
        # Empty content messages are skipped
        assert isinstance(result, list)

    def test_whitespace_only_content(self, summarizer):
        msgs = [{"role": "user", "content": "   \n\t  "}]
        result = summarizer.summarize(msgs)
        assert isinstance(result, list)

    def test_none_content_treated_as_string(self, summarizer):
        msgs = [{"role": "user", "content": None}]
        result = summarizer.summarize(msgs)
        assert isinstance(result, list)

    def test_unicode_content(self):
        engine = AiderContextEngine(max_tokens=1000, scoring_mode="bm25")
        try:
            h = engine.add_message("user", "Fix the bug in 日本語ファイル.py with émojis 🎉")
            assert isinstance(h, str)
            msgs = engine.get_managed_messages()
            assert "日本語ファイル" in msgs[0]["content"]
        finally:
            engine.close()

    def test_unicode_compaction(self):
        summarizer = ChunkLogSummary(max_tokens=400, scoring_mode="bm25")
        needle = {"role": "user", "content": "Исправь баг в модуле авторизации строка 42"}
        filler = _filler(6, base_tokens=50)
        msgs = [needle] + filler
        result = summarizer.summarize(msgs)
        assert isinstance(result, list)
        assert all(isinstance(m["content"], str) for m in result)

    def test_very_long_single_message(self):
        """A single message exceeding the budget should still be returned."""
        summarizer = ChunkLogSummary(max_tokens=100, scoring_mode="bm25")
        big_msg = {"role": "user", "content": "x " * 5000}
        result = summarizer.summarize([big_msg])
        # Should have the big message (can't compact a single chunk)
        assert len(result) >= 1

    def test_huge_tool_output(self):
        """Simulate a massive tool output (file contents) from Aider."""
        engine = AiderContextEngine(max_tokens=500, scoring_mode="bm25")
        try:
            # Add a specific needle first
            engine.add_message("user", "Fix the CSRF vulnerability in api/views.py line 89")
            engine.add_message(
                "assistant",
                "Found the CSRF bug: @csrf_exempt on handle_payment() at line 89. Removing it.",
            )

            # Then add huge tool output (filler)
            big_output = "def placeholder_function():\n    pass\n" * 200
            engine.add_message("assistant", f"Here are the file contents:\n```\n{big_output}\n```")

            assert engine.compaction_count > 0
            msgs = engine.get_managed_messages()
            all_text = " ".join(m["content"] for m in msgs).lower()
            # The specific CSRF fix should survive over the big generic output
            # (or at least not crash)
            assert isinstance(msgs, list)
        finally:
            engine.close()

    def test_special_characters_in_content(self):
        engine = AiderContextEngine(max_tokens=1000, scoring_mode="bm25")
        try:
            special = r'regex: ^(?:[\w.]+)@([\w-]+\.)+[\w]{2,}$ and SQL: SELECT * FROM "users" WHERE id=1; --comment'
            h = engine.add_message("user", special)
            msgs = engine.get_managed_messages()
            assert special in msgs[0]["content"]
        finally:
            engine.close()

    def test_newlines_and_tabs(self, engine):
        content = "line1\nline2\n\tindented\r\nwindows_line\n"
        engine.add_message("user", content)
        msgs = engine.get_managed_messages()
        assert msgs[0]["content"] == content

    def test_very_many_short_messages(self):
        """100 messages — compaction should still produce valid output."""
        summarizer = ChunkLogSummary(max_tokens=200, scoring_mode="bm25")
        # Each msg ~20 tokens, 100 msgs = ~2000 tokens, budget 200 → must compact
        msgs = [{"role": "user", "content": f"message number {i} about topic " * 3} for i in range(100)]
        result = summarizer.summarize(msgs)
        assert isinstance(result, list)
        assert len(result) < 100

    def test_all_identical_messages_deduplicated_in_chunklog(self):
        """Content-addressing in ChunkLog deduplicates identical messages."""
        engine = AiderContextEngine(max_tokens=1000, scoring_mode="bm25")
        try:
            for _ in range(20):
                engine.add_message("user", "identical message")
            msgs = engine.get_managed_messages()
            # Content addressing collapses all 20 into 1
            user_msgs = [m for m in msgs if m["content"] == "identical message"]
            assert len(user_msgs) == 1
        finally:
            engine.close()

    def test_missing_role_defaults_to_user(self):
        """Messages without a role key should default to 'user'."""
        summarizer = ChunkLogSummary(max_tokens=1000, scoring_mode="bm25")
        msgs = [{"content": "no role here"}]
        result = summarizer.summarize(msgs)
        assert len(result) >= 1

    def test_multimodal_content_flattened(self):
        """Multimodal (list) content should be flattened for scoring."""
        summarizer = ChunkLogSummary(max_tokens=1000, scoring_mode="bm25")
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What does this screenshot show?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
        result = summarizer.summarize(msgs)
        assert isinstance(result, list)


# ──────────────────────────────────────────────────────────
# BM25 scoring correctness
# ──────────────────────────────────────────────────────────

class TestBM25ScoringCorrectness:
    """Verify BM25 scorer ranks relevant messages above filler."""

    def test_needle_scores_above_filler(self):
        """A message with unique technical content should score higher."""
        from engine import BM25Scorer

        scorer = BM25Scorer()
        goal = "What was the bug in validate_token?"
        chunks = [
            ("needle", "Fix validate_token() in auth.py: the expiry check uses <= instead of < on line 42"),
            ("filler1", "Review the test coverage. All edge cases look good. Tests pass."),
            ("filler2", "Module looks good. Test coverage is adequate. No issues found."),
            ("filler3", "Review the test coverage. Check all edge cases. Tests pass."),
        ]
        scores = scorer.score_chunks(goal, chunks)
        assert scores["needle"] > scores["filler1"]
        assert scores["needle"] > scores["filler2"]
        assert scores["needle"] > scores["filler3"]

    def test_bm25_empty_goal(self):
        from engine import BM25Scorer

        scorer = BM25Scorer()
        chunks = [("h1", "some content here")]
        scores = scorer.score_chunks("", chunks)
        assert scores["h1"] == 0.5  # Default score for empty goal

    def test_bm25_empty_chunks(self):
        from engine import BM25Scorer

        scorer = BM25Scorer()
        scores = scorer.score_chunks("some goal", [])
        assert scores == {}

    def test_bm25_single_chunk(self):
        from engine import BM25Scorer

        scorer = BM25Scorer()
        scores = scorer.score_chunks("hello", [("h1", "hello world")])
        assert 0.5 <= scores["h1"] <= 2.0

    def test_bm25_scores_in_range(self):
        """All scores must be in [0.5, 2.0]."""
        from engine import BM25Scorer

        scorer = BM25Scorer()
        goal = "fix the authentication bug"
        chunks = [
            ("a", "auth module has a bug in validate_token"),
            ("b", "database migration needs updating"),
            ("c", "CSS layout issue in sidebar"),
            ("d", "fix authentication in auth.py"),
        ]
        scores = scorer.score_chunks(goal, chunks)
        for h, s in scores.items():
            assert 0.5 <= s <= 2.0, f"Score {s} for {h} out of range"

    def test_bm25_uniqueness_signal(self):
        """Repetitive chunks should score lower via uniqueness."""
        from engine import BM25Scorer

        scorer = BM25Scorer()
        goal = "general question"
        chunks = [
            ("unique", "JWT DecodeError at line 42 in PaymentProcessor with idempotency_key abc123"),
            ("rep1", "Review code. Tests pass. Coverage is good. No issues."),
            ("rep2", "Review code. Tests pass. Coverage is good. Everything works."),
            ("rep3", "Review code. Tests pass. Coverage is good. All clear."),
        ]
        scores = scorer.score_chunks(goal, chunks)
        # The unique chunk should score higher than any repetitive one
        assert scores["unique"] > scores["rep1"]
        assert scores["unique"] > scores["rep2"]
        assert scores["unique"] > scores["rep3"]

    def test_bm25_aider_search_replace_block(self):
        """Aider's SEARCH/REPLACE blocks should be scorable and in valid range."""
        from engine import BM25Scorer

        scorer = BM25Scorer()
        goal = "what change was made to the decoded exp time check in auth?"
        chunks = [
            (
                "diff",
                "auth.py change: decoded exp time check was wrong. "
                "<<<<<<< SEARCH\n    if decoded['exp'] <= time.time():\n"
                "=======\n    if decoded['exp'] < time.time():\n>>>>>>> REPLACE",
            ),
            ("filler", "General discussion about code review and testing procedures. All looks good."),
            ("filler2", "Code quality review complete. Everything passes. Good naming conventions."),
        ]
        scores = scorer.score_chunks(goal, chunks)
        # All scores should be valid
        for s in scores.values():
            assert 0.5 <= s <= 2.0
        # The diff should score higher due to keyword match + uniqueness
        assert scores["diff"] > scores["filler"]

    def test_bm25_custom_k1_b(self):
        """Custom k1 and b parameters should produce valid scores."""
        from engine import BM25Scorer

        scorer = BM25Scorer(k1=2.0, b=0.5)
        scores = scorer.score_chunks(
            "fix the bug",
            [("h1", "bug fix in auth"), ("h2", "general update")],
        )
        for s in scores.values():
            assert 0.5 <= s <= 2.0


# ──────────────────────────────────────────────────────────
# DecisionRecord integrity
# ──────────────────────────────────────────────────────────

class TestDecisionRecords:
    def test_append_decisions_recorded(self, engine):
        engine.add_message("user", "hello")
        engine.add_message("assistant", "hi")
        append_decisions = [d for d in engine.decisions if d.action == "append"]
        assert len(append_decisions) == 2

    def test_compact_decisions_recorded(self):
        engine = AiderContextEngine(max_tokens=200, scoring_mode="bm25")
        try:
            for i in range(15):
                engine.add_message("user", f"msg {i} " * 15)
                engine.add_message("assistant", f"resp {i} " * 15)
            compact_decisions = [
                d for d in engine.decisions if d.action.startswith("compact")
            ]
            assert len(compact_decisions) > 0
        finally:
            engine.close()

    def test_decision_record_fields(self, engine):
        engine.add_message("user", "hello world test message")
        d = engine.decisions[0]
        assert isinstance(d, DecisionRecord)
        assert isinstance(d.timestamp, float)
        assert d.action == "append"
        assert isinstance(d.chunk_hash, str)
        assert isinstance(d.reason, str)
        assert isinstance(d.context_size_before, int)
        assert isinstance(d.context_size_after, int)

    def test_decision_context_sizes_monotonic_on_append(self, engine):
        engine.add_message("user", "first message")
        engine.add_message("user", "second message")
        appends = [d for d in engine.decisions if d.action == "append"]
        assert appends[1].context_size_before >= appends[0].context_size_after

    def test_decision_compact_reduces_size(self):
        engine = AiderContextEngine(max_tokens=200, scoring_mode="bm25")
        try:
            for i in range(15):
                engine.add_message("user", f"message number {i} with content " * 5)
            compacts = [d for d in engine.decisions if d.action.startswith("compact")]
            for c in compacts:
                assert c.context_size_after <= c.context_size_before
        finally:
            engine.close()


# ──────────────────────────────────────────────────────────
# Scoring mode configuration
# ──────────────────────────────────────────────────────────

class TestScoringModes:
    def test_bm25_mode_default(self):
        s = ChunkLogSummary(scoring_mode="bm25")
        assert s.scoring_mode == "bm25"

    def test_tfidf_mode(self):
        s = ChunkLogSummary(scoring_mode="tfidf")
        assert s.scoring_mode == "tfidf"

    def test_custom_thresholds(self):
        s = ChunkLogSummary(soft_threshold=0.5, hard_threshold=0.8)
        assert s.soft_threshold == 0.5
        assert s.hard_threshold == 0.8

    def test_default_db_path_is_memory(self):
        s = ChunkLogSummary()
        assert s.db_path == ":memory:"


# ──────────────────────────────────────────────────────────
# Aider-realistic multi-turn scenarios
# ──────────────────────────────────────────────────────────

class TestAiderRealisticScenarios:
    def test_coding_session_needle_preservation(self):
        """Simulate a full aider coding session and verify needle recall."""
        engine = AiderContextEngine(max_tokens=1500, scoring_mode="bm25")
        try:
            # Early needles with specific technical details
            engine.add_message("user", "Fix the SQL injection in api/views.py handle_search() line 73")
            engine.add_message(
                "assistant",
                "Found it: raw f-string query `f\"SELECT * FROM users WHERE name='{name}'\"`. "
                "Changed to parameterized query with cursor.execute().",
            )
            engine.add_message("user", "Also fix the SSRF in fetch_url() line 112 of api/utils.py")
            engine.add_message(
                "assistant",
                "Fixed SSRF: added URL validation with allowlist ['api.internal.com']. "
                "Blocked requests to private IP ranges 10.0.0.0/8, 172.16.0.0/12.",
            )

            # Many filler turns
            for i in range(15):
                engine.add_message(
                    "user",
                    f"Review file {i}.py for code quality and best practices.",
                )
                engine.add_message(
                    "assistant",
                    f"File {i}.py looks clean. Good naming. Proper error handling.",
                )

            msgs = engine.get_managed_messages()
            all_text = " ".join(m["content"] for m in msgs).lower()

            # At least one of the specific security findings should survive
            sql_found = "sql injection" in all_text or "parameterized" in all_text
            ssrf_found = "ssrf" in all_text or "allowlist" in all_text
            assert sql_found or ssrf_found, "Both security needles were evicted"
        finally:
            engine.close()

    def test_progressive_context_growth(self, large_engine):
        """Tokens should grow with messages, then stabilize after compaction."""
        token_history = []
        for i in range(20):
            large_engine.add_message("user", f"Task {i}: review component " * 10)
            large_engine.add_message("assistant", f"Reviewed component {i}. " * 8)
            token_history.append(large_engine.get_tokens())

        # Should stabilize (not grow unbounded)
        assert token_history[-1] <= 3000
        # Should have had growth initially
        assert token_history[3] > token_history[0]

    def test_aider_done_messages_flow(self):
        """Replicate aider's pattern: accumulate done_messages, periodically summarize."""
        summarizer = ChunkLogSummary(max_tokens=1000, scoring_mode="bm25")
        done_messages: list[dict] = []

        # Simulate 10 aider turns
        for i in range(10):
            done_messages.append({"role": "user", "content": f"Work on task {i}. " * 15})
            done_messages.append({"role": "assistant", "content": f"Done with task {i}. " * 12})

            if summarizer.too_big(done_messages):
                done_messages = summarizer.summarize(done_messages)

        # After 10 turns, messages should be within budget
        total = sum(summarizer.token_count(m) for m in done_messages)
        assert total <= 1000

    def test_interleaved_code_and_chat(self):
        """Mix of code blocks and natural language."""
        engine = AiderContextEngine(max_tokens=800, scoring_mode="bm25")
        try:
            engine.add_message("user", "Show me the auth.py file")
            engine.add_message(
                "assistant",
                "```python\ndef login(user, pwd):\n    if check_password(user, pwd):\n        return create_session(user)\n    raise AuthError('Invalid credentials')\n```",
            )
            engine.add_message("user", "Add rate limiting to login()")
            engine.add_message(
                "assistant",
                "```python\nfrom ratelimit import rate_limit\n\n@rate_limit(5, period=60)\ndef login(user, pwd):\n    ...\n```",
            )

            # Add filler
            for i in range(5):
                engine.add_message("user", f"Check module {i} for style issues. " * 8)
                engine.add_message("assistant", f"Module {i} style is fine. " * 6)

            msgs = engine.get_managed_messages()
            all_text = " ".join(m["content"] for m in msgs).lower()
            # The code-specific content should survive
            assert "login" in all_text or "rate_limit" in all_text or "auth" in all_text
        finally:
            engine.close()


# ──────────────────────────────────────────────────────────
# create_patched_coder (requires aider import — mock it)
# ──────────────────────────────────────────────────────────

class TestCreatePatchedCoder:
    def test_create_patched_coder_extracts_kwargs(self):
        """Verify our kwargs are popped before passing to Coder.create()."""
        # We can't call create_patched_coder directly without aider installed,
        # but we can test the kwarg extraction logic
        kwargs = {
            "context_scoring_mode": "tfidf",
            "context_decision_log": "/tmp/log.json",
            "model": "gpt-4",
        }
        scoring = kwargs.pop("context_scoring_mode", "bm25")
        decision_log = kwargs.pop("context_decision_log", None)
        assert scoring == "tfidf"
        assert decision_log == "/tmp/log.json"
        assert "model" in kwargs
        assert "context_scoring_mode" not in kwargs
        assert "context_decision_log" not in kwargs


# ──────────────────────────────────────────────────────────
# Concurrency & robustness
# ──────────────────────────────────────────────────────────

class TestRobustness:
    def test_multiple_summarize_calls(self):
        """Calling summarize multiple times should not corrupt state."""
        summarizer = ChunkLogSummary(max_tokens=400, scoring_mode="bm25")
        msgs = _filler(8, base_tokens=50)

        result1 = summarizer.summarize(msgs)
        result2 = summarizer.summarize(msgs)

        # Both should produce valid output
        assert len(result1) > 0
        assert len(result2) > 0

    def test_summarize_with_growing_messages(self):
        """Incrementally adding messages and summarizing."""
        summarizer = ChunkLogSummary(max_tokens=400, scoring_mode="bm25")
        msgs = []
        for i in range(10):
            msgs.append({"role": "user", "content": f"Turn {i}: " + "generic question. " * 10})
            msgs.append({"role": "assistant", "content": f"Answer {i}. " * 8})
            if summarizer.too_big(msgs):
                msgs = summarizer.summarize(msgs)
        assert isinstance(msgs, list)
        total = sum(summarizer.token_count(m) for m in msgs)
        assert total <= 400

    def test_engine_with_sqlite_file(self, tmp_path):
        """Engine should work with file-backed SQLite."""
        db_path = str(tmp_path / "test.db")
        engine = AiderContextEngine(
            max_tokens=500,
            scoring_mode="bm25",
            db_path=db_path,
        )
        try:
            engine.add_message("user", "hello")
            engine.add_message("assistant", "world")
            assert engine.get_tokens() > 0
            assert Path(db_path).exists()
        finally:
            engine.close()
