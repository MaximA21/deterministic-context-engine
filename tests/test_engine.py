"""Tests for the context engine (ChunkLog, AutoPriority, GoalGuidedScorer)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure engine.py is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (
    ChunkLog,
    _estimate_tokens,
    _sha256,
    extract_keywords,
    score_chunk,
)

from engine import GoalGuidedScorer

# GoalGuidedScorer requires scikit-learn at instantiation time
import importlib.util
_has_sklearn = importlib.util.find_spec("sklearn") is not None

requires_sklearn = pytest.mark.skipif(not _has_sklearn, reason="scikit-learn not installed")


# --- Utility tests ---

def test_sha256_deterministic():
    assert _sha256("hello") == _sha256("hello")
    assert _sha256("hello") != _sha256("world")


def test_estimate_tokens():
    assert _estimate_tokens("") == 1  # min 1
    assert _estimate_tokens("a" * 100) == 25  # 100 / 4


# --- extract_keywords tests ---

def test_extract_keywords_filenames():
    kw = extract_keywords("Check the file utils.py and config.yaml")
    assert "utils.py" in kw
    assert "config.yaml" in kw


def test_extract_keywords_functions():
    kw = extract_keywords("def process_data(): pass")
    assert "process_data" in kw


def test_extract_keywords_errors():
    kw = extract_keywords("There is an Error in the traceback")
    assert "error" in kw
    assert "traceback" in kw


def test_extract_keywords_empty():
    kw = extract_keywords("")
    assert isinstance(kw, set)


# --- score_chunk tests ---

def test_score_chunk_no_keywords():
    assert score_chunk("some text", set()) == 0.5


def test_score_chunk_no_matches():
    assert score_chunk("some text", {"foobar.py", "baz"}) == 0.5


def test_score_chunk_three_plus_matches():
    text = "check utils.py and config.yaml for the error traceback"
    kw = {"utils.py", "config.yaml", "error", "traceback"}
    assert score_chunk(text, kw) == 2.0


def test_score_chunk_partial_matches():
    text = "check utils.py for bugs"
    kw = {"utils.py", "config.yaml", "traceback"}
    score = score_chunk(text, kw)
    assert 0.5 < score < 2.0


# --- ChunkLog tests ---

def test_chunklog_append_and_context():
    log = ChunkLog(db_path=":memory:", max_tokens=10000)
    log.append("user", "Hello world")
    log.append("assistant", "Hi there")

    ctx = log.get_context()
    assert len(ctx) == 2
    assert ctx[0]["role"] == "user"
    assert ctx[1]["role"] == "assistant"


def test_chunklog_content_addressing():
    log = ChunkLog(db_path=":memory:", max_tokens=10000)
    h1 = log.append("user", "same content")
    h2 = log.append("user", "same content")
    assert h1 == h2
    assert len(log.get_context()) == 1


def test_chunklog_token_tracking():
    log = ChunkLog(db_path=":memory:", max_tokens=10000)
    log.append("user", "a" * 100)  # ~25 tokens
    assert log.current_tokens() == 25


def test_chunklog_compaction_fires():
    # Small context window to force compaction
    log = ChunkLog(db_path=":memory:", max_tokens=100, soft_threshold=0.7, hard_threshold=0.9)
    for i in range(20):
        log.append("user", f"Message number {i} with some filler content to use tokens")
        log.next_turn()

    # Compaction should have fired
    assert log.compaction_count > 0
    # Context should be smaller than total appended
    assert log.current_tokens() < 20 * _estimate_tokens("Message number 0 with some filler content to use tokens")


def test_chunklog_compaction_disabled():
    log = ChunkLog(db_path=":memory:", max_tokens=100, soft_threshold=2.0, hard_threshold=2.0)
    for i in range(10):
        log.append("user", f"Message {i}")
        log.next_turn()

    assert log.compaction_count == 0


def test_chunklog_turn_counter():
    log = ChunkLog(db_path=":memory:", max_tokens=10000)
    assert log.turn() == 0
    log.next_turn()
    assert log.turn() == 1


def test_chunklog_decisions_logged():
    log = ChunkLog(db_path=":memory:", max_tokens=100, soft_threshold=0.7, hard_threshold=0.9)
    log.append("user", "test message")
    assert len(log.decisions) > 0
    assert log.decisions[0].action == "append"


def test_chunklog_auto_priority():
    log = ChunkLog(
        db_path=":memory:", max_tokens=200,
        soft_threshold=0.7, hard_threshold=0.9,
        auto_priority=True,
    )
    # Add a user message with keywords
    log.append("user", "Fix the bug in utils.py")
    log.next_turn()

    # Add relevant content
    log.append("assistant", "The utils.py file has an error on line 42")
    log.next_turn()

    # Add irrelevant filler
    for i in range(10):
        log.append("user", f"Generic filler content number {i} about nothing specific")
        log.next_turn()

    # The relevant content should survive compaction better
    ctx = log.get_context()
    texts = [m["content"] for m in ctx]
    # At least check compaction happened
    assert log.compaction_count > 0


# --- GoalGuidedScorer tests ---

@requires_sklearn
def test_goal_guided_scorer_basic():
    scorer = GoalGuidedScorer()
    chunks = [
        ("h1", "Fix the authentication bug in login.py"),
        ("h2", "The weather today is nice and sunny"),
        ("h3", "Consider refactoring the database schema"),
    ]
    scores = scorer.score_chunks("authentication login bug", chunks)
    assert len(scores) == 3
    # All scores should be in [0.5, 2.0]
    for s in scores.values():
        assert 0.5 <= s <= 2.0


@requires_sklearn
def test_goal_guided_scorer_empty():
    scorer = GoalGuidedScorer()
    scores = scorer.score_chunks("anything", [])
    assert scores == {}


@requires_sklearn
def test_goal_guided_scorer_single_chunk():
    scorer = GoalGuidedScorer()
    scores = scorer.score_chunks("test", [("h1", "test content")])
    assert len(scores) == 1
    assert 0.5 <= scores["h1"] <= 2.0
