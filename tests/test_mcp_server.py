"""Tests for the Context Engine MCP server."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("fastmcp")

from engine import ChunkLog
from mcp_server import (
    do_compact_now,
    do_get_context,
    do_get_decisions,
    do_set_goal,
    do_store_chunk,
    reset_log,
)


@pytest.fixture(autouse=True)
def fresh_log():
    """Give each test a fresh in-memory ChunkLog."""
    log = ChunkLog(
        db_path=":memory:",
        max_tokens=10_000,
        soft_threshold=0.7,
        hard_threshold=0.9,
        scoring_mode="tfidf",
    )
    reset_log(log)
    yield log
    reset_log(None)


# --- store_chunk ---


def test_store_chunk_basic():
    result = do_store_chunk(role="user", content="Hello world")
    assert "chunk_hash" in result
    assert result["tokens"] > 0
    assert result["turn"] == 0


def test_store_chunk_invalid_role():
    result = do_store_chunk(role="invalid", content="test")
    assert "error" in result


def test_store_chunk_empty_content():
    result = do_store_chunk(role="user", content="")
    assert "error" in result


def test_store_chunk_priority_clamping():
    result = do_store_chunk(role="user", content="hi", priority=5.0)
    assert "chunk_hash" in result  # clamped to 2.0, no error


def test_store_chunk_dedup():
    r1 = do_store_chunk(role="user", content="same message")
    r2 = do_store_chunk(role="user", content="same message")
    assert r1["chunk_hash"] == r2["chunk_hash"]


# --- get_context ---


def test_get_context_empty():
    result = do_get_context()
    assert result["messages"] == []
    assert result["total_chunks"] == 0


def test_get_context_ordered():
    do_store_chunk(role="user", content="first")
    do_store_chunk(role="assistant", content="second")
    result = do_get_context()
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"


def test_get_context_max_chunks():
    for i in range(5):
        do_store_chunk(role="user", content=f"message {i}")
    result = do_get_context(max_chunks=2)
    assert len(result["messages"]) == 2
    assert result["total_chunks"] == 5


# --- compact_now ---


def test_compact_now_with_data():
    tiny_log = ChunkLog(
        db_path=":memory:",
        max_tokens=200,
        soft_threshold=0.7,
        hard_threshold=0.9,
        scoring_mode="tfidf",
    )
    reset_log(tiny_log)

    for i in range(20):
        do_store_chunk(role="user", content=f"Filler message number {i} with extra words to consume tokens")
        tiny_log.next_turn()

    result = do_compact_now()
    assert result["tokens_before"] >= result["tokens_after"]


def test_compact_now_empty():
    result = do_compact_now()
    assert result["tokens_before"] == 0
    assert result["tokens_freed"] == 0


# --- get_decisions ---


def test_get_decisions_after_append():
    do_store_chunk(role="user", content="tracked message")
    result = do_get_decisions()
    assert result["total_decisions"] >= 1
    actions = [d["action"] for d in result["decisions"]]
    assert "append" in actions


def test_get_decisions_filter():
    do_store_chunk(role="user", content="msg1")
    do_store_chunk(role="assistant", content="msg2")
    result = do_get_decisions(action_filter="append")
    for d in result["decisions"]:
        assert d["action"] == "append"


def test_get_decisions_limit():
    for i in range(10):
        do_store_chunk(role="user", content=f"msg {i}")
    result = do_get_decisions(limit=3)
    assert len(result["decisions"]) == 3


# --- set_goal ---


def test_set_goal_basic():
    result = do_set_goal(goal="Fix the authentication bug in login.py")
    assert result["goal"] == "Fix the authentication bug in login.py"
    assert result["keywords_extracted"] > 0


def test_set_goal_empty():
    result = do_set_goal(goal="")
    assert "error" in result


def test_set_goal_updates_scoring():
    do_store_chunk(role="user", content="some context about databases")
    result = do_set_goal(goal="Find the SQL injection vulnerability")
    assert result["scoring_mode"] == "tfidf"
