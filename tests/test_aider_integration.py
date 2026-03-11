"""Tests for the aider integration — verifies ChunkLogSummary is a valid
drop-in replacement for aider's ChatSummary."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aider_integration import ChunkLogSummary, AiderContextEngine


def test_chunklog_summary_interface():
    """ChunkLogSummary implements the same interface as ChatSummary."""
    summary = ChunkLogSummary(max_tokens=500)
    assert hasattr(summary, "too_big")
    assert hasattr(summary, "summarize")
    assert callable(summary.too_big)
    assert callable(summary.summarize)


def test_too_big_small_messages():
    """Small messages should not be too big."""
    summary = ChunkLogSummary(max_tokens=5000)
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert not summary.too_big(messages)


def test_too_big_large_messages():
    """Large messages should trigger too_big."""
    summary = ChunkLogSummary(max_tokens=100)
    messages = [
        {"role": "user", "content": "x" * 1000},
        {"role": "assistant", "content": "y" * 1000},
    ]
    assert summary.too_big(messages)


def test_summarize_fits():
    """Messages that fit should be returned unchanged."""
    summary = ChunkLogSummary(max_tokens=5000)
    messages = [
        {"role": "user", "content": "Fix the bug in auth.py"},
        {"role": "assistant", "content": "Done, changed line 42."},
    ]
    result = summary.summarize(messages)
    assert len(result) >= 2
    # Content should be preserved
    contents = [m["content"] for m in result]
    assert any("auth.py" in c for c in contents)


def test_summarize_compacts():
    """Messages exceeding budget should be compacted, not summarized by LLM."""
    summary = ChunkLogSummary(max_tokens=500)
    # Create messages that exceed 500 tokens
    messages = []
    for i in range(20):
        messages.append({"role": "user", "content": f"Generic filler message number {i} " * 20})
        messages.append({"role": "assistant", "content": f"Ok, noted item {i}."})
    # Add a needle
    messages.append({"role": "user", "content": "CRITICAL: The secret codename is PHOENIX."})
    messages.append({"role": "assistant", "content": "Got it, PHOENIX."})

    result = summary.summarize(messages)

    # Should be smaller than input
    assert len(result) < len(messages)

    # Should end with assistant message (aider convention)
    assert result[-1]["role"] == "assistant"

    # Decision records should be populated
    assert len(summary.decision_records) > 0
    record = summary.decision_records[0]
    assert record["tokens_before"] > record["tokens_after"]
    assert record["compaction_events"] > 0


def test_summarize_preserves_needles():
    """BM25 scoring should preserve unique/important content during compaction."""
    summary = ChunkLogSummary(max_tokens=800, scoring_mode="bm25")
    messages = []

    # 15 generic filler messages
    for i in range(15):
        messages.append({
            "role": "user",
            "content": (
                f"Here's the status update for sprint {i}. "
                "We reviewed the backlog and updated estimates. "
                "The team discussed velocity and capacity planning. "
                "No blockers were identified in the standup meeting. " * 3
            ),
        })
        messages.append({"role": "assistant", "content": "Noted."})

    # 1 needle with unique content
    messages.append({
        "role": "user",
        "content": "CRITICAL BUG: auth.py line 42 has an off-by-one error in validate_token(). "
                   "Token expiry uses <= instead of <, allowing tokens to be valid one extra second.",
    })
    messages.append({"role": "assistant", "content": "I'll fix that."})

    result = summary.summarize(messages)

    # Needle content should survive compaction
    all_content = " ".join(m["content"] for m in result)
    assert "off-by-one" in all_content or "validate_token" in all_content


def test_multimodal_messages():
    """Should handle multimodal (vision) message format."""
    summary = ChunkLogSummary(max_tokens=5000)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
            ],
        },
        {"role": "assistant", "content": "I see a chart."},
    ]
    # Should not crash
    result = summary.summarize(messages)
    assert len(result) >= 1


def test_aider_context_engine():
    """AiderContextEngine standalone wrapper works."""
    engine = AiderContextEngine(max_tokens=500)

    # Add messages
    engine.add_message("user", "Fix the bug in auth.py line 42")
    engine.add_message("assistant", "I'll look at that.")

    messages = engine.get_managed_messages()
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert "auth.py" in messages[0]["content"]

    tokens = engine.get_tokens()
    assert tokens > 0

    engine.close()


def test_aider_context_engine_compaction():
    """AiderContextEngine compacts when messages exceed budget."""
    engine = AiderContextEngine(max_tokens=500)

    # Add lots of filler
    for i in range(20):
        engine.add_message("user", f"Filler message {i} about sprint planning " * 10)
        engine.add_message("assistant", "Ok.")

    # Add needle
    engine.add_message("user", "CRITICAL: server IP is 10.42.88.7")

    assert engine.compaction_count > 0
    assert engine.get_tokens() <= 500

    engine.close()


def test_decision_log():
    """Decision records are populated during compaction."""
    summary = ChunkLogSummary(max_tokens=300)
    messages = []
    for i in range(15):
        messages.append({"role": "user", "content": f"Filler content {i} " * 15})
        messages.append({"role": "assistant", "content": "Ok."})

    summary.summarize(messages)

    records = summary.decision_records
    assert len(records) > 0
    assert records[0]["compaction_events"] > 0
    assert records[0]["tokens_before"] > records[0]["tokens_after"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
