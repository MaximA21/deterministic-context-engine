"""Shared fixtures and markers for the context engine test suite."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Ensure engine.py is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Conditional dependency markers
_has_sklearn = importlib.util.find_spec("sklearn") is not None
_has_sentence_transformers = importlib.util.find_spec("sentence_transformers") is not None
_has_rank_bm25 = importlib.util.find_spec("rank_bm25") is not None

requires_sklearn = pytest.mark.skipif(not _has_sklearn, reason="scikit-learn not installed")
requires_sentence_transformers = pytest.mark.skipif(
    not _has_sentence_transformers, reason="sentence-transformers not installed"
)
requires_bm25 = pytest.mark.skipif(not _has_rank_bm25, reason="rank-bm25 not installed")


@pytest.fixture
def memory_log():
    """Create a ChunkLog with in-memory SQLite for testing."""
    from engine import ChunkLog

    log = ChunkLog(db_path=":memory:", max_tokens=10000)
    yield log
    log.close()


@pytest.fixture
def tiny_log():
    """ChunkLog with a very small context window to force compaction."""
    from engine import ChunkLog

    log = ChunkLog(
        db_path=":memory:",
        max_tokens=100,
        soft_threshold=0.7,
        hard_threshold=0.9,
    )
    yield log
    log.close()
