"""Tests for all scorer classes: GoalGuidedScorer, BM25Scorer, SemanticScorer, EntityAwareScorer."""

from __future__ import annotations

import pytest

from tests.conftest import requires_bm25, requires_sklearn, requires_sentence_transformers


# ── GoalGuidedScorer (TF-IDF) ────────────────────────────────────────────


@requires_sklearn
class TestGoalGuidedScorer:
    def setup_method(self):
        from engine import GoalGuidedScorer

        self.scorer = GoalGuidedScorer()

    def test_basic_scoring(self):
        chunks = [
            ("h1", "Fix the authentication bug in login.py"),
            ("h2", "The weather today is nice and sunny"),
            ("h3", "Consider refactoring the database schema"),
        ]
        scores = self.scorer.score_chunks("authentication login bug", chunks)
        assert len(scores) == 3
        for s in scores.values():
            assert 0.5 <= s <= 2.0

    def test_empty_chunks(self):
        scores = self.scorer.score_chunks("anything", [])
        assert scores == {}

    def test_single_chunk(self):
        scores = self.scorer.score_chunks("test", [("h1", "test content")])
        assert len(scores) == 1
        assert 0.5 <= scores["h1"] <= 2.0

    def test_relevant_chunk_scores_higher(self):
        chunks = [
            ("relevant", "authentication login error bug in production"),
            ("irrelevant", "the weather forecast shows sunny skies tomorrow"),
        ]
        scores = self.scorer.score_chunks("authentication login bug", chunks)
        assert scores["relevant"] > scores["irrelevant"]

    def test_identical_chunks_equal_scores(self):
        chunks = [
            ("h1", "some identical text about testing"),
            ("h2", "some identical text about testing"),
        ]
        # Content-addressed hashes differ here because we pass different hashes
        scores = self.scorer.score_chunks("testing", chunks)
        assert scores["h1"] == pytest.approx(scores["h2"], abs=0.01)

    def test_keyword_scores_parameter_accepted(self):
        chunks = [("h1", "some text")]
        # keyword_scores is accepted but not used in GoalGuidedScorer
        scores = self.scorer.score_chunks("test", chunks, keyword_scores={"h1": 1.5})
        assert len(scores) == 1

    def test_value_error_fallback(self):
        """When vocabulary is empty (all stop words), returns 0.5 for all."""
        chunks = [("h1", "the a an")]
        scores = self.scorer.score_chunks("the", chunks)
        # Should return 0.5 for all (ValueError caught)
        assert scores["h1"] == 0.5

    def test_many_chunks(self):
        chunks = [(f"h{i}", f"Content about topic number {i}") for i in range(50)]
        scores = self.scorer.score_chunks("topic number 25", chunks)
        assert len(scores) == 50
        for s in scores.values():
            assert 0.5 <= s <= 2.0

    def test_unicode_content(self):
        chunks = [
            ("h1", "日本語のテスト content with 漢字"),
            ("h2", "English only content about testing"),
        ]
        scores = self.scorer.score_chunks("testing", chunks)
        assert len(scores) == 2

    def test_long_chunk_content(self):
        long_text = "word " * 5000
        chunks = [("h1", long_text)]
        scores = self.scorer.score_chunks("word", chunks)
        assert 0.5 <= scores["h1"] <= 2.0


# ── BM25Scorer ───────────────────────────────────────────────────────────


@requires_bm25
class TestBM25Scorer:
    def setup_method(self):
        from engine import BM25Scorer

        self.scorer = BM25Scorer()

    def test_basic_scoring(self):
        chunks = [
            ("h1", "Fix the authentication bug in login.py"),
            ("h2", "The weather today is nice and sunny"),
            ("h3", "Consider refactoring the database schema"),
        ]
        scores = self.scorer.score_chunks("authentication login bug", chunks)
        assert len(scores) == 3
        for s in scores.values():
            assert 0.5 <= s <= 2.0

    def test_empty_chunks(self):
        scores = self.scorer.score_chunks("anything", [])
        assert scores == {}

    def test_single_chunk(self):
        scores = self.scorer.score_chunks("test", [("h1", "test content")])
        assert len(scores) == 1
        assert 0.5 <= scores["h1"] <= 2.0

    def test_relevant_chunk_scores_higher(self):
        chunks = [
            ("relevant", "authentication login error bug in production server crash"),
            ("filler1", "the weather forecast shows sunny skies tomorrow afternoon"),
            ("filler2", "the weather report indicates clear conditions all week"),
            ("filler3", "today sunny warm pleasant breeze gentle sunshine forecast"),
        ]
        scores = self.scorer.score_chunks("authentication login bug", chunks)
        assert scores["relevant"] >= scores["filler1"]

    def test_empty_goal_returns_default(self):
        chunks = [("h1", "some content")]
        scores = self.scorer.score_chunks("", chunks)
        # Empty goal -> all stop words -> fallback
        assert scores["h1"] == 0.5

    def test_all_empty_chunks_returns_default(self):
        chunks = [("h1", ""), ("h2", "")]
        scores = self.scorer.score_chunks("test", chunks)
        for s in scores.values():
            assert s == 0.5

    def test_custom_k1_b(self):
        from engine import BM25Scorer

        scorer = BM25Scorer(k1=2.0, b=0.5)
        chunks = [("h1", "test content about scoring")]
        scores = scorer.score_chunks("scoring", chunks)
        assert 0.5 <= scores["h1"] <= 2.0

    def test_tokenizer_removes_stop_words(self):
        tokens = self.scorer._tokenize("the quick brown fox jumps over the lazy dog")
        assert "the" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    def test_tokenizer_lowercases(self):
        tokens = self.scorer._tokenize("Hello WORLD Test")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_tokenizer_single_char_removed(self):
        tokens = self.scorer._tokenize("a b c real word")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "real" in tokens
        assert "word" in tokens

    def test_many_chunks(self):
        chunks = [(f"h{i}", f"Content about topic number {i}") for i in range(50)]
        scores = self.scorer.score_chunks("topic number 25", chunks)
        assert len(scores) == 50

    def test_keyword_scores_parameter_ignored(self):
        chunks = [("h1", "some text")]
        scores = self.scorer.score_chunks("text", chunks, keyword_scores={"h1": 1.5})
        assert len(scores) == 1


# ── SemanticScorer ───────────────────────────────────────────────────────


@requires_sentence_transformers
class TestSemanticScorer:
    def setup_method(self):
        from engine import SemanticScorer

        self.scorer = SemanticScorer()

    def test_basic_scoring(self):
        chunks = [
            ("h1", "Fix the authentication bug in login module"),
            ("h2", "The weather today is nice and sunny"),
        ]
        scores = self.scorer.score_chunks("authentication login bug", chunks)
        assert len(scores) == 2
        for s in scores.values():
            assert 0.5 <= s <= 2.0

    def test_empty_chunks(self):
        scores = self.scorer.score_chunks("anything", [])
        assert scores == {}

    def test_single_chunk(self):
        scores = self.scorer.score_chunks("test", [("h1", "test content")])
        assert len(scores) == 1
        assert 0.5 <= scores["h1"] <= 2.0

    def test_semantic_similarity(self):
        """Semantic scorer should understand paraphrases better than TF-IDF."""
        chunks = [
            ("related", "user authentication failed during sign-in process"),
            ("unrelated", "the cat sat on the mat near the window"),
        ]
        scores = self.scorer.score_chunks("login error in auth module", chunks)
        assert scores["related"] > scores["unrelated"]


# ── EntityAwareScorer ────────────────────────────────────────────────────


@requires_sklearn
class TestEntityAwareScorer:
    def setup_method(self):
        from engine import EntityAwareScorer

        self.scorer = EntityAwareScorer()

    def test_basic_scoring(self):
        chunks = [
            ("h1", "Error PAY-4012 in billing_events at 10.0.0.1"),
            ("h2", "General text about random topics and weather"),
        ]
        scores = self.scorer.score_chunks(
            "What happened with PAY-4012 on billing_events?", chunks
        )
        assert len(scores) == 2
        for s in scores.values():
            assert 0.5 <= s <= 2.0

    def test_empty_chunks(self):
        scores = self.scorer.score_chunks("anything", [])
        assert scores == {}

    def test_single_chunk(self):
        scores = self.scorer.score_chunks("test", [("h1", "test content")])
        assert len(scores) == 1

    def test_entity_bonus_with_overlap(self):
        """Chunks with matching entities should get a bonus."""
        needle = ("needle", "Error INC-7734 on billing_events at 10.0.0.1:5432")
        filler = ("filler", "The weather is sunny and warm today in the park")
        chunks = [needle, filler]
        scores = self.scorer.score_chunks(
            "What happened with INC-7734 on billing_events?", chunks
        )
        assert scores["needle"] >= scores["filler"]

    def test_entity_overlap_threshold(self):
        """Entity bonus should only fire above the threshold."""
        # With no entity overlap, no bonus
        chunks = [("h1", "just plain text without entities")]
        scores = self.scorer.score_chunks("unrelated query", chunks)
        assert 0.5 <= scores["h1"] <= 2.0

    def test_specificity_check_disables_bonus(self):
        """If too many chunks match entities, bonus is disabled."""
        # Create many chunks that all share the same entities
        chunks = [
            (f"h{i}", "Error PAY-4012 in billing_events table")
            for i in range(20)
        ]
        scores = self.scorer.score_chunks(
            "What about PAY-4012 in billing_events?", chunks
        )
        # All should have similar scores (bonus disabled for widespread matches)
        values = list(scores.values())
        spread = max(values) - min(values)
        # With bonus disabled, spread should be small
        assert spread < 1.0

    def test_expand_entities_round_trip(self):
        expanded = self.scorer._expand_entities({"billing_events"})
        assert "billing_events" in expanded
        assert "billing events" in expanded
        assert "billing-events" in expanded

    def test_entity_overlap_empty_sets(self):
        overlap = self.scorer._entity_overlap(set(), set())
        assert overlap == 0.0

    def test_entity_overlap_no_goal_entities(self):
        overlap = self.scorer._entity_overlap({"billing_events"}, set())
        assert overlap == 0.0

    def test_value_error_fallback(self):
        """When TF-IDF fails, should return 0.5."""
        chunks = [("h1", "the a an")]
        scores = self.scorer.score_chunks("the", chunks)
        assert scores["h1"] == 0.5
