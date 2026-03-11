"""Tests for utility functions: _sha256, _estimate_tokens, extract_keywords, score_chunk."""

from __future__ import annotations

import pytest

from engine import _estimate_tokens, _sha256, extract_keywords, score_chunk


# ── _sha256 ──────────────────────────────────────────────────────────────


class TestSha256:
    def test_deterministic(self):
        assert _sha256("hello") == _sha256("hello")

    def test_different_inputs(self):
        assert _sha256("hello") != _sha256("world")

    def test_empty_string(self):
        h = _sha256("")
        assert isinstance(h, str) and len(h) == 64

    def test_unicode(self):
        h = _sha256("こんにちは世界 🌍")
        assert isinstance(h, str) and len(h) == 64

    def test_long_string(self):
        h = _sha256("x" * 100_000)
        assert isinstance(h, str) and len(h) == 64

    def test_whitespace_only(self):
        assert _sha256("   ") != _sha256("")


# ── _estimate_tokens ─────────────────────────────────────────────────────


class TestEstimateTokens:
    def test_empty_string_returns_one(self):
        assert _estimate_tokens("") == 1

    def test_short_string(self):
        # 10 chars / 3.2 = 3.125 -> int 3
        assert _estimate_tokens("a" * 10) == 3

    def test_hundred_chars(self):
        # 100 / 3.2 = 31.25 -> 31
        assert _estimate_tokens("a" * 100) == 31

    def test_single_char(self):
        # 1 / 3.2 = 0.3125 -> 0, but min is 1
        assert _estimate_tokens("a") == 1

    def test_unicode_chars(self):
        # Unicode chars are multi-byte in UTF-8 but len() counts characters
        result = _estimate_tokens("日本語")  # 3 chars
        assert result == 1  # 3 / 3.2 = 0.9375 -> 0 -> min 1

    def test_always_positive(self):
        for length in range(0, 100):
            assert _estimate_tokens("x" * length) >= 1


# ── extract_keywords ─────────────────────────────────────────────────────


class TestExtractKeywords:
    def test_filenames(self):
        kw = extract_keywords("Check utils.py and config.yaml")
        assert "utils.py" in kw
        assert "config.yaml" in kw

    def test_function_names(self):
        kw = extract_keywords("def process_data(): pass")
        assert "process_data" in kw

    def test_class_names(self):
        kw = extract_keywords("class MyHandler: pass")
        assert "myhandler" in kw

    def test_error_indicators(self):
        kw = extract_keywords("There is an Error in the traceback")
        assert "error" in kw
        assert "traceback" in kw

    def test_critical_indicator(self):
        kw = extract_keywords("CRITICAL failure detected")
        assert "critical" in kw

    def test_quoted_content(self):
        kw = extract_keywords('Check `my_function` and "another_item"')
        assert "my_function" in kw
        assert "another_item" in kw

    def test_short_quoted_ignored(self):
        # Quoted strings <= 2 chars are ignored
        kw = extract_keywords('Use `ab` for testing')
        assert "ab" not in kw

    def test_ip_address(self):
        kw = extract_keywords("Server at 192.168.1.100")
        assert "192.168.1.100" in kw

    def test_date_pattern(self):
        kw = extract_keywords("Due by January 15th")
        assert "january 15th" in kw

    def test_empty_string(self):
        kw = extract_keywords("")
        assert isinstance(kw, set)
        assert len(kw) == 0

    def test_no_keywords_plain_text(self):
        # Plain text without any patterns
        kw = extract_keywords("the quick brown fox jumps over the lazy dog")
        # Should still return a set (may have some short number matches)
        assert isinstance(kw, set)

    def test_multiple_file_extensions(self):
        kw = extract_keywords("Edit main.rs and index.html and style.css")
        assert "main.rs" in kw
        assert "index.html" in kw
        assert "style.css" in kw

    def test_function_keyword_fn(self):
        kw = extract_keywords("fn handle_request() {}")
        assert "handle_request" in kw

    def test_javascript_function(self):
        kw = extract_keywords("function calculateTotal() {}")
        assert "calculatetotal" in kw

    def test_numeric_values(self):
        kw = extract_keywords("Port 8080 and timeout 300")
        assert "8080" in kw
        assert "300" in kw


# ── score_chunk ──────────────────────────────────────────────────────────


class TestScoreChunk:
    def test_no_keywords_returns_half(self):
        assert score_chunk("some text", set()) == 0.5

    def test_no_matches_returns_half(self):
        assert score_chunk("some text", {"foobar.py", "baz"}) == 0.5

    def test_three_plus_matches_returns_two(self):
        text = "check utils.py and config.yaml for the error traceback"
        kw = {"utils.py", "config.yaml", "error", "traceback"}
        assert score_chunk(text, kw) == 2.0

    def test_one_match(self):
        text = "check utils.py for bugs"
        kw = {"utils.py", "config.yaml", "traceback"}
        score = score_chunk(text, kw)
        # 1 match: 0.5 + (1/3) * 1.5 = 1.0
        assert score == pytest.approx(1.0)

    def test_two_matches(self):
        text = "check utils.py and config.yaml"
        kw = {"utils.py", "config.yaml", "traceback"}
        score = score_chunk(text, kw)
        # 2 matches: 0.5 + (2/3) * 1.5 = 1.5
        assert score == pytest.approx(1.5)

    def test_case_insensitive(self):
        text = "ERROR in Utils.py"
        kw = {"error", "utils.py"}
        score = score_chunk(text, kw)
        assert score > 0.5

    def test_score_range(self):
        text = "some random content"
        kw = {"anything"}
        score = score_chunk(text, kw)
        assert 0.5 <= score <= 2.0

    def test_many_keywords_cap_at_two(self):
        text = "a b c d e f g h i j"
        kw = set(text.split())
        score = score_chunk(text, kw)
        assert score == 2.0
