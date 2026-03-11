"""Tests for CerebrasSession and GeminiSession with mocked API clients.

Uses sys.modules injection to mock SDK imports without requiring the actual
SDKs to be importable (or waiting for slow SDK initialization).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from unittest.mock import MagicMock, patch, call

import pytest

from engine import ChunkLog


# ── Mock helpers ─────────────────────────────────────────────────────────


@dataclass
class MockUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class MockResponse:
    choices: list
    usage: MockUsage | None


@dataclass
class MockGeminiUsage:
    prompt_token_count: int
    candidates_token_count: int


def make_cerebras_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    msg = MagicMock()
    msg.content = text
    msg.reasoning = None
    choice = MagicMock()
    choice.message = msg
    usage = MockUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return MockResponse(choices=[choice], usage=usage)


def make_gemini_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = MockGeminiUsage(
        prompt_token_count=prompt_tokens,
        candidates_token_count=completion_tokens,
    )
    return resp


# ── CerebrasSession ─────────────────────────────────────────────────────


class TestCerebrasSession:
    @pytest.fixture(autouse=True)
    def setup_cerebras_mock(self):
        """Install fresh mock cerebras SDK for each test."""
        mock_sdk = ModuleType("cerebras")
        mock_cloud = ModuleType("cerebras.cloud")
        mock_sdk_mod = ModuleType("cerebras.cloud.sdk")

        self.mock_cerebras_cls = MagicMock()
        self.mock_client = MagicMock()
        self.mock_cerebras_cls.return_value = self.mock_client

        mock_sdk_mod.Cerebras = self.mock_cerebras_cls
        mock_cloud.sdk = mock_sdk_mod
        mock_sdk.cloud = mock_cloud

        # Force-replace to get fresh mocks each test
        sys.modules["cerebras"] = mock_sdk
        sys.modules["cerebras.cloud"] = mock_cloud
        sys.modules["cerebras.cloud.sdk"] = mock_sdk_mod

        # Clear any cached imports in the session module
        session_mod_key = "deterministic_context_engine.sessions.cerebras"
        if session_mod_key in sys.modules:
            del sys.modules[session_mod_key]

        yield

    def _make_session(self, log):
        from deterministic_context_engine.sessions.cerebras import CerebrasSession

        return CerebrasSession(log, api_key="test-key")

    def test_chat_basic(self):
        self.mock_client.chat.completions.create.return_value = make_cerebras_response("Hello!")
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Hi")
        session = self._make_session(log)
        response = session.chat()

        assert response == "Hello!"
        assert session.total_turns == 1
        log.close()

    def test_chat_with_system_prompt(self):
        self.mock_client.chat.completions.create.return_value = make_cerebras_response("I'm helpful")
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Help me")
        session = self._make_session(log)
        response = session.chat(system_prompt="You are helpful")

        assert response == "I'm helpful"
        call_args = self.mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        log.close()

    def test_metrics(self):
        self.mock_client.chat.completions.create.return_value = make_cerebras_response(
            "Response", prompt_tokens=100, completion_tokens=50
        )
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.chat()

        metrics = session.get_metrics()
        assert metrics["total_input_tokens"] == 100
        assert metrics["total_output_tokens"] == 50
        assert metrics["total_tokens"] == 150
        assert metrics["total_turns"] == 1
        assert metrics["avg_ttft"] >= 0  # Mock completes instantly, so ≥ 0
        log.close()

    def test_avg_ttft_zero_before_chat(self):
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        session = self._make_session(log)
        assert session.avg_ttft == 0.0
        log.close()

    def test_response_appended_to_context(self):
        self.mock_client.chat.completions.create.return_value = make_cerebras_response("AI response")
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Query")
        session = self._make_session(log)
        session.chat()

        ctx = log.get_context()
        assert any(m["content"] == "AI response" for m in ctx)
        log.close()

    def test_retry_on_failure(self):
        self.mock_client.chat.completions.create.side_effect = [
            RuntimeError("API error"),
            make_cerebras_response("Recovered"),
        ]
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.max_retries = 3

        import deterministic_context_engine.sessions.cerebras as cmod
        with patch.object(cmod.time, "sleep"):
            response = session.chat()

        assert response == "Recovered"
        log.close()

    def test_all_retries_exhausted(self):
        self.mock_client.chat.completions.create.side_effect = RuntimeError("Persistent error")
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.max_retries = 2

        import deterministic_context_engine.sessions.cerebras as cmod
        with patch.object(cmod.time, "sleep"):
            with pytest.raises(RuntimeError, match="failed after 2 retries"):
                session.chat()
        log.close()

    def test_empty_response_content(self):
        msg = MagicMock()
        msg.content = ""
        msg.reasoning = "Thinking about it"
        choice = MagicMock()
        choice.message = msg
        resp = MockResponse(
            choices=[choice],
            usage=MockUsage(prompt_tokens=10, completion_tokens=5),
        )
        self.mock_client.chat.completions.create.return_value = resp

        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        response = session.chat()

        assert response == "Thinking about it"
        log.close()

    def test_none_response_content(self):
        msg = MagicMock()
        msg.content = None
        msg.reasoning = None
        choice = MagicMock()
        choice.message = msg
        resp = MockResponse(
            choices=[choice],
            usage=MockUsage(prompt_tokens=10, completion_tokens=5),
        )
        self.mock_client.chat.completions.create.return_value = resp

        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        response = session.chat()
        assert response == ""
        log.close()

    def test_no_usage_metadata(self):
        resp = MockResponse(
            choices=[MagicMock(message=MagicMock(content="Hi", reasoning=None))],
            usage=None,
        )
        self.mock_client.chat.completions.create.return_value = resp

        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.chat()

        assert session.total_tokens == 0
        log.close()

    def test_max_completion_tokens(self):
        self.mock_client.chat.completions.create.return_value = make_cerebras_response("Hi")

        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.chat(max_completion_tokens=100)

        call_args = self.mock_client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] <= 100
        log.close()


# ── GeminiSession ────────────────────────────────────────────────────────


class TestGeminiSession:
    @pytest.fixture(autouse=True)
    def setup_genai_mock(self):
        """Install fresh mock google.genai SDK for each test."""
        # google may already exist as a namespace package
        if "google" not in sys.modules:
            sys.modules["google"] = ModuleType("google")

        mock_genai = ModuleType("google.genai")
        self.mock_genai_client_cls = MagicMock()
        self.mock_client = MagicMock()
        self.mock_genai_client_cls.return_value = self.mock_client
        mock_genai.Client = self.mock_genai_client_cls

        # Create types submodule with real-ish Content/Part/Config
        mock_types = ModuleType("google.genai.types")

        # Use simple dataclass-like containers for types
        class FakeContent:
            def __init__(self, role=None, parts=None):
                self.role = role
                self.parts = parts or []

        class FakePart:
            def __init__(self, text=None):
                self.text = text

        class FakeConfig:
            def __init__(self, system_instruction=None, temperature=None, max_output_tokens=None):
                self.system_instruction = system_instruction
                self.temperature = temperature
                self.max_output_tokens = max_output_tokens

        mock_types.Content = FakeContent
        mock_types.Part = FakePart
        mock_types.GenerateContentConfig = FakeConfig
        mock_genai.types = mock_types

        sys.modules["google.genai"] = mock_genai
        sys.modules["google.genai.types"] = mock_types
        sys.modules["google"].genai = mock_genai

        # Clear cached session module
        session_mod_key = "deterministic_context_engine.sessions.gemini"
        if session_mod_key in sys.modules:
            del sys.modules[session_mod_key]

        yield

    def _make_session(self, log):
        from deterministic_context_engine.sessions.gemini import GeminiSession

        return GeminiSession(log, api_key="test-key")

    def test_chat_basic(self):
        self.mock_client.models.generate_content.return_value = make_gemini_response("Hello from Gemini!")
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Hi")
        session = self._make_session(log)
        response = session.chat()

        assert response == "Hello from Gemini!"
        assert session.total_turns == 1
        log.close()

    def test_metrics(self):
        self.mock_client.models.generate_content.return_value = make_gemini_response(
            "Response", prompt_tokens=100, completion_tokens=50
        )
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.chat()

        metrics = session.get_metrics()
        assert metrics["total_input_tokens"] == 100
        assert metrics["total_output_tokens"] == 50
        assert metrics["total_tokens"] == 150
        assert metrics["total_turns"] == 1
        log.close()

    def test_avg_ttft_zero_before_chat(self):
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        session = self._make_session(log)
        assert session.avg_ttft == 0.0
        log.close()

    def test_retry_on_failure(self):
        self.mock_client.models.generate_content.side_effect = [
            RuntimeError("API error"),
            make_gemini_response("Recovered"),
        ]
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.max_retries = 3

        import deterministic_context_engine.sessions.gemini as gmod
        with patch.object(gmod.time, "sleep"):
            response = session.chat()

        assert response == "Recovered"
        log.close()

    def test_all_retries_exhausted(self):
        self.mock_client.models.generate_content.side_effect = RuntimeError("Persistent error")
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.max_retries = 2

        import deterministic_context_engine.sessions.gemini as gmod
        with patch.object(gmod.time, "sleep"):
            with pytest.raises(RuntimeError, match="failed after 2 retries"):
                session.chat()
        log.close()

    def test_response_appended_to_context(self):
        self.mock_client.models.generate_content.return_value = make_gemini_response("Gemini says hi")
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Query")
        session = self._make_session(log)
        session.chat()

        ctx = log.get_context()
        assert any(m["content"] == "Gemini says hi" for m in ctx)
        log.close()

    def test_empty_response(self):
        resp = MagicMock()
        resp.text = None
        resp.usage_metadata = MockGeminiUsage(prompt_token_count=10, candidates_token_count=0)
        self.mock_client.models.generate_content.return_value = resp

        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        response = session.chat()
        assert response == ""
        log.close()

    def test_no_usage_metadata(self):
        resp = MagicMock()
        resp.text = "Hi"
        resp.usage_metadata = None
        self.mock_client.models.generate_content.return_value = resp

        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.chat()

        assert session.total_tokens == 0
        log.close()

    def test_consecutive_same_role_merged(self):
        """Gemini requires alternating user/model roles."""
        self.mock_client.models.generate_content.return_value = make_gemini_response("Response")
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Message 1")
        log.append("user", "Message 2")
        session = self._make_session(log)
        session.chat()

        assert session.total_turns == 1
        log.close()

    def test_with_system_prompt(self):
        self.mock_client.models.generate_content.return_value = make_gemini_response("OK")
        log = ChunkLog(db_path=":memory:", max_tokens=8000, scoring_mode=None)
        log.append("user", "Test")
        session = self._make_session(log)
        session.chat(system_prompt="You are helpful")

        call_args = self.mock_client.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert config.system_instruction == "You are helpful"
        log.close()
