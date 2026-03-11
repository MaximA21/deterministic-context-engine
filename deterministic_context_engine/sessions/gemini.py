"""Google Gemini API session with ChunkLog context management."""
from __future__ import annotations

import os
import time
from typing import Any


class GeminiSession:
    """Wrapper around Google Gemini API with context management via ChunkLog."""

    def __init__(self, chunk_log, model: str = "gemini-2.0-flash", api_key: str | None = None, max_retries: int = 5):
        from google.genai import Client
        from google.genai.types import Content, Part, GenerateContentConfig

        self.chunk_log = chunk_log
        self.model = model
        self.max_retries = max_retries
        self._client = Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self._Content = Content
        self._Part = Part
        self._Config = GenerateContentConfig
        self._ttft_samples: list[float] = []
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_turns = 0

    @property
    def avg_ttft(self) -> float:
        return sum(self._ttft_samples) / len(self._ttft_samples) if self._ttft_samples else 0.0

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_turns(self) -> int:
        return self._total_turns

    def _build_contents(self):
        """Build Gemini-compatible contents with alternating roles."""
        ctx = self.chunk_log.get_context()
        contents = []
        for msg in ctx:
            role = "model" if msg["role"] == "assistant" else "user"
            if contents and contents[-1].role == role:
                contents[-1].parts.append(self._Part(text=msg["content"]))
            else:
                contents.append(self._Content(role=role, parts=[self._Part(text=msg["content"])]))
        return contents

    def chat(self, system_prompt: str | None = None, max_completion_tokens: int | None = None) -> str:
        contents = self._build_contents()

        config = self._Config(
            system_instruction=system_prompt,
            temperature=0.0,
            max_output_tokens=max_completion_tokens,
        )

        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                response = self._client.models.generate_content(
                    model=self.model, contents=contents, config=config,
                )
                self._ttft_samples.append(time.time() - t0)

                text = response.text or ""

                if response.usage_metadata:
                    self._total_input_tokens += response.usage_metadata.prompt_token_count
                    self._total_output_tokens += response.usage_metadata.candidates_token_count

                self._total_turns += 1
                self.chunk_log.append("assistant", text, priority=1.0)
                self.chunk_log.next_turn()
                return text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Gemini API failed after {self.max_retries} retries: {e}") from e
        return ""

    def get_metrics(self) -> dict[str, Any]:
        return {
            "avg_ttft": round(self.avg_ttft, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_turns": self.total_turns,
            "context_size_tokens": self.chunk_log.get_context_tokens(),
            "compaction_events": self.chunk_log.compaction_count,
        }
