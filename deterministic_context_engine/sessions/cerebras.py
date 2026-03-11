"""Cerebras API session with ChunkLog context management."""
from __future__ import annotations

import os
import time
from typing import Any


class CerebrasSession:
    """Wrapper around Cerebras API with context management via ChunkLog."""

    def __init__(self, chunk_log, model: str = "llama3.1-8b", api_key: str | None = None, max_retries: int = 5):
        from cerebras.cloud.sdk import Cerebras

        self.chunk_log = chunk_log
        self.model = model
        self.max_retries = max_retries
        self._client = Cerebras(api_key=api_key or os.environ.get("CEREBRAS_API_KEY"))
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

    def chat(self, system_prompt: str | None = None, max_completion_tokens: int | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.chunk_log.get_context())

        input_tokens = sum(max(1, len(m["content"]) // 4) for m in messages)
        model_limit = self.chunk_log.max_tokens
        available = max(256, model_limit - input_tokens - 512)
        completion_tokens = min(max_completion_tokens or 16384, available)

        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                response = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=completion_tokens, temperature=0.0,
                )
                self._ttft_samples.append(time.time() - t0)

                choice = response.choices[0]
                text = choice.message.content or ""
                if not text and hasattr(choice.message, 'reasoning') and choice.message.reasoning:
                    text = choice.message.reasoning

                if response.usage:
                    self._total_input_tokens += response.usage.prompt_tokens
                    self._total_output_tokens += response.usage.completion_tokens

                self._total_turns += 1
                self.chunk_log.append("assistant", text, priority=1.0)
                self.chunk_log.next_turn()
                return text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise RuntimeError(f"Cerebras API failed after {self.max_retries} retries: {e}") from e
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
