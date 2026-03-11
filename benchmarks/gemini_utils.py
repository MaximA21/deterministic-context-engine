"""Shared utilities for Gemini benchmarks using google-genai SDK."""

from __future__ import annotations

import os
import time

# Default model for all Gemini benchmarks
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"


def call_gemini(
    messages: list[dict[str, str]],
    system_prompt: str,
    api_key: str,
    model: str = GEMINI_MODEL,
    max_output_tokens: int = 512,
    max_retries: int = 5,
) -> dict:
    """Call Gemini API with a list of messages.

    Merges consecutive same-role messages (Gemini requires alternating roles).
    Returns dict with answer, ttft, input_tokens, output_tokens, error.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    # Merge consecutive same-role messages into Content objects
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        if contents and contents[-1].role == role:
            contents[-1] = types.Content(
                role=role,
                parts=[types.Part(text=contents[-1].parts[0].text + "\n\n" + msg["content"])],
            )
        else:
            contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.0,
        max_output_tokens=max_output_tokens,
    )

    for attempt in range(max_retries):
        try:
            t0 = time.time()
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            ttft = time.time() - t0
            answer = response.text or ""
            input_tokens = 0
            output_tokens = 0
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
            return {
                "answer": answer,
                "ttft": round(ttft, 4),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                return {
                    "answer": "",
                    "ttft": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error": str(e),
                }
