"""
llm_client.py
-------------
Thin wrapper around the Groq API (OpenAI-compatible chat completions).

Key design decision (anti-hallucination):
The LLM is NEVER trusted to type out a quote or a timestamp from memory.
It is only allowed to:
  1. write a short synthesized answer in its own words, and
  2. select verbatim substrings from the turns it was given, tagged with
     the turn_index they came from.

Every returned "quote" is validated downstream (see pipeline.py) against
the actual turn text with an exact substring check. If validation fails,
the quote is dropped rather than shown to the user. Timestamps are never
requested from the LLM at all -- they are attached in code from the
Turn object matching the returned turn_index.
"""
import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.environ.get("HASAMEX_MODEL", "openai/gpt-oss-120b")

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it before running the app, "
                "e.g. `export GROQ_API_KEY=gsk_...`"
            )
        _client = Groq(api_key=api_key)
    return _client


def call_json(system_prompt: str, user_prompt: str, max_tokens: int = 1200) -> dict:
    """
    Calls Groq (Llama 3.3 70B by default) with a system prompt that demands
    JSON-only output, then parses it. Uses Groq's native JSON mode
    (response_format={"type": "json_object"}) and strips stray markdown
    fences defensively in case the model adds them anyway.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON.\nRaw output:\n{raw}") from e