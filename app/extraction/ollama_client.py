"""
Thin, robust client for talking to a local Ollama server.

This module only knows how to send a prompt and get back parsed JSON
(or a clear error). All "what to ask the model" logic lives in
extraction/fact_extractor.py.
"""

import json
import logging
import re
from typing import Optional, Tuple

import requests

from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised for any problem talking to Ollama (connection, timeout, HTTP, etc.)."""


def call_ollama(prompt: str) -> str:
    """
    Send a prompt to the local Ollama server and return the raw text
    response. Raises OllamaError on any failure — callers must catch
    this and treat it as "this page failed", never crash the app.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_NUM_PREDICT,
            "num_ctx": OLLAMA_NUM_CTX,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. Is 'ollama serve' running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS} seconds.") from exc
    except requests.exceptions.RequestException as exc:
        raise OllamaError(f"Request to Ollama failed: {exc}") from exc

    if response.status_code != 200:
        raise OllamaError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")

    try:
        body = response.json()
    except ValueError as exc:
        raise OllamaError("Ollama response was not valid JSON at the HTTP level.") from exc

    text = body.get("response", "")
    if not text or not text.strip():
        raise OllamaError("Ollama returned an empty response.")

    return text


def extract_json(raw_text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Try hard to pull a JSON object out of a raw LLM response, which may
    contain markdown code fences or stray explanatory text around it.

    Returns (parsed_dict, error_message) — exactly one is non-None.
    """
    if not raw_text or not raw_text.strip():
        return None, "Empty text passed to JSON extractor."

    candidate = raw_text.strip()

    # 1. Try parsing directly (the fast path, since we asked for format="json").
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences like ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip()), None
        except json.JSONDecodeError:
            pass

    # 3. Fall back to the substring between the first "{" and last "}",
    #    in case the model added explanatory text around the JSON.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1]), None
        except json.JSONDecodeError as exc:
            return None, f"Could not parse JSON even after cleanup: {exc}"

    return None, "No JSON object could be found in the model's response."