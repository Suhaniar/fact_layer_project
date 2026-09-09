"""
Turns raw page text into a list of structured, evidence-grounded facts
by prompting the local LLM (through ollama_client) and validating what
comes back.
"""

import logging
from typing import Dict, List

from app.config import MAX_PAGE_TEXT_CHARS, MIN_PAGE_TEXT_LENGTH
from app.extraction.ollama_client import OllamaError, call_ollama, extract_json

logger = logging.getLogger(__name__)

REQUIRED_FACT_FIELDS = [
    "subject", "predicate", "value", "unit",
    "time_period", "scope", "location", "confidence", "evidence",
]


class ExtractionError(Exception):
    """Raised when a page's facts could not be extracted for any reason."""


def is_page_worth_processing(text: str) -> bool:
    """
    Cheap heuristic to skip pages that clearly won't contain useful
    facts (blank pages, whitespace, tiny fragments like a lone page
    number). This is the main fix for the old slowness: we no longer
    send every page to the LLM.
    """
    if not text:
        return False
    cleaned = " ".join(text.split())
    return len(cleaned) >= MIN_PAGE_TEXT_LENGTH


def _build_prompt(text: str, document_name: str, page_number: int) -> str:
    """Build a concise, evidence-focused extraction prompt."""
    trimmed_text = text[:MAX_PAGE_TEXT_CHARS]
    return f"""You extract structured facts from a single page of a business document.

Document: {document_name}
Page: {page_number}

Rules:
- Only extract facts that are EXPLICITLY stated in the text below.
- Do NOT calculate, infer, or guess any fact not directly stated.
- Every fact MUST include a short "evidence" string copied from the text
  that supports it.
- If there are no clear facts on this page, return {{"facts": []}}.
- Extract both numeric facts (metrics, financial values, percentages,
  counts, dates, fiscal years, ratios) and simple semantic statements.

Return ONLY valid JSON in exactly this shape, nothing else:
{{
  "facts": [
    {{
      "subject": "",
      "predicate": "",
      "value": "",
      "unit": "",
      "time_period": "",
      "scope": "",
      "location": "",
      "confidence": 0.0,
      "evidence": ""
    }}
  ]
}}

Page text:
\"\"\"
{trimmed_text}
\"\"\"
"""


def _validate_raw_fact(raw_fact: Dict) -> bool:
    """A fact is only usable if it has non-empty subject/predicate/value/evidence."""
    if not isinstance(raw_fact, dict):
        return False
    for key in ("subject", "predicate", "value", "evidence"):
        if not str(raw_fact.get(key, "")).strip():
            return False
    return True


def extract_facts_from_page(text: str, document_name: str, page_number: int) -> List[Dict]:
    """
    Extract facts from a single page of text.

    Returns a list of raw fact dicts (not yet normalized), each already
    stamped with "document", "page", and guaranteed to have evidence.

    Raises ExtractionError on any failure — pipeline.py catches this,
    logs it as a page failure, and continues with the next page.
    """
    prompt = _build_prompt(text, document_name, page_number)

    try:
        raw_response = call_ollama(prompt)
    except OllamaError as exc:
        raise ExtractionError(str(exc)) from exc

    parsed, error = extract_json(raw_response)
    if error or parsed is None:
        raise ExtractionError(f"Could not parse LLM output as JSON: {error}")

    raw_facts = parsed.get("facts", [])
    if not isinstance(raw_facts, list):
        raise ExtractionError("LLM output did not contain a 'facts' list.")

    facts: List[Dict] = []
    for raw_fact in raw_facts:
        if not _validate_raw_fact(raw_fact):
            # Discard individual malformed facts instead of failing the whole page.
            logger.debug("Discarding malformed fact on page %s: %s", page_number, raw_fact)
            continue

        fact = {field: raw_fact.get(field, "") for field in REQUIRED_FACT_FIELDS}
        fact["document"] = document_name
        fact["page"] = page_number
        facts.append(fact)

    return facts