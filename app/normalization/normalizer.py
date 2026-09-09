"""
Cleans and standardizes facts coming out of fact_extractor.py before
they are stored — without ever touching the evidence, document, or
page fields, which must stay exactly as extracted.

Also validates that a fact's claimed subject/predicate/value/unit/
time_period are actually supported by its own evidence text, and
classifies the fact as VALIDATED / REVIEW_REQUIRED / REJECTED.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

STATUS_VALIDATED = "VALIDATED"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATUS_REJECTED = "REJECTED"

MIN_TOKEN_LENGTH = 3


# ---------------------------------------------------------------------------
# Field cleanup
# ---------------------------------------------------------------------------

def _normalize_whitespace(text) -> str:
    if text is None:
        return ""
    return " ".join(str(text).split())


def _normalize_llm_confidence(value):
    """
    Coerce the LLM-provided confidence into [0,1], or None if missing/invalid.
    None means "we don't know" — callers must NOT treat None as 0.0.
    """
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence != confidence:  # NaN check
        return None
    return max(0.0, min(1.0, confidence))


def _compute_final_confidence(llm_confidence, evidence) -> float:
    """
    Independently estimate confidence instead of trusting the LLM's number
    outright. Coarse heuristic — full evidence-matching validation happens
    separately in validate_fact_evidence() below.
    """
    evidence_text = str(evidence or "").strip()
    has_evidence = len(evidence_text) >= 10

    if llm_confidence is None:
        return 0.5 if has_evidence else 0.2

    if not has_evidence:
        return min(llm_confidence, 0.3)  # claimed confidence but no real evidence

    return llm_confidence


def _normalize_value(value) -> str:
    """
    Light-touch numeric cleanup: strip stray whitespace and standardize
    percent-sign / comma spacing. We deliberately do NOT convert
    currencies or units here — that would risk inventing precision the
    source text doesn't actually have.
    """
    text = _normalize_whitespace(value)
    if not text:
        return text

    text = re.sub(r"\s+%", "%", text)                 # "45 %" -> "45%"
    text = re.sub(r"(\d),\s+(\d)", r"\1,\2", text)     # "1, 234" -> "1,234"

    return text


def _normalize_time_period(value) -> str:
    """
    Standardize common fiscal-year phrasings to a compact "FYxx" form,
    e.g. "FY 2023-24" or "FY2023-24" -> "FY24". Text that doesn't match
    a recognized pattern is left untouched rather than risk a wrong rewrite.
    """
    text = _normalize_whitespace(value)
    if not text:
        return text

    match = re.match(r"(?i)^FY\s*20?(\d{2})-(\d{2})$", text)
    if match:
        return f"FY{match.group(2)}"

    match = re.match(r"(?i)^FY\s*20?(\d{2})$", text)
    if match:
        return f"FY{match.group(1)}"

    return text


def normalize_fact(fact: Dict) -> Dict:
    """
    Return a new, normalized copy of a fact dict.

    Accepts "period" as an alias for "time_period". Evidence, document
    and page are passed through unchanged — normalization must never
    alter a fact's traceable source.

    Adds two separate confidence fields:
      - llm_confidence: the model's raw self-rating, clamped to [0,1],
        or None if it was missing/invalid (never silently 0.0).
      - confidence: an independently computed final confidence used for
        filtering/display.
    """
    time_period = fact.get("time_period")
    if not time_period:
        time_period = fact.get("period", "")

    llm_confidence = _normalize_llm_confidence(fact.get("confidence"))
    evidence = fact.get("evidence", "")

    return {
        "subject": _normalize_whitespace(fact.get("subject")),
        "predicate": _normalize_whitespace(fact.get("predicate")),
        "value": _normalize_value(fact.get("value")),
        "unit": _normalize_whitespace(fact.get("unit")),
        "time_period": _normalize_time_period(time_period),
        "scope": _normalize_whitespace(fact.get("scope")),
        "location": _normalize_whitespace(fact.get("location")),
        "llm_confidence": llm_confidence,
        "confidence": _compute_final_confidence(llm_confidence, evidence),
        "document": fact.get("document", ""),   # preserved as-is
        "page": fact.get("page", 0),             # preserved as-is
        "evidence": evidence,                    # preserved as-is, never invented
    }


# ---------------------------------------------------------------------------
# Evidence validation (STEP 2)
# ---------------------------------------------------------------------------

def _clean(text) -> str:
    return str(text or "").strip().lower()


def _tokens(text) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", _clean(text)) if len(t) >= MIN_TOKEN_LENGTH]


def _numeric_core(text) -> str:
    """
    Extract just digits and a decimal point from a value, dropping commas,
    currency symbols, units, and whitespace — so '1,234', '1234', and
    'Rs. 1,234' all reduce to a comparable core.
    """
    digits = re.sub(r"[^\d.]", "", str(text or ""))
    return digits.strip(".")


def _value_supported(value, evidence: str) -> bool:
    value_text = str(value or "").strip()
    if not value_text:
        return False

    numeric_value = _numeric_core(value_text)
    if numeric_value:
        numeric_evidence = _numeric_core(evidence)
        return numeric_value in numeric_evidence

    # Non-numeric value (e.g. "India", "Mumbai"): require it as literal text.
    return _clean(value_text) in _clean(evidence)


def _tokens_supported(text, evidence: str, min_overlap: int = 1) -> bool:
    """At least `min_overlap` significant token(s) from `text` must appear in evidence."""
    text_tokens = set(_tokens(text))
    if not text_tokens:
        return True  # nothing meaningful to check (e.g. empty field)
    evidence_tokens = set(_tokens(evidence))
    return len(text_tokens & evidence_tokens) >= min_overlap


def _unit_supported(unit, evidence: str) -> bool:
    unit_text = str(unit or "").strip()
    if not unit_text:
        return True  # no unit claimed
    evidence_clean = _clean(evidence)
    if _clean(unit_text) in evidence_clean:
        return True
    return _tokens_supported(unit_text, evidence)


def _time_period_supported(time_period, evidence: str) -> bool:
    period_text = str(time_period or "").strip()
    if not period_text:
        return True
    evidence_clean = _clean(evidence)
    if _clean(period_text) in evidence_clean:
        return True
    # Our normalized "FYxx" form often won't literally match source text
    # like "FY 2023-24" — fall back to checking the two-digit / four-digit year.
    match = re.match(r"(?i)^fy(\d{2})$", period_text.strip())
    if match:
        yy = match.group(1)
        return yy in evidence_clean or f"20{yy}" in evidence_clean
    return False


def validate_fact_evidence(fact: Dict) -> Tuple[str, List[str]]:
    """
    Returns (status, reasons).

    - No evidence at all -> REJECTED
    - Value, subject, or predicate not supported by evidence -> REJECTED
      (these are the core claim of the fact)
    - Unit or time_period not clearly supported -> REVIEW_REQUIRED
      (supporting context; missing support is suspicious, not disqualifying)
    - Everything supported -> VALIDATED

    Note: this only checks surface-text support. It does NOT verify counts,
    arithmetic, or deeper consistency (e.g. "7 directors" vs. 8 names listed
    in the evidence) — that is a separate, later verification step.
    """
    evidence = fact.get("evidence", "")

    if not str(evidence).strip():
        return STATUS_REJECTED, ["No evidence text provided."]

    critical_reasons: List[str] = []
    if not _value_supported(fact.get("value"), evidence):
        critical_reasons.append(f"Value '{fact.get('value')}' not found in evidence.")
    if not _tokens_supported(fact.get("subject"), evidence):
        critical_reasons.append(f"Subject '{fact.get('subject')}' not supported by evidence.")
    if not _tokens_supported(fact.get("predicate"), evidence):
        critical_reasons.append(f"Predicate '{fact.get('predicate')}' not supported by evidence.")

    if critical_reasons:
        return STATUS_REJECTED, critical_reasons

    soft_reasons: List[str] = []
    if not _unit_supported(fact.get("unit"), evidence):
        soft_reasons.append(f"Unit '{fact.get('unit')}' not clearly supported by evidence.")
    if not _time_period_supported(fact.get("time_period"), evidence):
        soft_reasons.append(f"Time period '{fact.get('time_period')}' not clearly supported by evidence.")

    if soft_reasons:
        return STATUS_REVIEW_REQUIRED, soft_reasons

    return STATUS_VALIDATED, []