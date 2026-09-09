"""
Reasoning layer: decides how two related facts relate to each other.

The key idea: a numeric difference is NOT automatically a contradiction.
We walk through several checks (same metric? compatible context? same
time period?) before deciding, the way a careful analyst would read two
numbers before calling them conflicting.
"""

import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relationship labels
# ---------------------------------------------------------------------------

RELATIONSHIP_AGREEMENT = "AGREEMENT"
RELATIONSHIP_CONTRADICTION = "GENUINE_CONTRADICTION"
RELATIONSHIP_CONTEXTUAL = "CONTEXTUAL_DIFFERENCE"
RELATIONSHIP_EVOLUTION = "EVOLUTION"
RELATIONSHIP_UNRELATED = "UNRELATED"
RELATIONSHIP_EXTRACTION_FAILURE = "EXTRACTION_FAILURE"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Two numeric values within 2% of each other are treated as agreement.
# This prevents small rounding/reporting differences from being
# incorrectly classified as contradictions.
NUMERIC_AGREEMENT_TOLERANCE = 0.02


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _clean(value) -> str:
    """Convert a value to a normalized lowercase string."""
    return str(value if value is not None else "").strip().lower()


def _is_extraction_failure(fact: Dict) -> bool:
    """
    Determine whether a fact is unusable for comparison.

    A fact is considered invalid if it is missing any of:
    - subject
    - predicate
    - value
    - evidence
    """
    if not fact:
        return True

    for key in ("subject", "predicate", "value", "evidence"):
        if not str(fact.get(key, "")).strip():
            return True

    return False


def _tokens(text: str) -> set:
    """Convert text into a set of alphanumeric tokens."""
    return set(re.findall(r"[a-z0-9]+", _clean(text)))


def _subjects_match(a: Dict, b: Dict) -> bool:
    """
    Check whether two facts refer to the same subject.

    Exact matches are accepted immediately. Otherwise, an overlapping
    token is sufficient, e.g.:
        'Delhivery'
        'Delhivery Ltd'
    """
    subj_a = _clean(a.get("subject"))
    subj_b = _clean(b.get("subject"))

    if subj_a == subj_b:
        return True

    tokens_a = _tokens(subj_a)
    tokens_b = _tokens(subj_b)

    return bool(tokens_a & tokens_b)


def _predicates_match(a: Dict, b: Dict) -> bool:
    """
    Check whether two facts describe the same metric/predicate.
    """
    pred_a = _clean(a.get("predicate"))
    pred_b = _clean(b.get("predicate"))

    if pred_a == pred_b:
        return True

    tokens_a = _tokens(pred_a)
    tokens_b = _tokens(pred_b)

    if not tokens_a or not tokens_b:
        return False

    overlap = tokens_a & tokens_b

    return len(overlap) >= min(len(tokens_a), len(tokens_b)) * 0.5


def _units_compatible(a: Dict, b: Dict) -> bool:
    """
    Check whether the units are compatible.

    If one fact does not specify a unit, the comparison is allowed.
    """
    unit_a = _clean(a.get("unit"))
    unit_b = _clean(b.get("unit"))

    if not unit_a or not unit_b:
        return True

    return unit_a == unit_b


def _try_parse_number(value) -> Optional[float]:
    """
    Extract a numeric value from text.

    Handles values containing:
    - commas
    - currency symbols
    - percentage signs
    - surrounding text

    Examples:
        '1,234.5' -> 1234.5
        '$500'    -> 500.0
        '25%'     -> 25.0
    """
    if value is None:
        return None

    text = str(value).strip().replace(",", "")

    # Keep only digits, decimal point and minus sign.
    text = re.sub(r"[^\d.\-]", "", text)

    if text in ("", "-", "."):
        return None

    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_facts(
    fact_a: Dict,
    fact_b: Dict,
    similarity: Optional[float] = None
) -> Dict:
    """
    Compare two facts and classify their relationship.

    Parameters
    ----------
    fact_a : Dict
        First extracted fact.

    fact_b : Dict
        Second extracted fact.

    similarity : Optional[float]
        Semantic similarity score between the two facts.

    Returns
    -------
    Dict
        {
            "relationship": ...,
            "similarity": ...,
            "explanation": ...
        }
    """

    sim = similarity if similarity is not None else 0.0

    # -----------------------------------------------------------------------
    # Step 0: Check whether the facts can be compared
    # -----------------------------------------------------------------------

    if (
        _is_extraction_failure(fact_a)
        or _is_extraction_failure(fact_b)
    ):
        return {
            "relationship": RELATIONSHIP_EXTRACTION_FAILURE,
            "similarity": sim,
            "explanation": (
                "One or both facts are missing core fields or evidence, "
                "so no reliable comparison can be made."
            ),
        }

    # -----------------------------------------------------------------------
    # Step 1: Check subject
    # -----------------------------------------------------------------------

    if not _subjects_match(fact_a, fact_b):
        return {
            "relationship": RELATIONSHIP_UNRELATED,
            "similarity": sim,
            "explanation": (
                "The facts refer to different subjects, "
                "so they are not directly comparable."
            ),
        }

    # -----------------------------------------------------------------------
    # Step 2: Check predicate / metric
    # -----------------------------------------------------------------------

    if not _predicates_match(fact_a, fact_b):
        return {
            "relationship": RELATIONSHIP_UNRELATED,
            "similarity": sim,
            "explanation": (
                "The subjects are related but the facts describe "
                "different metrics or statements, so they are not "
                "directly comparable."
            ),
        }

    # -----------------------------------------------------------------------
    # Step 3: Check units
    # -----------------------------------------------------------------------

    if not _units_compatible(fact_a, fact_b):
        return {
            "relationship": RELATIONSHIP_CONTEXTUAL,
            "similarity": sim,
            "explanation": (
                f"Same metric reported in different units "
                f"('{fact_a.get('unit')}' vs '{fact_b.get('unit')}'), "
                "so the values cannot be safely compared as a contradiction."
            ),
        }

    # -----------------------------------------------------------------------
    # Extract context
    # -----------------------------------------------------------------------

    time_a = _clean(fact_a.get("time_period"))
    time_b = _clean(fact_b.get("time_period"))

    scope_a = _clean(fact_a.get("scope"))
    scope_b = _clean(fact_b.get("scope"))

    loc_a = _clean(fact_a.get("location"))
    loc_b = _clean(fact_b.get("location"))

    different_time = (
        bool(time_a)
        and bool(time_b)
        and time_a != time_b
    )

    different_scope = (
        bool(scope_a)
        and bool(scope_b)
        and scope_a != scope_b
    )

    different_location = (
        bool(loc_a)
        and bool(loc_b)
        and loc_a != loc_b
    )

    # -----------------------------------------------------------------------
    # Step 4: Compare values
    # -----------------------------------------------------------------------

    num_a = _try_parse_number(fact_a.get("value"))
    num_b = _try_parse_number(fact_b.get("value"))

    if num_a is not None and num_b is not None:

        larger = max(
            abs(num_a),
            abs(num_b),
            1e-9
        )

        values_agree = (
            abs(num_a - num_b) / larger
        ) <= NUMERIC_AGREEMENT_TOLERANCE

    else:
        values_agree = (
            _clean(fact_a.get("value"))
            == _clean(fact_b.get("value"))
        )

    # -----------------------------------------------------------------------
    # Step 5: Different time periods
    # -----------------------------------------------------------------------
    #
    # Different years normally indicate temporal evolution rather than
    # contradiction.
    #
    # Example:
    #   FY23 = 229.1
    #   FY24 = 184
    #
    # This is an evolution, not a contradiction.
    # -----------------------------------------------------------------------

    if different_time:

        if values_agree:
            return {
                "relationship": RELATIONSHIP_AGREEMENT,
                "similarity": sim,
                "explanation": (
                    f"The same metric is reported for different time periods "
                    f"('{fact_a.get('time_period')}' vs "
                    f"'{fact_b.get('time_period')}') but the values are "
                    "effectively the same, so this is agreement across time."
                ),
            }

        return {
            "relationship": RELATIONSHIP_EVOLUTION,
            "similarity": sim,
            "explanation": (
                f"The same metric is reported for different time periods "
                f"('{fact_a.get('time_period')}' vs "
                f"'{fact_b.get('time_period')}'), so the difference "
                "represents a temporal change rather than a contradiction."
            ),
        }

    # -----------------------------------------------------------------------
    # Step 6: Different scope or location
    # -----------------------------------------------------------------------

    if different_scope or different_location:

        if values_agree:
            return {
                "relationship": RELATIONSHIP_AGREEMENT,
                "similarity": sim,
                "explanation": (
                    "The facts have different scope or location but agree "
                    "in value, so they support the same conclusion."
                ),
            }

        return {
            "relationship": RELATIONSHIP_CONTEXTUAL,
            "similarity": sim,
            "explanation": (
                "The facts differ in scope or location "
                f"('{fact_a.get('scope') or fact_a.get('location')}' vs "
                f"'{fact_b.get('scope') or fact_b.get('location')}'), "
                "so the value difference is contextual rather than "
                "a direct contradiction."
            ),
        }

    # -----------------------------------------------------------------------
    # Step 7: Same context
    # -----------------------------------------------------------------------

    if values_agree:
        return {
            "relationship": RELATIONSHIP_AGREEMENT,
            "similarity": sim,
            "explanation": (
                "The facts describe the same metric, time period and "
                "context, and the values agree."
            ),
        }

    # -----------------------------------------------------------------------
    # Step 8: Genuine contradiction
    # -----------------------------------------------------------------------

    return {
        "relationship": RELATIONSHIP_CONTRADICTION,
        "similarity": sim,
        "explanation": (
            "The facts describe the same metric, time period and context, "
            "but report different values "
            f"('{fact_a.get('value')} {fact_a.get('unit') or ''}' vs "
            f"'{fact_b.get('value')} {fact_b.get('unit') or ''}'), "
            "which is a genuine contradiction."
        ).strip(),
    }