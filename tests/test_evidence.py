from app.extraction.fact_extractor import is_page_worth_processing
from app.normalization.normalizer import normalize_fact


def test_empty_page_is_skipped():
    assert is_page_worth_processing("") is False
    assert is_page_worth_processing("   ") is False


def test_tiny_page_is_skipped():
    assert is_page_worth_processing("Page 4") is False


def test_substantial_page_is_processed():
    text = (
        "Delhivery reported a logistics intensity of 184 gCO2e/tonne-km in FY24, "
        "down from 229.1 gCO2e/tonne-km in FY23, reflecting efficiency improvements."
    )
    assert is_page_worth_processing(text) is True


def test_normalization_never_invents_evidence():
    fact = {
        "subject": "Delhivery", "predicate": "tractors", "value": "753",
        "unit": "", "time_period": "FY24", "scope": "", "location": "",
        "confidence": 0.9, "document": "annual_report.pdf", "page": 88,
        "evidence": "The fleet included 753 tractors as of FY24.",
    }
    normalized = normalize_fact(fact)
    assert normalized["evidence"] == fact["evidence"]
    assert normalized["document"] == fact["document"]
    assert normalized["page"] == fact["page"]


def test_normalization_with_missing_optional_fields_still_preserves_core():
    fact = {
        "subject": "Delhivery", "predicate": "operates in", "value": "India",
        "document": "prospectus.pdf", "page": 22,
        "evidence": "Delhivery operates across India.",
    }
    normalized = normalize_fact(fact)
    assert normalized["evidence"] == "Delhivery operates across India."
    assert normalized["unit"] == ""
    assert normalized["scope"] == ""