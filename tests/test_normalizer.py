from app.normalization.normalizer import normalize_fact


def test_normalize_whitespace_and_confidence():
    fact = {
        "subject": "  Delhivery  ", "predicate": "logistics intensity", "value": "184",
        "unit": "gCO2e/tonne-km", "time_period": "FY24", "scope": "", "location": "",
        "confidence": "1.5",  # out of range, should be clamped to 1.0
        "document": "report.pdf", "page": 57,
        "evidence": "Logistics intensity was 184 gCO2e/tonne-km in FY24.",
    }
    result = normalize_fact(fact)
    assert result["subject"] == "Delhivery"
    assert result["confidence"] == 1.0
    assert result["evidence"] == fact["evidence"]


def test_normalize_percentage_spacing():
    fact = {
        "subject": "Delhivery", "predicate": "market share", "value": "12 %",
        "unit": "", "time_period": "", "scope": "", "location": "",
        "confidence": 0.9, "document": "d.pdf", "page": 1, "evidence": "12 % market share",
    }
    assert normalize_fact(fact)["value"] == "12%"


def test_normalize_fiscal_year():
    fact = {
        "subject": "Delhivery", "predicate": "tractors", "value": "753",
        "unit": "", "time_period": "FY 2023-24", "scope": "", "location": "",
        "confidence": 0.9, "document": "d.pdf", "page": 3, "evidence": "753 tractors in FY 2023-24",
    }
    assert normalize_fact(fact)["time_period"] == "FY24"


def test_normalize_preserves_evidence_document_page():
    fact = {
        "subject": "Delhivery", "predicate": "operates in", "value": "India",
        "unit": "", "time_period": "", "scope": "", "location": "India",
        "confidence": 0.8, "document": "prospectus.pdf", "page": 22,
        "evidence": "Delhivery operates across India.",
    }
    result = normalize_fact(fact)
    assert result["document"] == "prospectus.pdf"
    assert result["page"] == 22
    assert result["evidence"] == "Delhivery operates across India."


def test_period_alias_supported():
    fact = {
        "subject": "Delhivery", "predicate": "tractors", "value": "562",
        "unit": "", "period": "FY23", "scope": "", "location": "",
        "confidence": 0.9, "document": "d.pdf", "page": 3, "evidence": "562 tractors",
    }
    assert normalize_fact(fact)["time_period"] == "FY23"