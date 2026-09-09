"""
The single processing pipeline for the whole project.

Every entry point — Streamlit, a script, a test — goes through
process_pdf() and build_relationships() defined here. Nothing else in
the project should independently loop over PDF pages or call the LLM.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.extraction.fact_extractor import ExtractionError, extract_facts_from_page, is_page_worth_processing
from app.ingestion.pdf_parser import PDFParsingError, extract_pdf_pages
from app.matching.embedder import embed_facts
from app.matching.matcher import find_candidate_pairs
from app.normalization.normalizer import normalize_fact
from app.reasoning.comparator import compare_facts
from app.storage import database as db

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


def _report(progress_callback: Optional[ProgressCallback], current: int, total: int, message: str) -> None:
    if progress_callback:
        try:
            progress_callback(current, total, message)
        except Exception:
            # A broken UI callback should never take down the pipeline.
            logger.exception("progress_callback raised an exception; ignoring.")


def process_pdf(pdf_path: str, progress_callback: Optional[ProgressCallback] = None) -> Dict:
    """
    Process a single PDF end to end: parse pages, extract facts,
    normalize them, and store everything in the database.

    Returns:
        {
            "document": name, "pages": total_pages,
            "facts": [normalized fact dicts, each with an "id"],
            "fact_count": int, "skipped_pages": int,
            "failures": [{"page": int, "reason": str}, ...],
            "duplicate": bool,
        }
    """
    db.init_db()
    pdf_path = str(pdf_path)
    document_name = Path(pdf_path).name

    file_hash = db.compute_file_hash(pdf_path)
    existing = db.get_document_by_hash(file_hash)
    if existing:
        logger.info("Skipping '%s': identical file already processed (id=%s).", document_name, existing["id"])
        _report(progress_callback, 1, 1, f"'{document_name}' was already processed — skipping duplicate.")
        return {
            "document": document_name, "pages": existing.get("total_pages", 0),
            "facts": [], "fact_count": 0, "skipped_pages": 0, "failures": [], "duplicate": True,
        }

    try:
        pages = extract_pdf_pages(pdf_path)
    except PDFParsingError as exc:
        logger.error("Failed to parse PDF '%s': %s", document_name, exc)
        _report(progress_callback, 1, 1, f"Failed to open PDF: {exc}")
        return {
            "document": document_name, "pages": 0, "facts": [], "fact_count": 0,
            "skipped_pages": 0, "failures": [{"page": None, "reason": str(exc)}], "duplicate": False,
        }

    total_pages = len(pages)
    document_id = db.insert_document(document_name, pdf_path, file_hash, total_pages)

    stored_facts: List[Dict] = []
    failures: List[Dict] = []
    skipped_pages = 0

    for index, page in enumerate(pages, start=1):
        page_number = page["page"]
        text = page["text"]

        if not is_page_worth_processing(text):
            skipped_pages += 1
            _report(progress_callback, index, total_pages, f"Page {page_number}/{total_pages}: skipped (too little text)")
            continue

        _report(progress_callback, index, total_pages, f"Page {page_number}/{total_pages}: extracting facts...")

        try:
            raw_facts = extract_facts_from_page(text, document_name, page_number)
        except ExtractionError as exc:
            logger.warning("Extraction failed on page %s of '%s': %s", page_number, document_name, exc)
            failures.append({"page": page_number, "reason": str(exc)})
            db.insert_failure(document_id, page_number, str(exc))
            continue

        for raw_fact in raw_facts:
            try:
                normalized = normalize_fact(raw_fact)
            except Exception as exc:  # normalization must never crash the pipeline
                logger.warning("Normalization failed on page %s: %s", page_number, exc)
                failures.append({"page": page_number, "reason": f"Normalization error: {exc}"})
                db.insert_failure(document_id, page_number, f"Normalization error: {exc}")
                continue

            try:
                fact_id = db.insert_fact(document_id, normalized)
            except Exception as exc:  # database errors must not crash the pipeline
                logger.error("Database insert failed on page %s: %s", page_number, exc)
                failures.append({"page": page_number, "reason": f"Database error: {exc}"})
                db.insert_failure(document_id, page_number, f"Database error: {exc}")
                continue

            normalized["id"] = fact_id
            normalized["document_id"] = document_id
            stored_facts.append(normalized)

        _report(progress_callback, index, total_pages, f"Page {page_number}/{total_pages}: {len(raw_facts)} fact(s) found")

    db.update_document_status(document_id, "completed")

    return {
        "document": document_name, "pages": total_pages, "facts": stored_facts,
        "fact_count": len(stored_facts), "skipped_pages": skipped_pages,
        "failures": failures, "duplicate": False,
    }


def build_relationships(progress_callback: Optional[ProgressCallback] = None) -> Dict:
    """
    Build cross-document relationships between all facts currently in
    the database. Kept separate from process_pdf() on purpose — running
    expensive matching after every single page would be wasteful, so
    this runs once, after documents have been ingested.

    Returns: {"candidates": int, "relationships_created": int}
    """
    db.init_db()
    all_facts = db.get_facts()

    if len(all_facts) < 2:
        _report(progress_callback, 1, 1, "Not enough facts to build relationships yet.")
        return {"candidates": 0, "relationships_created": 0}

    _report(progress_callback, 0, 3, "Generating embeddings for all facts...")
    try:
        embeddings = embed_facts(all_facts)
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        _report(progress_callback, 3, 3, f"Embedding failed: {exc}")
        return {"candidates": 0, "relationships_created": 0}

    _report(progress_callback, 1, 3, "Finding candidate related facts...")
    candidates = find_candidate_pairs(all_facts, embeddings)

    _report(progress_callback, 2, 3, f"Comparing {len(candidates)} candidate pair(s)...")
    created = 0
    for pair in candidates:
        fact_a, fact_b, similarity = pair["fact_a"], pair["fact_b"], pair["similarity"]

        if db.relationship_exists(fact_a["id"], fact_b["id"]):
            continue

        try:
            result = compare_facts(fact_a, fact_b, similarity)
        except Exception as exc:
            logger.warning("Comparison failed for facts %s/%s: %s", fact_a.get("id"), fact_b.get("id"), exc)
            result = {
                "relationship": "EXTRACTION_FAILURE",
                "similarity": similarity,
                "explanation": f"Comparison could not be completed: {exc}",
            }

        db.insert_relationship(fact_a["id"], fact_b["id"], result["relationship"], result["similarity"], result["explanation"])
        created += 1

    _report(progress_callback, 3, 3, f"Created {created} relationship(s).")
    return {"candidates": len(candidates), "relationships_created": created}