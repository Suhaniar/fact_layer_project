"""
PDF ingestion: turns a PDF file into a list of per-page text blocks,
preserving the real PDF page numbers so every extracted fact can be
traced back to an exact page.
"""

import logging
from typing import Dict, List

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class PDFParsingError(Exception):
    """Raised when a PDF cannot be opened or read at all."""


def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    """
    Extract text from every page of a PDF.

    Returns a list like:
        [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]

    Page numbers are 1-indexed and match real PDF page numbers, even
    for pages that have no extractable text (their "text" is "").
    """
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        # Covers malformed PDFs, unreadable files, wrong file type, etc.
        raise PDFParsingError(f"Could not open PDF '{pdf_path}': {exc}") from exc

    if document.page_count == 0:
        document.close()
        raise PDFParsingError(f"PDF '{pdf_path}' has no pages.")

    pages: List[Dict] = []
    for page_index in range(document.page_count):
        page_number = page_index + 1  # PyMuPDF is 0-indexed; PDFs are 1-indexed
        try:
            page = document.load_page(page_index)
            text = page.get_text("text") or ""
        except Exception as exc:
            # A single unreadable page (e.g. corrupted content stream)
            # must not stop us from reading the rest of the PDF.
            logger.warning("Failed to read page %s of '%s': %s", page_number, pdf_path, exc)
            text = ""
        pages.append({"page": page_number, "text": text.strip()})

    document.close()
    return pages