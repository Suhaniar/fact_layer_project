# Fact Knowledge Layer

Extracts evidence-grounded facts from financial/business PDFs, checks each fact against its own source text before trusting it, and reasons about how facts relate across documents — agreement, contradiction, or a difference explained by context (time period, units, scope).

Built for the Superjoin VIT 2026 Engineering Intern assignment, using Delhivery's FY24 Annual Report and Q4 FY24 earnings presentation as the starter dataset.

---

## Setup and Run Instructions

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed locally.

```bash
git clone <your-repo-url>
cd fact_knowledge_layer_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

**Start Ollama and pull the model:**
```bash
ollama serve
ollama pull llama3.2:3b
```

**Run the app** (from the project root):
```bash
streamlit run app/ui/streamlit_app.py
```

Open the local URL Streamlit prints, upload one or more PDFs on the **Upload Documents** tab, and let it process. Facts, comparisons, and evolution across time periods appear in the other tabs once processing finishes.

No API keys or paid services are required — everything runs locally against Ollama.

---

## Video Demo

[Link to demo video — 3 minutes or less]

Shows a PDF being processed end-to-end and all four required cases: corroboration, contradiction, contextual difference, and an extraction failure.

---

## Approach

**Pipeline:** PDF → PyMuPDF text extraction → local LLM (Llama 3.2 3B via Ollama) fact extraction → normalization → evidence validation → SQLite storage → sentence-transformer embeddings → rule-based reasoning → Streamlit UI.

**Key decisions:**

- **Every fact carries its own evidence.** The LLM prompt requires a verbatim evidence snippet for each fact, and nothing is displayed without it.
- **Confidence is not trusted blindly.** The LLM's self-reported confidence is stored separately (`llm_confidence`, which can be `None` if missing/invalid — never silently coerced to 0). A second, independently computed `confidence` score factors in whether real evidence text was actually provided, so a confident-sounding but evidence-free claim gets penalized rather than taken at face value.
- **Facts are validated against their own evidence before being trusted.** Before a fact reaches the reasoning engine, I check whether its subject, predicate, value, unit, and time period are actually supported by the quoted evidence text. Facts are classified `VALIDATED`, `REVIEW_REQUIRED`, or `REJECTED`. Only `VALIDATED` facts are compared across documents; the rest are kept in storage for audit rather than silently dropped.
- **Reasoning doesn't treat every numeric mismatch as a contradiction.** The comparator checks subject, predicate, unit, time period, scope, and location before deciding how two related facts relate: `AGREEMENT` (same context, same value), `GENUINE_CONTRADICTION` (same context, different value), `CONTEXTUAL_DIFFERENCE` (different unit/scope), `EVOLUTION` (different time period), or `UNRELATED` (different subject/metric entirely). In practice, on the current sample documents, the Comparisons tab reliably surfaces `AGREEMENT` and `UNRELATED` pairs; the other categories are implemented in the comparator logic but the embedding similarity threshold isn't yet tuned to surface them automatically for this document set — see Limitations below.
- **One pipeline, one entry point.** Streamlit never touches the database, Ollama, or PyMuPDF directly — it only calls `process_pdf()` and `build_relationships()`. This kept the system testable and made it easy to iterate on the extraction/validation logic without touching the UI.

**AI tools used:** Claude (Anthropic) for iterative development of the extraction prompt, validation logic, and reasoning rules, working incrementally from an initial working prototype rather than a single generated codebase. Llama 3.2 3B (via Ollama) is the extraction model used at runtime.

---

## Limitations and Next Steps

**What doesn't work yet:**

- **Comparisons currently only reliably surface `AGREEMENT` and `UNRELATED`.** `GENUINE_CONTRADICTION`, `CONTEXTUAL_DIFFERENCE`, and `EVOLUTION` exist in the comparator's decision logic and are covered by unit tests, but the embedding similarity threshold used to find candidate pairs isn't yet tuned well enough to reliably surface those relationships automatically for arbitrary document sets — the contradiction and contextual-difference examples in the demo video were found by inspecting the fact table directly, not the live Comparisons view.
- **Table extraction is still primarily plain-text based.** PyMuPDF's plain text extraction can mix up rows/columns in dense tables, which has caused at least one observed error: a revenue figure associated with the wrong year's column in a multi-year table.
- **Evidence validation is surface-level, not semantic.** It checks that a claimed value's digits (or text) appear somewhere in the evidence string, but doesn't verify counts against enumerated lists — e.g. "Number of Directors = 7" passed validation even though the evidence text lists eight named directors, because "7" does appear as a substring elsewhere. A dedicated counting/verification step is the next fix.
- **Page text is still truncated at a fixed character limit** before being sent to the LLM, which can drop information on dense pages rather than chunking intelligently or extracting tables separately first.
- **Unit conversion is intentionally conservative** (no automatic million/billion/crore/lakh conversion yet), so genuinely comparable values reported in different units are flagged as "not directly comparable" rather than converted and compared — safer, but less complete.

**Next steps, in priority order:**
1. Tune the embedding similarity threshold and predicate-matching logic so `GENUINE_CONTRADICTION`, `CONTEXTUAL_DIFFERENCE`, and `EVOLUTION` surface automatically in Comparisons, not just agreement/unrelated.
2. A lightweight verification step for numeric claims that should match an enumerated list in evidence (director counts, committee members, etc.).
3. Better table-aware extraction using PyMuPDF's structured text/table detection instead of flattened plain text.
4. A conservative unit-conversion table for common financial unit pairs (Cr ↔ million ↔ billion, lakh, crore) so genuine cross-unit contradictions can be surfaced instead of silently skipped.

---

## Additional Notes

- Every fact — including rejected ones — is retained in the database for audit, per the acceptance policy (`VALIDATED` / `REVIEW_REQUIRED` / `REJECTED`), rather than silently discarded.
- Documents can be deleted individually from the Documents tab, which also removes their facts and any relationships those facts were part of.
- The system was built and iterated incrementally, one fix at a time, against real extraction errors found in the sample PDFs, rather than designed upfront and never revisited.
- No document-specific rules, filenames, or hardcoded facts are used — extraction, normalization, validation, and reasoning are all generic and should apply to any similar business PDF.
