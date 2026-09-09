# FactLens — Evidence-Grounded Fact Knowledge Layer

A prototype for the Superjoin VIT 2026 Engineering Intern assignment.

## What it does

- Accepts arbitrary PDFs through a Streamlit UI.
- Extracts page-preserving text with PyMuPDF.
- Extracts atomic numerical/semantic facts using a local Ollama LLM.
- Stores exact source evidence and page numbers.
- Normalizes values, units, time, scope and location.
- Uses embeddings to find candidate cross-document matches.
- Classifies relationships as:
  - CORROBORATED
  - CONTRADICTION
  - CONTEXTUAL_DIFFERENCE
  - UNCERTAIN
- Explains why a relationship was assigned.
- Keeps an explicit failure log.
- Stores knowledge in SQLite so later PDFs can be added incrementally.

## Setup

### 1. Python

Use Python 3.11+.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Ollama

Install Ollama and pull a model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running.

You can change the model in `.env`.

### 3. Environment

Copy `.env.example` to `.env`.

### 4. Run

```bash
streamlit run app/ui/streamlit_app.py
```

Then upload PDFs in the browser.

## Architecture

PDF -> page-preserving extraction -> fact extraction -> normalization -> semantic candidate matching -> deterministic comparison / LLM reasoning -> evidence-backed UI.

## Important design choice

The LLM is not treated as the database of truth. It extracts claims from source text, but each fact retains exact evidence and page information. Candidate matching uses embeddings, while obvious numeric/context comparisons are handled deterministically before invoking the LLM for ambiguous cases.

## Limitations

- Scanned/image-only PDFs require OCR; this prototype currently relies on extractable PDF text.
- Tables and complex layouts can be imperfectly extracted.
- Local LLM output can occasionally be malformed or ambiguous.
- Semantic similarity can produce false candidate matches.
- Relationship judgments should be treated as evidence-backed assessments, not absolute truth.

## Demo checklist

Show:
1. A corroborated fact with evidence from both documents.
2. A genuine/likely contradiction with evidence and reasoning.
3. An apparent contradiction explained by time/scope/units.
4. One extraction or reasoning failure and how the system flags it.

Also show the Fact Explorer and, if time permits, upload a new PDF after the initial set to demonstrate incremental knowledge.

## Credentials

Do not commit `.env`. Use `.env.example` as the template.
