"""
Streamlit UI for the Evidence-Grounded Fact Knowledge Layer.

IMPORTANT: this file only calls functions from app.pipeline. It never
parses PDFs, calls Ollama, or touches the database directly — all of
that logic lives in the pipeline and the modules it calls.
"""

import sys
from pathlib import Path

# Make sure the project root is on sys.path so "from app...." imports
# work no matter what directory `streamlit run` is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.config import UPLOAD_DIRECTORY
from app.pipeline import build_relationships, process_pdf
from app.storage import database as db

st.set_page_config(page_title="Evidence-Grounded Fact Knowledge Layer", layout="wide")
db.init_db()


def _save_uploaded_file(uploaded_file) -> str:
    destination = Path(UPLOAD_DIRECTORY) / uploaded_file.name
    with open(destination, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(destination)


def _process_uploaded_files(uploaded_files) -> None:
    """Run each uploaded PDF through the pipeline, showing live progress."""
    for uploaded_file in uploaded_files:
        st.markdown(f"### Processing `{uploaded_file.name}`")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(current, total, message, _bar=progress_bar, _text=status_text):
            fraction = current / total if total else 1.0
            _bar.progress(min(max(fraction, 0.0), 1.0))
            _text.write(message)

        pdf_path = _save_uploaded_file(uploaded_file)
        summary = process_pdf(pdf_path, progress_callback=progress_callback)

        if summary.get("duplicate"):
            st.info(f"`{uploaded_file.name}` was already processed before — skipped as a duplicate.")
            continue

        st.success(
            f"Done: {summary['fact_count']} fact(s) extracted from {summary['pages']} page(s); "
            f"{summary['skipped_pages']} page(s) skipped; {len(summary['failures'])} failure(s)."
        )
        if summary["failures"]:
            with st.expander("View extraction failures for this document"):
                st.dataframe(pd.DataFrame(summary["failures"]))

    st.markdown("### Building cross-document relationships")
    rel_bar = st.progress(0)
    rel_text = st.empty()

    def rel_progress_callback(current, total, message):
        fraction = current / total if total else 1.0
        rel_bar.progress(min(max(fraction, 0.0), 1.0))
        rel_text.write(message)

    rel_summary = build_relationships(progress_callback=rel_progress_callback)
    st.success(
        f"Relationship building complete: {rel_summary['relationships_created']} relationship(s) "
        f"created from {rel_summary['candidates']} candidate pair(s)."
    )


def tab_upload() -> None:
    st.header("Upload Documents")
    st.write(
        "Upload one or more PDF documents. Each PDF is parsed page by page, facts are "
        "extracted with evidence, and duplicate files are automatically skipped."
    )
    uploaded_files = st.file_uploader("Choose PDF file(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Process Documents", type="primary"):
        _process_uploaded_files(uploaded_files)


def tab_fact_explorer() -> None:
    st.header("Fact Explorer")
    facts = db.get_facts()
    if not facts:
        st.info("No facts yet — upload and process a document first.")
        return

    df = pd.DataFrame(facts)

    col1, col2, col3 = st.columns(3)
    with col1:
        document_filter = st.selectbox("Document", ["All"] + sorted(df["document_name"].unique().tolist()))
    with col2:
        subject_filter = st.text_input("Subject contains")
    with col3:
        predicate_filter = st.text_input("Predicate contains")

    col4, col5 = st.columns(2)
    with col4:
        time_options = sorted([t for t in df["time_period"].unique() if t])
        time_filter = st.selectbox("Time period", ["All"] + time_options)
    with col5:
        status_options = sorted(df["status"].dropna().unique().tolist()) if "status" in df.columns else []
        status_filter = st.multiselect("Status", status_options, default=status_options)

    filtered = df.copy()
    if document_filter != "All":
        filtered = filtered[filtered["document_name"] == document_filter]
    if subject_filter:
        filtered = filtered[filtered["subject"].str.contains(subject_filter, case=False, na=False)]
    if predicate_filter:
        filtered = filtered[filtered["predicate"].str.contains(predicate_filter, case=False, na=False)]
    if time_filter != "All":
        filtered = filtered[filtered["time_period"] == time_filter]
    if status_filter and "status" in filtered.columns:
        filtered = filtered[filtered["status"].isin(status_filter)]

    display_columns = [
        "subject", "predicate", "value", "unit", "time_period", "scope",
        "location", "confidence", "llm_confidence", "status",
        "document_name", "page", "evidence",
    ]
    display_columns = [c for c in display_columns if c in filtered.columns]
    st.write(f"Showing {len(filtered)} of {len(df)} facts")
    st.dataframe(filtered[display_columns].rename(columns={"document_name": "document"}), use_container_width=True)

def tab_comparisons() -> None:
    st.header("Comparisons")
    relationships = db.get_relationships()
    if not relationships:
        st.info("No relationships yet — process at least two documents with related facts.")
        return

    relationship_types = sorted({r["relationship"] for r in relationships})
    selected_types = st.multiselect("Relationship type", relationship_types, default=relationship_types)

    badge_colors = {
        "AGREEMENT": "green", "GENUINE_CONTRADICTION": "red", "CONTEXTUAL_DIFFERENCE": "orange",
        "EVOLUTION": "blue", "UNRELATED": "gray", "EXTRACTION_FAILURE": "violet",
    }

    for rel in relationships:
        if rel["relationship"] not in selected_types:
            continue

        color = badge_colors.get(rel["relationship"], "gray")
        with st.container(border=True):
            st.markdown(f":{color}[**{rel['relationship']}**]  ·  similarity: {rel['similarity']:.2f}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Fact A** — {rel['a_document']}, page {rel['a_page']}")
                st.write(f"{rel['a_subject']} — {rel['a_predicate']}: **{rel['a_value']} {rel['a_unit']}**")
                st.caption(f"Time: {rel['a_time_period']} · Scope: {rel['a_scope']} · Location: {rel['a_location']}")
                st.markdown(f"> {rel['a_evidence']}")
            with col_b:
                st.markdown(f"**Fact B** — {rel['b_document']}, page {rel['b_page']}")
                st.write(f"{rel['b_subject']} — {rel['b_predicate']}: **{rel['b_value']} {rel['b_unit']}**")
                st.caption(f"Time: {rel['b_time_period']} · Scope: {rel['b_scope']} · Location: {rel['b_location']}")
                st.markdown(f"> {rel['b_evidence']}")

            st.markdown(f"**Explanation:** {rel['explanation']}")


def tab_fact_evolution() -> None:
    st.header("Fact Evolution")
    facts = db.get_facts()
    if not facts:
        st.info("No facts yet — upload and process a document first.")
        return

    df = pd.DataFrame(facts)
    df["group_key"] = df["subject"].fillna("").str.lower() + " | " + df["predicate"].fillna("").str.lower()

    for group_key, group_df in df.groupby("group_key"):
        if group_df["time_period"].nunique() < 2:
            continue  # only show subject/predicate pairs with multiple time periods

        subject = group_df.iloc[0]["subject"]
        predicate = group_df.iloc[0]["predicate"]
        st.subheader(f"{subject} — {predicate}")

        ordered = group_df.sort_values("time_period")
        display_cols = ["time_period", "value", "unit", "document_name", "page", "evidence"]
        st.dataframe(ordered[display_cols].rename(columns={"document_name": "document"}), use_container_width=True)


def tab_documents() -> None:
    st.header("Documents")
    documents = db.get_documents()
    if not documents:
        st.info("No documents processed yet.")
        return

    for doc in documents:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.write(
                f"**{doc['name']}** — {doc['total_pages']} pages, "
                f"{doc['fact_count']} facts, {doc['failure_count']} failures, "
                f"status: {doc['status']}, uploaded: {doc['created_at']}"
            )
        with col2:
            confirm_key = f"confirm_delete_{doc['id']}"
            if st.session_state.get(confirm_key):
                if st.button("Confirm delete", key=f"do_delete_{doc['id']}", type="primary"):
                    db.delete_document(doc["id"])
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
                if st.button("Cancel", key=f"cancel_delete_{doc['id']}"):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()
            else:
                if st.button("Delete", key=f"delete_{doc['id']}"):
                    st.session_state[confirm_key] = True
                    st.rerun()


def tab_extraction_failures() -> None:
    st.header("Extraction Failures")
    st.caption(
        "Pages that were attempted but failed during extraction. Pages skipped for having "
        "too little text are NOT shown here — see the processing summary shown after upload for those."
    )
    failures = db.get_failures()
    if not failures:
        st.success("No extraction failures recorded.")
        return

    df = pd.DataFrame(failures).rename(columns={"document_name": "Document", "page": "Page", "reason": "Failure Reason"})
    st.dataframe(df[["Document", "Page", "Failure Reason"]], use_container_width=True)


def main() -> None:
    st.title("Evidence-Grounded Fact Knowledge Layer")
    st.caption(
        "Upload business PDFs to extract evidence-grounded facts, then explore how those "
        "facts agree, contradict, or evolve across documents."
    )

    tabs = st.tabs(
        ["Upload Documents", "Fact Explorer", "Comparisons", "Fact Evolution", "Documents", "Extraction Failures"]
    )
    with tabs[0]:
        tab_upload()
    with tabs[1]:
        tab_fact_explorer()
    with tabs[2]:
        tab_comparisons()
    with tabs[3]:
        tab_fact_evolution()
    with tabs[4]:
        tab_documents()
    with tabs[5]:
        tab_extraction_failures()


if __name__ == "__main__":
    main()