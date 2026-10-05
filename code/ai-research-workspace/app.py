"""Streamlit entry point for the AI Research Workspace."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import streamlit as st
from openai import (
    APIConnectionError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)

import config
from comparator import compare_papers, detect_research_gaps
from document_processor import DocumentProcessingError, ProcessedDocument, process_pdf
from rag import MissingAPIKeyError, RAGPipeline
from summarizer import extract_concepts, summarize_document
from utils import chat_history_to_markdown, format_file_size

st.set_page_config(page_title=config.APP_TITLE, page_icon="🔬", layout="wide")

PAGE_DOCUMENTS = "📄 Documents"
PAGE_CHAT = "💬 Research Chat"
PAGE_SUMMARIZE = "📝 Summarize"
PAGE_COMPARE = "🔍 Compare"
PAGE_GAPS = "🧠 Research Gaps"
PAGE_CONCEPTS = "🔑 Key Concepts"

PAGES = [
    PAGE_DOCUMENTS,
    PAGE_CHAT,
    PAGE_SUMMARIZE,
    PAGE_COMPARE,
    PAGE_GAPS,
    PAGE_CONCEPTS,
]


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    """Create all session-state keys used by the app (only once per session)."""
    defaults = {
        "documents": {},          # filename -> ProcessedDocument
        "pipeline": None,         # RAGPipeline (FAISS index)
        "indexed_signature": None,  # identifies which documents are indexed
        "chat_history": [],       # list of {"role", "content", "sources"}
        "summaries": {},          # filename -> markdown summary
        "comparison": None,       # {"a", "b", "text"}
        "gaps": None,             # {"names", "text"}
        "concepts": None,         # {"names", "text"}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_documents() -> dict[str, ProcessedDocument]:
    return st.session_state.documents


def total_pages() -> int:
    return sum(doc.total_pages for doc in get_documents().values())


def total_chunks() -> int:
    return sum(doc.num_chunks for doc in get_documents().values())


def documents_signature() -> tuple:
    return tuple(sorted((name, doc.num_chunks) for name, doc in get_documents().items()))


def reset_derived_results() -> None:
    """Clear cached analysis results that depend on the set of documents."""
    st.session_state.comparison = None
    st.session_state.gaps = None
    st.session_state.concepts = None


def ensure_index() -> RAGPipeline:
    """Return an up-to-date FAISS pipeline, (re)building it when needed."""
    signature = documents_signature()
    pipeline = st.session_state.pipeline
    if (
        pipeline is not None
        and pipeline.is_ready
        and st.session_state.indexed_signature == signature
    ):
        return pipeline

    chunks = [chunk for doc in get_documents().values() for chunk in doc.chunks]
    pipeline = RAGPipeline()
    with st.spinner("Creating embeddings and building the FAISS index..."):
        pipeline.build_index(chunks)
    st.session_state.pipeline = pipeline
    st.session_state.indexed_signature = signature
    return pipeline


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def show_error(exc: Exception) -> None:
    """Show a friendly, human-readable error message."""
    if isinstance(exc, MissingAPIKeyError):
        st.error(f"🔑 {exc}")
    elif isinstance(exc, AuthenticationError):
        st.error(
            "🔑 OpenAI rejected the API key. Check `OPENAI_API_KEY` in your "
            "`.env` file, save it, and try again."
        )
    elif isinstance(exc, RateLimitError):
        st.error(
            "⏳ OpenAI rate limit or quota reached. Check your usage/billing "
            "on the OpenAI dashboard, wait a moment, and try again."
        )
    elif isinstance(exc, APIConnectionError):
        st.error("🌐 Could not connect to OpenAI. Check your internet connection and try again.")
    elif isinstance(exc, NotFoundError):
        st.error(
            f"🤖 The model `{config.OPENAI_MODEL}` was not found or your account "
            "has no access to it. Change `OPENAI_MODEL` in your `.env` file."
        )
    else:
        st.error(f"Something went wrong: {exc}")


def require_documents(minimum: int = 1) -> bool:
    """Show an info box and return False when not enough documents are loaded."""
    count = len(get_documents())
    if count >= minimum:
        return True
    if count == 0:
        st.info("📄 No documents yet. Open **📄 Documents** in the sidebar and upload at least one PDF.")
    else:
        st.info(
            f"This feature needs at least {minimum} documents, but only {count} "
            "is uploaded. Add more in **📄 Documents**."
        )
    return False


def render_sources(sources: list[dict]) -> None:
    """Render the sources (filename + page) used for an answer."""
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**[{source['index']}] {source['filename']}** — page {source['page']}")
            st.caption(source["snippet"])


def render_sidebar() -> str:
    st.sidebar.title("🔬 AI Research Workspace")
    st.sidebar.caption("RAG-powered assistant for research papers")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

    st.sidebar.divider()
    col_docs, col_chunks = st.sidebar.columns(2)
    col_docs.metric("Documents", len(get_documents()))
    col_chunks.metric("Chunks", total_chunks())

    st.sidebar.divider()
    if config.is_api_key_configured():
        st.sidebar.success("OpenAI API key detected")
    else:
        st.sidebar.warning("OpenAI API key missing")
    st.sidebar.caption(f"Model: `{config.OPENAI_MODEL}`")
    return page


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def process_uploaded_files(uploaded_files) -> None:
    """Extract, chunk and (if possible) index the uploaded PDFs."""
    documents = get_documents()
    added = 0
    progress = st.progress(0.0)

    for position, uploaded in enumerate(uploaded_files, start=1):
        already_loaded = (
            uploaded.name in documents and documents[uploaded.name].size_bytes == uploaded.size
        )
        if already_loaded:
            st.info(f"'{uploaded.name}' is already processed — skipped.")
        else:
            try:
                with st.spinner(f"Reading {uploaded.name}..."):
                    documents[uploaded.name] = process_pdf(uploaded.name, uploaded.getvalue())
                st.session_state.summaries.pop(uploaded.name, None)
                added += 1
            except DocumentProcessingError as exc:
                st.error(f"❌ {exc}")
        progress.progress(position / len(uploaded_files))
    progress.empty()

    if not added:
        return

    reset_derived_results()
    st.session_state.pipeline = None
    st.session_state.indexed_signature = None
    st.success(f"✅ Processed {added} new document(s).")

    if config.is_api_key_configured():
        try:
            ensure_index()
            st.success("✅ Search index is ready. You can start chatting with your papers.")
        except Exception as exc:  # noqa: BLE001 - show any failure to the user
            show_error(exc)
    else:
        st.warning(
            "Documents were processed, but the search index could not be built "
            "because `OPENAI_API_KEY` is missing. Add it to your `.env` file."
        )


def page_documents() -> None:
    st.header("📄 Documents")
    st.write("Upload one or more research-paper PDFs. Text is extracted page by page, split into chunks and indexed with FAISS.")

    uploaded_files = st.file_uploader(
        "Upload research papers (PDF)", type=["pdf"], accept_multiple_files=True
    )
    if st.button("⚙️ Process documents", type="primary", disabled=not uploaded_files):
        process_uploaded_files(uploaded_files)

    documents = get_documents()
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(documents))
    col2.metric("Pages", total_pages())
    col3.metric("Chunks", total_chunks())

    if documents:
        st.subheader("Loaded documents")
        rows = [
            {
                "Document": doc.filename,
                "Size": format_file_size(doc.size_bytes),
                "Pages": doc.total_pages,
                "Chunks": doc.num_chunks,
                "Characters": f"{doc.num_chars:,}",
            }
            for doc in documents.values()
        ]
        st.dataframe(rows, hide_index=True)

        if st.button("🗑️ Clear all documents"):
            st.session_state.documents = {}
            st.session_state.pipeline = None
            st.session_state.indexed_signature = None
            st.session_state.chat_history = []
            st.session_state.summaries = {}
            reset_derived_results()
            st.rerun()
    else:
        st.info("No documents uploaded yet.")


def page_chat() -> None:
    st.header("💬 Research Chat")
    st.caption("Answers are grounded in your uploaded papers and cite filename + page.")
    if not require_documents():
        return

    names = list(get_documents())
    with st.expander("🎯 Search scope"):
        selected = st.multiselect("Restrict the search to specific papers (leave empty to search all)", names)

    history = st.session_state.chat_history
    if history:
        col_clear, col_export, _ = st.columns([1, 1, 4])
        if col_clear.button("🧹 Clear chat"):
            st.session_state.chat_history = []
            st.rerun()
        col_export.download_button(
            "⬇️ Export chat",
            data=chat_history_to_markdown(history),
            file_name="research_chat.md",
            mime="text/markdown",
        )

    for message in history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))

    question = st.chat_input("Ask a question about your uploaded papers...")
    if not question:
        return

    previous_history = list(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            pipeline = ensure_index()
            with st.spinner("Searching the papers and generating an answer..."):
                result = pipeline.answer(
                    question,
                    chat_history=previous_history,
                    sources=selected or None,
                )
            sources = [dataclasses.asdict(source) for source in result.sources]
            st.markdown(result.answer)
            render_sources(sources)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result.answer, "sources": sources}
            )
        except Exception as exc:  # noqa: BLE001
            show_error(exc)
            st.session_state.chat_history.pop()  # drop the unanswered question


def page_summarize() -> None:
    st.header("📝 Paper Summarization")
    st.caption("Generates a structured summary: problem, objective, methodology, dataset, results, contributions, limitations and future work.")
    if not require_documents():
        return

    documents = get_documents()
    name = st.selectbox("Select a paper", list(documents))
    summaries = st.session_state.summaries
    button_label = "🔄 Regenerate summary" if name in summaries else "📝 Generate summary"

    if st.button(button_label, type="primary"):
        try:
            with st.spinner("Reading the paper and writing the summary..."):
                summaries[name] = summarize_document(documents[name])
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

    if name in summaries:
        st.divider()
        st.markdown(summaries[name])
        st.download_button(
            "⬇️ Download summary (.md)",
            data=summaries[name],
            file_name=f"{Path(name).stem}_summary.md",
            mime="text/markdown",
        )


def page_compare() -> None:
    st.header("🔍 Compare Papers")
    st.caption("Compares methodology, datasets, models, results and limitations of two papers.")
    if not require_documents(minimum=2):
        return

    documents = get_documents()
    names = list(documents)
    col_a, col_b = st.columns(2)
    name_a = col_a.selectbox("Paper A", names, index=0)
    name_b = col_b.selectbox("Paper B", names, index=1)

    if name_a == name_b:
        st.warning("Please select two different papers.")
    elif st.button("🔍 Compare papers", type="primary"):
        try:
            with st.spinner("Comparing the two papers..."):
                text = compare_papers(documents[name_a], documents[name_b])
            st.session_state.comparison = {"a": name_a, "b": name_b, "text": text}
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

    comparison = st.session_state.comparison
    if comparison:
        st.divider()
        st.subheader(f"{comparison['a']}  vs  {comparison['b']}")
        st.markdown(comparison["text"])
        st.download_button(
            "⬇️ Download comparison (.md)",
            data=comparison["text"],
            file_name="paper_comparison.md",
            mime="text/markdown",
        )


def page_gaps() -> None:
    st.header("🧠 Research Gap Detection")
    st.warning(
        "⚠️ The output below consists of **AI-generated observations**. They are "
        "suggestions based on the uploaded papers, not verified facts. Always "
        "check them against the literature before using them."
    )
    if not require_documents():
        return

    names = list(get_documents())
    selected = st.multiselect("Papers to analyze", names, default=names)

    if st.button("🧠 Detect research gaps", type="primary", disabled=not selected):
        try:
            docs = [get_documents()[name] for name in selected]
            with st.spinner("Analyzing limitations and unexplored areas..."):
                text = detect_research_gaps(docs)
            st.session_state.gaps = {"names": selected, "text": text}
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

    gaps = st.session_state.gaps
    if gaps:
        st.divider()
        st.caption("Papers analyzed: " + ", ".join(gaps["names"]))
        st.markdown(gaps["text"])
        st.download_button(
            "⬇️ Download analysis (.md)",
            data=gaps["text"],
            file_name="research_gaps.md",
            mime="text/markdown",
        )


def page_concepts() -> None:
    st.header("🔑 Key Concepts")
    st.caption("Extracts keywords, algorithms, datasets and research concepts.")
    if not require_documents():
        return

    names = list(get_documents())
    selected = st.multiselect("Papers to analyze", names, default=names)

    if st.button("🔑 Extract concepts", type="primary", disabled=not selected):
        try:
            docs = [get_documents()[name] for name in selected]
            with st.spinner("Extracting keywords and concepts..."):
                text = extract_concepts(docs)
            st.session_state.concepts = {"names": selected, "text": text}
        except Exception as exc:  # noqa: BLE001
            show_error(exc)

    concepts = st.session_state.concepts
    if concepts:
        st.divider()
        st.caption("Papers analyzed: " + ", ".join(concepts["names"]))
        st.markdown(concepts["text"])
        st.download_button(
            "⬇️ Download concepts (.md)",
            data=concepts["text"],
            file_name="key_concepts.md",
            mime="text/markdown",
        )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    init_session_state()
    page = render_sidebar()

    st.title("🔬 AI Research Workspace")
    if not config.is_api_key_configured():
        st.warning(
            "🔑 `OPENAI_API_KEY` is not set. You can upload and read PDFs, but "
            "chat, summaries and analysis need an API key. Copy `.env.example` "
            "to `.env`, add your key and reload this page."
        )

    routes = {
        PAGE_DOCUMENTS: page_documents,
        PAGE_CHAT: page_chat,
        PAGE_SUMMARIZE: page_summarize,
        PAGE_COMPARE: page_compare,
        PAGE_GAPS: page_gaps,
        PAGE_CONCEPTS: page_concepts,
    }
    routes[page]()


if __name__ == "__main__":
    main()