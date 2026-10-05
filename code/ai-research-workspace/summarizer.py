"""Structured paper summarization and keyword/concept extraction."""

from __future__ import annotations

from document_processor import ProcessedDocument, build_combined_context
import config
from prompts import CONCEPT_EXTRACTION_PROMPT, SUMMARY_PROMPT
from rag import run_chain


def summarize_document(doc: ProcessedDocument) -> str:
    """Return a structured markdown summary of one paper.

    Sections: Problem, Objective, Methodology, Dataset, Results,
    Contributions, Limitations and Future Work.
    """
    paper_text = doc.get_text(config.MAX_DOC_CHARS)
    summary = run_chain(
        SUMMARY_PROMPT,
        {"paper_name": doc.filename, "paper_text": paper_text},
    )
    return f"# Summary: {doc.filename}\n\n{summary}"


def extract_concepts(docs: list[ProcessedDocument]) -> str:
    """Extract keywords, algorithms, datasets, metrics and concepts from papers."""
    if not docs:
        raise ValueError("Select at least one paper to analyze.")
    papers_text = build_combined_context(docs, config.MAX_COMBINED_CHARS)
    return run_chain(CONCEPT_EXTRACTION_PROMPT, {"papers_text": papers_text})