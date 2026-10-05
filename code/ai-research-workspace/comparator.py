"""Paper comparison and research-gap detection."""

from __future__ import annotations

import config
from document_processor import ProcessedDocument, build_combined_context
from prompts import COMPARISON_PROMPT, RESEARCH_GAP_PROMPT
from rag import run_chain


def compare_papers(doc_a: ProcessedDocument, doc_b: ProcessedDocument) -> str:
    """Compare two papers on methodology, datasets, models, results and limitations."""
    if doc_a.filename == doc_b.filename:
        raise ValueError("Please select two different papers to compare.")

    return run_chain(
        COMPARISON_PROMPT,
        {
            "paper_a_name": doc_a.filename,
            "paper_a_text": doc_a.get_text(config.MAX_COMPARE_DOC_CHARS),
            "paper_b_name": doc_b.filename,
            "paper_b_text": doc_b.get_text(config.MAX_COMPARE_DOC_CHARS),
        },
    )


def detect_research_gaps(docs: list[ProcessedDocument]) -> str:
    """Identify limitations, unexplored areas and possible research directions.

    The prompt forces the model to label its output as AI-generated observations.
    """
    if not docs:
        raise ValueError("Select at least one paper to analyze.")
    papers_text = build_combined_context(docs, config.MAX_COMBINED_CHARS)
    return run_chain(RESEARCH_GAP_PROMPT, {"papers_text": papers_text})