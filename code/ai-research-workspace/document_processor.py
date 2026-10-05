"""PDF text extraction and chunking with page/document metadata."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config


class DocumentProcessingError(Exception):
    """Raised when a PDF cannot be read; the message is safe to show to users."""


@dataclass
class ProcessedDocument:
    """A processed PDF: page-level documents plus the chunks used for retrieval."""

    filename: str
    size_bytes: int
    total_pages: int
    pages: list[Document] = field(default_factory=list)
    chunks: list[Document] = field(default_factory=list)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_chars(self) -> int:
        return sum(len(page.page_content) for page in self.pages)

    def get_text(self, max_chars: int | None = config.MAX_DOC_CHARS) -> str:
        """Return the full text with [Page N] markers.

        If the paper is longer than `max_chars`, keep the beginning (abstract,
        introduction, methods) and the end (results, conclusion) and drop the
        middle so the prompt stays within the model's context window.
        """
        full_text = "\n\n".join(
            f"[Page {page.metadata['page']}]\n{page.page_content}" for page in self.pages
        )
        if max_chars is None or len(full_text) <= max_chars:
            return full_text

        head_size = int(max_chars * 0.65)
        tail_size = max_chars - head_size
        return (
            full_text[:head_size]
            + "\n\n[... middle part of the paper omitted for length ...]\n\n"
            + full_text[-tail_size:]
        )


def clean_text(text: str) -> str:
    """Normalise raw PDF text so it chunks and reads well."""
    text = text.replace("\x00", "")
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)      # re-join hyphenated words
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)            # single newline -> space
    text = re.sub(r"[ \t]+", " ", text)                     # collapse spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                  # collapse blank lines
    return text.strip()


def _open_reader(filename: str, file_bytes: bytes) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:  # noqa: BLE001 - pypdf raises many exception types
        raise DocumentProcessingError(
            f"'{filename}' could not be opened as a PDF. The file may be corrupted ({exc})."
        ) from exc

    if reader.is_encrypted:
        try:
            decrypted = reader.decrypt("")
        except Exception:  # noqa: BLE001
            decrypted = 0
        if not decrypted:
            raise DocumentProcessingError(
                f"'{filename}' is password-protected. Please upload an unlocked copy."
            )
    return reader


def extract_pages(reader: PdfReader, filename: str) -> list[Document]:
    """Extract one Document per page that contains text (1-based page numbers)."""
    pages: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - skip unreadable pages instead of failing
            raw_text = ""
        text = clean_text(raw_text)
        if text:
            pages.append(
                Document(page_content=text, metadata={"source": filename, "page": page_number})
            )

    if not pages:
        raise DocumentProcessingError(
            f"No extractable text found in '{filename}'. It may be a scanned "
            "(image-only) PDF, which this app cannot read."
        )
    return pages


def chunk_pages(pages: list[Document]) -> list[Document]:
    """Split page documents into overlapping chunks, keeping source/page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id
    return chunks


def process_pdf(filename: str, file_bytes: bytes) -> ProcessedDocument:
    """Read a PDF and return its pages and chunks.

    Raises DocumentProcessingError with a friendly message on failure.
    """
    reader = _open_reader(filename, file_bytes)
    total_pages = len(reader.pages)
    pages = extract_pages(reader, filename)
    chunks = chunk_pages(pages)
    return ProcessedDocument(
        filename=filename,
        size_bytes=len(file_bytes),
        total_pages=total_pages,
        pages=pages,
        chunks=chunks,
    )


def build_combined_context(
    documents: list[ProcessedDocument],
    max_total_chars: int = config.MAX_COMBINED_CHARS,
) -> str:
    """Join several papers into one prompt-ready string with a fair size budget each."""
    if not documents:
        return ""
    per_document = min(max_total_chars // len(documents), config.MAX_DOC_CHARS)
    parts = [
        f"=== PAPER: {doc.filename} ===\n{doc.get_text(per_document)}" for doc in documents
    ]
    return "\n\n".join(parts)