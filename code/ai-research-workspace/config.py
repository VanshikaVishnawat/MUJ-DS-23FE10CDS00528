"""Central configuration for the AI Research Workspace.

Secrets (the Gemini API key) come from the environment / a local `.env` file.
Everything else is a plain setting you can tweak here or override via `.env`.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load .env once at import time (harmless if the file does not exist).
load_dotenv(ENV_PATH)

# Hide a harmless Windows-only Hugging Face warning about symlinks.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
APP_TITLE = "AI Research Workspace"

# --------------------------------------------------------------------------- #
# LLM: Google Gemini API (free tier)
# --------------------------------------------------------------------------- #
# Override in .env with GEMINI_MODEL. Other free-tier options you can try:
# gemini-3.5-flash-lite, gemini-3-flash-preview, gemini-2.5-flash
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Backward-compatible alias: app.py still displays `config.OPENAI_MODEL`.
OPENAI_MODEL = GEMINI_MODEL


def _resolve_temperature() -> float | None:
    """Pick the sampling temperature.

    Google recommends leaving temperature at its default (1.0) for Gemini 3
    models, because lower values can cause looping. We therefore return None
    (= "do not send a temperature") for gemini-3* models and use a low value
    for older models. Set TEMPERATURE in .env to force a specific value.
    """
    raw = os.getenv("TEMPERATURE", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    if GEMINI_MODEL.lower().startswith("gemini-3"):
        return None
    return 0.2


TEMPERATURE = _resolve_temperature()

LLM_TIMEOUT_SECONDS = 120   # max wait for one Gemini request
LLM_MAX_RETRIES = 2         # automatic retries (e.g. short rate-limit spikes)

# --------------------------------------------------------------------------- #
# Embeddings: Sentence Transformers, running locally (no API calls)
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# --------------------------------------------------------------------------- #
# Chunking and retrieval
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # characters shared between neighbouring chunks
TOP_K = 6              # number of chunks retrieved from FAISS per question

# --------------------------------------------------------------------------- #
# Limits for prompts that read (parts of) whole papers
# --------------------------------------------------------------------------- #
MAX_DOC_CHARS = 60000          # per paper for summaries
MAX_COMPARE_DOC_CHARS = 45000  # per paper when comparing two papers
MAX_COMBINED_CHARS = 90000     # total across papers for gaps / concepts
MAX_HISTORY_MESSAGES = 6       # recent chat messages passed to the LLM

# --------------------------------------------------------------------------- #
# API key helpers
# --------------------------------------------------------------------------- #
API_KEY_MISSING_MESSAGE = (
    "GEMINI_API_KEY is missing. Get a free key at https://aistudio.google.com/apikey, "
    "copy `.env.example` to `.env`, put the key in it, and try again."
)

_PLACEHOLDER_KEYS = {"", "your_key_here", "your_gemini_api_key_here"}


def get_api_key() -> str | None:
    """Return the Gemini API key, or None if it is missing.

    The .env file is re-read on every call, so you can add the key while the
    app is running without restarting Streamlit.
    """
    load_dotenv(ENV_PATH, override=True)
    key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if key.lower() in _PLACEHOLDER_KEYS:
        return None
    return key


def is_api_key_configured() -> bool:
    return get_api_key() is not None