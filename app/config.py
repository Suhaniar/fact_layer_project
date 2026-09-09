"""
Centralized configuration for the Evidence-Grounded Fact Knowledge Layer.

Every other module imports settings FROM HERE instead of reading
environment variables or hardcoding values itself. This keeps all the
tunable knobs (Ollama URL, model name, thresholds, etc.) in one place.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if one exists.
load_dotenv()


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

UPLOAD_DIRECTORY: Path = Path(
    _get_env("UPLOAD_DIRECTORY", str(PROJECT_ROOT / "data" / "uploads"))
)

DATABASE_PATH: Path = Path(
    _get_env("DATABASE_PATH", str(PROJECT_ROOT / "data" / "fact_knowledge_layer.db"))
)

UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = _get_env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = _get_env("OLLAMA_MODEL", "llama3.2:3b")

# How long (seconds) we wait for a single Ollama call before giving up.
OLLAMA_TIMEOUT_SECONDS: int = _get_env_int("OLLAMA_TIMEOUT_SECONDS", 60)

# We want deterministic, evidence-grounded extraction, not creative output.
OLLAMA_TEMPERATURE: float = _get_env_float("OLLAMA_TEMPERATURE", 0.0)

# Caps how many tokens the model may generate. Keeping this modest is a
# major lever for speed: a huge num_predict means Ollama keeps writing
# even after the useful JSON is already complete.
OLLAMA_NUM_PREDICT: int = _get_env_int("OLLAMA_NUM_PREDICT", 700)

# Context window size (tokens). Big enough for prompt + page text +
# response, but far less than the model's max context.
OLLAMA_NUM_CTX: int = _get_env_int("OLLAMA_NUM_CTX", 4096)

# Keeps the model loaded in memory between calls instead of reloading
# it (which is slow) on every single page.
OLLAMA_KEEP_ALIVE: str = _get_env("OLLAMA_KEEP_ALIVE", "10m")


# ---------------------------------------------------------------------------
# Embedding / matching configuration
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Minimum cosine similarity for two facts to be considered a "candidate"
# related pair before we spend effort comparing them in detail.
MATCH_SIMILARITY_THRESHOLD: float = _get_env_float("MATCH_SIMILARITY_THRESHOLD", 0.55)

# Safety cap on candidate pairs per run, so a huge fact set can't blow
# up runtime unexpectedly.
MAX_CANDIDATE_PAIRS: int = _get_env_int("MAX_CANDIDATE_PAIRS", 5000)


# ---------------------------------------------------------------------------
# Performance / extraction heuristics
# ---------------------------------------------------------------------------

# Pages with fewer non-whitespace characters than this are treated as
# "empty" and skipped entirely (blank pages, pure image pages, etc.).
MIN_PAGE_TEXT_LENGTH: int = _get_env_int("MIN_PAGE_TEXT_LENGTH", 40)

# We only send up to this many characters of page text to the LLM. Long
# pages are truncated (not dropped), so evidence still traces to a real
# page, but the prompt stays small and fast.
MAX_PAGE_TEXT_CHARS: int = _get_env_int("MAX_PAGE_TEXT_CHARS", 3000)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = _get_env("LOG_LEVEL", "INFO")

# Configure logging once, centrally. logging.basicConfig() is a no-op if
# the root logger already has handlers, so this stays safe even when
# Streamlit re-imports this module on every rerun.
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)