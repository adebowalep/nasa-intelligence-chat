"""
Centralised configuration for the NASA RAG system.

All runtime parameters are read from environment variables (or a .env file
loaded by python-dotenv). Hardcoded constants live here — never scattered
across modules — so a single change propagates everywhere.

Usage
-----
    from nasa_rag.config import get_settings

    cfg = get_settings()
    print(cfg.chat_model)          # "gpt-4o-mini"
    print(cfg.chroma_dir)          # "./chroma_db"
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env from the project root (two levels above this file: src/nasa_rag/config.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    """
    Immutable application settings resolved from environment variables.

    Attributes
    ----------
    openai_api_key:
        OpenAI API key. Required for LLM generation, embeddings, and RAGAS
        evaluation. Read from ``OPENAI_API_KEY``.
    chat_model:
        OpenAI chat-completion model identifier.
        Read from ``OPENAI_CHAT_MODEL`` (default: ``gpt-4o-mini``).
    embedding_model:
        OpenAI embedding model identifier.
        Read from ``OPENAI_EMBEDDING_MODEL`` (default: ``text-embedding-3-small``).
    chroma_dir:
        Path to the ChromaDB persistence directory.
        Read from ``CHROMA_DIR`` (default: ``./chroma_db``).
    collection_name:
        ChromaDB collection name.
        Read from ``COLLECTION_NAME`` (default: ``nasa_missions``).
    chunk_size:
        Maximum characters per text chunk during indexing.
        Read from ``CHUNK_SIZE`` (default: 1000).
    chunk_overlap:
        Character overlap between consecutive chunks.
        Read from ``CHUNK_OVERLAP`` (default: 150).
    max_history_turns:
        Maximum number of prior conversation turns forwarded to the LLM.
    default_top_k:
        Default number of chunks retrieved per query.
    eval_timeout_seconds:
        RAGAS evaluation timeout in seconds.
    """

    # ── OpenAI ───────────────────────────────────────────────────────────
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    chat_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )

    # ── ChromaDB ─────────────────────────────────────────────────────────
    chroma_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_DIR", "./chroma_db")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "nasa_missions")
    )

    # ── Chunking ─────────────────────────────────────────────────────────
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150"))
    )

    # ── Runtime constants ─────────────────────────────────────────────────
    max_history_turns: int = 6
    default_top_k: int = 5
    eval_timeout_seconds: int = 300

    def is_configured(self) -> bool:
        """Return True if the OpenAI API key is present."""
        return bool(self.openai_api_key)

    def validate(self) -> None:
        """
        Raise ``ValueError`` if required settings are missing.

        Raises
        ------
        ValueError
            When ``openai_api_key`` is empty.
        """
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})."
            )


# ---------------------------------------------------------------------------
# Supported missions — single source of truth
# ---------------------------------------------------------------------------

#: Normalisation map: user-supplied names → internal ChromaDB metadata values.
MISSION_NORMALISE: dict[str, str] = {
    "apollo 11": "apollo_11",
    "apollo11": "apollo_11",
    "apollo_11": "apollo_11",
    "apollo 13": "apollo_13",
    "apollo13": "apollo_13",
    "apollo_13": "apollo_13",
    "challenger": "challenger",
    "sts-51l": "challenger",
    "sts51l": "challenger",
}

#: Ordered list of known missions for UI display.
KNOWN_MISSIONS: list[str] = ["apollo_11", "apollo_13", "challenger"]

#: Human-readable display names keyed by internal mission identifier.
MISSION_DISPLAY: dict[str, str] = {
    "apollo_11": "Apollo 11",
    "apollo_13": "Apollo 13",
    "challenger": "Challenger (STS-51L)",
}


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application ``Settings`` singleton.

    The instance is cached after the first call. Environment variables are
    resolved at construction time, so changes to ``os.environ`` after the
    first call are **not** reflected.

    Returns
    -------
    Settings
        Populated, immutable settings object.
    """
    settings = Settings()
    if not settings.is_configured():
        logger.warning(
            "OPENAI_API_KEY is not set. "
            "LLM generation and evaluation will be unavailable."
        )
    return settings
