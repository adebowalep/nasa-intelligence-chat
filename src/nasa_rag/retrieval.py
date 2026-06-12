"""
Semantic retrieval client for the NASA RAG system.

Wraps ChromaDB to provide:
    - Collection discovery across the working directory.
    - Collection initialisation with graceful error handling.
    - Semantic similarity search with optional mission filtering.
    - Context formatting for LLM consumption.

All public functions return typed results and never raise; errors are
logged and communicated through return-value conventions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from nasa_rag.config import MISSION_NORMALISE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection discovery
# ---------------------------------------------------------------------------

def discover_chroma_backends(search_dir: str | Path = ".") -> dict[str, dict[str, str]]:
    """
    Scan a directory for ChromaDB persistence stores.

    Any subdirectory whose name contains ``"chroma"`` (case-insensitive) is
    probed. For each valid collection found, an entry is added to the
    returned dict keyed by ``"<dir_name>::<collection_name>"``.

    Parameters
    ----------
    search_dir:
        Directory to scan. Defaults to the current working directory.

    Returns
    -------
    dict[str, dict[str, str]]
        Maps backend_key → ``{directory, collection_name, display_name, doc_count}``.
        Returns an empty dict if no ChromaDB directories are found.
    """
    backends: dict[str, dict[str, str]] = {}
    base = Path(search_dir)

    if not base.is_dir():
        logger.warning("search_dir does not exist: %s", search_dir)
        return backends

    chroma_dirs = [d for d in base.iterdir() if d.is_dir() and "chroma" in d.name.lower()]

    for chroma_dir in chroma_dirs:
        try:
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collections = client.list_collections()

            for col in collections:
                key = f"{chroma_dir.name}::{col.name}"
                try:
                    count = col.count()
                except Exception:
                    count = 0

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": col.name,
                    "display_name": f"{col.name} ({count} docs) [{chroma_dir.name}]",
                    "doc_count": str(count),
                }

        except Exception as exc:
            key = f"{chroma_dir.name}::error"
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "error",
                "display_name": f"{chroma_dir.name} (error: {str(exc)[:40]})",
                "doc_count": "0",
            }
            logger.error("Failed to probe ChromaDB at %s: %s", chroma_dir, exc)

    return backends


# ---------------------------------------------------------------------------
# Collection initialisation
# ---------------------------------------------------------------------------

def initialize_collection(
    chroma_dir: str, collection_name: str
) -> tuple[Any, bool, str]:
    """
    Connect to a named ChromaDB collection.

    Parameters
    ----------
    chroma_dir:
        Filesystem path to the ChromaDB persistence directory.
    collection_name:
        Name of the collection to open.

    Returns
    -------
    tuple[collection | None, success: bool, error_message: str]
        On success: ``(collection_object, True, "")``.
        On failure: ``(None, False, human-readable error)``.
    """
    try:
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_collection(name=collection_name)
        logger.info(
            "Collection '%s' opened (%d documents).",
            collection_name,
            collection.count(),
        )
        return collection, True, ""
    except Exception as exc:
        logger.error("Failed to open collection '%s': %s", collection_name, exc)
        return None, False, str(exc)


# ---------------------------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------------------------

def retrieve_documents(
    collection: Any,
    query: str,
    n_results: int = 5,
    mission_filter: str | None = None,
) -> dict[str, Any] | None:
    """
    Run a semantic similarity search against a ChromaDB collection.

    Parameters
    ----------
    collection:
        ChromaDB ``Collection`` object returned by :func:`initialize_collection`.
    query:
        Natural-language question from the user.
    n_results:
        Number of top-k chunks to retrieve (default: 5).
    mission_filter:
        Optional mission name to restrict retrieval. Accepts human-readable
        forms such as ``"Apollo 11"``, ``"apollo_13"``, ``"Challenger"``,
        ``"all"``, or ``None`` (no filter).

    Returns
    -------
    dict | None
        Raw ChromaDB query result dict on success, or ``None`` on error.
    """
    if not query or not query.strip():
        logger.warning("retrieve_documents called with empty query.")
        return None

    where_filter: dict[str, Any] | None = None

    if mission_filter and mission_filter.lower() not in ("all", "all missions", ""):
        normalised = MISSION_NORMALISE.get(mission_filter.lower())
        if normalised is None:
            normalised = mission_filter.lower().replace(" ", "_")
        where_filter = {"mission": {"$eq": normalised}}
        logger.debug("Mission filter applied: %s → %s", mission_filter, normalised)

    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
        )
        logger.debug(
            "Retrieved %d chunks for query: %.60s…",
            len(results.get("documents", [[]])[0]),
            query,
        )
        return results
    except Exception as exc:
        logger.error("Document retrieval failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(
    documents: list[str],
    metadatas: list[dict[str, Any]],
    max_chunk_chars: int = 2000,
) -> str:
    """
    Format retrieved chunks into a structured context string for the LLM.

    Each unique chunk is labelled with its source number, mission, file path,
    and document category so the model can cite sources precisely.

    Parameters
    ----------
    documents:
        List of chunk texts (first element of ``ChromaDB result['documents']``).
    metadatas:
        Corresponding metadata dicts (from ``result['metadatas']``).
    max_chunk_chars:
        Maximum characters per chunk before truncation (default: 2000).

    Returns
    -------
    str
        Formatted context string, or an empty string when ``documents`` is empty.

    Example
    -------
    ::

        === RETRIEVED NASA MISSION CONTEXT ===

        [Source 1]
        Mission: Apollo 13
        File: data_text/apollo13/AS13_TEC_textract_full_text.txt
        Category: Technical
        Chunk: 4/92
        Content:
        … chunk text …
    """
    if not documents:
        return ""

    context_parts: list[str] = ["=== RETRIEVED NASA MISSION CONTEXT ===\n"]
    seen: set[str] = set()
    source_idx = 1

    for doc, meta in zip(documents, metadatas):
        fingerprint = doc[:120].strip()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        mission = meta.get("mission", "Unknown").replace("_", " ").title()
        file_path = meta.get("file_path", meta.get("source", "Unknown"))
        category = (
            meta.get("document_category", "document").replace("_", " ").title()
        )
        chunk_index = meta.get("chunk_index", "?")
        total_chunks = meta.get("total_chunks", "?")

        header = (
            f"[Source {source_idx}]\n"
            f"Mission: {mission}\n"
            f"File: {file_path}\n"
            f"Category: {category}\n"
            f"Chunk: {chunk_index}/{total_chunks}"
        )
        context_parts.append(header)

        content = doc if len(doc) <= max_chunk_chars else doc[:max_chunk_chars] + " … [truncated]"
        context_parts.append(f"Content:\n{content}\n")

        source_idx += 1

    return "\n".join(context_parts)
