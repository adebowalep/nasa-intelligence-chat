"""
ChromaDB embedding pipeline for NASA mission documents.

Reads plain-text NASA archive files, splits them into overlapping chunks,
generates OpenAI embeddings, and persists everything in a ChromaDB collection.

Supported missions
------------------
- Apollo 11  (data_text/apollo11/)
- Apollo 13  (data_text/apollo13/)
- Challenger STS-51L  (data_text/challenger/)

Usage — programmatic
--------------------
    from nasa_rag.embedding import EmbeddingPipeline

    pipeline = EmbeddingPipeline(
        openai_api_key="sk-...",
        chroma_dir="./chroma_db",
        collection_name="nasa_missions",
    )
    stats = pipeline.process_all("data_text", update_mode="replace")
    print(stats)

Usage — CLI
-----------
    python -m nasa_rag.embedding \\
        --data-dir data_text \\
        --update-mode replace \\
        --stats-only
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata extraction helpers — module-level (stateless, easy to test)
# ---------------------------------------------------------------------------

_MISSION_PATTERNS: dict[str, list[str]] = {
    "apollo_11": ["apollo11", "apollo_11"],
    "apollo_13": ["apollo13", "apollo_13"],
    "challenger": ["challenger"],
}

_DATA_TYPE_PATTERNS: dict[str, str] = {
    "transcript": "transcript",
    "textract": "textract_extracted",
    "audio": "audio_transcript",
    "flight_plan": "flight_plan",
}

_CATEGORY_PATTERNS: dict[str, str] = {
    "pao": "public_affairs_officer",
    "cm": "command_module",
    "tec": "technical",
    "flight_plan": "flight_plan",
    "mission_audio": "mission_audio",
    "ntrs": "nasa_archive",
    "19900066485": "technical_report",
    "19710015566": "mission_report",
}


def extract_mission(file_path: Path) -> str:
    """Return the internal mission identifier for *file_path*."""
    path_str = str(file_path).lower()
    for mission, patterns in _MISSION_PATTERNS.items():
        if any(p in path_str for p in patterns):
            return mission
    return "unknown"


def extract_data_type(file_path: Path) -> str:
    """Infer the data type (e.g. transcript, flight plan) from the file path."""
    path_str = str(file_path).lower()
    for token, label in _DATA_TYPE_PATTERNS.items():
        if token in path_str:
            return label
    return "document"


def extract_document_category(filename: str) -> str:
    """Infer a human-readable document category from the filename."""
    fn = filename.lower()
    for token, label in _CATEGORY_PATTERNS.items():
        if token in fn:
            return label
    if "full_text" in fn:
        return "complete_document"
    return "general_document"


# ---------------------------------------------------------------------------
# Embedding pipeline class
# ---------------------------------------------------------------------------

class EmbeddingPipeline:
    """
    End-to-end pipeline: text files → ChromaDB collection with OpenAI embeddings.

    Parameters
    ----------
    openai_api_key:
        OpenAI API key.
    chroma_dir:
        Filesystem path for ChromaDB persistence (created if absent).
    collection_name:
        ChromaDB collection to create or open.
    embedding_model:
        OpenAI embedding model (default: ``"text-embedding-3-small"``).
    chunk_size:
        Maximum characters per chunk (default: 1000).
    chunk_overlap:
        Overlap between consecutive chunks in characters (default: 150).
    """

    def __init__(
        self,
        openai_api_key: str,
        chroma_dir: str = "./chroma_db",
        collection_name: str = "nasa_missions",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        self.openai_api_key = openai_api_key
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(chroma_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedding_fn = OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name=embedding_model,
        )
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' ready (%d documents).",
            collection_name,
            self.collection.count(),
        )

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def chunk_text(
        self, text: str, base_metadata: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Split *text* into overlapping chunks with per-chunk metadata.

        Chunks break preferentially at sentence boundaries (``'.'``), then at
        newline boundaries, before falling back to a hard cut at ``chunk_size``
        characters.

        Parameters
        ----------
        text:
            Full document text.
        base_metadata:
            Metadata dict shared by all chunks. ``chunk_index`` and
            ``total_chunks`` keys are added per chunk.

        Returns
        -------
        list[tuple[str, dict]]
            ``[(chunk_text, chunk_metadata), …]``
        """
        if len(text) <= self.chunk_size:
            return [(text.strip(), {**base_metadata, "chunk_index": 0, "total_chunks": 1})]

        raw_chunks: list[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            if end < len(text):
                boundary = text.rfind(".", start + self.chunk_size // 2, end)
                if boundary == -1:
                    boundary = text.rfind("\n", start + self.chunk_size // 2, end)
                if boundary != -1:
                    end = boundary + 1

            chunk = text[start:end].strip()
            if chunk:
                raw_chunks.append(chunk)

            next_start = max(end - self.chunk_overlap, start + 1)
            start = next_start

        total = len(raw_chunks)
        return [
            (chunk, {**base_metadata, "chunk_index": idx, "total_chunks": total})
            for idx, chunk in enumerate(raw_chunks)
        ]

    # ------------------------------------------------------------------
    # File processing
    # ------------------------------------------------------------------

    def process_text_file(
        self, file_path: Path
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Read *file_path*, extract metadata, and return chunks ready for
        insertion into ChromaDB.

        Returns an empty list for empty files or on read errors.
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.error("Cannot read %s: %s", file_path, exc)
            return []

        if not content.strip():
            logger.warning("Skipping empty file: %s", file_path)
            return []

        metadata: dict[str, Any] = {
            "source": file_path.stem,
            "file_path": str(file_path),
            "file_type": "text",
            "mission": extract_mission(file_path),
            "data_type": extract_data_type(file_path),
            "document_category": extract_document_category(file_path.name),
            "file_size": len(content),
            "processed_timestamp": datetime.now().isoformat(),
        }

        return self.chunk_text(content, metadata)

    # ------------------------------------------------------------------
    # ChromaDB CRUD
    # ------------------------------------------------------------------

    def _make_doc_id(self, file_path: Path, chunk_meta: dict[str, Any]) -> str:
        """Generate a stable, human-readable ChromaDB document ID."""
        mission = chunk_meta.get("mission", "unknown")
        source = chunk_meta.get("source", file_path.stem).replace(" ", "_")
        idx = chunk_meta.get("chunk_index", 0)
        return f"{mission}_{source}_chunk_{idx:04d}"

    def _get_file_doc_ids(self, file_path: Path) -> list[str]:
        """Return all ChromaDB IDs that belong to *file_path*."""
        try:
            all_docs = self.collection.get()
            source = file_path.stem
            mission = extract_mission(file_path)
            return [
                all_docs["ids"][i]
                for i, meta in enumerate(all_docs["metadatas"] or [])
                if meta.get("source") == source and meta.get("mission") == mission
            ]
        except Exception as exc:
            logger.error("Error fetching IDs for %s: %s", file_path, exc)
            return []

    def _doc_exists(self, doc_id: str) -> bool:
        """Return True if *doc_id* already exists in the collection."""
        try:
            return len(self.collection.get(ids=[doc_id])["ids"]) > 0
        except Exception:
            return False

    def add_chunks(
        self,
        chunks: list[tuple[str, dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> dict[str, int]:
        """
        Insert *chunks* into ChromaDB, respecting *update_mode*.

        Parameters
        ----------
        update_mode:
            ``"skip"``    — ignore chunks that already exist (default).
            ``"update"``  — re-embed and overwrite existing chunks.
            ``"replace"`` — delete all existing chunks for *file_path*, then
                            insert fresh.

        Returns
        -------
        dict with keys ``added``, ``updated``, ``skipped``.
        """
        if not chunks:
            return {"added": 0, "updated": 0, "skipped": 0}

        stats = {"added": 0, "updated": 0, "skipped": 0}

        if update_mode == "replace":
            existing = self._get_file_doc_ids(file_path)
            if existing:
                self.collection.delete(ids=existing)
                logger.info("Removed %d existing chunks for %s.", len(existing), file_path.name)

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            ids, texts, metas = [], [], []

            for text, meta in batch:
                doc_id = self._make_doc_id(file_path, meta)

                if update_mode == "skip" and self._doc_exists(doc_id):
                    stats["skipped"] += 1
                    continue
                if update_mode == "update" and self._doc_exists(doc_id):
                    try:
                        self.collection.update(ids=[doc_id], documents=[text], metadatas=[meta])
                        stats["updated"] += 1
                    except Exception as exc:
                        logger.error("Update failed for %s: %s", doc_id, exc)
                    continue

                ids.append(doc_id)
                texts.append(text)
                metas.append(meta)

            if ids:
                try:
                    self.collection.add(ids=ids, documents=texts, metadatas=metas)
                    stats["added"] += len(ids)
                except Exception as exc:
                    logger.error("Batch insert failed at offset %d: %s", batch_start, exc)

        return stats

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------

    def scan_text_files(self, data_dir: str | Path) -> list[Path]:
        """
        Scan *data_dir* for ``.txt`` files across mission subdirectories.

        Looks for subdirectories named ``apollo11``, ``apollo13``, and
        ``challenger`` under *data_dir*.

        Returns
        -------
        list[Path]
            Filtered list of ``.txt`` file paths.
        """
        base = Path(data_dir)
        mission_dirs = ["apollo11", "apollo13", "challenger"]
        files: list[Path] = []

        for dir_name in mission_dirs:
            dir_path = base / dir_name
            if dir_path.exists():
                txt_files = list(dir_path.glob("**/*.txt"))
                files.extend(txt_files)
                logger.info("  %s: %d .txt files found.", dir_name, len(txt_files))
            else:
                logger.warning("Mission directory not found: %s", dir_path)

        return [
            fp for fp in files
            if not fp.name.startswith(".") and fp.suffix.lower() == ".txt"
        ]

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def process_all(
        self, data_dir: str | Path, update_mode: str = "skip"
    ) -> dict[str, Any]:
        """
        Process all NASA text files and populate the ChromaDB collection.

        Parameters
        ----------
        data_dir:
            Root directory containing ``apollo11/``, ``apollo13/``,
            ``challenger/`` sub-folders.
        update_mode:
            ``"skip"`` | ``"update"`` | ``"replace"``

        Returns
        -------
        dict
            Aggregate statistics: ``files_processed``, ``total_chunks``,
            ``documents_added``, ``documents_updated``, ``documents_skipped``,
            ``errors``, ``missions`` (per-mission breakdown).
        """
        stats: dict[str, Any] = {
            "files_processed": 0,
            "total_chunks": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "missions": {},
        }

        files = self.scan_text_files(data_dir)
        if not files:
            logger.warning("No text files found in: %s", data_dir)
            return stats

        for file_path in files:
            mission = extract_mission(file_path)
            if mission not in stats["missions"]:
                stats["missions"][mission] = {"files": 0, "chunks": 0, "added": 0, "updated": 0, "skipped": 0}

            try:
                logger.info("Processing: %s", file_path.name)
                chunks = self.process_text_file(file_path)
                if not chunks:
                    continue

                file_stats = self.add_chunks(chunks, file_path, update_mode=update_mode)

                stats["files_processed"] += 1
                stats["total_chunks"] += len(chunks)
                stats["documents_added"] += file_stats["added"]
                stats["documents_updated"] += file_stats["updated"]
                stats["documents_skipped"] += file_stats["skipped"]

                m = stats["missions"][mission]
                m["files"] += 1
                m["chunks"] += len(chunks)
                m["added"] += file_stats["added"]
                m["updated"] += file_stats["updated"]
                m["skipped"] += file_stats["skipped"]

                logger.info(
                    "  → %d chunks | added=%d updated=%d skipped=%d",
                    len(chunks), file_stats["added"], file_stats["updated"], file_stats["skipped"],
                )

            except Exception as exc:
                logger.error("Error processing %s: %s", file_path, exc)
                stats["errors"] += 1

        return stats

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def collection_info(self) -> dict[str, Any]:
        """Return basic collection metadata."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "chroma_dir": self.chroma_dir,
            "embedding_model": self.embedding_model,
        }

    def collection_stats(self) -> dict[str, Any]:
        """Return per-mission, per-category document counts."""
        try:
            all_docs = self.collection.get()
            if not all_docs.get("metadatas"):
                return {"total_documents": 0, "note": "Collection is empty."}

            result: dict[str, Any] = {
                "total_documents": len(all_docs["metadatas"]),
                "missions": {},
                "data_types": {},
                "document_categories": {},
            }
            for meta in all_docs["metadatas"]:
                for key, bucket in (
                    ("mission", "missions"),
                    ("data_type", "data_types"),
                    ("document_category", "document_categories"),
                ):
                    val = meta.get(key, "unknown")
                    result[bucket][val] = result[bucket].get(val, 0) + 1

            return result
        except Exception as exc:
            logger.error("Error computing collection stats: %s", exc)
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# CLI entry-point (python -m nasa_rag.embedding)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NASA Mission Intelligence — Embedding Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data_text", help="Root data directory")
    p.add_argument("--openai-key", default=None)
    p.add_argument("--chroma-dir", default="./chroma_db")
    p.add_argument("--collection-name", default="nasa_missions")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--update-mode", choices=["skip", "update", "replace"], default="skip")
    p.add_argument("--stats-only", action="store_true")
    p.add_argument("--test-query", default=None)
    return p


def main() -> None:
    """CLI entry point for the embedding pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

    args = _build_parser().parse_args()
    api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required. Pass --openai-key or set the env var.")

    pipeline = EmbeddingPipeline(
        openai_api_key=api_key,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if args.stats_only:
        info = pipeline.collection_info()
        stats = pipeline.collection_stats()
        logger.info("Collection : %s (%d docs)", info["collection_name"], info["document_count"])
        for mission, count in stats.get("missions", {}).items():
            logger.info("  %s: %d chunks", mission, count)
        return

    t0 = time.time()
    stats = pipeline.process_all(args.data_dir, update_mode=args.update_mode)
    elapsed = time.time() - t0

    logger.info("=" * 50)
    logger.info("Files processed : %d", stats["files_processed"])
    logger.info("Total chunks    : %d", stats["total_chunks"])
    logger.info("Added           : %d", stats["documents_added"])
    logger.info("Updated         : %d", stats["documents_updated"])
    logger.info("Skipped         : %d", stats["documents_skipped"])
    logger.info("Errors          : %d", stats["errors"])
    logger.info("Elapsed         : %.1fs", elapsed)

    if args.test_query:
        results = pipeline.collection.query(query_texts=[args.test_query], n_results=3)
        for i, doc in enumerate(results["documents"][0]):
            logger.info("Result %d: %s…", i + 1, doc[:200])


if __name__ == "__main__":
    main()
