#!/usr/bin/env python3
"""
Build or update the NASA mission ChromaDB embedding index.

This is a thin CLI wrapper around :class:`nasa_rag.embedding.EmbeddingPipeline`.
Run from the project root after installing dependencies.

Examples
--------
Build fresh index (replace existing):
    python scripts/build_index.py --update-mode replace

Incremental update (skip already-indexed chunks):
    python scripts/build_index.py --update-mode skip

Inspect current index without processing:
    python scripts/build_index.py --stats-only

Run a test query:
    python scripts/build_index.py --stats-only --test-query "oxygen tank explosion"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Resolve project root (scripts/ → parent)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from nasa_rag.embedding import EmbeddingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_ROOT / "embedding_pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NASA Intelligence Chat — Embedding Index Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        default=str(_ROOT / "data_text"),
        help="Root directory with apollo11/, apollo13/, challenger/ sub-folders.",
    )
    p.add_argument("--openai-key", default=None, help="OpenAI API key (env fallback: OPENAI_API_KEY)")
    p.add_argument("--chroma-dir", default=str(_ROOT / "chroma_db"))
    p.add_argument("--collection-name", default="nasa_missions")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--chunk-overlap", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument(
        "--update-mode",
        choices=["skip", "update", "replace"],
        default="skip",
        help="skip=ignore existing | update=re-embed existing | replace=wipe+rebuild per file",
    )
    p.add_argument("--stats-only", action="store_true", help="Print stats and exit.")
    p.add_argument("--test-query", default=None, help="Run a test query and print top-3 results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is required. Pass --openai-key or set the environment variable.")
        sys.exit(1)

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
        logger.info("=" * 50)
        logger.info("Collection  : %s", info["collection_name"])
        logger.info("Directory   : %s", info["chroma_dir"])
        logger.info("Documents   : %d", info["document_count"])
        logger.info("Embed model : %s", info["embedding_model"])
        logger.info("-" * 50)
        for mission, count in stats.get("missions", {}).items():
            logger.info("  %-20s : %d chunks", mission, count)
        if args.test_query:
            results = pipeline.collection.query(query_texts=[args.test_query], n_results=3)
            logger.info("\nTest query: '%s'", args.test_query)
            for i, doc in enumerate(results["documents"][0]):
                logger.info("  [%d] %s…", i + 1, doc[:200])
        return

    logger.info("Starting embedding pipeline | update-mode=%s", args.update_mode)
    t0 = time.time()
    stats = pipeline.process_all(args.data_dir, update_mode=args.update_mode)
    elapsed = time.time() - t0

    logger.info("=" * 50)
    logger.info("FILES PROCESSED : %d", stats["files_processed"])
    logger.info("TOTAL CHUNKS    : %d", stats["total_chunks"])
    logger.info("ADDED           : %d", stats["documents_added"])
    logger.info("UPDATED         : %d", stats["documents_updated"])
    logger.info("SKIPPED         : %d", stats["documents_skipped"])
    logger.info("ERRORS          : %d", stats["errors"])
    logger.info("ELAPSED         : %.1fs", elapsed)
    logger.info("-" * 50)
    for mission, m in stats["missions"].items():
        logger.info(
            "  %-20s : %d files, %d chunks (+%d added)",
            mission, m["files"], m["chunks"], m["added"],
        )


if __name__ == "__main__":
    main()
