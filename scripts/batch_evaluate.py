#!/usr/bin/env python3
"""
Batch evaluation runner for the NASA Intelligence Chat System.

Loads evaluation questions from ``data/evaluation_dataset.txt``, runs each
through the full RAG pipeline (retrieve → generate → evaluate), and writes
per-question RAGAS scores plus an aggregate row to ``evaluation_results.csv``.

Usage
-----
    python scripts/batch_evaluate.py

Environment (via .env or shell exports)
-----------------------------------------
    OPENAI_API_KEY     — required
    CHROMA_DIR         — default: ./chroma_db
    COLLECTION_NAME    — default: nasa_missions
    OPENAI_CHAT_MODEL  — default: gpt-4o-mini
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Resolve project root (scripts/ → parent)
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from nasa_rag.retrieval import format_context, initialize_collection, retrieve_documents
from nasa_rag.generation import generate_response
from nasa_rag.evaluation import evaluate_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_ROOT / "batch_evaluation.log"),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
CHROMA_DIR = os.getenv("CHROMA_DIR", str(_ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "nasa_missions")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

DATASET_FILE = _ROOT / "data" / "evaluation_dataset.txt"
RESULTS_FILE = _ROOT / "evaluation_results.csv"
TOP_K = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(filepath: Path) -> list[str]:
    """Load questions, skipping blank lines and ``#`` comments."""
    if not filepath.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {filepath}")
    questions = []
    with open(filepath, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)
    return questions


def run_pipeline(
    collection: Any,
    question: str,
    openai_key: str,
    model: str,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Run one question through retrieval → generation → evaluation."""
    record: dict[str, Any] = {
        "question": question,
        "answer": "",
        "num_contexts": 0,
        "response_relevancy": None,
        "faithfulness": None,
        "error": "",
    }

    try:
        results = retrieve_documents(collection, question, n_results=top_k)
        contexts: list[str] = []
        context_str = ""

        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            context_str = format_context(docs, metas)
            contexts = docs
        else:
            logger.warning("No documents retrieved for: %.60s", question)

        record["num_contexts"] = len(contexts)

        answer = generate_response(
            openai_key=openai_key,
            user_message=question,
            context=context_str,
            conversation_history=[],
            model=model,
        )
        record["answer"] = answer
        logger.info("  Answer preview: %.100s…", answer)

        scores = evaluate_response(
            question=question,
            answer=answer,
            contexts=contexts,
            openai_key=openai_key,
        )

        if "error" in scores:
            logger.warning("  RAGAS error: %s", scores["error"])
            record["error"] = scores["error"]
        else:
            record["response_relevancy"] = scores.get("response_relevancy")
            record["faithfulness"] = scores.get("faithfulness")
            logger.info(
                "  relevancy=%.3f  faithfulness=%.3f",
                record["response_relevancy"] or 0,
                record["faithfulness"] or 0,
            )

    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        record["error"] = str(exc)

    return record


def compute_aggregate(records: list[dict]) -> dict[str, Any]:
    rel = [r["response_relevancy"] for r in records if r["response_relevancy"] is not None]
    faith = [r["faithfulness"] for r in records if r["faithfulness"] is not None]
    return {
        "question": f"AGGREGATE ({len(records)} questions)",
        "answer": "",
        "num_contexts": 0,
        "response_relevancy": round(sum(rel) / len(rel), 4) if rel else None,
        "faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
        "error": "",
    }


def write_csv(records: list[dict], filepath: Path) -> None:
    fieldnames = ["question", "answer", "num_contexts", "response_relevancy", "faithfulness", "error"]
    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["answer"] = str(row.get("answer", ""))[:300].replace("\n", " ")
            writer.writerow(row)
    logger.info("Results written to: %s", filepath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("NASA Intelligence Chat — Batch Evaluator")
    logger.info("=" * 60)

    if not OPENAI_KEY:
        logger.error("OPENAI_API_KEY is not set.")
        sys.exit(1)

    questions = load_questions(DATASET_FILE)
    logger.info("Loaded %d questions from %s", len(questions), DATASET_FILE)

    collection, ok, err = initialize_collection(CHROMA_DIR, COLLECTION_NAME)
    if not ok:
        logger.error("Could not open ChromaDB: %s", err)
        sys.exit(1)

    logger.info("Collection has %d documents.", collection.count())
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY

    records: list[dict] = []
    t0 = time.time()

    for i, question in enumerate(questions, start=1):
        logger.info("[%d/%d] %s", i, len(questions), question)
        records.append(
            run_pipeline(collection, question, OPENAI_KEY, CHAT_MODEL)
        )
        time.sleep(1)  # Avoid rate limits

    elapsed = time.time() - t0
    agg = compute_aggregate(records)
    write_csv(records + [agg], RESULTS_FILE)

    logger.info("=" * 60)
    logger.info("Questions evaluated : %d", len(records))
    logger.info("Elapsed             : %.1fs", elapsed)
    logger.info(
        "Mean relevancy      : %s",
        f"{agg['response_relevancy']:.4f}" if agg["response_relevancy"] else "N/A",
    )
    logger.info(
        "Mean faithfulness   : %s",
        f"{agg['faithfulness']:.4f}" if agg["faithfulness"] else "N/A",
    )
    logger.info("Results saved to    : %s", RESULTS_FILE)


if __name__ == "__main__":
    main()
