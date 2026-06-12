"""
RAGAS response-quality evaluator for the NASA RAG system.

Computes two metrics per (question, answer, contexts) triple:

- ``response_relevancy`` — how relevant the answer is to the question
  (requires LLM + embeddings).
- ``faithfulness`` — whether factual claims in the answer are grounded in
  the retrieved context (requires LLM).

Architecture note — uvloop compatibility
----------------------------------------
Streamlit runs its event loop under ``uvloop``, which ``nest_asyncio``
cannot patch. The workaround is to run RAGAS inside a *dedicated thread*
that creates its own standard ``asyncio`` event loop, completely isolated
from the calling thread. This approach works in Streamlit, Jupyter, and
plain batch scripts without any loop-patching.

Usage
-----
    from nasa_rag.evaluation import evaluate_response

    scores = evaluate_response(
        question="What caused the Apollo 13 oxygen tank explosion?",
        answer="According to [Source 1] ...",
        contexts=["retrieved chunk 1", "retrieved chunk 2"],
        openai_key="sk-...",
    )
    # → {"response_relevancy": 0.91, "faithfulness": 0.88}
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_EVAL_TIMEOUT_SECONDS = 300

# ---------------------------------------------------------------------------
# Lazy RAGAS availability check — avoids importing the full LangChain stack
# at module load time (which adds 5–15 s on first import).
# The heavy imports are deferred to _run_ragas_in_thread() where they are
# actually needed.
# ---------------------------------------------------------------------------
RAGAS_AVAILABLE: bool = (
    importlib.util.find_spec("ragas") is not None
    and importlib.util.find_spec("langchain_openai") is not None
)
if not RAGAS_AVAILABLE:
    logger.warning(
        "RAGAS not available. Install with: pip install ragas langchain-openai"
    )


# ---------------------------------------------------------------------------
# Internal: isolated-thread RAGAS execution
# ---------------------------------------------------------------------------

def _run_ragas_in_thread(
    question: str,
    answer: str,
    contexts: list[str],
    openai_key: str,
) -> dict[str, float]:
    """
    Execute RAGAS evaluation inside a dedicated thread with a fresh asyncio
    event loop, avoiding uvloop incompatibilities entirely.

    Parameters
    ----------
    question, answer, contexts, openai_key:
        Evaluation inputs (already validated by the public wrapper).

    Returns
    -------
    dict[str, float]
        ``{"response_relevancy": float, "faithfulness": float}`` on success,
        or ``{"error": str}`` on failure.
    """
    # Defer heavy imports to here — they only run when evaluation is actually
    # requested, not on every module import.
    from ragas import SingleTurnSample, EvaluationDataset, evaluate  # noqa: PLC0415
    from ragas.metrics import ResponseRelevancy, Faithfulness  # noqa: PLC0415
    from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415
    from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: PLC0415

    # Respect OPENAI_BASE_URL so non-standard endpoints (Vocareum, Azure, etc.)
    # are used consistently for evaluation calls.  LangChain's ChatOpenAI and
    # OpenAIEmbeddings both accept a ``base_url`` keyword.
    base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
    endpoint_kwargs: dict[str, str] = {"base_url": base_url} if base_url else {}

    # Use the configured chat model rather than a hardcoded name, so the same
    # model works for endpoints that may not expose every OpenAI model variant
    # (e.g. Vocareum exposes gpt-4o-mini).
    from nasa_rag.config import get_settings  # noqa: PLC0415
    cfg = get_settings()
    eval_model = cfg.chat_model

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=eval_model,
                api_key=openai_key,
                temperature=0,
                **endpoint_kwargs,
            )
        )
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=cfg.embedding_model,
                api_key=openai_key,
                **endpoint_kwargs,
            )
        )

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        dataset = EvaluationDataset(samples=[sample])

        metrics = [
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            Faithfulness(llm=evaluator_llm),
        ]

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        scores: dict[str, float] = {}
        if hasattr(result, "to_pandas"):
            df = result.to_pandas()
            for col in ("response_relevancy", "faithfulness"):
                if col in df.columns:
                    val = df[col].iloc[0]
                    if val is not None and str(val) != "nan":
                        scores[col] = round(float(val), 4)
        elif hasattr(result, "__getitem__"):
            for key in ("response_relevancy", "faithfulness"):
                try:
                    val = result[key]
                    if val is not None:
                        scores[key] = round(float(val), 4)
                except (KeyError, TypeError):
                    pass

        return scores if scores else {"error": "RAGAS returned no scores."}

    except Exception as exc:
        logger.error("RAGAS thread evaluation failed: %s", exc)
        return {"error": str(exc)}
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
    openai_key: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a single RAG response triple with RAGAS metrics.

    Metrics computed
    ----------------
    ``response_relevancy``
        How directly and completely the answer addresses the question.
        Score range: 0.0 – 1.0.
    ``faithfulness``
        Proportion of factual claims in the answer that are supported by
        the retrieved context.  Score range: 0.0 – 1.0.

    Both metrics make additional LLM calls and add ~10–30 s of latency.

    Parameters
    ----------
    question:
        The user's original question.
    answer:
        The LLM-generated response to evaluate.
    contexts:
        List of retrieved document chunks used when generating the answer.
    openai_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` environment
        variable if ``None``.

    Returns
    -------
    dict[str, float | str]
        ``{"response_relevancy": float, "faithfulness": float}`` on success.
        ``{"error": str}`` on failure (never raises).
    """
    if not RAGAS_AVAILABLE:
        return {
            "error": (
                "RAGAS is not installed. "
                "Run: pip install ragas langchain-openai langchain-core"
            )
        }

    # ── Input validation ──────────────────────────────────────────────────
    if not question or not question.strip():
        return {"error": "question must not be empty."}
    if not answer or not answer.strip():
        return {"error": "answer must not be empty."}
    if not contexts:
        contexts = ["No context retrieved."]

    # ── Resolve API key ───────────────────────────────────────────────────
    resolved_key = (
        openai_key
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("CHROMA_OPENAI_API_KEY")
        or ""
    )
    if not resolved_key:
        return {
            "error": (
                "OpenAI API key not found. "
                "Set OPENAI_API_KEY in .env or pass openai_key explicitly."
            )
        }

    # ── Run RAGAS in an isolated thread ───────────────────────────────────
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _run_ragas_in_thread, question, answer, contexts, resolved_key
            )
            return future.result(timeout=_EVAL_TIMEOUT_SECONDS)

    except concurrent.futures.TimeoutError:
        logger.warning("RAGAS evaluation timed out after %ds.", _EVAL_TIMEOUT_SECONDS)
        return {"error": f"Evaluation timed out after {_EVAL_TIMEOUT_SECONDS}s."}
    except Exception as exc:
        logger.error("Unexpected error during RAGAS evaluation: %s", exc)
        return {"error": f"Evaluation failed: {exc}"}
