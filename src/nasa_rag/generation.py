"""
OpenAI LLM client for the NASA RAG system.

Provides grounded, citation-aware responses anchored exclusively to retrieved
NASA archive context. The system prompt instructs the model to:

- Answer only from the supplied context.
- Cite [Source N] labels for every factual claim.
- Declare uncertainty explicitly when context is insufficient.
- Never fabricate names, dates, or technical details.

Usage
-----
    from nasa_rag.generation import generate_response

    answer = generate_response(
        openai_key="sk-...",
        user_message="What caused the Apollo 13 oxygen tank failure?",
        context=formatted_context,
        conversation_history=[],
    )
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — single source of truth for LLM behaviour
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a NASA mission intelligence assistant. Your role is to answer \
questions using ONLY the retrieved NASA mission context provided to you.

Guidelines:
1. Answer clearly and accurately based solely on the provided context.
2. Always cite your sources using the [Source N] labels from the context \
   (e.g. "According to [Source 1] — Apollo 13 Technical Transcript …").
3. If the retrieved context does not contain enough evidence to answer the \
   question, say so clearly:
   "The available NASA documents do not provide sufficient information to answer \
this question."
4. Never fabricate facts, dates, names, or technical details not present in \
   the context.
5. When relevant, mention the mission name, document type (transcript, technical \
   report, flight plan), and any key identifiers.
6. Keep answers focused and grounded; prefer evidence over speculation.
"""

_CONTEXT_PREFIX = (
    "Use the following retrieved NASA mission documents to answer "
    "the user's question. Cite [Source N] labels where relevant.\n\n"
)

_NO_CONTEXT_MESSAGE = (
    "No relevant documents were retrieved from the NASA archive for "
    "this query. Inform the user that the available documents do not "
    "provide sufficient information to answer the question."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: list[dict[str, Any]],
    model: str = "gpt-4o-mini",
    max_history_turns: int = 6,
    temperature: float = 0.1,
    max_tokens: int = 1500,
) -> str:
    """
    Generate a grounded answer using OpenAI chat completions.

    The response is anchored to ``context`` — retrieved NASA document chunks
    — and will cite ``[Source N]`` labels where applicable.

    Parameters
    ----------
    openai_key:
        OpenAI API key.
    user_message:
        The user's current question.
    context:
        Formatted string of retrieved NASA document chunks produced by
        :func:`nasa_rag.retrieval.format_context`.  Pass an empty string
        when no documents were retrieved; the model will state uncertainty.
    conversation_history:
        List of prior ``{"role": str, "content": str}`` dicts representing
        the multi-turn conversation. Only the most recent ``max_history_turns``
        user/assistant pairs are forwarded to avoid exceeding the context window.
    model:
        OpenAI chat model identifier (default: ``"gpt-4o-mini"``).
    max_history_turns:
        Maximum number of prior turns included in the prompt.
    temperature:
        Sampling temperature (default 0.1 for deterministic, faithful answers).
    max_tokens:
        Upper bound on completion length.

    Returns
    -------
    str
        The assistant's response text, or a descriptive error message when
        the API call fails.

    Raises
    ------
    This function never raises; all exceptions are caught, logged, and
    returned as human-readable error strings.
    """
    if not user_message or not user_message.strip():
        return "Please provide a question."

    # ── Build message list ────────────────────────────────────────────────
    messages: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # Inject retrieved context (or signal its absence)
    context_body = context.strip() if context else ""
    messages.append(
        {
            "role": "system",
            "content": (_CONTEXT_PREFIX + context_body) if context_body else _NO_CONTEXT_MESSAGE,
        }
    )

    # Append recent conversation history (user + assistant turns only)
    recent_turns = [
        msg
        for msg in conversation_history
        if msg.get("role") in ("user", "assistant")
    ][-max_history_turns:]
    messages.extend(recent_turns)

    # Current user turn
    messages.append({"role": "user", "content": user_message})

    # ── Call the API ──────────────────────────────────────────────────────
    try:
        client = OpenAI(api_key=openai_key)
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = completion.choices[0].message.content
        logger.debug("Generated response (%d chars) for: %.60s…", len(answer or ""), user_message)
        return answer or ""

    except OpenAIError as exc:
        logger.error("OpenAI API error: %s", exc)
        return (
            f"I encountered an error communicating with the OpenAI API: {exc}. "
            "Please check your API key and try again."
        )
    except Exception as exc:
        logger.error("Unexpected error in generate_response: %s", exc)
        return "An unexpected error occurred while generating a response. Please try again."
