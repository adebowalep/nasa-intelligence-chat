"""
NASA Intelligence Chat System — Streamlit Application.

Run from the project root:
    streamlit run app/streamlit_app.py

Features
--------
- Mission filter: All / Apollo 11 / Apollo 13 / Challenger
- Configurable top-k retrieval slider
- Multi-turn conversation with persistent session history
- Retrieved source cards with file path, chunk index, and content preview
- Real-time RAGAS evaluation (ResponseRelevancy + Faithfulness)
- Graceful warnings when the API key or ChromaDB collection is missing
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project root importable regardless of working directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import streamlit as st

from nasa_rag.retrieval import (
    discover_chroma_backends,
    format_context,
    initialize_collection,
    retrieve_documents,
)
from nasa_rag.generation import generate_response
from nasa_rag.evaluation import evaluate_response, RAGAS_AVAILABLE

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASA Intelligence Chat",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Mission filter options ────────────────────────────────────────────────
_MISSION_OPTIONS: dict[str, str] = {
    "All Missions": "all",
    "Apollo 11": "apollo_11",
    "Apollo 13": "apollo_13",
    "Challenger (STS-51L)": "challenger",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session() -> None:
    defaults: dict = {
        "messages": [],
        "last_evaluation": None,
        "last_sources": [],
        "current_backend": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Source extraction helper
# ---------------------------------------------------------------------------

def _extract_sources(documents: list, metadatas: list) -> list[dict]:
    seen: set[str] = set()
    sources: list[dict] = []
    for doc, meta in zip(documents, metadatas):
        fingerprint = doc[:80].strip()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        sources.append(
            {
                "mission": meta.get("mission", "unknown").replace("_", " ").title(),
                "file_path": meta.get("file_path", meta.get("source", "?")),
                "category": (
                    meta.get("document_category", "document").replace("_", " ").title()
                ),
                "chunk_index": meta.get("chunk_index", "?"),
                "total_chunks": meta.get("total_chunks", "?"),
                "content": doc[:600] + ("…" if len(doc) > 600 else ""),
            }
        )
    return sources


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> dict:
    cfg: dict = {}

    with st.sidebar:
        st.title("⚙️ Settings")

        # ── API Key ──────────────────────────────────────────────────────
        st.subheader("🔑 OpenAI")
        default_key = os.getenv("OPENAI_API_KEY", "")
        openai_key = st.text_input(
            "API Key",
            value=default_key,
            type="password",
            help="Falls back to OPENAI_API_KEY environment variable.",
        )
        if not openai_key:
            st.warning("Enter your OpenAI API key to enable responses.")

        default_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        _models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo-preview"]
        model = st.selectbox(
            "Model",
            _models,
            index=_models.index(default_model) if default_model in _models else 0,
        )

        # ── ChromaDB ─────────────────────────────────────────────────────
        st.subheader("🗄️ Document Collection")
        with st.spinner("Scanning for ChromaDB collections…"):
            backends = discover_chroma_backends(_PROJECT_ROOT)

        if not backends:
            st.error(
                "No ChromaDB collections found.\n\n"
                "Build the index first:\n"
                "```\npython scripts/build_index.py --update-mode replace\n```"
            )
            cfg["ready"] = False
            cfg["openai_key"] = openai_key
            return cfg

        backend_labels = {k: v["display_name"] for k, v in backends.items()}
        selected_key = st.selectbox(
            "Collection",
            options=list(backend_labels.keys()),
            format_func=lambda x: backend_labels[x],
        )
        selected_backend = backends[selected_key]

        if st.session_state.current_backend != selected_key:
            st.session_state.current_backend = selected_key

        # ── Retrieval settings ───────────────────────────────────────────
        st.subheader("🔍 Retrieval")
        mission_label = st.selectbox("Mission filter", list(_MISSION_OPTIONS.keys()))
        top_k = st.slider("Top-k chunks", 1, 10, 5)

        # ── Evaluation ───────────────────────────────────────────────────
        st.subheader("📊 RAGAS Evaluation")
        enable_eval = st.checkbox("Enable evaluation", value=RAGAS_AVAILABLE)
        if not RAGAS_AVAILABLE:
            st.caption("Install RAGAS: `pip install ragas`")

        # ── Session control ──────────────────────────────────────────────
        st.divider()
        if st.button("🗑️ Clear history"):
            st.session_state.messages = []
            st.session_state.last_evaluation = None
            st.session_state.last_sources = []
            st.rerun()

    cfg.update(
        {
            "openai_key": openai_key,
            "model": model,
            "chroma_dir": selected_backend["directory"],
            "collection_name": selected_backend["collection_name"],
            "mission_filter": _MISSION_OPTIONS[mission_label],
            "top_k": top_k,
            "enable_eval": enable_eval,
            "ready": True,
        }
    )
    return cfg


# ---------------------------------------------------------------------------
# Source cards
# ---------------------------------------------------------------------------

def _render_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("No source documents retrieved.")
        return
    for i, src in enumerate(sources, start=1):
        with st.expander(
            f"📄 [Source {i}] {src['mission']} — {src['category']}", expanded=False
        ):
            st.caption(f"**File:** `{src['file_path']}`")
            st.caption(f"**Chunk:** {src['chunk_index']} / {src['total_chunks']}")
            st.markdown(
                f"<div style='background:#f0f2f6;padding:10px;border-radius:6px;"
                f"font-size:0.84rem;font-family:monospace;white-space:pre-wrap;'>"
                f"{src['content']}</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Evaluation display
# ---------------------------------------------------------------------------

def _render_eval(scores: dict) -> None:
    if not scores:
        return
    if "error" in scores:
        st.warning(f"RAGAS: {scores['error']}")
        return
    cols = st.columns(len(scores))
    for col, (metric, value) in zip(cols, scores.items()):
        col.metric(metric.replace("_", " ").title(), f"{value:.3f}")
        col.progress(float(value))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session()

    st.title("🚀 NASA Intelligence Chat")
    st.markdown(
        "Ask natural-language questions about **Apollo 11**, **Apollo 13**, "
        "and **Challenger (STS-51L)** — answers grounded in real NASA archive documents."
    )
    st.divider()

    cfg = _render_sidebar()
    if not cfg.get("ready"):
        st.stop()
    if not cfg["openai_key"]:
        st.error("Enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    collection, ok, err = initialize_collection(cfg["chroma_dir"], cfg["collection_name"])
    if not ok:
        st.error(
            f"Could not open collection **{cfg['collection_name']}**.\n\n"
            f"`{err}`\n\n"
            "Run `python scripts/build_index.py --update-mode replace` first."
        )
        st.stop()
    if collection.count() == 0:
        st.warning("The ChromaDB collection is empty. Run the embedding pipeline.")

    # ── Layout ───────────────────────────────────────────────────────────
    chat_col, info_col = st.columns([2, 1])

    with chat_col:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── Chat input ───────────────────────────────────────────────────────
    prompt = st.chat_input("Ask about NASA missions…")
    if not prompt:
        with info_col:
            st.subheader("📄 Retrieved Sources")
            _render_sources(st.session_state.last_sources)
            if st.session_state.last_evaluation and cfg["enable_eval"]:
                st.divider()
                st.subheader("📊 RAGAS Evaluation")
                _render_eval(st.session_state.last_evaluation)
        return

    # Record user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_col:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Retrieve
    with st.spinner("🔍 Searching NASA archives…"):
        results = retrieve_documents(
            collection, prompt, n_results=cfg["top_k"], mission_filter=cfg["mission_filter"]
        )

    contexts: list[str] = []
    context_str = ""
    sources: list[dict] = []

    if results and results.get("documents") and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        context_str = format_context(docs, metas)
        contexts = docs
        sources = _extract_sources(docs, metas)

    st.session_state.last_sources = sources

    # Generate
    with st.spinner("🤖 Generating answer…"):
        answer = generate_response(
            openai_key=cfg["openai_key"],
            user_message=prompt,
            context=context_str,
            conversation_history=st.session_state.messages[:-1],
            model=cfg["model"],
        )

    with chat_col:
        with st.chat_message("assistant"):
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Evaluate
    if cfg["enable_eval"] and RAGAS_AVAILABLE:
        with st.spinner("📊 Evaluating with RAGAS…"):
            eval_scores = evaluate_response(
                question=prompt,
                answer=answer,
                contexts=contexts,
                openai_key=cfg["openai_key"],
            )
        st.session_state.last_evaluation = eval_scores

    st.rerun()


if __name__ == "__main__":
    main()
