# System Architecture

## Overview

The NASA Intelligence Chat System is a production-quality Retrieval-Augmented Generation (RAG) pipeline. It transforms 12 raw NASA mission documents into a searchable vector store and uses a grounded LLM to answer natural-language questions with source citations.

## Component Diagram

```mermaid
flowchart TB
    subgraph Ingestion ["📥 Ingestion (one-time)"]
        A[NASA Text Files\ndata_text/] --> B[EmbeddingPipeline\nsrc/nasa_rag/embedding.py]
        B -->|OpenAI text-embedding-3-small| C[(ChromaDB\nchroma_db/)]
    end

    subgraph Query ["🔍 Query Pipeline (per request)"]
        D([User Question]) --> E[Streamlit UI\napp/streamlit_app.py]
        E --> F[retrieve_documents\nsrc/nasa_rag/retrieval.py]
        F -->|top-k semantic search| C
        C -->|chunks + metadata| G[format_context]
        G --> H[generate_response\nsrc/nasa_rag/generation.py]
        H -->|OpenAI GPT| I([Grounded Answer\nwith [Source N] citations])
        I --> E
    end

    subgraph Evaluation ["📊 Evaluation"]
        I --> J[evaluate_response\nsrc/nasa_rag/evaluation.py]
        J -->|RAGAS ResponseRelevancy\n+ Faithfulness| K([Quality Scores])
        K --> E
    end
```

## Data Flow

### 1. Ingestion (run once)

```
data_text/
├── apollo11/   (6 documents)
├── apollo13/   (3 documents)
└── challenger/ (3 documents)
         ↓
    EmbeddingPipeline.process_all()
         ↓ chunk_text() → 1000-char chunks, 150-char overlap
         ↓ OpenAI text-embedding-3-small
         ↓ ChromaDB.add()
    chroma_db/ (≈4 200 chunks, cosine similarity index)
```

### 2. Query

```
User question
    → ChromaDB.query(top_k=5)   — cosine similarity search
    → format_context()           — [Source N] labelled context string
    → OpenAI GPT (chat)          — grounded answer + citations
    → Streamlit UI               — rendered markdown
```

### 3. Evaluation (optional, per response)

```
(question, answer, contexts)
    → RAGAS SingleTurnSample
    → ResponseRelevancy (LLM + embeddings)
    → Faithfulness (LLM)
    → float scores 0.0 – 1.0
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Centralised settings from env vars; single source of truth for all constants |
| `embedding.py` | Document ingestion: chunking, metadata extraction, ChromaDB batch insert |
| `retrieval.py` | ChromaDB discovery, collection init, semantic search, context formatting |
| `generation.py` | OpenAI chat client; injects retrieved context; manages conversation history |
| `evaluation.py` | RAGAS wrapper; uvloop-safe thread isolation for async evaluation |

## Key Design Decisions

### Chunking strategy
Chunks break at sentence boundaries (`.`) first, then newlines, before hard-cutting at `chunk_size`. This preserves semantic coherence at chunk boundaries.

### uvloop isolation
RAGAS uses asyncio internally. Streamlit runs on uvloop, which `nest_asyncio` cannot patch. The solution: RAGAS evaluation runs in a dedicated thread with a fresh standard asyncio event loop — completely isolated from the Streamlit event loop.

### Context deduplication
`format_context()` deduplicates chunks by their first 120 characters before labelling them `[Source N]`. This prevents the LLM from citing the same chunk under multiple source numbers.

### Graceful degradation
Every public function returns a typed result and never raises. Errors are logged and communicated via return-value conventions (e.g. `evaluate_response` returns `{"error": str}` instead of raising).

## Performance Characteristics

| Operation | Typical latency | Notes |
|---|---|---|
| Embedding ingestion | ~5 min for 12 files | One-time, cached in ChromaDB |
| Semantic retrieval | < 200 ms | ChromaDB HNSW index, in-process |
| LLM generation | 2–8 s | Depends on model and response length |
| RAGAS evaluation | 15–45 s | 2–3 additional OpenAI calls |

## Document Corpus

| Mission | Files | Chunk count | Document types |
|---|---|---|---|
| Apollo 11 | 6 | ~2 100 | Transcripts (CM, PAO, TEC), flight plan, NTRS reports |
| Apollo 13 | 3 | ~1 800 | Transcripts (CM, PAO, TEC) |
| Challenger | 3 | ~320 | Mission audio transcripts (107, 108, 109) |
| **Total** | **12** | **~4 220** | |
