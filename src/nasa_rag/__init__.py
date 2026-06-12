"""
nasa_rag — NASA Mission Intelligence RAG System.

A production-quality Retrieval-Augmented Generation system for querying
historic NASA mission documents (Apollo 11, Apollo 13, Challenger STS-51L).

Sub-modules
-----------
config      Centralised environment-based configuration.
embedding   ChromaDB embedding pipeline for NASA document ingestion.
retrieval   Semantic retrieval client (ChromaDB).
generation  OpenAI LLM client with grounded, citation-aware responses.
evaluation  RAGAS response-quality evaluation (relevancy + faithfulness).
"""

from nasa_rag.config import Settings, get_settings  # noqa: F401

__version__ = "1.0.0"
__author__ = "Ade"
__all__ = ["Settings", "get_settings"]
