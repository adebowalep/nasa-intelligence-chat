"""
Pytest configuration and shared fixtures for the NASA RAG test suite.

Design principles
-----------------
- All tests are fully offline: no real API calls, no real ChromaDB writes.
- External dependencies (OpenAI, ChromaDB) are mocked at the boundary.
- Fixtures are composable and minimal.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Environment isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Prevent any test from accidentally reading a real .env key.

    Sets a deterministic fake key so modules that call ``os.getenv`` at
    import time (e.g. config.py) get a predictable value.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-1234567890")
    monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("CHROMA_DIR", "./test_chroma_db")
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")


# ---------------------------------------------------------------------------
# ChromaDB mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_collection() -> MagicMock:
    """
    A MagicMock that quacks like a ChromaDB Collection.

    Pre-wired with sensible defaults so tests only need to override what
    they care about.
    """
    col = MagicMock()
    col.count.return_value = 42

    # Default query result: one chunk from Apollo 13
    col.query.return_value = {
        "ids": [["apollo_13_AS13_TEC_chunk_0001"]],
        "documents": [["The oxygen tank in Service Module exploded at 55:55 MET."]],
        "metadatas": [
            [
                {
                    "mission": "apollo_13",
                    "file_path": "data_text/apollo13/AS13_TEC_textract_full_text.txt",
                    "document_category": "technical",
                    "chunk_index": 1,
                    "total_chunks": 92,
                    "source": "AS13_TEC_textract_full_text",
                }
            ]
        ],
        "distances": [[0.12]],
    }

    col.get.return_value = {"ids": [], "metadatas": [], "documents": []}
    return col


@pytest.fixture
def mock_chroma_client(mock_collection: MagicMock) -> MagicMock:
    """A MagicMock ChromaDB PersistentClient pre-wired with mock_collection."""
    client = MagicMock()
    client.list_collections.return_value = [mock_collection]
    client.get_collection.return_value = mock_collection
    client.get_or_create_collection.return_value = mock_collection
    return client


# ---------------------------------------------------------------------------
# OpenAI mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_response() -> MagicMock:
    """Simulate a successful OpenAI chat completion response."""
    message = MagicMock()
    message.content = (
        "According to [Source 1] — Apollo 13 Technical Transcript — "
        "the oxygen tank No. 2 ruptured at 55:55 MET, causing a rapid "
        "pressure loss and electrical failure across multiple systems."
    )
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@pytest.fixture
def mock_openai_client(mock_openai_response: MagicMock) -> MagicMock:
    """A MagicMock OpenAI client with a stubbed chat.completions.create."""
    client = MagicMock()
    client.chat.completions.create.return_value = mock_openai_response
    return client


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_QUESTION = "What caused the Apollo 13 oxygen tank explosion?"

SAMPLE_ANSWER = (
    "According to [Source 1] — Apollo 13 Technical Transcript — "
    "the oxygen tank No. 2 ruptured at 55:55 MET."
)

SAMPLE_CONTEXTS = [
    "The oxygen tank in Service Module exploded at 55:55 MET.",
    "The crew moved to the Lunar Module Aquarius as a lifeboat.",
]

SAMPLE_METADATA = {
    "mission": "apollo_13",
    "file_path": "data_text/apollo13/AS13_TEC_textract_full_text.txt",
    "document_category": "technical",
    "chunk_index": 1,
    "total_chunks": 92,
    "source": "AS13_TEC_textract_full_text",
}
