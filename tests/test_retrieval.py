"""
Tests for nasa_rag.retrieval — ChromaDB retrieval and context formatting.

All ChromaDB interactions are mocked; no real database is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nasa_rag.retrieval import (
    discover_chroma_backends,
    format_context,
    initialize_collection,
    retrieve_documents,
)
from tests.conftest import SAMPLE_CONTEXTS, SAMPLE_METADATA


# ---------------------------------------------------------------------------
# discover_chroma_backends
# ---------------------------------------------------------------------------

class TestDiscoverChromaBackends:
    def test_returns_empty_dict_for_missing_dir(self, tmp_path: Path) -> None:
        result = discover_chroma_backends(tmp_path / "nonexistent")
        assert result == {}

    def test_ignores_directories_without_chroma_in_name(self, tmp_path: Path) -> None:
        (tmp_path / "my_database").mkdir()
        result = discover_chroma_backends(tmp_path)
        assert result == {}

    def test_discovers_directory_with_chroma_in_name(self, tmp_path: Path) -> None:
        chroma_dir = tmp_path / "chroma_db"
        chroma_dir.mkdir()

        mock_col = MagicMock()
        mock_col.name = "nasa_missions"
        mock_col.count.return_value = 100

        mock_client = MagicMock()
        mock_client.list_collections.return_value = [mock_col]

        with patch("nasa_rag.retrieval.chromadb.PersistentClient", return_value=mock_client):
            result = discover_chroma_backends(tmp_path)

        assert len(result) == 1
        key = "chroma_db::nasa_missions"
        assert key in result
        assert result[key]["collection_name"] == "nasa_missions"
        assert result[key]["doc_count"] == "100"

    def test_handles_chromadb_connection_error_gracefully(self, tmp_path: Path) -> None:
        chroma_dir = tmp_path / "chroma_db"
        chroma_dir.mkdir()

        with patch(
            "nasa_rag.retrieval.chromadb.PersistentClient",
            side_effect=Exception("connection refused"),
        ):
            result = discover_chroma_backends(tmp_path)

        assert "chroma_db::error" in result
        assert "error" in result["chroma_db::error"]["collection_name"]


# ---------------------------------------------------------------------------
# initialize_collection
# ---------------------------------------------------------------------------

class TestInitializeCollection:
    def test_success_returns_collection_true_empty_error(
        self, mock_chroma_client: MagicMock, mock_collection: MagicMock
    ) -> None:
        with patch(
            "nasa_rag.retrieval.chromadb.PersistentClient",
            return_value=mock_chroma_client,
        ):
            col, ok, err = initialize_collection("./chroma_db", "nasa_missions")

        assert ok is True
        assert err == ""
        assert col is mock_collection

    def test_failure_returns_none_false_message(self) -> None:
        with patch(
            "nasa_rag.retrieval.chromadb.PersistentClient",
            side_effect=Exception("collection not found"),
        ):
            col, ok, err = initialize_collection("./bad_path", "missing")

        assert ok is False
        assert col is None
        assert "collection not found" in err


# ---------------------------------------------------------------------------
# retrieve_documents
# ---------------------------------------------------------------------------

class TestRetrieveDocuments:
    def test_returns_results_on_success(self, mock_collection: MagicMock) -> None:
        result = retrieve_documents(mock_collection, "What happened on Apollo 13?")
        assert result is not None
        assert "documents" in result
        assert len(result["documents"][0]) > 0

    def test_returns_none_for_empty_query(self, mock_collection: MagicMock) -> None:
        result = retrieve_documents(mock_collection, "")
        assert result is None

    def test_returns_none_for_whitespace_query(self, mock_collection: MagicMock) -> None:
        result = retrieve_documents(mock_collection, "   ")
        assert result is None

    def test_applies_mission_filter_apollo11(self, mock_collection: MagicMock) -> None:
        retrieve_documents(mock_collection, "moon landing", mission_filter="apollo 11")
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"mission": {"$eq": "apollo_11"}}

    def test_applies_mission_filter_challenger(self, mock_collection: MagicMock) -> None:
        retrieve_documents(mock_collection, "launch", mission_filter="challenger")
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"mission": {"$eq": "challenger"}}

    def test_no_filter_for_all_missions(self, mock_collection: MagicMock) -> None:
        retrieve_documents(mock_collection, "crew", mission_filter="all")
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs.get("where") is None

    def test_no_filter_when_mission_filter_is_none(self, mock_collection: MagicMock) -> None:
        retrieve_documents(mock_collection, "crew", mission_filter=None)
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs.get("where") is None

    def test_respects_n_results_parameter(self, mock_collection: MagicMock) -> None:
        retrieve_documents(mock_collection, "oxygen tank", n_results=7)
        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["n_results"] == 7

    def test_returns_none_on_chromadb_error(self, mock_collection: MagicMock) -> None:
        mock_collection.query.side_effect = Exception("DB error")
        result = retrieve_documents(mock_collection, "question")
        assert result is None


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_returns_empty_string_for_empty_documents(self) -> None:
        assert format_context([], []) == ""

    def test_contains_source_labels(self) -> None:
        ctx = format_context(SAMPLE_CONTEXTS, [SAMPLE_METADATA, SAMPLE_METADATA])
        assert "[Source 1]" in ctx
        assert "[Source 2]" in ctx

    def test_contains_mission_name(self) -> None:
        ctx = format_context([SAMPLE_CONTEXTS[0]], [SAMPLE_METADATA])
        assert "Apollo 13" in ctx

    def test_contains_file_path(self) -> None:
        ctx = format_context([SAMPLE_CONTEXTS[0]], [SAMPLE_METADATA])
        assert "AS13_TEC" in ctx

    def test_deduplicates_identical_chunks(self) -> None:
        # Same chunk twice → only one [Source N] label
        ctx = format_context([SAMPLE_CONTEXTS[0], SAMPLE_CONTEXTS[0]], [SAMPLE_METADATA, SAMPLE_METADATA])
        assert "[Source 1]" in ctx
        assert "[Source 2]" not in ctx

    def test_truncates_long_chunks(self) -> None:
        long_doc = "A" * 5000
        ctx = format_context([long_doc], [SAMPLE_METADATA], max_chunk_chars=500)
        assert "[truncated]" in ctx

    def test_header_present(self) -> None:
        ctx = format_context([SAMPLE_CONTEXTS[0]], [SAMPLE_METADATA])
        assert "RETRIEVED NASA MISSION CONTEXT" in ctx

    def test_mission_displayed_in_title_case(self) -> None:
        meta = {**SAMPLE_METADATA, "mission": "apollo_13"}
        ctx = format_context([SAMPLE_CONTEXTS[0]], [meta])
        assert "Apollo 13" in ctx
        assert "apollo_13" not in ctx
