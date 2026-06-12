"""
Tests for nasa_rag.embedding — chunking, metadata extraction, and pipeline logic.

ChromaDB and OpenAI are mocked throughout.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nasa_rag.embedding import (
    EmbeddingPipeline,
    extract_document_category,
    extract_data_type,
    extract_mission,
)


# ---------------------------------------------------------------------------
# Module-level extraction helpers (pure functions — no mocking needed)
# ---------------------------------------------------------------------------

class TestExtractMission:
    def test_apollo11_from_path(self, tmp_path: Path) -> None:
        fp = tmp_path / "apollo11" / "transcript.txt"
        assert extract_mission(fp) == "apollo_11"

    def test_apollo13_from_path(self, tmp_path: Path) -> None:
        fp = tmp_path / "apollo13" / "AS13_TEC.txt"
        assert extract_mission(fp) == "apollo_13"

    def test_challenger_from_path(self, tmp_path: Path) -> None:
        fp = tmp_path / "challenger" / "audio.txt"
        assert extract_mission(fp) == "challenger"

    def test_unknown_for_unrecognised_path(self, tmp_path: Path) -> None:
        fp = tmp_path / "gemini" / "doc.txt"
        assert extract_mission(fp) == "unknown"

    def test_case_insensitive_match(self, tmp_path: Path) -> None:
        fp = tmp_path / "Apollo11" / "file.txt"
        # Path str is lowercased before matching
        assert extract_mission(fp) == "apollo_11"


class TestExtractDataType:
    def test_transcript_in_path(self, tmp_path: Path) -> None:
        assert extract_data_type(tmp_path / "a11transcript_tec.txt") == "transcript"

    def test_textract_in_path(self, tmp_path: Path) -> None:
        assert extract_data_type(tmp_path / "file_textract_full.txt") == "textract_extracted"

    def test_audio_in_path(self, tmp_path: Path) -> None:
        assert extract_data_type(tmp_path / "mission_audio_log.txt") == "audio_transcript"

    def test_flight_plan_in_path(self, tmp_path: Path) -> None:
        assert extract_data_type(tmp_path / "Apollo_11_Flight_Plan.txt") == "flight_plan"

    def test_default_document(self, tmp_path: Path) -> None:
        assert extract_data_type(tmp_path / "unknown_file.txt") == "document"


class TestExtractDocumentCategory:
    def test_pao_filename(self) -> None:
        assert extract_document_category("AS13_PAO_textract.txt") == "public_affairs_officer"

    def test_cm_filename(self) -> None:
        assert extract_document_category("a11transscript_cm.txt") == "command_module"

    def test_tec_filename(self) -> None:
        assert extract_document_category("AS13_TEC_textract.txt") == "technical"

    def test_flight_plan_filename(self) -> None:
        assert extract_document_category("Apollo_11_Flight_Plan.txt") == "flight_plan"

    def test_ntrs_filename(self) -> None:
        assert extract_document_category("NASA_NTRS_Archive.txt") == "nasa_archive"

    def test_full_text_filename(self) -> None:
        assert extract_document_category("something_full_text.txt") == "complete_document"

    def test_fallback(self) -> None:
        assert extract_document_category("mystery_file.txt") == "general_document"


# ---------------------------------------------------------------------------
# EmbeddingPipeline — chunk_text (no ChromaDB needed)
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(mock_chroma_client: MagicMock) -> EmbeddingPipeline:
    """EmbeddingPipeline with ChromaDB and OpenAI embedding function mocked out."""
    with (
        patch("nasa_rag.embedding.chromadb.PersistentClient", return_value=mock_chroma_client),
        patch("nasa_rag.embedding.OpenAIEmbeddingFunction", return_value=MagicMock()),
    ):
        return EmbeddingPipeline(
            openai_api_key="sk-fake",
            chroma_dir="./test_chroma",
            collection_name="test_col",
            chunk_size=500,
            chunk_overlap=50,
        )


class TestChunkText:
    def test_short_text_returns_single_chunk(self, pipeline: EmbeddingPipeline) -> None:
        text = "Short document."
        chunks = pipeline.chunk_text(text, {"source": "test"})
        assert len(chunks) == 1
        chunk_text, meta = chunks[0]
        assert chunk_text == text
        assert meta["chunk_index"] == 0
        assert meta["total_chunks"] == 1

    def test_long_text_produces_multiple_chunks(self, pipeline: EmbeddingPipeline) -> None:
        text = "Sentence number X. " * 200  # Well above chunk_size=500
        chunks = pipeline.chunk_text(text, {"source": "long_doc"})
        assert len(chunks) > 1

    def test_chunks_carry_base_metadata(self, pipeline: EmbeddingPipeline) -> None:
        text = "Word " * 300
        base = {"source": "doc", "mission": "apollo_13"}
        chunks = pipeline.chunk_text(text, base)
        for _, meta in chunks:
            assert meta["source"] == "doc"
            assert meta["mission"] == "apollo_13"

    def test_chunk_index_is_sequential(self, pipeline: EmbeddingPipeline) -> None:
        text = "Word " * 300
        chunks = pipeline.chunk_text(text, {"source": "x"})
        indices = [meta["chunk_index"] for _, meta in chunks]
        assert indices == list(range(len(indices)))

    def test_total_chunks_consistent(self, pipeline: EmbeddingPipeline) -> None:
        text = "Word " * 300
        chunks = pipeline.chunk_text(text, {"source": "x"})
        expected_total = len(chunks)
        for _, meta in chunks:
            assert meta["total_chunks"] == expected_total

    def test_no_chunk_exceeds_chunk_size_by_much(self, pipeline: EmbeddingPipeline) -> None:
        text = "A" * 2000
        chunks = pipeline.chunk_text(text, {"source": "x"})
        # Some chunks may be up to chunk_size + a few chars at sentence breaks,
        # but nothing wild — allow a 20 % buffer
        for chunk_text, _ in chunks:
            assert len(chunk_text) <= pipeline.chunk_size * 1.2

    def test_empty_text_returns_no_chunks(self, pipeline: EmbeddingPipeline) -> None:
        chunks = pipeline.chunk_text("", {"source": "x"})
        # Either empty list or one empty-string chunk — both acceptable
        assert chunks == [] or all(t.strip() == "" for t, _ in chunks)


# ---------------------------------------------------------------------------
# EmbeddingPipeline — process_text_file
# ---------------------------------------------------------------------------

class TestProcessTextFile:
    def test_processes_valid_file(self, pipeline: EmbeddingPipeline, tmp_path: Path) -> None:
        sample = tmp_path / "apollo13" / "AS13_TEC_textract_full_text.txt"
        sample.parent.mkdir(parents=True)
        sample.write_text("A" * 600, encoding="utf-8")  # Triggers at least 2 chunks
        chunks = pipeline.process_text_file(sample)
        assert len(chunks) >= 1

    def test_empty_file_returns_empty_list(
        self, pipeline: EmbeddingPipeline, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        chunks = pipeline.process_text_file(empty)
        assert chunks == []

    def test_metadata_includes_mission(
        self, pipeline: EmbeddingPipeline, tmp_path: Path
    ) -> None:
        sample = tmp_path / "apollo11" / "a11transcript_tec.txt"
        sample.parent.mkdir()
        sample.write_text("Some content here.", encoding="utf-8")
        chunks = pipeline.process_text_file(sample)
        assert chunks[0][1]["mission"] == "apollo_11"

    def test_metadata_includes_source(
        self, pipeline: EmbeddingPipeline, tmp_path: Path
    ) -> None:
        sample = tmp_path / "challenger" / "mission_audio.txt"
        sample.parent.mkdir()
        sample.write_text("Audio content.", encoding="utf-8")
        chunks = pipeline.process_text_file(sample)
        assert chunks[0][1]["source"] == "mission_audio"

    def test_nonexistent_file_returns_empty_list(
        self, pipeline: EmbeddingPipeline, tmp_path: Path
    ) -> None:
        result = pipeline.process_text_file(tmp_path / "ghost.txt")
        assert result == []


# ---------------------------------------------------------------------------
# EmbeddingPipeline — scan_text_files
# ---------------------------------------------------------------------------

class TestScanTextFiles:
    def test_scans_known_mission_dirs(
        self, pipeline: EmbeddingPipeline, tmp_path: Path
    ) -> None:
        for mission_dir in ("apollo11", "apollo13", "challenger"):
            d = tmp_path / mission_dir
            d.mkdir()
            (d / "doc.txt").write_text("content")

        files = pipeline.scan_text_files(tmp_path)
        assert len(files) == 3

    def test_ignores_non_txt_files(
        self, pipeline: EmbeddingPipeline, tmp_path: Path
    ) -> None:
        d = tmp_path / "apollo11"
        d.mkdir()
        (d / "doc.txt").write_text("text")
        (d / "image.png").write_bytes(b"\x89PNG")

        files = pipeline.scan_text_files(tmp_path)
        assert all(fp.suffix == ".txt" for fp in files)

    def test_warns_for_missing_mission_dir(
        self, pipeline: EmbeddingPipeline, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Only apollo11 exists
        (tmp_path / "apollo11").mkdir()
        (tmp_path / "apollo11" / "doc.txt").write_text("content")

        with caplog.at_level("WARNING"):
            pipeline.scan_text_files(tmp_path)

        # Should warn about missing apollo13 and challenger
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("apollo13" in w.message or "challenger" in w.message for w in warnings)
