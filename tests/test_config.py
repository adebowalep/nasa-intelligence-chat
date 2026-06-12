"""
Tests for nasa_rag.config — Settings, validation, and MISSION_NORMALISE.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from nasa_rag.config import (
    KNOWN_MISSIONS,
    MISSION_DISPLAY,
    MISSION_NORMALISE,
    Settings,
    get_settings,
)


class TestSettings:
    def test_reads_openai_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abc")
        s = Settings()
        assert s.openai_api_key == "sk-test-abc"

    def test_reads_chat_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_CHAT_MODEL", "gpt-4o")
        s = Settings()
        assert s.chat_model == "gpt-4o"

    def test_defaults_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_CHAT_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("CHROMA_DIR", raising=False)
        monkeypatch.delenv("COLLECTION_NAME", raising=False)
        s = Settings(openai_api_key="sk-fake")
        assert s.chat_model == "gpt-4o-mini"
        assert s.embedding_model == "text-embedding-3-small"
        assert s.chroma_dir == "./chroma_db"
        assert s.collection_name == "nasa_missions"

    def test_is_configured_true_with_key(self) -> None:
        s = Settings(openai_api_key="sk-anything")
        assert s.is_configured() is True

    def test_is_configured_false_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(openai_api_key="")
        assert s.is_configured() is False

    def test_validate_raises_when_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(openai_api_key="")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            s.validate()

    def test_validate_raises_when_overlap_ge_chunk(self) -> None:
        s = Settings(openai_api_key="sk-x", chunk_size=500, chunk_overlap=500)
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            s.validate()

    def test_validate_passes_with_valid_settings(self) -> None:
        s = Settings(openai_api_key="sk-x", chunk_size=1000, chunk_overlap=150)
        s.validate()  # should not raise

    def test_settings_are_immutable(self) -> None:
        s = Settings(openai_api_key="sk-x")
        with pytest.raises((AttributeError, TypeError)):
            s.chat_model = "gpt-4"  # type: ignore[misc]

    def test_chunk_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "2000")
        monkeypatch.setenv("CHUNK_OVERLAP", "200")
        s = Settings()
        assert s.chunk_size == 2000
        assert s.chunk_overlap == 200


class TestGetSettings:
    def test_returns_settings_instance(self) -> None:
        # Clear cache before testing singleton behaviour
        get_settings.cache_clear()
        cfg = get_settings()
        assert isinstance(cfg, Settings)

    def test_singleton_same_object(self) -> None:
        get_settings.cache_clear()
        cfg1 = get_settings()
        cfg2 = get_settings()
        assert cfg1 is cfg2


class TestMissionConstants:
    def test_apollo11_variants_normalise(self) -> None:
        for variant in ("apollo 11", "apollo11", "apollo_11"):
            assert MISSION_NORMALISE[variant] == "apollo_11"

    def test_apollo13_variants_normalise(self) -> None:
        for variant in ("apollo 13", "apollo13", "apollo_13"):
            assert MISSION_NORMALISE[variant] == "apollo_13"

    def test_challenger_variants_normalise(self) -> None:
        for variant in ("challenger", "sts-51l"):
            assert MISSION_NORMALISE[variant] == "challenger"

    def test_known_missions_complete(self) -> None:
        assert set(KNOWN_MISSIONS) == {"apollo_11", "apollo_13", "challenger"}

    def test_display_names_exist_for_all_known_missions(self) -> None:
        for mission in KNOWN_MISSIONS:
            assert mission in MISSION_DISPLAY
