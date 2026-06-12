"""
Tests for nasa_rag.evaluation — RAGAS response-quality evaluation.

RAGAS and OpenAI are mocked throughout; no real API calls are made.
"""

from __future__ import annotations

import concurrent.futures
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import SAMPLE_ANSWER, SAMPLE_CONTEXTS, SAMPLE_QUESTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ragas_result(
    relevancy: float = 0.85, faithfulness: float = 0.90
) -> MagicMock:
    """Build a mock RAGAS Result object with a stubbed to_pandas()."""
    import pandas as pd

    result = MagicMock()
    df = pd.DataFrame(
        {
            "response_relevancy": [relevancy],
            "faithfulness": [faithfulness],
        }
    )
    result.to_pandas.return_value = df
    return result


# ---------------------------------------------------------------------------
# Tests — input validation
# ---------------------------------------------------------------------------

class TestEvaluateResponseValidation:
    """Validation guards must return error dicts, never raise."""

    def test_empty_question_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        from nasa_rag.evaluation import evaluate_response

        result = evaluate_response("", SAMPLE_ANSWER, SAMPLE_CONTEXTS, "sk-test")
        assert "error" in result

    def test_whitespace_question_returns_error(self) -> None:
        from nasa_rag.evaluation import evaluate_response

        result = evaluate_response("  ", SAMPLE_ANSWER, SAMPLE_CONTEXTS, "sk-test")
        assert "error" in result

    def test_empty_answer_returns_error(self) -> None:
        from nasa_rag.evaluation import evaluate_response

        result = evaluate_response(SAMPLE_QUESTION, "", SAMPLE_CONTEXTS, "sk-test")
        assert "error" in result

    def test_missing_api_key_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("CHROMA_OPENAI_API_KEY", raising=False)
        from nasa_rag.evaluation import evaluate_response

        result = evaluate_response(SAMPLE_QUESTION, SAMPLE_ANSWER, SAMPLE_CONTEXTS, openai_key=None)
        assert "error" in result

    def test_empty_contexts_replaced_with_placeholder(self) -> None:
        """Empty contexts list should not crash — replaced by placeholder."""
        from nasa_rag import evaluation as eval_mod

        if not eval_mod.RAGAS_AVAILABLE:
            pytest.skip("RAGAS not installed")

        with patch.object(eval_mod, "_run_ragas_in_thread", return_value={"response_relevancy": 0.8, "faithfulness": 0.9}):
            with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
                future = MagicMock()
                future.result.return_value = {"response_relevancy": 0.8, "faithfulness": 0.9}
                mock_executor.return_value.__enter__.return_value.submit.return_value = future

                result = eval_mod.evaluate_response(
                    SAMPLE_QUESTION, SAMPLE_ANSWER, [], "sk-test"
                )
        # Should not error about empty contexts
        assert "error" not in result or "contexts" not in result.get("error", "")


# ---------------------------------------------------------------------------
# Tests — RAGAS not installed
# ---------------------------------------------------------------------------

class TestRagasNotAvailable:
    def test_returns_error_when_ragas_missing(self) -> None:
        from nasa_rag import evaluation as eval_mod

        original = eval_mod.RAGAS_AVAILABLE
        try:
            eval_mod.RAGAS_AVAILABLE = False
            result = eval_mod.evaluate_response(
                SAMPLE_QUESTION, SAMPLE_ANSWER, SAMPLE_CONTEXTS, "sk-test"
            )
            assert "error" in result
            assert "RAGAS" in result["error"] or "not installed" in result["error"]
        finally:
            eval_mod.RAGAS_AVAILABLE = original


# ---------------------------------------------------------------------------
# Tests — happy path (RAGAS mocked)
# ---------------------------------------------------------------------------

class TestEvaluateResponseSuccess:
    """Test the evaluate_response public function with RAGAS fully mocked."""

    def test_returns_float_scores(self) -> None:
        from nasa_rag import evaluation as eval_mod

        if not eval_mod.RAGAS_AVAILABLE:
            pytest.skip("RAGAS not installed")

        expected = {"response_relevancy": 0.85, "faithfulness": 0.90}

        with patch.object(eval_mod, "_run_ragas_in_thread", return_value=expected):
            with patch("concurrent.futures.ThreadPoolExecutor") as mock_exec:
                future = MagicMock()
                future.result.return_value = expected
                mock_exec.return_value.__enter__.return_value.submit.return_value = future

                result = eval_mod.evaluate_response(
                    SAMPLE_QUESTION, SAMPLE_ANSWER, SAMPLE_CONTEXTS, "sk-test"
                )

        assert "error" not in result
        assert result.get("response_relevancy") == pytest.approx(0.85, abs=0.01)
        assert result.get("faithfulness") == pytest.approx(0.90, abs=0.01)

    def test_timeout_returns_error(self) -> None:
        from nasa_rag import evaluation as eval_mod

        if not eval_mod.RAGAS_AVAILABLE:
            pytest.skip("RAGAS not installed")

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_exec:
            future = MagicMock()
            future.result.side_effect = concurrent.futures.TimeoutError()
            mock_exec.return_value.__enter__.return_value.submit.return_value = future

            result = eval_mod.evaluate_response(
                SAMPLE_QUESTION, SAMPLE_ANSWER, SAMPLE_CONTEXTS, "sk-test"
            )

        assert "error" in result
        assert "timed out" in result["error"].lower()

    def test_api_key_fallback_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When openai_key=None, the function should read from environment."""
        from nasa_rag import evaluation as eval_mod

        if not eval_mod.RAGAS_AVAILABLE:
            pytest.skip("RAGAS not installed")

        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        expected = {"response_relevancy": 0.75, "faithfulness": 0.80}

        with patch("concurrent.futures.ThreadPoolExecutor") as mock_exec:
            future = MagicMock()
            future.result.return_value = expected
            mock_exec.return_value.__enter__.return_value.submit.return_value = future

            result = eval_mod.evaluate_response(
                SAMPLE_QUESTION, SAMPLE_ANSWER, SAMPLE_CONTEXTS, openai_key=None
            )

        assert "error" not in result
