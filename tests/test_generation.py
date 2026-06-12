"""
Tests for nasa_rag.generation — LLM response generation.

All OpenAI API calls are mocked; no real API key is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nasa_rag.generation import generate_response
from tests.conftest import SAMPLE_ANSWER, SAMPLE_CONTEXTS, SAMPLE_QUESTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion object returning *content*."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestGenerateResponseSuccess:
    def test_returns_string_on_success(self, mock_openai_client: MagicMock) -> None:
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            result = generate_response("sk-fake", SAMPLE_QUESTION, "context", [])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_expected_content(self, mock_openai_client: MagicMock) -> None:
        mock_openai_client.chat.completions.create.return_value = _make_completion(SAMPLE_ANSWER)
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            result = generate_response("sk-fake", SAMPLE_QUESTION, "context", [])
        assert result == SAMPLE_ANSWER

    def test_passes_correct_model(self, mock_openai_client: MagicMock) -> None:
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            generate_response("sk-fake", SAMPLE_QUESTION, "ctx", [], model="gpt-4o")
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_passes_temperature(self, mock_openai_client: MagicMock) -> None:
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            generate_response("sk-fake", SAMPLE_QUESTION, "ctx", [], temperature=0.5)
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5

    def test_includes_system_message(self, mock_openai_client: MagicMock) -> None:
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            generate_response("sk-fake", SAMPLE_QUESTION, "NASA context here", [])
        messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert roles[0] == "system"

    def test_includes_context_in_messages(self, mock_openai_client: MagicMock) -> None:
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            generate_response("sk-fake", SAMPLE_QUESTION, "Important NASA context", [])
        messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        all_content = " ".join(m["content"] for m in messages)
        assert "Important NASA context" in all_content

    def test_empty_context_sends_no_context_message(
        self, mock_openai_client: MagicMock
    ) -> None:
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            generate_response("sk-fake", SAMPLE_QUESTION, "", [])
        messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        all_content = " ".join(m["content"] for m in messages)
        assert "No relevant documents" in all_content

    def test_conversation_history_appended(self, mock_openai_client: MagicMock) -> None:
        history = [
            {"role": "user", "content": "Who was on Apollo 11?"},
            {"role": "assistant", "content": "Armstrong, Aldrin, Collins."},
        ]
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            generate_response("sk-fake", SAMPLE_QUESTION, "ctx", history)
        messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        contents = [m["content"] for m in messages]
        assert "Armstrong, Aldrin, Collins." in contents

    def test_history_capped_at_max_turns(self, mock_openai_client: MagicMock) -> None:
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(20)
        ]
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            generate_response("sk-fake", SAMPLE_QUESTION, "ctx", history, max_history_turns=4)
        messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        # 2 system + 4 history + 1 user = 7 max
        assert len(messages) <= 7

    def test_user_message_is_last(self, mock_openai_client: MagicMock) -> None:
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            generate_response("sk-fake", "My question?", "ctx", [])
        messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[-1]["content"] == "My question?"
        assert messages[-1]["role"] == "user"


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------

class TestGenerateResponseEdgeCases:
    def test_empty_user_message_returns_prompt_string(self) -> None:
        result = generate_response("sk-fake", "", "ctx", [])
        assert "provide a question" in result.lower()

    def test_whitespace_only_message_returns_prompt_string(self) -> None:
        result = generate_response("sk-fake", "   ", "ctx", [])
        assert "provide a question" in result.lower()

    def test_openai_error_returns_error_string(self, mock_openai_client: MagicMock) -> None:
        from openai import OpenAIError

        mock_openai_client.chat.completions.create.side_effect = OpenAIError("rate limit")
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            result = generate_response("sk-fake", SAMPLE_QUESTION, "ctx", [])
        assert "error" in result.lower()
        assert "OpenAI" in result or "API" in result

    def test_unexpected_exception_returns_error_string(
        self, mock_openai_client: MagicMock
    ) -> None:
        mock_openai_client.chat.completions.create.side_effect = RuntimeError("unexpected")
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            result = generate_response("sk-fake", SAMPLE_QUESTION, "ctx", [])
        assert "error" in result.lower()

    def test_none_content_returns_empty_string(self, mock_openai_client: MagicMock) -> None:
        mock_openai_client.chat.completions.create.return_value = _make_completion(None)
        with patch("nasa_rag.generation.OpenAI", return_value=mock_openai_client):
            result = generate_response("sk-fake", SAMPLE_QUESTION, "ctx", [])
        assert result == ""
