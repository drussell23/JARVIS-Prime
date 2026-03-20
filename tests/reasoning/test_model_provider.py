"""
Tests for ModelProvider protocol and implementations.

TDD: these tests are written BEFORE the implementation exists.
Covers MockModelProvider (test double) and LlamaCppModelProvider (production wrapper).
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from jarvis_prime.reasoning.model_provider import (
    LlamaCppModelProvider,
    MockModelProvider,
    ModelProvider,
)


# ---------------------------------------------------------------------------
# TestMockModelProvider
# ---------------------------------------------------------------------------


class TestMockModelProvider:
    def test_mock_returns_canned_response(self):
        provider = MockModelProvider(response="test")
        result = asyncio.get_event_loop().run_until_complete(
            provider.infer([{"role": "user", "content": "hello"}])
        )
        assert result == {"content": "test"}

    def test_mock_failure_mode(self):
        provider = MockModelProvider(should_fail=True)
        with pytest.raises(RuntimeError, match="Mock model failure"):
            asyncio.get_event_loop().run_until_complete(
                provider.infer([{"role": "user", "content": "hi"}])
            )

    def test_mock_is_model_loaded(self):
        assert MockModelProvider().is_model_loaded() is True

    def test_mock_is_model_loaded_false_when_fail(self):
        assert MockModelProvider(should_fail=True).is_model_loaded() is False

    def test_mock_loaded_model_name(self):
        assert MockModelProvider().loaded_model_name() == "mock-model"

    def test_mock_loaded_model_name_when_fail(self):
        assert MockModelProvider(should_fail=True).loaded_model_name() == ""

    def test_mock_satisfies_protocol(self):
        assert isinstance(MockModelProvider(), ModelProvider)

    def test_mock_tracks_calls(self):
        provider = MockModelProvider(response="hello")
        messages = [{"role": "user", "content": "ping"}]
        asyncio.get_event_loop().run_until_complete(provider.infer(messages))
        asyncio.get_event_loop().run_until_complete(provider.infer(messages))
        assert provider.call_count == 2
        assert provider.last_messages == messages

    def test_mock_tracks_calls_starts_at_zero(self):
        provider = MockModelProvider()
        assert provider.call_count == 0
        assert provider.last_messages is None

    def test_mock_default_response(self):
        provider = MockModelProvider()
        result = asyncio.get_event_loop().run_until_complete(
            provider.infer([{"role": "user", "content": "?"}])
        )
        assert "content" in result
        assert isinstance(result["content"], str)


# ---------------------------------------------------------------------------
# TestLlamaCppModelProvider
# ---------------------------------------------------------------------------


class TestLlamaCppModelProvider:
    def test_no_model_raises(self):
        """get_model_fn returns None → RuntimeError('No model loaded')."""
        provider = LlamaCppModelProvider(get_model_fn=lambda: None)
        with pytest.raises(RuntimeError, match="No model loaded"):
            asyncio.get_event_loop().run_until_complete(
                provider.infer([{"role": "user", "content": "hi"}])
            )

    def test_is_model_loaded_false_when_none(self):
        provider = LlamaCppModelProvider(get_model_fn=lambda: None)
        assert provider.is_model_loaded() is False

    def test_is_model_loaded_true_when_present(self):
        fake_model = MagicMock()
        provider = LlamaCppModelProvider(get_model_fn=lambda: fake_model)
        assert provider.is_model_loaded() is True

    def test_satisfies_protocol(self):
        provider = LlamaCppModelProvider(get_model_fn=lambda: None)
        assert isinstance(provider, ModelProvider)

    def test_timeout_raises(self):
        """Inject a SlowModel that sleeps 5s; timeout_s=0.1 → asyncio.TimeoutError."""

        class SlowModel:
            model_path = "slow-model.gguf"

            def create_chat_completion(self, messages, max_tokens, temperature):
                time.sleep(5)
                return {"choices": [{"message": {"content": "too late"}}]}

        slow_model = SlowModel()
        provider = LlamaCppModelProvider(get_model_fn=lambda: slow_model)
        with pytest.raises(asyncio.TimeoutError):
            asyncio.get_event_loop().run_until_complete(
                provider.infer(
                    [{"role": "user", "content": "hi"}],
                    timeout_s=0.1,
                )
            )

    def test_loaded_model_name_with_model_path(self):
        fake_model = MagicMock()
        fake_model.model_path = "/models/qwen2.5-7b.gguf"
        provider = LlamaCppModelProvider(get_model_fn=lambda: fake_model)
        assert provider.loaded_model_name() == "/models/qwen2.5-7b.gguf"

    def test_loaded_model_name_no_model(self):
        provider = LlamaCppModelProvider(get_model_fn=lambda: None)
        assert provider.loaded_model_name() == "unknown"

    def test_loaded_model_name_no_attr(self):
        """Model object without model_path attr → 'unknown'."""
        fake_model = MagicMock(spec=[])  # no attributes
        provider = LlamaCppModelProvider(get_model_fn=lambda: fake_model)
        assert provider.loaded_model_name() == "unknown"

    def test_infer_calls_create_chat_completion(self):
        """Happy path: model returns valid completion → dict with 'content' key."""
        fake_model = MagicMock()
        fake_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "pong"}}]
        }
        provider = LlamaCppModelProvider(get_model_fn=lambda: fake_model)
        messages = [{"role": "user", "content": "ping"}]
        result = asyncio.get_event_loop().run_until_complete(
            provider.infer(messages, max_tokens=512, temperature=0.1)
        )
        assert result["content"] == "pong"
        fake_model.create_chat_completion.assert_called_once_with(
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )
