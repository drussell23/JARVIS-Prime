"""
Tests for Task 6: J-Prime Model Routing Layer.

Covers:
1. ModelDispatcher initializes with empty catalogue (no yaml).
2. ModelDispatcher.get_executor returns default when model_name is None.
3. ModelDispatcher.get_executor returns default when model_name is "jarvis-prime".
4. ModelDispatcher.get_executor returns default when GGUF not found.
5. ModelDispatcher caches executor and returns same object on second call.
6. ModelDispatcher evicts LRU when max_loaded is reached.
7. ModelDispatcher.list_loaded reflects cache state.
8. CompletionRequest carries task_profile field.
9. CompletionRequest.task_profile defaults to None.
10. PrimeModelManager._execute_tier_0 uses default executor when dispatcher is None.
11. PrimeModelManager._execute_tier_0 uses dispatched executor when requested_model set.
12. CompletionRequestModel Pydantic model carries task_profile field.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis_prime.core.model_dispatcher import ModelDispatcher
from jarvis_prime.core.model_manager import CompletionRequest, ChatMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeExecutor:
    def __init__(self, name: str = "default"):
        self.name = name
        self.loaded_path: Optional[Path] = None
        self.generate = AsyncMock(return_value=f"response from {name}")

    async def load(self, path: Path) -> None:
        self.loaded_path = path

    async def unload(self) -> None:
        pass

    async def validate(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Test 1 — ModelDispatcher initializes with no catalogue file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_initializes_without_catalogue(tmp_path):
    dispatcher = ModelDispatcher(models_dir=tmp_path)
    await dispatcher.initialize()
    assert dispatcher.list_loaded() == []


# ---------------------------------------------------------------------------
# Test 2 — Returns default for None model_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_returns_default_for_none(tmp_path):
    dispatcher = ModelDispatcher(models_dir=tmp_path)
    await dispatcher.initialize()
    default = _FakeExecutor("default")
    result = await dispatcher.get_executor(None, default)
    assert result is default


# ---------------------------------------------------------------------------
# Test 3 — Returns default for "jarvis-prime" alias
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_returns_default_for_jarvis_prime(tmp_path):
    dispatcher = ModelDispatcher(models_dir=tmp_path)
    await dispatcher.initialize()
    default = _FakeExecutor("default")
    result = await dispatcher.get_executor("jarvis-prime", default)
    assert result is default


# ---------------------------------------------------------------------------
# Test 4 — Returns default when GGUF not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_fallback_when_gguf_missing(tmp_path):
    dispatcher = ModelDispatcher(models_dir=tmp_path)
    await dispatcher.initialize()
    default = _FakeExecutor("default")
    result = await dispatcher.get_executor("qwen-2.5-coder-7b", default)
    assert result is default
    assert dispatcher.list_loaded() == []


# ---------------------------------------------------------------------------
# Test 5 — Caches executor; second call returns same object
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_caches_executor(tmp_path):
    # Place a dummy GGUF
    gguf = tmp_path / "mymodel.gguf"
    gguf.write_bytes(b"\x00" * 16)

    dispatcher = ModelDispatcher(models_dir=tmp_path, max_loaded=2)
    await dispatcher.initialize()

    fake_exec = _FakeExecutor("cached")

    with patch(
        "jarvis_prime.core.model_dispatcher.LlamaCppExecutor",
        return_value=fake_exec,
        create=True,
    ), patch.object(fake_exec, "load", AsyncMock()):
        # Patch _load_executor to return our fake executor
        async def _load_stub(name):
            return fake_exec

        dispatcher._load_executor = _load_stub

        default = _FakeExecutor("default")
        first = await dispatcher.get_executor("mymodel", default)
        second = await dispatcher.get_executor("mymodel", default)

    assert first is second
    assert "mymodel" in dispatcher.list_loaded()


# ---------------------------------------------------------------------------
# Test 6 — Evicts LRU when max_loaded reached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_evicts_lru(tmp_path):
    dispatcher = ModelDispatcher(models_dir=tmp_path, max_loaded=1)
    await dispatcher.initialize()

    exec_a = _FakeExecutor("a")
    exec_b = _FakeExecutor("b")

    async def _load_stub(name):
        return exec_a if name == "model_a" else exec_b

    dispatcher._load_executor = _load_stub

    default = _FakeExecutor("default")

    # Manually prime the cache to avoid needing real GGUF files
    dispatcher._executors["model_a"] = exec_a
    dispatcher._lru = ["model_a"]

    # Loading model_b should evict model_a
    # We need the path resolution to succeed — stub _resolve_gguf too
    gguf_b = tmp_path / "model_b.gguf"
    gguf_b.write_bytes(b"\x00" * 8)

    async def _load_b_stub(name):
        if name == "model_b":
            return exec_b
        return None

    dispatcher._load_executor = _load_b_stub

    result = await dispatcher.get_executor("model_b", default)
    # model_a should have been evicted
    assert "model_a" not in dispatcher._executors
    assert "model_b" in dispatcher._executors


# ---------------------------------------------------------------------------
# Test 7 — list_loaded reflects cache state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_list_loaded(tmp_path):
    dispatcher = ModelDispatcher(models_dir=tmp_path, max_loaded=3)
    await dispatcher.initialize()

    exec1 = _FakeExecutor("m1")
    exec2 = _FakeExecutor("m2")

    dispatcher._executors["model_one"] = exec1
    dispatcher._lru.insert(0, "model_one")
    dispatcher._executors["model_two"] = exec2
    dispatcher._lru.insert(0, "model_two")

    loaded = dispatcher.list_loaded()
    assert "model_one" in loaded
    assert "model_two" in loaded
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# Test 8 — CompletionRequest carries task_profile field
# ---------------------------------------------------------------------------


def test_completion_request_task_profile_field():
    profile = {"intent": "code_generation", "complexity": "heavy_code",
               "brain_id": "qwen_coder", "model": "qwen-2.5-coder-7b"}
    req = CompletionRequest(task_profile=profile)
    assert req.task_profile == profile


# ---------------------------------------------------------------------------
# Test 9 — CompletionRequest.task_profile defaults to None
# ---------------------------------------------------------------------------


def test_completion_request_task_profile_default():
    req = CompletionRequest()
    assert req.task_profile is None


# ---------------------------------------------------------------------------
# Test 10 — _execute_tier_0 uses default executor when dispatcher is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tier_0_no_dispatcher():
    from jarvis_prime.core.model_manager import PrimeModelManager, CompletionRequest

    manager = PrimeModelManager.__new__(PrimeModelManager)
    manager._dispatcher = None

    fake_exec = _FakeExecutor("default")

    class _FakeCtx:
        async def __aenter__(self):
            return fake_exec
        async def __aexit__(self, *a):
            pass

    mock_hot_swap = MagicMock()
    mock_hot_swap.acquire.return_value = _FakeCtx()
    manager._hot_swap = mock_hot_swap

    request = CompletionRequest(max_tokens=64, temperature=0.1)
    result = await manager._execute_tier_0("hello", request, requested_model=None)
    assert result == "response from default"


# ---------------------------------------------------------------------------
# Test 11 — _execute_tier_0 uses dispatched executor when requested_model set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_tier_0_with_dispatcher():
    from jarvis_prime.core.model_manager import PrimeModelManager, CompletionRequest

    manager = PrimeModelManager.__new__(PrimeModelManager)

    default_exec = _FakeExecutor("default")
    specialist_exec = _FakeExecutor("qwen-specialist")

    mock_dispatcher = AsyncMock()
    mock_dispatcher.get_executor = AsyncMock(return_value=specialist_exec)
    manager._dispatcher = mock_dispatcher

    class _FakeCtx:
        async def __aenter__(self):
            return default_exec
        async def __aexit__(self, *a):
            pass

    mock_hot_swap = MagicMock()
    mock_hot_swap.acquire.return_value = _FakeCtx()
    manager._hot_swap = mock_hot_swap

    request = CompletionRequest(max_tokens=64, temperature=0.1)
    result = await manager._execute_tier_0("hello", request, requested_model="qwen-2.5-coder-7b")

    mock_dispatcher.get_executor.assert_called_once_with(
        model_name="qwen-2.5-coder-7b",
        default_executor=default_exec,
    )
    assert result == "response from qwen-specialist"


# ---------------------------------------------------------------------------
# Test 12 — CompletionRequestModel Pydantic model carries task_profile
# ---------------------------------------------------------------------------


def test_pydantic_completion_request_model_task_profile():
    try:
        from jarvis_prime.core.model_manager import CompletionRequestModel
    except ImportError:
        pytest.skip("Pydantic not available")

    if CompletionRequestModel is None:
        pytest.skip("CompletionRequestModel not built (pydantic missing)")

    model = CompletionRequestModel(
        task_profile={"brain_id": "deepseek_r1", "model": "deepseek-r1-qwen-7b",
                      "intent": "segfault_analysis", "complexity": "complex"}
    )
    assert model.task_profile["brain_id"] == "deepseek_r1"

    model_default = CompletionRequestModel()
    assert model_default.task_profile is None
