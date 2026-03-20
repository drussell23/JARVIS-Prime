"""
ModelProvider — dependency injection interface for GPU inference.

Graph nodes use ModelProvider to perform LLM inference without importing
from server.py directly. This keeps reasoning nodes testable in isolation.

Three classes:
- ModelProvider: @runtime_checkable Protocol (the interface)
- MockModelProvider: test double with call tracking
- LlamaCppModelProvider: production wrapper running sync llama-cpp
  inference off the event loop via run_in_executor + asyncio.wait_for
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Callable, Dict, List, Optional
from typing import runtime_checkable, Protocol


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelProvider(Protocol):
    """Interface that graph nodes use for GPU inference.

    Implementations must be safe to call from async code. Synchronous
    model backends (e.g. llama-cpp-python) must run in an executor and
    must honour the ``timeout_s`` deadline via ``asyncio.wait_for``.
    """

    async def infer(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout_s: float = 10.0,
    ) -> Dict:
        """Run inference and return a dict containing at least ``content``."""
        ...

    def is_model_loaded(self) -> bool:
        """Return True if the underlying model is ready to serve requests."""
        ...

    def loaded_model_name(self) -> str:
        """Return a human-readable identifier for the currently loaded model."""
        ...


# ---------------------------------------------------------------------------
# MockModelProvider — test double
# ---------------------------------------------------------------------------


class MockModelProvider:
    """Test double for ModelProvider.

    Parameters
    ----------
    response:
        Canned string that ``infer`` wraps in ``{"content": response}``.
    should_fail:
        When True, ``infer`` raises ``RuntimeError("Mock model failure")``.
    """

    def __init__(
        self,
        response: str = "mock response",
        should_fail: bool = False,
    ) -> None:
        self._response = response
        self._should_fail = should_fail
        self.call_count: int = 0
        self.last_messages: Optional[List] = None

    async def infer(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout_s: float = 10.0,
    ) -> Dict:
        self.call_count += 1
        self.last_messages = messages
        if self._should_fail:
            raise RuntimeError("Mock model failure")
        return {"content": self._response}

    def is_model_loaded(self) -> bool:
        return not self._should_fail

    def loaded_model_name(self) -> str:
        return "" if self._should_fail else "mock-model"


# ---------------------------------------------------------------------------
# LlamaCppModelProvider — production wrapper
# ---------------------------------------------------------------------------


class LlamaCppModelProvider:
    """Production ModelProvider wrapping a llama-cpp-python Llama instance.

    The synchronous ``create_chat_completion`` call is executed in a
    ``ThreadPoolExecutor`` (via ``loop.run_in_executor``) so it never
    blocks the event loop.  The whole operation is bounded by
    ``asyncio.wait_for(timeout=timeout_s)``.

    Parameters
    ----------
    get_model_fn:
        Zero-argument callable returning the current ``Llama`` instance,
        or ``None`` if no model is loaded.  Using a callable (rather than
        holding a direct reference) allows hot-swapping the model without
        recreating the provider.
    """

    def __init__(self, get_model_fn: Callable[[], Optional[Any]]) -> None:
        self._get_model = get_model_fn

    async def infer(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout_s: float = 10.0,
    ) -> Dict:
        model = self._get_model()
        if model is None:
            raise RuntimeError("No model loaded")

        loop = asyncio.get_event_loop()

        def _run() -> Dict:
            completion = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = completion["choices"][0]["message"]["content"]
            return {"content": content}

        coro = loop.run_in_executor(None, functools.partial(_run))
        result: Dict = await asyncio.wait_for(coro, timeout=timeout_s)
        return result

    def is_model_loaded(self) -> bool:
        return self._get_model() is not None

    def loaded_model_name(self) -> str:
        model = self._get_model()
        if model is None:
            return "unknown"
        return getattr(model, "model_path", "unknown")


# ---------------------------------------------------------------------------
# PrimeManagerModelProvider — wraps PrimeModelManager.complete()
# ---------------------------------------------------------------------------


class PrimeManagerModelProvider:
    """Production ModelProvider that uses PrimeModelManager for inference.

    This is the preferred production provider. It uses the full model manager
    pipeline (routing, hot-swap, reasoning engine, telemetry) rather than
    calling the raw llama-cpp model directly.

    Parameters
    ----------
    get_manager_fn:
        Zero-argument callable returning the PrimeModelManager instance,
        or None if not yet initialized.
    """

    def __init__(self, get_manager_fn: Callable[[], Optional[Any]]) -> None:
        self._get_manager = get_manager_fn

    async def infer(
        self,
        messages: List[Dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        timeout_s: float = 10.0,
    ) -> Dict:
        manager = self._get_manager()
        if manager is None:
            raise RuntimeError("PrimeModelManager not initialized")

        # Convert dicts to ChatMessage dataclasses
        from jarvis_prime.core.model_manager import ChatMessage, CompletionRequest

        chat_messages = [
            ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
            for m in messages
        ]
        request = CompletionRequest(
            messages=chat_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            force_tier="tier_0",  # always use local GPU for reasoning nodes
        )

        response = await asyncio.wait_for(
            manager.complete(request),
            timeout=timeout_s,
        )

        # Extract content from CompletionResponse
        content = ""
        if response.choices:
            content = response.choices[0].message.content
        return {"content": content}

    def is_model_loaded(self) -> bool:
        manager = self._get_manager()
        if manager is None:
            return False
        try:
            return manager.get_health_status() == "ready"
        except Exception:
            return False

    def loaded_model_name(self) -> str:
        manager = self._get_manager()
        if manager is None:
            return ""
        try:
            status = manager.get_health_status()
            return status if isinstance(status, str) else "unknown"
        except Exception:
            return "unknown"
