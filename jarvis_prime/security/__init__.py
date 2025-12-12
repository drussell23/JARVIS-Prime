"""Security primitives for JARVIS.

Tiny Prime integration lives in:
- `jarvis_prime.security.check_semantic_security` (stable hook)
- `jarvis_prime.tiny_prime.*` (internal layout / full API)

This package keeps the stable hook import path while still allowing JARVIS to
depend directly on Tiny Prime's internals when desired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Pylance/static analyzers: `__all__` + `__getattr__` lazy exports.
if TYPE_CHECKING:
    from typing import Any

    check_semantic_security: Any
    check_semantic_security_async: Any
    get_semantic_guard: Any
    IntentResult: Any
    TinyPrimeGuard: Any
    TinyPrimeConfig: Any


def __getattr__(name: str):
    # Lazy to avoid importing heavy ML deps at import time.
    if name in {
        "check_semantic_security",
        "check_semantic_security_async",
        "get_semantic_guard",
        "IntentResult",
        "TinyPrimeGuard",
        "TinyPrimeConfig",
    }:
        from jarvis_prime.security.check_semantic_security import (  # noqa: WPS433
            IntentResult,
            TinyPrimeConfig,
            TinyPrimeGuard,
            check_semantic_security,
            check_semantic_security_async,
            get_semantic_guard,
        )

        return {
            "check_semantic_security": check_semantic_security,
            "check_semantic_security_async": check_semantic_security_async,
            "get_semantic_guard": get_semantic_guard,
            "IntentResult": IntentResult,
            "TinyPrimeGuard": TinyPrimeGuard,
            "TinyPrimeConfig": TinyPrimeConfig,
        }[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "check_semantic_security",
    "check_semantic_security_async",
    "get_semantic_guard",
    "IntentResult",
    "TinyPrimeGuard",
    "TinyPrimeConfig",
]
