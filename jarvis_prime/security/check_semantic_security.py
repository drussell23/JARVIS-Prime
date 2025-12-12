"""Stable semantic-security hook for JARVIS.

This module intentionally provides a stable import path:
- `from jarvis_prime.security.check_semantic_security import check_semantic_security`

Implementation delegates to Tiny Prime internals:
- `jarvis_prime.tiny_prime.semantic_guard`

So JARVIS *can* also import and use the internal layout directly when desired.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis_prime.tiny_prime.config import TinyPrimeConfig
    from jarvis_prime.tiny_prime.semantic_guard import IntentResult, TinyPrimeGuard


def get_semantic_guard(*, cfg_path: Optional[str] = None) -> "TinyPrimeGuard":
    # Local import to avoid importing torch/transformers until needed.
    from jarvis_prime.tiny_prime.semantic_guard import get_semantic_guard as _get

    return _get(cfg_path=cfg_path)


def check_semantic_security(text: str, *, cfg_path: Optional[str] = None) -> "IntentResult":
    from jarvis_prime.tiny_prime.semantic_guard import check_semantic_security as _check

    return _check(text, cfg_path=cfg_path)


async def check_semantic_security_async(
    text: str,
    *,
    cfg_path: Optional[str] = None,
) -> "IntentResult":
    from jarvis_prime.tiny_prime.semantic_guard import check_semantic_security_async as _check_async

    return await _check_async(text, cfg_path=cfg_path)


# Convenience re-exports (internal layout is still first-class)
def __getattr__(name: str):
    # Lazy to avoid importing torch/transformers at import time.
    if name in {"TinyPrimeConfig", "TinyPrimeGuard", "IntentResult"}:
        if name == "TinyPrimeConfig":
            from jarvis_prime.tiny_prime.config import TinyPrimeConfig  # noqa: WPS433

            return TinyPrimeConfig
        from jarvis_prime.tiny_prime.semantic_guard import (  # noqa: WPS433
            IntentResult,
            TinyPrimeGuard,
        )

        return {"TinyPrimeGuard": TinyPrimeGuard, "IntentResult": IntentResult}[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "get_semantic_guard",
    "check_semantic_security",
    "check_semantic_security_async",
    "TinyPrimeConfig",
    "TinyPrimeGuard",
    "IntentResult",
]
