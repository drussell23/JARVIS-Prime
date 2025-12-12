"""Tiny Prime - Semantic Security Guard for JARVIS VBI."""

from __future__ import annotations

__version__ = "0.1.0"

# Lazy imports to keep import time light.
def __getattr__(name: str):
    if name == "TinyPrimeConfig":
        from jarvis_prime.tiny_prime.config import TinyPrimeConfig

        return TinyPrimeConfig
    if name == "TinyPrimeGuard":
        from jarvis_prime.tiny_prime.semantic_guard import TinyPrimeGuard

        return TinyPrimeGuard
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "TinyPrimeConfig",
    "TinyPrimeGuard",
    "__version__",
]
