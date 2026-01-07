"""
JARVIS Prime - Specialized PRIME Models for JARVIS
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.7.0"

# Static analyzers (e.g. Pylance) don't infer exports provided via `__getattr__`.
# Declare type-check-only stubs so names in `__all__` are considered present,
# without importing heavy ML dependencies at runtime.
if TYPE_CHECKING:
    from typing import Any

    PrimeModel: Any
    PrimeTrainer: Any
    LlamaModel: Any
    LlamaTrainer: Any
    LlamaPresets: Any
    LlamaModelConfig: Any
    TinyPrimeConfig: Any
    TinyPrimeGuard: Any
    IntentResult: Any

    # AGI v0.7.0 Components
    AGIOrchestrator: Any
    ReasoningEngine: Any
    AppleSiliconOptimizer: Any
    ContinuousLearningEngine: Any
    MultiModalFusionEngine: Any

    def check_semantic_security(text: str, *, cfg_path: str | None = None) -> Any: ...

    async def check_semantic_security_async(text: str, *, cfg_path: str | None = None) -> Any: ...

    def get_semantic_guard(*, cfg_path: str | None = None) -> Any: ...

# Lazy imports to avoid loading heavy dependencies at import time
def __getattr__(name):
    if name == "PrimeModel":
        from jarvis_prime.models import PrimeModel
        return PrimeModel
    elif name == "PrimeTrainer":
        from jarvis_prime.trainer import PrimeTrainer
        return PrimeTrainer
    elif name == "LlamaModel":
        from jarvis_prime.models import LlamaModel
        return LlamaModel
    elif name == "LlamaTrainer":
        from jarvis_prime.trainer import LlamaTrainer
        return LlamaTrainer
    elif name == "LlamaPresets":
        from jarvis_prime.configs import LlamaPresets
        return LlamaPresets
    elif name == "LlamaModelConfig":
        from jarvis_prime.configs import LlamaModelConfig
        return LlamaModelConfig
    elif name == "TinyPrimeConfig":
        from jarvis_prime.tiny_prime import TinyPrimeConfig

        return TinyPrimeConfig
    elif name == "TinyPrimeGuard":
        from jarvis_prime.tiny_prime import TinyPrimeGuard

        return TinyPrimeGuard
    elif name in {
        "check_semantic_security",
        "check_semantic_security_async",
        "get_semantic_guard",
        "IntentResult",
    }:
        # Stable, top-level exports for the VBI pipeline.
        # Lazy import: does not import torch/transformers until functions are called.
        from jarvis_prime.security.check_semantic_security import (  # noqa: WPS433
            IntentResult,
            check_semantic_security,
            check_semantic_security_async,
            get_semantic_guard,
        )

        return {
            "check_semantic_security": check_semantic_security,
            "check_semantic_security_async": check_semantic_security_async,
            "get_semantic_guard": get_semantic_guard,
            "IntentResult": IntentResult,
        }[name]
    # AGI v0.7.0 Components - Lazy loaded to avoid heavy imports
    elif name == "AGIOrchestrator":
        from jarvis_prime.core.agi_models import AGIOrchestrator
        return AGIOrchestrator
    elif name == "ReasoningEngine":
        from jarvis_prime.core.reasoning_engine import ReasoningEngine
        return ReasoningEngine
    elif name == "AppleSiliconOptimizer":
        from jarvis_prime.core.apple_silicon_optimizer import AppleSiliconOptimizer
        return AppleSiliconOptimizer
    elif name == "ContinuousLearningEngine":
        from jarvis_prime.core.continuous_learning import ContinuousLearningEngine
        return ContinuousLearningEngine
    elif name == "MultiModalFusionEngine":
        from jarvis_prime.core.multimodal_fusion import MultiModalFusionEngine
        return MultiModalFusionEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "PrimeModel",
    "PrimeTrainer",
    "LlamaModel",
    "LlamaTrainer",
    "LlamaPresets",
    "LlamaModelConfig",
    "TinyPrimeConfig",
    "TinyPrimeGuard",
    "check_semantic_security",
    "check_semantic_security_async",
    "get_semantic_guard",
    "IntentResult",
    # AGI v0.7.0 Components
    "AGIOrchestrator",
    "ReasoningEngine",
    "AppleSiliconOptimizer",
    "ContinuousLearningEngine",
    "MultiModalFusionEngine",
    "__version__",
]
