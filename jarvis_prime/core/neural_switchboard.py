"""
Neural Switchboard facade for JARVIS Prime.

Provides a stable, public API surface for switchboard-style routing while
delegating implementation details to:
- dynamic_model_registry (task classification + model/spec routing)
- neural_orchestrator_core (tier-level fallback routing)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from jarvis_prime.core.dynamic_model_registry import (
    DynamicModelRegistry,
    ModelSpec,
    TaskClassification as RegistryTaskClassification,
    get_dynamic_model_registry,
    shutdown_dynamic_model_registry,
)


@dataclass
class NeuralSwitchboardDecision:
    """Unified switchboard routing decision."""

    source: Literal["dynamic_model_registry", "neural_orchestrator_core"]
    model_spec: Optional[ModelSpec]
    model_path: Optional[Path]
    task_classification: Optional[RegistryTaskClassification]
    orchestrator_result: Optional[Any] = None

    def __iter__(self):
        """Tuple compatibility: (model_spec, model_path, task_classification)."""
        yield self.model_spec
        yield self.model_path
        yield self.task_classification

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision for telemetry/debug APIs."""
        return {
            "source": self.source,
            "model": _serialize_model_spec(self.model_spec),
            "model_path": str(self.model_path) if self.model_path else None,
            "task_classification": _serialize_task_classification(self.task_classification),
            "orchestrator_result": (
                self.orchestrator_result.to_dict()
                if self.orchestrator_result is not None and hasattr(self.orchestrator_result, "to_dict")
                else None
            ),
        }


def _serialize_model_spec(spec: Optional[ModelSpec]) -> Optional[Dict[str, Any]]:
    if spec is None:
        return None
    return {
        "model_id": spec.model_id,
        "name": spec.name,
        "tier_level": spec.tier_level.name,
        "location": spec.location.value,
        "capabilities": [cap.value for cap in spec.capabilities],
        "context_length": spec.context_length,
        "memory_required_gb": spec.memory_required_gb,
    }


def _serialize_task_classification(
    task: Optional[RegistryTaskClassification],
) -> Optional[Dict[str, Any]]:
    if task is None:
        return None
    return {
        "task_type": task.task_type.value,
        "confidence": task.confidence,
        "complexity": task.complexity,
        "detected_signals": task.detected_signals,
        "recommended_tiers": task.recommended_tiers,
        "required_capabilities": [cap.value for cap in task.required_capabilities],
        "requires_fast_response": task.requires_fast_response,
        "is_coding_session": task.is_coding_session,
        "context_tokens_estimate": task.context_tokens_estimate,
    }


class NeuralSwitchboard:
    """
    Stable public Neural Switchboard API.

    This class is intentionally thin and keeps the implementation in existing
    core systems instead of duplicating routing logic.
    """

    def __init__(
        self,
        registry: Optional[DynamicModelRegistry] = None,
        models_dir: Optional[Path] = None,
    ) -> None:
        self._registry = registry
        self._models_dir = models_dir
        self._init_lock = asyncio.Lock()
        self._uses_registry_singleton = registry is None

    async def initialize(self) -> "NeuralSwitchboard":
        """Initialize backing registry if needed."""
        async with self._init_lock:
            if self._registry is None:
                self._registry = await get_dynamic_model_registry(models_dir=self._models_dir)
        return self

    async def route(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        strategy: Literal["auto", "switchboard", "orchestrator"] = "auto",
    ) -> NeuralSwitchboardDecision:
        """
        Route a prompt using switchboard and/or orchestrator strategy.

        Returns a NeuralSwitchboardDecision that can also be tuple-unpacked.
        """
        if strategy not in {"auto", "switchboard", "orchestrator"}:
            raise ValueError(f"Unsupported route strategy: {strategy}")

        context = context or {}

        if strategy in {"auto", "switchboard"}:
            registry = await self._get_registry()
            spec, path, task_classification = await registry.neural_switchboard_route(
                prompt, context
            )
            if spec is not None or strategy == "switchboard":
                return NeuralSwitchboardDecision(
                    source="dynamic_model_registry",
                    model_spec=spec,
                    model_path=path,
                    task_classification=task_classification,
                )

        from jarvis_prime.core.neural_orchestrator_core import get_neural_orchestrator

        orchestrator = await get_neural_orchestrator()
        orchestrator_result = await orchestrator.route(prompt, context)
        return NeuralSwitchboardDecision(
            source="neural_orchestrator_core",
            model_spec=None,
            model_path=None,
            task_classification=None,
            orchestrator_result=orchestrator_result,
        )

    async def route_raw(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[ModelSpec], Optional[Path], Optional[RegistryTaskClassification]]:
        """Compatibility helper that returns the historical tuple shape."""
        decision = await self.route(prompt, context=context, strategy="switchboard")
        return decision.model_spec, decision.model_path, decision.task_classification

    async def classify_task(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RegistryTaskClassification:
        registry = await self._get_registry()
        return await registry.classify_task(prompt, context)

    async def get_memory_pressure(self) -> Dict[str, Any]:
        registry = await self._get_registry()
        return await registry.get_memory_pressure()

    def get_sticky_status(self) -> Dict[str, Any]:
        if self._registry is None:
            return {"initialized": False}
        return self._registry.get_sticky_status()

    def get_statistics(self) -> Dict[str, Any]:
        if self._registry is None:
            return {"initialized": False, "version": "98.0-facade"}
        stats = self._registry.get_statistics().copy()
        stats["facade_version"] = "98.1"
        return stats

    def get_status(self) -> Dict[str, Any]:
        """Synchronous status snapshot."""
        stats = self.get_statistics()
        stats["sticky_routing_status"] = self.get_sticky_status()
        return stats

    async def shutdown(self, shutdown_registry: bool = False) -> None:
        """
        Shutdown this facade.

        By default this only detaches the local reference. Set
        shutdown_registry=True to also stop the global registry singleton.
        """
        if shutdown_registry and self._uses_registry_singleton:
            await shutdown_dynamic_model_registry()
        elif shutdown_registry and self._registry is not None:
            await self._registry.shutdown()
        self._registry = None

    async def _get_registry(self) -> DynamicModelRegistry:
        if self._registry is None:
            await self.initialize()
        assert self._registry is not None
        return self._registry


_switchboard: Optional[NeuralSwitchboard] = None
_switchboard_lock = asyncio.Lock()


async def get_neural_switchboard(
    models_dir: Optional[Path] = None,
) -> NeuralSwitchboard:
    """Get or create a process-level NeuralSwitchboard facade."""
    global _switchboard
    async with _switchboard_lock:
        if _switchboard is None:
            _switchboard = NeuralSwitchboard(models_dir=models_dir)
            await _switchboard.initialize()
        return _switchboard


async def shutdown_neural_switchboard(shutdown_registry: bool = False) -> None:
    """Shutdown the process-level NeuralSwitchboard facade."""
    global _switchboard
    async with _switchboard_lock:
        if _switchboard is not None:
            await _switchboard.shutdown(shutdown_registry=shutdown_registry)
            _switchboard = None


__all__ = [
    "NeuralSwitchboard",
    "NeuralSwitchboardDecision",
    "get_neural_switchboard",
    "shutdown_neural_switchboard",
]
