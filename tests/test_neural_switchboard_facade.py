from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from jarvis_prime.core.neural_switchboard import NeuralSwitchboard


@pytest.mark.asyncio
async def test_neural_switchboard_facade_routes_with_registry(monkeypatch):
    spec = SimpleNamespace(
        model_id="qwen-2.5-32b",
        name="Qwen 2.5 32B",
        tier_level=SimpleNamespace(name="TIER_05_LOCAL_CAPABLE"),
        location=SimpleNamespace(value="local_only"),
        capabilities=[SimpleNamespace(value="code_expert")],
        context_length=131072,
        memory_required_gb=19.0,
    )
    classification = SimpleNamespace(
        task_type=SimpleNamespace(value="code_simple"),
        confidence=0.91,
        complexity=0.62,
        detected_signals=["code"],
        recommended_tiers=["tier_05_local_capable"],
        required_capabilities=[SimpleNamespace(value="code_expert")],
        requires_fast_response=False,
        is_coding_session=True,
        context_tokens_estimate=256,
    )

    fake_registry = SimpleNamespace(
        neural_switchboard_route=AsyncMock(return_value=(spec, Path("/tmp/model.gguf"), classification)),
        classify_task=AsyncMock(return_value=classification),
        get_memory_pressure=AsyncMock(return_value={"pressure_level": "normal"}),
        get_sticky_status=lambda: {"sticky_model": "qwen-2.5-32b"},
        get_statistics=lambda: {"version": "99.0"},
    )

    monkeypatch.setattr(
        "jarvis_prime.core.neural_switchboard.get_dynamic_model_registry",
        AsyncMock(return_value=fake_registry),
    )

    switchboard = NeuralSwitchboard()
    decision = await switchboard.route("write a binary search", strategy="switchboard")
    routed_spec, model_path, task_class = decision

    assert decision.source == "dynamic_model_registry"
    assert routed_spec is spec
    assert model_path == Path("/tmp/model.gguf")
    assert task_class is classification
    assert decision.to_dict()["model"]["model_id"] == "qwen-2.5-32b"

    status = switchboard.get_status()
    assert status["version"] == "99.0"
    assert "sticky_routing_status" in status


@pytest.mark.asyncio
async def test_neural_switchboard_facade_orchestrator_strategy(monkeypatch):
    fake_orchestrator_result = SimpleNamespace(
        to_dict=lambda: {"tier": "TIER_1_CLOUD", "model_id": "llama-3.3-70b"},
    )
    fake_orchestrator = SimpleNamespace(
        route=AsyncMock(return_value=fake_orchestrator_result),
    )

    import jarvis_prime.core.neural_orchestrator_core as neural_orchestrator_core

    monkeypatch.setattr(
        neural_orchestrator_core,
        "get_neural_orchestrator",
        AsyncMock(return_value=fake_orchestrator),
    )

    switchboard = NeuralSwitchboard()
    decision = await switchboard.route(
        "explain CAP theorem tradeoffs",
        strategy="orchestrator",
    )

    assert decision.source == "neural_orchestrator_core"
    assert decision.model_spec is None
    assert decision.orchestrator_result is fake_orchestrator_result
    assert decision.to_dict()["orchestrator_result"]["model_id"] == "llama-3.3-70b"
