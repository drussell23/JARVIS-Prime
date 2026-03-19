"""Tests for /v1/reason/health and /v1/protocol/version endpoints."""
import pytest
from jarvis_prime.reasoning.endpoints import get_reason_health, get_protocol_version
from jarvis_prime.reasoning.protocol import PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_protocol_version_returns_current():
    result = await get_protocol_version()
    assert result["current_version"] == PROTOCOL_VERSION
    assert "min_supported_version" in result
    assert "max_supported_version" in result
    assert isinstance(result["features"], list)
    assert "brain_selection" in result["features"]


@pytest.mark.asyncio
async def test_reason_health_basic():
    result = await get_reason_health()
    assert "status" in result
    assert result["status"] in ("ready", "starting", "degraded")
    assert "protocol_version" in result
    assert "brains_loaded" in result
    assert isinstance(result["brains_loaded"], list)


@pytest.mark.asyncio
async def test_protocol_version_has_brain_policy_hash():
    result = await get_protocol_version()
    assert "brain_policy_hash" in result
    assert isinstance(result["brain_policy_hash"], str)


@pytest.mark.asyncio
async def test_protocol_version_has_version_bounds():
    result = await get_protocol_version()
    assert result["min_supported_version"] != ""
    assert result["max_supported_version"] != ""


@pytest.mark.asyncio
async def test_reason_health_has_reasoning_graph_ready():
    result = await get_reason_health()
    assert "reasoning_graph_ready" in result
    assert isinstance(result["reasoning_graph_ready"], bool)


@pytest.mark.asyncio
async def test_reason_health_has_brain_policy_hash():
    result = await get_reason_health()
    assert "brain_policy_hash" in result
    assert isinstance(result["brain_policy_hash"], str)


@pytest.mark.asyncio
async def test_reason_health_brains_loaded_is_list():
    result = await get_reason_health()
    # brains_loaded is always a list (may be empty if hybrid_router unavailable)
    assert isinstance(result["brains_loaded"], list)


@pytest.mark.asyncio
async def test_protocol_version_features_is_list_of_strings():
    result = await get_protocol_version()
    features = result["features"]
    assert isinstance(features, list)
    for f in features:
        assert isinstance(f, str)
