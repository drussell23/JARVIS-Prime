#!/usr/bin/env python3
"""
Cross-Repo Integration Test Suite v95.2
========================================

Comprehensive tests for cross-repo integration between:
- JARVIS-AI-Agent (Body)
- JARVIS-Prime (Mind)
- Reactor-Core (Nervous System)

Tests:
1. GCP Import Bridge - All 4 components
2. Cross-repo communication
3. Elastic cloud burst
4. Circuit breakers and rate limiting
5. Event bus coordination
6. HybridTieredRouter with tier selection
7. Budget and RAM monitoring
8. Dynamic Model Registry with neural selection

Run:
    python3 tests/test_cross_repo_integration.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def record_pass(self, name: str) -> None:
        self.passed += 1
        print(f"  [PASS] {name}")

    def record_fail(self, name: str, reason: str) -> None:
        self.failed += 1
        self.errors.append(f"{name}: {reason}")
        print(f"  [FAIL] {name}: {reason}")

    def summary(self) -> None:
        total = self.passed + self.failed
        print()
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print("Failures:")
            for error in self.errors:
                print(f"  - {error}")


async def test_gcp_import_bridge(results: TestResults) -> None:
    """Test GCP Import Bridge components."""
    print()
    print("1. Testing GCP Import Bridge...")

    try:
        from jarvis_prime.core.gcp_import_bridge import GCPImportBridge

        bridge = await GCPImportBridge.get_instance()
        status = await bridge.get_infrastructure_status()

        # Test each component
        components = status.get("components", {})
        for name, info in components.items():
            if info.get("available"):
                results.record_pass(f"GCPImportBridge.{name}")
            else:
                results.record_fail(f"GCPImportBridge.{name}", "not available")

        # Test repos detected
        if status.get("jarvis_repo"):
            results.record_pass("JARVIS repo detection")
        else:
            results.record_fail("JARVIS repo detection", "not found")

        if status.get("reactor_repo"):
            results.record_pass("Reactor repo detection")
        else:
            results.record_fail("Reactor repo detection", "not found")

    except Exception as e:
        results.record_fail("GCPImportBridge", str(e))


async def test_circuit_breakers(results: TestResults) -> None:
    """Test circuit breaker functionality."""
    print()
    print("2. Testing Circuit Breakers...")

    try:
        from jarvis_prime.core.gcp_import_bridge import (
            CircuitBreaker,
            CircuitState,
        )

        # Test basic circuit breaker
        cb = CircuitBreaker("test_breaker")

        # Should start closed
        if cb.state == CircuitState.CLOSED:
            results.record_pass("CircuitBreaker initial state")
        else:
            results.record_fail("CircuitBreaker initial state", f"got {cb.state}")

        # Should allow execution when closed
        if await cb.can_execute():
            results.record_pass("CircuitBreaker can_execute (closed)")
        else:
            results.record_fail("CircuitBreaker can_execute (closed)", "returned False")

        # Record success
        await cb.record_success()
        results.record_pass("CircuitBreaker record_success")

        # Record failures until open
        for _ in range(5):
            await cb.record_failure()

        if cb.state == CircuitState.OPEN:
            results.record_pass("CircuitBreaker opens after failures")
        else:
            results.record_fail("CircuitBreaker opens after failures", f"got {cb.state}")

    except Exception as e:
        results.record_fail("CircuitBreaker", str(e))


async def test_rate_limiter(results: TestResults) -> None:
    """Test rate limiter functionality."""
    print()
    print("3. Testing Rate Limiter...")

    try:
        from jarvis_prime.core.gcp_import_bridge import (
            TokenBucketRateLimiter,
            RateLimitConfig,
        )

        # Test with high rate limit
        rl = TokenBucketRateLimiter(RateLimitConfig(
            requests_per_second=100.0,
            burst_size=50,
        ))

        # Should acquire tokens
        if await rl.acquire():
            results.record_pass("RateLimiter.acquire")
        else:
            results.record_fail("RateLimiter.acquire", "returned False")

        # Should acquire multiple tokens
        acquired = 0
        for _ in range(10):
            if await rl.acquire():
                acquired += 1

        if acquired == 10:
            results.record_pass("RateLimiter.acquire (burst)")
        else:
            results.record_fail("RateLimiter.acquire (burst)", f"only {acquired}/10")

    except Exception as e:
        results.record_fail("RateLimiter", str(e))


async def test_async_cache(results: TestResults) -> None:
    """Test async TTL cache."""
    print()
    print("4. Testing Async TTL Cache...")

    try:
        from jarvis_prime.core.gcp_import_bridge import AsyncTTLCache

        cache: AsyncTTLCache[str] = AsyncTTLCache(default_ttl=60.0)

        # Test set and get
        await cache.set("test_key", "test_value")
        value = await cache.get("test_key")

        if value == "test_value":
            results.record_pass("AsyncTTLCache.set/get")
        else:
            results.record_fail("AsyncTTLCache.set/get", f"got {value}")

        # Test get_or_set
        async def factory():
            return "computed_value"

        result = await cache.get_or_set("new_key", factory)
        if result == "computed_value":
            results.record_pass("AsyncTTLCache.get_or_set")
        else:
            results.record_fail("AsyncTTLCache.get_or_set", f"got {result}")

        # Test cache hit
        result2 = await cache.get_or_set("new_key", factory)
        stats = cache.get_stats()

        if stats["hits"] > 0:
            results.record_pass("AsyncTTLCache.hits")
        else:
            results.record_fail("AsyncTTLCache.hits", "no hits recorded")

    except Exception as e:
        results.record_fail("AsyncTTLCache", str(e))


async def test_event_bus(results: TestResults) -> None:
    """Test event bus functionality."""
    print()
    print("5. Testing Event Bus...")

    try:
        from jarvis_prime.core.gcp_import_bridge import EventBus, Event

        bus = await EventBus.get_instance()

        # Track received events
        received = []

        async def handler(event: Event):
            received.append(event)

        # Subscribe
        bus.subscribe("test_event", handler)
        results.record_pass("EventBus.subscribe")

        # Publish
        await bus.publish(Event(name="test_event", payload={"test": True}))

        if len(received) == 1:
            results.record_pass("EventBus.publish")
        else:
            results.record_fail("EventBus.publish", f"received {len(received)} events")

        # Check event data
        if received[0].payload.get("test"):
            results.record_pass("EventBus.event_data")
        else:
            results.record_fail("EventBus.event_data", "payload mismatch")

    except Exception as e:
        results.record_fail("EventBus", str(e))


async def test_hybrid_tiered_router(results: TestResults) -> None:
    """Test HybridTieredRouter integration."""
    print()
    print("6. Testing HybridTieredRouter...")

    try:
        from jarvis_prime.core.hybrid_tiered_router import get_hybrid_tiered_router

        router = await get_hybrid_tiered_router()

        # Test initialization
        if router._initialized:
            results.record_pass("HybridTieredRouter.initialized")
        else:
            results.record_fail("HybridTieredRouter.initialized", "not initialized")

        # Test routing
        result = await router.route("What is 2 + 2?")
        if result.tier_id:
            results.record_pass("HybridTieredRouter.route")
        else:
            results.record_fail("HybridTieredRouter.route", "no tier_id")

        # Test elastic burst status
        burst_status = await router.get_elastic_burst_status()
        if "burst_possible" in burst_status:
            results.record_pass("HybridTieredRouter.elastic_burst_status")
        else:
            results.record_fail("HybridTieredRouter.elastic_burst_status", "missing fields")

        # Test ResourceMonitor GCP integration
        stats = router._resource_monitor.get_statistics()
        if stats.get("gcp_bridge_available"):
            results.record_pass("ResourceMonitor.gcp_bridge")
        else:
            results.record_fail("ResourceMonitor.gcp_bridge", "not available")

    except Exception as e:
        results.record_fail("HybridTieredRouter", str(e))


async def test_budget_and_ram(results: TestResults) -> None:
    """Test budget and RAM monitoring."""
    print()
    print("7. Testing Budget and RAM Monitoring...")

    try:
        from jarvis_prime.core.gcp_import_bridge import (
            get_budget_status,
            get_ram_snapshot,
            can_afford_cloud_tier,
        )

        # Test budget status (handles both flat and nested structures)
        budget = await get_budget_status()
        # Check for either flat format or nested format
        has_data = (
            "daily_remaining" in budget or
            "available" in budget or
            "budgets" in budget or  # Nested format from JARVIS CostTracker
            "mode" in budget
        )
        if has_data:
            results.record_pass("get_budget_status")
        else:
            results.record_fail("get_budget_status", f"missing fields: {list(budget.keys())}")

        # Test RAM snapshot
        ram = await get_ram_snapshot()
        if "percent_used" in ram:
            results.record_pass("get_ram_snapshot")
        else:
            results.record_fail("get_ram_snapshot", "missing fields")

        # Test can_afford_cloud_tier
        can_afford = await can_afford_cloud_tier(0.01)
        if isinstance(can_afford, bool):
            results.record_pass("can_afford_cloud_tier")
        else:
            results.record_fail("can_afford_cloud_tier", f"got {type(can_afford)}")

    except Exception as e:
        results.record_fail("Budget/RAM", str(e))


async def test_dynamic_model_registry(results: TestResults) -> None:
    """Test Dynamic Model Registry v95.2."""
    print()
    print("8. Testing Dynamic Model Registry...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            get_dynamic_model_registry,
            KNOWN_MODELS,
            TIER_MODEL_MAPPING,
            NeuralModelSelector,
        )

        # Test KNOWN_MODELS catalog
        if len(KNOWN_MODELS) >= 5:
            results.record_pass("KNOWN_MODELS catalog")
        else:
            results.record_fail("KNOWN_MODELS catalog", f"only {len(KNOWN_MODELS)} models")

        # Test TIER_MODEL_MAPPING
        required_tiers = ["tier_0_local_fast", "tier_05_local_capable", "tier_1_cloud_intelligent"]
        has_all_tiers = all(t in TIER_MODEL_MAPPING for t in required_tiers)
        if has_all_tiers:
            results.record_pass("TIER_MODEL_MAPPING")
        else:
            results.record_fail("TIER_MODEL_MAPPING", "missing tiers")

        # Test registry initialization
        registry = await get_dynamic_model_registry()
        if registry._initialized:
            results.record_pass("DynamicModelRegistry.initialized")
        else:
            results.record_fail("DynamicModelRegistry.initialized", "not initialized")

        # Test get_model_spec
        spec = registry.get_model_spec("phi-3.5-mini")
        if spec and spec.parameter_count == "3.8B":
            results.record_pass("DynamicModelRegistry.get_model_spec")
        else:
            results.record_fail("DynamicModelRegistry.get_model_spec", "spec not found")

        # Test Neural Model Selector
        selector = NeuralModelSelector()
        available_specs = list(KNOWN_MODELS.values())[:4]
        selected = await selector.select(
            prompt="implement async caching",
            complexity_score=0.6,
            available_models=available_specs,
            requirements=["code"],
        )
        if selected and selected.model_id:
            results.record_pass("NeuralModelSelector.select")
        else:
            results.record_fail("NeuralModelSelector.select", "no model selected")

        # Test registry statistics
        stats = registry.get_statistics()
        if "total_known_models" in stats and stats["total_known_models"] >= 5:
            results.record_pass("DynamicModelRegistry.get_statistics")
        else:
            results.record_fail("DynamicModelRegistry.get_statistics", "missing stats")

    except Exception as e:
        results.record_fail("DynamicModelRegistry", str(e))


async def main():
    """Run all integration tests."""
    print("=" * 70)
    print("  CROSS-REPO INTEGRATION TEST SUITE v95.2")
    print("  GCP Bridge + HybridTieredRouter + Dynamic Model Registry")
    print("=" * 70)

    results = TestResults()

    # Run all tests
    await test_gcp_import_bridge(results)
    await test_circuit_breakers(results)
    await test_rate_limiter(results)
    await test_async_cache(results)
    await test_event_bus(results)
    await test_hybrid_tiered_router(results)
    await test_budget_and_ram(results)
    await test_dynamic_model_registry(results)

    # Print summary
    print()
    print("=" * 70)
    results.summary()
    print("=" * 70)

    # Exit with appropriate code
    if results.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
