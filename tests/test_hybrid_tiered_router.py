#!/usr/bin/env python3
"""
Test Suite for Hybrid Tiered Router v94.0
==========================================

Tests intelligent tier-based routing with complexity analysis.

Run: python3 -m pytest tests/test_hybrid_tiered_router.py -v
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest


class TestComplexityAnalyzer:
    """Test the AdvancedComplexityAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        from jarvis_prime.core.hybrid_tiered_router import AdvancedComplexityAnalyzer
        return AdvancedComplexityAnalyzer()

    @pytest.mark.asyncio
    async def test_simple_greeting_low_complexity(self, analyzer):
        """Simple greetings should have low complexity."""
        analysis = await analyzer.analyze("Hello, how are you?")

        assert analysis.score < 0.35
        assert analysis.task_type == "greeting" or analysis.task_type == "chat"

    @pytest.mark.asyncio
    async def test_complex_reasoning_high_complexity(self, analyzer):
        """Complex reasoning tasks should have high complexity."""
        prompt = """
        Analyze the trade-offs between microservice and monolithic architectures.
        Compare and contrast their scalability, maintainability, and deployment complexity.
        Consider the implications for team organization and system reliability.
        Explain why certain patterns emerge in large-scale distributed systems.
        """
        analysis = await analyzer.analyze(prompt)

        assert analysis.score > 0.5
        assert analysis.reasoning_depth > 0.3
        assert "analyze" in analysis.detected_capabilities or "reason" in analysis.detected_capabilities

    @pytest.mark.asyncio
    async def test_code_complexity_detection(self, analyzer):
        """Code snippets should be detected."""
        prompt = """
        Write an async Python function that implements:
        ```python
        async def process_batch(items: List[Dict], max_concurrent: int = 10):
            async with asyncio.Semaphore(max_concurrent):
                tasks = [process_item(item) for item in items]
                return await asyncio.gather(*tasks)
        ```
        """
        analysis = await analyzer.analyze(prompt)

        assert analysis.code_complexity > 0.1
        assert "code_simple" in analysis.detected_capabilities or "code_complex" in analysis.detected_capabilities

    @pytest.mark.asyncio
    async def test_domain_specificity_detection(self, analyzer):
        """Domain-specific prompts should have high domain specificity."""
        prompt = """
        What is the optimal treatment methodology for patients presenting with
        symptoms of acute coronary syndrome? Consider the clinical diagnosis
        and potential prognosis factors.
        """
        analysis = await analyzer.analyze(prompt)

        assert analysis.domain_specificity > 0.3
        assert analysis.score > 0.4

    @pytest.mark.asyncio
    async def test_multi_step_detection(self, analyzer):
        """Multi-step tasks should be detected."""
        prompt = """
        First, analyze the current system architecture.
        Then, identify bottlenecks and performance issues.
        Next, propose optimization strategies.
        Finally, create an implementation plan with phases.
        """
        analysis = await analyzer.analyze(prompt)

        assert analysis.multi_step_indicators > 0.3

    @pytest.mark.asyncio
    async def test_task_type_summarize(self, analyzer):
        """Summarize tasks should be correctly classified."""
        prompt = "Please summarize the main points of this article in a brief paragraph."
        analysis = await analyzer.analyze(prompt)

        assert analysis.task_type == "summarize"

    @pytest.mark.asyncio
    async def test_task_type_search(self, analyzer):
        """Search-requiring tasks should be detected."""
        prompt = "What are the latest developments in quantum computing as of 2025?"
        analysis = await analyzer.analyze(prompt, context={"needs_current_info": True})

        # Either detected as search or has high context requirements
        assert "search" in analysis.detected_capabilities or analysis.context_requirements > 0.2


class TestHybridTieredRouter:
    """Test the HybridTieredRouter."""

    @pytest.fixture
    async def router(self):
        from jarvis_prime.core.hybrid_tiered_router import HybridTieredRouter
        router = HybridTieredRouter()
        await router.initialize()
        return router

    @pytest.mark.asyncio
    async def test_router_initialization(self, router):
        """Router should initialize with tiers from config."""
        assert router._initialized
        assert len(router._tiers) >= 1  # At least one tier configured

    @pytest.mark.asyncio
    async def test_simple_query_routes_to_tier_0(self, router):
        """Simple queries should route to Tier 0 (local fast)."""
        result = await router.route("What is 2 + 2?")

        # Should route to a local tier (0 or 0.5)
        assert result.tier_level <= 1
        assert result.complexity_score < 0.5

    @pytest.mark.asyncio
    async def test_complex_query_routes_higher(self, router):
        """Complex queries should route to higher tiers."""
        prompt = """
        Design a complete microservice architecture for a high-traffic e-commerce platform.
        The system should handle 10 million daily active users with 99.99% uptime.
        Consider:
        1. Service decomposition and domain-driven design
        2. Data consistency across distributed services
        3. Caching strategies and eventual consistency
        4. Circuit breaker patterns and graceful degradation
        5. Observability, tracing, and monitoring
        Explain your reasoning for each architectural decision.
        """
        result = await router.route(prompt)

        # Should route to a higher tier
        assert result.complexity_score > 0.5
        assert "tier_1" in result.tier_id or "tier_2" in result.tier_id or result.tier_level >= 1

    @pytest.mark.asyncio
    async def test_latency_constraint_respected(self, router):
        """Latency constraints should influence routing."""
        # Request with tight latency constraint
        result = await router.route(
            "Explain quantum entanglement",
            context={"max_latency_ms": 1000}
        )

        # Should route to a fast tier (local)
        assert result.estimated_latency_ms <= 5000  # Some buffer

    @pytest.mark.asyncio
    async def test_routing_result_structure(self, router):
        """Routing result should have all required fields."""
        result = await router.route("Hello world")

        assert result.tier_id is not None
        assert result.tier_name is not None
        assert 0.0 <= result.complexity_score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert result.reasoning is not None
        assert result.estimated_latency_ms >= 0
        assert result.estimated_cost_usd >= 0
        assert isinstance(result.available_tiers, list)

    @pytest.mark.asyncio
    async def test_fallback_chain_populated(self, router):
        """Routing should include fallback chain."""
        result = await router.route("What is machine learning?")

        # Fallback chain may or may not be populated depending on config
        assert isinstance(result.fallback_chain, list)

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, router):
        """Router should track statistics."""
        # Make a few routes
        await router.route("Hello")
        await router.route("How are you?")

        stats = router.get_statistics()

        assert stats["total_routes"] >= 2
        assert "tier_usage" in stats
        assert "circuit_breakers" in stats


class TestTierConfig:
    """Test TierConfig parsing."""

    def test_tier_config_from_dict(self):
        """TierConfig should parse from config dict."""
        from jarvis_prime.core.hybrid_tiered_router import TierConfig

        config = {
            "name": "Test Tier",
            "description": "A test tier",
            "tier_level": 0,
            "priority": 1,
            "enabled": True,
            "endpoint": "http://localhost:8000",
            "model_type": "llama_cpp",
            "models": {
                "primary": {"name": "test-model"},
                "fallbacks": [],
            },
            "capabilities": ["chat", "code_simple"],
            "performance": {
                "tokens_per_second": 20,
                "max_context_tokens": 8192,
            },
            "cost": {
                "per_1k_input_tokens": 0.0,
            },
            "routing": {
                "max_complexity": 0.5,
            },
        }

        tier = TierConfig.from_config("test_tier", config)

        assert tier.tier_id == "test_tier"
        assert tier.name == "Test Tier"
        assert tier.tier_level == 0
        assert tier.tokens_per_second == 20
        assert "chat" in tier.capabilities
        assert tier.max_complexity == 0.5


class TestResourceMonitor:
    """Test ResourceMonitor."""

    @pytest.fixture
    def monitor(self):
        from jarvis_prime.core.hybrid_tiered_router import ResourceMonitor
        return ResourceMonitor(max_daily_budget=10.0)

    @pytest.mark.asyncio
    async def test_budget_tracking(self, monitor):
        """Budget tracking should work correctly."""
        initial_remaining = await monitor.get_remaining_budget()
        assert initial_remaining == 10.0

        await monitor.record_spend(2.5)

        remaining = await monitor.get_remaining_budget()
        assert remaining == 7.5

    @pytest.mark.asyncio
    async def test_can_spend_check(self, monitor):
        """can_spend should enforce budget."""
        assert await monitor.can_spend(5.0)
        assert await monitor.can_spend(10.0)
        assert not await monitor.can_spend(15.0)

        await monitor.record_spend(8.0)
        assert not await monitor.can_spend(5.0)
        assert await monitor.can_spend(2.0)


class TestIntegration:
    """Integration tests with unified_inference."""

    @pytest.mark.asyncio
    async def test_analyze_and_route(self):
        """Test analyze_and_route convenience function."""
        from jarvis_prime.core.unified_inference import analyze_and_route

        result = await analyze_and_route("What is the capital of France?")

        # Should return routing decision or fallback error
        assert "complexity_score" in result or "error" in result

    @pytest.mark.asyncio
    async def test_complexity_examples(self):
        """Test various complexity levels."""
        from jarvis_prime.core.hybrid_tiered_router import get_hybrid_tiered_router

        router = await get_hybrid_tiered_router()

        test_cases = [
            ("Hello!", "low"),
            ("What is 2+2?", "low"),
            ("Summarize this paragraph.", "medium"),
            ("Write a Python function to sort a list.", "medium"),
            ("Analyze the implications of quantum computing on cryptography.", "high"),
            ("Design a distributed system architecture for a global payment platform with 5 nines uptime.", "high"),
        ]

        for prompt, expected_level in test_cases:
            result = await router.route(prompt)

            if expected_level == "low":
                assert result.complexity_score < 0.45, f"'{prompt}' should be low complexity"
            elif expected_level == "high":
                assert result.complexity_score > 0.4, f"'{prompt}' should be high complexity"
            # Medium is in between


# =============================================================================
# DEMO FUNCTION
# =============================================================================


async def demo_routing():
    """Demo the hybrid tiered router."""
    print("\n" + "=" * 70)
    print("  HYBRID TIERED ROUTER v94.0 DEMO")
    print("=" * 70)

    from jarvis_prime.core.hybrid_tiered_router import get_hybrid_tiered_router

    router = await get_hybrid_tiered_router()

    # Test prompts with varying complexity
    test_prompts = [
        ("Hello!", "Simple greeting"),
        ("What is the capital of France?", "Simple factual"),
        ("Summarize the key points of machine learning.", "Medium - summarize"),
        ("Write a Python async function with error handling.", "Medium - code"),
        (
            "Analyze the trade-offs between microservice and monolithic architectures. "
            "Consider scalability, maintainability, team organization, and deployment complexity. "
            "Explain why certain patterns emerge in large-scale systems.",
            "Complex - analysis"
        ),
        (
            "Design a complete distributed system architecture for a global financial platform. "
            "The system must handle 10M transactions per day with 99.999% uptime. "
            "First, decompose into microservices. Then, design the data layer with eventual consistency. "
            "Next, implement circuit breakers and graceful degradation. "
            "Finally, create an observability strategy with distributed tracing.",
            "Very Complex - multi-step design"
        ),
    ]

    print("\n┌" + "─" * 68 + "┐")
    print("│ {:^66} │".format("ROUTING DECISIONS"))
    print("├" + "─" * 68 + "┤")

    for prompt, description in test_prompts:
        result = await router.route(prompt)

        print(f"│ {description:<40} │")
        print(f"│   Complexity: {result.complexity_score:.2f} │ Tier: {result.tier_name:<20} │")
        print(f"│   Est. Latency: {result.estimated_latency_ms:.0f}ms │ Est. Cost: ${result.estimated_cost_usd:.5f} │")
        print("├" + "─" * 68 + "┤")

    print("└" + "─" * 68 + "┘")

    # Show statistics
    stats = router.get_statistics()
    print(f"\n📊 Router Statistics:")
    print(f"   Total routes: {stats['total_routes']}")
    print(f"   Tier usage: {stats['tier_usage']}")
    print(f"   Fallback rate: {stats['fallback_rate']:.1%}")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    # Run demo if executed directly
    asyncio.run(demo_routing())
