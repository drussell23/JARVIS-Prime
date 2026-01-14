#!/usr/bin/env python3
"""
Tests for Neural Orchestrator Core v100.0
==========================================

Comprehensive test suite covering:
- Task classification
- Memory monitoring
- Sticky routing
- Request buffering
- Circuit breakers
- Cross-repo state management
- Full routing pipeline
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None


class TestRunner:
    """Test runner with detailed reporting."""

    def __init__(self):
        self.results: List[TestResult] = []

    def record(self, name: str, passed: bool, duration_ms: float, error: Optional[str] = None):
        """Record a test result."""
        self.results.append(TestResult(name, passed, duration_ms, error))

    def summary(self) -> Dict[str, Any]:
        """Get test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        total_time = sum(r.duration_ms for r in self.results)

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0,
            "total_time_ms": total_time,
        }

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("NEURAL ORCHESTRATOR CORE v100.0 - TEST RESULTS")
        print("=" * 70)

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status}: {result.name} ({result.duration_ms:.1f}ms)")
            if result.error:
                print(f"         Error: {result.error}")

        summary = self.summary()
        print("\n" + "-" * 70)
        print(f"  Total: {summary['total']} | Passed: {summary['passed']} | "
              f"Failed: {summary['failed']} | Time: {summary['total_time_ms']:.1f}ms")
        print("=" * 70)


async def test_dynamic_config(runner: TestRunner):
    """Test DynamicConfig creation and env var override."""
    start = time.time()
    try:
        import os
        from jarvis_prime.core.neural_orchestrator_core import DynamicConfig

        # Test default creation
        config = DynamicConfig.from_env_and_yaml()
        assert config.memory_warn_threshold == 0.80
        assert config.memory_critical_threshold == 0.90
        assert config.tier_0_fast_max_complexity == 0.35

        # Test env var override
        os.environ["NEURAL_ORCHESTRATOR_MEMORY_WARN_THRESHOLD"] = "0.75"
        config2 = DynamicConfig.from_env_and_yaml()
        assert config2.memory_warn_threshold == 0.75
        del os.environ["NEURAL_ORCHESTRATOR_MEMORY_WARN_THRESHOLD"]

        runner.record("DynamicConfig creation", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("DynamicConfig creation", False, (time.time() - start) * 1000, str(e))


async def test_task_classifier(runner: TestRunner):
    """Test UnifiedTaskClassifier."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            DynamicConfig, UnifiedTaskClassifier, TaskCategory, RoutingTier
        )

        config = DynamicConfig.from_env_and_yaml()
        classifier = UnifiedTaskClassifier(config)

        # Test greeting classification
        result = await classifier.classify("Hello!")
        assert result.task_type == TaskCategory.GREETING
        assert result.complexity < 0.2
        assert result.recommended_tier == RoutingTier.TIER_0_ULTRA_FAST

        # Test voice command classification
        result = await classifier.classify("Hey Jarvis", {"voice_mode": True})
        assert result.task_type == TaskCategory.VOICE
        assert result.requires_fast_response

        # Test code classification
        result = await classifier.classify(
            "Implement a distributed cache with Redis backend. "
            "Handle connection pooling and automatic reconnection. "
            "```python\nclass RedisCache:\n    pass\n```"
        )
        assert result.task_type == TaskCategory.CODE
        assert result.complexity > 0.2  # Adjusted threshold

        # Test reasoning classification - use a prompt without code keywords
        result = await classifier.classify(
            "Why is the sky blue? Explain the physics and reason through "
            "how light scattering works in the atmosphere."
        )
        assert result.task_type == TaskCategory.REASONING

        runner.record("UnifiedTaskClassifier", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("UnifiedTaskClassifier", False, (time.time() - start) * 1000, str(e))


async def test_memory_monitor(runner: TestRunner):
    """Test UnifiedMemoryMonitor."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            DynamicConfig, UnifiedMemoryMonitor, MemoryPressureLevel
        )

        config = DynamicConfig.from_env_and_yaml()
        monitor = UnifiedMemoryMonitor(config)
        await monitor.initialize()

        # Test pressure level
        level = await monitor.get_pressure_level()
        assert isinstance(level, MemoryPressureLevel)

        # Test detailed snapshot
        snapshot = await monitor.get_detailed_snapshot()
        assert "pressure_level" in snapshot
        assert "is_critical" in snapshot
        assert "should_burst" in snapshot
        assert "trend" in snapshot

        # Test caching (second call should be cached)
        start_cached = time.time()
        level2 = await monitor.get_pressure_level()
        cached_time = time.time() - start_cached
        assert cached_time < 0.01  # Should be very fast (cached)

        runner.record("UnifiedMemoryMonitor", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("UnifiedMemoryMonitor", False, (time.time() - start) * 1000, str(e))


async def test_sticky_routing(runner: TestRunner):
    """Test UnifiedStickyRouting."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            DynamicConfig, UnifiedStickyRouting, UnifiedTaskClassifier,
            TaskCategory, RoutingTier
        )

        config = DynamicConfig.from_env_and_yaml()
        sticky = UnifiedStickyRouting(config)
        classifier = UnifiedTaskClassifier(config)

        # Classify a coding task
        classification = await classifier.classify(
            "Implement a binary search algorithm in Python"
        )

        # Initially no sticky model
        model = await sticky.get_sticky_model(classification)
        assert model is None

        # Update sticky routing
        await sticky.update_sticky("llama-3-8b", RoutingTier.TIER_0_FAST, classification)

        # Now should have sticky model
        classification2 = await classifier.classify("Fix the bug in the search function")
        model = await sticky.get_sticky_model(classification2)
        assert model == "llama-3-8b"

        # Test session stats
        stats = sticky.get_stats()
        assert stats["sticky_model"] == "llama-3-8b"
        assert stats["session_active"] is True

        # End session
        await sticky.end_session()
        model = await sticky.get_sticky_model(classification)
        assert model is None

        runner.record("UnifiedStickyRouting", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("UnifiedStickyRouting", False, (time.time() - start) * 1000, str(e))


async def test_request_buffer(runner: TestRunner):
    """Test UnifiedRequestBuffer."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            DynamicConfig, UnifiedRequestBuffer
        )

        config = DynamicConfig.from_env_and_yaml()
        buffer = UnifiedRequestBuffer(config)

        # Initially not buffering
        assert not buffer.is_buffering()

        # Start buffering
        await buffer.start_buffering()
        assert buffer.is_buffering()

        # Enqueue requests
        future1 = await buffer.enqueue("req1", "Test prompt 1", {"key": "value"})
        future2 = await buffer.enqueue("req2", "Test prompt 2")

        stats = buffer.get_stats()
        assert stats["queue_length"] == 2
        assert stats["buffered_total"] == 2

        # Flush with processor
        processed_prompts = []

        async def processor(prompt: str, context: Dict[str, Any]) -> str:
            processed_prompts.append(prompt)
            return f"Processed: {prompt}"

        count = await buffer.flush_to(processor)
        assert count == 2
        assert len(processed_prompts) == 2

        # Check futures completed
        assert future1.done()
        assert future2.done()
        assert "Processed" in future1.result()

        # Stop buffering
        await buffer.stop_buffering()
        assert not buffer.is_buffering()

        runner.record("UnifiedRequestBuffer", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("UnifiedRequestBuffer", False, (time.time() - start) * 1000, str(e))


async def test_circuit_breaker(runner: TestRunner):
    """Test CircuitBreaker and CircuitBreakerManager."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            CircuitBreaker, CircuitState, DynamicConfig, CircuitBreakerManager
        )

        # Test individual circuit breaker
        breaker = CircuitBreaker(
            name="test_breaker",
            failure_threshold=3,
            recovery_timeout=1.0,
            half_open_max_calls=2,
        )

        assert breaker.get_state() == CircuitState.CLOSED
        assert await breaker.can_proceed()

        # Record failures
        await breaker.record_failure()
        await breaker.record_failure()
        await breaker.record_failure()

        # Should be open now
        assert breaker.get_state() == CircuitState.OPEN
        assert not await breaker.can_proceed()

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should be half-open
        assert await breaker.can_proceed()
        assert breaker.get_state() == CircuitState.HALF_OPEN

        # Record success in half-open
        await breaker.record_success()
        await breaker.record_success()

        # Should be closed again
        assert breaker.get_state() == CircuitState.CLOSED

        # Test CircuitBreakerManager
        config = DynamicConfig.from_env_and_yaml()
        manager = CircuitBreakerManager(config)

        can_proceed = await manager.can_proceed("tier_1_cloud")
        assert can_proceed

        await manager.record_failure("tier_1_cloud")
        stats = manager.get_all_stats()
        assert "tier_1_cloud" in stats

        runner.record("CircuitBreaker", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("CircuitBreaker", False, (time.time() - start) * 1000, str(e))


async def test_cross_repo_state_manager(runner: TestRunner):
    """Test CrossRepoStateManager."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            DynamicConfig, CrossRepoStateManager, MemoryPressureLevel
        )
        import tempfile
        import shutil

        # Use temp directory for test
        temp_dir = tempfile.mkdtemp()
        config = DynamicConfig.from_env_and_yaml()
        config = DynamicConfig(
            **{**config.__dict__, "cross_repo_state_dir": temp_dir}
        )

        manager = CrossRepoStateManager(config)
        await manager.initialize()

        # Write state
        await manager.write_memory_state(
            pressure=MemoryPressureLevel.WARN,
            should_burst=True,
            metadata={"test": True},
        )

        # Read state (should find our written state)
        state = await manager.read_cross_repo_state()
        assert state is not None
        assert state.jarvis_prime_pressure == MemoryPressureLevel.WARN
        assert state.should_burst is True

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

        runner.record("CrossRepoStateManager", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("CrossRepoStateManager", False, (time.time() - start) * 1000, str(e))


async def test_neural_orchestrator_core(runner: TestRunner):
    """Test full NeuralOrchestratorCore pipeline."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            NeuralOrchestratorCore, DynamicConfig, RoutingTier,
            RoutingDecisionReason
        )

        # Reset any existing instance
        await NeuralOrchestratorCore.reset_instance()

        # Get instance
        orchestrator = await NeuralOrchestratorCore.get_instance()

        # Test routing - greeting
        result = await orchestrator.route("Hello!")
        assert result is not None
        assert result.tier in {RoutingTier.TIER_0_ULTRA_FAST, RoutingTier.TIER_0_FAST}

        # Test routing - code
        result = await orchestrator.route(
            "Implement a REST API endpoint for user authentication"
        )
        assert result is not None
        assert result.task_classification is not None

        # Test routing with context
        result = await orchestrator.route(
            "What time is it?",
            context={"voice_mode": True, "session_id": "test123"}
        )
        assert result is not None

        # Test memory status
        status = await orchestrator.get_memory_status()
        assert "pressure_level" in status
        assert "is_critical" in status

        # Test comprehensive stats
        stats = orchestrator.get_comprehensive_stats()
        assert stats["version"] == "100.0"
        assert "routing" in stats
        assert "task_classifier" in stats
        assert "memory_monitor" in stats

        # Test health status
        health = orchestrator.get_health_status()
        assert health["status"] == "healthy"
        assert health["version"] == "100.0"

        # Shutdown
        await orchestrator.shutdown()

        runner.record("NeuralOrchestratorCore pipeline", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("NeuralOrchestratorCore pipeline", False, (time.time() - start) * 1000, str(e))


async def test_context_variables(runner: TestRunner):
    """Test context variables for distributed tracing."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            set_request_context, get_request_context,
            request_id_var, session_id_var
        )

        # Set context
        set_request_context(
            request_id="req-123",
            session_id="sess-456",
            custom_field="value",
        )

        # Get context
        ctx = get_request_context()
        assert ctx["request_id"] == "req-123"
        assert ctx["session_id"] == "sess-456"
        assert ctx["custom_field"] == "value"

        # Direct access
        assert request_id_var.get() == "req-123"
        assert session_id_var.get() == "sess-456"

        runner.record("Context variables", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("Context variables", False, (time.time() - start) * 1000, str(e))


async def test_decorators(runner: TestRunner):
    """Test defensive decorators."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            defensive, async_defensive, with_timeout, with_retry
        )

        # Test defensive decorator
        @defensive(fallback="fallback_value")
        def risky_sync():
            raise ValueError("Oops")

        result = risky_sync()
        assert result == "fallback_value"

        # Test async_defensive decorator
        @async_defensive(fallback="async_fallback")
        async def risky_async():
            raise ValueError("Async oops")

        result = await risky_async()
        assert result == "async_fallback"

        # Test with_timeout
        @with_timeout(0.5)
        async def slow_operation():
            await asyncio.sleep(0.1)
            return "completed"

        result = await slow_operation()
        assert result == "completed"

        # Test timeout actually works
        @with_timeout(0.1)
        async def too_slow():
            await asyncio.sleep(1.0)
            return "never"

        try:
            await too_slow()
            assert False, "Should have timed out"
        except asyncio.TimeoutError:
            pass  # Expected

        runner.record("Decorators", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("Decorators", False, (time.time() - start) * 1000, str(e))


async def test_module_functions(runner: TestRunner):
    """Test module-level convenience functions."""
    start = time.time()
    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            get_neural_orchestrator, neural_route, shutdown_neural_orchestrator,
            NeuralOrchestratorCore
        )

        # Reset any existing instance
        await NeuralOrchestratorCore.reset_instance()

        # Get orchestrator
        orchestrator = await get_neural_orchestrator()
        assert orchestrator is not None

        # Convenience routing
        result = await neural_route("Test prompt")
        assert result is not None
        assert result.request_id is not None

        # Shutdown
        await shutdown_neural_orchestrator()

        runner.record("Module functions", True, (time.time() - start) * 1000)
    except Exception as e:
        runner.record("Module functions", False, (time.time() - start) * 1000, str(e))


async def run_all_tests():
    """Run all tests."""
    runner = TestRunner()

    print("\n" + "=" * 70)
    print("NEURAL ORCHESTRATOR CORE v100.0 - RUNNING TESTS")
    print("=" * 70 + "\n")

    # Run tests in order
    tests = [
        test_dynamic_config,
        test_task_classifier,
        test_memory_monitor,
        test_sticky_routing,
        test_request_buffer,
        test_circuit_breaker,
        test_cross_repo_state_manager,
        test_neural_orchestrator_core,
        test_context_variables,
        test_decorators,
        test_module_functions,
    ]

    for test in tests:
        print(f"Running: {test.__name__}...")
        try:
            await test(runner)
        except Exception as e:
            runner.record(test.__name__, False, 0, str(e))

    runner.print_summary()

    # Return exit code
    summary = runner.summary()
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
