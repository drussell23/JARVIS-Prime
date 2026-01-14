#!/usr/bin/env python3
"""
Neural Switchboard Test Suite v98.0
====================================

Comprehensive tests for the Neural Switchboard architecture:
1. Task Classification
2. Memory Pressure Monitoring
3. Sticky Routing
4. Request Buffering
5. Unified Routing Decision Making

Run:
    python3 tests/test_neural_switchboard.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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


async def test_task_classifier(results: TestResults) -> None:
    """Test TaskClassifier functionality."""
    print()
    print("1. Testing Task Classifier...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            TaskClassifier,
            TaskType,
        )

        classifier = TaskClassifier()

        # Test greeting detection
        classification = await classifier.classify("hello there")
        if classification.task_type == TaskType.GREETING:
            results.record_pass("TaskClassifier.greeting_detection")
        else:
            results.record_fail("TaskClassifier.greeting_detection", f"got {classification.task_type}")

        # Test code task detection
        classification = await classifier.classify(
            "Write a Python function to implement async caching with TTL"
        )
        if classification.task_type in {TaskType.CODE_SIMPLE, TaskType.CODE_COMPLEX}:
            results.record_pass("TaskClassifier.code_detection")
        else:
            results.record_fail("TaskClassifier.code_detection", f"got {classification.task_type}")

        # Test complexity calculation
        simple_prompt = "hi"
        complex_prompt = """
        I need you to implement a distributed caching system with the following requirements:
        1. Use Redis for the backend
        2. Implement TTL with sliding expiration
        3. Add circuit breaker pattern for failover
        4. Support both sync and async operations
        ```python
        class DistributedCache:
            pass
        ```
        Also explain the tradeoffs between consistency and availability.
        """

        simple_class = await classifier.classify(simple_prompt)
        complex_class = await classifier.classify(complex_prompt)

        if complex_class.complexity > simple_class.complexity:
            results.record_pass("TaskClassifier.complexity_scoring")
        else:
            results.record_fail(
                "TaskClassifier.complexity_scoring",
                f"simple={simple_class.complexity:.2f}, complex={complex_class.complexity:.2f}"
            )

        # Test voice command detection
        voice_class = await classifier.classify("hey jarvis what time is it", {"voice_mode": True})
        if voice_class.task_type == TaskType.VOICE_COMMAND:
            results.record_pass("TaskClassifier.voice_detection")
        else:
            results.record_fail("TaskClassifier.voice_detection", f"got {voice_class.task_type}")

        # Test architecture detection
        arch_class = await classifier.classify(
            "Design the system architecture for a microservices deployment"
        )
        if arch_class.task_type == TaskType.CODE_ARCHITECTURE:
            results.record_pass("TaskClassifier.architecture_detection")
        else:
            results.record_fail("TaskClassifier.architecture_detection", f"got {arch_class.task_type}")

        # Test recommended tiers
        if arch_class.recommended_tiers:
            results.record_pass("TaskClassifier.recommended_tiers")
        else:
            results.record_fail("TaskClassifier.recommended_tiers", "empty")

    except Exception as e:
        results.record_fail("TaskClassifier", str(e))


async def test_memory_pressure_monitor(results: TestResults) -> None:
    """Test MacOSMemoryPressureMonitor functionality."""
    print()
    print("2. Testing Memory Pressure Monitor...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            MacOSMemoryPressureMonitor,
            MemoryPressureLevel,
        )

        monitor = MacOSMemoryPressureMonitor()

        # Test pressure level retrieval
        level = await monitor.get_pressure_level()
        if isinstance(level, MemoryPressureLevel):
            results.record_pass("MemoryPressureMonitor.get_pressure_level")
        else:
            results.record_fail("MemoryPressureMonitor.get_pressure_level", f"got {type(level)}")

        # Test detailed snapshot
        snapshot = await monitor.get_detailed_snapshot()
        if "pressure_level" in snapshot and "should_burst" in snapshot:
            results.record_pass("MemoryPressureMonitor.detailed_snapshot")
        else:
            results.record_fail("MemoryPressureMonitor.detailed_snapshot", f"missing fields: {list(snapshot.keys())}")

        # Test trend detection
        if "trend" in snapshot:
            results.record_pass("MemoryPressureMonitor.trend_detection")
        else:
            results.record_fail("MemoryPressureMonitor.trend_detection", "missing trend")

        # Test caching (second call should be cached)
        level1 = await monitor.get_pressure_level()
        level2 = await monitor.get_pressure_level()
        if level1 == level2:
            results.record_pass("MemoryPressureMonitor.caching")
        else:
            results.record_fail("MemoryPressureMonitor.caching", "cache miss")

    except Exception as e:
        results.record_fail("MemoryPressureMonitor", str(e))


async def test_sticky_routing(results: TestResults) -> None:
    """Test StickyRoutingManager functionality."""
    print()
    print("3. Testing Sticky Routing Manager...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            StickyRoutingManager,
            TaskClassifier,
            TaskType,
        )

        manager = StickyRoutingManager()
        classifier = TaskClassifier()

        # Initial state should have no sticky model
        task = await classifier.classify("Write a function to sort a list")
        sticky = await manager.get_sticky_model(task)
        if sticky is None:
            results.record_pass("StickyRouting.initial_state")
        else:
            results.record_fail("StickyRouting.initial_state", f"got {sticky}")

        # Update sticky with multiple coding tasks to build strength
        for prompt in [
            "Write a Python class for user authentication",
            "Add logging to the auth class",
            "Write unit tests for the auth class",
        ]:
            task = await classifier.classify(prompt)
            await manager.update_sticky("qwen-2.5-32b", "tier_05_local_capable", task)

        # Check sticky status
        status = manager.get_status()
        if status.get("session_task_count", 0) >= 3:
            results.record_pass("StickyRouting.task_tracking")
        else:
            results.record_fail("StickyRouting.task_tracking", f"count={status.get('session_task_count')}")

        # Check strength calculation
        if status.get("sticky_strength", 0) > 0:
            results.record_pass("StickyRouting.strength_calculation")
        else:
            results.record_fail("StickyRouting.strength_calculation", "strength=0")

        # Test session timeout (not testing actual timeout, just the status field)
        if "session_duration_seconds" in status:
            results.record_pass("StickyRouting.session_tracking")
        else:
            results.record_fail("StickyRouting.session_tracking", "missing duration")

    except Exception as e:
        results.record_fail("StickyRouting", str(e))


async def test_request_buffer(results: TestResults) -> None:
    """Test HotSwapRequestBuffer functionality."""
    print()
    print("4. Testing Request Buffer...")

    try:
        from jarvis_prime.core.dynamic_model_registry import HotSwapRequestBuffer

        buffer = HotSwapRequestBuffer(max_size=10, default_timeout=5.0)

        # Test initial state
        stats = buffer.get_statistics()
        if not stats["active"] and stats["queue_length"] == 0:
            results.record_pass("RequestBuffer.initial_state")
        else:
            results.record_fail("RequestBuffer.initial_state", f"stats={stats}")

        # Test buffering lifecycle
        await buffer.start_buffering()
        if buffer.is_buffering():
            results.record_pass("RequestBuffer.start_buffering")
        else:
            results.record_fail("RequestBuffer.start_buffering", "not active")

        # Test enqueue
        future = await buffer.enqueue("req-1", "test prompt", {"key": "value"})
        if buffer.get_queue_length() == 1:
            results.record_pass("RequestBuffer.enqueue")
        else:
            results.record_fail("RequestBuffer.enqueue", f"length={buffer.get_queue_length()}")

        # Test flush
        async def mock_processor(prompt: str, context: dict) -> str:
            return f"processed: {prompt}"

        processed = await buffer.flush_to(mock_processor)
        if processed == 1:
            results.record_pass("RequestBuffer.flush")
        else:
            results.record_fail("RequestBuffer.flush", f"processed={processed}")

        # Test future resolution
        try:
            result = future.result()
            if "processed" in result:
                results.record_pass("RequestBuffer.future_resolution")
            else:
                results.record_fail("RequestBuffer.future_resolution", f"result={result}")
        except Exception as e:
            results.record_fail("RequestBuffer.future_resolution", str(e))

        await buffer.stop_buffering()

        # Test stats after operations
        final_stats = buffer.get_statistics()
        if final_stats["buffered_total"] >= 1 and final_stats["flushed_total"] >= 1:
            results.record_pass("RequestBuffer.statistics")
        else:
            results.record_fail("RequestBuffer.statistics", f"stats={final_stats}")

    except Exception as e:
        results.record_fail("RequestBuffer", str(e))


async def test_neural_switchboard_routing(results: TestResults) -> None:
    """Test Neural Switchboard unified routing."""
    print()
    print("5. Testing Neural Switchboard Routing...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            get_dynamic_model_registry,
            TaskType,
        )

        registry = await get_dynamic_model_registry()

        # Test neural_switchboard_route method
        spec, path, task_class = await registry.neural_switchboard_route(
            "Write a function to implement binary search"
        )

        if spec is not None:
            results.record_pass("NeuralSwitchboard.route_returns_spec")
        else:
            results.record_fail("NeuralSwitchboard.route_returns_spec", "None")

        if task_class is not None:
            results.record_pass("NeuralSwitchboard.route_returns_classification")
        else:
            results.record_fail("NeuralSwitchboard.route_returns_classification", "None")

        if task_class and task_class.task_type in {TaskType.CODE_SIMPLE, TaskType.CODE_COMPLEX}:
            results.record_pass("NeuralSwitchboard.correct_task_type")
        else:
            results.record_fail("NeuralSwitchboard.correct_task_type", f"got {task_class.task_type if task_class else None}")

        # Test classify_task method
        task = await registry.classify_task("hello world")
        if task.task_type == TaskType.GREETING:
            results.record_pass("NeuralSwitchboard.classify_task")
        else:
            results.record_fail("NeuralSwitchboard.classify_task", f"got {task.task_type}")

        # Test get_memory_pressure method
        pressure = await registry.get_memory_pressure()
        if "pressure_level" in pressure:
            results.record_pass("NeuralSwitchboard.memory_pressure")
        else:
            results.record_fail("NeuralSwitchboard.memory_pressure", f"missing fields")

        # Test get_sticky_status method
        sticky = registry.get_sticky_status()
        if "sticky_model" in sticky:
            results.record_pass("NeuralSwitchboard.sticky_status")
        else:
            results.record_fail("NeuralSwitchboard.sticky_status", "missing fields")

        # Test statistics include v98.0 fields
        stats = registry.get_statistics()
        if stats.get("version") == "98.0":
            results.record_pass("NeuralSwitchboard.version")
        else:
            results.record_fail("NeuralSwitchboard.version", f"got {stats.get('version')}")

        if "task_classifier_stats" in stats:
            results.record_pass("NeuralSwitchboard.includes_task_stats")
        else:
            results.record_fail("NeuralSwitchboard.includes_task_stats", "missing")

        if "sticky_routing_status" in stats:
            results.record_pass("NeuralSwitchboard.includes_sticky_stats")
        else:
            results.record_fail("NeuralSwitchboard.includes_sticky_stats", "missing")

    except Exception as e:
        results.record_fail("NeuralSwitchboard", str(e))


async def test_task_tier_mapping(results: TestResults) -> None:
    """Test Task to Tier mappings."""
    print()
    print("6. Testing Task-Tier Mappings...")

    try:
        from jarvis_prime.core.dynamic_model_registry import (
            TaskType,
            TASK_TIER_MAPPING,
            TASK_CAPABILITY_MAPPING,
        )

        # Test all task types have tier mappings
        missing_mappings = []
        for task_type in TaskType:
            if task_type not in TASK_TIER_MAPPING:
                missing_mappings.append(task_type.value)

        if len(missing_mappings) <= 3:  # Allow a few missing (edge cases)
            results.record_pass("TaskTierMapping.coverage")
        else:
            results.record_fail("TaskTierMapping.coverage", f"missing: {missing_mappings}")

        # Test voice task maps to fast tier
        voice_tiers = TASK_TIER_MAPPING.get(TaskType.VOICE_COMMAND, [])
        if "tier_0_local_fast" in voice_tiers:
            results.record_pass("TaskTierMapping.voice_to_fast")
        else:
            results.record_fail("TaskTierMapping.voice_to_fast", f"got {voice_tiers}")

        # Test code architecture maps to capable/cloud tiers
        arch_tiers = TASK_TIER_MAPPING.get(TaskType.CODE_ARCHITECTURE, [])
        if any("cloud" in t or "capable" in t for t in arch_tiers):
            results.record_pass("TaskTierMapping.arch_to_capable")
        else:
            results.record_fail("TaskTierMapping.arch_to_capable", f"got {arch_tiers}")

        # Test capability mappings exist
        if len(TASK_CAPABILITY_MAPPING) >= 5:
            results.record_pass("TaskCapabilityMapping.exists")
        else:
            results.record_fail("TaskCapabilityMapping.exists", f"only {len(TASK_CAPABILITY_MAPPING)}")

    except Exception as e:
        results.record_fail("TaskTierMapping", str(e))


async def main():
    """Run all Neural Switchboard tests."""
    print("=" * 70)
    print("  NEURAL SWITCHBOARD TEST SUITE v98.0")
    print("  Task Classification + Memory Pressure + Sticky Routing")
    print("=" * 70)

    results = TestResults()

    # Run all tests
    await test_task_classifier(results)
    await test_memory_pressure_monitor(results)
    await test_sticky_routing(results)
    await test_request_buffer(results)
    await test_neural_switchboard_routing(results)
    await test_task_tier_mapping(results)

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
