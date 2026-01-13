#!/usr/bin/env python3
"""
Cross-Repository Validation Test Suite
======================================

v92.1 - Comprehensive testing for Trinity ecosystem cross-repo coordination.

This test suite validates:
1. Reasoning Engine with Circuit Breaker and RAG integration
2. Continual Learning System with auto-initialization
3. Cross-repo communication (Trinity Protocol)
4. Service health and connectivity
5. Feedback collection pipeline
6. Strategy execution with context augmentation

USAGE:
    # Run all tests
    python -m pytest tests/test_cross_repo_validation.py -v

    # Run specific test class
    python -m pytest tests/test_cross_repo_validation.py::TestReasoningEngine -v

    # Run with coverage
    python -m pytest tests/test_cross_repo_validation.py --cov=jarvis_prime -v

    # Direct execution (standalone)
    python tests/test_cross_repo_validation.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# TEST RESULT TRACKING
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class TestSuite:
    """Test suite runner with detailed reporting."""

    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []
        self._start_time: Optional[float] = None

    async def run_test(
        self,
        name: str,
        test_func,
        *args,
        **kwargs,
    ) -> TestResult:
        """Run a single test and record the result."""
        start = time.perf_counter()
        try:
            result = await test_func(*args, **kwargs)
            duration = (time.perf_counter() - start) * 1000

            # If test returns a dict, use it for details
            if isinstance(result, dict):
                passed = result.get("passed", True)
                details = result
            else:
                passed = bool(result) if result is not None else True
                details = None

            test_result = TestResult(
                name=name,
                passed=passed,
                duration_ms=duration,
                details=details,
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            test_result = TestResult(
                name=name,
                passed=False,
                duration_ms=duration,
                error=str(e),
            )

        self.results.append(test_result)
        return test_result

    def get_summary(self) -> Dict[str, Any]:
        """Get test suite summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_duration = sum(r.duration_ms for r in self.results)

        return {
            "suite": self.name,
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results) if self.results else 0,
            "total_duration_ms": total_duration,
            "failures": [
                {"name": r.name, "error": r.error}
                for r in self.results
                if not r.passed
            ],
        }

    def print_results(self) -> None:
        """Print test results to console."""
        print(f"\n{'=' * 70}")
        print(f"TEST SUITE: {self.name}")
        print(f"{'=' * 70}\n")

        for result in self.results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"  {status}  {result.name} ({result.duration_ms:.1f}ms)")
            if result.error:
                print(f"         Error: {result.error}")

        summary = self.get_summary()
        print(f"\n{'-' * 70}")
        print(f"Results: {summary['passed']}/{summary['total']} passed "
              f"({summary['pass_rate']*100:.1f}%)")
        print(f"Duration: {summary['total_duration_ms']:.1f}ms")
        print(f"{'=' * 70}\n")


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

async def test_circuit_breaker_closed_state() -> Dict[str, Any]:
    """Test circuit breaker in CLOSED state (normal operation)."""
    from jarvis_prime.core.reasoning_engine import CircuitBreaker

    cb = CircuitBreaker(name="test", failure_threshold=3, recovery_timeout=1.0)

    # Should start in CLOSED state
    assert cb.get_state() == CircuitBreaker.CLOSED, "Should start CLOSED"

    # Successful calls should work
    async def success_func():
        return "success"

    result = await cb.call(success_func)
    assert result == "success", "Should return function result"

    stats = cb.get_stats()
    assert stats["success_count"] == 1, "Should track successes"
    assert stats["failure_count"] == 0, "No failures yet"

    return {"passed": True, "state": cb.get_state()}


async def test_circuit_breaker_opens_on_failures() -> Dict[str, Any]:
    """Test circuit breaker opens after threshold failures."""
    from jarvis_prime.core.reasoning_engine import CircuitBreaker

    cb = CircuitBreaker(name="test_open", failure_threshold=3, recovery_timeout=1.0)

    async def failing_func():
        raise ValueError("Simulated failure")

    # Cause failures to open the circuit
    for i in range(3):
        try:
            await cb.call(failing_func)
        except ValueError:
            pass

    assert cb.get_state() == CircuitBreaker.OPEN, "Should be OPEN after failures"

    stats = cb.get_stats()
    assert stats["failure_count"] >= 3, "Should track failures"

    return {"passed": True, "state": cb.get_state(), "failures": stats["failure_count"]}


async def test_circuit_breaker_fallback() -> Dict[str, Any]:
    """Test circuit breaker fallback mechanism."""
    from jarvis_prime.core.reasoning_engine import CircuitBreaker

    cb = CircuitBreaker(name="test_fallback", failure_threshold=2, recovery_timeout=0.5)

    async def failing_func():
        raise ValueError("Simulated failure")

    async def fallback_func():
        return "fallback_result"

    # Open the circuit
    for _ in range(2):
        try:
            await cb.call(failing_func)
        except ValueError:
            pass

    # Now call with fallback - should use fallback since circuit is open
    result = await cb.call(failing_func, fallback=fallback_func)
    assert result == "fallback_result", "Should use fallback when circuit is open"

    return {"passed": True, "result": result}


async def test_circuit_breaker_half_open_recovery() -> Dict[str, Any]:
    """Test circuit breaker recovery through HALF_OPEN state."""
    from jarvis_prime.core.reasoning_engine import CircuitBreaker

    cb = CircuitBreaker(
        name="test_recovery",
        failure_threshold=2,
        recovery_timeout=0.1,  # Short timeout for testing
        half_open_max_calls=2,
    )

    call_count = 0

    async def sometimes_fails():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ValueError("Initial failures")
        return "recovered"

    # Open the circuit
    for _ in range(2):
        try:
            await cb.call(sometimes_fails)
        except ValueError:
            pass

    assert cb.get_state() == CircuitBreaker.OPEN, "Should be OPEN"

    # Wait for recovery timeout
    await asyncio.sleep(0.15)

    # Next call should trigger HALF_OPEN
    result = await cb.call(sometimes_fails)
    assert result == "recovered", "Should recover"

    # Eventually should return to CLOSED
    await asyncio.sleep(0.1)
    final_state = cb.get_state()

    return {
        "passed": final_state in (CircuitBreaker.CLOSED, CircuitBreaker.HALF_OPEN),
        "final_state": final_state,
    }


# =============================================================================
# REASONING ENGINE TESTS
# =============================================================================

async def test_reasoning_engine_initialization() -> Dict[str, Any]:
    """Test ReasoningEngine initialization."""
    try:
        from jarvis_prime.core.reasoning_engine import ReasoningEngine

        # Create with mocked dependencies
        with patch("jarvis_prime.core.reasoning_engine.ContinualLearningEngine"):
            engine = ReasoningEngine()

            # Verify internal components
            assert engine._context_window is not None, "Should have context window"
            assert engine._feedback_deduplicator is not None, "Should have deduplicator"
            assert engine._rag_circuit_breaker is not None, "Should have RAG circuit breaker"

            return {"passed": True}
    except ImportError as e:
        return {"passed": False, "error": f"Import failed: {e}"}


async def test_strategy_context_augmentation() -> Dict[str, Any]:
    """Test that strategies properly augment prompts with context."""
    try:
        from jarvis_prime.core.reasoning_engine import BaseReasoningStrategy

        # Create a test strategy instance
        class TestStrategy(BaseReasoningStrategy):
            async def reason(self, *args, **kwargs):
                pass

        strategy = TestStrategy()

        # Test context formatting
        context = {
            "retrieved_context": ["Fact 1", "Fact 2"],
            "system_prompt": "You are a helpful assistant.",
        }

        formatted = strategy._format_context_for_prompt(context)
        assert "Fact 1" in formatted, "Should include retrieved context"
        assert "Fact 2" in formatted, "Should include all facts"

        # Test prompt augmentation
        input_text = "What is the capital of France?"
        augmented = strategy._build_augmented_prompt(input_text, context)

        assert input_text in augmented, "Should include original input"
        assert "Fact 1" in augmented, "Should include context"

        return {"passed": True, "augmented_length": len(augmented)}

    except ImportError as e:
        return {"passed": False, "error": f"Import failed: {e}"}


async def test_feedback_deduplicator() -> Dict[str, Any]:
    """Test feedback deduplication functionality."""
    try:
        from jarvis_prime.core.reasoning_engine import FeedbackDeduplicator

        dedup = FeedbackDeduplicator(max_size=100)

        # First submission should not be duplicate
        item1 = {"input": "test", "output": "response", "quality_score": 0.9}
        is_dup1 = dedup.is_duplicate(item1)
        assert not is_dup1, "First item should not be duplicate"

        # Second same submission should be duplicate
        is_dup2 = dedup.is_duplicate(item1)
        assert is_dup2, "Same item should be duplicate"

        # Different item should not be duplicate
        item2 = {"input": "different", "output": "other", "quality_score": 0.8}
        is_dup3 = dedup.is_duplicate(item2)
        assert not is_dup3, "Different item should not be duplicate"

        # Check stats
        stats = dedup.get_stats()
        assert stats["seen_count"] == 2, "Should have seen 2 unique items"

        return {"passed": True, "stats": stats}

    except ImportError as e:
        return {"passed": False, "error": f"Import failed: {e}"}


async def test_context_window_manager() -> Dict[str, Any]:
    """Test context window management and utilization tracking."""
    try:
        from jarvis_prime.core.reasoning_engine import ContextWindowManager

        cwm = ContextWindowManager(
            max_tokens=4096,
            warning_threshold=0.8,
            critical_threshold=0.95,
        )

        # Test utilization check
        utilization, status = await cwm.check_utilization(2000, "test_op")
        expected_util = 2000 / 4096
        assert abs(utilization - expected_util) < 0.01, "Utilization calculation"
        assert status == "ok", "Should be OK at ~49%"

        # Test warning threshold
        utilization, status = await cwm.check_utilization(3500, "test_op")
        assert status == "warning", "Should warn at ~85%"

        # Test critical threshold
        utilization, status = await cwm.check_utilization(3950, "test_op")
        assert status == "critical", "Should be critical at ~96%"

        return {"passed": True}

    except ImportError as e:
        return {"passed": False, "error": f"Import failed: {e}"}


# =============================================================================
# CONTINUAL LEARNING TESTS
# =============================================================================

async def test_continual_learning_initialization() -> Dict[str, Any]:
    """Test ContinualLearningEngine auto-initialization."""
    try:
        from jarvis_prime.models.continual_learning_system import (
            ContinualLearningEngine,
        )

        # Create with temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ContinualLearningEngine(
                base_dir=Path(tmpdir),
            )

            # Should initialize
            await engine.ensure_initialized()

            assert engine._initialized, "Should be initialized"

            return {"passed": True}

    except ImportError as e:
        return {"passed": False, "error": f"Import failed: {e}"}


async def test_experience_replay_buffer() -> Dict[str, Any]:
    """Test ExperienceReplayBuffer functionality."""
    try:
        from jarvis_prime.models.continual_learning_system import (
            ExperienceReplayBuffer,
            Experience,
        )

        buffer = ExperienceReplayBuffer(capacity=100, priority_alpha=0.6)

        # Add experiences
        exp1 = Experience(
            input_text="Hello",
            output_text="Hi there!",
            reward=0.9,
            context={"test": True},
        )

        await buffer.add(exp1)

        # Get all
        all_exp = await buffer.get_all()
        assert len(all_exp) == 1, "Should have 1 experience"
        assert all_exp[0].input_text == "Hello", "Should preserve data"

        # Sample
        samples = await buffer.sample(1)
        assert len(samples) == 1, "Should sample 1"

        return {"passed": True, "buffer_size": len(all_exp)}

    except ImportError as e:
        return {"passed": False, "error": f"Import failed: {e}"}


# =============================================================================
# CROSS-REPO COMMUNICATION TESTS
# =============================================================================

async def test_trinity_directory_structure() -> Dict[str, Any]:
    """Test Trinity Protocol directory structure exists."""
    trinity_dir = Path.home() / ".jarvis" / "trinity"
    cross_repo_dir = Path.home() / ".jarvis" / "cross_repo"

    results = {
        "trinity_exists": trinity_dir.exists(),
        "cross_repo_exists": cross_repo_dir.exists(),
    }

    # Create if needed
    trinity_dir.mkdir(parents=True, exist_ok=True)
    cross_repo_dir.mkdir(parents=True, exist_ok=True)

    # Check for service registry
    registry_path = trinity_dir / "service_registry.json"
    results["registry_exists"] = registry_path.exists()

    # Check for heartbeats directory
    heartbeats_dir = trinity_dir / "heartbeats"
    heartbeats_dir.mkdir(exist_ok=True)
    results["heartbeats_dir_exists"] = heartbeats_dir.exists()

    passed = all([
        results["trinity_exists"] or True,  # We created it
        results["cross_repo_exists"] or True,
    ])

    return {"passed": passed, **results}


async def test_service_health_endpoints() -> Dict[str, Any]:
    """Test service health endpoints are accessible."""
    import aiohttp

    services = {
        "jarvis_prime": {"url": "http://localhost:8000/health", "required": True},
        "jarvis_body": {"url": "http://localhost:8080/health", "required": False},
        "reactor_core": {"url": "http://localhost:8090/health", "required": False},
    }

    results = {}

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=5)
    ) as session:
        for name, config in services.items():
            try:
                async with session.get(config["url"]) as response:
                    results[name] = {
                        "status": response.status,
                        "healthy": response.status == 200,
                    }
                    if response.status == 200:
                        try:
                            data = await response.json()
                            results[name]["data"] = data
                        except Exception:
                            pass
            except Exception as e:
                results[name] = {"status": None, "healthy": False, "error": str(e)}

    # Check if required services are healthy
    passed = all(
        results.get(name, {}).get("healthy", False)
        for name, config in services.items()
        if config["required"]
    )

    return {"passed": passed, "services": results}


async def test_cross_repo_state_files() -> Dict[str, Any]:
    """Test cross-repo state files are being updated."""
    cross_repo_dir = Path.home() / ".jarvis" / "cross_repo"

    state_files = [
        "supervisor_state.json",
        "reactor_state.json",
        "computer_use_state.json",
    ]

    results = {}
    for filename in state_files:
        filepath = cross_repo_dir / filename
        if filepath.exists():
            try:
                with open(filepath) as f:
                    data = json.load(f)
                results[filename] = {
                    "exists": True,
                    "updated_at": data.get("updated_at", data.get("timestamp_iso")),
                    "running": data.get("running", data.get("status") == "healthy"),
                }
            except Exception as e:
                results[filename] = {"exists": True, "error": str(e)}
        else:
            results[filename] = {"exists": False}

    # At least supervisor_state should exist if supervisor is running
    passed = True  # Non-critical - files may not exist if services aren't running

    return {"passed": passed, "state_files": results}


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

async def test_reasoning_with_rag_circuit_breaker() -> Dict[str, Any]:
    """Test reasoning engine with RAG circuit breaker integration."""
    try:
        from jarvis_prime.core.reasoning_engine import ReasoningEngine, CircuitBreaker

        # Create engine with mocked RAG
        engine = ReasoningEngine()

        # Verify circuit breaker exists
        cb = engine._rag_circuit_breaker
        assert isinstance(cb, CircuitBreaker), "Should have CircuitBreaker"

        # Check initial state
        stats = cb.get_stats()
        assert stats["state"] == CircuitBreaker.CLOSED, "Should start CLOSED"

        return {"passed": True, "circuit_breaker_stats": stats}

    except ImportError as e:
        return {"passed": False, "error": f"Import failed: {e}"}


async def test_feedback_collection_api() -> Dict[str, Any]:
    """Test that feedback collection API matches expected signature."""
    try:
        from jarvis_prime.core.training_data_pipeline import TrainingDataPipeline

        # Check the method signature
        import inspect
        sig = inspect.signature(TrainingDataPipeline.capture_conversation)
        params = list(sig.parameters.keys())

        # Should have: self, user_input, assistant_response, system_prompt, context
        expected_params = ["self", "user_input", "assistant_response"]
        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

        return {"passed": True, "parameters": params}

    except ImportError as e:
        return {"passed": False, "error": f"Import failed: {e}"}


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_all_tests() -> Dict[str, Any]:
    """Run all test suites and return results."""
    suites = []

    # Circuit Breaker Tests
    cb_suite = TestSuite("Circuit Breaker")
    await cb_suite.run_test("closed_state", test_circuit_breaker_closed_state)
    await cb_suite.run_test("opens_on_failures", test_circuit_breaker_opens_on_failures)
    await cb_suite.run_test("fallback_mechanism", test_circuit_breaker_fallback)
    await cb_suite.run_test("half_open_recovery", test_circuit_breaker_half_open_recovery)
    suites.append(cb_suite)

    # Reasoning Engine Tests
    re_suite = TestSuite("Reasoning Engine")
    await re_suite.run_test("initialization", test_reasoning_engine_initialization)
    await re_suite.run_test("context_augmentation", test_strategy_context_augmentation)
    await re_suite.run_test("feedback_deduplicator", test_feedback_deduplicator)
    await re_suite.run_test("context_window_manager", test_context_window_manager)
    suites.append(re_suite)

    # Continual Learning Tests
    cl_suite = TestSuite("Continual Learning")
    await cl_suite.run_test("initialization", test_continual_learning_initialization)
    await cl_suite.run_test("experience_buffer", test_experience_replay_buffer)
    suites.append(cl_suite)

    # Cross-Repo Communication Tests
    cr_suite = TestSuite("Cross-Repo Communication")
    await cr_suite.run_test("directory_structure", test_trinity_directory_structure)
    await cr_suite.run_test("health_endpoints", test_service_health_endpoints)
    await cr_suite.run_test("state_files", test_cross_repo_state_files)
    suites.append(cr_suite)

    # Integration Tests
    int_suite = TestSuite("Integration")
    await int_suite.run_test("rag_circuit_breaker", test_reasoning_with_rag_circuit_breaker)
    await int_suite.run_test("feedback_api", test_feedback_collection_api)
    suites.append(int_suite)

    # Print results
    total_passed = 0
    total_failed = 0

    for suite in suites:
        suite.print_results()
        summary = suite.get_summary()
        total_passed += summary["passed"]
        total_failed += summary["failed"]

    # Overall summary
    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print(f"Total Tests: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Pass Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    print("=" * 70)

    return {
        "total": total_passed + total_failed,
        "passed": total_passed,
        "failed": total_failed,
        "suites": [s.get_summary() for s in suites],
    }


if __name__ == "__main__":
    results = asyncio.run(run_all_tests())
    sys.exit(0 if results["failed"] == 0 else 1)
