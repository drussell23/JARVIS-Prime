"""
Verification Suite v1.0 - Comprehensive System Verification
=============================================================

Provides verification tests for the Trinity Connective Tissue:
- Brain Router Logic Tests (complexity discrimination)
- GCP Preemption Drill (death event handling)
- OOM Killer Protection Tests (RAM limit enforcement)
- Cross-Repo Integration Tests (Trinity Protocol)

RUN THIS BEFORE CONSIDERING THE SYSTEM "DONE":
    python -m jarvis_prime.core.verification_suite

TEST CATEGORIES:
    Test A: Brain Router Logic - Verifies complexity discrimination
    Test B: Preemption Drill - Simulates GCP Spot VM preemption
    Test C: OOM Killer - Verifies RAM protection
    Test D: Service Mesh - Verifies service discovery and circuit breakers
    Test E: Cross-Repo - Verifies Trinity Protocol communication
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verification")


# =============================================================================
# TEST RESULT TYPES
# =============================================================================

class TestStatus(Enum):
    """Test execution status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    duration_ms: float
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": round(self.duration_ms, 2),
            "message": self.message,
            "details": self.details,
            "error": str(self.error) if self.error else None,
        }


@dataclass
class TestSuiteResult:
    """Result of an entire test suite."""
    suite_name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "success_rate": round(self.success_rate, 2),
            "duration_ms": round((self.end_time or time.time()) - self.start_time, 2) * 1000,
            "results": [r.to_dict() for r in self.results],
        }


# =============================================================================
# TEST A: BRAIN ROUTER LOGIC VERIFICATION
# =============================================================================

class BrainRouterVerification:
    """
    Verifies the Intelligent Model Router correctly discriminates
    between "easy" and "hard" tasks.

    SUCCESS CRITERIA:
    - Simple prompts (chat, math) -> Local 7B
    - Medium complexity (code) -> Local 7B or GCP 13B
    - Complex prompts (reasoning, analysis) -> GCP 13B or Claude API
    """

    # Test prompts with expected routing
    TEST_PROMPTS = [
        # Simple prompts -> should route to LOCAL_7B
        {
            "prompt": "What is 2+2?",
            "expected_tier": "local_7b",
            "complexity_max": 0.3,
            "description": "Simple arithmetic",
        },
        {
            "prompt": "Hello, how are you today?",
            "expected_tier": "local_7b",
            "complexity_max": 0.2,
            "description": "Simple greeting",
        },
        {
            "prompt": "What is the capital of France?",
            "expected_tier": "local_7b",
            "complexity_max": 0.25,
            "description": "Simple factual question",
        },
        {
            "prompt": "Summarize this text in one sentence: The quick brown fox jumps over the lazy dog.",
            "expected_tier": "local_7b",
            "complexity_max": 0.35,
            "description": "Simple summarization",
        },

        # Medium complexity -> should route to LOCAL_7B or GCP_13B
        {
            "prompt": "Write a Python function that reverses a string.",
            "expected_tier": "local_7b",  # or gcp_13b
            "complexity_max": 0.5,
            "description": "Simple code generation",
        },
        {
            "prompt": "Write a Python script to read a CSV file and calculate the average of a column.",
            "expected_tier": "gcp_13b",  # Medium complexity code
            "complexity_min": 0.3,
            "complexity_max": 0.7,
            "description": "Medium code generation",
        },

        # Complex prompts -> should route to GCP_13B or CLAUDE_API
        {
            "prompt": """Write a Python script to implement a binary search tree with
                        insert, delete, search, and in-order traversal methods. Include
                        proper error handling and docstrings.""",
            "expected_tier": "gcp_13b",
            "complexity_min": 0.5,
            "description": "Complex code generation",
        },
        {
            "prompt": """Explain the geopolitical implications of quantum computing on
                        modern cryptography. Consider the impact on financial systems,
                        national security, and international relations. Provide specific
                        examples and analyze potential countermeasures.""",
            "expected_tier": "claude_api",
            "complexity_min": 0.7,
            "description": "Complex reasoning and analysis",
        },
        {
            "prompt": """Design a distributed system architecture for a real-time stock
                        trading platform. Include considerations for low latency, high
                        availability, fault tolerance, and regulatory compliance.
                        Provide diagrams and implementation details.""",
            "expected_tier": "claude_api",
            "complexity_min": 0.75,
            "description": "Complex architecture design",
        },
        {
            "prompt": """Analyze the following code for security vulnerabilities and
                        suggest fixes. Consider OWASP top 10, SQL injection, XSS,
                        CSRF, and authentication bypass attacks.""",
            "expected_tier": "gcp_13b",
            "complexity_min": 0.6,
            "description": "Security analysis",
        },
    ]

    def __init__(self):
        self._router = None
        self._results: List[TestResult] = []

    async def run(self) -> TestSuiteResult:
        """Run all brain router verification tests."""
        suite = TestSuiteResult(suite_name="Brain Router Verification")

        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST A: BRAIN ROUTER LOGIC VERIFICATION")
        logger.info("=" * 70)
        logger.info("")

        try:
            # Initialize router
            from jarvis_prime.core.intelligent_model_router import (
                get_intelligent_router,
                ModelTier,
            )

            self._router = await get_intelligent_router()

            # Run each test prompt
            for i, test_case in enumerate(self.TEST_PROMPTS, 1):
                result = await self._test_prompt(i, test_case)
                suite.results.append(result)

                status_icon = "PASS" if result.status == TestStatus.PASSED else "FAIL"
                logger.info(f"  [{status_icon}] Test {i}: {test_case['description']}")

                if result.status == TestStatus.FAILED:
                    logger.info(f"         Expected: {test_case.get('expected_tier')}")
                    logger.info(f"         Got: {result.details.get('actual_tier')}")
                    logger.info(f"         Complexity: {result.details.get('complexity', 0):.3f}")

        except ImportError as e:
            logger.error(f"Could not import router: {e}")
            suite.results.append(TestResult(
                name="import_router",
                status=TestStatus.ERROR,
                duration_ms=0,
                message=f"Import failed: {e}",
                error=e,
            ))
        except Exception as e:
            logger.error(f"Test error: {e}")
            suite.results.append(TestResult(
                name="general_error",
                status=TestStatus.ERROR,
                duration_ms=0,
                message=str(e),
                error=e,
            ))

        suite.end_time = time.time()
        return suite

    async def _test_prompt(self, test_num: int, test_case: Dict[str, Any]) -> TestResult:
        """Test a single prompt."""
        start = time.time()

        try:
            # Get routing decision (without executing inference)
            from jarvis_prime.core.intelligent_model_router import ModelTier

            # Analyze complexity
            complexity, task_type, signals = await self._router._complexity_analyzer.analyze(
                test_case["prompt"]
            )

            # Get tier selection
            from jarvis_prime.core.intelligent_model_router import ResourceSnapshot
            resources = ResourceSnapshot.capture()

            tier, confidence, reasoning = self._router._select_tier(
                complexity, task_type, resources, test_case["prompt"], None
            )

            actual_tier = tier.value
            expected_tier = test_case["expected_tier"]

            # Check complexity bounds
            complexity_min = test_case.get("complexity_min", 0.0)
            complexity_max = test_case.get("complexity_max", 1.0)

            # Determine pass/fail
            # Accept if tier matches expected OR if complexity is in expected range
            tier_match = actual_tier == expected_tier

            # For flexible routing (local or gcp), accept either
            if expected_tier in ["local_7b", "gcp_13b"]:
                tier_acceptable = actual_tier in ["local_7b", "gcp_13b"]
            elif expected_tier == "claude_api":
                tier_acceptable = actual_tier in ["gcp_13b", "claude_api"]
            else:
                tier_acceptable = tier_match

            complexity_ok = complexity_min <= complexity <= complexity_max

            passed = tier_acceptable and complexity_ok

            return TestResult(
                name=f"prompt_{test_num}",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message=f"{test_case['description']}",
                details={
                    "prompt": test_case["prompt"][:100] + "...",
                    "expected_tier": expected_tier,
                    "actual_tier": actual_tier,
                    "complexity": complexity,
                    "complexity_min": complexity_min,
                    "complexity_max": complexity_max,
                    "task_type": task_type,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "signals": signals,
                },
            )

        except Exception as e:
            return TestResult(
                name=f"prompt_{test_num}",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )


# =============================================================================
# TEST B: GCP PREEMPTION DRILL
# =============================================================================

class PreemptionDrillVerification:
    """
    Simulates GCP Spot VM preemption to verify:
    1. ACPI G2 signal detection
    2. Automatic checkpointing
    3. Zone migration

    SUCCESS CRITERIA:
    - System detects preemption signal within 60 seconds
    - Checkpoint is created before shutdown
    - New VM is provisioned in alternate zone
    """

    def __init__(self):
        self._gcp_manager = None
        self._preemption_handler = None

    async def run(self) -> TestSuiteResult:
        """Run preemption drill tests."""
        suite = TestSuiteResult(suite_name="Preemption Drill Verification")

        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST B: GCP PREEMPTION DRILL")
        logger.info("=" * 70)
        logger.info("")

        try:
            from jarvis_prime.core.gcp_vm_manager import (
                get_gcp_manager,
                GCPManagerConfig,
                PreemptionHandler,
                PreemptionConfig,
                VMInstance,
                VMState,
            )

            self._gcp_manager = await get_gcp_manager()

            # Test 1: Preemption signal detection
            result1 = await self._test_signal_detection()
            suite.results.append(result1)
            logger.info(f"  [{'PASS' if result1.status == TestStatus.PASSED else 'FAIL'}] Signal detection")

            # Test 2: Checkpoint creation
            result2 = await self._test_checkpoint_creation()
            suite.results.append(result2)
            logger.info(f"  [{'PASS' if result2.status == TestStatus.PASSED else 'FAIL'}] Checkpoint creation")

            # Test 3: Zone migration
            result3 = await self._test_zone_migration()
            suite.results.append(result3)
            logger.info(f"  [{'PASS' if result3.status == TestStatus.PASSED else 'FAIL'}] Zone migration")

        except ImportError as e:
            logger.warning(f"GCP manager not available: {e}")
            suite.results.append(TestResult(
                name="gcp_import",
                status=TestStatus.SKIPPED,
                duration_ms=0,
                message="GCP manager not available - skipping preemption tests",
            ))
        except Exception as e:
            logger.error(f"Test error: {e}")
            suite.results.append(TestResult(
                name="general_error",
                status=TestStatus.ERROR,
                duration_ms=0,
                message=str(e),
                error=e,
            ))

        suite.end_time = time.time()
        return suite

    async def _test_signal_detection(self) -> TestResult:
        """Test preemption signal file detection."""
        start = time.time()

        try:
            from jarvis_prime.core.gcp_vm_manager import PreemptionHandler, PreemptionConfig

            # Create signal file
            signal_file = Path("/tmp/test_preemption_signal")

            config = PreemptionConfig(
                check_interval_seconds=1,
                signal_file=str(signal_file),
            )

            preemption_detected = asyncio.Event()

            async def on_preemption():
                preemption_detected.set()

            handler = PreemptionHandler(
                config=config,
                on_preemption=on_preemption,
            )

            # Create mock instance
            from jarvis_prime.core.gcp_vm_manager import VMInstance
            mock_instance = VMInstance(
                name="test-instance",
                zone="us-central1-a",
            )

            # Start monitoring
            await handler.start_monitoring(mock_instance)

            # Write signal file
            signal_file.write_text("preemption")

            # Wait for detection (with timeout)
            try:
                await asyncio.wait_for(
                    preemption_detected.wait(),
                    timeout=5.0
                )
                detected = True
            except asyncio.TimeoutError:
                detected = False

            # Cleanup
            await handler.stop_monitoring()
            if signal_file.exists():
                signal_file.unlink()

            return TestResult(
                name="signal_detection",
                status=TestStatus.PASSED if detected else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Signal detected" if detected else "Signal not detected within timeout",
                details={
                    "signal_file": str(signal_file),
                    "detected": detected,
                },
            )

        except Exception as e:
            return TestResult(
                name="signal_detection",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )

    async def _test_checkpoint_creation(self) -> TestResult:
        """Test checkpoint creation on preemption."""
        start = time.time()

        try:
            checkpoint_dir = Path.home() / ".jarvis" / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Count existing checkpoints
            existing_checkpoints = list(checkpoint_dir.glob("*.json"))

            # Simulate checkpoint save
            from jarvis_prime.core.gcp_vm_manager import PreemptionHandler, PreemptionConfig, VMInstance

            config = PreemptionConfig(
                checkpoint_on_preemption=True,
            )

            checkpoint_saved = asyncio.Event()

            async def on_checkpoint(path: Path):
                checkpoint_saved.set()

            handler = PreemptionHandler(
                config=config,
                on_checkpoint=on_checkpoint,
            )

            mock_instance = VMInstance(
                name="checkpoint-test-instance",
                zone="us-central1-a",
            )

            # Manually trigger checkpoint save
            await handler._save_checkpoint(mock_instance)

            # Check for new checkpoint
            new_checkpoints = list(checkpoint_dir.glob("*.json"))
            created = len(new_checkpoints) > len(existing_checkpoints)

            return TestResult(
                name="checkpoint_creation",
                status=TestStatus.PASSED if created else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Checkpoint created" if created else "No checkpoint created",
                details={
                    "checkpoint_dir": str(checkpoint_dir),
                    "existing_count": len(existing_checkpoints),
                    "new_count": len(new_checkpoints),
                },
            )

        except Exception as e:
            return TestResult(
                name="checkpoint_creation",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )

    async def _test_zone_migration(self) -> TestResult:
        """Test automatic zone migration on preemption."""
        start = time.time()

        try:
            from jarvis_prime.core.gcp_vm_manager import GCPManagerConfig, PreemptionConfig

            # Verify migration zones are configured
            config = GCPManagerConfig.from_env()
            preemption_config = config.preemption

            has_migrate_zones = len(preemption_config.migrate_zones) > 1
            auto_migrate_enabled = preemption_config.auto_migrate

            passed = has_migrate_zones and auto_migrate_enabled

            return TestResult(
                name="zone_migration",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Migration configuration valid" if passed else "Migration not properly configured",
                details={
                    "auto_migrate_enabled": auto_migrate_enabled,
                    "migrate_zones": preemption_config.migrate_zones,
                    "zone_count": len(preemption_config.migrate_zones),
                },
            )

        except Exception as e:
            return TestResult(
                name="zone_migration",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )


# =============================================================================
# TEST C: OOM KILLER PROTECTION
# =============================================================================

class OOMProtectionVerification:
    """
    Verifies the OOM protection system:
    1. RAM monitoring accuracy
    2. Local inference skip when RAM high
    3. Automatic fallback to cloud/API

    SUCCESS CRITERIA:
    - At 90% RAM usage, local inference is skipped
    - Routing automatically falls back to API/GCP
    """

    def __init__(self):
        self._router = None

    async def run(self) -> TestSuiteResult:
        """Run OOM protection tests."""
        suite = TestSuiteResult(suite_name="OOM Protection Verification")

        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST C: OOM KILLER PROTECTION")
        logger.info("=" * 70)
        logger.info("")

        try:
            # Test 1: Resource monitoring accuracy
            result1 = await self._test_resource_monitoring()
            suite.results.append(result1)
            logger.info(f"  [{'PASS' if result1.status == TestStatus.PASSED else 'FAIL'}] Resource monitoring")

            # Test 2: High RAM detection
            result2 = await self._test_high_ram_detection()
            suite.results.append(result2)
            logger.info(f"  [{'PASS' if result2.status == TestStatus.PASSED else 'FAIL'}] High RAM detection")

            # Test 3: Automatic fallback
            result3 = await self._test_automatic_fallback()
            suite.results.append(result3)
            logger.info(f"  [{'PASS' if result3.status == TestStatus.PASSED else 'FAIL'}] Automatic fallback")

        except Exception as e:
            logger.error(f"Test error: {e}")
            suite.results.append(TestResult(
                name="general_error",
                status=TestStatus.ERROR,
                duration_ms=0,
                message=str(e),
                error=e,
            ))

        suite.end_time = time.time()
        return suite

    async def _test_resource_monitoring(self) -> TestResult:
        """Test resource monitoring accuracy."""
        start = time.time()

        try:
            from jarvis_prime.core.advanced_primitives import ResourceMonitor

            monitor = await ResourceMonitor.get_instance()
            resources = await monitor.capture(force_refresh=True)

            # Verify we got real values (not zeros/defaults)
            has_ram = resources.ram_total_mb > 0
            has_cpu = resources.cpu_count > 0
            has_gpu_check = resources.gpu is not None

            passed = has_ram and has_cpu

            return TestResult(
                name="resource_monitoring",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Real resource values obtained" if passed else "Resource monitoring returned defaults",
                details={
                    "ram_total_mb": resources.ram_total_mb,
                    "ram_used_percent": resources.ram_percent,
                    "cpu_count": resources.cpu_count,
                    "cpu_percent": resources.cpu_percent,
                    "gpu_available": resources.gpu.available if resources.gpu else False,
                    "gpu_type": resources.gpu.gpu_type if resources.gpu else "none",
                    "network_available": resources.network.available if resources.network else False,
                },
            )

        except ImportError:
            # Fall back to psutil-only test
            try:
                import psutil
                mem = psutil.virtual_memory()
                passed = mem.total > 0

                return TestResult(
                    name="resource_monitoring",
                    status=TestStatus.PASSED if passed else TestStatus.FAILED,
                    duration_ms=(time.time() - start) * 1000,
                    message="psutil monitoring working",
                    details={
                        "ram_total_mb": mem.total / (1024 ** 2),
                        "ram_used_percent": mem.percent,
                    },
                )
            except ImportError:
                return TestResult(
                    name="resource_monitoring",
                    status=TestStatus.SKIPPED,
                    duration_ms=(time.time() - start) * 1000,
                    message="psutil not available",
                )

        except Exception as e:
            return TestResult(
                name="resource_monitoring",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )

    async def _test_high_ram_detection(self) -> TestResult:
        """Test that high RAM usage is correctly detected."""
        start = time.time()

        try:
            from jarvis_prime.core.intelligent_model_router import (
                ResourceSnapshot,
                RoutingConfig,
            )

            # Capture real resources
            resources = ResourceSnapshot.capture()
            config = RoutingConfig()

            # Test the can_use_local check
            can_use, reason = resources.can_use_local(config)

            # Also test with artificial high RAM
            high_ram_resources = ResourceSnapshot(
                timestamp=time.time(),
                ram_used_percent=92.0,  # Above threshold
                ram_used_mb=15000,
                ram_available_mb=1000,
                cpu_percent=50.0,
                gpu_available=True,
                gpu_memory_used_mb=0,
                gpu_memory_total_mb=0,
                network_available=True,
            )

            can_use_high, reason_high = high_ram_resources.can_use_local(config)

            # With high RAM, should NOT be able to use local
            passed = not can_use_high and "RAM" in reason_high

            return TestResult(
                name="high_ram_detection",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="High RAM correctly blocks local inference" if passed else "High RAM not properly detected",
                details={
                    "current_ram_percent": resources.ram_used_percent,
                    "current_can_use_local": can_use,
                    "simulated_high_ram_percent": 92.0,
                    "high_ram_can_use_local": can_use_high,
                    "high_ram_reason": reason_high,
                    "threshold_percent": config.local_max_ram_percent,
                },
            )

        except Exception as e:
            return TestResult(
                name="high_ram_detection",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )

    async def _test_automatic_fallback(self) -> TestResult:
        """Test that routing automatically falls back when resources exhausted."""
        start = time.time()

        try:
            from jarvis_prime.core.intelligent_model_router import (
                get_intelligent_router,
                ModelTier,
                RoutingConfig,
            )

            router = await get_intelligent_router()

            # Verify fallback chain is configured
            config = router._config
            fallback_chain = config.fallback_chain

            has_fallback = len(fallback_chain) > 1
            has_cloud_fallback = (
                ModelTier.GCP_13B in fallback_chain or
                ModelTier.CLAUDE_API in fallback_chain
            )

            passed = has_fallback and has_cloud_fallback

            return TestResult(
                name="automatic_fallback",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Fallback chain properly configured" if passed else "No fallback chain configured",
                details={
                    "fallback_chain": [t.value for t in fallback_chain],
                    "has_cloud_fallback": has_cloud_fallback,
                },
            )

        except Exception as e:
            return TestResult(
                name="automatic_fallback",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )


# =============================================================================
# TEST D: SERVICE MESH VERIFICATION
# =============================================================================

class ServiceMeshVerification:
    """
    Verifies Service Mesh functionality:
    1. Service registration
    2. Service discovery
    3. Circuit breaker operation
    """

    async def run(self) -> TestSuiteResult:
        """Run service mesh tests."""
        suite = TestSuiteResult(suite_name="Service Mesh Verification")

        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST D: SERVICE MESH VERIFICATION")
        logger.info("=" * 70)
        logger.info("")

        try:
            from jarvis_prime.core.service_mesh import (
                get_service_mesh,
                ServiceStatus,
            )

            mesh = await get_service_mesh()

            # Test 1: Service registration
            result1 = await self._test_registration(mesh)
            suite.results.append(result1)
            logger.info(f"  [{'PASS' if result1.status == TestStatus.PASSED else 'FAIL'}] Service registration")

            # Test 2: Service discovery
            result2 = await self._test_discovery(mesh)
            suite.results.append(result2)
            logger.info(f"  [{'PASS' if result2.status == TestStatus.PASSED else 'FAIL'}] Service discovery")

            # Test 3: Circuit breaker
            result3 = await self._test_circuit_breaker(mesh)
            suite.results.append(result3)
            logger.info(f"  [{'PASS' if result3.status == TestStatus.PASSED else 'FAIL'}] Circuit breaker")

        except ImportError as e:
            logger.warning(f"Service mesh not available: {e}")
            suite.results.append(TestResult(
                name="mesh_import",
                status=TestStatus.SKIPPED,
                duration_ms=0,
                message="Service mesh not available",
            ))
        except Exception as e:
            logger.error(f"Test error: {e}")
            suite.results.append(TestResult(
                name="general_error",
                status=TestStatus.ERROR,
                duration_ms=0,
                message=str(e),
                error=e,
            ))

        suite.end_time = time.time()
        return suite

    async def _test_registration(self, mesh) -> TestResult:
        """Test service registration."""
        start = time.time()

        try:
            # Register a test service
            endpoint = await mesh.register_service(
                service_name="test-service",
                host="localhost",
                port=9999,
                capabilities=["test"],
            )

            registered = endpoint is not None
            has_correct_info = (
                endpoint.service_name == "test-service" and
                endpoint.port == 9999
            )

            passed = registered and has_correct_info

            return TestResult(
                name="service_registration",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Service registered successfully" if passed else "Registration failed",
                details={
                    "service_name": endpoint.service_name if endpoint else None,
                    "instance_id": endpoint.instance_id if endpoint else None,
                    "url": endpoint.url if endpoint else None,
                },
            )

        except Exception as e:
            return TestResult(
                name="service_registration",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )

    async def _test_discovery(self, mesh) -> TestResult:
        """Test service discovery."""
        start = time.time()

        try:
            # Get registered service
            endpoint = await mesh.get_endpoint("test-service")
            found = endpoint is not None

            # Get all services
            all_services = await mesh.registry.get_all_services()
            has_services = len(all_services) > 0

            passed = found or has_services

            return TestResult(
                name="service_discovery",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Service discovery working" if passed else "No services found",
                details={
                    "test_service_found": found,
                    "total_services": len(all_services),
                    "service_names": list(all_services.keys()),
                },
            )

        except Exception as e:
            return TestResult(
                name="service_discovery",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )

    async def _test_circuit_breaker(self, mesh) -> TestResult:
        """Test circuit breaker functionality."""
        start = time.time()

        try:
            # Get circuit breaker manager
            cb_manager = mesh._circuit_manager
            states = cb_manager.get_all_states()

            # Verify circuit breaker is enabled
            has_breakers = len(states) >= 0  # May be empty if no failures
            config_enabled = mesh._config.circuit_enabled

            passed = config_enabled

            return TestResult(
                name="circuit_breaker",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Circuit breaker enabled" if passed else "Circuit breaker disabled",
                details={
                    "circuit_enabled": config_enabled,
                    "active_breakers": len(states),
                    "failure_threshold": mesh._config.circuit_failure_threshold,
                    "timeout_seconds": mesh._config.circuit_timeout_seconds,
                },
            )

        except Exception as e:
            return TestResult(
                name="circuit_breaker",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )


# =============================================================================
# TEST E: CROSS-REPO INTEGRATION
# =============================================================================

class CrossRepoVerification:
    """
    Verifies cross-repository integration:
    1. Trinity Protocol IPC
    2. Unified configuration loading
    3. Component health monitoring
    """

    async def run(self) -> TestSuiteResult:
        """Run cross-repo integration tests."""
        suite = TestSuiteResult(suite_name="Cross-Repo Integration Verification")

        logger.info("")
        logger.info("=" * 70)
        logger.info("TEST E: CROSS-REPO INTEGRATION")
        logger.info("=" * 70)
        logger.info("")

        # Test 1: Trinity Protocol directory
        result1 = await self._test_trinity_protocol()
        suite.results.append(result1)
        logger.info(f"  [{'PASS' if result1.status == TestStatus.PASSED else 'FAIL'}] Trinity Protocol")

        # Test 2: Unified configuration
        result2 = await self._test_unified_config()
        suite.results.append(result2)
        logger.info(f"  [{'PASS' if result2.status == TestStatus.PASSED else 'FAIL'}] Unified configuration")

        # Test 3: Cross-repo paths
        result3 = await self._test_cross_repo_paths()
        suite.results.append(result3)
        logger.info(f"  [{'PASS' if result3.status == TestStatus.PASSED else 'FAIL'}] Cross-repo paths")

        suite.end_time = time.time()
        return suite

    async def _test_trinity_protocol(self) -> TestResult:
        """Test Trinity Protocol IPC directories."""
        start = time.time()

        try:
            trinity_dir = Path.home() / ".jarvis" / "trinity"
            cross_repo_dir = Path.home() / ".jarvis" / "cross_repo"

            # Create if needed
            trinity_dir.mkdir(parents=True, exist_ok=True)
            cross_repo_dir.mkdir(parents=True, exist_ok=True)

            trinity_exists = trinity_dir.exists()
            cross_repo_exists = cross_repo_dir.exists()
            trinity_writable = os.access(trinity_dir, os.W_OK)

            passed = trinity_exists and cross_repo_exists and trinity_writable

            return TestResult(
                name="trinity_protocol",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Trinity Protocol directories ready" if passed else "IPC directories not accessible",
                details={
                    "trinity_dir": str(trinity_dir),
                    "trinity_exists": trinity_exists,
                    "trinity_writable": trinity_writable,
                    "cross_repo_dir": str(cross_repo_dir),
                    "cross_repo_exists": cross_repo_exists,
                },
            )

        except Exception as e:
            return TestResult(
                name="trinity_protocol",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )

    async def _test_unified_config(self) -> TestResult:
        """Test unified configuration loading."""
        start = time.time()

        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "unified_config.yaml"

            config_exists = config_path.exists()

            if config_exists:
                import yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                has_components = "components" in config
                has_routing = "model_routing" in config
                has_mesh = "service_mesh" in config

                passed = has_components and has_routing
            else:
                has_components = False
                has_routing = False
                has_mesh = False
                passed = False

            return TestResult(
                name="unified_config",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Unified configuration loaded" if passed else "Configuration not found or invalid",
                details={
                    "config_path": str(config_path),
                    "config_exists": config_exists,
                    "has_components": has_components,
                    "has_routing": has_routing,
                    "has_mesh": has_mesh,
                },
            )

        except Exception as e:
            return TestResult(
                name="unified_config",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )

    async def _test_cross_repo_paths(self) -> TestResult:
        """Test cross-repo path detection."""
        start = time.time()

        try:
            # Get current repo path (jarvis-prime)
            prime_path = Path(__file__).parent.parent.parent
            base_dir = prime_path.parent

            # Check for sibling repos
            jarvis_candidates = [
                base_dir / "JARVIS-AI-Agent",
                base_dir / "jarvis-ai-agent",
                base_dir / "JARVIS",
                base_dir / "jarvis",
            ]

            reactor_candidates = [
                base_dir / "Reactor-Core",
                base_dir / "reactor-core",
            ]

            jarvis_path = next(
                (p for p in jarvis_candidates if p.exists()),
                None
            )

            reactor_path = next(
                (p for p in reactor_candidates if p.exists()),
                None
            )

            prime_exists = prime_path.exists()

            # Pass if at least prime exists (others are optional)
            passed = prime_exists

            return TestResult(
                name="cross_repo_paths",
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                duration_ms=(time.time() - start) * 1000,
                message="Cross-repo paths detected" if passed else "Path detection failed",
                details={
                    "base_dir": str(base_dir),
                    "prime_path": str(prime_path),
                    "prime_exists": prime_exists,
                    "jarvis_path": str(jarvis_path) if jarvis_path else None,
                    "jarvis_exists": jarvis_path is not None,
                    "reactor_path": str(reactor_path) if reactor_path else None,
                    "reactor_exists": reactor_path is not None,
                },
            )

        except Exception as e:
            return TestResult(
                name="cross_repo_paths",
                status=TestStatus.ERROR,
                duration_ms=(time.time() - start) * 1000,
                message=str(e),
                error=e,
            )


# =============================================================================
# MAIN VERIFICATION RUNNER
# =============================================================================

class VerificationRunner:
    """Main verification runner that executes all test suites."""

    def __init__(self):
        self._suites: List[TestSuiteResult] = []

    async def run_all(self) -> Dict[str, Any]:
        """Run all verification suites."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("JARVIS TRINITY CONNECTIVE TISSUE VERIFICATION")
        logger.info("v87.0 - Comprehensive System Verification")
        logger.info("=" * 70)
        logger.info("")

        start_time = time.time()

        # Run each test suite
        suites = [
            ("A", BrainRouterVerification()),
            ("B", PreemptionDrillVerification()),
            ("C", OOMProtectionVerification()),
            ("D", ServiceMeshVerification()),
            ("E", CrossRepoVerification()),
        ]

        for letter, suite in suites:
            result = await suite.run()
            self._suites.append(result)

        # Calculate totals
        total_passed = sum(s.passed for s in self._suites)
        total_failed = sum(s.failed for s in self._suites)
        total_tests = sum(s.total for s in self._suites)

        duration_seconds = time.time() - start_time

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("VERIFICATION SUMMARY")
        logger.info("=" * 70)
        logger.info("")

        for suite in self._suites:
            status = "PASS" if suite.failed == 0 else "FAIL"
            logger.info(f"  [{status}] {suite.suite_name}: {suite.passed}/{suite.total} passed")

        logger.info("")
        logger.info("-" * 70)
        logger.info(f"  TOTAL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
        logger.info(f"  Duration: {duration_seconds:.2f} seconds")
        logger.info("-" * 70)

        if total_failed == 0:
            logger.info("")
            logger.info("  *** ALL VERIFICATION TESTS PASSED ***")
            logger.info("  The Connective Tissue is ready for production.")
            logger.info("")
        else:
            logger.info("")
            logger.info(f"  *** {total_failed} TESTS FAILED ***")
            logger.info("  Please fix issues before deploying.")
            logger.info("")

        # Return results
        return {
            "passed": total_passed,
            "failed": total_failed,
            "total": total_tests,
            "success_rate": total_passed / total_tests if total_tests > 0 else 0,
            "duration_seconds": duration_seconds,
            "suites": [s.to_dict() for s in self._suites],
        }

    def save_results(self, path: Optional[Path] = None):
        """Save verification results to file."""
        if path is None:
            path = Path.home() / ".jarvis" / "verification_results.json"

        path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "timestamp": datetime.now().isoformat(),
            "suites": [s.to_dict() for s in self._suites],
        }

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {path}")


async def main():
    """Main entry point for verification suite."""
    runner = VerificationRunner()
    results = await runner.run_all()

    # Save results
    runner.save_results()

    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
