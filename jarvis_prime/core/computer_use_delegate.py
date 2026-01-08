"""
JARVIS Prime Computer Use Delegation
=====================================

Enables JARVIS Prime to delegate Computer Use tasks to main JARVIS.

Features:
- Computer Use task delegation to main JARVIS
- Vision analysis result caching
- Action Chaining optimization awareness
- OmniParser integration support
- Real-time event subscription (v4.0)
- Health monitoring integration (v4.0)
- Atomic transaction support (v4.0)
- Adaptive timeout based on task complexity (v4.0)

Architecture:
    JARVIS Prime (request) → ~/.jarvis/cross_repo/ → JARVIS (execution)
                                       ↓
                                   Results

Author: JARVIS Prime Team
Version: 4.0.0 - Production-Grade Delegation
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import random
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("jarvis-prime.computer-use")


# =============================================================================
# Environment Configuration (Zero Hardcoding)
# =============================================================================


def _env_int(key: str, default: int) -> int:
    """Get integer from environment."""
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    """Get float from environment."""
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment."""
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# Delegation settings
CU_DELEGATE_BASE_TIMEOUT = _env_float("CU_DELEGATE_BASE_TIMEOUT", 60.0)
CU_DELEGATE_COMPLEXITY_MULTIPLIER = _env_float("CU_DELEGATE_COMPLEXITY_MULTIPLIER", 1.5)
CU_DELEGATE_POLLING_INTERVAL = _env_float("CU_DELEGATE_POLLING_INTERVAL", 0.3)
CU_DELEGATE_RETRY_ATTEMPTS = _env_int("CU_DELEGATE_RETRY_ATTEMPTS", 3)
CU_DELEGATE_LOCK_TIMEOUT = _env_float("CU_DELEGATE_LOCK_TIMEOUT", 5.0)

# Health check settings
CU_DELEGATE_HEALTH_CHECK_INTERVAL = _env_float("CU_DELEGATE_HEALTH_CHECK_INTERVAL", 30.0)
CU_DELEGATE_STALE_THRESHOLD = _env_float("CU_DELEGATE_STALE_THRESHOLD", 120.0)

# Event streaming settings
CU_DELEGATE_EVENT_BUFFER_SIZE = _env_int("CU_DELEGATE_EVENT_BUFFER_SIZE", 50)


# ============================================================================
# Constants
# ============================================================================

JARVIS_BASE_DIR = Path(os.getenv("JARVIS_BASE_DIR", str(Path.home() / ".jarvis")))
COMPUTER_USE_STATE_DIR = JARVIS_BASE_DIR / "cross_repo"
COMPUTER_USE_REQUESTS_FILE = COMPUTER_USE_STATE_DIR / "computer_use_requests.json"
COMPUTER_USE_RESULTS_FILE = COMPUTER_USE_STATE_DIR / "computer_use_results.json"
COMPUTER_USE_STATE_FILE = COMPUTER_USE_STATE_DIR / "computer_use_state.json"
COMPUTER_USE_HEALTH_FILE = COMPUTER_USE_STATE_DIR / "repo_health.json"

REQUEST_TIMEOUT = CU_DELEGATE_BASE_TIMEOUT
POLLING_INTERVAL = CU_DELEGATE_POLLING_INTERVAL


# ============================================================================
# Enums
# ============================================================================

class RequestStatus(Enum):
    """Computer Use request status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class DelegationMode(Enum):
    """Computer Use delegation mode."""
    FULL_DELEGATION = "full_delegation"  # JARVIS Prime sends request, JARVIS executes
    VISION_ONLY = "vision_only"  # Only delegate vision analysis
    HYBRID = "hybrid"  # Use local inference + remote execution


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ComputerUseRequest:
    """A Computer Use task request from JARVIS Prime."""
    request_id: str
    timestamp: str
    goal: str
    context: Dict[str, Any]

    # Delegation settings
    mode: DelegationMode = DelegationMode.FULL_DELEGATION
    use_action_chaining: bool = True
    use_omniparser: bool = True  # v6.2: Enable by default with fallback

    # v6.2: Preferred parser mode (omniparser, claude_vision, ocr, auto)
    preferred_parser_mode: str = "auto"

    # Requester info
    requester: str = "jarvis-prime"
    priority: int = 1  # 1=normal, 2=high, 3=urgent

    # Timeout
    timeout_seconds: float = 60.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["mode"] = self.mode.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputerUseRequest":
        """Create from dictionary."""
        mode_str = data.pop("mode", "full_delegation")
        mode = DelegationMode(mode_str)

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}, mode=mode)


@dataclass
class ComputerUseResult:
    """Result of a Computer Use task."""
    request_id: str
    timestamp: str
    status: RequestStatus

    # Results
    success: bool = False
    actions_executed: int = 0
    execution_time_ms: float = 0.0

    # Outputs
    final_state: str = ""
    error_message: str = ""

    # Optimization metrics
    used_action_chaining: bool = False
    used_omniparser: bool = False
    time_saved_ms: float = 0.0
    tokens_saved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComputerUseResult":
        """Create from dictionary."""
        status_str = data.pop("status", "pending")
        status = RequestStatus(status_str)

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}, status=status)


# ============================================================================
# Computer Use Delegate
# ============================================================================

class ComputerUseDelegate:
    """
    Delegates Computer Use tasks from JARVIS Prime to main JARVIS.

    Features:
    - Async task delegation via shared state files
    - Result polling with timeout
    - Request queueing and prioritization
    - Optimization metrics tracking
    """

    def __init__(
        self,
        default_mode: DelegationMode = DelegationMode.FULL_DELEGATION,
        enable_action_chaining: bool = True,
        enable_omniparser: bool = True,  # v6.2: Enable by default with fallback
    ):
        """
        Initialize Computer Use delegate.

        Args:
            default_mode: Default delegation mode
            enable_action_chaining: Request action chaining optimization
            enable_omniparser: Request OmniParser UI parsing (with intelligent fallback)
        """
        self.default_mode = default_mode
        self.enable_action_chaining = enable_action_chaining
        self.enable_omniparser = enable_omniparser

        self._requests: Dict[str, ComputerUseRequest] = {}
        self._results: Dict[str, ComputerUseResult] = {}

        # Ensure state directory exists
        COMPUTER_USE_STATE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"[COMPUTER USE DELEGATE] Initialized "
            f"(mode={default_mode.value}, "
            f"action_chaining={enable_action_chaining}, "
            f"omniparser={enable_omniparser})"
        )

    async def execute_task(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = REQUEST_TIMEOUT,
        mode: Optional[DelegationMode] = None,
    ) -> ComputerUseResult:
        """
        Delegate a Computer Use task to main JARVIS.

        Args:
            goal: The task goal
            context: Optional context dictionary
            timeout: Request timeout in seconds
            mode: Delegation mode (defaults to instance default)

        Returns:
            ComputerUseResult with task outcome
        """
        request_id = f"prime-{int(time.time() * 1000)}"

        request = ComputerUseRequest(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            goal=goal,
            context=context or {},
            mode=mode or self.default_mode,
            use_action_chaining=self.enable_action_chaining,
            use_omniparser=self.enable_omniparser,
            requester="jarvis-prime",
            timeout_seconds=timeout,
        )

        logger.info(f"[COMPUTER USE DELEGATE] Delegating task: {goal}")

        # Write request
        await self._write_request(request)
        self._requests[request_id] = request

        # Wait for result
        result = await self._wait_for_result(request_id, timeout)

        logger.info(
            f"[COMPUTER USE DELEGATE] Task completed: "
            f"success={result.success}, "
            f"time={result.execution_time_ms:.0f}ms"
        )

        return result

    async def check_jarvis_availability(self) -> bool:
        """
        Check if main JARVIS Computer Use is available.

        Returns:
            True if JARVIS is running and has Computer Use enabled
        """
        try:
            if not COMPUTER_USE_STATE_FILE.exists():
                return False

            content = COMPUTER_USE_STATE_FILE.read_text()
            state = json.loads(content)

            # Check if state is recent (not stale)
            last_update = state.get("last_update", "")
            if last_update:
                update_time = datetime.fromisoformat(last_update)
                age_seconds = (datetime.now() - update_time).total_seconds()

                if age_seconds < 120:  # Fresh within 2 minutes
                    return True

        except Exception as e:
            logger.debug(f"Failed to check JARVIS availability: {e}")

        return False

    async def get_jarvis_capabilities(self) -> Dict[str, bool]:
        """
        Get JARVIS Computer Use capabilities.

        Returns:
            Dictionary of capability flags
        """
        try:
            if COMPUTER_USE_STATE_FILE.exists():
                content = COMPUTER_USE_STATE_FILE.read_text()
                state = json.loads(content)

                return {
                    "available": True,
                    "action_chaining_enabled": state.get("action_chaining_enabled", False),
                    "omniparser_enabled": state.get("omniparser_enabled", False),
                    "omniparser_initialized": state.get("omniparser_initialized", False),
                }
        except Exception:
            pass

        return {
            "available": False,
            "action_chaining_enabled": False,
            "omniparser_enabled": False,
            "omniparser_initialized": False,
        }

    async def _write_request(self, request: ComputerUseRequest) -> None:
        """Write request to shared file."""
        try:
            # Load existing requests
            requests = []
            if COMPUTER_USE_REQUESTS_FILE.exists():
                try:
                    content = COMPUTER_USE_REQUESTS_FILE.read_text()
                    requests = json.loads(content)
                except json.JSONDecodeError:
                    requests = []

            # Add new request
            requests.append(request.to_dict())

            # Keep only last 100 requests
            requests = requests[-100:]

            # Write back
            COMPUTER_USE_REQUESTS_FILE.write_text(json.dumps(requests, indent=2))

        except Exception as e:
            logger.error(f"Failed to write Computer Use request: {e}")
            raise

    async def _wait_for_result(
        self,
        request_id: str,
        timeout: float,
    ) -> ComputerUseResult:
        """
        Wait for task result with timeout.

        Args:
            request_id: Request ID to wait for
            timeout: Timeout in seconds

        Returns:
            ComputerUseResult
        """
        start_time = time.time()

        while (time.time() - start_time) < timeout:
            # Check for result
            result = await self._read_result(request_id)

            if result:
                self._results[request_id] = result
                return result

            # Poll interval
            await asyncio.sleep(POLLING_INTERVAL)

        # Timeout
        logger.warning(f"[COMPUTER USE DELEGATE] Task timeout after {timeout}s")
        return ComputerUseResult(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            status=RequestStatus.TIMEOUT,
            success=False,
            error_message=f"Task timeout after {timeout} seconds",
        )

    async def _read_result(self, request_id: str) -> Optional[ComputerUseResult]:
        """Read result from shared file."""
        try:
            if not COMPUTER_USE_RESULTS_FILE.exists():
                return None

            content = COMPUTER_USE_RESULTS_FILE.read_text()
            results = json.loads(content)

            # Find result for this request
            for result_data in results:
                if result_data.get("request_id") == request_id:
                    return ComputerUseResult.from_dict(result_data)

        except Exception as e:
            logger.debug(f"Failed to read Computer Use result: {e}")

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get delegation statistics."""
        total_requests = len(self._requests)
        total_results = len(self._results)
        successful = sum(1 for r in self._results.values() if r.success)

        return {
            "total_requests": total_requests,
            "total_results": total_results,
            "successful": successful,
            "success_rate": (successful / total_results * 100) if total_results > 0 else 0,
            "default_mode": self.default_mode.value,
            "action_chaining_enabled": self.enable_action_chaining,
            "omniparser_enabled": self.enable_omniparser,
        }


# ============================================================================
# Atomic Transaction Support (v4.0)
# ============================================================================


class AtomicRequestTransaction:
    """
    Atomic file transaction for request/result files.

    Features:
    - fcntl-based file locking
    - Atomic write (temp + rename)
    - Rollback on failure
    - Retry with backoff
    """

    def __init__(
        self,
        file_path: Path,
        lock_timeout: float = CU_DELEGATE_LOCK_TIMEOUT,
    ):
        self._file_path = file_path
        self._lock_timeout = lock_timeout
        self._lock_file: Optional[Path] = None
        self._fd: Optional[int] = None

    @asynccontextmanager
    async def transaction(self):
        """Context manager for atomic transaction."""
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = self._file_path.with_suffix(".lock")

        # Acquire lock
        for attempt in range(CU_DELEGATE_RETRY_ATTEMPTS):
            try:
                await self._acquire_lock()
                break
            except BlockingIOError:
                if attempt >= CU_DELEGATE_RETRY_ATTEMPTS - 1:
                    raise RuntimeError(f"Could not acquire lock for {self._file_path}")
                wait = (2 ** attempt) * (0.1 + random.random() * 0.1)
                await asyncio.sleep(wait)

        try:
            yield AtomicContext(self._file_path)
        finally:
            await self._release_lock()

    async def _acquire_lock(self) -> None:
        """Acquire exclusive file lock."""
        self._fd = os.open(str(self._lock_file), os.O_RDWR | os.O_CREAT, 0o644)

        start = time.time()
        while time.time() - start < self._lock_timeout:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                await asyncio.sleep(0.05)

        os.close(self._fd)
        self._fd = None
        raise BlockingIOError(f"Lock timeout for {self._file_path}")

    async def _release_lock(self) -> None:
        """Release file lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except Exception:
                pass
            finally:
                self._fd = None

        if self._lock_file and self._lock_file.exists():
            try:
                self._lock_file.unlink()
            except Exception:
                pass


class AtomicContext:
    """Context for atomic file operations."""

    def __init__(self, file_path: Path):
        self._file_path = file_path
        self._original: Optional[str] = None

    def read(self) -> Any:
        """Read current file content."""
        try:
            if self._file_path.exists():
                self._original = self._file_path.read_text()
                return json.loads(self._original)
            return {}
        except json.JSONDecodeError:
            return {}

    def write(self, data: Any) -> bool:
        """Write data atomically."""
        try:
            temp_path = self._file_path.with_suffix(".tmp")
            content = json.dumps(data, indent=2)
            temp_path.write_text(content)
            temp_path.replace(self._file_path)
            return True
        except Exception as e:
            logger.error(f"[ATOMIC] Write error: {e}")
            if self._original:
                try:
                    self._file_path.write_text(self._original)
                except Exception:
                    pass
            return False


# ============================================================================
# Enhanced Health-Aware Delegate (v4.0)
# ============================================================================


class HealthAwareComputerUseDelegate(ComputerUseDelegate):
    """
    Enhanced delegate with health monitoring and adaptive behavior.

    Features:
    - JARVIS health monitoring
    - Adaptive timeout based on task complexity
    - Atomic request transactions
    - Event streaming integration
    - Automatic retry with backoff
    """

    def __init__(
        self,
        default_mode: DelegationMode = DelegationMode.FULL_DELEGATION,
        enable_action_chaining: bool = True,
        enable_omniparser: bool = True,
    ):
        super().__init__(
            default_mode=default_mode,
            enable_action_chaining=enable_action_chaining,
            enable_omniparser=enable_omniparser,
        )

        # Atomic transaction managers
        self._request_transaction = AtomicRequestTransaction(COMPUTER_USE_REQUESTS_FILE)
        self._result_transaction = AtomicRequestTransaction(COMPUTER_USE_RESULTS_FILE)

        # Health monitoring
        self._jarvis_healthy = True
        self._last_health_check = 0.0
        self._health_check_interval = CU_DELEGATE_HEALTH_CHECK_INTERVAL

        # Event buffer
        self._event_buffer: Deque[Dict[str, Any]] = deque(
            maxlen=CU_DELEGATE_EVENT_BUFFER_SIZE
        )

        # Latency tracking
        self._latency_samples: Deque[float] = deque(maxlen=20)
        self._avg_latency_ms = 0.0

        logger.info("[DELEGATE] Health-aware delegate initialized")

    def estimate_timeout(self, goal: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Estimate adaptive timeout based on task complexity.

        Args:
            goal: Task goal
            context: Task context

        Returns:
            Estimated timeout in seconds
        """
        base_timeout = CU_DELEGATE_BASE_TIMEOUT

        # Adjust based on goal complexity
        words = len(goal.split())
        if words > 20:
            base_timeout *= CU_DELEGATE_COMPLEXITY_MULTIPLIER
        if words > 50:
            base_timeout *= CU_DELEGATE_COMPLEXITY_MULTIPLIER

        # Adjust based on context size
        if context:
            context_size = len(json.dumps(context))
            if context_size > 1000:
                base_timeout *= 1.2
            if context_size > 5000:
                base_timeout *= 1.5

        # Adjust based on historical latency
        if self._avg_latency_ms > 5000:
            base_timeout *= 1.3

        return min(base_timeout, 300.0)  # Cap at 5 minutes

    async def check_jarvis_health(self, force: bool = False) -> bool:
        """
        Check JARVIS health with caching.

        Args:
            force: Force health check even if cache is valid

        Returns:
            True if JARVIS is healthy
        """
        now = time.time()

        # Use cached result if recent
        if not force and now - self._last_health_check < self._health_check_interval:
            return self._jarvis_healthy

        self._last_health_check = now

        try:
            # Check health file
            if COMPUTER_USE_HEALTH_FILE.exists():
                health_data = json.loads(COMPUTER_USE_HEALTH_FILE.read_text())
                jarvis_health = health_data.get("jarvis", {})

                last_heartbeat = jarvis_health.get("last_heartbeat", 0)
                is_stale = now - last_heartbeat > CU_DELEGATE_STALE_THRESHOLD

                self._jarvis_healthy = jarvis_health.get("is_healthy", False) and not is_stale
            else:
                # Fall back to state file check
                self._jarvis_healthy = await self.check_jarvis_availability()

            return self._jarvis_healthy

        except Exception as e:
            logger.warning(f"[DELEGATE] Health check failed: {e}")
            return self._jarvis_healthy

    async def execute_task_with_health_check(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        mode: Optional[DelegationMode] = None,
        require_healthy: bool = True,
    ) -> ComputerUseResult:
        """
        Execute task with health checking and adaptive timeout.

        Args:
            goal: Task goal
            context: Task context
            timeout: Override timeout (uses adaptive if None)
            mode: Delegation mode
            require_healthy: Fail fast if JARVIS is unhealthy

        Returns:
            ComputerUseResult
        """
        # Check health first
        if require_healthy:
            is_healthy = await self.check_jarvis_health()
            if not is_healthy:
                logger.warning("[DELEGATE] JARVIS is unhealthy, skipping task")
                return ComputerUseResult(
                    request_id=f"prime-{int(time.time() * 1000)}",
                    timestamp=datetime.now().isoformat(),
                    status=RequestStatus.FAILED,
                    success=False,
                    error_message="JARVIS is not available or unhealthy",
                )

        # Use adaptive timeout
        actual_timeout = timeout or self.estimate_timeout(goal, context)

        # Record start time
        start_time = time.time()

        try:
            result = await self.execute_task(
                goal=goal,
                context=context,
                timeout=actual_timeout,
                mode=mode,
            )

            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            self._latency_samples.append(latency_ms)
            self._avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

            # Buffer event
            self._event_buffer.append({
                "type": "task_completed",
                "timestamp": time.time(),
                "goal": goal[:100],
                "success": result.success,
                "latency_ms": latency_ms,
            })

            return result

        except Exception as e:
            logger.error(f"[DELEGATE] Task execution failed: {e}")

            # Buffer error event
            self._event_buffer.append({
                "type": "task_failed",
                "timestamp": time.time(),
                "goal": goal[:100],
                "error": str(e),
            })

            return ComputerUseResult(
                request_id=f"prime-{int(time.time() * 1000)}",
                timestamp=datetime.now().isoformat(),
                status=RequestStatus.FAILED,
                success=False,
                error_message=str(e),
            )

    async def _write_request_atomic(self, request: ComputerUseRequest) -> bool:
        """Write request with atomic transaction."""
        try:
            async with self._request_transaction.transaction() as tx:
                data = tx.read()
                requests = data.get("requests", [])
                requests.append(request.to_dict())
                requests = requests[-100:]  # Keep last 100
                data["requests"] = requests
                data["last_update"] = time.time()
                return tx.write(data)
        except Exception as e:
            logger.error(f"[DELEGATE] Atomic write failed: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics."""
        base_stats = super().get_statistics()

        base_stats.update({
            "jarvis_healthy": self._jarvis_healthy,
            "avg_latency_ms": round(self._avg_latency_ms, 1),
            "recent_events": len(self._event_buffer),
            "last_health_check": self._last_health_check,
        })

        return base_stats


# ============================================================================
# Global Instance
# ============================================================================

_delegate_instance: Optional[ComputerUseDelegate] = None
_health_aware_delegate: Optional[HealthAwareComputerUseDelegate] = None


def get_computer_use_delegate(
    mode: DelegationMode = DelegationMode.FULL_DELEGATION,
    enable_action_chaining: bool = True,
    enable_omniparser: bool = True,  # v6.2: Enable by default
) -> ComputerUseDelegate:
    """Get or create the global Computer Use delegate."""
    global _delegate_instance

    if _delegate_instance is None:
        _delegate_instance = ComputerUseDelegate(
            default_mode=mode,
            enable_action_chaining=enable_action_chaining,
            enable_omniparser=enable_omniparser,
        )

    return _delegate_instance


# ============================================================================
# Convenience Functions
# ============================================================================

async def delegate_computer_use_task(
    goal: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = REQUEST_TIMEOUT,
) -> ComputerUseResult:
    """Convenience function to delegate a Computer Use task."""
    delegate = get_computer_use_delegate()
    return await delegate.execute_task(goal, context, timeout)


async def check_computer_use_available() -> bool:
    """Check if Computer Use delegation is available."""
    delegate = get_computer_use_delegate()
    return await delegate.check_jarvis_availability()
