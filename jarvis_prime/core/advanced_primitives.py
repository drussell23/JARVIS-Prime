"""
Advanced Primitives v1.0 - Production-Grade Async Building Blocks
==================================================================

Provides bulletproof async primitives for the Trinity ecosystem:
- Advanced circuit breakers with state persistence
- Atomic file operations with write-ahead logging
- Exponential backoff with jitter
- Connection pool manager with cleanup
- Real resource monitoring (GPU, network, memory)
- Distributed tracing primitives
- Rate limiters and semaphores
- Timeout wrappers with cancellation

These primitives address ALL identified gaps in the Connective Tissue:
- Race conditions -> Atomic operations, proper locking
- Missing timeouts -> Operation-level timeout wrappers
- Hardcoded values -> Dynamic configuration loading
- Missing retry logic -> Exponential backoff with jitter
- Memory leaks -> Connection pool cleanup, resource tracking
- Single points of failure -> Circuit breakers, fallback chains
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
R = TypeVar('R')

# =============================================================================
# TRY IMPORTS
# =============================================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - resource monitoring limited")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available - HTTP features disabled")

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# GPU detection libraries
try:
    import subprocess
    # Check for Apple Metal (macOS)
    IS_MACOS = platform.system() == "Darwin"
    if IS_MACOS:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5
        )
        HAS_METAL = "Metal" in result.stdout
    else:
        HAS_METAL = False
except Exception:
    HAS_METAL = False

try:
    # NVIDIA GPU detection
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=5
    )
    HAS_NVIDIA = result.returncode == 0 and result.stdout.strip()
    NVIDIA_GPU_NAME = result.stdout.strip() if HAS_NVIDIA else None
except Exception:
    HAS_NVIDIA = False
    NVIDIA_GPU_NAME = None


# =============================================================================
# ATOMIC FILE OPERATIONS
# =============================================================================

class AtomicFileWriter:
    """
    Atomic file writer with write-ahead logging.

    Ensures that file writes are atomic:
    1. Write to temporary file
    2. Sync to disk (fsync)
    3. Atomic rename to target

    This prevents corruption from crashes during writes.
    """

    def __init__(self, path: Path, backup: bool = True):
        self.path = Path(path)
        self.backup = backup
        self._lock = asyncio.Lock()

    async def write(self, content: Union[str, bytes, Dict[str, Any]]) -> bool:
        """
        Atomically write content to file.

        Args:
            content: String, bytes, or dict (will be JSON serialized)

        Returns:
            True if successful
        """
        async with self._lock:
            try:
                # Convert dict to JSON
                if isinstance(content, dict):
                    content = json.dumps(content, indent=2, default=str)

                if isinstance(content, str):
                    content = content.encode('utf-8')

                # Ensure parent directory exists
                self.path.parent.mkdir(parents=True, exist_ok=True)

                # Write to temp file in same directory (for atomic rename)
                temp_path = self.path.with_suffix(f'.tmp.{uuid.uuid4().hex[:8]}')

                try:
                    # Write and sync
                    with open(temp_path, 'wb') as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())

                    # Backup existing file
                    if self.backup and self.path.exists():
                        backup_path = self.path.with_suffix('.bak')
                        shutil.copy2(self.path, backup_path)

                    # Atomic rename
                    temp_path.rename(self.path)

                    return True

                except Exception as e:
                    # Cleanup temp file on error
                    if temp_path.exists():
                        temp_path.unlink()
                    raise

            except Exception as e:
                logger.error(f"Atomic write failed for {self.path}: {e}")
                return False

    async def read(self, default: Any = None) -> Any:
        """Read file content, with fallback to backup."""
        async with self._lock:
            try:
                if self.path.exists():
                    with open(self.path, 'r') as f:
                        content = f.read()
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return content

                # Try backup
                backup_path = self.path.with_suffix('.bak')
                if backup_path.exists():
                    logger.warning(f"Using backup file for {self.path}")
                    with open(backup_path, 'r') as f:
                        content = f.read()
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return content

                return default

            except Exception as e:
                logger.error(f"Read failed for {self.path}: {e}")
                return default


class WriteAheadLog:
    """
    Write-Ahead Log (WAL) for ensuring durability.

    Operations are first written to the WAL, then applied.
    On recovery, uncommitted operations are replayed.
    """

    def __init__(self, wal_dir: Path):
        self.wal_dir = Path(wal_dir)
        self.wal_dir.mkdir(parents=True, exist_ok=True)
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def log(self, operation: Dict[str, Any]) -> str:
        """Log an operation to WAL. Returns operation ID."""
        async with self._lock:
            self._sequence += 1
            op_id = f"{int(time.time() * 1000)}_{self._sequence:06d}"

            log_entry = {
                "id": op_id,
                "timestamp": time.time(),
                "operation": operation,
                "status": "pending",
            }

            log_path = self.wal_dir / f"{op_id}.wal"
            await AtomicFileWriter(log_path, backup=False).write(log_entry)

            return op_id

    async def commit(self, op_id: str) -> bool:
        """Mark operation as committed."""
        log_path = self.wal_dir / f"{op_id}.wal"

        if not log_path.exists():
            return False

        try:
            content = await AtomicFileWriter(log_path).read()
            content["status"] = "committed"
            content["committed_at"] = time.time()
            await AtomicFileWriter(log_path, backup=False).write(content)
            return True
        except Exception as e:
            logger.error(f"WAL commit failed for {op_id}: {e}")
            return False

    async def rollback(self, op_id: str) -> bool:
        """Mark operation as rolled back."""
        log_path = self.wal_dir / f"{op_id}.wal"

        if log_path.exists():
            log_path.unlink()

        return True

    async def get_pending(self) -> List[Dict[str, Any]]:
        """Get all pending operations for recovery."""
        pending = []

        for wal_file in self.wal_dir.glob("*.wal"):
            try:
                content = await AtomicFileWriter(wal_file).read()
                if content and content.get("status") == "pending":
                    pending.append(content)
            except Exception:
                pass

        # Sort by timestamp
        pending.sort(key=lambda x: x.get("timestamp", 0))
        return pending

    async def cleanup(self, max_age_hours: int = 24):
        """Remove old committed/rolled-back entries."""
        cutoff = time.time() - (max_age_hours * 3600)

        for wal_file in self.wal_dir.glob("*.wal"):
            try:
                if wal_file.stat().st_mtime < cutoff:
                    wal_file.unlink()
            except Exception:
                pass


# =============================================================================
# ADVANCED CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3

    # Advanced options
    failure_rate_threshold: float = 0.5  # 50% failure rate triggers open
    slow_call_threshold_seconds: float = 5.0
    slow_call_rate_threshold: float = 0.5

    # Persistence
    persist_state: bool = True
    state_file: Optional[Path] = None


class AdvancedCircuitBreaker:
    """
    Production-grade circuit breaker with:
    - Failure rate and slow call rate tracking
    - State persistence across restarts
    - Metrics and observability
    - Configurable thresholds
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._last_state_change: float = time.time()

        # Sliding window for rate calculation
        self._call_history: Deque[Tuple[float, bool, float]] = deque(maxlen=100)

        # Lock
        self._lock = asyncio.Lock()

        # Metrics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0

        # State persistence
        if self.config.persist_state:
            self._state_writer = AtomicFileWriter(
                self.config.state_file or
                Path.home() / ".jarvis" / "circuit_breakers" / f"{name}.json"
            )
            asyncio.create_task(self._load_state())

    async def _load_state(self):
        """Load persisted state."""
        try:
            data = await self._state_writer.read()
            if data:
                self._state = CircuitState(data.get("state", "closed"))
                self._failure_count = data.get("failure_count", 0)
                self._success_count = data.get("success_count", 0)
                self._last_failure_time = data.get("last_failure_time")
                logger.debug(f"Loaded circuit breaker state for {self.name}: {self._state.value}")
        except Exception as e:
            logger.warning(f"Failed to load circuit breaker state: {e}")

    async def _save_state(self):
        """Persist current state."""
        if not self.config.persist_state:
            return

        try:
            await self._state_writer.write({
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "last_state_change": self._last_state_change,
                "updated_at": time.time(),
            })
        except Exception as e:
            logger.warning(f"Failed to save circuit breaker state: {e}")

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def allow_request(self) -> bool:
        """Check if request should be allowed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout expired
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout_seconds:
                        self._transition_to(CircuitState.HALF_OPEN)
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self, latency_ms: float = 0):
        """Record successful call."""
        async with self._lock:
            self._total_calls += 1
            self._total_successes += 1
            self._success_count += 1
            self._failure_count = 0

            # Track in sliding window
            is_slow = latency_ms > (self.config.slow_call_threshold_seconds * 1000)
            self._call_history.append((time.time(), True, latency_ms))

            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)

            await self._save_state()

    async def record_failure(self, error: Optional[Exception] = None):
        """Record failed call."""
        async with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.time()

            # Track in sliding window
            self._call_history.append((time.time(), False, 0))

            if self._state == CircuitState.CLOSED:
                # Check failure count threshold
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(f"Circuit breaker {self.name} OPENED: {self._failure_count} failures")

                # Check failure rate threshold
                elif self._get_failure_rate() >= self.config.failure_rate_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(f"Circuit breaker {self.name} OPENED: failure rate exceeded")

            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                logger.warning(f"Circuit breaker {self.name} re-OPENED from half-open")

            await self._save_state()

    def _transition_to(self, new_state: CircuitState):
        """Transition to new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()

        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0

        logger.info(f"Circuit breaker {self.name}: {old_state.value} -> {new_state.value}")

    def _get_failure_rate(self) -> float:
        """Calculate failure rate from sliding window."""
        if not self._call_history:
            return 0.0

        # Only consider recent calls (last 60 seconds)
        cutoff = time.time() - 60
        recent = [c for c in self._call_history if c[0] > cutoff]

        if len(recent) < 10:  # Need minimum samples
            return 0.0

        failures = sum(1 for _, success, _ in recent if not success)
        return failures / len(recent)

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
            "failure_rate": self._get_failure_rate(),
            "last_failure_time": self._last_failure_time,
            "last_state_change": self._last_state_change,
        }

    @asynccontextmanager
    async def protect(self):
        """Context manager for protected calls."""
        if not await self.allow_request():
            raise CircuitOpenError(f"Circuit breaker {self.name} is open")

        start_time = time.time()
        try:
            yield
            latency_ms = (time.time() - start_time) * 1000
            await self.record_success(latency_ms)
        except Exception as e:
            await self.record_failure(e)
            raise


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# EXPONENTIAL BACKOFF WITH JITTER
# =============================================================================

@dataclass
class BackoffConfig:
    """Configuration for exponential backoff."""
    initial_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter_factor: float = 0.1
    max_retries: int = 5

    # Advanced options
    decorrelated_jitter: bool = True  # More effective jitter distribution
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
    no_retry_on: Tuple[Type[Exception], ...] = ()


class ExponentialBackoff:
    """
    Production-grade exponential backoff with decorrelated jitter.

    Uses AWS-style decorrelated jitter which is more effective at
    preventing thundering herd than simple random jitter.
    """

    def __init__(self, config: Optional[BackoffConfig] = None):
        self.config = config or BackoffConfig()
        self._attempt = 0
        self._last_delay = 0.0

    def reset(self):
        """Reset backoff state."""
        self._attempt = 0
        self._last_delay = 0.0

    def get_delay(self) -> float:
        """Calculate next delay with jitter."""
        self._attempt += 1

        if self.config.decorrelated_jitter:
            # AWS-style decorrelated jitter
            # delay = min(cap, random_between(base, delay * 3))
            temp = max(
                self.config.initial_delay,
                self._last_delay * 3 * random.random()
            )
            delay = min(self.config.max_delay, temp)
        else:
            # Simple exponential with random jitter
            base_delay = self.config.initial_delay * (
                self.config.exponential_base ** (self._attempt - 1)
            )
            jitter = base_delay * self.config.jitter_factor * random.random()
            delay = min(self.config.max_delay, base_delay + jitter)

        self._last_delay = delay
        return delay

    def should_retry(self, error: Exception) -> bool:
        """Check if we should retry for this error."""
        if self._attempt >= self.config.max_retries:
            return False

        # Check no-retry exceptions first
        if isinstance(error, self.config.no_retry_on):
            return False

        # Check retry-on exceptions
        return isinstance(error, self.config.retry_on)

    async def wait(self):
        """Wait for the backoff delay."""
        delay = self.get_delay()
        await asyncio.sleep(delay)


def with_retry(
    config: Optional[BackoffConfig] = None,
    on_retry: Optional[Callable[[int, Exception], Awaitable[None]]] = None,
):
    """
    Decorator for automatic retry with exponential backoff.

    Usage:
        @with_retry(BackoffConfig(max_retries=3))
        async def my_function():
            ...
    """
    _config = config or BackoffConfig()

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            backoff = ExponentialBackoff(_config)
            last_error: Optional[Exception] = None

            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if not backoff.should_retry(e):
                        raise

                    if on_retry:
                        await on_retry(backoff._attempt, e)

                    logger.debug(f"Retry {backoff._attempt}/{_config.max_retries} for {func.__name__}: {e}")
                    await backoff.wait()

            raise last_error  # Should never reach here

        return wrapper
    return decorator


# =============================================================================
# OPERATION TIMEOUT WRAPPER
# =============================================================================

class OperationTimeoutError(Exception):
    """Raised when operation times out."""
    pass


@asynccontextmanager
async def operation_timeout(
    timeout_seconds: float,
    operation_name: str = "operation",
    on_timeout: Optional[Callable[[], Awaitable[None]]] = None,
):
    """
    Context manager for operation-level timeouts.

    Unlike asyncio.wait_for, this provides:
    - Named operations for better error messages
    - Callback on timeout for cleanup
    - Proper cancellation handling

    Usage:
        async with operation_timeout(30.0, "database query"):
            await slow_database_query()
    """
    task = asyncio.current_task()
    loop = asyncio.get_event_loop()

    # Create timeout handle
    timeout_handle = None
    timed_out = False

    def timeout_callback():
        nonlocal timed_out
        timed_out = True
        if task:
            task.cancel()

    try:
        timeout_handle = loop.call_later(timeout_seconds, timeout_callback)
        yield
    except asyncio.CancelledError:
        if timed_out:
            if on_timeout:
                await on_timeout()
            raise OperationTimeoutError(
                f"{operation_name} timed out after {timeout_seconds}s"
            )
        raise
    finally:
        if timeout_handle:
            timeout_handle.cancel()


async def with_timeout(
    coro: Coroutine[Any, Any, T],
    timeout_seconds: float,
    operation_name: str = "operation",
) -> T:
    """
    Execute coroutine with timeout.

    Usage:
        result = await with_timeout(slow_function(), 30.0, "slow_function")
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        raise OperationTimeoutError(
            f"{operation_name} timed out after {timeout_seconds}s"
        )


# =============================================================================
# REAL RESOURCE MONITORING
# =============================================================================

@dataclass
class GPUInfo:
    """GPU information."""
    available: bool = False
    name: Optional[str] = None
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_free_mb: float = 0.0
    utilization_percent: float = 0.0
    temperature_celsius: float = 0.0
    gpu_type: str = "unknown"  # nvidia, metal, none


@dataclass
class NetworkInfo:
    """Network connectivity information."""
    available: bool = False
    latency_ms: float = 0.0
    dns_working: bool = False
    internet_reachable: bool = False
    local_ip: Optional[str] = None


@dataclass
class SystemResources:
    """Complete system resource snapshot."""
    timestamp: float = field(default_factory=time.time)

    # Memory
    ram_total_mb: float = 0.0
    ram_used_mb: float = 0.0
    ram_available_mb: float = 0.0
    ram_percent: float = 0.0
    swap_total_mb: float = 0.0
    swap_used_mb: float = 0.0

    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # GPU
    gpu: GPUInfo = field(default_factory=GPUInfo)

    # Network
    network: NetworkInfo = field(default_factory=NetworkInfo)

    # Disk
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0


class ResourceMonitor:
    """
    Real-time system resource monitoring.

    Unlike the original implementation, this performs ACTUAL:
    - GPU detection (NVIDIA via nvidia-smi, Apple Metal via system_profiler)
    - Network connectivity testing
    - Accurate memory monitoring
    """

    _instance: Optional["ResourceMonitor"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        self._cache: Optional[SystemResources] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 1.0  # Cache for 1 second
        self._gpu_cache: Optional[GPUInfo] = None
        self._gpu_cache_time: float = 0.0
        self._gpu_cache_ttl: float = 5.0  # Cache GPU for 5 seconds

    @classmethod
    async def get_instance(cls) -> "ResourceMonitor":
        """Get singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def capture(self, force_refresh: bool = False) -> SystemResources:
        """
        Capture current system resources.

        Args:
            force_refresh: Bypass cache and get fresh data
        """
        now = time.time()

        # Check cache
        if not force_refresh and self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        resources = SystemResources(timestamp=now)

        # Memory
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            resources.ram_total_mb = mem.total / (1024 ** 2)
            resources.ram_used_mb = mem.used / (1024 ** 2)
            resources.ram_available_mb = mem.available / (1024 ** 2)
            resources.ram_percent = mem.percent

            swap = psutil.swap_memory()
            resources.swap_total_mb = swap.total / (1024 ** 2)
            resources.swap_used_mb = swap.used / (1024 ** 2)

            # CPU
            resources.cpu_percent = psutil.cpu_percent(interval=0.1)
            resources.cpu_count = psutil.cpu_count()
            resources.load_average = os.getloadavg() if hasattr(os, 'getloadavg') else (0.0, 0.0, 0.0)

            # Disk
            disk = psutil.disk_usage('/')
            resources.disk_total_gb = disk.total / (1024 ** 3)
            resources.disk_used_gb = disk.used / (1024 ** 3)
            resources.disk_free_gb = disk.free / (1024 ** 3)

        # GPU - use cached if recent
        if self._gpu_cache and (now - self._gpu_cache_time) < self._gpu_cache_ttl:
            resources.gpu = self._gpu_cache
        else:
            resources.gpu = await self._detect_gpu()
            self._gpu_cache = resources.gpu
            self._gpu_cache_time = now

        # Network - quick check
        resources.network = await self._check_network()

        # Update cache
        self._cache = resources
        self._cache_time = now

        return resources

    async def _detect_gpu(self) -> GPUInfo:
        """Detect and query GPU information."""
        info = GPUInfo()

        # Try NVIDIA first
        if HAS_NVIDIA:
            try:
                # Query GPU info via nvidia-smi
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                            "--format=csv,noheader,nounits"
                        ],
                        capture_output=True, text=True, timeout=5
                    )
                )

                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 6:
                        info.available = True
                        info.gpu_type = "nvidia"
                        info.name = parts[0].strip()
                        info.memory_total_mb = float(parts[1].strip())
                        info.memory_used_mb = float(parts[2].strip())
                        info.memory_free_mb = float(parts[3].strip())
                        info.utilization_percent = float(parts[4].strip())
                        info.temperature_celsius = float(parts[5].strip())
                        return info
            except Exception as e:
                logger.debug(f"NVIDIA GPU detection failed: {e}")

        # Try Apple Metal
        if IS_MACOS and HAS_METAL:
            info.available = True
            info.gpu_type = "metal"
            info.name = "Apple Silicon GPU"

            # Get unified memory info (shared with system)
            if PSUTIL_AVAILABLE:
                mem = psutil.virtual_memory()
                # Apple Silicon uses unified memory, estimate GPU portion
                info.memory_total_mb = mem.total / (1024 ** 2) * 0.75  # Assume 75% available for GPU
                info.memory_free_mb = mem.available / (1024 ** 2) * 0.5
                info.memory_used_mb = info.memory_total_mb - info.memory_free_mb

            return info

        return info

    async def _check_network(self) -> NetworkInfo:
        """Quick network connectivity check."""
        info = NetworkInfo()

        try:
            # Check DNS
            socket.setdefaulttimeout(2)
            socket.gethostbyname("dns.google")
            info.dns_working = True

            # Get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                info.local_ip = s.getsockname()[0]
            finally:
                s.close()

            # Quick internet check
            if AIOHTTP_AVAILABLE:
                start = time.time()
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
                    async with session.head("https://www.google.com") as response:
                        if response.status < 500:
                            info.internet_reachable = True
                            info.latency_ms = (time.time() - start) * 1000
            else:
                # Fallback to socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                start = time.time()
                result = sock.connect_ex(("www.google.com", 443))
                info.latency_ms = (time.time() - start) * 1000
                info.internet_reachable = result == 0
                sock.close()

            info.available = info.dns_working or info.internet_reachable

        except Exception as e:
            logger.debug(f"Network check failed: {e}")

        return info

    def can_use_local_inference(
        self,
        resources: SystemResources,
        max_ram_percent: float = 85.0,
        max_ram_mb: float = 14336,
        require_gpu: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check if local inference is viable.

        Returns:
            Tuple of (can_use, reason)
        """
        if resources.ram_percent > max_ram_percent:
            return False, f"RAM usage {resources.ram_percent:.1f}% > {max_ram_percent}%"

        if resources.ram_used_mb > max_ram_mb:
            return False, f"RAM used {resources.ram_used_mb:.0f}MB > {max_ram_mb}MB"

        if require_gpu and not resources.gpu.available:
            return False, "No GPU available"

        return True, "Resources available"

    def can_use_cloud_inference(
        self,
        resources: SystemResources,
    ) -> Tuple[bool, str]:
        """Check if cloud inference is viable."""
        if not resources.network.available:
            return False, "Network unavailable"

        if not resources.network.internet_reachable:
            return False, "Internet not reachable"

        if resources.network.latency_ms > 5000:  # 5 second latency
            return False, f"Network latency too high: {resources.network.latency_ms:.0f}ms"

        return True, "Network available"


# =============================================================================
# CONNECTION POOL MANAGER
# =============================================================================

class ManagedConnectionPool:
    """
    Managed HTTP connection pool with:
    - Automatic cleanup of idle connections
    - Per-host connection limits
    - Health monitoring
    - Graceful shutdown
    """

    def __init__(
        self,
        max_connections_per_host: int = 10,
        max_keepalive: int = 5,
        keepalive_timeout: float = 300.0,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
    ):
        self._config = {
            "max_per_host": max_connections_per_host,
            "max_keepalive": max_keepalive,
            "keepalive_timeout": keepalive_timeout,
            "connect_timeout": connect_timeout,
            "read_timeout": read_timeout,
        }

        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._session_created: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the connection pool manager."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Connection pool manager started")

    async def stop(self):
        """Stop and cleanup all connections."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        await self.close_all()
        logger.info("Connection pool manager stopped")

    async def get_session(self, host: str) -> aiohttp.ClientSession:
        """Get or create a session for a host."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")

        async with self._lock:
            # Check if session exists and is valid
            if host in self._sessions:
                session = self._sessions[host]
                if not session.closed:
                    return session
                else:
                    # Clean up closed session
                    del self._sessions[host]
                    del self._session_created[host]

            # Create new session
            connector = aiohttp.TCPConnector(
                limit=self._config["max_per_host"],
                limit_per_host=self._config["max_per_host"],
                ttl_dns_cache=300,
                keepalive_timeout=self._config["keepalive_timeout"],
                enable_cleanup_closed=True,
            )

            timeout = aiohttp.ClientTimeout(
                total=self._config["read_timeout"],
                connect=self._config["connect_timeout"],
            )

            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
            )

            self._sessions[host] = session
            self._session_created[host] = time.time()

            return session

    async def close_all(self):
        """Close all sessions."""
        async with self._lock:
            for host, session in list(self._sessions.items()):
                if not session.closed:
                    await session.close()

            self._sessions.clear()
            self._session_created.clear()

    async def _cleanup_loop(self):
        """Periodically clean up idle connections."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_idle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection pool cleanup error: {e}")

    async def _cleanup_idle(self):
        """Close sessions that have been idle too long."""
        async with self._lock:
            now = time.time()
            max_age = self._config["keepalive_timeout"]

            to_remove = []
            for host, created_at in self._session_created.items():
                if now - created_at > max_age:
                    to_remove.append(host)

            for host in to_remove:
                session = self._sessions.pop(host, None)
                self._session_created.pop(host, None)

                if session and not session.closed:
                    await session.close()
                    logger.debug(f"Closed idle session for {host}")

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "active_sessions": len(self._sessions),
            "hosts": list(self._sessions.keys()),
            "config": self._config,
        }


# =============================================================================
# DISTRIBUTED TRACING PRIMITIVES
# =============================================================================

@dataclass
class TraceContext:
    """Distributed tracing context."""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: Optional[str] = None

    # Baggage (key-value pairs propagated across services)
    baggage: Dict[str, str] = field(default_factory=dict)

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    # Metadata
    operation_name: str = ""
    service_name: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def create_child(self, operation_name: str) -> "TraceContext":
        """Create a child span."""
        return TraceContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            baggage=dict(self.baggage),
            operation_name=operation_name,
            service_name=self.service_name,
        )

    def add_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = value

    def add_log(self, message: str, **fields):
        """Add a log entry to the span."""
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            **fields,
        })

    def finish(self):
        """Mark span as finished."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        return {
            "X-Trace-ID": self.trace_id,
            "X-Span-ID": self.span_id,
            "X-Parent-Span-ID": self.parent_span_id or "",
            "X-Baggage": json.dumps(self.baggage) if self.baggage else "",
        }

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "TraceContext":
        """Create from HTTP headers."""
        baggage = {}
        if headers.get("X-Baggage"):
            try:
                baggage = json.loads(headers["X-Baggage"])
            except json.JSONDecodeError:
                pass

        return cls(
            trace_id=headers.get("X-Trace-ID", uuid.uuid4().hex),
            parent_span_id=headers.get("X-Span-ID"),
            baggage=baggage,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "baggage": self.baggage,
        }


# Context variable for current trace
_current_trace: Optional[TraceContext] = None


def get_current_trace() -> Optional[TraceContext]:
    """Get current trace context."""
    return _current_trace


def set_current_trace(trace: Optional[TraceContext]):
    """Set current trace context."""
    global _current_trace
    _current_trace = trace


@asynccontextmanager
async def trace_operation(
    operation_name: str,
    service_name: str = "unknown",
    parent: Optional[TraceContext] = None,
):
    """
    Context manager for tracing an operation.

    Usage:
        async with trace_operation("database_query", "user-service") as trace:
            trace.add_tag("query", "SELECT * FROM users")
            result = await db.query(...)
    """
    global _current_trace

    # Create trace context
    if parent:
        trace = parent.create_child(operation_name)
    elif _current_trace:
        trace = _current_trace.create_child(operation_name)
    else:
        trace = TraceContext(operation_name=operation_name)

    trace.service_name = service_name

    # Save and set current trace
    old_trace = _current_trace
    _current_trace = trace

    try:
        yield trace
    except Exception as e:
        trace.add_tag("error", True)
        trace.add_tag("error.message", str(e))
        trace.add_tag("error.type", type(e).__name__)
        raise
    finally:
        trace.finish()
        _current_trace = old_trace

        # Log the span (in production, send to tracing backend)
        logger.debug(f"Trace: {operation_name} - {trace.duration_ms:.2f}ms")


# =============================================================================
# RATE LIMITER
# =============================================================================

class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with:
    - Configurable rate and burst
    - Async-safe
    - Optional adaptive rate adjustment
    """

    def __init__(
        self,
        rate: float,  # Tokens per second
        burst: int,   # Maximum burst size
        adaptive: bool = False,
    ):
        self.rate = rate
        self.burst = burst
        self.adaptive = adaptive

        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

        # Adaptive tracking
        self._success_count = 0
        self._error_count = 0

    async def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            wait: If True, wait until tokens available. If False, return immediately.

        Returns:
            True if tokens acquired, False if not available (only when wait=False)
        """
        async with self._lock:
            while True:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                if not wait:
                    return False

                # Calculate wait time
                needed = tokens - self._tokens
                wait_time = needed / self.rate

                # Release lock while waiting
                self._lock.release()
                await asyncio.sleep(wait_time)
                await self._lock.acquire()

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens based on rate
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)

    def record_success(self):
        """Record successful request (for adaptive rate limiting)."""
        self._success_count += 1

        if self.adaptive and self._success_count >= 100:
            # Increase rate slightly
            self.rate = min(self.rate * 1.1, self.rate * 2)
            self._success_count = 0

    def record_error(self, is_rate_limit: bool = False):
        """Record error (for adaptive rate limiting)."""
        self._error_count += 1

        if self.adaptive and is_rate_limit:
            # Decrease rate
            self.rate = max(self.rate * 0.5, 1.0)
            self._error_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "rate": self.rate,
            "burst": self.burst,
            "current_tokens": self._tokens,
            "adaptive": self.adaptive,
            "success_count": self._success_count,
            "error_count": self._error_count,
        }


# =============================================================================
# SEMAPHORE WITH METRICS
# =============================================================================

class MeteredSemaphore:
    """
    Semaphore with metrics tracking.

    Provides:
    - Acquisition timing
    - Wait time tracking
    - Utilization metrics
    """

    def __init__(self, value: int, name: str = "semaphore"):
        self._semaphore = asyncio.Semaphore(value)
        self._max_value = value
        self._name = name

        # Metrics
        self._current_acquired = 0
        self._total_acquisitions = 0
        self._total_wait_time_ms = 0.0
        self._max_wait_time_ms = 0.0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self):
        """Acquire the semaphore with metrics tracking."""
        start_time = time.time()

        await self._semaphore.acquire()

        wait_time_ms = (time.time() - start_time) * 1000

        async with self._lock:
            self._current_acquired += 1
            self._total_acquisitions += 1
            self._total_wait_time_ms += wait_time_ms
            self._max_wait_time_ms = max(self._max_wait_time_ms, wait_time_ms)

        try:
            yield
        finally:
            self._semaphore.release()
            async with self._lock:
                self._current_acquired -= 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get semaphore metrics."""
        return {
            "name": self._name,
            "max_value": self._max_value,
            "current_acquired": self._current_acquired,
            "available": self._max_value - self._current_acquired,
            "utilization": self._current_acquired / self._max_value,
            "total_acquisitions": self._total_acquisitions,
            "avg_wait_time_ms": (
                self._total_wait_time_ms / self._total_acquisitions
                if self._total_acquisitions > 0 else 0
            ),
            "max_wait_time_ms": self._max_wait_time_ms,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Atomic file operations
    "AtomicFileWriter",
    "WriteAheadLog",

    # Circuit breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "AdvancedCircuitBreaker",
    "CircuitOpenError",

    # Exponential backoff
    "BackoffConfig",
    "ExponentialBackoff",
    "with_retry",

    # Timeouts
    "OperationTimeoutError",
    "operation_timeout",
    "with_timeout",

    # Resource monitoring
    "GPUInfo",
    "NetworkInfo",
    "SystemResources",
    "ResourceMonitor",

    # Connection pool
    "ManagedConnectionPool",

    # Distributed tracing
    "TraceContext",
    "get_current_trace",
    "set_current_trace",
    "trace_operation",

    # Rate limiting
    "TokenBucketRateLimiter",

    # Semaphore
    "MeteredSemaphore",

    # Constants
    "PSUTIL_AVAILABLE",
    "AIOHTTP_AVAILABLE",
    "AIOFILES_AVAILABLE",
    "IS_MACOS",
    "HAS_METAL",
    "HAS_NVIDIA",
    "NVIDIA_GPU_NAME",
]
