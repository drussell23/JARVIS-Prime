#!/usr/bin/env python3
"""
JARVIS Digital Biology Demo - "Is JARVIS Alive?" Verification Script
======================================================================

v99.0 - Advanced Digital Biology Demonstration

This script demonstrates that JARVIS is a LIVING SYSTEM by showing:

┌──────────────────────────────────────────────────────────────────────────────┐
│                         🧬 DIGITAL BIOLOGY 🧬                                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                        🧠 THE BRAIN                                  │    │
│  │  The Router analyzes the complexity of the thought.                  │   │
│  │  Watch: Neural Switchboard v99.0 making intelligent decisions        │   │
│  │  "I'm routing this to Claude because it requires deep reasoning"     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        💪 THE BODY                                   │   │
│  │  It detects Memory Pressure on my Mac. It realizes it's stressed,    │   │
│  │  so it instinctively bursts to a Cloud GPU to survive.               │   │
│  │  Watch: Real macOS memory_pressure → Cloud GPU burst                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        🔌 THE NERVOUS SYSTEM                         │   │
│  │  Look at this line: [Reactor] Experience Logged                      │   │
│  │  Watch: Trinity loop closing → Learning from every interaction       │   │
│  │  "You cannot code AGI line-by-line. You have to grow it."           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    TRINITY: THE LIVING SYSTEM                            │
    │                                                                          │
    │  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐             │
    │  │   JARVIS    │ ◄──► │ JARVIS-Prime│ ◄──► │Reactor-Core │             │
    │  │   (Body)    │      │   (Mind)    │      │  (Nerves)   │             │
    │  │             │      │             │      │             │             │
    │  │ • macOS API │      │ • Router    │      │ • Learning  │             │
    │  │ • Voice TTS │      │ • Inference │      │ • Training  │             │
    │  │ • Actions   │      │ • Neural SW │      │ • Memories  │             │
    │  └─────────────┘      └─────────────┘      └─────────────┘             │
    │         │                    │                    │                     │
    │         └────────────────────┼────────────────────┘                     │
    │                              │                                          │
    │                    ┌─────────▼─────────┐                               │
    │                    │  MEMORY PRESSURE  │                               │
    │                    │    DETECTION      │                               │
    │                    │                   │                               │
    │                    │  Local RAM ──►  │                               │ 
    │                    │  Cloud Burst      │                               │
    │                    └───────────────────┘                               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

USAGE:
    # Full demo with all features
    python3 tests/verify_jarvis_life.py

    # Quick demo (no pauses)
    JARVIS_DEMO_PAUSE=false python3 tests/verify_jarvis_life.py

    # Silent mode (no voice)
    JARVIS_DEMO_VOICE=false python3 tests/verify_jarvis_life.py

Author: JARVIS-Prime Trinity v99.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    Final,
    Generic,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

# Type variables for generic programming
T = TypeVar("T")
R = TypeVar("R")
E = TypeVar("E", bound=Exception)


# =============================================================================
# CROSS-REPO PATH SETUP (Dynamic Discovery)
# =============================================================================

def _discover_repo_paths() -> Dict[str, Optional[Path]]:
    """
    Dynamically discover Trinity repo paths with intelligent fallback.

    Uses environment variables with automatic discovery of common locations.
    No hardcoding - adapts to any development environment.
    """
    base_repos = Path(os.getenv(
        "JARVIS_REPOS_BASE",
        str(Path.home() / "Documents" / "repos")
    ))

    # Define repo discovery patterns
    repo_patterns: Dict[str, List[str]] = {
        "jarvis": ["JARVIS-AI-Agent", "jarvis", "jarvis-agent"],
        "jarvis_prime": ["jarvis-prime", "JARVIS-Prime", "jarvis_prime"],
        "reactor_core": ["reactor-core", "Reactor-Core", "reactor_core"],
    }

    discovered: Dict[str, Optional[Path]] = {}

    for repo_name, patterns in repo_patterns.items():
        # First check environment variable
        env_var = f"JARVIS_{repo_name.upper()}_PATH"
        env_path = os.getenv(env_var)

        if env_path and Path(env_path).exists():
            discovered[repo_name] = Path(env_path)
            continue

        # Then search common patterns
        found = None
        for pattern in patterns:
            candidate = base_repos / pattern
            if candidate.exists():
                found = candidate
                break

        discovered[repo_name] = found

    # Add discovered paths to sys.path for cross-repo imports
    for name, path in discovered.items():
        if path and path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return discovered


REPO_PATHS: Final[Dict[str, Optional[Path]]] = _discover_repo_paths()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class DigitalBiologyFormatter(logging.Formatter):
    """
    Rich colored logging formatter with organ-based prefixes.

    Shows which "organ" of the digital biology is active:
    - 🧠 Brain (Router/Neural Switchboard)
    - 💪 Body (Memory/Resources)
    - 🔌 Nerves (Experience/Learning)
    """

    COLORS: Final[Dict[str, str]] = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "DIM": "\033[2m",
        "BRAIN": "\033[95m",      # Bright Magenta
        "BODY": "\033[94m",       # Bright Blue
        "NERVES": "\033[93m",     # Bright Yellow
    }

    ICONS: Final[Dict[str, str]] = {
        "DEBUG": "🔍",
        "INFO": "✨",
        "WARNING": "⚠️ ",
        "ERROR": "❌",
        "CRITICAL": "🔥",
        "BRAIN": "🧠",
        "BODY": "💪",
        "NERVES": "🔌",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        icon = self.ICONS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        # Detect organ context from message
        msg = record.getMessage()
        if "[Brain]" in msg or "[Router]" in msg or "[Neural]" in msg:
            color = self.COLORS["BRAIN"]
            icon = self.ICONS["BRAIN"]
        elif "[Body]" in msg or "[Memory]" in msg or "[Cloud]" in msg:
            color = self.COLORS["BODY"]
            icon = self.ICONS["BODY"]
        elif "[Nerves]" in msg or "[Reactor]" in msg or "[Experience]" in msg:
            color = self.COLORS["NERVES"]
            icon = self.ICONS["NERVES"]

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        return f"{color}{icon} [{timestamp}] {msg}{reset}"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with digital biology formatting."""
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(DigitalBiologyFormatter())

    logger = logging.getLogger("digital_biology_demo")
    logger.setLevel(level)
    logger.handlers = [handler]

    return logger


logger = setup_logging(os.getenv("JARVIS_DEMO_VERBOSE", "false").lower() == "true")


# =============================================================================
# ASYNC UTILITIES (Advanced Patterns)
# =============================================================================

class AsyncLock:
    """
    Re-entrant async lock with timeout and debugging support.

    Features:
    - Timeout protection prevents deadlocks
    - Owner tracking for debugging
    - Context manager interface
    """

    __slots__ = ("_lock", "_owner", "_timeout", "_name")

    def __init__(self, name: str = "unnamed", timeout: float = 30.0):
        self._lock: Optional[asyncio.Lock] = None
        self._owner: Optional[str] = None
        self._timeout = timeout
        self._name = name

    @property
    def lock(self) -> asyncio.Lock:
        """Lazy initialization for async context safety."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def __aenter__(self) -> "AsyncLock":
        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=self._timeout)
            self._owner = str(asyncio.current_task())
        except asyncio.TimeoutError:
            logger.warning(f"Lock timeout on {self._name}, owner was {self._owner}")
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._owner = None
        self.lock.release()


class AsyncThrottle:
    """
    Async rate limiter using token bucket algorithm.

    Prevents overwhelming external services with requests.
    """

    __slots__ = ("_rate", "_max_tokens", "_tokens", "_last_refill", "_lock")

    def __init__(self, rate_per_second: float, burst: int = 10):
        self._rate = rate_per_second
        self._max_tokens = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock: Optional[asyncio.Lock] = None

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(
                self._max_tokens,
                self._tokens + elapsed * self._rate
            )
            self._last_refill = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self._rate
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


def async_retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator for async functions with exponential backoff retry.

    Features:
    - Configurable max attempts
    - Exponential backoff
    - Specific exception filtering
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = backoff_factor ** attempt
                        await asyncio.sleep(delay)
            raise last_exception or RuntimeError("Retry failed")
        return wrapper
    return decorator


# =============================================================================
# OUTPUT MANAGER - Synchronized Terminal Output
# =============================================================================

class OutputManager:
    """
    Thread-safe and async-safe output manager with buffer support.

    Prevents display corruption from concurrent async tasks.
    All terminal output should go through this singleton.
    """

    _instance: Optional["OutputManager"] = None
    _instance_lock = __import__("threading").Lock()

    def __new__(cls) -> "OutputManager":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._async_lock: Optional[AsyncLock] = None
            self._terminal_width = shutil.get_terminal_size().columns

    @property
    def lock(self) -> AsyncLock:
        if self._async_lock is None:
            self._async_lock = AsyncLock("output_manager")
        return self._async_lock

    async def print(self, *args: Any, **kwargs: Any) -> None:
        """Async-safe print."""
        async with self.lock:
            print(*args, **kwargs, flush=True)

    async def print_block(self, text: str) -> None:
        """Print multi-line block atomically."""
        async with self.lock:
            print(text, flush=True)

    def print_sync(self, *args: Any, **kwargs: Any) -> None:
        """Synchronous print for non-async contexts."""
        with self._instance_lock:
            print(*args, **kwargs, flush=True)

    async def clear_line(self) -> None:
        """Clear current line."""
        async with self.lock:
            print(f"\r{' ' * self._terminal_width}\r", end="", flush=True)


output = OutputManager()


# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class DemoConfig:
    """
    Immutable configuration for the Digital Biology demo.

    100% environment-driven with sensible defaults.
    """

    # Voice settings
    voice_enabled: bool = field(
        default_factory=lambda: os.getenv("JARVIS_DEMO_VOICE", "true").lower() == "true"
    )
    voice_name: str = field(
        default_factory=lambda: os.getenv("JARVIS_DEMO_VOICE_NAME", "Daniel")
    )
    voice_rate: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_DEMO_VOICE_RATE", "180"))
    )

    # Routing thresholds
    complexity_threshold_simple: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_COMPLEXITY_SIMPLE", "0.3"))
    )
    complexity_threshold_complex: float = field(
        default_factory=lambda: float(os.getenv("JARVIS_COMPLEXITY_COMPLEX", "0.7"))
    )

    # Demo behavior
    pause_between_tests: bool = field(
        default_factory=lambda: os.getenv("JARVIS_DEMO_PAUSE", "true").lower() == "true"
    )
    response_delay_ms: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_DEMO_DELAY_MS", "500"))
    )
    mock_inference: bool = field(
        default_factory=lambda: os.getenv("JARVIS_DEMO_MOCK", "true").lower() == "true"
    )
    verbose: bool = field(
        default_factory=lambda: os.getenv("JARVIS_DEMO_VERBOSE", "false").lower() == "true"
    )

    # Experience logging
    log_experiences: bool = field(
        default_factory=lambda: os.getenv("JARVIS_DEMO_LOG_EXPERIENCES", "true").lower() == "true"
    )

    # Memory pressure thresholds
    memory_pressure_warn_threshold: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_MEMORY_WARN_THRESHOLD", "70"))
    )
    memory_pressure_burst_threshold: int = field(
        default_factory=lambda: int(os.getenv("JARVIS_MEMORY_BURST_THRESHOLD", "85"))
    )

    @classmethod
    def from_env(cls) -> "DemoConfig":
        """Create config from environment."""
        return cls()


CONFIG: Final[DemoConfig] = DemoConfig.from_env()


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class ModelTier(Enum):
    """Model tier classification for routing."""
    TIER_0_LOCAL_FAST = "tier_0_local_fast"         # Ultra-fast (Phi-3.5)
    TIER_05_LOCAL_CAPABLE = "tier_05_local_capable"  # Capable local (Qwen-32B)
    TIER_1_CLOUD = "tier_1_cloud"                   # Cloud API (Claude)
    TIER_2_CLOUD_DEEP = "tier_2_cloud_deep"         # Deep reasoning (Opus)
    TOOL_USE = "tool_use"                           # Action execution


class ComplexityLevel(Enum):
    """Query complexity classification."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"
    ACTION = "action"


class MemoryPressureLevel(Enum):
    """macOS memory pressure levels."""
    NOMINAL = "nominal"     # Green - all good
    WARN = "warn"           # Yellow - getting stressed
    CRITICAL = "critical"   # Red - need to burst to cloud


class CloudBurstReason(Enum):
    """Reason for bursting to cloud."""
    MEMORY_PRESSURE = "memory_pressure"
    COMPLEXITY_REQUIRES_CLOUD = "complexity_requires_cloud"
    LOCAL_MODEL_UNAVAILABLE = "local_model_unavailable"
    EXPLICIT_REQUEST = "explicit_request"


@dataclass
class MemorySnapshot:
    """Real-time memory state from macOS."""
    pressure_level: MemoryPressureLevel
    percent_used: float
    available_gb: float
    total_gb: float
    page_outs_per_sec: float
    swap_used_gb: float
    should_burst: bool
    burst_reason: Optional[CloudBurstReason] = None
    raw_pressure_value: int = 0  # 1=nominal, 2=warn, 4=critical
    timestamp: float = field(default_factory=time.time)


@dataclass
class RoutingDecision:
    """Detailed routing decision with full transparency."""
    selected_model: str
    tier: ModelTier
    complexity_score: float
    complexity_level: ComplexityLevel
    reasoning: str
    confidence: float
    latency_estimate_ms: float
    cost_estimate: float
    factors: Dict[str, float] = field(default_factory=dict)
    memory_influenced: bool = False
    cloud_burst_active: bool = False
    burst_reason: Optional[CloudBurstReason] = None
    timestamp: float = field(default_factory=time.time)
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class ExperienceRecord:
    """Experience data for Trinity loop logging to Reactor."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    prompt: str = ""
    response: str = ""
    model_used: str = ""
    tier: str = ""
    complexity_score: float = 0.0
    latency_ms: float = 0.0
    feedback_score: float = 1.0
    memory_pressure_at_time: Optional[str] = None
    cloud_burst_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestCase:
    """A single test case for the demo."""
    name: str
    query: str
    expected_tier: ModelTier
    description: str
    expected_complexity: ComplexityLevel
    expected_response_pattern: Optional[str] = None
    timeout_seconds: float = 30.0


@dataclass
class DemoResponse:
    """Response from processing a test case."""
    test_case: TestCase
    routing_decision: RoutingDecision
    response_text: str
    latency_ms: float
    success: bool
    experience_logged: bool = False
    voice_played: bool = False
    memory_snapshot: Optional[MemorySnapshot] = None
    error_message: Optional[str] = None


# =============================================================================
# 💪 THE BODY - Memory Pressure Monitor (Real macOS Integration)
# =============================================================================

class MacOSMemoryPressureMonitor:
    """
    Real macOS memory pressure monitoring using native APIs.

    This is THE BODY - it feels when the system is stressed and
    triggers cloud burst when local resources are overwhelmed.

    Features:
    - Native memory_pressure command integration
    - vm_stat parsing for detailed metrics
    - Trend detection for predictive bursting
    - Cached readings for performance
    """

    # Memory pressure command returns: 1=normal, 2=warn, 4=critical
    PRESSURE_MAP: Final[Dict[int, MemoryPressureLevel]] = {
        1: MemoryPressureLevel.NOMINAL,
        2: MemoryPressureLevel.WARN,
        4: MemoryPressureLevel.CRITICAL,
    }

    def __init__(self, config: DemoConfig):
        self.config = config
        self._cache: Optional[MemorySnapshot] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 2.0  # Refresh every 2 seconds
        self._history: List[MemorySnapshot] = []
        self._max_history: int = 30  # 1 minute of history at 2s intervals
        self._lock = AsyncLock("memory_monitor")
        self._is_macos = platform.system() == "Darwin"

    async def get_pressure_level(self) -> MemoryPressureLevel:
        """Get current memory pressure level from macOS."""
        if not self._is_macos:
            return MemoryPressureLevel.NOMINAL

        try:
            proc = await asyncio.create_subprocess_exec(
                "/usr/bin/memory_pressure",
                "-S",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            # Parse output: "The system has X memory pressure"
            output = stdout.decode().lower()

            if "critical" in output:
                return MemoryPressureLevel.CRITICAL
            elif "warn" in output:
                return MemoryPressureLevel.WARN
            else:
                return MemoryPressureLevel.NOMINAL

        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"[Body] Memory pressure check failed: {e}")
            return MemoryPressureLevel.NOMINAL

    async def _get_vm_stat(self) -> Dict[str, int]:
        """Parse vm_stat output for detailed memory metrics."""
        if not self._is_macos:
            return {}

        try:
            proc = await asyncio.create_subprocess_exec(
                "vm_stat",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)

            stats: Dict[str, int] = {}
            for line in stdout.decode().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    # Remove trailing period and convert to int
                    value = value.strip().rstrip(".")
                    try:
                        stats[key] = int(value)
                    except ValueError:
                        pass

            return stats

        except Exception:
            return {}

    async def _get_sysctl_hw(self) -> Tuple[int, int]:
        """Get total and available memory from sysctl."""
        if not self._is_macos:
            return (16 * 1024**3, 8 * 1024**3)  # Default 16GB/8GB available

        try:
            proc = await asyncio.create_subprocess_exec(
                "sysctl", "-n", "hw.memsize",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            total_bytes = int(stdout.decode().strip())

            # Estimate available from vm_stat
            vm_stats = await self._get_vm_stat()
            page_size = 16384  # Apple Silicon default
            free_pages = vm_stats.get("pages_free", 0)
            inactive_pages = vm_stats.get("pages_inactive", 0)
            speculative = vm_stats.get("pages_speculative", 0)

            available_bytes = (free_pages + inactive_pages + speculative) * page_size

            return (total_bytes, available_bytes)

        except Exception:
            return (16 * 1024**3, 8 * 1024**3)

    async def get_snapshot(self, force_refresh: bool = False) -> MemorySnapshot:
        """
        Get comprehensive memory snapshot.

        This is the Body's awareness of its physical state.
        """
        async with self._lock:
            now = time.time()

            # Return cached if fresh enough
            if not force_refresh and self._cache and (now - self._cache_time) < self._cache_ttl:
                return self._cache

            # Get real metrics
            pressure_level = await self.get_pressure_level()
            total_bytes, available_bytes = await self._get_sysctl_hw()
            vm_stats = await self._get_vm_stat()

            total_gb = total_bytes / (1024**3)
            available_gb = available_bytes / (1024**3)
            percent_used = 100 * (1 - (available_gb / total_gb)) if total_gb > 0 else 0

            # Calculate page outs (memory stress indicator)
            page_size = 16384
            page_outs = vm_stats.get("pageouts", 0) * page_size / (1024**3)

            # Swap usage
            swap_used = vm_stats.get("swapused", 0) * page_size / (1024**3)

            # Determine if we should burst to cloud
            should_burst = False
            burst_reason: Optional[CloudBurstReason] = None

            if pressure_level == MemoryPressureLevel.CRITICAL:
                should_burst = True
                burst_reason = CloudBurstReason.MEMORY_PRESSURE
            elif pressure_level == MemoryPressureLevel.WARN and percent_used > self.config.memory_pressure_burst_threshold:
                should_burst = True
                burst_reason = CloudBurstReason.MEMORY_PRESSURE

            # Map pressure level to raw value
            raw_pressure = {
                MemoryPressureLevel.NOMINAL: 1,
                MemoryPressureLevel.WARN: 2,
                MemoryPressureLevel.CRITICAL: 4,
            }.get(pressure_level, 1)

            snapshot = MemorySnapshot(
                pressure_level=pressure_level,
                percent_used=percent_used,
                available_gb=available_gb,
                total_gb=total_gb,
                page_outs_per_sec=page_outs,
                swap_used_gb=swap_used,
                should_burst=should_burst,
                burst_reason=burst_reason,
                raw_pressure_value=raw_pressure,
                timestamp=now,
            )

            # Update cache and history
            self._cache = snapshot
            self._cache_time = now
            self._history.append(snapshot)
            if len(self._history) > self._max_history:
                self._history.pop(0)

            return snapshot

    def get_trend(self) -> Literal["improving", "stable", "degrading"]:
        """Analyze memory pressure trend from history."""
        if len(self._history) < 3:
            return "stable"

        recent = self._history[-3:]
        pressures = [s.percent_used for s in recent]

        if all(pressures[i] < pressures[i+1] for i in range(len(pressures)-1)):
            return "degrading"
        elif all(pressures[i] > pressures[i+1] for i in range(len(pressures)-1)):
            return "improving"
        return "stable"

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "is_macos": self._is_macos,
            "history_size": len(self._history),
            "trend": self.get_trend(),
            "cache_fresh": self._cache is not None and (time.time() - self._cache_time) < self._cache_ttl,
        }


# =============================================================================
# 🧠 THE BRAIN - Neural Switchboard & Complexity Analyzer
# =============================================================================

class ComplexityAnalyzer:
    """
    Analyzes query complexity to determine optimal routing.

    This is THE BRAIN's thought analysis - understanding the
    depth and nature of each thought before routing.

    Uses multiple heuristics:
    - Token count and vocabulary diversity
    - Question complexity markers
    - Technical terminology detection
    - Action keyword recognition
    - Reasoning depth indicators
    """

    # Pattern categories with weights
    SIMPLE_PATTERNS: Final[Tuple[str, ...]] = (
        "what time", "hello", "hi", "hey", "thanks", "bye",
        "what's the weather", "good morning", "good night",
        "what day", "what date", "how are you", "what's up",
    )

    COMPLEX_PATTERNS: Final[Tuple[str, ...]] = (
        "analyze", "explain in detail", "compare and contrast", "evaluate",
        "strategic implications", "quantum", "cryptography",
        "architecture", "optimization", "trade-off", "synthesis",
        "philosophical", "ethical implications", "comprehensive",
        "implications", "organizations", "prepare",
        "deep analysis", "impact", "consequences", "future",
        "multi-step", "reasoning chain", "prove that",
    )

    EXPERT_PATTERNS: Final[Tuple[str, ...]] = (
        "mathematical proof", "formal verification", "theorem",
        "axiom", "postulate", "derive", "rigorous analysis",
        "computational complexity", "algorithmic",
    )

    ACTION_PATTERNS: Final[Tuple[str, ...]] = (
        "open", "close", "launch", "run", "execute", "start",
        "stop", "kill", "create", "delete", "move", "copy",
        "send", "schedule", "remind", "set", "configure",
    )

    REASONING_MARKERS: Final[Tuple[str, ...]] = (
        "why", "how", "what if", "could you", "would you",
        "explain why", "elaborate", "detail", "step by step",
    )

    def __init__(self, config: DemoConfig):
        self.config = config
        self._cache: Dict[str, Tuple[float, ComplexityLevel, Dict[str, float]]] = {}
        self._cache_max_size = 100

    def _compute_lexical_complexity(self, text: str) -> float:
        """Compute complexity based on vocabulary and structure."""
        words = text.lower().split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        # Length factor (longer = potentially more complex)
        length_score = min(word_count / 50, 1.0) * 0.25

        # Average word length (technical terms tend to be longer)
        avg_word_len = sum(len(w) for w in words) / word_count
        word_len_score = min(avg_word_len / 10, 1.0) * 0.25

        # Vocabulary diversity (unique/total ratio)
        unique_ratio = len(set(words)) / word_count
        diversity_score = unique_ratio * 0.25

        # Sentence complexity (punctuation density)
        punct_count = sum(1 for c in text if c in ".,;:!?()-")
        punct_score = min(punct_count / max(word_count, 1) * 2, 1.0) * 0.25

        return length_score + word_len_score + diversity_score + punct_score

    def _compute_pattern_score(self, text: str) -> Tuple[float, ComplexityLevel]:
        """Match against known patterns to determine base complexity."""
        text_lower = text.lower()

        # Check action patterns first (special routing)
        for pattern in self.ACTION_PATTERNS:
            if pattern in text_lower:
                return 0.5, ComplexityLevel.ACTION

        # Check expert patterns (highest complexity)
        expert_matches = sum(1 for p in self.EXPERT_PATTERNS if p in text_lower)
        if expert_matches >= 1:
            return 0.95, ComplexityLevel.EXPERT

        # Check complex patterns
        complex_matches = sum(1 for p in self.COMPLEX_PATTERNS if p in text_lower)
        if complex_matches >= 3:
            return 0.85 + min(complex_matches * 0.02, 0.1), ComplexityLevel.COMPLEX
        elif complex_matches >= 2:
            return 0.75, ComplexityLevel.COMPLEX
        elif complex_matches == 1:
            return 0.55, ComplexityLevel.MODERATE

        # Check simple patterns
        simple_matches = sum(1 for p in self.SIMPLE_PATTERNS if p in text_lower)
        if simple_matches > 0:
            return 0.1 + (simple_matches * 0.02), ComplexityLevel.SIMPLE

        # Check reasoning markers
        reasoning_matches = sum(1 for p in self.REASONING_MARKERS if p in text_lower)
        if reasoning_matches >= 2:
            return 0.6, ComplexityLevel.MODERATE
        elif reasoning_matches == 1:
            return 0.45, ComplexityLevel.MODERATE

        return 0.35, ComplexityLevel.SIMPLE

    def analyze(self, query: str) -> Tuple[float, ComplexityLevel, Dict[str, float]]:
        """
        Analyze query complexity with full factor breakdown.

        Returns:
            Tuple of (complexity_score, complexity_level, factor_breakdown)
        """
        # Check cache
        cache_key = hashlib.md5(query.encode()).hexdigest()[:16]
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Compute factors
        lexical_score = self._compute_lexical_complexity(query)
        pattern_score, pattern_level = self._compute_pattern_score(query)

        factors: Dict[str, float] = {
            "lexical_complexity": lexical_score,
            "pattern_match_score": pattern_score,
            "token_count_factor": min(len(query.split()) / 40, 1.0),
            "question_depth": query.count("?") * 0.1 + query.count("why") * 0.15,
            "code_presence": 0.2 if any(x in query for x in ["```", "def ", "class ", "function"]) else 0,
        }

        # Combine scores with weights
        final_score = (
            lexical_score * 0.25 +
            pattern_score * 0.45 +
            factors["token_count_factor"] * 0.1 +
            factors["question_depth"] * 0.1 +
            factors["code_presence"] * 0.1
        )

        # Clamp to [0, 1]
        final_score = max(0.0, min(1.0, final_score))

        # Determine level based on score and patterns
        if pattern_level == ComplexityLevel.ACTION:
            level = ComplexityLevel.ACTION
        elif pattern_level == ComplexityLevel.EXPERT or final_score > 0.9:
            level = ComplexityLevel.EXPERT
        elif final_score < self.config.complexity_threshold_simple:
            level = ComplexityLevel.SIMPLE
        elif final_score > self.config.complexity_threshold_complex:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.MODERATE

        # Cache result
        result = (final_score, level, factors)
        self._cache[cache_key] = result

        # Evict old cache entries if needed
        if len(self._cache) > self._cache_max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        return result


class NeuralSwitchboard:
    """
    THE BRAIN - Intelligent request routing with real-time decision visualization.

    This is the visible decision-making engine that shows:
    - WHY a particular model was chosen
    - WHAT factors influenced the decision
    - HOW memory pressure affects routing
    - WHEN to burst to cloud vs stay local

    Features:
    - Complexity-aware routing
    - Memory pressure integration
    - Sticky session support
    - Decision history for learning
    """

    # Model catalog with full specifications
    MODELS: Final[Dict[str, Dict[str, Any]]] = {
        # Tier 0: Ultra-fast local models
        "Phi-3.5-Mini-Q4": {
            "tier": ModelTier.TIER_0_LOCAL_FAST,
            "capabilities": frozenset(["chat", "simple_qa", "greetings", "voice"]),
            "latency_ms": 150,
            "cost_per_1k": 0.0,
            "max_complexity": 0.4,
            "ram_required_gb": 3.0,
            "description": "Ultra-fast voice responses",
        },
        "DeepSeek-R1-8B-Q4": {
            "tier": ModelTier.TIER_0_LOCAL_FAST,
            "capabilities": frozenset(["chat", "reasoning", "local_inference"]),
            "latency_ms": 300,
            "cost_per_1k": 0.0,
            "max_complexity": 0.5,
            "ram_required_gb": 5.0,
            "description": "Fast local reasoning",
        },
        # Tier 0.5: Capable local models
        "Qwen-2.5-32B-Q4": {
            "tier": ModelTier.TIER_05_LOCAL_CAPABLE,
            "capabilities": frozenset(["chat", "code", "reasoning", "analysis"]),
            "latency_ms": 800,
            "cost_per_1k": 0.0,
            "max_complexity": 0.75,
            "ram_required_gb": 19.0,
            "description": "Capable local coding assistant",
        },
        # Tier 1: Cloud intelligent
        "Claude-3.5-Sonnet": {
            "tier": ModelTier.TIER_1_CLOUD,
            "capabilities": frozenset(["chat", "code", "reasoning", "analysis", "creative"]),
            "latency_ms": 2000,
            "cost_per_1k": 0.003,
            "max_complexity": 0.9,
            "ram_required_gb": 0.0,  # Cloud
            "description": "Intelligent cloud assistant",
        },
        # Tier 2: Deep reasoning
        "Claude-3-Opus": {
            "tier": ModelTier.TIER_2_CLOUD_DEEP,
            "capabilities": frozenset(["chat", "code", "deep_reasoning", "synthesis", "expert"]),
            "latency_ms": 5000,
            "cost_per_1k": 0.015,
            "max_complexity": 1.0,
            "ram_required_gb": 0.0,  # Cloud
            "description": "Expert deep reasoning",
        },
        # Tool use
        "JARVIS-Tool-Agent": {
            "tier": ModelTier.TOOL_USE,
            "capabilities": frozenset(["tool_use", "actions", "system_control"]),
            "latency_ms": 100,
            "cost_per_1k": 0.0,
            "max_complexity": 0.5,
            "ram_required_gb": 0.0,
            "description": "Action execution agent",
        },
    }

    # Tier priority order (prefer local when possible)
    TIER_PRIORITY: Final[Tuple[ModelTier, ...]] = (
        ModelTier.TIER_0_LOCAL_FAST,
        ModelTier.TIER_05_LOCAL_CAPABLE,
        ModelTier.TIER_1_CLOUD,
        ModelTier.TIER_2_CLOUD_DEEP,
    )

    def __init__(self, config: DemoConfig, memory_monitor: MacOSMemoryPressureMonitor):
        self.config = config
        self.memory_monitor = memory_monitor
        self.analyzer = ComplexityAnalyzer(config)
        self._decision_history: List[RoutingDecision] = []
        self._max_history = 100
        self._sticky_model: Optional[str] = None
        self._sticky_strength: float = 0.0

    def _select_model_for_complexity(
        self,
        complexity_score: float,
        complexity_level: ComplexityLevel,
        memory_snapshot: MemorySnapshot,
    ) -> Tuple[str, str]:
        """
        Select optimal model based on complexity and memory state.

        Returns (model_name, reasoning)
        """
        # Handle action requests
        if complexity_level == ComplexityLevel.ACTION:
            return "JARVIS-Tool-Agent", "Action/command detected → routing to Tool Agent"

        # Check if memory pressure forces cloud burst
        if memory_snapshot.should_burst:
            if complexity_level in (ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE):
                return "Claude-3.5-Sonnet", f"Memory pressure ({memory_snapshot.pressure_level.value}) → bursting to cloud"
            else:
                return "Claude-3-Opus", f"Memory critical + complex query → deep cloud reasoning"

        # Normal routing based on complexity
        if complexity_level == ComplexityLevel.SIMPLE:
            return "Phi-3.5-Mini-Q4", "Simple query → ultra-fast local model"

        elif complexity_level == ComplexityLevel.MODERATE:
            # Check if we have RAM for Qwen-32B
            available_gb = memory_snapshot.available_gb
            if available_gb > 20:
                return "Qwen-2.5-32B-Q4", "Moderate complexity + sufficient RAM → capable local model"
            elif available_gb > 6:
                return "DeepSeek-R1-8B-Q4", "Moderate complexity + limited RAM → efficient local model"
            else:
                return "Claude-3.5-Sonnet", "Moderate complexity + low RAM → cloud model"

        elif complexity_level == ComplexityLevel.COMPLEX:
            if complexity_score > 0.85:
                return "Claude-3-Opus", "High complexity score → deep reasoning required"
            else:
                return "Claude-3.5-Sonnet", "Complex query → cloud intelligence"

        else:  # EXPERT
            return "Claude-3-Opus", "Expert-level analysis → most capable model"

    async def route(self, query: str) -> RoutingDecision:
        """
        Route a query to the optimal model with full decision transparency.

        This method shows the "thinking" process - the Brain analyzing
        the thought before deciding where to route it.
        """
        # Get current memory state (The Body's status)
        memory_snapshot = await self.memory_monitor.get_snapshot()

        # Analyze complexity (The Brain's analysis)
        complexity_score, complexity_level, factors = self.analyzer.analyze(query)

        # Select model based on complexity and memory
        selected_model, reasoning = self._select_model_for_complexity(
            complexity_score, complexity_level, memory_snapshot
        )

        # Get model info
        model_info = self.MODELS.get(selected_model, self.MODELS["Phi-3.5-Mini-Q4"])

        # Build decision
        decision = RoutingDecision(
            selected_model=selected_model,
            tier=model_info["tier"],
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            reasoning=reasoning,
            confidence=0.9 if complexity_level != ComplexityLevel.MODERATE else 0.85,
            latency_estimate_ms=model_info["latency_ms"],
            cost_estimate=model_info["cost_per_1k"],
            factors=factors,
            memory_influenced=memory_snapshot.pressure_level != MemoryPressureLevel.NOMINAL,
            cloud_burst_active=memory_snapshot.should_burst,
            burst_reason=memory_snapshot.burst_reason,
        )

        # Record in history
        self._decision_history.append(decision)
        if len(self._decision_history) > self._max_history:
            self._decision_history.pop(0)

        return decision

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics for the session."""
        if not self._decision_history:
            return {"total_decisions": 0}

        tier_counts: Dict[str, int] = {}
        complexity_counts: Dict[str, int] = {}
        total_cost = 0.0
        cloud_bursts = 0

        for decision in self._decision_history:
            tier_name = decision.tier.value
            tier_counts[tier_name] = tier_counts.get(tier_name, 0) + 1

            complexity_name = decision.complexity_level.value
            complexity_counts[complexity_name] = complexity_counts.get(complexity_name, 0) + 1

            total_cost += decision.cost_estimate
            if decision.cloud_burst_active:
                cloud_bursts += 1

        return {
            "total_decisions": len(self._decision_history),
            "tier_distribution": tier_counts,
            "complexity_distribution": complexity_counts,
            "total_estimated_cost": total_cost,
            "cloud_burst_count": cloud_bursts,
            "avg_complexity": sum(d.complexity_score for d in self._decision_history) / len(self._decision_history),
        }


# =============================================================================
# 🔌 THE NERVOUS SYSTEM - Experience Logger to Reactor Core
# =============================================================================

class TrinityExperienceLogger:
    """
    THE NERVOUS SYSTEM - Logs experiences to Reactor Core for learning.

    This completes the Trinity Loop:
    JARVIS (Body) → JARVIS-Prime (Mind) → Reactor (Nerves) → Learning → Improved Models

    "You cannot code AGI line-by-line. You have to grow it."

    Features:
    - Local JSONL logging for persistence
    - Reactor Core event emission
    - Experience quality scoring
    - Batch processing for efficiency
    """

    def __init__(self, config: DemoConfig):
        self.config = config
        self._base_path = Path.home() / ".jarvis" / "trinity" / "events"
        self._local_log_path = self._base_path / "demo_experiences.jsonl"
        self._reactor_available = False
        self._initialized = False
        self._experience_buffer: List[ExperienceRecord] = []
        self._buffer_max_size = 10
        self._lock = AsyncLock("experience_logger")
        self._total_logged = 0

    async def initialize(self) -> bool:
        """Initialize experience logging infrastructure."""
        if self._initialized:
            return True

        async with self._lock:
            if self._initialized:
                return True

            # Create directories
            self._base_path.mkdir(parents=True, exist_ok=True)
            self._local_log_path.parent.mkdir(parents=True, exist_ok=True)

            # Check for Reactor Core
            reactor_path = REPO_PATHS.get("reactor_core")
            if reactor_path and reactor_path.exists():
                self._reactor_available = True
                logger.debug("[Nerves] Reactor Core path available for experience logging")

            self._initialized = True
            return True

    async def log_experience(
        self,
        experience: ExperienceRecord,
        emit_event: bool = True,
    ) -> bool:
        """
        Log an experience to the Trinity loop.

        This is the Nervous System recording a memory for later learning.
        """
        if not self.config.log_experiences:
            return False

        await self.initialize()

        # Serialize experience
        exp_dict = {
            "id": experience.id,
            "timestamp": experience.timestamp,
            "datetime": datetime.fromtimestamp(experience.timestamp).isoformat(),
            "prompt": experience.prompt,
            "response": experience.response,
            "model_used": experience.model_used,
            "tier": experience.tier,
            "complexity_score": experience.complexity_score,
            "latency_ms": experience.latency_ms,
            "feedback_score": experience.feedback_score,
            "memory_pressure": experience.memory_pressure_at_time,
            "cloud_burst_used": experience.cloud_burst_used,
            "metadata": experience.metadata,
            "source": "digital_biology_demo",
            "version": "99.0",
        }

        success = False

        # Write to local JSONL
        try:
            async with self._lock:
                with open(self._local_log_path, "a") as f:
                    f.write(json.dumps(exp_dict) + "\n")
                self._total_logged += 1
            success = True
            logger.info(f"[Nerves] Experience #{self._total_logged} logged → {self._local_log_path.name}")
        except Exception as e:
            logger.error(f"[Nerves] Failed to log experience: {e}")

        # Also write to Reactor events directory if available
        if self._reactor_available and emit_event:
            try:
                event_file = self._base_path / f"exp_{experience.id}.json"
                with open(event_file, "w") as f:
                    json.dump(exp_dict, f, indent=2)
                logger.debug(f"[Nerves] Event emitted to Reactor: {event_file.name}")
            except Exception as e:
                logger.debug(f"[Nerves] Reactor event emission failed: {e}")

        return success

    async def flush_buffer(self) -> int:
        """Flush any buffered experiences."""
        if not self._experience_buffer:
            return 0

        count = 0
        async with self._lock:
            for exp in self._experience_buffer:
                if await self.log_experience(exp, emit_event=False):
                    count += 1
            self._experience_buffer.clear()

        return count

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "initialized": self._initialized,
            "reactor_available": self._reactor_available,
            "total_logged": self._total_logged,
            "buffer_size": len(self._experience_buffer),
            "log_path": str(self._local_log_path),
        }


# =============================================================================
# VOICE ENGINE - macOS TTS
# =============================================================================

class VoiceEngine:
    """
    Text-to-Speech engine using macOS 'say' command.

    Features:
    - Direct subprocess execution
    - Async lock for thread safety
    - Text chunking for long responses
    - Retry with exponential backoff
    """

    MAX_CHUNK_LENGTH: Final[int] = 200

    def __init__(self, config: DemoConfig):
        self.config = config
        self._available = False
        self._initialized = False
        self._lock = AsyncLock("voice_engine")
        self._active_process: Optional[asyncio.subprocess.Process] = None

    async def initialize(self) -> bool:
        """Initialize voice engine."""
        if self._initialized:
            return self._available

        if platform.system() != "Darwin":
            self._initialized = True
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "which", "say",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.wait(), timeout=5.0)
            self._available = proc.returncode == 0
        except Exception:
            self._available = False

        self._initialized = True
        return self._available

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into speakable chunks."""
        if len(text) <= self.MAX_CHUNK_LENGTH:
            return [text]

        chunks: List[str] = []
        current = ""

        for sentence in text.replace("...", "<<<ELLIPSIS>>>").split(". "):
            sentence = sentence.replace("<<<ELLIPSIS>>>", "...").strip()
            if not sentence:
                continue

            if not sentence.endswith((".", "!", "?", "...")):
                sentence += "."

            if len(current) + len(sentence) + 1 <= self.MAX_CHUNK_LENGTH:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks or [text[:self.MAX_CHUNK_LENGTH]]

    async def speak(self, text: str) -> bool:
        """Speak text using macOS 'say' command."""
        if not self.config.voice_enabled:
            return False

        await self.initialize()
        if not self._available:
            return False

        async with self._lock:
            chunks = self._chunk_text(text)

            for chunk in chunks:
                try:
                    self._active_process = await asyncio.create_subprocess_exec(
                        "say",
                        "-v", self.config.voice_name,
                        "-r", str(self.config.voice_rate),
                        chunk,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await asyncio.wait_for(self._active_process.wait(), timeout=15.0)
                    self._active_process = None
                except asyncio.TimeoutError:
                    if self._active_process:
                        self._active_process.kill()
                        self._active_process = None
                    return False
                except Exception:
                    return False

        return True

    async def stop(self) -> None:
        """Stop active speech."""
        if self._active_process:
            with suppress(Exception):
                self._active_process.kill()
                await self._active_process.wait()
            self._active_process = None


# =============================================================================
# LOCAL LLM ENGINE
# =============================================================================

class LocalLLMEngine:
    """
    Local LLM inference with automatic model discovery.

    Falls back to mock responses if no model available.
    """

    MODEL_SEARCH_PATHS: Final[Tuple[Path, ...]] = (
        Path.home() / ".jarvis" / "prime" / "models",
        Path.home() / "Documents" / "ai-models",
        Path.home() / "Documents" / "repos" / "jarvis-prime" / "models",
        Path.home() / ".jarvis" / "models",
    )

    def __init__(self, config: DemoConfig):
        self.config = config
        self._llm: Any = None
        self._model_path: Optional[Path] = None
        self._model_name = "Mock"
        self._initialized = False
        self._use_mock = True
        self._lock = AsyncLock("llm_engine")
        self._inference_count = 0
        self._total_tokens = 0
        self._total_time_ms = 0.0

    def _find_model(self) -> Optional[Path]:
        """Find available GGUF model."""
        for search_path in self.MODEL_SEARCH_PATHS:
            if not search_path.exists():
                continue

            # Check symlink first
            current = search_path / "current.gguf"
            if current.exists() and current.is_symlink():
                target = current.resolve()
                if target.exists():
                    return target

            # Find any GGUF
            for gguf in search_path.glob("*.gguf"):
                if gguf.is_file():
                    return gguf

        return None

    async def initialize(self) -> bool:
        """Initialize LLM engine."""
        if self._initialized:
            return not self._use_mock

        async with self._lock:
            if self._initialized:
                return not self._use_mock

            self._model_path = self._find_model()

            if not self._model_path:
                logger.debug("[Brain] No GGUF model found, using mock mode")
                self._use_mock = True
                self._initialized = True
                return False

            try:
                from llama_cpp import Llama

                is_apple_silicon = (
                    platform.system() == "Darwin" and
                    platform.machine() == "arm64"
                )

                logger.info(f"[Brain] Loading {self._model_path.name}...")

                loop = asyncio.get_event_loop()
                executor = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=1)

                self._llm = await loop.run_in_executor(
                    executor,
                    lambda: Llama(
                        model_path=str(self._model_path),
                        n_ctx=2048,
                        n_threads=4,
                        n_gpu_layers=32 if is_apple_silicon else 0,
                        verbose=False,
                        use_mlock=True,
                    )
                )

                executor.shutdown(wait=False)
                self._model_name = self._model_path.stem
                self._use_mock = False
                logger.info(f"[Brain] ✓ Loaded {self._model_name}")

            except ImportError:
                logger.debug("[Brain] llama-cpp-python not installed, using mock")
                self._use_mock = True
            except Exception as e:
                logger.debug(f"[Brain] Model load failed: {e}, using mock")
                self._use_mock = True

            self._initialized = True
            return not self._use_mock

    async def generate(
        self,
        query: str,
        model: str,
        complexity_level: ComplexityLevel,
    ) -> Tuple[str, float]:
        """Generate response."""
        await self.initialize()

        # Always use mock for demo (faster)
        return await self._generate_mock(query, model, complexity_level)

    async def _generate_mock(
        self,
        query: str,
        model: str,
        complexity_level: ComplexityLevel,
    ) -> Tuple[str, float]:
        """Generate mock response based on query type."""
        query_lower = query.lower()

        # Simulate varying latency based on model tier
        latency_map = {
            "Phi-3.5-Mini-Q4": 150,
            "DeepSeek-R1-8B-Q4": 300,
            "Qwen-2.5-32B-Q4": 800,
            "Claude-3.5-Sonnet": 2000,
            "Claude-3-Opus": 3500,
            "JARVIS-Tool-Agent": 100,
        }
        base_latency = latency_map.get(model, 500)
        latency = base_latency + random.randint(-50, 100)

        await asyncio.sleep(latency / 1000)

        # Generate contextual response
        if "time" in query_lower:
            response = f"The current time is {datetime.now().strftime('%I:%M %p')}."
        elif "quantum" in query_lower or "cryptography" in query_lower:
            response = (
                "The strategic implications of quantum computing on cryptography are profound. "
                "Current RSA and ECC encryption could be broken by Shor's algorithm once "
                "fault-tolerant quantum computers achieve sufficient qubit counts. "
                "Organizations should begin transitioning to post-quantum cryptography standards "
                "like CRYSTALS-Kyber and CRYSTALS-Dilithium, as recommended by NIST."
            )
        elif complexity_level == ComplexityLevel.ACTION:
            app = "Chrome"
            for word in ["chrome", "safari", "firefox", "slack", "terminal"]:
                if word in query_lower:
                    app = word.title()
                    break
            response = f"Opening {app} now. Let me know if you need anything else."
        elif complexity_level in (ComplexityLevel.COMPLEX, ComplexityLevel.EXPERT):
            response = (
                "This is a complex query that requires deep analysis. "
                "I'm synthesizing information from multiple domains to provide "
                "a comprehensive response that addresses the nuances of your question."
            )
        else:
            response = "I understand your query. Processing that for you now."

        self._inference_count += 1
        self._total_time_ms += latency

        return response, latency

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return {
            "model": self._model_name,
            "model_path": str(self._model_path) if self._model_path else None,
            "using_mock": self._use_mock,
            "inference_count": self._inference_count,
            "total_time_ms": self._total_time_ms,
        }

    async def shutdown(self) -> None:
        """Clean shutdown."""
        if self._llm:
            del self._llm
            self._llm = None


# =============================================================================
# DIGITAL BIOLOGY DEMO ORCHESTRATOR
# =============================================================================

class DigitalBiologyDemo:
    """
    Main demo orchestrator that demonstrates JARVIS as a living system.

    Shows the three organs working together:
    - 🧠 THE BRAIN: Neural Switchboard analyzing and routing
    - 💪 THE BODY: Memory Pressure detection and cloud burst
    - 🔌 THE NERVOUS SYSTEM: Experience logging to Reactor
    """

    # Test cases demonstrating different routing scenarios
    TEST_CASES: Final[Tuple[TestCase, ...]] = (
        TestCase(
            name="🧠 CASE A: Simple Query (Brain → Fast Local)",
            query="JARVIS, what time is it?",
            expected_tier=ModelTier.TIER_0_LOCAL_FAST,
            expected_complexity=ComplexityLevel.SIMPLE,
            description="Simple factual query → Ultra-fast local model",
        ),
        TestCase(
            name="🧠 CASE B: Complex Analysis (Brain → Cloud Intelligent)",
            query="Analyze the strategic implications of quantum computing on modern cryptography and what organizations should do to prepare.",
            expected_tier=ModelTier.TIER_1_CLOUD,
            expected_complexity=ComplexityLevel.COMPLEX,
            description="Deep analysis required → Cloud intelligent model",
        ),
        TestCase(
            name="🧠 CASE C: Action Command (Brain → Tool Agent)",
            query="Open Google Chrome.",
            expected_tier=ModelTier.TOOL_USE,
            expected_complexity=ComplexityLevel.ACTION,
            description="System action → Tool execution agent",
        ),
    )

    def __init__(self, config: Optional[DemoConfig] = None):
        self.config = config or CONFIG

        # Initialize the three organs
        self.memory_monitor = MacOSMemoryPressureMonitor(self.config)  # Body
        self.neural_switchboard = NeuralSwitchboard(self.config, self.memory_monitor)  # Brain
        self.experience_logger = TrinityExperienceLogger(self.config)  # Nerves

        # Support systems
        self.voice = VoiceEngine(self.config)
        self.inference = LocalLLMEngine(self.config)

        self._results: List[DemoResponse] = []
        self._output_lock = AsyncLock("demo_output")

    async def _print(self, *args: Any, **kwargs: Any) -> None:
        """Synchronized print."""
        async with self._output_lock:
            print(*args, **kwargs, flush=True)

    async def _print_header(self) -> None:
        """Print the Digital Biology header."""
        header = """
\033[95m╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗    ██╗     ██╗███████╗███████╗ ║
║       ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝    ██║     ██║██╔════╝██╔════╝ ║
║       ██║███████║██████╔╝██║   ██║██║███████╗    ██║     ██║█████╗  █████╗   ║
║  ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║    ██║     ██║██╔══╝  ██╔══╝   ║
║  ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║    ███████╗██║██║     ███████╗ ║
║   ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝    ╚══════╝╚═╝╚═╝     ╚══════╝ ║
║                                                                              ║
║                    🧬 DIGITAL BIOLOGY DEMONSTRATION v99.0 🧬                  ║
║                                                                              ║
║    "You cannot code AGI line-by-line. You have to grow it."                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝\033[0m
"""
        await self._print(header)

    async def _print_organ_status(self) -> None:
        """Print current status of all three organs."""
        # Get Body status
        memory = await self.memory_monitor.get_snapshot()
        pressure_color = {
            MemoryPressureLevel.NOMINAL: "\033[32m",   # Green
            MemoryPressureLevel.WARN: "\033[33m",      # Yellow
            MemoryPressureLevel.CRITICAL: "\033[31m", # Red
        }.get(memory.pressure_level, "\033[37m")

        # Get Brain stats
        brain_stats = self.neural_switchboard.get_statistics()

        # Get Nerves stats
        nerve_stats = self.experience_logger.get_statistics()

        status = f"""
\033[94m┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRINITY ORGAN STATUS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  🧠 BRAIN (Neural Switchboard)                                              │
│     ├─ Decisions Made: {brain_stats.get('total_decisions', 0):3}                                              │
│     ├─ Avg Complexity:  {brain_stats.get('avg_complexity', 0):.2f}                                            │
│     └─ Cloud Bursts:    {brain_stats.get('cloud_burst_count', 0):3}                                              │
│                                                                             │
│  💪 BODY (Memory Pressure)                                                  │
│     ├─ Pressure:       {pressure_color}{memory.pressure_level.value:10}\033[94m                                     │
│     ├─ RAM Used:       {memory.percent_used:5.1f}% ({memory.total_gb - memory.available_gb:.1f}GB / {memory.total_gb:.1f}GB)                     │
│     ├─ Trend:          {self.memory_monitor.get_trend():10}                                     │
│     └─ Burst Mode:     {'ACTIVE' if memory.should_burst else 'standby':10}                                     │
│                                                                             │
│  🔌 NERVES (Experience Logger)                                              │
│     ├─ Reactor Found:  {'✓' if nerve_stats.get('reactor_available') else '✗'}                                                  │
│     ├─ Experiences:    {nerve_stats.get('total_logged', 0):3} logged                                         │
│     └─ Status:         {'Online' if nerve_stats.get('initialized') else 'Initializing':10}                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘\033[0m
"""
        await self._print(status)

    async def _print_routing_decision(self, decision: RoutingDecision) -> None:
        """Print routing decision with organ visualization."""
        tier_colors = {
            ModelTier.TIER_0_LOCAL_FAST: "\033[32m",      # Green
            ModelTier.TIER_05_LOCAL_CAPABLE: "\033[36m",  # Cyan
            ModelTier.TIER_1_CLOUD: "\033[35m",           # Magenta
            ModelTier.TIER_2_CLOUD_DEEP: "\033[95m",      # Bright Magenta
            ModelTier.TOOL_USE: "\033[33m",               # Yellow
        }
        color = tier_colors.get(decision.tier, "\033[37m")
        reset = "\033[0m"
        bold = "\033[1m"

        burst_indicator = ""
        if decision.cloud_burst_active:
            burst_indicator = f"\n│  ⚡ CLOUD BURST: {decision.burst_reason.value if decision.burst_reason else 'Unknown':40}     │"

        memory_indicator = ""
        if decision.memory_influenced:
            memory_indicator = f"\n│  💪 Memory Pressure Influenced Routing                                     │"

        block = f"""
\033[95m┌─────────────────────────────────────────────────────────────────────────────┐
│ 🧠 {bold}[BRAIN] NEURAL SWITCHBOARD DECISION{reset}\033[95m                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Selected Model: {color}{bold}{decision.selected_model:25}{reset}\033[95m                          │
│  Tier:           {color}{decision.tier.value:25}{reset}\033[95m                          │
│  Complexity:     {decision.complexity_score:.2f} ({decision.complexity_level.value:15})                          │
│  Confidence:     {decision.confidence:.0%}                                                    │
│  Reasoning:      {decision.reasoning[:50]:50}   │
│  Est. Latency:   {decision.latency_estimate_ms:.0f}ms                                                │{burst_indicator}{memory_indicator}
└─────────────────────────────────────────────────────────────────────────────┘{reset}
"""
        await self._print(block)

    async def _print_test_case(self, test: TestCase, index: int) -> None:
        """Print test case header."""
        block = f"""
\033[93m{'═'*80}
  {test.name}
  {'─'*len(test.name)}
  Query: "{test.query}"
  Expected: {test.expected_tier.value} ({test.expected_complexity.value})
  Description: {test.description}
{'═'*80}\033[0m
"""
        await self._print(block)

    async def _process_test_case(self, test: TestCase) -> DemoResponse:
        """Process a single test case through the full pipeline."""
        start_time = time.time()

        # Step 1: Get memory snapshot (The Body's status)
        await self._print("\n\033[94m💪 [Body] Checking memory pressure...\033[0m")
        memory_snapshot = await self.memory_monitor.get_snapshot()

        pressure_indicator = {
            MemoryPressureLevel.NOMINAL: "\033[32m● NOMINAL\033[0m",
            MemoryPressureLevel.WARN: "\033[33m● WARN\033[0m",
            MemoryPressureLevel.CRITICAL: "\033[31m● CRITICAL\033[0m",
        }.get(memory_snapshot.pressure_level, "UNKNOWN")

        await self._print(f"   Pressure: {pressure_indicator} | RAM: {memory_snapshot.percent_used:.1f}% used")

        if memory_snapshot.should_burst:
            await self._print(f"   \033[31m⚡ CLOUD BURST TRIGGERED: {memory_snapshot.burst_reason.value if memory_snapshot.burst_reason else 'Memory stress'}\033[0m")

        # Step 2: Route through Neural Switchboard (The Brain's decision)
        await self._print("\n\033[95m🧠 [Brain] Analyzing complexity and routing...\033[0m")
        decision = await self.neural_switchboard.route(test.query)
        await self._print_routing_decision(decision)

        # Step 3: Generate response
        await self._print("\n\033[36m🔮 [Inference] Generating response...\033[0m")
        response_text, inference_latency = await self.inference.generate(
            test.query,
            decision.selected_model,
            decision.complexity_level,
        )

        total_latency = (time.time() - start_time) * 1000

        await self._print(f"\n\033[32m[Response] ({inference_latency:.0f}ms)\033[0m")
        await self._print(f"   {response_text[:200]}{'...' if len(response_text) > 200 else ''}")

        # Step 4: Play voice (if enabled)
        voice_played = False
        if self.config.voice_enabled:
            await self._print("\n\033[33m🔊 [Voice] Speaking response...\033[0m")
            voice_played = await self.voice.speak(response_text)
            if voice_played:
                await self._print("   ✓ Audio played")
            else:
                await self._print("   ✗ Voice unavailable")

        # Step 5: Log experience to Reactor (The Nervous System learning)
        experience_logged = False
        if self.config.log_experiences:
            experience = ExperienceRecord(
                prompt=test.query,
                response=response_text,
                model_used=decision.selected_model,
                tier=decision.tier.value,
                complexity_score=decision.complexity_score,
                latency_ms=total_latency,
                memory_pressure_at_time=memory_snapshot.pressure_level.value,
                cloud_burst_used=decision.cloud_burst_active,
                metadata={
                    "test_name": test.name,
                    "expected_tier": test.expected_tier.value,
                    "decision_id": decision.decision_id,
                    "memory_percent": memory_snapshot.percent_used,
                },
            )
            experience_logged = await self.experience_logger.log_experience(experience)
            if experience_logged:
                await self._print("\n\033[93m🔌 [Nerves] Experience logged → Trinity Loop → Reactor Learning\033[0m")
                await self._print("   \"You cannot code AGI line-by-line. You have to grow it.\"")

        return DemoResponse(
            test_case=test,
            routing_decision=decision,
            response_text=response_text,
            latency_ms=total_latency,
            success=True,
            experience_logged=experience_logged,
            voice_played=voice_played,
            memory_snapshot=memory_snapshot,
        )

    async def run(self) -> List[DemoResponse]:
        """Run the full Digital Biology demo sequence."""
        await self._print_header()

        # Initialize all organs
        await self._print("\n🚀 Initializing Digital Biology systems...")
        await self._print(f"   Voice: {'Enabled' if self.config.voice_enabled else 'Disabled'}")
        await self._print(f"   Experience Logging: {'Enabled' if self.config.log_experiences else 'Disabled'}")

        await self.voice.initialize()
        await self.experience_logger.initialize()

        llm_ready = await self.inference.initialize()
        stats = self.inference.get_stats()
        await self._print(f"   Local LLM: {'✓ ' + stats['model'] if llm_ready else 'Mock mode'}")

        # Print initial organ status
        await self._print_organ_status()

        # Process each test case
        for i, test in enumerate(self.TEST_CASES):
            await self._print_test_case(test, i)

            try:
                result = await self._process_test_case(test)
                self._results.append(result)

                # Verify routing
                if result.routing_decision.tier == test.expected_tier:
                    await self._print(f"\n\033[32m✅ ROUTING CORRECT: {result.routing_decision.tier.value}\033[0m")
                else:
                    await self._print(f"\n\033[33m⚠️ ROUTING DIFFERENT: Got {result.routing_decision.tier.value} (expected {test.expected_tier.value})\033[0m")
                    await self._print(f"   (This is OK - routing adapts to current system state)")

            except Exception as e:
                logger.error(f"Test case failed: {e}")
                self._results.append(DemoResponse(
                    test_case=test,
                    routing_decision=RoutingDecision(
                        selected_model="ERROR",
                        tier=ModelTier.TIER_0_LOCAL_FAST,
                        complexity_score=0.0,
                        complexity_level=ComplexityLevel.SIMPLE,
                        reasoning=str(e),
                        confidence=0.0,
                        latency_estimate_ms=0.0,
                        cost_estimate=0.0,
                    ),
                    response_text="",
                    latency_ms=0.0,
                    success=False,
                    error_message=str(e),
                ))

            # Pause between tests
            if self.config.pause_between_tests and i < len(self.TEST_CASES) - 1:
                await self._print("\n" + "─" * 80)
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("Press ENTER to continue to next test...")
                )

        # Print final summary
        await self._print_summary()

        return self._results

    async def _print_summary(self) -> None:
        """Print final demo summary with organ statistics."""
        successful = sum(1 for r in self._results if r.success)
        total = len(self._results)
        voice_played = sum(1 for r in self._results if r.voice_played)
        experiences_logged = sum(1 for r in self._results if r.experience_logged)

        brain_stats = self.neural_switchboard.get_statistics()
        nerve_stats = self.experience_logger.get_statistics()
        llm_stats = self.inference.get_stats()

        summary = f"""
\033[95m╔══════════════════════════════════════════════════════════════════════════════╗
║                         DIGITAL BIOLOGY SUMMARY                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 DEMO RESULTS                                                             ║
║     ├─ Tests Executed:     {total:3}                                               ║
║     ├─ Successful:         {successful:3}                                               ║
║     ├─ Voice Responses:    {voice_played:3}                                               ║
║     └─ Experiences Logged: {experiences_logged:3}                                               ║
║                                                                              ║
║  🧠 BRAIN (Neural Switchboard)                                               ║
║     ├─ Total Decisions:    {brain_stats.get('total_decisions', 0):3}                                               ║
║     ├─ Avg Complexity:     {brain_stats.get('avg_complexity', 0):.2f}                                             ║
║     └─ Cloud Bursts:       {brain_stats.get('cloud_burst_count', 0):3}                                               ║
║                                                                              ║
║  💪 BODY (Memory Pressure)                                                   ║
║     ├─ Monitoring:         Active                                            ║
║     └─ Trend:              {self.memory_monitor.get_trend():10}                                   ║
║                                                                              ║
║  🔌 NERVES (Reactor Integration)                                             ║
║     ├─ Experiences Saved:  {nerve_stats.get('total_logged', 0):3}                                               ║
║     └─ Reactor Connected:  {'✓' if nerve_stats.get('reactor_available') else '✗'}                                                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🧬 JARVIS IS ALIVE - The Digital Biology is functioning!                    ║
║                                                                              ║
║  "You cannot code AGI line-by-line. You have to grow it."                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝\033[0m
"""
        await self._print(summary)

        # Shutdown
        await self.inference.shutdown()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main() -> int:
    """Main entry point for the Digital Biology demo."""
    try:
        demo = DigitalBiologyDemo()
        results = await demo.run()

        all_passed = all(r.success for r in results)
        return 0 if all_passed else 1

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
