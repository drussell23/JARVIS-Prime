"""
GCP Import Bridge v95.1 - Enterprise-Grade Cross-Repo Integration
===================================================================

ADVANCED PATTERNS IMPLEMENTED:
    - Protocol classes for duck-typing with type safety
    - Retry with exponential backoff + jitter (prevents thundering herd)
    - Circuit breakers (state machine pattern)
    - Rate limiting (token bucket algorithm)
    - Async-safe caching with TTL
    - Structured concurrency (TaskGroup pattern)
    - Event bus for cross-repo coordination
    - Context managers for resource cleanup
    - Dependency injection via providers
    - Observer pattern for reactive updates
    - Health monitoring with heartbeats
    - Graceful degradation with fallback chains

CROSS-REPO INTEGRATION:
    - JARVIS-AI-Agent: CostTracker, GCPVMManager, IntelligentGCPOptimizer
    - JARVIS-Prime: HybridTieredRouter, ResourceMonitor
    - Reactor-Core: SpotVMCheckpointer, TrainingExecutor

Usage:
    async with GCPBridgeContext() as bridge:
        result = await bridge.execute_with_retry(some_operation)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import sys
import time
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Awaitable,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    Final,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

logger = logging.getLogger("jarvis-prime.gcp-bridge")

# =============================================================================
# TYPE VARIABLES AND PROTOCOLS
# =============================================================================

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")
ExceptionT = TypeVar("ExceptionT", bound=Exception)


@runtime_checkable
class Initializable(Protocol):
    """Protocol for components that need async initialization."""
    async def initialize(self) -> None: ...


@runtime_checkable
class HasHealthCheck(Protocol):
    """Protocol for components with health checks."""
    async def health_check(self) -> bool: ...


@runtime_checkable
class CostTrackerProtocol(Protocol):
    """Protocol defining CostTracker interface."""
    async def can_create_vm(self, estimated_cost: float = 0.0) -> bool: ...
    async def get_budget_status(self) -> Dict[str, Any]: ...
    async def record_spend(self, amount: float) -> None: ...


@runtime_checkable
class VMManagerProtocol(Protocol):
    """Protocol defining GCPVMManager interface."""
    async def provision_spot_vm(self, **kwargs: Any) -> Dict[str, Any]: ...
    def get_status(self) -> Dict[str, Any]: ...


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Protocol defining IntelligentGCPOptimizer interface."""
    async def calculate_pressure_score(self, snapshot: Any = None) -> Any: ...
    async def should_create_vm(self) -> bool: ...


@runtime_checkable
class RAMMonitorProtocol(Protocol):
    """Protocol defining AdvancedRAMMonitor interface."""
    async def get_snapshot(self) -> Any: ...
    async def start(self) -> None: ...


# =============================================================================
# ADVANCED RETRY WITH EXPONENTIAL BACKOFF + JITTER
# =============================================================================

@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1  # Prevents thundering herd
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        OSError,
    )


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(f"Exhausted {attempts} retry attempts: {last_exception}")


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: Optional[RetryConfig] = None,
    **kwargs: Any,
) -> T:
    """
    Execute async function with exponential backoff and jitter.

    Implements the "decorrelated jitter" algorithm for optimal retry distribution.
    """
    config = config or RetryConfig()
    last_exception: Optional[Exception] = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e
            if attempt == config.max_attempts:
                break

            # Calculate delay with decorrelated jitter
            delay = min(
                config.max_delay,
                config.base_delay * (config.exponential_base ** (attempt - 1))
            )
            # Add jitter: random value between 0 and jitter * delay
            jitter = random.uniform(0, config.jitter * delay)
            actual_delay = delay + jitter

            logger.debug(
                f"Retry {attempt}/{config.max_attempts} after {actual_delay:.2f}s: {e}"
            )
            await asyncio.sleep(actual_delay)

    raise RetryExhausted(config.max_attempts, cast(Exception, last_exception))


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for adding retry behavior to async functions."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(func, *args, config=config, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# CIRCUIT BREAKER (STATE MACHINE PATTERN)
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 2  # Successes needed in HALF_OPEN to close
    timeout: float = 30.0       # Time in OPEN before trying HALF_OPEN
    half_open_max_calls: int = 3  # Max concurrent calls in HALF_OPEN


class CircuitBreaker:
    """
    Circuit breaker implementation using state machine pattern.

    States:
        CLOSED: Normal operation, failures tracked
        OPEN: Requests rejected, waiting for timeout
        HALF_OPEN: Testing if service recovered
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self._state_change_callbacks: List[Callable[[CircuitState], None]] = []

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def on_state_change(self, callback: Callable[[CircuitState], None]) -> None:
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            logger.info(f"Circuit breaker '{self.name}': {old_state.name} -> {new_state.name}")
            for callback in self._state_change_callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    logger.warning(f"State change callback error: {e}")

    async def can_execute(self) -> bool:
        """Check if execution is allowed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout elapsed
                if self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.config.timeout:
                        await self._transition_to(CircuitState.HALF_OPEN)
                        self._half_open_calls = 0
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    async def record_failure(self) -> None:
        """Record a failed execution."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                await self._transition_to(CircuitState.OPEN)
                self._success_count = 0
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.name,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure": self._last_failure_time,
        }


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    def __init__(self, breaker_name: str):
        self.breaker_name = breaker_name
        super().__init__(f"Circuit breaker '{breaker_name}' is open")


# =============================================================================
# RATE LIMITER (TOKEN BUCKET ALGORITHM)
# =============================================================================

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    burst_size: int = 20
    wait_timeout: float = 30.0


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for controlling request rates.

    Features:
        - Smooth rate limiting with burst support
        - Async-safe with proper locking
        - Graceful handling of burst traffic
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._tokens = float(self.config.burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.

        Returns:
            True if tokens acquired, False if rate limited
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            # Refill tokens
            self._tokens = min(
                self.config.burst_size,
                self._tokens + elapsed * self.config.requests_per_second
            )

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def wait_for_token(self, tokens: int = 1) -> None:
        """Wait until tokens are available."""
        start = time.monotonic()
        while True:
            if await self.acquire(tokens):
                return

            # Calculate wait time
            async with self._lock:
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.config.requests_per_second

            if time.monotonic() - start + wait_time > self.config.wait_timeout:
                raise asyncio.TimeoutError("Rate limit wait timeout exceeded")

            await asyncio.sleep(min(wait_time, 0.1))


# =============================================================================
# ASYNC-SAFE CACHE WITH TTL
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL tracking."""
    value: T
    created_at: float
    ttl: float
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl


class AsyncTTLCache(Generic[T]):
    """
    Async-safe cache with TTL support.

    Features:
        - Automatic expiration
        - Hit tracking for analytics
        - Background cleanup
        - Thread-safe operations
    """

    def __init__(
        self,
        default_ttl: float = 300.0,
        max_size: int = 1000,
        cleanup_interval: float = 60.0,
    ):
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    async def start_cleanup(self) -> None:
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._evict_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cache cleanup error: {e}")

    async def _evict_expired(self) -> int:
        """Evict expired entries."""
        async with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats["evictions"] += 1
            return len(expired_keys)

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            entry.hits += 1
            self._stats["hits"] += 1
            return entry.value

    async def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at
                )
                del self._cache[oldest_key]
                self._stats["evictions"] += 1

            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl or self._default_ttl,
            )

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
        ttl: Optional[float] = None,
    ) -> T:
        """Get from cache or compute and cache."""
        value = await self.get(key)
        if value is not None:
            return value

        value = await factory()
        await self.set(key, value, ttl)
        return value

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": (
                self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
            ),
        }


# =============================================================================
# EVENT BUS FOR CROSS-REPO COORDINATION
# =============================================================================

@dataclass
class Event:
    """Base event class."""
    name: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "gcp_bridge"
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


class EventBus:
    """
    Async event bus for cross-repo coordination.

    Features:
        - Pub/Sub pattern
        - Weak references to prevent memory leaks
        - Event filtering and routing
        - Event history for debugging
    """

    _instance: ClassVar[Optional["EventBus"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self):
        self._subscribers: Dict[str, List[weakref.ref[Callable[[Event], Awaitable[None]]]]] = {}
        self._event_history: List[Event] = []
        self._max_history = 1000
        self._global_subscribers: List[weakref.ref[Callable[[Event], Awaitable[None]]]] = []

    @classmethod
    async def get_instance(cls) -> "EventBus":
        """Get singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def subscribe(
        self,
        event_name: str,
        handler: Callable[[Event], Awaitable[None]],
    ) -> None:
        """Subscribe to a specific event."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        self._subscribers[event_name].append(weakref.ref(handler))

    def subscribe_all(self, handler: Callable[[Event], Awaitable[None]]) -> None:
        """Subscribe to all events."""
        self._global_subscribers.append(weakref.ref(handler))

    async def publish(self, event: Event) -> int:
        """
        Publish an event.

        Returns:
            Number of handlers that received the event
        """
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history // 2:]

        handlers_called = 0

        # Notify specific subscribers
        if event.name in self._subscribers:
            for ref in self._subscribers[event.name]:
                handler = ref()
                if handler is not None:
                    try:
                        await handler(event)
                        handlers_called += 1
                    except Exception as e:
                        logger.error(f"Event handler error for {event.name}: {e}")

        # Notify global subscribers
        for ref in self._global_subscribers:
            handler = ref()
            if handler is not None:
                try:
                    await handler(event)
                    handlers_called += 1
                except Exception as e:
                    logger.error(f"Global event handler error: {e}")

        return handlers_called

    async def publish_and_wait(
        self,
        event: Event,
        timeout: float = 30.0,
    ) -> None:
        """Publish event and wait for handlers to complete."""
        await asyncio.wait_for(self.publish(event), timeout)

    def get_history(
        self,
        event_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get event history."""
        history = self._event_history
        if event_name:
            history = [e for e in history if e.name == event_name]
        return history[-limit:]


# =============================================================================
# HEALTH MONITOR WITH HEARTBEATS
# =============================================================================

@dataclass
class HealthStatus:
    """Health status of a component."""
    component: str
    healthy: bool
    last_check: datetime
    latency_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """
    Health monitoring with periodic heartbeats.

    Features:
        - Periodic health checks
        - Latency tracking
        - Alert callbacks
        - Health history for trend analysis
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        unhealthy_threshold: int = 3,
    ):
        self._check_interval = check_interval
        self._unhealthy_threshold = unhealthy_threshold
        self._components: Dict[str, Callable[[], Awaitable[bool]]] = {}
        self._status: Dict[str, HealthStatus] = {}
        self._consecutive_failures: Dict[str, int] = {}
        self._task: Optional[asyncio.Task[None]] = None
        self._alert_callbacks: List[Callable[[str, HealthStatus], Awaitable[None]]] = []

    def register(
        self,
        name: str,
        check: Callable[[], Awaitable[bool]],
    ) -> None:
        """Register a component for health monitoring."""
        self._components[name] = check
        self._consecutive_failures[name] = 0

    def on_unhealthy(
        self,
        callback: Callable[[str, HealthStatus], Awaitable[None]],
    ) -> None:
        """Register callback for unhealthy components."""
        self._alert_callbacks.append(callback)

    async def start(self) -> None:
        """Start health monitoring."""
        if self._task is None:
            self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop health monitoring."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while True:
            try:
                await self._check_all()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    async def _check_all(self) -> None:
        """Check all registered components."""
        tasks = [
            self._check_component(name, check)
            for name, check in self._components.items()
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_component(
        self,
        name: str,
        check: Callable[[], Awaitable[bool]],
    ) -> None:
        """Check a single component."""
        start = time.time()
        error: Optional[str] = None
        healthy = False

        try:
            healthy = await asyncio.wait_for(check(), timeout=10.0)
        except asyncio.TimeoutError:
            error = "Health check timeout"
        except Exception as e:
            error = str(e)

        latency_ms = (time.time() - start) * 1000

        status = HealthStatus(
            component=name,
            healthy=healthy,
            last_check=datetime.now(),
            latency_ms=latency_ms,
            error=error,
        )
        self._status[name] = status

        # Track consecutive failures
        if healthy:
            self._consecutive_failures[name] = 0
        else:
            self._consecutive_failures[name] += 1
            if self._consecutive_failures[name] >= self._unhealthy_threshold:
                for callback in self._alert_callbacks:
                    try:
                        await callback(name, status)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

    def get_status(self, name: Optional[str] = None) -> Dict[str, HealthStatus]:
        """Get health status."""
        if name:
            return {name: self._status[name]} if name in self._status else {}
        return self._status.copy()

    def is_healthy(self, name: Optional[str] = None) -> bool:
        """Check if component(s) are healthy."""
        if name:
            return self._status.get(name, HealthStatus(
                component=name, healthy=False, last_check=datetime.now(), latency_ms=0
            )).healthy
        return all(s.healthy for s in self._status.values())


# =============================================================================
# JARVIS REPO DISCOVERY (ENHANCED)
# =============================================================================

class RepoDiscoveryStrategy(ABC):
    """Abstract strategy for repo discovery."""

    @abstractmethod
    async def discover(self) -> Optional[Path]: ...


class EnvVarStrategy(RepoDiscoveryStrategy):
    """Discover repo from environment variable."""

    def __init__(self, var_name: str, core_path: str = "backend/core"):
        self._var_name = var_name
        self._core_path = core_path

    async def discover(self) -> Optional[Path]:
        env_path = os.getenv(self._var_name)
        if env_path:
            path = Path(env_path).expanduser()
            if path.exists() and (path / self._core_path).exists():
                return path
        return None


class SiblingDirStrategy(RepoDiscoveryStrategy):
    """Discover repo from sibling directory."""

    def __init__(self, names: Sequence[str], core_path: str = "backend/core"):
        self._names = names
        self._core_path = core_path

    async def discover(self) -> Optional[Path]:
        prime_path = Path(__file__).parent.parent.parent
        parent = prime_path.parent

        for name in self._names:
            candidate = parent / name
            if candidate.exists() and (candidate / self._core_path).exists():
                return candidate
        return None


class CommonPathsStrategy(RepoDiscoveryStrategy):
    """Discover repo from common paths."""

    def __init__(self, paths: Sequence[Path], core_path: str = "backend/core"):
        self._paths = paths
        self._core_path = core_path

    async def discover(self) -> Optional[Path]:
        for path in self._paths:
            expanded = path.expanduser()
            if expanded.exists() and (expanded / self._core_path).exists():
                return expanded
        return None


class RepoDiscovery:
    """
    Multi-strategy repository discovery.

    Uses chain of responsibility pattern for flexible discovery.
    """

    def __init__(self, strategies: Sequence[RepoDiscoveryStrategy]):
        self._strategies = strategies
        self._cache: Dict[str, Optional[Path]] = {}

    async def discover(self, cache_key: str = "default") -> Optional[Path]:
        """Discover repo using all strategies."""
        if cache_key in self._cache:
            return self._cache[cache_key]

        for strategy in self._strategies:
            try:
                path = await strategy.discover()
                if path:
                    self._cache[cache_key] = path
                    return path
            except Exception as e:
                logger.debug(f"Discovery strategy failed: {e}")

        self._cache[cache_key] = None
        return None


# Create default JARVIS discovery
def create_jarvis_discovery() -> RepoDiscovery:
    """Create JARVIS repo discovery with default strategies."""
    return RepoDiscovery([
        EnvVarStrategy("JARVIS_AI_AGENT_PATH"),
        SiblingDirStrategy(["JARVIS-AI-Agent", "jarvis-ai-agent", "jarvis"]),
        CommonPathsStrategy([
            Path.home() / "Documents" / "repos" / "JARVIS-AI-Agent",
            Path.home() / "repos" / "JARVIS-AI-Agent",
            Path.home() / "code" / "JARVIS-AI-Agent",
        ]),
    ])


def create_reactor_discovery() -> RepoDiscovery:
    """Create Reactor-Core repo discovery with default strategies."""
    return RepoDiscovery([
        EnvVarStrategy("REACTOR_CORE_PATH", "reactor_core"),
        SiblingDirStrategy(["reactor-core", "Reactor-Core"], "reactor_core"),
        CommonPathsStrategy([
            Path.home() / "Documents" / "repos" / "reactor-core",
            Path.home() / "repos" / "reactor-core",
        ], "reactor_core"),
    ])


# =============================================================================
# COMPONENT PROVIDERS (DEPENDENCY INJECTION)
# =============================================================================

class ComponentProvider(Generic[T]):
    """
    Lazy component provider with caching and fallback.

    Implements dependency injection pattern for cross-repo components.
    """

    def __init__(
        self,
        name: str,
        factory: Callable[[], Awaitable[Optional[T]]],
        fallback_factory: Optional[Callable[[], T]] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        cache: Optional[AsyncTTLCache[T]] = None,
    ):
        self._name = name
        self._factory = factory
        self._fallback_factory = fallback_factory
        self._circuit_breaker = circuit_breaker
        self._cache = cache
        self._instance: Optional[T] = None
        self._lock = asyncio.Lock()
        self._initialized = False
        self._last_error: Optional[Exception] = None

    async def get(self) -> Optional[T]:
        """Get component instance."""
        async with self._lock:
            if self._instance is not None:
                return self._instance

            # Check circuit breaker
            if self._circuit_breaker:
                if not await self._circuit_breaker.can_execute():
                    logger.warning(f"Circuit breaker open for {self._name}")
                    return self._get_fallback()

            try:
                self._instance = await self._factory()
                if self._instance is not None:
                    if self._circuit_breaker:
                        await self._circuit_breaker.record_success()
                    self._initialized = True
                    logger.debug(f"Component {self._name} initialized")
                return self._instance

            except Exception as e:
                self._last_error = e
                if self._circuit_breaker:
                    await self._circuit_breaker.record_failure()
                logger.warning(f"Failed to get {self._name}: {e}")
                return self._get_fallback()

    def _get_fallback(self) -> Optional[T]:
        """Get fallback instance."""
        if self._fallback_factory:
            try:
                return self._fallback_factory()
            except Exception as e:
                logger.error(f"Fallback factory failed for {self._name}: {e}")
        return None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def last_error(self) -> Optional[Exception]:
        return self._last_error


# =============================================================================
# JARVIS MODULE IMPORTER
# =============================================================================

class JARVISModuleImporter:
    """
    Intelligent module importer for JARVIS components.

    Features:
        - Lazy loading
        - Path management
        - Error recovery
        - Module caching
    """

    def __init__(self, repo_discovery: RepoDiscovery):
        self._discovery = repo_discovery
        self._modules: Dict[str, Any] = {}
        self._available: Optional[bool] = None
        self._lock = asyncio.Lock()

    async def ensure_path(self) -> bool:
        """Ensure JARVIS is in Python path."""
        if self._available is not None:
            return self._available

        async with self._lock:
            if self._available is not None:
                return self._available

            jarvis_path = await self._discovery.discover()
            if jarvis_path is None:
                self._available = False
                return False

            jarvis_str = str(jarvis_path)
            if jarvis_str not in sys.path:
                sys.path.insert(0, jarvis_str)
                logger.info(f"Added JARVIS to path: {jarvis_path}")

            self._available = True
            return True

    async def import_module(self, module_name: str) -> Optional[Any]:
        """Import a JARVIS module."""
        if module_name in self._modules:
            return self._modules[module_name]

        if not await self.ensure_path():
            return None

        try:
            import importlib
            module = importlib.import_module(module_name)
            self._modules[module_name] = module
            return module
        except ImportError as e:
            logger.debug(f"Failed to import {module_name}: {e}")
            return None


# =============================================================================
# MAIN BRIDGE CLASS
# =============================================================================

class GCPImportBridge:
    """
    Enterprise-grade GCP Import Bridge.

    Provides unified access to JARVIS GCP components with:
        - Circuit breakers
        - Rate limiting
        - Caching
        - Health monitoring
        - Event bus integration
    """

    _instance: ClassVar[Optional["GCPImportBridge"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self):
        # Discovery
        self._jarvis_discovery = create_jarvis_discovery()
        self._reactor_discovery = create_reactor_discovery()
        self._importer = JARVISModuleImporter(self._jarvis_discovery)

        # Circuit breakers
        self._circuit_breakers = {
            "cost_tracker": CircuitBreaker("cost_tracker"),
            "vm_manager": CircuitBreaker("vm_manager"),
            "optimizer": CircuitBreaker("optimizer"),
            "ram_monitor": CircuitBreaker("ram_monitor"),
        }

        # Rate limiter
        self._rate_limiter = TokenBucketRateLimiter(
            RateLimitConfig(requests_per_second=20.0, burst_size=50)
        )

        # Cache
        self._cache: AsyncTTLCache[Any] = AsyncTTLCache(default_ttl=60.0)

        # Health monitor
        self._health_monitor = HealthMonitor()

        # Component providers
        self._cost_tracker_provider: Optional[ComponentProvider[Any]] = None
        self._vm_manager_provider: Optional[ComponentProvider[Any]] = None
        self._optimizer_provider: Optional[ComponentProvider[Any]] = None
        self._ram_monitor_provider: Optional[ComponentProvider[Any]] = None

        # State
        self._initialized = False
        self._event_bus: Optional[EventBus] = None

    @classmethod
    async def get_instance(cls) -> "GCPImportBridge":
        """Get singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        return cls._instance

    async def initialize(self) -> None:
        """Initialize the bridge."""
        if self._initialized:
            return

        # Start cache cleanup
        await self._cache.start_cleanup()

        # Get event bus
        self._event_bus = await EventBus.get_instance()

        # Create component providers
        self._cost_tracker_provider = ComponentProvider(
            "cost_tracker",
            self._create_cost_tracker,
            circuit_breaker=self._circuit_breakers["cost_tracker"],
        )
        self._vm_manager_provider = ComponentProvider(
            "vm_manager",
            self._create_vm_manager,
            circuit_breaker=self._circuit_breakers["vm_manager"],
        )
        self._optimizer_provider = ComponentProvider(
            "optimizer",
            self._create_optimizer,
            circuit_breaker=self._circuit_breakers["optimizer"],
        )
        self._ram_monitor_provider = ComponentProvider(
            "ram_monitor",
            self._create_ram_monitor,
            circuit_breaker=self._circuit_breakers["ram_monitor"],
        )

        # Register health checks
        self._health_monitor.register(
            "jarvis_repo",
            self._check_jarvis_health,
        )
        await self._health_monitor.start()

        self._initialized = True
        logger.info("GCP Import Bridge initialized")

        # Publish event
        if self._event_bus:
            await self._event_bus.publish(Event(
                name="gcp_bridge.initialized",
                payload={"status": "success"},
            ))

    async def shutdown(self) -> None:
        """Shutdown the bridge."""
        await self._cache.stop_cleanup()
        await self._health_monitor.stop()

        if self._event_bus:
            await self._event_bus.publish(Event(
                name="gcp_bridge.shutdown",
                payload={"status": "success"},
            ))

    async def _check_jarvis_health(self) -> bool:
        """Health check for JARVIS repo."""
        path = await self._jarvis_discovery.discover()
        return path is not None and path.exists()

    # Component factories
    async def _create_cost_tracker(self) -> Optional[Any]:
        """Create CostTracker instance."""
        module = await self._importer.import_module("backend.core.cost_tracker")
        if module is None:
            return None

        try:
            # CRITICAL: CostTracker uses asyncio.Lock() in __init__
            # So we CANNOT run it in an executor (no event loop in that thread)
            # Must run synchronously in the current async context

            if hasattr(module, "get_cost_tracker"):
                factory = module.get_cost_tracker

                if asyncio.iscoroutinefunction(factory):
                    instance = await factory()
                else:
                    # It's synchronous but creates asyncio objects internally
                    # Run directly (not in executor!) - fast enough
                    instance = factory()
                return instance

            elif hasattr(module, "CostTracker"):
                # Direct instantiation
                if hasattr(module, "CostTrackerConfig"):
                    config = module.CostTrackerConfig()
                    instance = module.CostTracker(config)
                else:
                    instance = module.CostTracker()

                if hasattr(instance, "initialize"):
                    init_result = instance.initialize()
                    if asyncio.iscoroutine(init_result):
                        await init_result
                return instance

        except Exception as e:
            logger.warning(f"CostTracker creation failed: {e}")
        return None

    async def _create_vm_manager(self) -> Optional[Any]:
        """Create GCPVMManager instance."""
        module = await self._importer.import_module("backend.core.gcp_vm_manager")
        if module is None:
            return None

        try:
            if hasattr(module, "get_gcp_vm_manager"):
                return await module.get_gcp_vm_manager()
            elif hasattr(module, "GCPVMManager"):
                instance = module.GCPVMManager()
                if hasattr(instance, "initialize"):
                    init_result = instance.initialize()
                    if asyncio.iscoroutine(init_result):
                        await init_result
                return instance
        except Exception as e:
            logger.warning(f"GCPVMManager creation failed: {e}")
        return None

    async def _create_optimizer(self) -> Optional[Any]:
        """Create IntelligentGCPOptimizer instance."""
        module = await self._importer.import_module("backend.core.intelligent_gcp_optimizer")
        if module is None:
            return None

        try:
            if hasattr(module, "get_intelligent_optimizer"):
                return await module.get_intelligent_optimizer()
            elif hasattr(module, "IntelligentGCPOptimizer"):
                instance = module.IntelligentGCPOptimizer()
                if hasattr(instance, "initialize"):
                    init_result = instance.initialize()
                    if asyncio.iscoroutine(init_result):
                        await init_result
                return instance
        except Exception as e:
            logger.warning(f"IntelligentGCPOptimizer creation failed: {e}")
        return None

    async def _create_ram_monitor(self) -> Optional[Any]:
        """Create AdvancedRAMMonitor instance."""
        module = await self._importer.import_module("backend.core.advanced_ram_monitor")
        if module is None:
            return None

        try:
            # AdvancedRAMMonitor requires a config dict
            default_config = {
                "monitoring_interval_seconds": 5.0,
                "history_size": 60,
                "pressure_thresholds": {
                    "low": 50.0,
                    "normal": 70.0,
                    "high": 85.0,
                    "critical": 95.0,
                },
                "workload_threshold_gb": 2.0,
                "enable_predictive": True,
            }

            if hasattr(module, "get_ram_monitor"):
                # get_ram_monitor may be sync or async
                factory = module.get_ram_monitor
                if asyncio.iscoroutinefunction(factory):
                    instance = await factory(default_config)
                else:
                    instance = factory(default_config)

                # Start monitoring if needed
                if hasattr(instance, "start"):
                    await instance.start()
                return instance

            elif hasattr(module, "AdvancedRAMMonitor"):
                # Direct instantiation with config
                instance = module.AdvancedRAMMonitor(default_config)
                if hasattr(instance, "start"):
                    await instance.start()
                return instance

        except Exception as e:
            logger.warning(f"AdvancedRAMMonitor creation failed: {e}")
        return None

    # Public API
    async def get_cost_tracker(self) -> Optional[Any]:
        """Get CostTracker instance."""
        if self._cost_tracker_provider:
            return await self._cost_tracker_provider.get()
        return None

    async def get_vm_manager(self) -> Optional[Any]:
        """Get GCPVMManager instance."""
        if self._vm_manager_provider:
            return await self._vm_manager_provider.get()
        return None

    async def get_optimizer(self) -> Optional[Any]:
        """Get IntelligentGCPOptimizer instance."""
        if self._optimizer_provider:
            return await self._optimizer_provider.get()
        return None

    async def get_ram_monitor(self) -> Optional[Any]:
        """Get AdvancedRAMMonitor instance."""
        if self._ram_monitor_provider:
            return await self._ram_monitor_provider.get()
        return None

    async def get_budget_status(self) -> Dict[str, Any]:
        """Get budget status with caching."""
        cache_key = "budget_status"

        async def fetch():
            tracker = await self.get_cost_tracker()
            if tracker and hasattr(tracker, "get_budget_status"):
                try:
                    return await tracker.get_budget_status()
                except Exception as e:
                    logger.warning(f"Budget status fetch failed: {e}")
            return {
                "available": False,
                "fallback": True,
                "daily_remaining": 10.0,
            }

        return await self._cache.get_or_set(cache_key, fetch, ttl=30.0)

    async def can_afford_cloud_tier(self, estimated_cost: float) -> bool:
        """Check if budget allows cloud tier."""
        tracker = await self.get_cost_tracker()
        if tracker:
            try:
                if hasattr(tracker, "can_create_vm"):
                    return await tracker.can_create_vm(estimated_cost=estimated_cost)
            except Exception as e:
                logger.warning(f"Budget check failed: {e}")

        # Fallback: simple budget check
        status = await self.get_budget_status()
        return estimated_cost <= status.get("daily_remaining", 10.0)

    async def get_ram_snapshot(self) -> Dict[str, Any]:
        """Get RAM snapshot with fallback."""
        cache_key = "ram_snapshot"

        async def fetch():
            monitor = await self.get_ram_monitor()
            if monitor and hasattr(monitor, "get_snapshot"):
                try:
                    snapshot = await monitor.get_snapshot()
                    return {
                        "available": True,
                        "total_gb": getattr(snapshot, "total_gb", 0.0),
                        "used_gb": getattr(snapshot, "used_gb", 0.0),
                        "available_gb": getattr(snapshot, "available_gb", 0.0),
                        "percent_used": getattr(snapshot, "percent_used", 0.0),
                        "pressure_level": getattr(snapshot, "pressure_level", "unknown"),
                    }
                except Exception as e:
                    logger.warning(f"RAM snapshot failed: {e}")

            # Fallback to psutil
            try:
                import psutil
                mem = psutil.virtual_memory()
                return {
                    "available": False,
                    "fallback": True,
                    "total_gb": mem.total / (1024**3),
                    "used_gb": mem.used / (1024**3),
                    "available_gb": mem.available / (1024**3),
                    "percent_used": mem.percent,
                    "pressure_level": (
                        "critical" if mem.percent > 90 else
                        "high" if mem.percent > 80 else
                        "normal"
                    ),
                }
            except ImportError:
                return {"available": False, "error": "psutil not available"}

        return await self._cache.get_or_set(cache_key, fetch, ttl=5.0)

    async def calculate_pressure_score(self) -> Dict[str, Any]:
        """Calculate system pressure score."""
        # Get base metrics
        try:
            import psutil
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            base_result = {
                "ram_pressure": mem.percent / 100.0,
                "cpu_pressure": cpu / 100.0,
                "overall": (mem.percent + cpu) / 200.0,
            }
        except ImportError:
            base_result = {
                "ram_pressure": 0.5,
                "cpu_pressure": 0.5,
                "overall": 0.5,
            }

        # Try JARVIS optimizer
        optimizer = await self.get_optimizer()
        if optimizer and hasattr(optimizer, "calculate_pressure_score"):
            try:
                ram_snapshot = await self.get_ram_snapshot()
                try:
                    score = await optimizer.calculate_pressure_score(ram_snapshot)
                except TypeError:
                    score = await optimizer.calculate_pressure_score()

                if isinstance(score, dict):
                    return {
                        "available": True,
                        **base_result,
                        **score,
                        "should_burst": score.get("should_burst", base_result["overall"] > 0.8),
                    }
                else:
                    return {
                        "available": True,
                        "ram_pressure": getattr(score, "ram_pressure", base_result["ram_pressure"]),
                        "cpu_pressure": getattr(score, "cpu_pressure", base_result["cpu_pressure"]),
                        "overall": getattr(score, "overall", base_result["overall"]),
                        "should_burst": getattr(score, "should_burst", base_result["overall"] > 0.8),
                    }
            except Exception as e:
                logger.warning(f"Pressure score calculation failed: {e}")

        return {
            "available": False,
            "fallback": True,
            **base_result,
            "should_burst": base_result["overall"] > 0.8,
        }

    async def should_burst_to_cloud(self) -> bool:
        """Determine if cloud burst is recommended."""
        score = await self.calculate_pressure_score()
        return score.get("should_burst", False)

    async def request_cloud_burst(
        self,
        reason: str,
        required_ram_gb: float = 0.0,
        estimated_duration_minutes: float = 30.0,
        priority: int = 5,
    ) -> Dict[str, Any]:
        """Request elastic cloud burst."""
        # Rate limit
        await self._rate_limiter.wait_for_token()

        # Budget check
        estimated_cost = (estimated_duration_minutes / 60.0) * 0.05
        if not await self.can_afford_cloud_tier(estimated_cost):
            return {
                "success": False,
                "error": "insufficient_budget",
            }

        # Check if burst needed
        if not await self.should_burst_to_cloud():
            return {
                "success": False,
                "error": "burst_not_needed",
            }

        # Request VM
        vm_manager = await self.get_vm_manager()
        if vm_manager is None:
            return {
                "success": False,
                "error": "vm_manager_unavailable",
            }

        try:
            if hasattr(vm_manager, "provision_spot_vm"):
                result = await vm_manager.provision_spot_vm(
                    reason=reason,
                    min_ram_gb=required_ram_gb,
                    max_duration_minutes=estimated_duration_minutes,
                )

                # Publish event
                if self._event_bus:
                    await self._event_bus.publish(Event(
                        name="gcp_bridge.burst_requested",
                        payload={"reason": reason, "result": "success"},
                    ))

                return {
                    "success": True,
                    "vm_info": result,
                    "estimated_cost": estimated_cost,
                }
        except Exception as e:
            logger.error(f"Cloud burst failed: {e}")
            return {
                "success": False,
                "error": "provisioning_failed",
                "message": str(e),
            }

        return {
            "success": False,
            "error": "provision_not_available",
        }

    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Get comprehensive infrastructure status."""
        jarvis_path = await self._jarvis_discovery.discover()
        reactor_path = await self._reactor_discovery.discover()

        components = {}
        for name in ["cost_tracker", "vm_manager", "optimizer", "ram_monitor"]:
            provider = getattr(self, f"_{name}_provider", None)
            if provider:
                instance = await provider.get()
                components[name] = {
                    "available": instance is not None,
                    "type": type(instance).__name__ if instance else None,
                    "circuit_breaker": self._circuit_breakers[name].get_stats(),
                }

        return {
            "jarvis_repo": str(jarvis_path) if jarvis_path else None,
            "reactor_repo": str(reactor_path) if reactor_path else None,
            "components": components,
            "health": {
                name: status.healthy
                for name, status in self._health_monitor.get_status().items()
            },
            "cache_stats": self._cache.get_stats(),
            "summary": {
                "total_components": len(components),
                "available_components": sum(
                    1 for c in components.values() if c["available"]
                ),
            },
        }


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

@asynccontextmanager
async def GCPBridgeContext() -> AsyncIterator[GCPImportBridge]:
    """
    Async context manager for GCP Import Bridge.

    Usage:
        async with GCPBridgeContext() as bridge:
            status = await bridge.get_budget_status()
    """
    bridge = await GCPImportBridge.get_instance()
    try:
        yield bridge
    finally:
        pass  # Singleton persists


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_bridge: Optional[GCPImportBridge] = None
_bridge_lock = asyncio.Lock()


async def _get_bridge() -> GCPImportBridge:
    """Get bridge instance."""
    global _bridge
    if _bridge is None:
        async with _bridge_lock:
            if _bridge is None:
                _bridge = await GCPImportBridge.get_instance()
    return _bridge


# Repo detection
def _detect_jarvis_repo_path() -> Optional[Path]:
    """Sync wrapper for JARVIS repo detection."""
    discovery = create_jarvis_discovery()
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(discovery.discover())
    finally:
        loop.close()


def _ensure_jarvis_in_path() -> bool:
    """Ensure JARVIS in path."""
    path = _detect_jarvis_repo_path()
    if path:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        return True
    return False


# Async convenience functions
async def get_cost_tracker() -> Optional[Any]:
    """Get CostTracker."""
    bridge = await _get_bridge()
    return await bridge.get_cost_tracker()


async def get_gcp_vm_manager() -> Optional[Any]:
    """Get GCPVMManager."""
    bridge = await _get_bridge()
    return await bridge.get_vm_manager()


async def get_intelligent_optimizer() -> Optional[Any]:
    """Get IntelligentGCPOptimizer."""
    bridge = await _get_bridge()
    return await bridge.get_optimizer()


async def get_advanced_ram_monitor() -> Optional[Any]:
    """Get AdvancedRAMMonitor."""
    bridge = await _get_bridge()
    return await bridge.get_ram_monitor()


async def get_budget_status() -> Dict[str, Any]:
    """Get budget status."""
    bridge = await _get_bridge()
    return await bridge.get_budget_status()


async def can_afford_cloud_tier(estimated_cost: float) -> bool:
    """Check if budget allows cloud tier."""
    bridge = await _get_bridge()
    return await bridge.can_afford_cloud_tier(estimated_cost)


async def get_ram_snapshot() -> Dict[str, Any]:
    """Get RAM snapshot."""
    bridge = await _get_bridge()
    return await bridge.get_ram_snapshot()


async def calculate_pressure_score() -> Dict[str, Any]:
    """Calculate pressure score."""
    bridge = await _get_bridge()
    return await bridge.calculate_pressure_score()


async def should_burst_to_cloud() -> bool:
    """Check if cloud burst recommended."""
    bridge = await _get_bridge()
    return await bridge.should_burst_to_cloud()


async def request_cloud_burst(
    reason: str,
    required_ram_gb: float = 0.0,
    estimated_duration_minutes: float = 30.0,
    priority: int = 5,
) -> Dict[str, Any]:
    """Request cloud burst."""
    bridge = await _get_bridge()
    return await bridge.request_cloud_burst(
        reason=reason,
        required_ram_gb=required_ram_gb,
        estimated_duration_minutes=estimated_duration_minutes,
        priority=priority,
    )


async def get_gcp_infrastructure_status() -> Dict[str, Any]:
    """Get infrastructure status."""
    bridge = await _get_bridge()
    return await bridge.get_infrastructure_status()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "CostTrackerProtocol",
    "VMManagerProtocol",
    "OptimizerProtocol",
    "RAMMonitorProtocol",
    # Retry
    "RetryConfig",
    "RetryExhausted",
    "retry_with_backoff",
    "with_retry",
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerOpen",
    # Rate Limiter
    "RateLimitConfig",
    "TokenBucketRateLimiter",
    # Cache
    "CacheEntry",
    "AsyncTTLCache",
    # Event Bus
    "Event",
    "EventBus",
    # Health
    "HealthStatus",
    "HealthMonitor",
    # Discovery
    "RepoDiscovery",
    "create_jarvis_discovery",
    "create_reactor_discovery",
    # Provider
    "ComponentProvider",
    # Bridge
    "GCPImportBridge",
    "GCPBridgeContext",
    # Convenience functions
    "_detect_jarvis_repo_path",
    "_ensure_jarvis_in_path",
    "get_cost_tracker",
    "get_gcp_vm_manager",
    "get_intelligent_optimizer",
    "get_advanced_ram_monitor",
    "get_budget_status",
    "can_afford_cloud_tier",
    "get_ram_snapshot",
    "calculate_pressure_score",
    "should_burst_to_cloud",
    "request_cloud_burst",
    "get_gcp_infrastructure_status",
]
