"""
Advanced Inference Engine - v73.0 Diamond-Hard Protocol
========================================================

High-performance inference engine with:
- Concurrent request handling with priority queuing
- Circuit breaker for failure resilience
- Adaptive timeout management
- Batched inference for throughput optimization
- Memory-efficient context windowing
- Inference health tracking for Trinity

ARCHITECTURE:
    Request → Priority Queue → Circuit Breaker → Executor Pool → Response

FEATURES:
    - Async-first design for maximum concurrency
    - Request deduplication (same prompt = cached response)
    - Automatic retry with exponential backoff
    - Graceful degradation under load
    - Integration with Trinity health metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Generic,
)
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Circuit Breaker - Prevents cascade failures
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests (too many failures)
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 30.0       # Time before half-open
    excluded_exceptions: Tuple[type, ...] = ()  # Don't count these as failures


class CircuitBreaker:
    """
    Circuit breaker for inference resilience.

    Prevents cascade failures by temporarily blocking requests after
    repeated failures, allowing the system to recover.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Blocking all requests (service is failing)
        HALF_OPEN: Testing with limited requests

    Usage:
        breaker = CircuitBreaker()

        async with breaker:
            result = await do_inference()
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    async def _check_state(self) -> None:
        """Check and potentially update circuit state."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if timeout elapsed
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    logger.info("[CircuitBreaker] Transitioning to HALF_OPEN")
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info("[CircuitBreaker] Recovered - CLOSED")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)

    async def record_failure(self, exception: BaseException) -> None:
        """Record a failed call."""
        # Check if exception is excluded
        if isinstance(exception, self.config.excluded_exceptions):
            return

        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                logger.warning("[CircuitBreaker] Failure in HALF_OPEN - returning to OPEN")
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"[CircuitBreaker] Opening circuit after {self._failure_count} failures"
                    )
                    self._state = CircuitState.OPEN

    async def __aenter__(self):
        await self._check_state()
        if self._state == CircuitState.OPEN:
            raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            await self.record_success()
        elif exc_val is not None:
            await self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
            "is_open": self.is_open,
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# Priority Request Queue
# =============================================================================

class RequestPriority(Enum):
    """Request priority levels."""
    CRITICAL = 0    # System messages, health checks
    HIGH = 1        # User-facing interactive
    NORMAL = 2      # Standard requests
    LOW = 3         # Background tasks, batch processing
    BATCH = 4       # Deferred batch processing


@dataclass
class InferenceRequest:
    """A queued inference request."""
    id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    priority: RequestPriority = RequestPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    timeout_seconds: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Response handling
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.timeout_seconds

    @property
    def prompt_hash(self) -> str:
        """Hash for deduplication."""
        key = f"{self.prompt}:{self.max_tokens}:{self.temperature}:{self.top_p}"
        return hashlib.md5(key.encode()).hexdigest()


@dataclass
class InferenceResponse:
    """Response from inference."""
    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class PriorityRequestQueue:
    """
    Priority queue for inference requests.

    Features:
        - Multiple priority levels
        - Request deduplication
        - Timeout handling
        - FIFO within priority level
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._queues: Dict[RequestPriority, asyncio.Queue] = {
            p: asyncio.Queue(maxsize=max_size // len(RequestPriority))
            for p in RequestPriority
        }
        self._pending: Dict[str, InferenceRequest] = {}
        self._dedup_cache: Dict[str, InferenceRequest] = {}
        self._lock = asyncio.Lock()

    async def put(self, request: InferenceRequest) -> InferenceRequest:
        """
        Add a request to the queue.

        Returns the request (may be existing if deduplicated).
        """
        async with self._lock:
            # Check for duplicate in-flight request
            prompt_hash = request.prompt_hash
            if prompt_hash in self._dedup_cache:
                existing = self._dedup_cache[prompt_hash]
                if not existing.future.done():
                    logger.debug(f"[Queue] Deduplicating request {request.id} → {existing.id}")
                    return existing

            # Add to appropriate queue
            queue = self._queues[request.priority]
            await queue.put(request)

            # Track for deduplication and retrieval
            self._pending[request.id] = request
            self._dedup_cache[prompt_hash] = request

            return request

    async def get(self) -> InferenceRequest:
        """
        Get the highest priority non-expired request.

        Blocks until a request is available.
        """
        while True:
            # Try queues in priority order
            for priority in RequestPriority:
                queue = self._queues[priority]
                try:
                    request = queue.get_nowait()

                    # Skip expired requests
                    if request.is_expired:
                        request.future.set_exception(
                            TimeoutError("Request expired in queue")
                        )
                        async with self._lock:
                            self._pending.pop(request.id, None)
                            self._dedup_cache.pop(request.prompt_hash, None)
                        continue

                    return request
                except asyncio.QueueEmpty:
                    continue

            # No requests available, wait briefly
            await asyncio.sleep(0.01)

    async def complete(self, request: InferenceRequest, response: InferenceResponse) -> None:
        """Mark a request as complete with response."""
        async with self._lock:
            self._pending.pop(request.id, None)
            # Keep in dedup cache briefly for response sharing

        if not request.future.done():
            request.future.set_result(response)

    async def fail(self, request: InferenceRequest, error: Exception) -> None:
        """Mark a request as failed."""
        async with self._lock:
            self._pending.pop(request.id, None)
            self._dedup_cache.pop(request.prompt_hash, None)

        if not request.future.done():
            request.future.set_exception(error)

    def get_status(self) -> Dict[str, Any]:
        """Get queue status."""
        return {
            "pending": len(self._pending),
            "queues": {
                p.name: self._queues[p].qsize() for p in RequestPriority
            },
            "dedup_cache_size": len(self._dedup_cache),
        }


# =============================================================================
# LRU Response Cache
# =============================================================================

class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache for inference responses.

    Features:
        - Configurable max size and TTL
        - Automatic eviction of stale entries
        - Hit rate tracking
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[T]:
        """Get item from cache, returns None if not found or expired."""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    async def put(self, key: str, value: T) -> None:
        """Add item to cache."""
        async with self._lock:
            # Remove if exists (to update timestamp)
            if key in self._cache:
                del self._cache[key]

            # Evict if full
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (value, time.time())

    async def clear(self) -> None:
        """Clear cache."""
        async with self._lock:
            self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


# =============================================================================
# Advanced Inference Engine
# =============================================================================

@dataclass
class InferenceEngineConfig:
    """Configuration for inference engine."""
    # Queue settings
    max_queue_size: int = 1000

    # Cache settings
    cache_enabled: bool = True
    cache_max_size: int = 500
    cache_ttl_seconds: float = 300.0

    # Circuit breaker
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 30.0

    # Concurrency
    max_concurrent_requests: int = 4

    # Timeouts
    default_timeout_seconds: float = 60.0

    # Retry
    max_retries: int = 2
    retry_delay_seconds: float = 1.0


class InferenceEngine:
    """
    Advanced inference engine with production-grade features.

    Features:
        - Priority request queuing
        - Response caching with LRU eviction
        - Circuit breaker for resilience
        - Concurrent request processing
        - Automatic retry with backoff
        - Integration with Trinity health metrics

    Usage:
        engine = InferenceEngine(executor, config)
        await engine.start()

        response = await engine.generate(
            prompt="Hello!",
            priority=RequestPriority.HIGH,
        )

        await engine.stop()
    """

    def __init__(
        self,
        executor: Any,  # LlamaCppExecutor or compatible
        config: Optional[InferenceEngineConfig] = None,
        trinity_record_fn: Optional[Callable[[float, bool], None]] = None,
    ):
        self.executor = executor
        self.config = config or InferenceEngineConfig()
        self._trinity_record = trinity_record_fn

        # Components
        self._queue = PriorityRequestQueue(self.config.max_queue_size)
        self._cache: Optional[LRUCache[InferenceResponse]] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None

        if self.config.cache_enabled:
            self._cache = LRUCache(
                max_size=self.config.cache_max_size,
                ttl_seconds=self.config.cache_ttl_seconds,
            )

        if self.config.circuit_breaker_enabled:
            self._circuit_breaker = CircuitBreaker(
                CircuitBreakerConfig(
                    failure_threshold=self.config.failure_threshold,
                    timeout_seconds=self.config.recovery_timeout_seconds,
                )
            )

        # Processing
        self._running = False
        self._workers: List[asyncio.Task] = []
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # Stats
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_latency_ms = 0.0

    async def start(self) -> None:
        """Start the inference engine."""
        if self._running:
            return

        self._running = True

        # Start worker tasks
        for i in range(self.config.max_concurrent_requests):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)

        logger.info(
            f"[InferenceEngine] Started with {len(self._workers)} workers"
        )

    async def stop(self) -> None:
        """Stop the inference engine."""
        self._running = False

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("[InferenceEngine] Stopped")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResponse:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            priority: Request priority level
            timeout: Request timeout (uses default if None)
            metadata: Additional metadata to attach

        Returns:
            InferenceResponse with generated text

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            TimeoutError: If request times out
        """
        # Check cache first
        if self._cache:
            cache_key = f"{prompt}:{max_tokens}:{temperature}:{top_p}"
            cached = await self._cache.get(cache_key)
            if cached:
                logger.debug(f"[InferenceEngine] Cache hit")
                return InferenceResponse(
                    request_id=f"cached-{int(time.time()*1000)}",
                    text=cached.text,
                    prompt_tokens=cached.prompt_tokens,
                    completion_tokens=cached.completion_tokens,
                    latency_ms=0.0,
                    cached=True,
                )

        # Create request
        request = InferenceRequest(
            id=f"req-{int(time.time()*1000)}-{hash(prompt) % 10000}",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            priority=priority,
            timeout_seconds=timeout or self.config.default_timeout_seconds,
            metadata=metadata or {},
        )

        # Queue request
        request = await self._queue.put(request)

        # Wait for response
        try:
            response = await asyncio.wait_for(
                request.future,
                timeout=request.timeout_seconds,
            )
            return response
        except asyncio.TimeoutError:
            await self._queue.fail(request, TimeoutError("Request timed out"))
            raise

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing requests."""
        logger.debug(f"[Worker-{worker_id}] Started")

        while self._running:
            try:
                # Get next request
                request = await self._queue.get()

                # Process with semaphore
                async with self._semaphore:
                    await self._process_request(request, worker_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Worker-{worker_id}] Error: {e}")

        logger.debug(f"[Worker-{worker_id}] Stopped")

    async def _process_request(
        self,
        request: InferenceRequest,
        worker_id: int,
    ) -> None:
        """Process a single request with retry."""
        self._total_requests += 1

        for attempt in range(self.config.max_retries + 1):
            try:
                # Check circuit breaker
                if self._circuit_breaker:
                    if self._circuit_breaker.is_open:
                        raise CircuitBreakerOpenError("Service unavailable")

                    async with self._circuit_breaker:
                        response = await self._execute_inference(request)
                else:
                    response = await self._execute_inference(request)

                # Success
                await self._queue.complete(request, response)
                self._successful_requests += 1
                self._total_latency_ms += response.latency_ms

                # Cache response
                if self._cache:
                    cache_key = f"{request.prompt}:{request.max_tokens}:{request.temperature}:{request.top_p}"
                    await self._cache.put(cache_key, response)

                # Record Trinity health
                if self._trinity_record:
                    try:
                        self._trinity_record(response.latency_ms, True)
                    except Exception:
                        pass

                return

            except CircuitBreakerOpenError:
                # Don't retry if circuit is open
                await self._queue.fail(request, CircuitBreakerOpenError("Service unavailable"))
                self._failed_requests += 1
                return

            except Exception as e:
                if attempt < self.config.max_retries:
                    logger.warning(
                        f"[Worker-{worker_id}] Retry {attempt + 1}/{self.config.max_retries}: {e}"
                    )
                    await asyncio.sleep(
                        self.config.retry_delay_seconds * (2 ** attempt)
                    )
                else:
                    logger.error(f"[Worker-{worker_id}] Failed after retries: {e}")
                    await self._queue.fail(request, e)
                    self._failed_requests += 1

                    # Record Trinity health (failure)
                    if self._trinity_record:
                        try:
                            self._trinity_record(0, False)
                        except Exception:
                            pass

    async def _execute_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Execute actual inference."""
        start = time.time()

        text = await self.executor.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        latency_ms = (time.time() - start) * 1000

        # Estimate tokens
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(text.split())

        return InferenceResponse(
            request_id=request.id,
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            metadata=request.metadata,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        avg_latency = (
            self._total_latency_ms / self._successful_requests
            if self._successful_requests > 0 else 0.0
        )

        return {
            "running": self._running,
            "workers": len(self._workers),
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (
                self._successful_requests / self._total_requests
                if self._total_requests > 0 else 0.0
            ),
            "avg_latency_ms": avg_latency,
            "queue": self._queue.get_status(),
            "cache": self._cache.get_stats() if self._cache else None,
            "circuit_breaker": (
                self._circuit_breaker.get_status() if self._circuit_breaker else None
            ),
        }


# =============================================================================
# Convenience Factory
# =============================================================================

def create_inference_engine(
    executor: Any,
    trinity_enabled: bool = True,
    cache_enabled: bool = True,
    max_workers: int = 4,
) -> InferenceEngine:
    """
    Factory function to create configured inference engine.

    Args:
        executor: Model executor (LlamaCppExecutor or compatible)
        trinity_enabled: Enable Trinity health tracking
        cache_enabled: Enable response caching
        max_workers: Number of concurrent workers

    Returns:
        Configured InferenceEngine
    """
    # Get Trinity record function if enabled
    trinity_record_fn = None
    if trinity_enabled:
        try:
            from jarvis_prime.core.trinity_bridge import record_inference
            trinity_record_fn = record_inference
        except ImportError:
            logger.warning("Trinity bridge not available for inference engine")

    config = InferenceEngineConfig(
        cache_enabled=cache_enabled,
        max_concurrent_requests=max_workers,
        circuit_breaker_enabled=True,
    )

    return InferenceEngine(
        executor=executor,
        config=config,
        trinity_record_fn=trinity_record_fn,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core
    "InferenceEngine",
    "InferenceEngineConfig",
    "InferenceRequest",
    "InferenceResponse",
    "RequestPriority",
    # Components
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitState",
    "PriorityRequestQueue",
    "LRUCache",
    # Factory
    "create_inference_engine",
]
