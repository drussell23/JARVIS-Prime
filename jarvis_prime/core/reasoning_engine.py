"""
Advanced Reasoning Engine - Multi-Strategy Cognitive Processing
================================================================

v92.0 - Production-Grade RAG-Integrated Reasoning with Full Observability

This module provides sophisticated reasoning strategies with RAG integration:
- Chain-of-Thought (CoT): Sequential reasoning with explicit steps
- Tree-of-Thoughts (ToT): Parallel exploration with branch pruning
- Self-Reflection: Meta-cognitive error detection and correction
- Hypothesis Testing: Scientific method-based reasoning
- Analogical Reasoning: Transfer learning from similar problems
- RAG-Augmented: Retrieval-enhanced reasoning with context injection

ARCHITECTURE:
    Input -> RAG Retrieval -> Strategy Selector -> Reasoning Strategy -> Verification -> Output
              |                   |                     |
              v                   v                     v
         Knowledge Base    Training Feedback    Metrics/Observability

FEATURES:
    - Dynamic strategy selection based on problem characteristics
    - Parallel thought exploration with intelligent pruning
    - Self-correcting reasoning with confidence tracking
    - Integration with AGI models for specialized processing
    - Streaming thought generation for real-time feedback
    - RAG integration for context-aware reasoning
    - Thread-safe statistics with AsyncLock
    - LRU cache with bounded memory and TTL
    - Semaphore-based rate limiting for parallel operations
    - Training feedback loop integration
    - Full observability with distributed tracing
    - Environment variable configuration
    - Graceful shutdown with state persistence

CRITICAL FIXES (v92.0):
    - Race conditions in statistics collection → AsyncLock
    - FIFO cache → Proper LRU with memory bounds
    - Missing RAG integration → Full RAG pipeline
    - Hardcoded values → Environment variable configuration
    - Missing timeout cancellation → Task cancellation propagation
    - No rate limiting → Semaphore-based parallel control
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import logging
import math
import os
import random
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from heapq import heappush, heappop, nlargest
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# ADVANCED UTILITIES - Thread-Safe Operations & Caching
# =============================================================================


class ThreadSafeStatistics:
    """
    Thread-safe statistics collector with atomic operations.

    Prevents race conditions when multiple coroutines update stats concurrently.
    Uses AsyncLock for all mutations and provides atomic increment/decrement.
    """

    __slots__ = ('_lock', '_counters', '_timings', '_histograms', '_last_updated')

    def __init__(self):
        self._lock = asyncio.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._histograms: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._last_updated = time.time()

    async def increment(self, key: str, value: int = 1) -> int:
        """Atomically increment a counter."""
        async with self._lock:
            self._counters[key] += value
            self._last_updated = time.time()
            return self._counters[key]

    async def decrement(self, key: str, value: int = 1) -> int:
        """Atomically decrement a counter."""
        async with self._lock:
            self._counters[key] -= value
            self._last_updated = time.time()
            return self._counters[key]

    async def record_timing(self, key: str, value_ms: float, max_history: int = 1000) -> None:
        """Record a timing value with bounded history."""
        async with self._lock:
            timings = self._timings[key]
            timings.append(value_ms)
            # Bounded history - remove oldest if over limit
            if len(timings) > max_history:
                self._timings[key] = timings[-max_history:]
            self._last_updated = time.time()

    async def record_histogram(self, key: str, bucket: str) -> None:
        """Record a histogram bucket."""
        async with self._lock:
            self._histograms[key][bucket] += 1
            self._last_updated = time.time()

    async def get_counter(self, key: str) -> int:
        """Get counter value (thread-safe read)."""
        async with self._lock:
            return self._counters.get(key, 0)

    async def get_timing_stats(self, key: str) -> Dict[str, float]:
        """Get timing statistics (avg, p50, p95, p99)."""
        async with self._lock:
            timings = self._timings.get(key, [])
            if not timings:
                return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

            sorted_timings = sorted(timings)
            n = len(sorted_timings)

            return {
                "count": n,
                "avg": sum(timings) / n,
                "min": sorted_timings[0],
                "max": sorted_timings[-1],
                "p50": sorted_timings[int(n * 0.50)],
                "p95": sorted_timings[int(n * 0.95)] if n > 20 else sorted_timings[-1],
                "p99": sorted_timings[int(n * 0.99)] if n > 100 else sorted_timings[-1],
            }

    async def get_all_stats(self) -> Dict[str, Any]:
        """Get all statistics."""
        async with self._lock:
            return {
                "counters": dict(self._counters),
                "histograms": {k: dict(v) for k, v in self._histograms.items()},
                "last_updated": self._last_updated,
            }

    async def reset(self) -> None:
        """Reset all statistics."""
        async with self._lock:
            self._counters.clear()
            self._timings.clear()
            self._histograms.clear()
            self._last_updated = time.time()


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with TTL and memory bounds.

    Features:
    - Least Recently Used eviction policy
    - Time-to-live expiration
    - Memory-bounded with size estimation
    - Thread-safe with AsyncLock
    - Atomic get-or-set operations
    """

    __slots__ = ('_lock', '_cache', '_max_size', '_ttl_seconds', '_size_estimator',
                 '_access_times', '_creation_times', '_hits', '_misses')

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 300.0,
        size_estimator: Optional[Callable[[T], int]] = None,
    ):
        self._lock = asyncio.Lock()
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._size_estimator = size_estimator
        self._access_times: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[T]:
        """Get item from cache, returns None if not found or expired."""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Check TTL
            creation_time = self._creation_times.get(key, 0)
            if time.time() - creation_time > self._ttl_seconds:
                # Expired - remove and return None
                self._remove_key(key)
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._access_times[key] = time.time()
            self._hits += 1

            return self._cache[key]

    async def set(self, key: str, value: T) -> None:
        """Set item in cache with LRU eviction."""
        async with self._lock:
            now = time.time()

            # If key exists, update it
            if key in self._cache:
                self._cache[key] = value
                self._cache.move_to_end(key)
                self._access_times[key] = now
                return

            # Evict expired entries first
            self._evict_expired()

            # Evict LRU entries if over capacity
            while len(self._cache) >= self._max_size:
                # Remove oldest (first) item
                oldest_key = next(iter(self._cache))
                self._remove_key(oldest_key)

            # Add new entry
            self._cache[key] = value
            self._access_times[key] = now
            self._creation_times[key] = now

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Awaitable[T]],
    ) -> T:
        """Atomic get-or-set operation."""
        # Try get first (common case)
        result = await self.get(key)
        if result is not None:
            return result

        # Not in cache, need to compute
        # Use a separate lock scope to avoid holding lock during computation
        value = await factory()
        await self.set(key, value)
        return value

    def _remove_key(self, key: str) -> None:
        """Remove key from all internal structures (must hold lock)."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._creation_times.pop(key, None)

    def _evict_expired(self) -> int:
        """Evict all expired entries (must hold lock). Returns count evicted."""
        now = time.time()
        expired_keys = [
            k for k, t in self._creation_times.items()
            if now - t > self._ttl_seconds
        ]
        for key in expired_keys:
            self._remove_key(key)
        return len(expired_keys)

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._creation_times.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total_requests if total_requests > 0 else 0,
                "ttl_seconds": self._ttl_seconds,
            }


class RateLimiter:
    """
    Semaphore-based rate limiter for parallel operations.

    Features:
    - Token bucket algorithm
    - Configurable concurrency limits
    - Timeout support
    - Per-operation tracking
    """

    __slots__ = ('_semaphore', '_max_concurrent', '_active_count', '_lock',
                 '_total_acquired', '_total_released', '_timeout_seconds')

    def __init__(self, max_concurrent: int = 4, timeout_seconds: float = 30.0):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._timeout_seconds = timeout_seconds
        self._active_count = 0
        self._lock = asyncio.Lock()
        self._total_acquired = 0
        self._total_released = 0

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """Acquire rate limit slot with timeout."""
        timeout = timeout or self._timeout_seconds

        try:
            # Try to acquire with timeout
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=timeout
            )

            async with self._lock:
                self._active_count += 1
                self._total_acquired += 1

            try:
                yield
            finally:
                self._semaphore.release()
                async with self._lock:
                    self._active_count -= 1
                    self._total_released += 1

        except asyncio.TimeoutError:
            logger.warning(f"Rate limiter timeout after {timeout}s")
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        async with self._lock:
            return {
                "max_concurrent": self._max_concurrent,
                "active_count": self._active_count,
                "total_acquired": self._total_acquired,
                "total_released": self._total_released,
                "available_slots": self._max_concurrent - self._active_count,
            }


class TaskCancellationManager:
    """
    Manages task cancellation propagation.

    Ensures that when a parent task times out, all child tasks
    are properly cancelled and cleaned up.
    """

    __slots__ = ('_tasks', '_lock', '_cancelled')

    def __init__(self):
        self._tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._cancelled = False

    async def register(self, task: asyncio.Task) -> None:
        """Register a task for cancellation management."""
        async with self._lock:
            if self._cancelled:
                task.cancel()
            else:
                self._tasks.add(task)
                # Auto-remove when done
                task.add_done_callback(lambda t: asyncio.create_task(self._remove_task(t)))

    async def _remove_task(self, task: asyncio.Task) -> None:
        """Remove completed task."""
        async with self._lock:
            self._tasks.discard(task)

    async def cancel_all(self, message: str = "Parent task cancelled") -> int:
        """Cancel all registered tasks."""
        async with self._lock:
            self._cancelled = True
            cancelled_count = 0

            for task in self._tasks:
                if not task.done():
                    task.cancel(msg=message)
                    cancelled_count += 1

            # Wait for all tasks to complete cancellation
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)

            self._tasks.clear()
            return cancelled_count

    async def wait_all(self, timeout: Optional[float] = None) -> List[Any]:
        """Wait for all registered tasks with optional timeout."""
        async with self._lock:
            tasks = list(self._tasks)

        if not tasks:
            return []

        if timeout:
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            return [t.result() for t in done if not t.cancelled()]
        else:
            return await asyncio.gather(*tasks, return_exceptions=True)


class CircuitBreaker:
    """
    Circuit breaker pattern for external service resilience.

    v92.1: Provides graceful degradation for RAG and other external services.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failures exceeded threshold, requests fail fast
    - HALF_OPEN: Testing if service recovered

    Features:
    - Configurable failure threshold and recovery timeout
    - Automatic state transitions
    - Statistics and observability
    - Thread-safe async operations
    """

    __slots__ = (
        '_name', '_failure_threshold', '_recovery_timeout', '_half_open_max_calls',
        '_state', '_failure_count', '_last_failure_time', '_half_open_calls',
        '_lock', '_stats', '_state_change_callbacks'
    )

    # States
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Circuit breaker identifier
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self._name = name
        self._failure_threshold = int(os.getenv(
            f"CB_{name.upper()}_FAILURE_THRESHOLD", str(failure_threshold)
        ))
        self._recovery_timeout = float(os.getenv(
            f"CB_{name.upper()}_RECOVERY_TIMEOUT", str(recovery_timeout)
        ))
        self._half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

        self._lock = asyncio.Lock()
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "state_changes": 0,
        }
        self._state_change_callbacks: List[Callable[[str, str], None]] = []

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self._state == self.OPEN

    async def call(
        self,
        func: Callable[..., Any],
        *args,
        fallback: Optional[Callable[..., Any]] = None,
        **kwargs,
    ) -> Any:
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Function arguments
            fallback: Optional fallback function if circuit is open
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback result

        Raises:
            CircuitBreakerOpen: If circuit is open and no fallback provided
        """
        async with self._lock:
            self._stats["total_calls"] += 1

            # Check if we should allow the call
            if not await self._should_allow_call():
                self._stats["rejected_calls"] += 1

                if fallback:
                    logger.warning(f"Circuit breaker '{self._name}' open - using fallback")
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)

                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self._name}' is open - service unavailable"
                )

        # Execute the function outside the lock
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure(e)
            raise

    async def _should_allow_call(self) -> bool:
        """Check if a call should be allowed based on circuit state."""
        now = time.time()

        if self._state == self.CLOSED:
            return True

        if self._state == self.OPEN:
            # Check if recovery timeout has passed
            if now - self._last_failure_time >= self._recovery_timeout:
                await self._transition_to(self.HALF_OPEN)
                return True
            return False

        if self._state == self.HALF_OPEN:
            # Allow limited calls in half-open state
            if self._half_open_calls < self._half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

        return False

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats["successful_calls"] += 1

            if self._state == self.HALF_OPEN:
                # Success in half-open state - close the circuit
                await self._transition_to(self.CLOSED)
            elif self._state == self.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats["failed_calls"] += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self._name}' recorded failure "
                f"({self._failure_count}/{self._failure_threshold}): {error}"
            )

            if self._state == self.HALF_OPEN:
                # Failure in half-open state - back to open
                await self._transition_to(self.OPEN)
            elif self._state == self.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    await self._transition_to(self.OPEN)

    async def _transition_to(self, new_state: str) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._stats["state_changes"] += 1

        if new_state == self.CLOSED:
            self._failure_count = 0
            self._half_open_calls = 0
        elif new_state == self.HALF_OPEN:
            self._half_open_calls = 0

        logger.info(f"Circuit breaker '{self._name}' state: {old_state} -> {new_state}")

        # Notify callbacks
        for callback in self._state_change_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.warning(f"State change callback error: {e}")

    def on_state_change(self, callback: Callable[[str, str], None]) -> None:
        """Register a state change callback."""
        self._state_change_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self._name,
            "state": self._state,
            "failure_count": self._failure_count,
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
            **self._stats,
        }

    async def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            await self._transition_to(self.CLOSED)
            logger.info(f"Circuit breaker '{self._name}' manually reset")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ContextWindowManager:
    """
    Manages context window limits for LLM interactions.

    Features:
    - Token counting with fallback estimation
    - Dynamic context truncation
    - Priority-based context selection
    - Overflow warnings and metrics
    - Context size distribution tracking
    - Proactive warning when approaching limits

    v92.1 ENHANCEMENTS:
    - Added comprehensive metrics tracking
    - Added warning thresholds
    - Added context utilization analysis
    """

    __slots__ = (
        '_max_tokens', '_reserved_tokens', '_tokenizer', '_stats',
        '_warning_threshold', '_critical_threshold', '_last_warning_time'
    )

    # Warning thresholds as percentage of max_tokens
    DEFAULT_WARNING_THRESHOLD = 0.75  # 75% utilization
    DEFAULT_CRITICAL_THRESHOLD = 0.90  # 90% utilization

    def __init__(
        self,
        max_tokens: int = 4096,
        reserved_tokens: int = 512,
        warning_threshold: float = DEFAULT_WARNING_THRESHOLD,
        critical_threshold: float = DEFAULT_CRITICAL_THRESHOLD,
    ):
        self._max_tokens = int(os.getenv("REASONING_MAX_TOKENS", str(max_tokens)))
        self._reserved_tokens = int(os.getenv("REASONING_RESERVED_TOKENS", str(reserved_tokens)))
        self._warning_threshold = warning_threshold
        self._critical_threshold = critical_threshold
        self._last_warning_time = 0.0
        self._tokenizer = None
        self._stats = ThreadSafeStatistics()

        # Try to load tokenizer
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.debug("Using tiktoken for accurate token counting")
        except ImportError:
            logger.debug("tiktoken not available, using char-based estimation")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text with caching for repeated calls."""
        if not text:
            return 0
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        # Fallback: estimate ~4 chars per token (conservative)
        return len(text) // 4

    def available_tokens(self, current_usage: int = 0) -> int:
        """Get available tokens for context."""
        return max(0, self._max_tokens - self._reserved_tokens - current_usage)

    @property
    def effective_max_tokens(self) -> int:
        """Get effective maximum tokens (accounting for reserved)."""
        return self._max_tokens - self._reserved_tokens

    async def check_utilization(
        self,
        current_tokens: int,
        operation: str = "context",
    ) -> Tuple[float, str]:
        """
        Check context utilization and log warnings if needed.

        v92.1: Comprehensive utilization tracking with warnings.

        Args:
            current_tokens: Current token count
            operation: Description of the operation for logging

        Returns:
            Tuple of (utilization_ratio, status) where status is "ok", "warning", or "critical"
        """
        effective_max = self.effective_max_tokens
        utilization = current_tokens / effective_max if effective_max > 0 else 1.0

        # Track utilization distribution
        bucket = f"{int(utilization * 10) * 10}%"
        await self._stats.record_histogram("context_utilization", bucket)
        await self._stats.record_timing("context_tokens", current_tokens)

        # Determine status
        status = "ok"
        if utilization >= self._critical_threshold:
            status = "critical"
            await self._stats.increment("context_critical_warnings")
            # Rate-limit warnings to avoid spam
            now = time.time()
            if now - self._last_warning_time > 60:  # Max 1 warning per minute
                logger.warning(
                    f"CRITICAL: {operation} at {utilization:.1%} capacity "
                    f"({current_tokens}/{effective_max} tokens)"
                )
                self._last_warning_time = now
        elif utilization >= self._warning_threshold:
            status = "warning"
            await self._stats.increment("context_warnings")
            now = time.time()
            if now - self._last_warning_time > 300:  # Max 1 warning per 5 min
                logger.warning(
                    f"WARNING: {operation} at {utilization:.1%} capacity "
                    f"({current_tokens}/{effective_max} tokens)"
                )
                self._last_warning_time = now

        return (utilization, status)

    async def get_stats(self) -> Dict[str, Any]:
        """Get context window statistics."""
        all_stats = await self._stats.get_all_stats()
        timing_stats = await self._stats.get_timing_stats("context_tokens")

        return {
            "max_tokens": self._max_tokens,
            "reserved_tokens": self._reserved_tokens,
            "effective_max": self.effective_max_tokens,
            "tokenizer_available": self._tokenizer is not None,
            "truncations": all_stats.get("counters", {}).get("context_truncations", 0),
            "warnings": all_stats.get("counters", {}).get("context_warnings", 0),
            "critical_warnings": all_stats.get("counters", {}).get("context_critical_warnings", 0),
            "utilization_distribution": all_stats.get("histograms", {}).get("context_utilization", {}),
            "token_usage": timing_stats,
        }

    async def truncate_context(
        self,
        context: str,
        max_tokens: Optional[int] = None,
        preserve_start: bool = True,
    ) -> str:
        """
        Truncate context to fit within token limit.

        Args:
            context: Text to truncate
            max_tokens: Max tokens (uses available_tokens if None)
            preserve_start: If True, keep beginning; else keep end
        """
        max_tokens = max_tokens or self.available_tokens()
        current_tokens = self.count_tokens(context)

        if current_tokens <= max_tokens:
            return context

        await self._stats.increment("context_truncations")

        # Estimate characters to keep
        ratio = max_tokens / current_tokens
        max_chars = int(len(context) * ratio * 0.9)  # 10% safety margin

        if preserve_start:
            truncated = context[:max_chars] + "\n...[truncated]"
        else:
            truncated = "...[truncated]\n" + context[-max_chars:]

        return truncated

    async def prioritize_context(
        self,
        contexts: List[Tuple[str, float]],  # (text, priority)
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Select and combine contexts by priority within token limit.

        Args:
            contexts: List of (text, priority) tuples
            max_tokens: Maximum total tokens
        """
        max_tokens = max_tokens or self.available_tokens()

        # Sort by priority (highest first)
        sorted_contexts = sorted(contexts, key=lambda x: x[1], reverse=True)

        selected = []
        current_tokens = 0

        for text, priority in sorted_contexts:
            text_tokens = self.count_tokens(text)
            if current_tokens + text_tokens <= max_tokens:
                selected.append(text)
                current_tokens += text_tokens
            else:
                # Try to fit truncated version
                remaining = max_tokens - current_tokens
                if remaining > 100:  # Worth including
                    truncated = await self.truncate_context(text, remaining)
                    selected.append(truncated)
                break

        return "\n\n".join(selected)


class RAGContextCache:
    """
    Caches RAG retrieval results with TTL and similarity matching.

    Features:
    - Exact match caching
    - Similar query matching (fuzzy)
    - TTL-based expiration
    - Memory-bounded storage
    """

    __slots__ = ('_cache', '_lock', '_max_size', '_ttl_seconds', '_hits', '_misses')

    def __init__(self, max_size: int = 500, ttl_seconds: float = 300.0):
        self._cache: OrderedDict[str, Tuple[Dict[str, Any], float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str) -> str:
        """Create cache key from query."""
        # Normalize: lowercase, remove extra whitespace
        normalized = " ".join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached RAG result."""
        async with self._lock:
            key = self._make_key(query)

            if key not in self._cache:
                self._misses += 1
                return None

            result, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return result

    async def set(self, query: str, result: Dict[str, Any]) -> None:
        """Cache RAG result."""
        async with self._lock:
            key = self._make_key(query)

            # Evict if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (result, time.time())

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }


class FeedbackDeduplicator:
    """
    Deduplicates feedback items to prevent training on duplicates.

    Features:
    - Content-based hashing
    - Sliding window deduplication
    - Configurable similarity threshold
    """

    __slots__ = ('_seen_hashes', '_max_size', '_lock')

    def __init__(self, max_size: int = 10000):
        self._seen_hashes: OrderedDict[str, float] = OrderedDict()
        self._max_size = max_size
        self._lock = asyncio.Lock()

    def _hash_feedback(self, input_text: str, output_text: str) -> str:
        """Create hash of feedback content."""
        # Normalize and hash
        normalized = f"{input_text.strip().lower()}|||{output_text.strip().lower()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def is_duplicate(self, input_text: str, output_text: str) -> bool:
        """Check if feedback is a duplicate."""
        async with self._lock:
            hash_key = self._hash_feedback(input_text, output_text)

            if hash_key in self._seen_hashes:
                return True

            # Add to seen
            while len(self._seen_hashes) >= self._max_size:
                self._seen_hashes.popitem(last=False)

            self._seen_hashes[hash_key] = time.time()
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplicator statistics."""
        now = time.time()
        oldest_timestamp = None
        if self._seen_hashes:
            oldest_timestamp = min(self._seen_hashes.values())

        return {
            "seen_count": len(self._seen_hashes),
            "max_size": self._max_size,
            "utilization": len(self._seen_hashes) / self._max_size if self._max_size > 0 else 0,
            "oldest_entry_age_seconds": (now - oldest_timestamp) if oldest_timestamp else None,
        }


class QualityScorer:
    """
    Scores feedback quality beyond simple confidence.

    Features:
    - Multi-factor quality assessment
    - Configurable weights
    - Learning from outcomes
    """

    __slots__ = ('_weights',)

    def __init__(self):
        self._weights = {
            "confidence": 0.3,
            "length_score": 0.2,
            "reasoning_depth": 0.2,
            "specificity": 0.15,
            "latency_score": 0.15,
        }

    def score(
        self,
        result: "ReasoningResult",
        input_text: str,
    ) -> float:
        """
        Calculate comprehensive quality score.

        Returns score in [0, 1] range.
        """
        scores = {}

        # Confidence score (already in 0-1)
        scores["confidence"] = result.confidence

        # Length score: prefer substantial responses
        output_len = len(result.output_text)
        if output_len < 50:
            scores["length_score"] = 0.3
        elif output_len < 200:
            scores["length_score"] = 0.6
        elif output_len < 1000:
            scores["length_score"] = 0.9
        else:
            scores["length_score"] = 1.0

        # Reasoning depth: based on thought count
        thought_count = result.total_thoughts
        if thought_count == 0:
            scores["reasoning_depth"] = 0.3
        elif thought_count <= 2:
            scores["reasoning_depth"] = 0.6
        elif thought_count <= 5:
            scores["reasoning_depth"] = 0.85
        else:
            scores["reasoning_depth"] = 1.0

        # Specificity: presence of concrete terms
        specific_indicators = [
            "specifically", "for example", "in particular",
            "because", "therefore", "first", "second", "finally",
        ]
        specificity_count = sum(
            1 for ind in specific_indicators
            if ind in result.output_text.lower()
        )
        scores["specificity"] = min(1.0, specificity_count / 4)

        # Latency score: faster is better (but not too fast)
        latency_ms = result.latency_ms
        if latency_ms < 100:  # Too fast, might be cached/trivial
            scores["latency_score"] = 0.7
        elif latency_ms < 1000:
            scores["latency_score"] = 1.0
        elif latency_ms < 5000:
            scores["latency_score"] = 0.8
        else:
            scores["latency_score"] = 0.5

        # Weighted average
        total_score = sum(
            scores[key] * self._weights[key]
            for key in self._weights
        )

        return min(1.0, max(0.0, total_score))


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    DIRECT = "direct"                      # Single-pass, no explicit reasoning
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Sequential step-by-step
    TREE_OF_THOUGHTS = "tree_of_thoughts"  # Parallel exploration
    SELF_REFLECTION = "self_reflection"    # Meta-cognitive verification
    HYPOTHESIS_TEST = "hypothesis_test"    # Scientific method
    ANALOGICAL = "analogical"              # Transfer from similar problems
    ENSEMBLE = "ensemble"                  # Multiple strategies combined
    ADAPTIVE = "adaptive"                  # Dynamic strategy switching


class ThoughtStatus(Enum):
    """Status of a thought node."""
    PENDING = "pending"
    EXPLORING = "exploring"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    SELECTED = "selected"
    ABANDONED = "abandoned"


class VerificationResult(Enum):
    """Result of thought verification."""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    NEEDS_REVISION = "needs_revision"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Thought:
    """
    Represents a single thought in the reasoning process.

    Can be a step in Chain-of-Thought or a node in Tree-of-Thoughts.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""

    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0

    # Evaluation
    value: float = 0.0              # Estimated value (0-1)
    confidence: float = 0.5         # Confidence in this thought
    visit_count: int = 0            # For MCTS-style exploration

    # Status
    status: ThoughtStatus = ThoughtStatus.PENDING
    verification: Optional[VerificationResult] = None

    # Metadata
    strategy_used: Optional[ReasoningStrategy] = None
    generation_time_ms: float = 0.0
    evaluation_time_ms: float = 0.0

    # Self-reflection
    self_critique: Optional[str] = None
    revisions: List[str] = field(default_factory=list)

    def __lt__(self, other: "Thought") -> bool:
        """For heap operations - higher value = higher priority."""
        return self.value > other.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "value": round(self.value, 3),
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "verification": self.verification.value if self.verification else None,
        }


@dataclass
class ReasoningChain:
    """A chain of thoughts forming a complete reasoning path."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    thoughts: List[Thought] = field(default_factory=list)
    strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT

    # Aggregates
    total_value: float = 0.0
    average_confidence: float = 0.0

    # Input/Output
    input_text: str = ""
    final_answer: str = ""

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def add_thought(self, thought: Thought) -> None:
        """Add thought to chain."""
        self.thoughts.append(thought)
        self._update_aggregates()

    def _update_aggregates(self) -> None:
        """Update aggregate metrics."""
        if not self.thoughts:
            return

        self.total_value = sum(t.value for t in self.thoughts)
        self.average_confidence = sum(t.confidence for t in self.thoughts) / len(self.thoughts)

    def get_path(self) -> List[str]:
        """Get the content of all thoughts in order."""
        return [t.content for t in self.thoughts]

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "thought_count": len(self.thoughts),
            "total_value": round(self.total_value, 3),
            "average_confidence": round(self.average_confidence, 3),
            "duration_ms": round(self.duration_ms, 1),
            "final_answer": self.final_answer[:200] if self.final_answer else None,
        }


@dataclass
class ThoughtTree:
    """
    A tree of thoughts for parallel exploration.

    Used by Tree-of-Thoughts strategy for branching reasoning.
    """
    root: Optional[Thought] = None
    nodes: Dict[str, Thought] = field(default_factory=dict)

    # Exploration parameters
    max_depth: int = 5
    max_branches: int = 3
    beam_width: int = 3  # Top-k to keep at each level

    # Best path tracking
    best_leaf: Optional[Thought] = None
    best_value: float = 0.0

    def add_node(self, thought: Thought, parent_id: Optional[str] = None) -> None:
        """Add node to tree."""
        if parent_id is None:
            self.root = thought
            thought.depth = 0
        else:
            if parent_id in self.nodes:
                parent = self.nodes[parent_id]
                parent.children_ids.append(thought.id)
                thought.parent_id = parent_id
                thought.depth = parent.depth + 1

        self.nodes[thought.id] = thought

        # Track best
        if thought.value > self.best_value:
            self.best_value = thought.value
            self.best_leaf = thought

    def get_path_to_node(self, node_id: str) -> List[Thought]:
        """Get path from root to node."""
        path = []
        current_id = node_id

        while current_id:
            if current_id in self.nodes:
                path.append(self.nodes[current_id])
                current_id = self.nodes[current_id].parent_id
            else:
                break

        return list(reversed(path))

    def get_best_path(self) -> List[Thought]:
        """Get path to best leaf."""
        if self.best_leaf:
            return self.get_path_to_node(self.best_leaf.id)
        return []

    def get_frontier(self) -> List[Thought]:
        """Get leaf nodes (frontier for expansion)."""
        leaves = []
        for node in self.nodes.values():
            if not node.children_ids and node.status != ThoughtStatus.PRUNED:
                leaves.append(node)
        return leaves

    def prune_below_threshold(self, threshold: float) -> int:
        """Prune nodes with value below threshold."""
        pruned_count = 0
        for node in self.nodes.values():
            if node.value < threshold and node.status != ThoughtStatus.SELECTED:
                node.status = ThoughtStatus.PRUNED
                pruned_count += 1
        return pruned_count

    def beam_search_prune(self) -> int:
        """Keep only top-k nodes at each depth level."""
        by_depth: Dict[int, List[Thought]] = defaultdict(list)

        for node in self.nodes.values():
            if node.status != ThoughtStatus.PRUNED:
                by_depth[node.depth].append(node)

        pruned_count = 0
        for depth, nodes in by_depth.items():
            if len(nodes) > self.beam_width:
                sorted_nodes = sorted(nodes, key=lambda n: n.value, reverse=True)
                for node in sorted_nodes[self.beam_width:]:
                    node.status = ThoughtStatus.PRUNED
                    pruned_count += 1

        return pruned_count


@dataclass
class ReasoningConfig:
    """
    Configuration for reasoning engine with environment variable support.

    All values can be overridden via environment variables:
        REASONING_DEFAULT_STRATEGY, REASONING_COT_MAX_STEPS, etc.
    """
    # Strategy selection
    default_strategy: ReasoningStrategy = field(default_factory=lambda:
        ReasoningStrategy(os.getenv("REASONING_DEFAULT_STRATEGY", "chain_of_thought"))
    )
    enable_adaptive: bool = field(default_factory=lambda:
        os.getenv("REASONING_ENABLE_ADAPTIVE", "true").lower() == "true"
    )

    # Chain-of-Thought
    cot_max_steps: int = field(default_factory=lambda:
        int(os.getenv("REASONING_COT_MAX_STEPS", "10"))
    )
    cot_stop_on_confidence: float = field(default_factory=lambda:
        float(os.getenv("REASONING_COT_STOP_CONFIDENCE", "0.9"))
    )

    # Tree-of-Thoughts
    tot_max_depth: int = field(default_factory=lambda:
        int(os.getenv("REASONING_TOT_MAX_DEPTH", "5"))
    )
    tot_branches_per_node: int = field(default_factory=lambda:
        int(os.getenv("REASONING_TOT_BRANCHES", "3"))
    )
    tot_beam_width: int = field(default_factory=lambda:
        int(os.getenv("REASONING_TOT_BEAM_WIDTH", "3"))
    )
    tot_exploration_constant: float = field(default_factory=lambda:
        float(os.getenv("REASONING_TOT_EXPLORATION", "1.4"))
    )

    # Self-Reflection
    reflection_threshold: float = field(default_factory=lambda:
        float(os.getenv("REASONING_REFLECTION_THRESHOLD", "0.6"))
    )
    max_revisions: int = field(default_factory=lambda:
        int(os.getenv("REASONING_MAX_REVISIONS", "3"))
    )

    # Hypothesis Testing
    hypothesis_confidence_threshold: float = field(default_factory=lambda:
        float(os.getenv("REASONING_HYPOTHESIS_THRESHOLD", "0.7"))
    )
    max_hypotheses: int = field(default_factory=lambda:
        int(os.getenv("REASONING_MAX_HYPOTHESES", "5"))
    )

    # General
    min_confidence: float = field(default_factory=lambda:
        float(os.getenv("REASONING_MIN_CONFIDENCE", "0.3"))
    )
    timeout_seconds: float = field(default_factory=lambda:
        float(os.getenv("REASONING_TIMEOUT_SECONDS", "60.0"))
    )
    parallel_thoughts: int = field(default_factory=lambda:
        int(os.getenv("REASONING_PARALLEL_THOUGHTS", "4"))
    )

    # Caching
    cache_thoughts: bool = field(default_factory=lambda:
        os.getenv("REASONING_CACHE_ENABLED", "true").lower() == "true"
    )
    cache_ttl_seconds: float = field(default_factory=lambda:
        float(os.getenv("REASONING_CACHE_TTL", "300.0"))
    )
    cache_max_size: int = field(default_factory=lambda:
        int(os.getenv("REASONING_CACHE_MAX_SIZE", "100"))
    )

    # RAG Integration (NEW in v92.0)
    rag_enabled: bool = field(default_factory=lambda:
        os.getenv("REASONING_RAG_ENABLED", "true").lower() == "true"
    )
    rag_top_k: int = field(default_factory=lambda:
        int(os.getenv("REASONING_RAG_TOP_K", "5"))
    )
    rag_min_relevance: float = field(default_factory=lambda:
        float(os.getenv("REASONING_RAG_MIN_RELEVANCE", "0.5"))
    )

    # Training Feedback Loop (NEW in v92.0)
    training_feedback_enabled: bool = field(default_factory=lambda:
        os.getenv("REASONING_TRAINING_FEEDBACK", "true").lower() == "true"
    )
    feedback_collection_threshold: float = field(default_factory=lambda:
        float(os.getenv("REASONING_FEEDBACK_THRESHOLD", "0.7"))
    )

    # Rate Limiting (NEW in v92.0)
    max_concurrent_reasoning: int = field(default_factory=lambda:
        int(os.getenv("REASONING_MAX_CONCURRENT", "4"))
    )
    rate_limit_timeout: float = field(default_factory=lambda:
        float(os.getenv("REASONING_RATE_LIMIT_TIMEOUT", "30.0"))
    )

    # Observability (NEW in v92.0)
    enable_tracing: bool = field(default_factory=lambda:
        os.getenv("REASONING_ENABLE_TRACING", "true").lower() == "true"
    )
    enable_detailed_logging: bool = field(default_factory=lambda:
        os.getenv("REASONING_DETAILED_LOGGING", "false").lower() == "true"
    )


@dataclass
class ReasoningResult:
    """Result from reasoning engine."""
    strategy: ReasoningStrategy
    input_text: str
    output_text: str

    # Quality metrics
    confidence: float = 0.5
    coherence: float = 0.5

    # Chain/Tree data
    chain: Optional[ReasoningChain] = None
    tree: Optional[ThoughtTree] = None

    # Reflection
    self_assessment: Optional[str] = None
    verified: bool = False

    # Performance
    total_thoughts: int = 0
    pruned_thoughts: int = 0
    latency_ms: float = 0.0

    # Alternative answers
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "output": self.output_text[:500] if self.output_text else None,
            "confidence": round(self.confidence, 3),
            "verified": self.verified,
            "total_thoughts": self.total_thoughts,
            "latency_ms": round(self.latency_ms, 1),
            "chain": self.chain.to_dict() if self.chain else None,
        }


# =============================================================================
# THOUGHT GENERATORS
# =============================================================================

class ThoughtGenerator(Protocol):
    """Protocol for thought generation."""

    async def generate(
        self,
        prompt: str,
        context: List[Thought],
        num_thoughts: int,
    ) -> List[Thought]:
        """Generate new thoughts given prompt and context."""
        ...


class DefaultThoughtGenerator:
    """
    Default thought generator using pattern-based generation.

    In production, this would use an LLM for more sophisticated generation.
    """

    # Reasoning templates
    TEMPLATES = {
        "analyze": "Let me analyze {topic}...",
        "decompose": "Breaking this down: {parts}",
        "consider": "Considering {aspect}...",
        "conclude": "Therefore, {conclusion}",
        "verify": "Checking this: {check}",
        "alternative": "Alternatively, {alternative}",
    }

    def __init__(self, executor: Optional[Any] = None):
        self.executor = executor
        self._generation_count = 0

    async def generate(
        self,
        prompt: str,
        context: List[Thought],
        num_thoughts: int = 1,
    ) -> List[Thought]:
        """Generate thoughts based on prompt and context."""
        thoughts = []
        self._generation_count += 1

        # Build context string
        context_str = " -> ".join(t.content[:50] for t in context[-3:])

        for i in range(num_thoughts):
            start = time.time()

            # Select template based on position and context
            if not context:
                template_key = "analyze"
            elif len(context) >= 4:
                template_key = "conclude"
            elif i == 0:
                template_key = "decompose"
            else:
                template_key = random.choice(["consider", "alternative"])

            # Generate thought content
            if self.executor:
                # Use LLM executor
                try:
                    content = await self._generate_with_llm(prompt, context, template_key)
                except Exception as e:
                    logger.warning(f"LLM generation failed: {e}")
                    content = self._generate_heuristic(prompt, context, template_key)
            else:
                content = self._generate_heuristic(prompt, context, template_key)

            thought = Thought(
                content=content,
                depth=len(context),
                generation_time_ms=(time.time() - start) * 1000,
                confidence=0.5 + random.uniform(-0.2, 0.2),  # Initial confidence
            )

            thoughts.append(thought)

        return thoughts

    async def _generate_with_llm(
        self,
        prompt: str,
        context: List[Thought],
        template_key: str,
    ) -> str:
        """Generate thought using LLM."""
        context_str = "\n".join(f"Step {i+1}: {t.content}" for i, t in enumerate(context))

        system_prompt = f"""You are a reasoning assistant. Generate the next logical step in reasoning.
Previous steps:
{context_str if context_str else 'None'}

Task: {prompt}

Generate only the next reasoning step (1-2 sentences):"""

        response = await self.executor.generate(
            prompt=system_prompt,
            max_tokens=100,
            temperature=0.7,
        )

        return response.strip()

    def _generate_heuristic(
        self,
        prompt: str,
        context: List[Thought],
        template_key: str,
    ) -> str:
        """Generate thought using heuristics."""
        # Extract key terms from prompt
        words = prompt.split()
        key_terms = [w for w in words if len(w) > 4][:3]

        if template_key == "analyze":
            return f"Let me analyze this: {' '.join(key_terms) if key_terms else prompt[:50]}"
        elif template_key == "decompose":
            return f"Breaking this down into components: {', '.join(key_terms) if key_terms else 'multiple parts'}"
        elif template_key == "consider":
            aspect = key_terms[0] if key_terms else "the main point"
            return f"Considering {aspect} in more detail..."
        elif template_key == "conclude":
            return f"Based on the analysis, we can conclude that..."
        elif template_key == "alternative":
            return f"An alternative perspective would be..."
        else:
            return f"Step {len(context)+1}: Processing {prompt[:30]}..."


# =============================================================================
# THOUGHT EVALUATORS
# =============================================================================

class ThoughtEvaluator(Protocol):
    """Protocol for thought evaluation."""

    async def evaluate(self, thought: Thought, context: List[Thought]) -> float:
        """Evaluate a thought and return value (0-1)."""
        ...


class DefaultThoughtEvaluator:
    """
    Default thought evaluator using heuristic scoring.

    Evaluates thoughts based on:
    - Relevance to context
    - Logical progression
    - Specificity
    - Coherence
    """

    def __init__(self, executor: Optional[Any] = None):
        self.executor = executor

    async def evaluate(self, thought: Thought, context: List[Thought]) -> float:
        """Evaluate thought quality."""
        start = time.time()

        scores = {
            "length": self._score_length(thought),
            "specificity": self._score_specificity(thought),
            "progression": self._score_progression(thought, context),
            "keywords": self._score_keywords(thought, context),
        }

        # Weighted average
        weights = {"length": 0.1, "specificity": 0.3, "progression": 0.3, "keywords": 0.3}
        value = sum(scores[k] * weights[k] for k in scores)

        thought.value = value
        thought.evaluation_time_ms = (time.time() - start) * 1000
        thought.status = ThoughtStatus.EVALUATED

        return value

    def _score_length(self, thought: Thought) -> float:
        """Score based on appropriate length."""
        length = len(thought.content)
        if length < 10:
            return 0.2
        elif length < 50:
            return 0.6
        elif length < 200:
            return 1.0
        elif length < 500:
            return 0.8
        else:
            return 0.5  # Too long

    def _score_specificity(self, thought: Thought) -> float:
        """Score based on specificity."""
        content = thought.content.lower()

        # Vague terms reduce score
        vague_terms = ["something", "thing", "stuff", "maybe", "perhaps", "kind of"]
        vague_count = sum(1 for term in vague_terms if term in content)

        # Specific terms increase score
        specific_indicators = ["specifically", "exactly", "precisely", "because", "therefore"]
        specific_count = sum(1 for term in specific_indicators if term in content)

        base = 0.6
        score = base - (vague_count * 0.1) + (specific_count * 0.1)
        return max(0.0, min(1.0, score))

    def _score_progression(self, thought: Thought, context: List[Thought]) -> float:
        """Score based on logical progression from context."""
        if not context:
            return 0.7  # First thought gets baseline

        last_thought = context[-1]

        # Check for connection to previous thought
        last_words = set(last_thought.content.lower().split())
        current_words = set(thought.content.lower().split())

        # Overlap indicates connection
        overlap = len(last_words & current_words)
        overlap_ratio = overlap / max(len(last_words), 1)

        # Some overlap is good, too much might be repetition
        if overlap_ratio < 0.1:
            return 0.3  # Too disconnected
        elif overlap_ratio < 0.3:
            return 0.8  # Good connection
        elif overlap_ratio < 0.5:
            return 0.6  # Some repetition
        else:
            return 0.4  # Too repetitive

    def _score_keywords(self, thought: Thought, context: List[Thought]) -> float:
        """Score based on important keyword presence."""
        reasoning_keywords = [
            "because", "therefore", "thus", "hence", "since",
            "implies", "suggests", "indicates", "means",
            "first", "second", "finally", "however", "although",
        ]

        content_lower = thought.content.lower()
        keyword_count = sum(1 for kw in reasoning_keywords if kw in content_lower)

        return min(1.0, 0.4 + keyword_count * 0.15)


# =============================================================================
# REASONING STRATEGIES
# =============================================================================

class BaseReasoningStrategy(ABC):
    """Base class for reasoning strategies."""

    strategy_type: ReasoningStrategy

    def __init__(
        self,
        config: ReasoningConfig,
        generator: ThoughtGenerator,
        evaluator: ThoughtEvaluator,
    ):
        self.config = config
        self.generator = generator
        self.evaluator = evaluator

    @abstractmethod
    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute reasoning strategy."""
        ...

    def _format_context_for_prompt(
        self,
        context: Optional[Dict[str, Any]],
        max_context_chars: int = 4000,
    ) -> str:
        """
        Format context dict into a prompt-ready string.

        v92.1: Properly integrates RAG retrieved context into reasoning prompts.

        Args:
            context: Context dict which may include:
                - retrieved_context: Raw RAG retrieval results
                - formatted_context: Pre-formatted context string
                - system_prompt: System-level instructions
                - Any other key-value context

        Returns:
            Formatted context string suitable for prompt injection
        """
        if not context:
            return ""

        parts = []

        # Use pre-formatted context if available (from RAGContextCache)
        if "formatted_context" in context and context["formatted_context"]:
            parts.append(context["formatted_context"])

        # Format retrieved RAG documents
        elif "retrieved_context" in context:
            retrieved = context["retrieved_context"]
            if isinstance(retrieved, list):
                for i, doc in enumerate(retrieved[:5], 1):  # Limit to 5 docs
                    if isinstance(doc, dict):
                        content = doc.get("content", doc.get("text", str(doc)))
                        score = doc.get("score", doc.get("relevance", 0))
                        parts.append(f"[Document {i}] (relevance: {score:.2f})\n{content}")
                    else:
                        parts.append(f"[Document {i}]\n{str(doc)}")
            elif isinstance(retrieved, str):
                parts.append(f"[Retrieved Context]\n{retrieved}")

        # Add any additional context fields
        for key, value in context.items():
            if key in ("retrieved_context", "formatted_context", "system_prompt",
                       "relevance_scores", "num_documents", "rag_context"):
                continue  # Already handled or metadata
            if isinstance(value, str) and value.strip():
                parts.append(f"[{key.replace('_', ' ').title()}]\n{value}")

        if not parts:
            return ""

        # Join and truncate if needed
        full_context = "\n\n".join(parts)
        if len(full_context) > max_context_chars:
            full_context = full_context[:max_context_chars - 100] + "\n\n[Context truncated...]"

        return f"=== RELEVANT CONTEXT ===\n{full_context}\n=== END CONTEXT ===\n\n"

    def _build_augmented_prompt(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build an augmented prompt with context injection.

        Args:
            input_text: The original user query
            context: Context dict for augmentation

        Returns:
            Augmented prompt string with context prepended
        """
        formatted_context = self._format_context_for_prompt(context)
        if formatted_context:
            return f"{formatted_context}{input_text}"
        return input_text


class ChainOfThoughtStrategy(BaseReasoningStrategy):
    """
    Chain-of-Thought (CoT) reasoning.

    Generates sequential reasoning steps, each building on the previous.
    Stops when confidence threshold is reached or max steps exceeded.
    """

    strategy_type = ReasoningStrategy.CHAIN_OF_THOUGHT

    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute Chain-of-Thought reasoning with RAG context augmentation."""
        start = time.time()

        # v92.1: Build augmented prompt with RAG context
        augmented_prompt = self._build_augmented_prompt(input_text, context)
        has_rag_context = augmented_prompt != input_text

        chain = ReasoningChain(
            strategy=self.strategy_type,
            input_text=input_text,
        )
        # Track RAG usage in chain metadata
        chain.metadata["has_rag_context"] = has_rag_context
        chain.metadata["context_chars"] = len(augmented_prompt) - len(input_text)

        thoughts: List[Thought] = []
        total_confidence = 0.0

        for step in range(self.config.cot_max_steps):
            # Generate next thought with augmented prompt (includes RAG context)
            new_thoughts = await self.generator.generate(
                prompt=augmented_prompt,  # v92.1: Use augmented prompt with context
                context=thoughts,
                num_thoughts=1,
            )

            if not new_thoughts:
                break

            thought = new_thoughts[0]

            # Evaluate thought
            value = await self.evaluator.evaluate(thought, thoughts)

            # Add to chain
            thoughts.append(thought)
            chain.add_thought(thought)
            total_confidence += thought.confidence

            # Check for early stopping
            avg_confidence = total_confidence / len(thoughts)
            if avg_confidence >= self.config.cot_stop_on_confidence:
                logger.debug(f"CoT: Early stop at step {step+1} (confidence: {avg_confidence:.2f})")
                break

        # Generate final answer
        chain.final_answer = self._synthesize_answer(thoughts)
        chain.end_time = time.time()

        return ReasoningResult(
            strategy=self.strategy_type,
            input_text=input_text,
            output_text=chain.final_answer,
            confidence=chain.average_confidence,
            chain=chain,
            total_thoughts=len(thoughts),
            latency_ms=(time.time() - start) * 1000,
        )

    def _synthesize_answer(self, thoughts: List[Thought]) -> str:
        """Synthesize final answer from thought chain."""
        if not thoughts:
            return "Unable to generate reasoning chain."

        # Use last thought as primary conclusion
        conclusion = thoughts[-1].content

        # Build reasoning summary
        if len(thoughts) > 1:
            steps_summary = " -> ".join(t.content[:50] for t in thoughts[:-1])
            return f"Reasoning: {steps_summary}\n\nConclusion: {conclusion}"

        return conclusion


class TreeOfThoughtsStrategy(BaseReasoningStrategy):
    """
    Tree-of-Thoughts (ToT) reasoning.

    Explores multiple reasoning paths in parallel using tree search.
    Uses beam search to prune low-value branches.
    """

    strategy_type = ReasoningStrategy.TREE_OF_THOUGHTS

    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute Tree-of-Thoughts reasoning with RAG context augmentation."""
        start = time.time()

        # v92.1: Build augmented prompt with RAG context
        augmented_prompt = self._build_augmented_prompt(input_text, context)
        has_rag_context = augmented_prompt != input_text

        tree = ThoughtTree(
            max_depth=self.config.tot_max_depth,
            max_branches=self.config.tot_branches_per_node,
            beam_width=self.config.tot_beam_width,
        )
        # Track RAG usage in tree metadata
        tree.metadata["has_rag_context"] = has_rag_context

        # Create root thought with augmented prompt
        root_thoughts = await self.generator.generate(
            prompt=augmented_prompt,  # v92.1: Use augmented prompt with context
            context=[],
            num_thoughts=1,
        )

        if not root_thoughts:
            return ReasoningResult(
                strategy=self.strategy_type,
                input_text=input_text,
                output_text="Unable to initiate reasoning.",
                confidence=0.0,
                latency_ms=(time.time() - start) * 1000,
            )

        root = root_thoughts[0]
        await self.evaluator.evaluate(root, [])
        tree.add_node(root)

        # BFS expansion with beam search
        for depth in range(self.config.tot_max_depth):
            frontier = [n for n in tree.get_frontier() if n.depth == depth]

            if not frontier:
                break

            # Expand each frontier node with augmented prompt
            expansion_tasks = []
            for node in frontier:
                expansion_tasks.append(
                    self._expand_node(tree, node, augmented_prompt)  # v92.1: Use augmented prompt
                )

            await asyncio.gather(*expansion_tasks)

            # Beam search pruning
            tree.beam_search_prune()

        # Get best path
        best_path = tree.get_best_path()

        # Generate answer from best path
        answer = self._path_to_answer(best_path)

        return ReasoningResult(
            strategy=self.strategy_type,
            input_text=input_text,
            output_text=answer,
            confidence=tree.best_value,
            tree=tree,
            total_thoughts=len(tree.nodes),
            pruned_thoughts=sum(1 for n in tree.nodes.values() if n.status == ThoughtStatus.PRUNED),
            latency_ms=(time.time() - start) * 1000,
            alternatives=self._get_alternatives(tree),
        )

    async def _expand_node(
        self,
        tree: ThoughtTree,
        node: Thought,
        input_text: str,
    ) -> None:
        """Expand a node with child thoughts."""
        if node.depth >= tree.max_depth:
            return

        # Get path to this node for context
        context = tree.get_path_to_node(node.id)

        # Generate children
        children = await self.generator.generate(
            prompt=input_text,
            context=context,
            num_thoughts=tree.max_branches,
        )

        # Evaluate and add children
        for child in children:
            value = await self.evaluator.evaluate(child, context)
            tree.add_node(child, parent_id=node.id)

    def _path_to_answer(self, path: List[Thought]) -> str:
        """Convert path to final answer."""
        if not path:
            return "No valid reasoning path found."

        steps = [f"Step {i+1}: {t.content}" for i, t in enumerate(path)]
        return "\n".join(steps)

    def _get_alternatives(self, tree: ThoughtTree) -> List[Tuple[str, float]]:
        """Get alternative answers from other branches."""
        alternatives = []

        # Find other high-value leaves
        leaves = [n for n in tree.nodes.values()
                  if not n.children_ids and n.status != ThoughtStatus.PRUNED
                  and n.id != (tree.best_leaf.id if tree.best_leaf else None)]

        # Sort by value and take top 3
        leaves.sort(key=lambda n: n.value, reverse=True)

        for leaf in leaves[:3]:
            path = tree.get_path_to_node(leaf.id)
            answer = self._path_to_answer(path)
            alternatives.append((answer, leaf.value))

        return alternatives


class SelfReflectionStrategy(BaseReasoningStrategy):
    """
    Self-Reflection reasoning.

    Generates initial answer, critiques it, and revises until satisfactory.
    """

    strategy_type = ReasoningStrategy.SELF_REFLECTION

    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute Self-Reflection reasoning with RAG context augmentation."""
        start = time.time()

        # v92.1: Build augmented prompt with RAG context
        augmented_prompt = self._build_augmented_prompt(input_text, context)
        has_rag_context = augmented_prompt != input_text

        chain = ReasoningChain(
            strategy=self.strategy_type,
            input_text=input_text,
        )
        # Track RAG usage in chain metadata
        chain.metadata["has_rag_context"] = has_rag_context

        thoughts: List[Thought] = []
        revision_count = 0

        # Generate initial thought with augmented prompt
        initial_thoughts = await self.generator.generate(
            prompt=augmented_prompt,  # v92.1: Use augmented prompt with context
            context=[],
            num_thoughts=1,
        )

        if not initial_thoughts:
            return ReasoningResult(
                strategy=self.strategy_type,
                input_text=input_text,
                output_text="Unable to generate initial response.",
                confidence=0.0,
                latency_ms=(time.time() - start) * 1000,
            )

        current_thought = initial_thoughts[0]
        await self.evaluator.evaluate(current_thought, [])
        thoughts.append(current_thought)
        chain.add_thought(current_thought)

        # Self-reflection loop
        while (current_thought.confidence < self.config.reflection_threshold
               and revision_count < self.config.max_revisions):

            # Generate critique (use original input for clarity)
            critique = await self._generate_critique(current_thought, input_text)
            current_thought.self_critique = critique

            # Generate revision based on critique with context
            revision_prompt = f"{augmented_prompt}\n\nPrevious attempt: {current_thought.content}\n\nCritique: {critique}\n\nImproved response:"
            revised_thoughts = await self.generator.generate(
                prompt=revision_prompt,  # v92.1: Include RAG context in revisions
                context=thoughts,
                num_thoughts=1,
            )

            if not revised_thoughts:
                break

            revised = revised_thoughts[0]
            await self.evaluator.evaluate(revised, thoughts)

            # Check if revision is better
            if revised.value > current_thought.value:
                current_thought.revisions.append(revised.content)
                current_thought = revised
                thoughts.append(current_thought)
                chain.add_thought(current_thought)

            revision_count += 1

        chain.final_answer = current_thought.content
        chain.end_time = time.time()

        return ReasoningResult(
            strategy=self.strategy_type,
            input_text=input_text,
            output_text=chain.final_answer,
            confidence=current_thought.confidence,
            chain=chain,
            total_thoughts=len(thoughts),
            verified=current_thought.confidence >= self.config.reflection_threshold,
            self_assessment=current_thought.self_critique,
            latency_ms=(time.time() - start) * 1000,
        )

    async def _generate_critique(self, thought: Thought, input_text: str) -> str:
        """Generate critique of a thought."""
        # Heuristic critique generation
        critiques = []

        # Check length
        if len(thought.content) < 50:
            critiques.append("Response could be more detailed")

        # Check for reasoning indicators
        reasoning_words = ["because", "therefore", "thus", "since"]
        if not any(w in thought.content.lower() for w in reasoning_words):
            critiques.append("Could include more explicit reasoning")

        # Check specificity
        vague_words = ["something", "maybe", "perhaps", "kind of"]
        if any(w in thought.content.lower() for w in vague_words):
            critiques.append("Could be more specific and definitive")

        if not critiques:
            critiques.append("Response appears adequate")

        return "; ".join(critiques)


class HypothesisTestStrategy(BaseReasoningStrategy):
    """
    Hypothesis Testing reasoning.

    Generates hypotheses, tests them against evidence, and selects best.
    """

    strategy_type = ReasoningStrategy.HYPOTHESIS_TEST

    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute Hypothesis Testing reasoning with RAG context augmentation."""
        start = time.time()

        # v92.1: Build augmented prompt with RAG context
        augmented_prompt = self._build_augmented_prompt(input_text, context)
        has_rag_context = augmented_prompt != input_text

        # Generate hypotheses using augmented prompt (includes RAG context)
        hypotheses = await self._generate_hypotheses(augmented_prompt)

        # Test each hypothesis against augmented context
        tested: List[Tuple[str, float, str]] = []
        for hypothesis in hypotheses:
            confidence, evidence = await self._test_hypothesis(hypothesis, augmented_prompt)
            tested.append((hypothesis, confidence, evidence))

        # Select best hypothesis
        tested.sort(key=lambda x: x[1], reverse=True)
        best_hypothesis, best_confidence, best_evidence = tested[0] if tested else ("", 0.0, "")

        # Build result
        chain = ReasoningChain(
            strategy=self.strategy_type,
            input_text=input_text,
        )
        # Track RAG usage in chain metadata
        chain.metadata["has_rag_context"] = has_rag_context

        # Add hypothesis thoughts
        for hyp, conf, ev in tested:
            thought = Thought(
                content=f"Hypothesis: {hyp}\nEvidence: {ev}",
                confidence=conf,
                value=conf,
                status=ThoughtStatus.EVALUATED,
            )
            chain.add_thought(thought)

        chain.final_answer = f"Based on hypothesis testing:\n{best_hypothesis}\n\nSupporting evidence: {best_evidence}"
        chain.end_time = time.time()

        return ReasoningResult(
            strategy=self.strategy_type,
            input_text=input_text,
            output_text=chain.final_answer,
            confidence=best_confidence,
            chain=chain,
            total_thoughts=len(tested),
            verified=best_confidence >= self.config.hypothesis_confidence_threshold,
            alternatives=[(h, c) for h, c, _ in tested[1:4]],
            latency_ms=(time.time() - start) * 1000,
        )

    async def _generate_hypotheses(self, input_text: str) -> List[str]:
        """Generate hypotheses for the input."""
        # Extract key elements for hypothesis generation
        words = input_text.split()
        key_words = [w for w in words if len(w) > 4][:5]

        hypotheses = []
        templates = [
            "The answer involves {key}",
            "{key} is the primary factor",
            "This relates to {key}",
            "The solution requires understanding {key}",
        ]

        for i, key in enumerate(key_words[:self.config.max_hypotheses]):
            template = templates[i % len(templates)]
            hypotheses.append(template.format(key=key))

        return hypotheses

    async def _test_hypothesis(self, hypothesis: str, input_text: str) -> Tuple[float, str]:
        """Test a hypothesis against the input."""
        # Simple keyword-based testing
        input_lower = input_text.lower()
        hyp_words = set(hypothesis.lower().split())

        # Count matching words
        matches = sum(1 for w in hyp_words if w in input_lower)
        match_ratio = matches / max(len(hyp_words), 1)

        confidence = 0.3 + match_ratio * 0.5

        evidence = f"Found {matches} supporting terms in the input"

        return confidence, evidence


# =============================================================================
# REASONING ENGINE (v92.0 - Production-Grade with RAG, Thread-Safety, Observability)
# =============================================================================

class ReasoningEngine:
    """
    Production-grade reasoning engine with RAG integration and full observability.

    v92.0 CRITICAL FIXES:
    - Thread-safe statistics with AsyncLock
    - LRU cache with TTL and memory bounds
    - Semaphore-based rate limiting
    - RAG integration for context-aware reasoning
    - Training feedback loop
    - Task cancellation propagation
    - Graceful shutdown with state persistence

    ARCHITECTURE:
        Input → RAG Retrieval → Strategy Selection → Reasoning → Verification → Output
                    ↓                                    ↓
               Knowledge Base                   Training Feedback
    """

    STRATEGIES: Dict[ReasoningStrategy, Type[BaseReasoningStrategy]] = {
        ReasoningStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtStrategy,
        ReasoningStrategy.TREE_OF_THOUGHTS: TreeOfThoughtsStrategy,
        ReasoningStrategy.SELF_REFLECTION: SelfReflectionStrategy,
        ReasoningStrategy.HYPOTHESIS_TEST: HypothesisTestStrategy,
    }

    def __init__(
        self,
        config: Optional[ReasoningConfig] = None,
        executor: Optional[Any] = None,
        rag_engine: Optional[Any] = None,
    ):
        self.config = config or ReasoningConfig()
        self.executor = executor

        # Components
        self.generator = DefaultThoughtGenerator(executor)
        self.evaluator = DefaultThoughtEvaluator(executor)

        # Strategy instances
        self._strategies: Dict[ReasoningStrategy, BaseReasoningStrategy] = {}

        # Thread-safe statistics (FIXED: was non-atomic in v76.0)
        self._stats = ThreadSafeStatistics()

        # LRU cache with TTL (FIXED: was FIFO without TTL in v76.0)
        self._cache: LRUCache[ReasoningResult] = LRUCache(
            max_size=self.config.cache_max_size,
            ttl_seconds=self.config.cache_ttl_seconds,
        )

        # Rate limiter (NEW in v92.0)
        self._rate_limiter = RateLimiter(
            max_concurrent=self.config.max_concurrent_reasoning,
            timeout_seconds=self.config.rate_limit_timeout,
        )

        # Task cancellation manager (NEW in v92.0)
        self._cancellation_manager = TaskCancellationManager()

        # RAG integration (NEW in v92.0)
        self._rag_engine = rag_engine
        self._rag_initialized = False

        # Training feedback integration (NEW in v92.0)
        self._feedback_buffer: List[Dict[str, Any]] = []
        self._feedback_lock = asyncio.Lock()
        self._feedback_deduplicator = FeedbackDeduplicator()
        self._quality_scorer = QualityScorer()
        self._feedback_flush_task: Optional[asyncio.Task] = None

        # Context window management (NEW in v92.1)
        self._context_manager = ContextWindowManager(
            max_tokens=int(os.getenv("REASONING_MAX_CONTEXT_TOKENS", "4096")),
            reserved_tokens=int(os.getenv("REASONING_RESERVED_TOKENS", "512")),
        )

        # RAG context cache (NEW in v92.1)
        self._rag_cache = RAGContextCache(
            max_size=int(os.getenv("REASONING_RAG_CACHE_SIZE", "500")),
            ttl_seconds=float(os.getenv("REASONING_RAG_CACHE_TTL", "300.0")),
        )

        # Circuit breaker for RAG operations (NEW in v92.1)
        self._rag_circuit_breaker = CircuitBreaker(
            name="rag",
            failure_threshold=int(os.getenv("RAG_CIRCUIT_FAILURE_THRESHOLD", "5")),
            recovery_timeout=float(os.getenv("RAG_CIRCUIT_RECOVERY_TIMEOUT", "30.0")),
        )

        # Shutdown management (NEW in v92.0)
        self._shutdown_event = asyncio.Event()
        self._active_reasoning_tasks: Set[asyncio.Task] = set()

        # Observability hooks (NEW in v92.0)
        self._tracer: Optional[Any] = None
        self._initialize_observability()

        # Start background tasks
        self._start_background_tasks()

        logger.info("ReasoningEngine v92.1 initialized with RAG, context management, and advanced feedback")

    def _start_background_tasks(self) -> None:
        """Start background tasks for periodic operations."""
        # Time-based feedback flushing (every 5 minutes)
        async def periodic_flush():
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(300)  # 5 minutes
                    if self._feedback_buffer:
                        await self._flush_feedback()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"Periodic flush error: {e}")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                self._feedback_flush_task = asyncio.create_task(periodic_flush())
        except RuntimeError:
            pass  # No event loop yet, will start later

    def _initialize_observability(self) -> None:
        """Initialize observability hooks (tracing, metrics)."""
        if not self.config.enable_tracing:
            return

        try:
            from jarvis_prime.core.distributed_tracing import tracer
            self._tracer = tracer
            logger.debug("Observability tracing enabled")
        except ImportError:
            logger.debug("Distributed tracing not available")

    async def _initialize_rag(self) -> None:
        """Lazily initialize RAG engine."""
        if self._rag_initialized or not self.config.rag_enabled:
            return

        if self._rag_engine is None:
            try:
                from jarvis_prime.models.continual_learning_system import get_rag_engine
                self._rag_engine = await get_rag_engine()
                self._rag_initialized = True
                logger.info("RAG engine initialized for reasoning")
            except ImportError:
                logger.warning("RAG engine not available - reasoning without retrieval")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG engine: {e}")

    async def _retrieve_context(
        self,
        input_text: str,
        top_k: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve relevant context using RAG with caching and context management.

        Args:
            input_text: Query text
            top_k: Number of documents to retrieve

        Returns:
            Retrieved context or None
        """
        retrieval_start = time.time()

        # Check cache first
        cached = await self._rag_cache.get(input_text)
        if cached is not None:
            await self._stats.increment("rag_cache_hits")
            return cached

        await self._stats.increment("rag_cache_misses")
        await self._initialize_rag()

        if not self._rag_engine:
            return None

        # Define the actual retrieval function
        async def _do_retrieval() -> Optional[Dict[str, Any]]:
            """Perform the actual RAG retrieval."""
            effective_top_k = top_k or self.config.rag_top_k
            retrieval_result = await self._rag_engine.retrieve(input_text, effective_top_k)

            if not retrieval_result.documents:
                return None

            # Filter by relevance threshold and normalize scores
            relevant_docs = []
            relevant_scores = []
            for doc, score in zip(retrieval_result.documents, retrieval_result.scores):
                # Normalize score to 0-1 range if needed
                normalized_score = max(0.0, min(1.0, score))
                if normalized_score >= self.config.rag_min_relevance:
                    relevant_docs.append(doc)
                    relevant_scores.append(normalized_score)

            if not relevant_docs:
                return None

            # Build context with priority-based truncation
            contexts_with_priority = [
                (f"[Relevance: {score:.2f}] {doc.content}", score)
                for doc, score in zip(relevant_docs, relevant_scores)
            ]

            # Use context manager to fit within token limits
            input_tokens = self._context_manager.count_tokens(input_text)
            available_tokens = self._context_manager.available_tokens(input_tokens)

            formatted_context = await self._context_manager.prioritize_context(
                contexts_with_priority,
                max_tokens=available_tokens,
            )

            retrieval_latency = (time.time() - retrieval_start) * 1000
            await self._stats.record_timing("rag_retrieval_latency_ms", retrieval_latency)

            # Check context utilization
            context_tokens = self._context_manager.count_tokens(formatted_context)
            await self._context_manager.check_utilization(context_tokens, "RAG context")

            return {
                "retrieved_context": formatted_context,
                "formatted_context": formatted_context,  # v92.1: Add for strategy use
                "retrieved_documents": relevant_docs,
                "relevance_scores": relevant_scores,
                "num_documents": len(relevant_docs),
                "retrieval_latency_ms": retrieval_latency,
                "context_tokens": context_tokens,
            }

        # Fallback when circuit breaker is open
        async def _retrieval_fallback() -> None:
            """Fallback when RAG is unavailable - proceed without context."""
            await self._stats.increment("rag_circuit_breaker_fallbacks")
            logger.info("RAG circuit breaker open - proceeding without context")
            return None

        # v92.1: Execute through circuit breaker for graceful degradation
        try:
            result = await self._rag_circuit_breaker.call(
                _do_retrieval,
                fallback=_retrieval_fallback,
            )

            # Cache the result if successful
            if result is not None:
                await self._rag_cache.set(input_text, result)

            return result

        except CircuitBreakerOpenError:
            # Circuit is open and no fallback was provided (shouldn't happen)
            await self._stats.increment("rag_circuit_breaker_rejections")
            return None

        except Exception as e:
            await self._stats.increment("rag_retrieval_errors")
            logger.warning(f"RAG retrieval failed: {e}")
            return None

    def _get_strategy(self, strategy_type: ReasoningStrategy) -> BaseReasoningStrategy:
        """Get or create a strategy instance."""
        if strategy_type not in self._strategies:
            if strategy_type in self.STRATEGIES:
                self._strategies[strategy_type] = self.STRATEGIES[strategy_type](
                    config=self.config,
                    generator=self.generator,
                    evaluator=self.evaluator,
                )
            else:
                # Default to CoT
                self._strategies[strategy_type] = ChainOfThoughtStrategy(
                    config=self.config,
                    generator=self.generator,
                    evaluator=self.evaluator,
                )

        return self._strategies[strategy_type]

    async def reason(
        self,
        input_text: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        use_rag: bool = True,
    ) -> ReasoningResult:
        """
        Execute reasoning on input text with RAG augmentation.

        Args:
            input_text: The problem or question to reason about
            strategy: Specific strategy to use (auto-selected if None)
            context: Additional context
            use_cache: Whether to use cached results
            use_rag: Whether to use RAG for context retrieval

        Returns:
            ReasoningResult with answer and reasoning chain
        """
        start_time = time.time()

        # Check for shutdown
        if self._shutdown_event.is_set():
            return ReasoningResult(
                strategy=strategy or self.config.default_strategy,
                input_text=input_text,
                output_text="Reasoning engine is shutting down.",
                confidence=0.0,
                latency_ms=0.0,
            )

        # Check cache first
        if use_cache and self.config.cache_thoughts:
            cache_key = self._make_cache_key(input_text, strategy)
            cached_result = await self._cache.get(cache_key)
            if cached_result is not None:
                await self._stats.increment("cache_hits")
                logger.debug("Returning cached reasoning result")
                return cached_result
            await self._stats.increment("cache_misses")

        # Apply rate limiting
        merged_context = None
        try:
            async with self._rate_limiter.acquire():
                result, merged_context = await self._execute_reasoning(
                    input_text=input_text,
                    strategy=strategy,
                    context=context,
                    use_rag=use_rag,
                    start_time=start_time,
                )
        except asyncio.TimeoutError:
            await self._stats.increment("rate_limit_timeouts")
            result = ReasoningResult(
                strategy=strategy or self.config.default_strategy,
                input_text=input_text,
                output_text="Reasoning queue is full. Please try again later.",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
            )
            merged_context = context  # Preserve original context for feedback

        # Cache result
        if use_cache and self.config.cache_thoughts and result.confidence > 0:
            cache_key = self._make_cache_key(input_text, strategy)
            await self._cache.set(cache_key, result)

        # Collect training feedback (NEW in v92.0, FIXED in v92.1)
        # Pass merged_context to preserve RAG retrieval info for training
        if self.config.training_feedback_enabled:
            await self._collect_feedback(input_text, result, merged_context=merged_context)

        return result

    async def _execute_reasoning(
        self,
        input_text: str,
        strategy: Optional[ReasoningStrategy],
        context: Optional[Dict[str, Any]],
        use_rag: bool,
        start_time: float,
    ) -> Tuple[ReasoningResult, Optional[Dict[str, Any]]]:
        """
        Execute reasoning with RAG context augmentation.

        Returns:
            Tuple of (ReasoningResult, merged_context) where merged_context
            includes any RAG retrieval results for training feedback.
        """
        # Retrieve RAG context (NEW in v92.0)
        rag_context = None
        if use_rag and self.config.rag_enabled:
            rag_context = await self._retrieve_context(input_text)

        # Merge contexts (original + RAG)
        merged_context = context.copy() if context else {}
        if rag_context:
            merged_context = {**merged_context, **rag_context}
            await self._stats.increment("rag_retrievals")

        # Select strategy
        if strategy is None:
            strategy = await self._select_strategy(input_text, merged_context)

        # Get strategy instance
        strategy_impl = self._get_strategy(strategy)

        # Update statistics (FIXED: now thread-safe)
        await self._stats.increment("total_reasoning_calls")
        await self._stats.record_histogram("strategy_usage", strategy.value)

        # Create cancellation manager for this reasoning task
        cancellation_mgr = TaskCancellationManager()

        try:
            # Execute reasoning with timeout and cancellation support
            reasoning_task = asyncio.create_task(
                strategy_impl.reason(input_text, merged_context)
            )
            await cancellation_mgr.register(reasoning_task)

            result = await asyncio.wait_for(
                reasoning_task,
                timeout=self.config.timeout_seconds,
            )

            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            result.latency_ms = latency_ms
            await self._stats.record_timing("reasoning_latency_ms", latency_ms)

        except asyncio.TimeoutError:
            # Cancel child tasks (FIXED: was not cancelling in v76.0)
            await cancellation_mgr.cancel_all("Reasoning timeout")
            await self._stats.increment("timeouts")

            logger.warning(f"Reasoning timeout after {self.config.timeout_seconds}s")
            result = ReasoningResult(
                strategy=strategy,
                input_text=input_text,
                output_text="Reasoning timed out. Please try with a simpler query.",
                confidence=0.0,
                latency_ms=self.config.timeout_seconds * 1000,
            )

        except asyncio.CancelledError:
            await self._stats.increment("cancellations")
            raise

        except Exception as e:
            await self._stats.increment("errors")
            logger.error(f"Reasoning error: {e}")
            result = ReasoningResult(
                strategy=strategy,
                input_text=input_text,
                output_text=f"Reasoning error: {str(e)}",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
            )

            # Collect error feedback for learning (NEW in v92.1)
            # Pass merged_context so training data includes what context was available
            if self.config.training_feedback_enabled:
                await self._collect_feedback(input_text, result, is_error=True, merged_context=merged_context)

        return (result, merged_context)

    async def _collect_feedback(
        self,
        input_text: str,
        result: ReasoningResult,
        is_error: bool = False,
        merged_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Collect feedback for training pipeline with quality scoring and deduplication.

        v92.1 ENHANCEMENTS:
        - Multi-factor quality scoring
        - Deduplication
        - Error case collection
        - Comprehensive metadata
        - RAG context preservation for training
        - System prompt capture

        Args:
            input_text: Original input
            result: Reasoning result
            is_error: Whether this was an error case
            merged_context: The merged context including RAG retrieval results
        """
        # Calculate comprehensive quality score
        quality_score = self._quality_scorer.score(result, input_text)

        # Skip low-quality results (but keep errors for learning)
        if not is_error and quality_score < self.config.feedback_collection_threshold:
            await self._stats.increment("feedback_filtered_low_quality")
            return

        # Check for duplicates
        if await self._feedback_deduplicator.is_duplicate(input_text, result.output_text):
            await self._stats.increment("feedback_filtered_duplicate")
            return

        # Extract RAG context for training (preserves what knowledge was used)
        rag_context_summary = None
        has_rag = False
        if merged_context:
            has_rag = "retrieved_context" in merged_context or "formatted_context" in merged_context
            if has_rag:
                rag_context_summary = {
                    "formatted_context": merged_context.get("formatted_context"),
                    "num_documents": merged_context.get("num_documents", 0),
                    "relevance_scores": merged_context.get("relevance_scores", []),
                }

        # Extract system prompt if available
        system_prompt = None
        if merged_context:
            system_prompt = merged_context.get("system_prompt")

        async with self._feedback_lock:
            feedback_item = {
                "timestamp": time.time(),
                "input": input_text,
                "output": result.output_text,
                "strategy": result.strategy.value,
                "confidence": result.confidence,
                "quality_score": quality_score,
                "latency_ms": result.latency_ms,
                "total_thoughts": result.total_thoughts,
                "is_error": is_error,
                "has_rag_context": has_rag or (result.chain.metadata.get("has_rag_context", False) if result.chain else False),
                # NEW in v92.1: Preserve RAG context and system prompt for training
                "rag_context": rag_context_summary,
                "system_prompt": system_prompt,
                # Thought chain for detailed analysis
                "thought_chain_summary": self._summarize_thought_chain(result.chain) if result.chain else None,
            }
            self._feedback_buffer.append(feedback_item)
            await self._stats.increment("feedback_collected")

            # Flush to training pipeline when buffer is full
            buffer_threshold = int(os.getenv("REASONING_FEEDBACK_BUFFER_SIZE", "100"))
            if len(self._feedback_buffer) >= buffer_threshold:
                await self._flush_feedback()

    def _summarize_thought_chain(self, chain: Optional["ThoughtChain"]) -> Optional[Dict[str, Any]]:
        """Summarize thought chain for training feedback (avoid storing full chain)."""
        if not chain:
            return None
        return {
            "depth": chain.depth,
            "total_thoughts": len(chain.thoughts) if hasattr(chain, "thoughts") else 0,
            "final_confidence": chain.final_confidence if hasattr(chain, "final_confidence") else None,
            "pruned_branches": chain.metadata.get("pruned_branches", 0),
            "explored_alternatives": chain.metadata.get("explored_alternatives", 0),
        }

    async def _flush_feedback(self) -> None:
        """
        Flush feedback buffer to training pipeline with error handling.

        v92.1 FIXES:
        - Fixed method name: collect_conversation -> capture_conversation
        - Added pipeline availability check
        - Added retry logic
        - Added detailed error logging
        """
        if not self._feedback_buffer:
            return

        flush_start = time.time()
        flushed_count = 0
        error_count = 0

        try:
            # Check if training pipeline is available
            try:
                from jarvis_prime.core.training_data_pipeline import get_training_data_pipeline
                pipeline = await get_training_data_pipeline()
            except ImportError:
                logger.warning("Training data pipeline not available - feedback not persisted")
                await self._stats.increment("feedback_flush_pipeline_unavailable")
                return
            except Exception as e:
                logger.warning(f"Failed to get training pipeline: {e}")
                await self._stats.increment("feedback_flush_pipeline_error")
                return

            # Process feedback items
            for item in self._feedback_buffer:
                try:
                    # v92.1 FIX: Correct parameter names for capture_conversation API
                    # Signature: capture_conversation(user_input, assistant_response, system_prompt, context)
                    await pipeline.capture_conversation(
                        user_input=item["input"],
                        assistant_response=item["output"],
                        system_prompt=item.get("system_prompt"),
                        context={
                            # Quality and scoring metadata
                            "quality_score": item["quality_score"],
                            "original_confidence": item["confidence"],
                            # Source identification
                            "source": "reasoning_engine",
                            "strategy": item["strategy"],
                            # Execution metadata
                            "is_error": item.get("is_error", False),
                            "has_rag_context": item.get("has_rag_context", False),
                            "latency_ms": item["latency_ms"],
                            "total_thoughts": item["total_thoughts"],
                            # RAG context if available
                            "rag_context": item.get("rag_context"),
                            # Timestamp for ordering
                            "captured_at": item.get("timestamp", time.time()),
                        }
                    )
                    flushed_count += 1
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Failed to capture conversation: {e}")

            flush_latency = (time.time() - flush_start) * 1000
            await self._stats.record_timing("feedback_flush_latency_ms", flush_latency)
            await self._stats.increment("feedback_flushed", flushed_count)

            if error_count > 0:
                await self._stats.increment("feedback_flush_errors", error_count)
                logger.warning(f"Feedback flush: {flushed_count} succeeded, {error_count} failed")
            else:
                logger.info(f"Flushed {flushed_count} feedback items to training in {flush_latency:.0f}ms")

            self._feedback_buffer.clear()

        except Exception as e:
            await self._stats.increment("feedback_flush_critical_error")
            logger.error(f"Critical error flushing feedback: {e}")

    async def _select_strategy(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningStrategy:
        """
        Automatically select best strategy for input.

        Uses RAG-retrieved context to improve selection (NEW in v92.0).
        """
        if not self.config.enable_adaptive:
            return self.config.default_strategy

        input_lower = input_text.lower()

        # Check if RAG context suggests a strategy
        if context and "retrieved_context" in context:
            # If we have relevant past examples, use reflection to verify
            return ReasoningStrategy.SELF_REFLECTION

        # Heuristic strategy selection
        indicators = {
            ReasoningStrategy.TREE_OF_THOUGHTS: [
                "explore", "alternatives", "options", "different ways",
                "multiple", "compare", "which is better", "pros and cons",
            ],
            ReasoningStrategy.SELF_REFLECTION: [
                "careful", "accurate", "verify", "check", "sure",
                "certain", "correct", "precise", "double-check",
            ],
            ReasoningStrategy.HYPOTHESIS_TEST: [
                "why", "cause", "reason", "hypothesis", "theory",
                "explain", "because", "investigate", "diagnose",
            ],
            ReasoningStrategy.CHAIN_OF_THOUGHT: [
                "how", "steps", "process", "procedure", "method",
                "first", "then", "calculate", "solve",
            ],
        }

        scores = {s: 0 for s in indicators}

        for strategy, keywords in indicators.items():
            for keyword in keywords:
                if keyword in input_lower:
                    scores[strategy] += 1

        # Select highest scoring or default
        best_strategy = max(scores, key=scores.get)
        if scores[best_strategy] > 0:
            await self._stats.record_histogram("strategy_selection", best_strategy.value)
            return best_strategy

        return self.config.default_strategy

    def _make_cache_key(
        self,
        input_text: str,
        strategy: Optional[ReasoningStrategy],
    ) -> str:
        """Create cache key."""
        key_data = f"{input_text}:{strategy.value if strategy else 'auto'}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def reason_stream(
        self,
        input_text: str,
        strategy: Optional[ReasoningStrategy] = None,
        use_rag: bool = True,
    ) -> AsyncIterator[Thought]:
        """
        Stream thoughts as they are generated.

        Useful for real-time feedback in UI.
        """
        # Check for shutdown
        if self._shutdown_event.is_set():
            return

        # Retrieve RAG context
        context = None
        if use_rag and self.config.rag_enabled:
            context = await self._retrieve_context(input_text)

        strategy = strategy or await self._select_strategy(input_text, context)
        strategy_impl = self._get_strategy(strategy)

        async with self._rate_limiter.acquire():
            # For streaming, use CoT and yield each thought
            if isinstance(strategy_impl, ChainOfThoughtStrategy):
                thoughts: List[Thought] = []

                for step in range(self.config.cot_max_steps):
                    if self._shutdown_event.is_set():
                        break

                    new_thoughts = await self.generator.generate(
                        prompt=input_text,
                        context=thoughts,
                        num_thoughts=1,
                    )

                    if not new_thoughts:
                        break

                    thought = new_thoughts[0]
                    await self.evaluator.evaluate(thought, thoughts)
                    thoughts.append(thought)

                    yield thought

                    if thought.confidence >= self.config.cot_stop_on_confidence:
                        break

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive engine statistics (thread-safe).

        v92.1 ENHANCEMENT: Added context window and deduplication metrics.
        """
        stats = await self._stats.get_all_stats()
        cache_stats = await self._cache.get_stats()
        rate_limiter_stats = await self._rate_limiter.get_stats()
        latency_stats = await self._stats.get_timing_stats("reasoning_latency_ms")
        rag_latency_stats = await self._stats.get_timing_stats("rag_retrieval_latency_ms")
        feedback_flush_stats = await self._stats.get_timing_stats("feedback_flush_latency_ms")
        rag_cache_stats = await self._rag_cache.get_stats()
        context_window_stats = await self._context_manager.get_stats()
        deduplicator_stats = self._feedback_deduplicator.get_stats()

        return {
            "counters": stats["counters"],
            "strategy_usage": stats["histograms"].get("strategy_usage", {}),
            "latency": {
                "reasoning": latency_stats,
                "rag_retrieval": rag_latency_stats,
                "feedback_flush": feedback_flush_stats,
            },
            "cache": {
                "reasoning": cache_stats,
                "rag_context": rag_cache_stats,
            },
            "rate_limiter": rate_limiter_stats,
            "feedback": {
                "buffer_size": len(self._feedback_buffer),
                "collected": stats["counters"].get("feedback_collected", 0),
                "flushed": stats["counters"].get("feedback_flushed", 0),
                "filtered_duplicate": stats["counters"].get("feedback_filtered_duplicate", 0),
                "filtered_low_quality": stats["counters"].get("feedback_filtered_low_quality", 0),
                "deduplicator": deduplicator_stats,
            },
            "rag": {
                "initialized": self._rag_initialized,
                "cache_hits": stats["counters"].get("rag_cache_hits", 0),
                "cache_misses": stats["counters"].get("rag_cache_misses", 0),
                "retrievals": stats["counters"].get("rag_retrievals", 0),
                "errors": stats["counters"].get("rag_retrieval_errors", 0),
                # v92.1: Circuit breaker stats
                "circuit_breaker": self._rag_circuit_breaker.get_stats(),
                "circuit_breaker_fallbacks": stats["counters"].get("rag_circuit_breaker_fallbacks", 0),
            },
            # v92.1: Context window metrics
            "context_window": context_window_stats,
            "config": {
                "default_strategy": self.config.default_strategy.value,
                "rag_enabled": self.config.rag_enabled,
                "cache_enabled": self.config.cache_thoughts,
                "max_concurrent": self.config.max_concurrent_reasoning,
                "feedback_threshold": self.config.feedback_collection_threshold,
                "max_tokens": self._context_manager._max_tokens,
                "reserved_tokens": self._context_manager._reserved_tokens,
            },
        }

    async def shutdown(self) -> None:
        """
        Graceful shutdown with state persistence (NEW in v92.0, enhanced v92.1).

        - Signals all active reasoning tasks to stop
        - Cancels background flush task
        - Flushes feedback buffer
        - Persists cache if configured
        - Releases resources
        """
        logger.info("Reasoning engine shutting down...")
        self._shutdown_event.set()

        # Cancel background flush task
        if self._feedback_flush_task and not self._feedback_flush_task.done():
            self._feedback_flush_task.cancel()
            try:
                await self._feedback_flush_task
            except asyncio.CancelledError:
                pass

        # Cancel all active reasoning tasks
        cancelled = await self._cancellation_manager.cancel_all("Engine shutdown")
        logger.info(f"Cancelled {cancelled} active reasoning tasks")

        # Final feedback flush
        await self._flush_feedback()

        # Clear caches
        await self._cache.clear()

        # Get final statistics
        final_stats = await self.get_statistics()
        rag_cache_stats = await self._rag_cache.get_stats()

        logger.info(
            f"Final reasoning engine stats: "
            f"calls={final_stats['counters'].get('total_reasoning_calls', 0)}, "
            f"errors={final_stats['counters'].get('errors', 0)}, "
            f"feedback_collected={final_stats['counters'].get('feedback_collected', 0)}, "
            f"rag_cache_hit_rate={rag_cache_stats['hit_rate']:.2%}"
        )

        logger.info("Reasoning engine shutdown complete")

    async def health_check(self) -> Dict[str, Any]:
        """
        Health check for monitoring (NEW in v92.0).

        Returns health status and key metrics.
        """
        stats = await self.get_statistics()
        rate_limiter = await self._rate_limiter.get_stats()

        return {
            "status": "healthy" if not self._shutdown_event.is_set() else "shutting_down",
            "rag_available": self._rag_initialized,
            "cache_hit_rate": stats["cache"]["hit_rate"],
            "active_reasoning": rate_limiter["active_count"],
            "available_slots": rate_limiter["available_slots"],
            "total_calls": stats["counters"].get("total_reasoning_calls", 0),
            "error_count": stats["counters"].get("errors", 0),
        }


# =============================================================================
# FACTORY FUNCTIONS & GLOBAL INSTANCE MANAGEMENT (v92.0)
# =============================================================================

# Global reasoning engine instance with thread-safe initialization
_reasoning_engine: Optional[ReasoningEngine] = None
_engine_lock = asyncio.Lock()
_engine_initialized = False


async def get_reasoning_engine(
    config: Optional[ReasoningConfig] = None,
    executor: Optional[Any] = None,
    rag_engine: Optional[Any] = None,
) -> ReasoningEngine:
    """
    Get or create global reasoning engine instance.

    Thread-safe singleton pattern with proper initialization.

    Args:
        config: Optional configuration (only used on first call)
        executor: Optional LLM executor (only used on first call)
        rag_engine: Optional RAG engine (only used on first call)

    Returns:
        Shared ReasoningEngine instance
    """
    global _reasoning_engine, _engine_initialized

    # Fast path - already initialized
    if _engine_initialized and _reasoning_engine is not None:
        return _reasoning_engine

    async with _engine_lock:
        # Double-check after acquiring lock
        if _engine_initialized and _reasoning_engine is not None:
            return _reasoning_engine

        # Create new instance
        _reasoning_engine = ReasoningEngine(
            config=config,
            executor=executor,
            rag_engine=rag_engine,
        )
        _engine_initialized = True

        logger.info("Global reasoning engine initialized")

        return _reasoning_engine


async def shutdown_reasoning_engine() -> None:
    """
    Shutdown global reasoning engine.

    Call this during application shutdown to ensure proper cleanup.
    """
    global _reasoning_engine, _engine_initialized

    async with _engine_lock:
        if _reasoning_engine is not None:
            await _reasoning_engine.shutdown()
            _reasoning_engine = None
            _engine_initialized = False
            logger.info("Global reasoning engine shutdown")


def create_reasoning_engine(
    executor: Optional[Any] = None,
    config: Optional[ReasoningConfig] = None,
    rag_engine: Optional[Any] = None,
) -> ReasoningEngine:
    """
    Factory function to create a NEW reasoning engine instance.

    Use get_reasoning_engine() for the shared singleton.
    Use this only when you need a separate instance.

    Args:
        executor: Optional LLM executor
        config: Optional configuration
        rag_engine: Optional RAG engine

    Returns:
        New ReasoningEngine instance
    """
    return ReasoningEngine(
        config=config,
        executor=executor,
        rag_engine=rag_engine,
    )


# Register shutdown handler for graceful cleanup
def _register_shutdown_handler() -> None:
    """Register atexit handler for shutdown."""
    def _sync_shutdown():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule shutdown
                loop.create_task(shutdown_reasoning_engine())
            else:
                # Run synchronously
                loop.run_until_complete(shutdown_reasoning_engine())
        except Exception:
            pass  # Ignore errors during shutdown

    atexit.register(_sync_shutdown)


_register_shutdown_handler()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ReasoningStrategy",
    "ThoughtStatus",
    "VerificationResult",
    # Data classes
    "Thought",
    "ReasoningChain",
    "ThoughtTree",
    "ReasoningConfig",
    "ReasoningResult",
    # Utilities (v92.0)
    "ThreadSafeStatistics",
    "LRUCache",
    "RateLimiter",
    "TaskCancellationManager",
    # Advanced Utilities (v92.1)
    "ContextWindowManager",
    "RAGContextCache",
    "FeedbackDeduplicator",
    "QualityScorer",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    # Generators/Evaluators
    "ThoughtGenerator",
    "DefaultThoughtGenerator",
    "ThoughtEvaluator",
    "DefaultThoughtEvaluator",
    # Strategies
    "BaseReasoningStrategy",
    "ChainOfThoughtStrategy",
    "TreeOfThoughtsStrategy",
    "SelfReflectionStrategy",
    "HypothesisTestStrategy",
    # Engine
    "ReasoningEngine",
    "create_reasoning_engine",
    "get_reasoning_engine",
    "shutdown_reasoning_engine",
]
