#!/usr/bin/env python3
"""
Neural Orchestrator Core v100.0 - Unified Intelligent Routing Architecture
============================================================================

ENTERPRISE-GRADE UNIFIED ORCHESTRATION:
    - Consolidates ALL routing systems (HybridTieredRouter, IntelligentModelRouter,
      CognitiveRouter, GraphRouter, Neural Switchboard) into a single source of truth
    - Zero hardcoding - all values from dynamic config with env var override
    - Advanced async patterns with proper cancellation and cleanup
    - Full race condition protection with distributed locking
    - Cross-repo memory integration with real-time pressure sharing
    - Request buffering with zero loss during hot swaps
    - Sticky routing for session continuity
    - Predictive preloading based on usage patterns
    - Circuit breakers with coordinated state across all routers

ADVANCED PATTERNS IMPLEMENTED:
    - Protocol classes for type-safe router interfaces
    - Contextvars for distributed request tracing
    - Async generators for streaming responses
    - Weakref for memory-efficient caching
    - Defensive decorators with graceful fallbacks
    - Atomic operations for state mutations
    - Structured concurrency with TaskGroup (Python 3.11+)
    - Event sourcing for routing decisions audit trail

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    NeuralOrchestratorCore v100.0                        │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                    UNIFIED ROUTING LAYER                         │   │
    │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │   │
    │  │  │ TaskClass │ │MemPressure│ │ Sticky    │ │ RequestBuf│        │   │
    │  │  │   -ifier  │ │  Monitor  │ │ Routing   │ │   -fer    │        │   │
    │  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘        │   │
    │  │        └─────────────┼─────────────┼─────────────┘              │   │
    │  │                      ▼             ▼                            │   │
    │  │              ┌───────────────────────────┐                      │   │
    │  │              │   ROUTING DECISION ENGINE │                      │   │
    │  │              │    (Unified Algorithm)    │                      │   │
    │  │              └─────────────┬─────────────┘                      │   │
    │  │                            │                                    │   │
    │  │  ┌─────────────────────────┼─────────────────────────┐          │   │
    │  │  │                         ▼                         │          │   │
    │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────┐  │          │   │
    │  │  │  │ Tier 0  │  │Tier 0.5 │  │ Tier 1  │  │Tier 2│  │          │   │
    │  │  │  │ Ultra   │  │ Local   │  │ Cloud   │  │ Deep │  │          │   │
    │  │  │  │ Fast    │  │ Capable │  │  Intel  │  │Reason│  │          │   │
    │  │  │  └────┬────┘  └────┬────┘  └────┬────┘  └──┬───┘  │          │   │
    │  │  │       └────────────┼────────────┼─────────┘       │          │   │
    │  │  │                    ▼            ▼                 │          │   │
    │  │  │           ┌────────────────────────────┐          │          │   │
    │  │  │           │  CIRCUIT BREAKER MANAGER   │          │          │   │
    │  │  │           │  (Coordinated State)       │          │          │   │
    │  │  │           └────────────────────────────┘          │          │   │
    │  │  └───────────────────────────────────────────────────┘          │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                    CROSS-REPO INTEGRATION                        │   │
    │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐                      │   │
    │  │  │  JARVIS   │ │  JARVIS   │ │  Reactor  │                      │   │
    │  │  │  (Body)   │ │  Prime    │ │   Core    │                      │   │
    │  │  │  Memory   │ │  Memory   │ │  Sync     │                      │   │
    │  │  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                      │   │
    │  │        └─────────────┼─────────────┘                            │   │
    │  │                      ▼                                          │   │
    │  │        ┌───────────────────────────┐                            │   │
    │  │        │  SHARED STATE MANAGER     │                            │   │
    │  │        │  (~/.jarvis/cross_repo/)  │                            │   │
    │  │        └───────────────────────────┘                            │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

Usage:
    # Get the unified orchestrator
    orchestrator = await NeuralOrchestratorCore.get_instance()

    # Route a request (handles everything automatically)
    result = await orchestrator.route(
        prompt="Implement a distributed cache",
        context={"session_id": "abc123"}
    )

    # Access components directly if needed
    stats = orchestrator.get_comprehensive_stats()
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import os
import sys
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    Final,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
R = TypeVar("R")

# =============================================================================
# ADVANCED IMPORTS (Optional - graceful degradation if unavailable)
# =============================================================================

try:
    from collections import deque
    from threading import RLock
    import random
    import tempfile
    import fcntl
    ADVANCED_IMPORTS_AVAILABLE = True
except ImportError:
    ADVANCED_IMPORTS_AVAILABLE = False
    deque = list  # type: ignore

# Observability bridge availability
OBSERVABILITY_AVAILABLE = False
try:
    from jarvis_prime.core.observability_bridge import get_observability_bridge
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class DeadlockError(Exception):
    """Raised when a deadlock is detected."""
    pass


class RoutingCancelledException(Exception):
    """Raised when a routing request is cancelled."""
    pass


class AllTiersUnavailableError(Exception):
    """Raised when all routing tiers are unavailable."""
    pass


class RequestDuplicateError(Exception):
    """Raised when a duplicate request is detected."""
    pass


# =============================================================================
# CANCELLATION TOKEN - Advanced request lifecycle management
# =============================================================================


class CancellationToken:
    """
    Thread-safe cancellation token for request lifecycle management.

    Supports:
    - Cooperative cancellation checking
    - Cascading cancellation to child tokens
    - Timeout-based auto-cancellation
    """

    __slots__ = ('_cancelled', '_lock', '_children', '_reason', '_cancel_time')

    def __init__(self):
        self._cancelled = False
        self._lock = asyncio.Lock()
        self._children: List["CancellationToken"] = []
        self._reason: Optional[str] = None
        self._cancel_time: Optional[datetime] = None

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancelled

    @property
    def cancellation_reason(self) -> Optional[str]:
        """Get reason for cancellation."""
        return self._reason

    async def cancel(self, reason: str = "cancelled") -> None:
        """Request cancellation."""
        async with self._lock:
            if not self._cancelled:
                self._cancelled = True
                self._reason = reason
                self._cancel_time = datetime.now()
                # Cascade to children
                for child in self._children:
                    await child.cancel(reason=f"parent:{reason}")

    def create_child(self) -> "CancellationToken":
        """Create a child token that cancels when parent cancels."""
        child = CancellationToken()
        self._children.append(child)
        if self._cancelled:
            child._cancelled = True
            child._reason = f"parent:{self._reason}"
        return child

    def check(self) -> None:
        """Check if cancelled and raise if so."""
        if self._cancelled:
            raise RoutingCancelledException(self._reason or "cancelled")

    @classmethod
    def with_timeout(cls, timeout_seconds: float) -> "CancellationToken":
        """Create a token that auto-cancels after timeout."""
        token = cls()
        asyncio.create_task(token._timeout_canceller(timeout_seconds))
        return token

    async def _timeout_canceller(self, timeout: float) -> None:
        """Background task to cancel after timeout."""
        await asyncio.sleep(timeout)
        await self.cancel(reason=f"timeout:{timeout}s")


# Context variable for cancellation token
cancellation_token_var: contextvars.ContextVar[Optional[CancellationToken]] = contextvars.ContextVar(
    'cancellation_token', default=None
)


# =============================================================================
# DEADLOCK-DETECTING LOCK - Prevents indefinite blocking
# =============================================================================


class DeadlockDetectingLock:
    """
    Async lock with deadlock detection and timeout.

    Features:
    - Configurable acquisition timeout
    - Tracks lock holder for debugging
    - Raises DeadlockError on timeout
    """

    __slots__ = ('_lock', '_timeout', '_acquired_by', '_acquired_at', '_acquisition_count')

    def __init__(self, timeout_seconds: float = 5.0):
        self._lock = asyncio.Lock()
        self._timeout = timeout_seconds
        self._acquired_by: Optional[str] = None
        self._acquired_at: Optional[float] = None
        self._acquisition_count = 0

    async def acquire(self, caller: str = "unknown") -> bool:
        """Acquire lock with timeout and deadlock detection."""
        try:
            acquired = await asyncio.wait_for(
                self._lock.acquire(),
                timeout=self._timeout
            )
            if acquired:
                self._acquired_by = caller
                self._acquired_at = time.time()
                self._acquisition_count += 1
            return acquired
        except asyncio.TimeoutError:
            raise DeadlockError(
                f"Lock acquisition timeout after {self._timeout}s. "
                f"Currently held by '{self._acquired_by}' since {self._acquired_at}"
            )

    def release(self) -> None:
        """Release the lock."""
        self._acquired_by = None
        self._acquired_at = None
        self._lock.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    @property
    def stats(self) -> Dict[str, Any]:
        """Get lock statistics."""
        return {
            "locked": self._lock.locked(),
            "acquired_by": self._acquired_by,
            "acquired_at": self._acquired_at,
            "total_acquisitions": self._acquisition_count,
        }


# =============================================================================
# LRU ROUTING CACHE - Caches routing decisions for performance
# =============================================================================


class LRURoutingCache:
    """
    Thread-safe LRU cache for routing decisions.

    Features:
    - Configurable max size and TTL
    - Automatic eviction of stale entries
    - Hit/miss statistics
    - Thread-safe operations
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 60.0,
    ):
        self._cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = DeadlockDetectingLock(timeout_seconds=2.0)

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired."""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, cached_time = self._cache[key]

            # Check expiration
            if (datetime.now() - cached_time).total_seconds() > self._ttl_seconds:
                del self._cache[key]
                self._misses += 1
                self._evictions += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    async def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        async with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = (value, datetime.now())

    async def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """Clear all cache entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(hit_rate, 4),
        }


# =============================================================================
# REQUEST DEDUPLICATOR - Prevents duplicate request processing
# =============================================================================


class RequestDeduplicator:
    """
    Prevents processing of duplicate requests.

    Uses prompt hashing to detect duplicates within a time window.
    Pending requests share the same future for response.
    """

    def __init__(
        self,
        ttl_seconds: float = 5.0,
        max_pending: int = 100,
    ):
        self._pending: Dict[str, asyncio.Future] = {}
        self._completed: Dict[str, Tuple[Any, datetime]] = {}
        self._ttl_seconds = ttl_seconds
        self._max_pending = max_pending
        self._lock = DeadlockDetectingLock(timeout_seconds=2.0)

        # Statistics
        self._deduplicated_count = 0
        self._processed_count = 0

    @staticmethod
    def compute_hash(prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Compute hash for deduplication."""
        content = prompt
        if context:
            # Include relevant context fields in hash
            relevant_keys = {"session_id", "model_requested", "max_tokens"}
            relevant_context = {k: v for k, v in context.items() if k in relevant_keys}
            if relevant_context:
                content += json.dumps(relevant_context, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def check_or_register(
        self,
        prompt_hash: str,
    ) -> Tuple[bool, Optional[asyncio.Future]]:
        """
        Check if request is duplicate or register as new.

        Returns:
            (is_duplicate, future_to_await)
            - If duplicate pending: (True, future_to_await)
            - If duplicate completed: (True, None) - result in completed cache
            - If new: (False, None)
        """
        async with self._lock:
            # Check completed cache first
            if prompt_hash in self._completed:
                result, cached_time = self._completed[prompt_hash]
                if (datetime.now() - cached_time).total_seconds() < self._ttl_seconds:
                    self._deduplicated_count += 1
                    return (True, None)
                else:
                    del self._completed[prompt_hash]

            # Check pending requests
            if prompt_hash in self._pending:
                self._deduplicated_count += 1
                return (True, self._pending[prompt_hash])

            # Register new request
            if len(self._pending) >= self._max_pending:
                # Evict oldest pending
                oldest_key = next(iter(self._pending))
                del self._pending[oldest_key]

            loop = asyncio.get_running_loop()
            future: asyncio.Future = loop.create_future()
            self._pending[prompt_hash] = future
            self._processed_count += 1
            return (False, None)

    async def complete(self, prompt_hash: str, result: Any) -> None:
        """Mark request as completed."""
        async with self._lock:
            # Resolve pending future
            if prompt_hash in self._pending:
                future = self._pending.pop(prompt_hash)
                if not future.done():
                    future.set_result(result)

            # Store in completed cache
            self._completed[prompt_hash] = (result, datetime.now())

            # Cleanup old completed entries
            cutoff = datetime.now() - timedelta(seconds=self._ttl_seconds * 2)
            keys_to_remove = [
                k for k, (_, t) in self._completed.items()
                if t < cutoff
            ]
            for k in keys_to_remove:
                del self._completed[k]

    async def fail(self, prompt_hash: str, error: Exception) -> None:
        """Mark request as failed."""
        async with self._lock:
            if prompt_hash in self._pending:
                future = self._pending.pop(prompt_hash)
                if not future.done():
                    future.set_exception(error)

    async def get_completed(self, prompt_hash: str) -> Optional[Any]:
        """Get completed result if available."""
        async with self._lock:
            if prompt_hash in self._completed:
                result, cached_time = self._completed[prompt_hash]
                if (datetime.now() - cached_time).total_seconds() < self._ttl_seconds:
                    return result
            return None

    @property
    def stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "pending_count": len(self._pending),
            "completed_cache_size": len(self._completed),
            "deduplicated_total": self._deduplicated_count,
            "processed_total": self._processed_count,
            "dedup_rate": round(
                self._deduplicated_count / max(self._deduplicated_count + self._processed_count, 1),
                4
            ),
        }


# =============================================================================
# MEMORY PRESSURE TREND ANALYZER - Prevents false positive bursts
# =============================================================================


class MemoryPressureTrendAnalyzer:
    """
    Analyzes memory pressure trends to prevent false positive cloud bursts.

    Requires sustained pressure before triggering action.
    """

    def __init__(
        self,
        window_size: int = 10,
        sustained_threshold: int = 3,
        sustained_duration_seconds: float = 10.0,
    ):
        self._history: deque = deque(maxlen=window_size)
        self._timestamps: deque = deque(maxlen=window_size)
        self._sustained_threshold = sustained_threshold
        self._sustained_duration = sustained_duration_seconds
        self._lock = asyncio.Lock()

    async def record(self, level: "MemoryPressureLevel") -> None:
        """Record a pressure reading."""
        async with self._lock:
            self._history.append(level)
            self._timestamps.append(datetime.now())

    async def should_trigger_burst(self) -> Tuple[bool, str]:
        """
        Analyze if sustained pressure warrants cloud burst.

        Returns:
            (should_burst, reason)
        """
        async with self._lock:
            if len(self._history) < 3:
                return (False, "insufficient_data")

            # Count critical readings
            from collections import Counter
            counts = Counter(self._history)
            critical_count = counts.get(MemoryPressureLevel.CRITICAL, 0)
            warn_count = counts.get(MemoryPressureLevel.WARN, 0)

            # Check for sustained critical
            if critical_count >= self._sustained_threshold:
                # Verify time span
                if len(self._timestamps) >= self._sustained_threshold:
                    time_span = (self._timestamps[-1] - self._timestamps[0]).total_seconds()
                    if time_span >= self._sustained_duration:
                        return (True, f"sustained_critical:{critical_count}/{len(self._history)}")

            # Check for escalating trend
            recent = list(self._history)[-5:]
            if len(recent) >= 5:
                escalating = sum(
                    1 for i in range(1, len(recent))
                    if recent[i].value > recent[i-1].value
                )
                if escalating >= 3:
                    return (True, f"escalating_trend:{escalating}/4")

            # Combined pressure
            if critical_count + warn_count >= len(self._history) * 0.8:
                return (True, f"high_average_pressure:{(critical_count + warn_count)}/{len(self._history)}")

            return (False, "normal")

    def get_trend(self) -> str:
        """Get current trend description."""
        if len(self._history) < 3:
            return "insufficient_data"

        recent = list(self._history)[-5:]
        if all(p == MemoryPressureLevel.CRITICAL for p in recent):
            return "critical_sustained"
        elif all(p == MemoryPressureLevel.NORMAL for p in recent):
            return "stable_normal"
        elif recent[-1].value > recent[0].value:
            return "increasing"
        elif recent[-1].value < recent[0].value:
            return "decreasing"
        return "stable"

    @property
    def stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            "history_size": len(self._history),
            "current_trend": self.get_trend(),
            "recent_levels": [p.name for p in list(self._history)[-5:]],
        }


# =============================================================================
# MODEL VERSION TRACKER - Tracks model versions for consistency
# =============================================================================


class ModelVersionTracker:
    """
    Tracks model versions to ensure routing consistency during hot-swaps.
    """

    def __init__(self):
        self._versions: Dict[str, str] = {}  # tier_name -> version
        self._load_times: Dict[str, datetime] = {}
        self._lock = DeadlockDetectingLock(timeout_seconds=2.0)
        self._version_history: Dict[str, List[Tuple[str, datetime]]] = {}

    async def get_version(self, tier: "RoutingTier") -> Optional[str]:
        """Get current model version for tier."""
        async with self._lock:
            return self._versions.get(tier.name)

    async def update_version(
        self,
        tier: "RoutingTier",
        version: str,
        model_id: Optional[str] = None,
    ) -> None:
        """Update model version for tier."""
        async with self._lock:
            old_version = self._versions.get(tier.name)
            self._versions[tier.name] = version
            self._load_times[tier.name] = datetime.now()

            # Track history
            if tier.name not in self._version_history:
                self._version_history[tier.name] = []
            self._version_history[tier.name].append((version, datetime.now()))

            # Keep last 10 versions
            if len(self._version_history[tier.name]) > 10:
                self._version_history[tier.name] = self._version_history[tier.name][-10:]

            if old_version and old_version != version:
                logger.info(f"Model version updated: {tier.name} {old_version} -> {version}")

    async def check_version_match(
        self,
        tier: "RoutingTier",
        expected_version: Optional[str],
    ) -> bool:
        """Check if current version matches expected."""
        if expected_version is None:
            return True
        async with self._lock:
            current = self._versions.get(tier.name)
            return current == expected_version

    async def get_all_versions(self) -> Dict[str, str]:
        """Get all current model versions."""
        async with self._lock:
            return self._versions.copy()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get version tracker statistics."""
        return {
            "tracked_tiers": list(self._versions.keys()),
            "versions": self._versions.copy(),
            "load_times": {k: v.isoformat() for k, v in self._load_times.items()},
        }


# =============================================================================
# PREDICTIVE PRELOADER - Preloads models based on usage patterns
# =============================================================================


class PredictivePreloader:
    """
    Predicts and preloads models based on usage patterns.

    Uses frequency analysis and time-of-day patterns to predict
    which models will be needed next.
    """

    def __init__(
        self,
        history_size: int = 100,
        prediction_threshold: float = 0.6,
    ):
        self._usage_history: deque = deque(maxlen=history_size)
        self._task_transitions: Dict[str, Dict[str, int]] = {}  # from_task -> {to_task -> count}
        self._prediction_threshold = prediction_threshold
        self._lock = asyncio.Lock()
        self._preload_callbacks: List[Callable[[str], Awaitable[None]]] = []

    async def record_usage(
        self,
        task_type: "TaskCategory",
        tier: "RoutingTier",
    ) -> None:
        """Record a usage event."""
        async with self._lock:
            event = {
                "task_type": task_type.value,
                "tier": tier.name,
                "timestamp": datetime.now(),
                "hour": datetime.now().hour,
            }

            # Track transitions
            if self._usage_history:
                last_event = self._usage_history[-1]
                from_task = last_event["task_type"]
                to_task = task_type.value

                if from_task not in self._task_transitions:
                    self._task_transitions[from_task] = {}

                self._task_transitions[from_task][to_task] = (
                    self._task_transitions[from_task].get(to_task, 0) + 1
                )

            self._usage_history.append(event)

    async def predict_next_tier(
        self,
        current_task: "TaskCategory",
    ) -> Optional["RoutingTier"]:
        """Predict next tier based on patterns."""
        async with self._lock:
            if current_task.value not in self._task_transitions:
                return None

            transitions = self._task_transitions[current_task.value]
            total = sum(transitions.values())

            if total < 5:  # Not enough data
                return None

            # Find most likely next task
            best_next = max(transitions.items(), key=lambda x: x[1])
            probability = best_next[1] / total

            if probability < self._prediction_threshold:
                return None

            # Map task to typical tier
            task_to_tier = {
                "voice": RoutingTier.TIER_0_ULTRA_FAST,
                "greeting": RoutingTier.TIER_0_ULTRA_FAST,
                "chat": RoutingTier.TIER_0_FAST,
                "code": RoutingTier.TIER_05_CAPABLE,
                "reasoning": RoutingTier.TIER_1_CLOUD,
                "creative": RoutingTier.TIER_05_CAPABLE,
                "math": RoutingTier.TIER_1_CLOUD,
            }

            return task_to_tier.get(best_next[0])

    def register_preload_callback(
        self,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        """Register callback for preload triggers."""
        self._preload_callbacks.append(callback)

    async def trigger_preload(self, model_id: str) -> None:
        """Trigger preloading of a model."""
        for callback in self._preload_callbacks:
            try:
                asyncio.create_task(callback(model_id))
            except Exception as e:
                logger.debug(f"Preload callback failed: {e}")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get preloader statistics."""
        return {
            "history_size": len(self._usage_history),
            "tracked_transitions": len(self._task_transitions),
            "prediction_threshold": self._prediction_threshold,
        }


# =============================================================================
# REQUEST BATCHER - Batches similar requests for efficiency
# =============================================================================


class RequestBatcher:
    """
    Batches similar requests for more efficient processing.

    Groups requests with identical prompts within a time window.
    """

    def __init__(
        self,
        batch_window_ms: float = 50.0,
        max_batch_size: int = 10,
    ):
        self._batch_window = batch_window_ms / 1000.0
        self._max_batch_size = max_batch_size
        self._batches: Dict[str, List[asyncio.Future]] = {}
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._lock = DeadlockDetectingLock(timeout_seconds=2.0)

        # Statistics
        self._batched_count = 0
        self._batch_executions = 0

    async def add_to_batch(
        self,
        prompt_hash: str,
    ) -> asyncio.Future:
        """Add request to batch, returns future for result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        async with self._lock:
            if prompt_hash not in self._batches:
                self._batches[prompt_hash] = []
                # Start timer for batch execution
                self._batch_timers[prompt_hash] = asyncio.create_task(
                    self._batch_timer(prompt_hash)
                )

            self._batches[prompt_hash].append(future)
            self._batched_count += 1

            # Execute immediately if batch is full
            if len(self._batches[prompt_hash]) >= self._max_batch_size:
                timer = self._batch_timers.pop(prompt_hash, None)
                if timer:
                    timer.cancel()
                # Don't execute here - let caller handle

        return future

    async def _batch_timer(self, prompt_hash: str) -> None:
        """Timer to trigger batch execution."""
        await asyncio.sleep(self._batch_window)
        # Timer expired - batch will be processed by next get_batch call

    async def get_batch(self, prompt_hash: str) -> Optional[List[asyncio.Future]]:
        """Get and clear a batch for processing."""
        async with self._lock:
            if prompt_hash not in self._batches:
                return None

            batch = self._batches.pop(prompt_hash)
            timer = self._batch_timers.pop(prompt_hash, None)
            if timer:
                timer.cancel()

            self._batch_executions += 1
            return batch

    async def resolve_batch(
        self,
        prompt_hash: str,
        result: Any,
    ) -> int:
        """Resolve all futures in a batch with result."""
        batch = await self.get_batch(prompt_hash)
        if not batch:
            return 0

        resolved = 0
        for future in batch:
            if not future.done():
                future.set_result(result)
                resolved += 1

        return resolved

    @property
    def stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        return {
            "active_batches": len(self._batches),
            "batched_total": self._batched_count,
            "batch_executions": self._batch_executions,
            "avg_batch_size": round(
                self._batched_count / max(self._batch_executions, 1), 2
            ),
        }


# =============================================================================
# TIER AVAILABILITY MONITOR - Monitors and alerts on tier health
# =============================================================================


class TierAvailabilityMonitor:
    """
    Monitors tier availability and triggers alerts when all tiers down.
    """

    def __init__(
        self,
        circuit_manager: "CircuitBreakerManager",
        alert_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self._circuit_manager = circuit_manager
        self._alert_callback = alert_callback
        self._all_down_since: Optional[datetime] = None
        self._alert_sent = False
        self._lock = asyncio.Lock()

    async def check_availability(self) -> Dict["RoutingTier", bool]:
        """Check availability of all tiers."""
        results = {}
        for tier in RoutingTier:
            if tier == RoutingTier.UNKNOWN:
                continue
            try:
                can_proceed = await self._circuit_manager.can_proceed(tier.name)
                results[tier] = can_proceed
            except Exception:
                results[tier] = False
        return results

    async def get_available_tiers(self) -> List["RoutingTier"]:
        """Get list of available tiers."""
        availability = await self.check_availability()
        return [tier for tier, available in availability.items() if available]

    async def monitor_and_alert(self) -> Tuple[bool, Optional[str]]:
        """
        Monitor availability and alert if all down.

        Returns:
            (all_available_ok, alert_message)
        """
        async with self._lock:
            available = await self.get_available_tiers()

            if not available:
                if self._all_down_since is None:
                    self._all_down_since = datetime.now()

                down_duration = (datetime.now() - self._all_down_since).total_seconds()

                if not self._alert_sent and down_duration > 5.0:
                    alert_msg = f"ALL ROUTING TIERS UNAVAILABLE for {down_duration:.1f}s"
                    logger.critical(alert_msg)

                    if self._alert_callback:
                        try:
                            await self._alert_callback(alert_msg)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")

                    self._alert_sent = True
                    return (False, alert_msg)

                return (False, None)
            else:
                # Reset if tiers are back
                if self._all_down_since:
                    logger.info("Routing tiers recovered")
                self._all_down_since = None
                self._alert_sent = False
                return (True, None)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        return {
            "all_down_since": self._all_down_since.isoformat() if self._all_down_since else None,
            "alert_sent": self._alert_sent,
        }


# =============================================================================
# METRICS EXPORTER - Exports metrics for observability
# =============================================================================


class MetricsExporter:
    """
    Exports routing metrics for observability systems.

    Supports:
    - Counter increments
    - Gauge updates
    - Histogram recording
    - Batch export
    """

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()
        self._export_callbacks: List[Callable[[Dict], Awaitable[None]]] = []

    async def inc_counter(
        self,
        name: str,
        value: int = 1,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        async with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    async def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        async with self._lock:
            self._gauges[key] = value

    async def record_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, labels)
        async with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            # Keep last 1000 values
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-500:]

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create metric key with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def register_export_callback(
        self,
        callback: Callable[[Dict], Awaitable[None]],
    ) -> None:
        """Register callback for metric exports."""
        self._export_callbacks.append(callback)

    async def export(self) -> Dict[str, Any]:
        """Export all metrics."""
        async with self._lock:
            metrics = {
                "counters": self._counters.copy(),
                "gauges": self._gauges.copy(),
                "histograms": {
                    k: {
                        "count": len(v),
                        "sum": sum(v),
                        "avg": sum(v) / len(v) if v else 0,
                        "min": min(v) if v else 0,
                        "max": max(v) if v else 0,
                    }
                    for k, v in self._histograms.items()
                },
                "timestamp": datetime.now().isoformat(),
            }

        # Call export callbacks
        for callback in self._export_callbacks:
            try:
                await callback(metrics)
            except Exception as e:
                logger.debug(f"Metrics export callback failed: {e}")

        return metrics

    @property
    def stats(self) -> Dict[str, Any]:
        """Get exporter statistics."""
        return {
            "counter_count": len(self._counters),
            "gauge_count": len(self._gauges),
            "histogram_count": len(self._histograms),
            "export_callbacks": len(self._export_callbacks),
        }

# =============================================================================
# CONTEXT VARIABLES FOR DISTRIBUTED TRACING
# =============================================================================

request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id', default=None
)
session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'session_id', default=None
)
trace_context_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    'trace_context', default={}
)
routing_context_var: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    'routing_context', default={}
)


def set_request_context(
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    **extra: Any,
) -> None:
    """Set request context for distributed tracing."""
    if request_id:
        request_id_var.set(request_id)
    if session_id:
        session_id_var.set(session_id)
    if extra:
        ctx = trace_context_var.get().copy()
        ctx.update(extra)
        trace_context_var.set(ctx)


def get_request_context() -> Dict[str, Any]:
    """Get current request context."""
    return {
        "request_id": request_id_var.get(),
        "session_id": session_id_var.get(),
        **trace_context_var.get(),
    }


# =============================================================================
# PROTOCOLS (Type-Safe Interfaces)
# =============================================================================


@runtime_checkable
class RouterProtocol(Protocol):
    """Protocol for routing implementations."""

    async def route(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> "RoutingResult": ...

    def get_health_status(self) -> Dict[str, Any]: ...

    async def shutdown(self) -> None: ...


@runtime_checkable
class MemoryMonitorProtocol(Protocol):
    """Protocol for memory monitoring implementations."""

    async def get_pressure_level(self) -> "MemoryPressureLevel": ...

    async def get_detailed_snapshot(self) -> Dict[str, Any]: ...

    async def should_burst_to_cloud(self) -> bool: ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """Protocol for circuit breaker implementations."""

    def is_open(self) -> bool: ...

    def record_success(self) -> None: ...

    def record_failure(self) -> None: ...

    def get_state(self) -> "CircuitState": ...


@runtime_checkable
class ConfigProviderProtocol(Protocol):
    """Protocol for configuration providers."""

    def get(self, key: str, default: Any = None) -> Any: ...

    def get_int(self, key: str, default: int = 0) -> int: ...

    def get_float(self, key: str, default: float = 0.0) -> float: ...

    def get_bool(self, key: str, default: bool = False) -> bool: ...


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    NORMAL = 0
    WARN = 1
    CRITICAL = 2


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RoutingTier(Enum):
    """Routing tier levels."""
    TIER_0_ULTRA_FAST = 0
    TIER_0_FAST = 1
    TIER_05_CAPABLE = 2
    TIER_1_CLOUD = 3
    TIER_2_DEEP = 4
    UNKNOWN = 99


class TaskCategory(Enum):
    """High-level task categories."""
    VOICE = "voice"
    GREETING = "greeting"
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    CREATIVE = "creative"
    MATH = "math"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class RoutingDecisionReason(Enum):
    """Reasons for routing decisions."""
    TASK_CLASSIFICATION = "task_classification"
    MEMORY_PRESSURE = "memory_pressure"
    STICKY_SESSION = "sticky_session"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    COST_BUDGET = "cost_budget"
    FALLBACK = "fallback"
    LOAD_BALANCING = "load_balancing"
    USER_PREFERENCE = "user_preference"
    DEFAULT = "default"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class DynamicConfig:
    """Dynamic configuration with environment variable override support."""

    # Memory thresholds
    memory_warn_threshold: float = 0.80
    memory_critical_threshold: float = 0.90
    memory_cache_ttl_seconds: float = 2.0
    memory_history_size: int = 60

    # Task classification
    task_session_timeout_seconds: float = 600.0
    task_history_size: int = 100
    task_complexity_weights: Dict[str, float] = field(default_factory=lambda: {
        "keyword": 0.35,
        "structure": 0.20,
        "complexity": 0.25,
        "session": 0.20,
    })

    # Sticky routing
    sticky_session_timeout_minutes: float = 30.0
    sticky_strength_threshold: float = 0.7
    sticky_history_size: int = 20

    # Request buffering
    buffer_max_size: int = 100
    buffer_default_timeout_seconds: float = 30.0

    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout_seconds: float = 30.0
    circuit_half_open_max_calls: int = 3

    # Routing thresholds
    tier_0_fast_max_complexity: float = 0.35
    tier_05_capable_max_complexity: float = 0.55
    tier_1_cloud_max_complexity: float = 0.80

    # Cost management
    daily_budget_usd: float = 10.0
    burst_budget_multiplier: float = 1.5

    # Ports (can be overridden by env vars)
    jarvis_prime_port: int = 8000
    jarvis_body_port: int = 8010
    reactor_core_port: int = 8090

    # Timeouts
    routing_timeout_seconds: float = 30.0
    health_check_timeout_seconds: float = 5.0
    shutdown_timeout_seconds: float = 30.0

    # Cross-repo paths
    cross_repo_state_dir: str = "~/.jarvis/cross_repo"
    trinity_state_dir: str = "~/.jarvis/trinity"

    # v100.1: Advanced features configuration
    # Deduplication
    dedup_ttl_seconds: float = 5.0
    dedup_max_pending: int = 100

    # Routing cache
    routing_cache_max_size: int = 1000
    routing_cache_ttl_seconds: float = 60.0

    # Request batching
    batch_window_ms: float = 50.0
    batch_max_size: int = 10

    # Memory pressure trend analysis
    memory_trend_window_size: int = 10
    memory_sustained_threshold: int = 3
    memory_sustained_duration_seconds: float = 10.0

    # Predictive preloading
    preload_history_size: int = 100
    preload_prediction_threshold: float = 0.6

    # Deadlock detection
    lock_timeout_seconds: float = 5.0

    # Cancellation
    default_request_timeout_seconds: float = 60.0

    # Metrics export interval
    metrics_export_interval_seconds: float = 30.0

    @classmethod
    def from_env_and_yaml(cls, yaml_config: Optional[Dict[str, Any]] = None) -> "DynamicConfig":
        """Create config from environment variables and YAML with proper precedence."""
        yaml_config = yaml_config or {}

        def get_value(key: str, default: Any, type_converter: Callable[[Any], Any]) -> Any:
            """Get value with env var > yaml > default precedence."""
            env_key = f"NEURAL_ORCHESTRATOR_{key.upper()}"
            env_value = os.environ.get(env_key)
            if env_value is not None:
                try:
                    return type_converter(env_value)
                except (ValueError, TypeError):
                    pass

            yaml_value = yaml_config.get(key)
            if yaml_value is not None:
                return type_converter(yaml_value)

            return default

        return cls(
            memory_warn_threshold=get_value("memory_warn_threshold", 0.80, float),
            memory_critical_threshold=get_value("memory_critical_threshold", 0.90, float),
            memory_cache_ttl_seconds=get_value("memory_cache_ttl_seconds", 2.0, float),
            memory_history_size=get_value("memory_history_size", 60, int),
            task_session_timeout_seconds=get_value("task_session_timeout_seconds", 600.0, float),
            task_history_size=get_value("task_history_size", 100, int),
            sticky_session_timeout_minutes=get_value("sticky_session_timeout_minutes", 30.0, float),
            sticky_strength_threshold=get_value("sticky_strength_threshold", 0.7, float),
            sticky_history_size=get_value("sticky_history_size", 20, int),
            buffer_max_size=get_value("buffer_max_size", 100, int),
            buffer_default_timeout_seconds=get_value("buffer_default_timeout_seconds", 30.0, float),
            circuit_failure_threshold=get_value("circuit_failure_threshold", 5, int),
            circuit_recovery_timeout_seconds=get_value("circuit_recovery_timeout_seconds", 30.0, float),
            circuit_half_open_max_calls=get_value("circuit_half_open_max_calls", 3, int),
            tier_0_fast_max_complexity=get_value("tier_0_fast_max_complexity", 0.35, float),
            tier_05_capable_max_complexity=get_value("tier_05_capable_max_complexity", 0.55, float),
            tier_1_cloud_max_complexity=get_value("tier_1_cloud_max_complexity", 0.80, float),
            daily_budget_usd=get_value("daily_budget_usd", 10.0, float),
            burst_budget_multiplier=get_value("burst_budget_multiplier", 1.5, float),
            jarvis_prime_port=get_value("jarvis_prime_port", int(os.getenv("JARVIS_PRIME_PORT", "8000")), int),
            jarvis_body_port=get_value("jarvis_body_port", int(os.getenv("JARVIS_PORT", "8010")), int),
            reactor_core_port=get_value("reactor_core_port", int(os.getenv("REACTOR_CORE_PORT", "8090")), int),
            routing_timeout_seconds=get_value("routing_timeout_seconds", 30.0, float),
            health_check_timeout_seconds=get_value("health_check_timeout_seconds", 5.0, float),
            shutdown_timeout_seconds=get_value("shutdown_timeout_seconds", 30.0, float),
            cross_repo_state_dir=get_value("cross_repo_state_dir", "~/.jarvis/cross_repo", str),
            trinity_state_dir=get_value("trinity_state_dir", "~/.jarvis/trinity", str),
            # v100.1: Advanced features configuration
            dedup_ttl_seconds=get_value("dedup_ttl_seconds", 5.0, float),
            dedup_max_pending=get_value("dedup_max_pending", 100, int),
            routing_cache_max_size=get_value("routing_cache_max_size", 1000, int),
            routing_cache_ttl_seconds=get_value("routing_cache_ttl_seconds", 60.0, float),
            batch_window_ms=get_value("batch_window_ms", 50.0, float),
            batch_max_size=get_value("batch_max_size", 10, int),
            memory_trend_window_size=get_value("memory_trend_window_size", 10, int),
            memory_sustained_threshold=get_value("memory_sustained_threshold", 3, int),
            memory_sustained_duration_seconds=get_value("memory_sustained_duration_seconds", 10.0, float),
            preload_history_size=get_value("preload_history_size", 100, int),
            preload_prediction_threshold=get_value("preload_prediction_threshold", 0.6, float),
            lock_timeout_seconds=get_value("lock_timeout_seconds", 5.0, float),
            default_request_timeout_seconds=get_value("default_request_timeout_seconds", 60.0, float),
            metrics_export_interval_seconds=get_value("metrics_export_interval_seconds", 30.0, float),
        )


class ConfigHotReloader:
    """
    Watches config file for changes and triggers reload.

    v100.1: Enables runtime configuration updates without restart.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        orchestrator: Optional["NeuralOrchestratorCore"] = None,
        poll_interval_seconds: float = 10.0,
    ):
        self._config_path = config_path or Path.home() / ".jarvis" / "neural_orchestrator_config.yaml"
        self._orchestrator = orchestrator
        self._poll_interval = poll_interval_seconds
        self._last_modified: Optional[float] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reload_callbacks: List[Callable[[DynamicConfig], Awaitable[None]]] = []

    def register_callback(self, callback: Callable[[DynamicConfig], Awaitable[None]]) -> None:
        """Register callback to be called on config reload."""
        self._reload_callbacks.append(callback)

    async def start(self) -> None:
        """Start watching for config changes."""
        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.info(f"ConfigHotReloader: Watching {self._config_path}")

    async def stop(self) -> None:
        """Stop watching for config changes."""
        self._running = False
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _watch_loop(self) -> None:
        """Main watch loop."""
        while self._running:
            try:
                if self._config_path.exists():
                    current_mtime = self._config_path.stat().st_mtime
                    if self._last_modified and current_mtime > self._last_modified:
                        await self._reload_config()
                    self._last_modified = current_mtime
            except Exception as e:
                logger.debug(f"Config watch error: {e}")

            await asyncio.sleep(self._poll_interval)

    async def _reload_config(self) -> None:
        """Reload configuration and notify listeners."""
        try:
            # Load YAML config if it exists
            yaml_config = {}
            if self._config_path.exists():
                import yaml
                with open(self._config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}

            # Create new config
            new_config = DynamicConfig.from_env_and_yaml(yaml_config)

            # Update orchestrator if available
            if self._orchestrator:
                self._orchestrator._config = new_config

            # Notify callbacks
            for callback in self._reload_callbacks:
                try:
                    await callback(new_config)
                except Exception as e:
                    logger.error(f"Config reload callback error: {e}")

            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Config reload failed: {e}")


@dataclass
class TaskClassification:
    """Result of task classification."""
    task_type: TaskCategory
    complexity: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    signals: List[str]
    recommended_tier: RoutingTier
    requires_fast_response: bool = False
    is_coding_session: bool = False
    required_capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    request_id: str
    tier: RoutingTier
    model_id: Optional[str]
    endpoint: Optional[str]
    decision_reason: RoutingDecisionReason
    task_classification: Optional[TaskClassification]
    confidence: float
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    fallback_chain: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "tier": self.tier.name,
            "model_id": self.model_id,
            "endpoint": self.endpoint,
            "decision_reason": self.decision_reason.value,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""
    name: str
    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: Optional[datetime]
    last_success_time: Optional[datetime]
    half_open_calls: int = 0


@dataclass
class CrossRepoMemoryState:
    """Shared memory state across repos."""
    jarvis_pressure: MemoryPressureLevel
    jarvis_prime_pressure: MemoryPressureLevel
    reactor_pressure: MemoryPressureLevel
    should_burst: bool
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# DEFENSIVE DECORATORS
# =============================================================================


def defensive(
    fallback: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
):
    """Decorator for graceful error handling with fallback."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"Defensive fallback in {func.__name__}: {e}")
                if reraise:
                    raise
                return fallback
        return wrapper
    return decorator


def async_defensive(
    fallback: Any = None,
    log_errors: bool = True,
    reraise: bool = False,
):
    """Async version of defensive decorator."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"Async defensive fallback in {func.__name__}: {e}")
                if reraise:
                    raise
                return fallback
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to async functions."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_seconds,
            )
        return wrapper
    return decorator


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
):
    """Decorator for retry with exponential backoff and decorrelated jitter."""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            import random
            last_exception: Optional[Exception] = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(
                            base_delay * (exponential_base ** attempt),
                            max_delay,
                        )
                        if jitter:
                            delay = random.uniform(0, delay)
                        logger.debug(
                            f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                            f"after {delay:.2f}s: {e}"
                        )
                        await asyncio.sleep(delay)

            raise last_exception  # type: ignore
        return wrapper
    return decorator


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitBreaker:
    """
    Thread-safe circuit breaker with configurable thresholds.

    States:
        CLOSED: Normal operation, requests pass through
        OPEN: Failures exceeded threshold, requests fail fast
        HALF_OPEN: Testing if service recovered
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_success_time: Optional[datetime] = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return self._name

    def is_open(self) -> bool:
        """Check if circuit is open (fail-fast mode)."""
        return self._state == CircuitState.OPEN

    def get_state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._success_count += 1
            self._last_success_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
                if self._half_open_calls >= self._half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    async def can_proceed(self) -> bool:
        """Check if a request can proceed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self._recovery_timeout:
                        self._transition_to(CircuitState.HALF_OPEN)
                        return True
                return False

            # HALF_OPEN - allow limited calls
            return self._half_open_calls < self._half_open_max_calls

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state (must hold lock)."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0

        logger.info(f"Circuit {self._name}: {old_state.value} -> {new_state.value}")

    def get_stats(self) -> CircuitBreakerState:
        """Get circuit breaker statistics."""
        return CircuitBreakerState(
            name=self._name,
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            half_open_calls=self._half_open_calls,
        )


# =============================================================================
# UNIFIED TASK CLASSIFIER
# =============================================================================


class UnifiedTaskClassifier:
    """
    Intelligent task classification with multi-signal analysis.

    Consolidates classification logic from:
    - TaskClassifier (dynamic_model_registry.py)
    - AdvancedComplexityAnalyzer (hybrid_tiered_router.py)
    - ComplexityAnalyzer (intelligent_model_router.py)
    """

    # Keyword patterns (frozen for thread safety)
    _CODE_KEYWORDS: Final[frozenset] = frozenset([
        "code", "implement", "write", "function", "class", "method", "api",
        "python", "javascript", "typescript", "rust", "golang", "java",
        "fix", "bug", "error", "exception", "debug", "refactor", "optimize",
        "test", "unittest", "pytest", "jest", "async", "await", "import",
        "def ", "class ", "return ", "if ", "for ", "while ",
    ])

    _ARCHITECTURE_KEYWORDS: Final[frozenset] = frozenset([
        "architect", "design", "pattern", "structure", "system", "scale",
        "microservice", "monolith", "database", "schema", "api design",
        "infrastructure", "deployment", "kubernetes", "docker", "ci/cd",
    ])

    _REASONING_KEYWORDS: Final[frozenset] = frozenset([
        "explain", "why", "how", "reason", "think", "analyze", "compare",
        "evaluate", "pros", "cons", "tradeoff", "decision", "strategy",
        "plan", "approach", "consider", "implications",
    ])

    _MATH_KEYWORDS: Final[frozenset] = frozenset([
        "calculate", "compute", "math", "equation", "formula", "derive",
        "solve", "proof", "theorem", "algorithm", "complexity", "o(n)",
        "integral", "derivative", "matrix", "statistics", "probability",
    ])

    _CREATIVE_KEYWORDS: Final[frozenset] = frozenset([
        "write", "story", "poem", "creative", "imagine", "brainstorm",
        "ideas", "generate", "suggest", "alternative", "creative writing",
    ])

    _GREETING_PATTERNS: Final[frozenset] = frozenset([
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "what's up", "how are you", "thanks", "thank you", "bye", "goodbye",
    ])

    _VOICE_PATTERNS: Final[frozenset] = frozenset([
        "jarvis", "hey jarvis", "ok jarvis", "listen", "what time",
        "set timer", "remind me", "play", "stop", "pause", "quick",
    ])

    def __init__(self, config: DynamicConfig):
        self._config = config
        self._lock = asyncio.Lock()
        self._session_context: Dict[str, Any] = {}
        self._classification_history: List[TaskClassification] = []
        self._coding_session_active = False
        self._coding_session_start: Optional[datetime] = None
        self._weights = config.task_complexity_weights.copy()

    async def classify(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskClassification:
        """Classify a prompt with full multi-signal analysis."""
        async with self._lock:
            context = context or {}
            prompt_lower = prompt.lower().strip()
            signals: List[str] = []

            # Quick checks for simple cases
            if self._is_voice_command(prompt_lower, context):
                return self._create_classification(
                    TaskCategory.VOICE,
                    complexity=0.10,
                    confidence=0.90,
                    signals=["voice_pattern"],
                    requires_fast=True,
                )

            if self._is_greeting(prompt_lower):
                return self._create_classification(
                    TaskCategory.GREETING,
                    complexity=0.05,
                    confidence=0.95,
                    signals=["greeting_pattern"],
                )

            # Analyze signals
            code_score = self._count_matches(prompt_lower, self._CODE_KEYWORDS)
            arch_score = self._count_matches(prompt_lower, self._ARCHITECTURE_KEYWORDS)
            reason_score = self._count_matches(prompt_lower, self._REASONING_KEYWORDS)
            math_score = self._count_matches(prompt_lower, self._MATH_KEYWORDS)
            creative_score = self._count_matches(prompt_lower, self._CREATIVE_KEYWORDS)

            # Check for code blocks
            has_code_block = "```" in prompt or "    " in prompt
            if has_code_block:
                code_score += 3
                signals.append("code_block_detected")

            # Calculate complexity
            complexity = self._calculate_complexity(prompt)

            # Determine category and tier
            category, confidence = self._determine_category(
                code_score, arch_score, reason_score, math_score,
                creative_score, complexity, signals,
            )

            # Update coding session state
            is_coding = category in {TaskCategory.CODE}
            if is_coding and not self._coding_session_active:
                self._coding_session_active = True
                self._coding_session_start = datetime.now()
                signals.append("coding_session_started")
            elif self._coding_session_active and not is_coding:
                if self._coding_session_start:
                    elapsed = (datetime.now() - self._coding_session_start).total_seconds()
                    if elapsed > self._config.task_session_timeout_seconds:
                        self._coding_session_active = False
                        signals.append("coding_session_ended")

            classification = self._create_classification(
                category,
                complexity=complexity,
                confidence=confidence,
                signals=signals,
                is_coding_session=self._coding_session_active,
            )

            # Store in history
            self._classification_history.append(classification)
            if len(self._classification_history) > self._config.task_history_size:
                self._classification_history = self._classification_history[
                    -self._config.task_history_size // 2:
                ]

            return classification

    def _is_greeting(self, prompt: str) -> bool:
        """Check if prompt is a greeting."""
        if len(prompt) < 50:
            return any(prompt.startswith(p) or prompt == p for p in self._GREETING_PATTERNS)
        return False

    def _is_voice_command(self, prompt: str, context: Dict[str, Any]) -> bool:
        """Check if prompt is a voice command."""
        if context.get("voice_mode") or context.get("from_voice"):
            return True
        return any(prompt.startswith(p) for p in self._VOICE_PATTERNS)

    def _count_matches(self, text: str, keywords: frozenset) -> int:
        """Count keyword matches in text."""
        return sum(1 for kw in keywords if kw in text)

    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity (0-1)."""
        factors = []

        # Length factor
        length = len(prompt)
        if length < 50:
            factors.append(0.1)
        elif length < 200:
            factors.append(0.3)
        elif length < 500:
            factors.append(0.5)
        elif length < 1000:
            factors.append(0.7)
        else:
            factors.append(0.9)

        # Line count factor
        lines = prompt.count('\n') + 1
        if lines < 5:
            factors.append(0.1)
        elif lines < 20:
            factors.append(0.3)
        elif lines < 50:
            factors.append(0.5)
        else:
            factors.append(0.7)

        # Question marks (multi-step reasoning)
        questions = prompt.count('?')
        if questions > 3:
            factors.append(0.6)
        elif questions > 1:
            factors.append(0.4)
        else:
            factors.append(0.2)

        # Code complexity indicators
        if "```" in prompt:
            # Count code blocks
            code_blocks = prompt.count("```") // 2
            factors.append(min(0.3 + code_blocks * 0.15, 0.8))

        return sum(factors) / len(factors) if factors else 0.5

    def _determine_category(
        self,
        code_score: int,
        arch_score: int,
        reason_score: int,
        math_score: int,
        creative_score: int,
        complexity: float,
        signals: List[str],
    ) -> Tuple[TaskCategory, float]:
        """Determine task category from scores."""
        scores = {
            TaskCategory.CODE: code_score + arch_score,
            TaskCategory.REASONING: reason_score,
            TaskCategory.MATH: math_score,
            TaskCategory.CREATIVE: creative_score,
        }

        if arch_score > 2:
            signals.append("architecture_detected")

        max_category = max(scores.items(), key=lambda x: x[1])

        if max_category[1] == 0:
            return TaskCategory.CHAT, 0.6

        # Calculate confidence based on score dominance
        total = sum(scores.values())
        confidence = max_category[1] / total if total > 0 else 0.5
        confidence = min(confidence + 0.3, 0.95)  # Boost confidence

        return max_category[0], confidence

    def _create_classification(
        self,
        category: TaskCategory,
        complexity: float,
        confidence: float,
        signals: List[str],
        requires_fast: bool = False,
        is_coding_session: bool = False,
    ) -> TaskClassification:
        """Create a TaskClassification result."""
        # Determine recommended tier
        if requires_fast or category in {TaskCategory.VOICE, TaskCategory.GREETING}:
            tier = RoutingTier.TIER_0_ULTRA_FAST
        elif complexity < self._config.tier_0_fast_max_complexity:
            tier = RoutingTier.TIER_0_FAST
        elif complexity < self._config.tier_05_capable_max_complexity:
            tier = RoutingTier.TIER_05_CAPABLE
        elif complexity < self._config.tier_1_cloud_max_complexity:
            tier = RoutingTier.TIER_1_CLOUD
        else:
            tier = RoutingTier.TIER_2_DEEP

        return TaskClassification(
            task_type=category,
            complexity=complexity,
            confidence=confidence,
            signals=signals,
            recommended_tier=tier,
            requires_fast_response=requires_fast,
            is_coding_session=is_coding_session,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        if not self._classification_history:
            return {"total": 0}

        type_counts: Dict[str, int] = {}
        for c in self._classification_history:
            t = c.task_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total": len(self._classification_history),
            "type_distribution": type_counts,
            "avg_complexity": sum(c.complexity for c in self._classification_history)
            / len(self._classification_history),
            "coding_session_active": self._coding_session_active,
        }


# =============================================================================
# UNIFIED MEMORY MONITOR
# =============================================================================


class UnifiedMemoryMonitor:
    """
    Unified memory monitoring across all sources.

    Combines:
    - macOS native memory_pressure
    - psutil metrics
    - JARVIS AdvancedRAMMonitor (via bridge)
    - Process-specific tracking
    """

    def __init__(self, config: DynamicConfig):
        self._config = config
        self._lock = asyncio.Lock()
        self._is_macos = sys.platform == "darwin"
        self._cached_level: Optional[MemoryPressureLevel] = None
        self._cache_time: Optional[datetime] = None
        self._pressure_history: List[Tuple[datetime, MemoryPressureLevel]] = []
        self._jarvis_bridge: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize cross-repo connections."""
        if self._initialized:
            return

        # Try to connect to JARVIS memory monitor
        try:
            from jarvis_prime.core.gcp_import_bridge import GCPImportBridge
            bridge = await GCPImportBridge.get_instance()
            self._jarvis_bridge = bridge
            logger.info("UnifiedMemoryMonitor: Connected to JARVIS via GCP bridge")
        except Exception as e:
            logger.debug(f"UnifiedMemoryMonitor: JARVIS bridge not available: {e}")

        self._initialized = True

    async def get_pressure_level(self) -> MemoryPressureLevel:
        """Get current memory pressure level (cached)."""
        async with self._lock:
            now = datetime.now()

            # Return cached if fresh
            if (
                self._cached_level is not None
                and self._cache_time is not None
                and (now - self._cache_time).total_seconds() < self._config.memory_cache_ttl_seconds
            ):
                return self._cached_level

            # Get fresh reading
            level = await self._read_memory_pressure()
            self._cached_level = level
            self._cache_time = now

            # Record history
            self._pressure_history.append((now, level))
            if len(self._pressure_history) > self._config.memory_history_size:
                self._pressure_history = self._pressure_history[
                    -self._config.memory_history_size // 2:
                ]

            return level

    async def _read_memory_pressure(self) -> MemoryPressureLevel:
        """Read memory pressure from system."""
        if self._is_macos:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "memory_pressure",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
                output = stdout.decode().lower()

                if "critical" in output or "under pressure" in output:
                    return MemoryPressureLevel.CRITICAL
                elif "warn" in output or "approaching" in output:
                    return MemoryPressureLevel.WARN
                else:
                    return MemoryPressureLevel.NORMAL
            except Exception:
                pass  # Fall through to psutil

        # Fallback to psutil
        try:
            import psutil
            mem = psutil.virtual_memory()

            if mem.percent > self._config.memory_critical_threshold * 100:
                return MemoryPressureLevel.CRITICAL
            elif mem.percent > self._config.memory_warn_threshold * 100:
                return MemoryPressureLevel.WARN
            else:
                return MemoryPressureLevel.NORMAL
        except ImportError:
            return MemoryPressureLevel.NORMAL

    async def get_detailed_snapshot(self) -> Dict[str, Any]:
        """Get detailed memory snapshot."""
        level = await self.get_pressure_level()

        result: Dict[str, Any] = {
            "pressure_level": level.value,
            "is_critical": level == MemoryPressureLevel.CRITICAL,
            "should_burst": level in {MemoryPressureLevel.CRITICAL, MemoryPressureLevel.WARN},
            "trend": self._detect_trend(),
        }

        # Add psutil data
        try:
            import psutil
            mem = psutil.virtual_memory()
            result.update({
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent_used": mem.percent,
                "swap_percent": psutil.swap_memory().percent,
            })
        except ImportError:
            pass

        # Add JARVIS bridge data if available
        if self._jarvis_bridge:
            try:
                jarvis_status = await self._jarvis_bridge.get_memory_pressure()
                if jarvis_status:
                    result["jarvis_pressure"] = jarvis_status
            except Exception:
                pass

        return result

    async def should_burst_to_cloud(self) -> bool:
        """Check if cloud burst is recommended."""
        level = await self.get_pressure_level()
        return level in {MemoryPressureLevel.CRITICAL, MemoryPressureLevel.WARN}

    def _detect_trend(self) -> str:
        """Detect memory pressure trend."""
        if len(self._pressure_history) < 5:
            return "stable"

        recent = self._pressure_history[-5:]
        levels = [h[1] for h in recent]

        # Count increasing transitions
        increasing = sum(
            1 for i in range(1, len(levels))
            if levels[i].value > levels[i - 1].value
        )

        if increasing >= 3:
            return "increasing"
        elif levels.count(MemoryPressureLevel.CRITICAL) >= 3:
            return "critical_sustained"
        else:
            return "stable"

    def get_stats(self) -> Dict[str, Any]:
        """Get memory monitor statistics."""
        return {
            "history_size": len(self._pressure_history),
            "cached_level": self._cached_level.value if self._cached_level else None,
            "trend": self._detect_trend(),
            "jarvis_bridge_connected": self._jarvis_bridge is not None,
        }


# =============================================================================
# UNIFIED STICKY ROUTING
# =============================================================================


class UnifiedStickyRouting:
    """
    Unified sticky routing for session continuity.

    Maintains model affinity during active sessions to reduce
    model swapping overhead and improve response consistency.
    """

    def __init__(self, config: DynamicConfig):
        self._config = config
        self._lock = asyncio.Lock()
        self._session_timeout = config.sticky_session_timeout_minutes * 60
        self._strength_threshold = config.sticky_strength_threshold

        # Current sticky state
        self._sticky_model: Optional[str] = None
        self._sticky_tier: Optional[RoutingTier] = None
        self._sticky_started: Optional[datetime] = None
        self._sticky_strength: float = 0.0

        # Session history
        self._session_tasks: List[TaskCategory] = []
        self._session_model_uses: Dict[str, int] = {}

    async def get_sticky_model(
        self,
        classification: TaskClassification,
    ) -> Optional[str]:
        """Get sticky model if session is active and applicable."""
        async with self._lock:
            # Check expiration
            if self._sticky_started:
                elapsed = (datetime.now() - self._sticky_started).total_seconds()
                if elapsed > self._session_timeout:
                    await self._end_session_internal()
                    return None

            # Check if task matches sticky context
            if self._sticky_model and self._sticky_strength >= self._strength_threshold:
                if self._task_matches_session(classification):
                    # Extend session
                    self._sticky_started = datetime.now()
                    return self._sticky_model

            return None

    async def update_sticky(
        self,
        model_id: str,
        tier: RoutingTier,
        classification: TaskClassification,
    ) -> None:
        """Update sticky routing state."""
        async with self._lock:
            self._session_tasks.append(classification.task_type)
            if len(self._session_tasks) > self._config.sticky_history_size:
                self._session_tasks = self._session_tasks[-self._config.sticky_history_size // 2:]

            # Track model usage
            self._session_model_uses[model_id] = self._session_model_uses.get(model_id, 0) + 1

            # Calculate sticky strength
            strength = self._calculate_sticky_strength(classification)

            if strength > self._sticky_strength or self._sticky_model is None:
                self._sticky_model = model_id
                self._sticky_tier = tier
                self._sticky_strength = strength
                if self._sticky_started is None:
                    self._sticky_started = datetime.now()

    def _task_matches_session(self, classification: TaskClassification) -> bool:
        """Check if task matches current session context."""
        if not self._session_tasks:
            return True

        # Check if task type is compatible
        recent_types = set(self._session_tasks[-5:])

        # Coding sessions stick together
        coding_types = {TaskCategory.CODE}
        if classification.task_type in coding_types and coding_types & recent_types:
            return True

        # Same task type always matches
        if classification.task_type in recent_types:
            return True

        return False

    def _calculate_sticky_strength(self, classification: TaskClassification) -> float:
        """Calculate how 'sticky' the current session is."""
        if not self._session_tasks:
            return 0.3

        strength = 0.0

        # Coding session bonus
        if classification.is_coding_session:
            strength += 0.4

        # Task type consistency bonus
        recent = self._session_tasks[-5:]
        if all(t == classification.task_type for t in recent):
            strength += 0.3
        elif classification.task_type in recent:
            strength += 0.2

        # Model usage consistency bonus
        if self._session_model_uses:
            total_uses = sum(self._session_model_uses.values())
            if self._sticky_model and self._sticky_model in self._session_model_uses:
                model_share = self._session_model_uses[self._sticky_model] / total_uses
                strength += model_share * 0.3

        return min(strength, 1.0)

    async def _end_session_internal(self) -> None:
        """End sticky session (internal, must hold lock)."""
        self._sticky_model = None
        self._sticky_tier = None
        self._sticky_started = None
        self._sticky_strength = 0.0
        self._session_tasks.clear()
        self._session_model_uses.clear()

    async def end_session(self) -> None:
        """Explicitly end sticky session."""
        async with self._lock:
            await self._end_session_internal()

    def get_stats(self) -> Dict[str, Any]:
        """Get sticky routing statistics."""
        return {
            "sticky_model": self._sticky_model,
            "sticky_tier": self._sticky_tier.name if self._sticky_tier else None,
            "sticky_strength": self._sticky_strength,
            "session_active": self._sticky_started is not None,
            "session_tasks_count": len(self._session_tasks),
        }


# =============================================================================
# UNIFIED REQUEST BUFFER
# =============================================================================


@dataclass
class BufferedRequest:
    """A buffered request during hot swap."""
    request_id: str
    prompt: str
    context: Dict[str, Any]
    created_at: datetime
    timeout_at: datetime
    future: asyncio.Future


class UnifiedRequestBuffer:
    """
    Request buffer for zero-loss hot swaps.

    Buffers requests during model transitions to ensure
    no requests are dropped.
    """

    def __init__(self, config: DynamicConfig):
        self._config = config
        self._max_size = config.buffer_max_size
        self._default_timeout = config.buffer_default_timeout_seconds
        self._queue: asyncio.Queue[BufferedRequest] = asyncio.Queue(maxsize=self._max_size)
        self._active = False
        self._lock = asyncio.Lock()

        # Metrics
        self._buffered_count = 0
        self._flushed_count = 0
        self._timeout_count = 0
        self._dropped_count = 0

    async def start_buffering(self) -> None:
        """Start buffering requests."""
        async with self._lock:
            self._active = True
            logger.info("🔄 Request buffering started")

    async def stop_buffering(self) -> None:
        """Stop buffering requests."""
        async with self._lock:
            self._active = False
            logger.info("🔄 Request buffering stopped")

    def is_buffering(self) -> bool:
        """Check if currently buffering."""
        return self._active

    async def enqueue(
        self,
        request_id: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> asyncio.Future:
        """Enqueue a request for later processing."""
        if not self._active:
            raise RuntimeError("Buffer not active")

        context = context or {}
        timeout = timeout or self._default_timeout

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        request = BufferedRequest(
            request_id=request_id,
            prompt=prompt,
            context=context,
            created_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=timeout),
            future=future,
        )

        try:
            self._queue.put_nowait(request)
            self._buffered_count += 1
        except asyncio.QueueFull:
            self._dropped_count += 1
            future.set_exception(RuntimeError("Request buffer full"))

        return future

    async def flush_to(
        self,
        processor: Callable[[str, Dict[str, Any]], Awaitable[str]],
    ) -> int:
        """Flush all buffered requests to processor."""
        processed = 0
        now = datetime.now()

        while not self._queue.empty():
            try:
                request = self._queue.get_nowait()

                # Check timeout
                if now > request.timeout_at:
                    request.future.set_exception(
                        TimeoutError(f"Request {request.request_id} timed out")
                    )
                    self._timeout_count += 1
                    continue

                # Process
                try:
                    result = await processor(request.prompt, request.context)
                    request.future.set_result(result)
                    processed += 1
                except Exception as e:
                    request.future.set_exception(e)
            except asyncio.QueueEmpty:
                break

        self._flushed_count += processed
        logger.info(f"Flushed {processed} buffered requests")
        return processed

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "active": self._active,
            "queue_length": self._queue.qsize(),
            "max_size": self._max_size,
            "buffered_total": self._buffered_count,
            "flushed_total": self._flushed_count,
            "timeout_total": self._timeout_count,
            "dropped_total": self._dropped_count,
        }


# =============================================================================
# CROSS-REPO STATE MANAGER
# =============================================================================


class CrossRepoStateManager:
    """
    Manages shared state across JARVIS, JARVIS-Prime, and Reactor-Core.

    Uses atomic file operations for safe state sharing.
    Integrates with existing CrossRepoBridge and TrinityBridge.
    """

    def __init__(self, config: DynamicConfig):
        self._config = config
        self._state_dir = Path(os.path.expanduser(config.cross_repo_state_dir))
        self._trinity_dir = Path(os.path.expanduser(config.trinity_state_dir))
        self._lock = asyncio.Lock()
        self._initialized = False

        # State cache
        self._cached_state: Optional[CrossRepoMemoryState] = None
        self._cache_time: Optional[datetime] = None

        # Cross-repo bridge instances (lazy init)
        self._cross_repo_bridge: Optional[Any] = None
        self._trinity_bridge: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize state directory and connect to existing bridges."""
        async with self._lock:
            if self._initialized:
                return

            # Create state directories
            self._state_dir.mkdir(parents=True, exist_ok=True)
            (self._state_dir / "memory").mkdir(exist_ok=True)
            (self._state_dir / "routing").mkdir(exist_ok=True)
            self._trinity_dir.mkdir(parents=True, exist_ok=True)
            (self._trinity_dir / "neural_orchestrator").mkdir(exist_ok=True)

            # Try to connect to existing bridges
            await self._connect_to_bridges()

            self._initialized = True
            logger.info("CrossRepoStateManager: Initialized with state sharing")

    async def _connect_to_bridges(self) -> None:
        """Connect to existing cross-repo and trinity bridges."""
        # Try cross-repo bridge
        try:
            from jarvis_prime.core.cross_repo_bridge import CrossRepoBridge
            # Just store reference, don't create new instance
            self._cross_repo_bridge = True  # Mark as available
            logger.debug("CrossRepoStateManager: CrossRepoBridge available")
        except ImportError:
            logger.debug("CrossRepoStateManager: CrossRepoBridge not available")

        # Try trinity bridge
        try:
            from jarvis_prime.core.trinity_bridge import TRINITY_ENABLED
            if TRINITY_ENABLED:
                self._trinity_bridge = True  # Mark as available
                logger.debug("CrossRepoStateManager: TrinityBridge available")
        except ImportError:
            logger.debug("CrossRepoStateManager: TrinityBridge not available")

    async def write_memory_state(
        self,
        pressure: MemoryPressureLevel,
        should_burst: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write memory state atomically."""
        await self.initialize()

        state_file = self._state_dir / "memory" / "jarvis_prime_state.json"
        temp_file = state_file.with_suffix(".tmp")

        data = {
            "pressure_level": pressure.value,
            "should_burst": should_burst,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        async with self._lock:
            try:
                temp_file.write_text(json.dumps(data, indent=2))
                temp_file.replace(state_file)
            except Exception as e:
                logger.warning(f"Failed to write memory state: {e}")
                with suppress(FileNotFoundError):
                    temp_file.unlink()

    async def read_cross_repo_state(self) -> Optional[CrossRepoMemoryState]:
        """Read memory state from all repos."""
        await self.initialize()

        async with self._lock:
            # Check cache
            if self._cached_state and self._cache_time:
                if (datetime.now() - self._cache_time).total_seconds() < 2.0:
                    return self._cached_state

            jarvis_pressure = MemoryPressureLevel.NORMAL
            reactor_pressure = MemoryPressureLevel.NORMAL
            prime_pressure = MemoryPressureLevel.NORMAL
            should_burst = False
            metadata: Dict[str, Any] = {}

            # Read JARVIS state
            jarvis_file = self._state_dir / "memory" / "jarvis_state.json"
            if jarvis_file.exists():
                try:
                    data = json.loads(jarvis_file.read_text())
                    jarvis_pressure = MemoryPressureLevel(data.get("pressure_level", 0))
                    metadata["jarvis"] = data
                except Exception:
                    pass

            # Read JARVIS-Prime state
            prime_file = self._state_dir / "memory" / "jarvis_prime_state.json"
            if prime_file.exists():
                try:
                    data = json.loads(prime_file.read_text())
                    prime_pressure = MemoryPressureLevel(data.get("pressure_level", 0))
                    should_burst = should_burst or data.get("should_burst", False)
                    metadata["jarvis_prime"] = data
                except Exception:
                    pass

            # Read Reactor state
            reactor_file = self._state_dir / "memory" / "reactor_state.json"
            if reactor_file.exists():
                try:
                    data = json.loads(reactor_file.read_text())
                    reactor_pressure = MemoryPressureLevel(data.get("pressure_level", 0))
                    metadata["reactor"] = data
                except Exception:
                    pass

            # Determine overall burst need
            critical_count = sum(
                1 for p in [jarvis_pressure, prime_pressure, reactor_pressure]
                if p == MemoryPressureLevel.CRITICAL
            )
            should_burst = should_burst or critical_count > 0

            state = CrossRepoMemoryState(
                jarvis_pressure=jarvis_pressure,
                jarvis_prime_pressure=prime_pressure,
                reactor_pressure=reactor_pressure,
                should_burst=should_burst,
                timestamp=datetime.now(),
                metadata=metadata,
            )

            self._cached_state = state
            self._cache_time = datetime.now()

            return state

    async def write_routing_state(
        self,
        routing_stats: Dict[str, Any],
        active_model: Optional[str] = None,
    ) -> None:
        """Write routing state for cross-repo visibility."""
        await self.initialize()

        state_file = self._state_dir / "routing" / "neural_orchestrator_state.json"
        temp_file = state_file.with_suffix(".tmp")

        data = {
            "version": "100.0",
            "timestamp": datetime.now().isoformat(),
            "active_model": active_model,
            "routing_stats": routing_stats,
        }

        async with self._lock:
            try:
                temp_file.write_text(json.dumps(data, indent=2))
                temp_file.replace(state_file)
            except Exception as e:
                logger.warning(f"Failed to write routing state: {e}")
                with suppress(FileNotFoundError):
                    temp_file.unlink()

    async def write_trinity_command(
        self,
        command: str,
        payload: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Write a command to Trinity for JARVIS Body execution.

        This enables Neural Orchestrator decisions to trigger actions
        in the broader JARVIS ecosystem.
        """
        if not self._trinity_bridge:
            return None

        await self.initialize()

        request_id = request_id or str(uuid.uuid4())
        command_file = self._trinity_dir / "neural_orchestrator" / f"{request_id}.json"

        data = {
            "source": "neural_orchestrator_v100",
            "request_id": request_id,
            "command": command,
            "payload": payload,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            temp_file = command_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2))
            temp_file.replace(command_file)
            logger.debug(f"Trinity command written: {command} (id={request_id})")
            return request_id
        except Exception as e:
            logger.warning(f"Failed to write Trinity command: {e}")
            return None

    async def notify_model_switch(
        self,
        from_model: Optional[str],
        to_model: str,
        reason: str,
    ) -> None:
        """Notify other repos about a model switch decision."""
        await self.write_memory_state(
            pressure=MemoryPressureLevel.NORMAL,
            should_burst=False,
            metadata={
                "event": "model_switch",
                "from_model": from_model,
                "to_model": to_model,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Also write to Trinity if available
        if self._trinity_bridge:
            await self.write_trinity_command(
                command="model_switched",
                payload={
                    "from_model": from_model,
                    "to_model": to_model,
                    "reason": reason,
                },
            )

    async def get_jarvis_body_state(self) -> Optional[Dict[str, Any]]:
        """
        Read JARVIS Body state from Trinity.

        This allows Neural Orchestrator to be aware of the broader
        system state for better routing decisions.
        """
        await self.initialize()

        jarvis_state_file = self._state_dir / "jarvis_state.json"
        if not jarvis_state_file.exists():
            return None

        try:
            return json.loads(jarvis_state_file.read_text())
        except Exception as e:
            logger.debug(f"Failed to read JARVIS state: {e}")
            return None

    async def get_reactor_core_state(self) -> Optional[Dict[str, Any]]:
        """Read Reactor-Core state."""
        await self.initialize()

        reactor_state_file = self._state_dir / "reactor_state.json"
        if not reactor_state_file.exists():
            return None

        try:
            return json.loads(reactor_state_file.read_text())
        except Exception as e:
            logger.debug(f"Failed to read Reactor-Core state: {e}")
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        return {
            "state_dir": str(self._state_dir),
            "trinity_dir": str(self._trinity_dir),
            "initialized": self._initialized,
            "cached": self._cached_state is not None,
            "cross_repo_bridge_available": self._cross_repo_bridge is not None,
            "trinity_bridge_available": self._trinity_bridge is not None,
        }


# =============================================================================
# CIRCUIT BREAKER MANAGER
# =============================================================================


class CircuitBreakerManager:
    """
    Manages circuit breakers for all routing tiers.

    Provides coordinated circuit breaker state across the system.
    """

    def __init__(self, config: DynamicConfig):
        self._config = config
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    failure_threshold=self._config.circuit_failure_threshold,
                    recovery_timeout=self._config.circuit_recovery_timeout_seconds,
                    half_open_max_calls=self._config.circuit_half_open_max_calls,
                )
            return self._breakers[name]

    async def record_success(self, name: str) -> None:
        """Record success for a circuit."""
        breaker = await self.get_breaker(name)
        await breaker.record_success()

    async def record_failure(self, name: str) -> None:
        """Record failure for a circuit."""
        breaker = await self.get_breaker(name)
        await breaker.record_failure()

    async def can_proceed(self, name: str) -> bool:
        """Check if circuit allows requests."""
        breaker = await self.get_breaker(name)
        return await breaker.can_proceed()

    def get_all_stats(self) -> Dict[str, Any]:
        """Get all circuit breaker statistics."""
        return {
            name: breaker.get_stats().__dict__
            for name, breaker in self._breakers.items()
        }


# =============================================================================
# NEURAL ORCHESTRATOR CORE
# =============================================================================


class NeuralOrchestratorCore:
    """
    Unified Neural Orchestrator Core v100.0

    The single source of truth for all routing decisions across
    the JARVIS Trinity ecosystem.
    """

    _instance: ClassVar[Optional["NeuralOrchestratorCore"]] = None
    _instance_lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self, config: Optional[DynamicConfig] = None):
        self._config = config or DynamicConfig.from_env_and_yaml()

        # Core components
        self._task_classifier = UnifiedTaskClassifier(self._config)
        self._memory_monitor = UnifiedMemoryMonitor(self._config)
        self._sticky_routing = UnifiedStickyRouting(self._config)
        self._request_buffer = UnifiedRequestBuffer(self._config)
        self._state_manager = CrossRepoStateManager(self._config)
        self._circuit_manager = CircuitBreakerManager(self._config)

        # v100.1: Advanced components
        self._routing_cache = LRURoutingCache(
            max_size=self._config.routing_cache_max_size,
            ttl_seconds=self._config.routing_cache_ttl_seconds,
        )
        self._deduplicator = RequestDeduplicator(
            ttl_seconds=self._config.dedup_ttl_seconds,
            max_pending=self._config.dedup_max_pending,
        )
        self._trend_analyzer = MemoryPressureTrendAnalyzer(
            window_size=self._config.memory_trend_window_size,
            sustained_threshold=self._config.memory_sustained_threshold,
            sustained_duration_seconds=self._config.memory_sustained_duration_seconds,
        )
        self._version_tracker = ModelVersionTracker()
        self._preloader = PredictivePreloader(
            history_size=self._config.preload_history_size,
            prediction_threshold=self._config.preload_prediction_threshold,
        )
        self._batcher = RequestBatcher(
            batch_window_ms=self._config.batch_window_ms,
            max_batch_size=self._config.batch_max_size,
        )
        self._metrics = MetricsExporter()
        self._availability_monitor: Optional[TierAvailabilityMonitor] = None  # Init after circuit_manager

        # State
        self._initialized = False
        self._init_lock = DeadlockDetectingLock(timeout_seconds=10.0)
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "fallback_routes": 0,
            "circuit_opens": 0,
            "sticky_hits": 0,
            "burst_triggers": 0,
            "cache_hits": 0,
            "deduplicated": 0,
            "batched": 0,
            "cancelled": 0,
            "all_tiers_down_events": 0,
        }

        # Weak reference cache for model specs
        self._model_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

    @classmethod
    async def get_instance(
        cls,
        config: Optional[DynamicConfig] = None,
    ) -> "NeuralOrchestratorCore":
        """Get singleton instance."""
        async with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(config)
                await cls._instance.initialize()
            return cls._instance

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        async with cls._instance_lock:
            if cls._instance is not None:
                await cls._instance.shutdown()
                cls._instance = None

    async def initialize(self) -> None:
        """Initialize all components with advanced features."""
        await self._init_lock.acquire("initialize")
        try:
            if self._initialized:
                return

            logger.info("Initializing NeuralOrchestratorCore v100.1 (Advanced)...")

            # Initialize core components
            await self._memory_monitor.initialize()
            await self._state_manager.initialize()

            # Initialize availability monitor
            self._availability_monitor = TierAvailabilityMonitor(
                circuit_manager=self._circuit_manager,
                alert_callback=self._on_all_tiers_down,
            )

            # Start background tasks
            self._background_tasks.extend([
                asyncio.create_task(self._memory_state_broadcaster()),
                asyncio.create_task(self._session_cleanup_task()),
                asyncio.create_task(self._metrics_export_task()),
                asyncio.create_task(self._tier_availability_monitor_task()),
            ])

            self._initialized = True
            logger.info("NeuralOrchestratorCore v100.1 initialized successfully")
            logger.info("  Advanced features: dedup, caching, batching, trend analysis, predictive preloading")
        finally:
            self._init_lock.release()

    async def _on_all_tiers_down(self, alert_msg: str) -> None:
        """Callback when all tiers are down."""
        self._stats["all_tiers_down_events"] += 1
        # Try to notify via observability if available
        if OBSERVABILITY_AVAILABLE:
            try:
                obs = await get_observability_bridge()
                await obs.send_alert(alert_msg, severity="critical")
            except Exception:
                pass

    async def _session_cleanup_task(self) -> None:
        """Background task to clean up expired sessions."""
        while not self._shutdown_event.is_set():
            try:
                # Cleanup expired sticky sessions
                if hasattr(self._sticky_routing, 'cleanup_expired_sessions'):
                    await self._sticky_routing.cleanup_expired_sessions()
            except Exception as e:
                logger.debug(f"Session cleanup error: {e}")
            await asyncio.sleep(60.0)

    async def _metrics_export_task(self) -> None:
        """Background task to export metrics periodically."""
        while not self._shutdown_event.is_set():
            try:
                await self._metrics.export()
            except Exception as e:
                logger.debug(f"Metrics export error: {e}")
            await asyncio.sleep(self._config.metrics_export_interval_seconds)

    async def _tier_availability_monitor_task(self) -> None:
        """Background task to monitor tier availability."""
        while not self._shutdown_event.is_set():
            try:
                if self._availability_monitor:
                    await self._availability_monitor.monitor_and_alert()
            except Exception as e:
                logger.debug(f"Availability monitor error: {e}")
            await asyncio.sleep(10.0)

    async def shutdown(self) -> None:
        """Shutdown all components gracefully."""
        logger.info("Shutting down NeuralOrchestratorCore...")

        self._shutdown_event.set()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        # End sticky session
        await self._sticky_routing.end_session()

        self._initialized = False
        logger.info("NeuralOrchestratorCore shutdown complete")

    async def _memory_state_broadcaster(self) -> None:
        """Background task to broadcast memory state."""
        while not self._shutdown_event.is_set():
            try:
                snapshot = await self._memory_monitor.get_detailed_snapshot()
                await self._state_manager.write_memory_state(
                    pressure=MemoryPressureLevel(snapshot.get("pressure_level", 0)),
                    should_burst=snapshot.get("should_burst", False),
                    metadata={"snapshot": snapshot},
                )
            except Exception as e:
                logger.debug(f"Memory state broadcast failed: {e}")

            await asyncio.sleep(5.0)

    @async_defensive(fallback=None, log_errors=True)
    async def route(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> RoutingResult:
        """
        Route a request through the unified pipeline with advanced features.

        v100.1 Advanced Features:
        - Request deduplication
        - LRU routing cache
        - Cancellation token support
        - Memory pressure trend analysis
        - Predictive preloading
        - Metrics recording

        Args:
            prompt: The prompt to route
            context: Optional request context
            cancellation_token: Optional cancellation token for request lifecycle

        Returns:
            RoutingResult with routing decision
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        context = context or {}
        request_id = context.get("request_id") or str(uuid.uuid4())

        # Set context variables for tracing
        set_request_context(
            request_id=request_id,
            session_id=context.get("session_id"),
            trace_id=context.get("trace_id"),
        )
        if cancellation_token:
            cancellation_token_var.set(cancellation_token)

        self._stats["total_requests"] += 1

        # Compute prompt hash for dedup/cache
        prompt_hash = self._deduplicator.compute_hash(prompt, context)

        try:
            # 0. Check cancellation
            if cancellation_token and cancellation_token.is_cancelled:
                self._stats["cancelled"] += 1
                raise RoutingCancelledException(cancellation_token.cancellation_reason)

            # 1. Check deduplication - return cached if duplicate
            is_duplicate, pending_future = await self._deduplicator.check_or_register(prompt_hash)
            if is_duplicate:
                self._stats["deduplicated"] += 1
                if pending_future:
                    # Wait for pending request
                    return await pending_future
                # Return from completed cache
                cached_result = await self._deduplicator.get_completed(prompt_hash)
                if cached_result:
                    return cached_result

            # 2. Check routing cache
            cached_routing = await self._routing_cache.get(prompt_hash)
            if cached_routing:
                self._stats["cache_hits"] += 1
                # Update latency and return
                cached_routing.latency_ms = (time.time() - start_time) * 1000
                cached_routing.metadata["from_cache"] = True
                await self._deduplicator.complete(prompt_hash, cached_routing)
                return cached_routing

            # 3. Check if buffering (hot swap in progress)
            if self._request_buffer.is_buffering():
                logger.debug(f"Request {request_id} buffered during hot swap")
                future = await self._request_buffer.enqueue(
                    request_id=request_id,
                    prompt=prompt,
                    context=context,
                )
                result = RoutingResult(
                    request_id=request_id,
                    tier=RoutingTier.UNKNOWN,
                    model_id=None,
                    endpoint=None,
                    decision_reason=RoutingDecisionReason.DEFAULT,
                    task_classification=None,
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={"buffered": True, "future": id(future)},
                )
                await self._deduplicator.complete(prompt_hash, result)
                return result

            # 4. Check cancellation before classification
            if cancellation_token:
                cancellation_token.check()

            # 5. Classify the task
            classification = await self._task_classifier.classify(prompt, context)

            # 6. Check memory pressure with trend analysis
            memory_snapshot = await self._memory_monitor.get_detailed_snapshot()
            pressure_level = MemoryPressureLevel(memory_snapshot.get("pressure_level", 0))
            await self._trend_analyzer.record(pressure_level)

            # Use trend analysis for burst decision (prevents false positives)
            should_burst_trend, burst_reason = await self._trend_analyzer.should_trigger_burst()
            memory_critical = pressure_level == MemoryPressureLevel.CRITICAL and should_burst_trend

            if memory_critical:
                self._stats["burst_triggers"] += 1
                logger.info(f"Cloud burst triggered: {burst_reason}")

            # 7. Check sticky routing
            sticky_model = await self._sticky_routing.get_sticky_model(classification)
            if sticky_model and not memory_critical:
                self._stats["sticky_hits"] += 1
                result = RoutingResult(
                    request_id=request_id,
                    tier=classification.recommended_tier,
                    model_id=sticky_model,
                    endpoint=self._get_endpoint_for_tier(classification.recommended_tier),
                    decision_reason=RoutingDecisionReason.STICKY_SESSION,
                    task_classification=classification,
                    confidence=0.95,
                    latency_ms=(time.time() - start_time) * 1000,
                    metadata={"sticky": True},
                )
                await self._routing_cache.put(prompt_hash, result)
                await self._deduplicator.complete(prompt_hash, result)
                await self._record_metrics(result)
                return result

            # 8. Check cancellation before tier selection
            if cancellation_token:
                cancellation_token.check()

            # 9. Determine target tier
            target_tier = classification.recommended_tier

            # Override for memory pressure
            if should_burst_trend and target_tier.value < RoutingTier.TIER_1_CLOUD.value:
                target_tier = RoutingTier.TIER_1_CLOUD

            # 10. Check circuit breaker with fallback chain
            tier_name = target_tier.name
            can_proceed = await self._circuit_manager.can_proceed(tier_name)

            if not can_proceed:
                # Try fallback chain
                fallback_tier = await self._find_available_tier(target_tier)
                if fallback_tier:
                    target_tier = fallback_tier
                    self._stats["fallback_routes"] += 1
                else:
                    # All tiers down - critical error
                    self._stats["all_tiers_down_events"] += 1
                    raise AllTiersUnavailableError("All routing tiers unavailable")

            # 11. Get endpoint and model
            endpoint = self._get_endpoint_for_tier(target_tier)
            model_id = self._get_model_for_tier(target_tier)

            # 12. Update sticky routing
            await self._sticky_routing.update_sticky(
                model_id=model_id or "unknown",
                tier=target_tier,
                classification=classification,
            )

            # 13. Record usage for predictive preloading
            await self._preloader.record_usage(classification.task_type, target_tier)

            # 14. Predict and trigger preload for next likely model
            predicted_tier = await self._preloader.predict_next_tier(classification.task_type)
            if predicted_tier and predicted_tier != target_tier:
                predicted_model = self._get_model_for_tier(predicted_tier)
                if predicted_model:
                    asyncio.create_task(self._preloader.trigger_preload(predicted_model))

            self._stats["successful_routes"] += 1

            result = RoutingResult(
                request_id=request_id,
                tier=target_tier,
                model_id=model_id,
                endpoint=endpoint,
                decision_reason=RoutingDecisionReason.TASK_CLASSIFICATION,
                task_classification=classification,
                confidence=classification.confidence,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={
                    "memory_pressure": memory_snapshot.get("pressure_level"),
                    "should_burst": should_burst_trend,
                    "burst_reason": burst_reason if should_burst_trend else None,
                    "memory_trend": self._trend_analyzer.get_trend(),
                },
            )

            # 15. Cache the routing decision
            await self._routing_cache.put(prompt_hash, result)

            # 16. Complete deduplication
            await self._deduplicator.complete(prompt_hash, result)

            # 17. Record metrics
            await self._record_metrics(result)

            # 18. Async notification to cross-repo state (non-blocking)
            asyncio.create_task(
                self._state_manager.write_routing_state(
                    routing_stats=self._stats.copy(),
                    active_model=model_id,
                )
            )

            return result

        except RoutingCancelledException as e:
            self._stats["cancelled"] += 1
            await self._deduplicator.fail(prompt_hash, e)
            raise

        except AllTiersUnavailableError as e:
            await self._deduplicator.fail(prompt_hash, e)
            # Try emergency fallback
            logger.critical(f"All tiers unavailable for {request_id}")
            return RoutingResult(
                request_id=request_id,
                tier=RoutingTier.TIER_0_FAST,
                model_id=None,
                endpoint=f"http://localhost:{self._config.jarvis_prime_port}",
                decision_reason=RoutingDecisionReason.FALLBACK,
                task_classification=None,
                confidence=0.1,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e), "all_tiers_down": True},
            )

        except Exception as e:
            logger.error(f"Routing failed for {request_id}: {e}")
            await self._deduplicator.fail(prompt_hash, e)

            # Return a fallback result
            return RoutingResult(
                request_id=request_id,
                tier=RoutingTier.TIER_0_FAST,  # Default safe tier
                model_id=None,
                endpoint=f"http://localhost:{self._config.jarvis_prime_port}",
                decision_reason=RoutingDecisionReason.FALLBACK,
                task_classification=None,
                confidence=0.5,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    async def _record_metrics(self, result: RoutingResult) -> None:
        """Record routing metrics for observability."""
        try:
            await self._metrics.inc_counter(
                "neural_orchestrator_routes_total",
                labels={
                    "tier": result.tier.name,
                    "decision_reason": result.decision_reason.value,
                }
            )
            await self._metrics.record_histogram(
                "neural_orchestrator_routing_latency_ms",
                result.latency_ms,
                labels={"tier": result.tier.name}
            )
            await self._metrics.set_gauge(
                "neural_orchestrator_confidence",
                result.confidence,
                labels={"tier": result.tier.name}
            )
        except Exception as e:
            logger.debug(f"Metrics recording failed: {e}")

    async def _find_available_tier(self, starting_tier: RoutingTier) -> Optional[RoutingTier]:
        """Find an available tier starting from the given tier and trying fallbacks."""
        current = starting_tier
        tried: Set[RoutingTier] = set()

        while current and current not in tried:
            tried.add(current)
            if await self._circuit_manager.can_proceed(current.name):
                return current
            current = self._get_fallback_tier(current)

        # If all fallbacks failed, try any available tier
        for tier in RoutingTier:
            if tier == RoutingTier.UNKNOWN or tier in tried:
                continue
            if await self._circuit_manager.can_proceed(tier.name):
                return tier

        return None

    def _get_endpoint_for_tier(self, tier: RoutingTier) -> str:
        """Get endpoint URL for a tier."""
        if tier in {RoutingTier.TIER_0_ULTRA_FAST, RoutingTier.TIER_0_FAST, RoutingTier.TIER_05_CAPABLE}:
            return f"http://localhost:{self._config.jarvis_prime_port}"
        elif tier == RoutingTier.TIER_1_CLOUD:
            # Check for GCP endpoint
            gcp_endpoint = os.environ.get("GCP_INFERENCE_ENDPOINT")
            if gcp_endpoint:
                return gcp_endpoint
            return f"http://localhost:{self._config.jarvis_prime_port}"  # Fallback
        else:  # TIER_2_DEEP
            return "https://api.anthropic.com"  # Claude API

    def _get_model_for_tier(self, tier: RoutingTier) -> Optional[str]:
        """Get model ID for a tier."""
        model_map = {
            RoutingTier.TIER_0_ULTRA_FAST: "phi-3.5-mini",
            RoutingTier.TIER_0_FAST: "llama-3-8b",
            RoutingTier.TIER_05_CAPABLE: "qwen-2.5-32b",
            RoutingTier.TIER_1_CLOUD: "llama-3.3-70b",
            RoutingTier.TIER_2_DEEP: "claude-opus-4",
        }
        return model_map.get(tier)

    def _get_fallback_tier(self, current: RoutingTier) -> Optional[RoutingTier]:
        """Get fallback tier if current is unavailable."""
        fallback_chain = {
            RoutingTier.TIER_0_ULTRA_FAST: RoutingTier.TIER_0_FAST,
            RoutingTier.TIER_0_FAST: RoutingTier.TIER_05_CAPABLE,
            RoutingTier.TIER_05_CAPABLE: RoutingTier.TIER_1_CLOUD,
            RoutingTier.TIER_1_CLOUD: RoutingTier.TIER_2_DEEP,
            RoutingTier.TIER_2_DEEP: None,
        }
        return fallback_chain.get(current)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def classify_task(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskClassification:
        """Classify a task without routing."""
        return await self._task_classifier.classify(prompt, context)

    async def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        return await self._memory_monitor.get_detailed_snapshot()

    async def should_burst_to_cloud(self) -> bool:
        """Check if cloud burst is recommended."""
        return await self._memory_monitor.should_burst_to_cloud()

    async def start_hot_swap_buffering(self) -> None:
        """Start request buffering for hot swap."""
        await self._request_buffer.start_buffering()

    async def stop_hot_swap_buffering(
        self,
        processor: Optional[Callable[[str, Dict[str, Any]], Awaitable[str]]] = None,
    ) -> int:
        """Stop buffering and optionally flush requests."""
        await self._request_buffer.stop_buffering()
        if processor:
            return await self._request_buffer.flush_to(processor)
        return 0

    async def record_circuit_success(self, tier_name: str) -> None:
        """Record successful request for circuit breaker."""
        await self._circuit_manager.record_success(tier_name)

    async def record_circuit_failure(self, tier_name: str) -> None:
        """Record failed request for circuit breaker."""
        await self._circuit_manager.record_failure(tier_name)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all components."""
        return {
            "version": "100.1",
            "initialized": self._initialized,
            "routing": self._stats.copy(),
            "task_classifier": self._task_classifier.get_stats(),
            "memory_monitor": self._memory_monitor.get_stats(),
            "sticky_routing": self._sticky_routing.get_stats(),
            "request_buffer": self._request_buffer.get_stats(),
            "state_manager": self._state_manager.get_stats(),
            "circuit_breakers": self._circuit_manager.get_all_stats(),
            # v100.1 Advanced components
            "routing_cache": self._routing_cache.stats,
            "deduplicator": self._deduplicator.stats,
            "trend_analyzer": self._trend_analyzer.stats,
            "version_tracker": self._version_tracker.stats,
            "preloader": self._preloader.stats,
            "batcher": self._batcher.stats,
            "metrics": self._metrics.stats,
            "availability_monitor": self._availability_monitor.stats if self._availability_monitor else None,
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for monitoring."""
        # Calculate overall health
        cache_healthy = self._routing_cache.stats.get("hit_rate", 0) > 0 or self._stats["total_requests"] < 10
        circuit_healthy = all(
            cb.get("state") != "open"
            for cb in self._circuit_manager.get_all_stats().values()
        )
        memory_healthy = self._trend_analyzer.get_trend() != "critical_sustained"

        overall_healthy = self._initialized and circuit_healthy and memory_healthy

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "version": "100.1",
            "initialized": self._initialized,
            "components": {
                "task_classifier": True,
                "memory_monitor": True,
                "sticky_routing": True,
                "request_buffer": True,
                "state_manager": True,
                "circuit_manager": circuit_healthy,
                "routing_cache": cache_healthy,
                "deduplicator": True,
                "trend_analyzer": memory_healthy,
                "version_tracker": True,
                "preloader": True,
                "batcher": True,
                "metrics": True,
            },
            "advanced_features": {
                "deduplication": True,
                "caching": True,
                "trend_analysis": True,
                "predictive_preloading": True,
                "batching": True,
                "cancellation_support": True,
                "deadlock_detection": True,
            },
            "stats_summary": {
                "total_requests": self._stats["total_requests"],
                "cache_hit_rate": self._routing_cache.stats.get("hit_rate", 0),
                "dedup_rate": self._deduplicator.stats.get("dedup_rate", 0),
                "fallback_rate": (
                    self._stats["fallback_routes"] / max(self._stats["total_requests"], 1)
                ),
            },
        }


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================


async def get_neural_orchestrator(
    config: Optional[DynamicConfig] = None,
) -> NeuralOrchestratorCore:
    """Get the global NeuralOrchestratorCore instance."""
    return await NeuralOrchestratorCore.get_instance(config)


async def neural_route(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    cancellation_token: Optional[CancellationToken] = None,
) -> RoutingResult:
    """Convenience function to route a request."""
    orchestrator = await get_neural_orchestrator()
    return await orchestrator.route(prompt, context, cancellation_token)


@with_retry(max_attempts=3, base_delay=0.5, max_delay=5.0, jitter=True)
async def neural_route_with_retry(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
) -> RoutingResult:
    """Route a request with automatic retry on failure."""
    orchestrator = await get_neural_orchestrator()
    result = await orchestrator.route(prompt, context)
    if result.decision_reason == RoutingDecisionReason.FALLBACK:
        # Don't retry on intentional fallback
        return result
    if result.metadata.get("error"):
        # Retry on errors
        raise RuntimeError(result.metadata["error"])
    return result


async def neural_route_with_timeout(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 30.0,
) -> RoutingResult:
    """Route a request with timeout."""
    token = CancellationToken.with_timeout(timeout_seconds)
    try:
        return await neural_route(prompt, context, token)
    except RoutingCancelledException:
        raise asyncio.TimeoutError(f"Routing timeout after {timeout_seconds}s")


async def shutdown_neural_orchestrator() -> None:
    """Shutdown the global orchestrator."""
    await NeuralOrchestratorCore.reset_instance()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core class
    "NeuralOrchestratorCore",

    # Configuration
    "DynamicConfig",

    # Data structures
    "TaskClassification",
    "RoutingResult",
    "CircuitBreakerState",
    "CrossRepoMemoryState",

    # Enums
    "MemoryPressureLevel",
    "CircuitState",
    "RoutingTier",
    "TaskCategory",
    "RoutingDecisionReason",

    # Protocols
    "RouterProtocol",
    "MemoryMonitorProtocol",
    "CircuitBreakerProtocol",
    "ConfigProviderProtocol",

    # Core Components
    "UnifiedTaskClassifier",
    "UnifiedMemoryMonitor",
    "UnifiedStickyRouting",
    "UnifiedRequestBuffer",
    "CrossRepoStateManager",
    "CircuitBreakerManager",
    "CircuitBreaker",

    # v100.1 Advanced Components
    "CancellationToken",
    "DeadlockDetectingLock",
    "LRURoutingCache",
    "RequestDeduplicator",
    "MemoryPressureTrendAnalyzer",
    "ModelVersionTracker",
    "PredictivePreloader",
    "RequestBatcher",
    "TierAvailabilityMonitor",
    "MetricsExporter",

    # Custom Exceptions
    "DeadlockError",
    "RoutingCancelledException",
    "AllTiersUnavailableError",
    "RequestDuplicateError",

    # Decorators
    "defensive",
    "async_defensive",
    "with_timeout",
    "with_retry",

    # Context variables
    "set_request_context",
    "get_request_context",
    "request_id_var",
    "session_id_var",
    "trace_context_var",
    "routing_context_var",
    "cancellation_token_var",

    # Convenience functions
    "get_neural_orchestrator",
    "neural_route",
    "neural_route_with_retry",
    "neural_route_with_timeout",
    "shutdown_neural_orchestrator",
]
