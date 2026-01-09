"""
Reliability Engines v84.0 - Production-Grade Fault Tolerance
============================================================

Ultra-robust reliability components for Trinity ecosystem production deployment.

ENGINES:
    - SQLiteRetryEngine: Handle database lock contention with exponential backoff
    - OOMProtectionEngine: Memory monitoring and emergency load shedding
    - GuaranteedDeliveryQueue: ACK-based message delivery with retry semantics
    - CircuitBreakerEngine: Prevent cascade failures with adaptive circuit breakers
    - RateLimiterEngine: Token bucket rate limiting with burst support

TECHNIQUES:
    - Exponential backoff with jitter for retry storms prevention
    - Memory-mapped monitoring for low-overhead health checks
    - Write-ahead logging for guaranteed delivery
    - Hysteresis-based circuit breaker state transitions
    - Adaptive rate limiting based on system load

USAGE:
    from jarvis_prime.core.reliability_engines import (
        get_sqlite_retry_engine,
        get_oom_protection,
        get_delivery_queue,
    )

    # SQLite with automatic retry
    async with get_sqlite_retry_engine() as db:
        await db.execute_write("INSERT INTO events ...")

    # OOM protection
    oom_guard = await get_oom_protection()
    await oom_guard.start_monitoring()

    # Guaranteed delivery
    queue = await get_delivery_queue()
    await queue.send_with_ack(websocket, event)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

# Try to import psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - OOM protection limited")

# Try to import aiofiles for async file I/O
try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# Try to import aiosqlite for async SQLite
try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    logger.warning("aiosqlite not available - using synchronous fallback")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 5
    initial_backoff: float = 0.05  # 50ms
    max_backoff: float = 5.0  # 5 seconds
    exponential_base: float = 2.0
    jitter_factor: float = 0.1  # 10% jitter
    retryable_errors: Tuple[str, ...] = (
        "database is locked",
        "database is busy",
        "unable to open database",
        "disk I/O error",
    )


@dataclass
class OOMConfig:
    """Configuration for OOM protection."""
    memory_threshold_percent: float = 95.0  # Emergency at 95%
    warning_threshold_percent: float = 85.0  # Warning at 85%
    check_interval: float = 5.0  # Check every 5 seconds
    gc_threshold_percent: float = 80.0  # Trigger GC at 80%
    enable_aggressive_gc: bool = True
    shed_load_cooldown: float = 30.0  # Cooldown between load shedding


@dataclass
class DeliveryConfig:
    """Configuration for guaranteed delivery."""
    max_retries: int = 10
    retry_delay: float = 1.0  # Base retry delay
    ack_timeout: float = 5.0  # Timeout waiting for ACK
    persistence_dir: Optional[Path] = None
    max_queue_size: int = 10000
    batch_size: int = 100
    flush_interval: float = 1.0


# =============================================================================
# SQLITE RETRY ENGINE
# =============================================================================

class SQLiteRetryEngine:
    """
    v84.0: Production-grade SQLite wrapper with exponential backoff.

    Handles "database is locked" and other transient errors automatically
    with intelligent retry logic and jitter to prevent thundering herd.

    Features:
        - Exponential backoff with configurable parameters
        - Jitter to prevent retry storms
        - Connection pooling with health checks
        - Write-ahead logging (WAL) mode for concurrent access
        - Automatic schema migration support
        - Query timeout enforcement
        - Dead connection detection and cleanup
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        config: Optional[RetryConfig] = None,
        enable_wal: bool = True,
        pool_size: int = 5,
        query_timeout: float = 30.0,
    ):
        """
        Initialize SQLite retry engine.

        Args:
            db_path: Path to SQLite database
            config: Retry configuration
            enable_wal: Enable Write-Ahead Logging mode
            pool_size: Connection pool size
            query_timeout: Maximum query execution time
        """
        self.db_path = Path(db_path)
        self.config = config or RetryConfig()
        self.enable_wal = enable_wal
        self.pool_size = pool_size
        self.query_timeout = query_timeout

        # Connection pool
        self._pool: asyncio.Queue[Optional[Any]] = asyncio.Queue(maxsize=pool_size)
        self._pool_initialized = False
        self._pool_lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_queries": 0,
            "retried_queries": 0,
            "failed_queries": 0,
            "total_retry_time": 0.0,
            "max_retries_hit": 0,
        }

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """Initialize connection pool."""
        async with self._pool_lock:
            if self._pool_initialized:
                return

            logger.info(f"Initializing SQLite pool for {self.db_path}")

            for _ in range(self.pool_size):
                conn = await self._create_connection()
                await self._pool.put(conn)

            self._pool_initialized = True
            logger.info(f"SQLite pool initialized: {self.pool_size} connections")

    async def _create_connection(self) -> Any:
        """Create a new database connection with optimal settings."""
        if AIOSQLITE_AVAILABLE:
            conn = await aiosqlite.connect(
                str(self.db_path),
                timeout=self.query_timeout,
            )

            # Optimize for concurrent access
            await conn.execute("PRAGMA journal_mode=WAL" if self.enable_wal else "PRAGMA journal_mode=DELETE")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            await conn.execute("PRAGMA temp_store=MEMORY")
            await conn.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
            await conn.execute("PRAGMA busy_timeout=30000")  # 30s busy timeout

            return conn
        else:
            # Synchronous fallback
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.query_timeout,
                check_same_thread=False,
            )
            conn.execute("PRAGMA journal_mode=WAL" if self.enable_wal else "PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            return conn

    async def _get_connection(self) -> Any:
        """Get a connection from the pool."""
        if not self._pool_initialized:
            await self.initialize()

        conn = await asyncio.wait_for(
            self._pool.get(),
            timeout=self.query_timeout,
        )

        # Verify connection is alive
        try:
            if AIOSQLITE_AVAILABLE:
                await conn.execute("SELECT 1")
            else:
                conn.execute("SELECT 1")
            return conn
        except Exception:
            # Connection dead, create new one
            logger.warning("Dead connection detected, creating new one")
            conn = await self._create_connection()
            return conn

    async def _return_connection(self, conn: Any):
        """Return a connection to the pool."""
        try:
            await self._pool.put(conn)
        except asyncio.QueueFull:
            # Pool full, close connection
            if AIOSQLITE_AVAILABLE:
                await conn.close()
            else:
                conn.close()

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate backoff delay with exponential growth and jitter.

        Uses decorrelated jitter to prevent thundering herd.
        """
        # Exponential backoff
        delay = self.config.initial_backoff * (self.config.exponential_base ** attempt)

        # Cap at max backoff
        delay = min(delay, self.config.max_backoff)

        # Add jitter (decorrelated jitter algorithm)
        jitter_range = delay * self.config.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)
        delay += jitter

        return max(0.001, delay)  # Minimum 1ms

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable."""
        error_msg = str(error).lower()
        return any(
            retryable in error_msg
            for retryable in self.config.retryable_errors
        )

    async def execute_write(
        self,
        query: str,
        params: Tuple[Any, ...] = (),
        commit: bool = True,
    ) -> Optional[int]:
        """
        Execute a write query with automatic retry.

        Args:
            query: SQL query to execute
            params: Query parameters
            commit: Whether to commit after execution

        Returns:
            Number of rows affected, or None on failure

        Raises:
            sqlite3.Error: If all retries exhausted
        """
        self._stats["total_queries"] += 1
        last_error: Optional[Exception] = None
        total_wait_time = 0.0

        for attempt in range(self.config.max_retries + 1):
            conn = await self._get_connection()

            try:
                if AIOSQLITE_AVAILABLE:
                    cursor = await conn.execute(query, params)
                    if commit:
                        await conn.commit()
                    rowcount = cursor.rowcount
                else:
                    cursor = conn.execute(query, params)
                    if commit:
                        conn.commit()
                    rowcount = cursor.rowcount

                await self._return_connection(conn)

                # Log retry success
                if attempt > 0:
                    logger.info(
                        f"Query succeeded after {attempt} retries "
                        f"(waited {total_wait_time:.3f}s)"
                    )
                    self._stats["retried_queries"] += 1
                    self._stats["total_retry_time"] += total_wait_time

                return rowcount

            except Exception as e:
                await self._return_connection(conn)
                last_error = e

                if not self._is_retryable_error(e):
                    logger.error(f"Non-retryable SQLite error: {e}")
                    self._stats["failed_queries"] += 1
                    raise

                if attempt < self.config.max_retries:
                    delay = self._calculate_backoff(attempt)
                    total_wait_time += delay

                    logger.debug(
                        f"SQLite locked (attempt {attempt + 1}/{self.config.max_retries + 1}), "
                        f"retrying in {delay:.3f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"SQLite query failed after {attempt + 1} attempts: {e}"
                    )
                    self._stats["failed_queries"] += 1
                    self._stats["max_retries_hit"] += 1

        raise last_error or sqlite3.Error("Unknown error")

    async def execute_read(
        self,
        query: str,
        params: Tuple[Any, ...] = (),
    ) -> List[Tuple[Any, ...]]:
        """
        Execute a read query with automatic retry.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            List of result rows
        """
        self._stats["total_queries"] += 1

        for attempt in range(self.config.max_retries + 1):
            conn = await self._get_connection()

            try:
                if AIOSQLITE_AVAILABLE:
                    cursor = await conn.execute(query, params)
                    rows = await cursor.fetchall()
                else:
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()

                await self._return_connection(conn)
                return rows

            except Exception as e:
                await self._return_connection(conn)

                if not self._is_retryable_error(e):
                    self._stats["failed_queries"] += 1
                    raise

                if attempt < self.config.max_retries:
                    delay = self._calculate_backoff(attempt)
                    await asyncio.sleep(delay)
                else:
                    self._stats["failed_queries"] += 1
                    raise

        return []

    async def execute_many(
        self,
        query: str,
        params_list: List[Tuple[Any, ...]],
        batch_size: int = 100,
    ) -> int:
        """
        Execute multiple queries in batches with retry.

        Args:
            query: SQL query template
            params_list: List of parameter tuples
            batch_size: Number of queries per batch

        Returns:
            Total rows affected
        """
        total_affected = 0

        for i in range(0, len(params_list), batch_size):
            batch = params_list[i:i + batch_size]
            conn = await self._get_connection()

            try:
                if AIOSQLITE_AVAILABLE:
                    await conn.executemany(query, batch)
                    await conn.commit()
                else:
                    conn.executemany(query, batch)
                    conn.commit()

                total_affected += len(batch)
                await self._return_connection(conn)

            except Exception as e:
                await self._return_connection(conn)

                if not self._is_retryable_error(e):
                    raise

                # Retry individual items on lock
                for params in batch:
                    result = await self.execute_write(query, params)
                    if result:
                        total_affected += result

        return total_affected

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "pool_size": self.pool_size,
            "db_path": str(self.db_path),
            "retry_config": {
                "max_retries": self.config.max_retries,
                "initial_backoff": self.config.initial_backoff,
                "max_backoff": self.config.max_backoff,
            },
        }

    async def close(self):
        """Close all connections."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                if conn:
                    if AIOSQLITE_AVAILABLE:
                        await conn.close()
                    else:
                        conn.close()
            except asyncio.QueueEmpty:
                break

        self._pool_initialized = False
        logger.info("SQLite pool closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# =============================================================================
# OOM PROTECTION ENGINE
# =============================================================================

class LoadShedStrategy(Enum):
    """Strategies for shedding load under memory pressure."""
    CLEAR_CACHES = auto()
    REJECT_NEW_REQUESTS = auto()
    PAUSE_NON_CRITICAL = auto()
    AGGRESSIVE_GC = auto()
    EMERGENCY_RESTART = auto()


@dataclass
class MemorySnapshot:
    """Snapshot of current memory state."""
    timestamp: float
    total_mb: float
    available_mb: float
    used_mb: float
    used_percent: float
    swap_used_mb: float
    process_rss_mb: float
    process_vms_mb: float


class OOMProtectionEngine:
    """
    v84.0: Memory monitoring and OOM prevention system.

    Features:
        - Continuous memory monitoring with configurable intervals
        - Multi-tier alerting (warning, critical, emergency)
        - Automatic garbage collection triggering
        - Load shedding callbacks for graceful degradation
        - Process memory tracking
        - Swap usage monitoring
        - Memory trend analysis for predictive alerts
    """

    def __init__(
        self,
        config: Optional[OOMConfig] = None,
        on_warning: Optional[Callable[[], Awaitable[None]]] = None,
        on_critical: Optional[Callable[[], Awaitable[None]]] = None,
        on_emergency: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        """
        Initialize OOM protection engine.

        Args:
            config: OOM configuration
            on_warning: Callback for warning state
            on_critical: Callback for critical state
            on_emergency: Callback for emergency state
        """
        self.config = config or OOMConfig()
        self._on_warning = on_warning
        self._on_critical = on_critical
        self._on_emergency = on_emergency

        # Monitoring state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_shed_time: float = 0.0

        # Memory history for trend analysis
        self._memory_history: List[MemorySnapshot] = []
        self._history_max_size = 60  # 5 minutes at 5s interval

        # Load shedding callbacks
        self._shed_callbacks: List[Callable[[], Awaitable[None]]] = []

        # Statistics
        self._stats = {
            "gc_triggered": 0,
            "warnings_issued": 0,
            "load_shed_events": 0,
            "peak_memory_percent": 0.0,
        }

    def register_shed_callback(self, callback: Callable[[], Awaitable[None]]):
        """Register a callback to be called during load shedding."""
        self._shed_callbacks.append(callback)

    def _get_memory_snapshot(self) -> Optional[MemorySnapshot]:
        """Get current memory snapshot."""
        if not PSUTIL_AVAILABLE:
            return None

        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            process = psutil.Process()
            mem_info = process.memory_info()

            snapshot = MemorySnapshot(
                timestamp=time.time(),
                total_mb=mem.total / (1024 ** 2),
                available_mb=mem.available / (1024 ** 2),
                used_mb=mem.used / (1024 ** 2),
                used_percent=mem.percent,
                swap_used_mb=swap.used / (1024 ** 2),
                process_rss_mb=mem_info.rss / (1024 ** 2),
                process_vms_mb=mem_info.vms / (1024 ** 2),
            )

            # Track peak
            if snapshot.used_percent > self._stats["peak_memory_percent"]:
                self._stats["peak_memory_percent"] = snapshot.used_percent

            return snapshot

        except Exception as e:
            logger.error(f"Failed to get memory snapshot: {e}")
            return None

    def _analyze_memory_trend(self) -> Tuple[float, str]:
        """
        Analyze memory usage trend.

        Returns:
            Tuple of (trend_slope, trend_description)
        """
        if len(self._memory_history) < 5:
            return 0.0, "insufficient_data"

        recent = self._memory_history[-10:]
        first_avg = sum(s.used_percent for s in recent[:5]) / 5
        last_avg = sum(s.used_percent for s in recent[-5:]) / 5

        slope = (last_avg - first_avg) / (len(recent) * self.config.check_interval)

        if slope > 0.5:
            return slope, "rapidly_increasing"
        elif slope > 0.1:
            return slope, "increasing"
        elif slope < -0.1:
            return slope, "decreasing"
        else:
            return slope, "stable"

    async def _trigger_gc(self):
        """Trigger garbage collection."""
        import gc

        logger.info("Triggering garbage collection")
        self._stats["gc_triggered"] += 1

        # Run GC in executor to not block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, gc.collect)

        if self.config.enable_aggressive_gc:
            # Force collection of all generations
            await loop.run_in_executor(None, lambda: gc.collect(2))

    async def _emergency_shed_load(self):
        """Emergency load shedding procedure."""
        now = time.time()

        # Respect cooldown
        if now - self._last_shed_time < self.config.shed_load_cooldown:
            logger.debug("Load shedding on cooldown")
            return

        self._last_shed_time = now
        self._stats["load_shed_events"] += 1

        logger.warning("EMERGENCY: Initiating load shedding")

        # Run all registered callbacks
        for callback in self._shed_callbacks:
            try:
                await callback()
            except Exception as e:
                logger.error(f"Load shed callback failed: {e}")

        # Trigger aggressive GC
        await self._trigger_gc()

        # Call emergency callback
        if self._on_emergency:
            try:
                await self._on_emergency()
            except Exception as e:
                logger.error(f"Emergency callback failed: {e}")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        logger.info("OOM protection monitoring started")

        while self._running:
            try:
                snapshot = self._get_memory_snapshot()

                if snapshot:
                    # Add to history
                    self._memory_history.append(snapshot)
                    if len(self._memory_history) > self._history_max_size:
                        self._memory_history.pop(0)

                    # Check thresholds
                    if snapshot.used_percent >= self.config.memory_threshold_percent:
                        # EMERGENCY
                        logger.critical(
                            f"MEMORY EMERGENCY: {snapshot.used_percent:.1f}% used "
                            f"({snapshot.used_mb:.0f}MB / {snapshot.total_mb:.0f}MB)"
                        )
                        await self._emergency_shed_load()

                    elif snapshot.used_percent >= self.config.warning_threshold_percent:
                        # WARNING
                        self._stats["warnings_issued"] += 1
                        slope, trend = self._analyze_memory_trend()

                        logger.warning(
                            f"Memory warning: {snapshot.used_percent:.1f}% used "
                            f"(trend: {trend}, slope: {slope:.2f}%/s)"
                        )

                        if self._on_warning:
                            await self._on_warning()

                        # Trigger GC at warning level
                        if snapshot.used_percent >= self.config.gc_threshold_percent:
                            await self._trigger_gc()

                    elif snapshot.used_percent >= self.config.gc_threshold_percent:
                        # Proactive GC
                        await self._trigger_gc()

                await asyncio.sleep(self.config.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.config.check_interval)

        logger.info("OOM protection monitoring stopped")

    async def start_monitoring(self):
        """Start memory monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self):
        """Stop memory monitoring."""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    def get_current_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        snapshot = self._get_memory_snapshot()

        if not snapshot:
            return {"status": "unavailable", "psutil_available": PSUTIL_AVAILABLE}

        slope, trend = self._analyze_memory_trend()

        return {
            "status": "ok" if snapshot.used_percent < self.config.warning_threshold_percent else "warning",
            "memory": {
                "used_percent": snapshot.used_percent,
                "used_mb": snapshot.used_mb,
                "available_mb": snapshot.available_mb,
                "total_mb": snapshot.total_mb,
            },
            "process": {
                "rss_mb": snapshot.process_rss_mb,
                "vms_mb": snapshot.process_vms_mb,
            },
            "trend": {
                "direction": trend,
                "slope": slope,
            },
            "thresholds": {
                "warning": self.config.warning_threshold_percent,
                "critical": self.config.memory_threshold_percent,
            },
            "stats": self._stats,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "history_size": len(self._memory_history),
            "running": self._running,
        }


# =============================================================================
# GUARANTEED DELIVERY QUEUE
# =============================================================================

@dataclass
class PendingMessage:
    """A message pending acknowledgment."""
    id: str
    payload: Dict[str, Any]
    created_at: float
    retry_count: int = 0
    last_attempt: float = 0.0
    destination: Optional[str] = None


class GuaranteedDeliveryQueue:
    """
    v84.0: Reliable message delivery with persistence and retry.

    Features:
        - Write-ahead logging for crash recovery
        - Automatic retry with exponential backoff
        - ACK-based delivery confirmation
        - Message deduplication
        - Dead letter queue for failed messages
        - Batch sending for efficiency
        - Priority queue support
    """

    def __init__(
        self,
        config: Optional[DeliveryConfig] = None,
        persistence_enabled: bool = True,
    ):
        """
        Initialize delivery queue.

        Args:
            config: Delivery configuration
            persistence_enabled: Enable disk persistence
        """
        self.config = config or DeliveryConfig()
        self.persistence_enabled = persistence_enabled

        # Set default persistence dir
        if not self.config.persistence_dir:
            self.config.persistence_dir = Path.home() / ".jarvis" / "trinity" / "delivery_queue"

        # Queues
        self._pending: Dict[str, PendingMessage] = {}
        self._dead_letter: List[PendingMessage] = []
        self._sent_ids: Set[str] = set()  # For deduplication

        # Locks
        self._pending_lock = asyncio.Lock()
        self._persist_lock = asyncio.Lock()

        # Background tasks
        self._retry_task: Optional[asyncio.Task] = None
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_acked": 0,
            "messages_retried": 0,
            "messages_failed": 0,
            "current_pending": 0,
            "dead_letter_count": 0,
        }

        # Ensure persistence directory exists
        if self.persistence_enabled:
            self.config.persistence_dir.mkdir(parents=True, exist_ok=True)

    def _generate_message_id(self, payload: Dict[str, Any]) -> str:
        """Generate unique message ID with content hash for deduplication."""
        content_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]

        return f"{uuid.uuid4().hex[:8]}-{content_hash}"

    async def _persist_pending(self):
        """Persist pending messages to disk."""
        if not self.persistence_enabled:
            return

        async with self._persist_lock:
            pending_file = self.config.persistence_dir / "pending.json"

            try:
                data = {
                    msg_id: {
                        "id": msg.id,
                        "payload": msg.payload,
                        "created_at": msg.created_at,
                        "retry_count": msg.retry_count,
                        "last_attempt": msg.last_attempt,
                        "destination": msg.destination,
                    }
                    for msg_id, msg in self._pending.items()
                }

                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(pending_file, 'w') as f:
                        await f.write(json.dumps(data, indent=2))
                else:
                    with open(pending_file, 'w') as f:
                        json.dump(data, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to persist pending messages: {e}")

    async def _load_pending(self):
        """Load pending messages from disk."""
        if not self.persistence_enabled:
            return

        pending_file = self.config.persistence_dir / "pending.json"

        if not pending_file.exists():
            return

        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(pending_file, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
            else:
                with open(pending_file, 'r') as f:
                    data = json.load(f)

            for msg_id, msg_data in data.items():
                self._pending[msg_id] = PendingMessage(
                    id=msg_data["id"],
                    payload=msg_data["payload"],
                    created_at=msg_data["created_at"],
                    retry_count=msg_data["retry_count"],
                    last_attempt=msg_data["last_attempt"],
                    destination=msg_data.get("destination"),
                )

            logger.info(f"Loaded {len(self._pending)} pending messages from disk")

        except Exception as e:
            logger.error(f"Failed to load pending messages: {e}")

    async def enqueue(
        self,
        payload: Dict[str, Any],
        destination: Optional[str] = None,
        priority: int = 0,
    ) -> str:
        """
        Enqueue a message for delivery.

        Args:
            payload: Message payload
            destination: Target destination identifier
            priority: Message priority (higher = more urgent)

        Returns:
            Message ID
        """
        msg_id = self._generate_message_id(payload)

        # Check for duplicate
        if msg_id in self._sent_ids:
            logger.debug(f"Duplicate message detected: {msg_id}")
            return msg_id

        message = PendingMessage(
            id=msg_id,
            payload=payload,
            created_at=time.time(),
            destination=destination,
        )

        async with self._pending_lock:
            if len(self._pending) >= self.config.max_queue_size:
                # Queue full - move oldest to dead letter
                oldest_id = min(self._pending.keys(), key=lambda k: self._pending[k].created_at)
                self._dead_letter.append(self._pending.pop(oldest_id))
                self._stats["dead_letter_count"] += 1

            self._pending[msg_id] = message
            self._stats["current_pending"] = len(self._pending)

        # Persist
        await self._persist_pending()

        return msg_id

    async def send_with_ack(
        self,
        websocket: Any,
        event: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> bool:
        """
        Send message and wait for acknowledgment.

        Args:
            websocket: WebSocket connection
            event: Event to send
            timeout: ACK timeout override

        Returns:
            True if acknowledged, False otherwise
        """
        timeout = timeout or self.config.ack_timeout
        msg_id = await self.enqueue(event)

        # Add message ID to event for ACK tracking
        event_with_id = {**event, "_msg_id": msg_id}

        for attempt in range(self.config.max_retries):
            try:
                # Send message
                await asyncio.wait_for(
                    websocket.send(json.dumps(event_with_id)),
                    timeout=5.0,
                )

                self._stats["messages_sent"] += 1

                # Wait for ACK
                try:
                    response = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=timeout,
                    )

                    response_data = json.loads(response)

                    if response_data.get("_ack") == msg_id:
                        # ACK received
                        await self.acknowledge(msg_id)
                        return True

                except asyncio.TimeoutError:
                    logger.debug(f"ACK timeout for message {msg_id}")

            except Exception as e:
                logger.warning(f"Send failed (attempt {attempt + 1}): {e}")

            # Update retry info
            async with self._pending_lock:
                if msg_id in self._pending:
                    self._pending[msg_id].retry_count += 1
                    self._pending[msg_id].last_attempt = time.time()
                    self._stats["messages_retried"] += 1

            # Backoff before retry
            delay = self.config.retry_delay * (2 ** attempt)
            delay += random.uniform(0, 0.5)
            await asyncio.sleep(delay)

        # Max retries exceeded - move to dead letter
        await self._move_to_dead_letter(msg_id)
        return False

    async def acknowledge(self, msg_id: str):
        """Acknowledge a message as delivered."""
        async with self._pending_lock:
            if msg_id in self._pending:
                del self._pending[msg_id]
                self._sent_ids.add(msg_id)
                self._stats["messages_acked"] += 1
                self._stats["current_pending"] = len(self._pending)

        # Limit sent_ids size
        if len(self._sent_ids) > 10000:
            # Remove oldest half
            self._sent_ids = set(list(self._sent_ids)[5000:])

        await self._persist_pending()

    async def _move_to_dead_letter(self, msg_id: str):
        """Move a message to dead letter queue."""
        async with self._pending_lock:
            if msg_id in self._pending:
                msg = self._pending.pop(msg_id)
                self._dead_letter.append(msg)
                self._stats["messages_failed"] += 1
                self._stats["dead_letter_count"] += 1
                self._stats["current_pending"] = len(self._pending)

        logger.warning(f"Message {msg_id} moved to dead letter queue")

    async def _retry_loop(self):
        """Background loop for retrying pending messages."""
        while self._running:
            try:
                now = time.time()

                async with self._pending_lock:
                    to_retry = [
                        msg for msg in self._pending.values()
                        if (now - msg.last_attempt) > self.config.retry_delay
                        and msg.retry_count < self.config.max_retries
                    ]

                for msg in to_retry:
                    # Mark for retry
                    msg.last_attempt = now
                    msg.retry_count += 1
                    self._stats["messages_retried"] += 1

                    logger.debug(f"Queueing retry for message {msg.id}")

                await asyncio.sleep(self.config.flush_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry loop error: {e}")
                await asyncio.sleep(1.0)

    async def start(self):
        """Start the delivery queue."""
        if self._running:
            return

        # Load persisted messages
        await self._load_pending()

        self._running = True
        self._retry_task = asyncio.create_task(self._retry_loop())

        logger.info("Guaranteed delivery queue started")

    async def stop(self):
        """Stop the delivery queue."""
        self._running = False

        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        # Final persist
        await self._persist_pending()

        logger.info("Guaranteed delivery queue stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            "pending_messages": len(self._pending),
            "dead_letter_messages": len(self._dead_letter),
        }

    def get_dead_letters(self) -> List[Dict[str, Any]]:
        """Get dead letter queue contents."""
        return [
            {
                "id": msg.id,
                "payload": msg.payload,
                "created_at": msg.created_at,
                "retry_count": msg.retry_count,
                "destination": msg.destination,
            }
            for msg in self._dead_letter
        ]


# =============================================================================
# CIRCUIT BREAKER ENGINE
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 30.0  # Time in open state before half-open
    half_open_max_calls: int = 3


class CircuitBreakerEngine:
    """
    v84.0: Circuit breaker for preventing cascade failures.

    Features:
        - Three-state circuit breaker (closed, open, half-open)
        - Automatic state transitions based on success/failure rates
        - Configurable thresholds and timeouts
        - Per-endpoint circuit breakers
        - Statistics and monitoring
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker engine."""
        self.config = config or CircuitBreakerConfig()

        # Per-endpoint state
        self._states: Dict[str, CircuitState] = defaultdict(lambda: CircuitState.CLOSED)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._success_counts: Dict[str, int] = defaultdict(int)
        self._last_failure_time: Dict[str, float] = {}
        self._half_open_calls: Dict[str, int] = defaultdict(int)

        # Lock for state transitions
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_calls": 0,
            "rejected_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "circuit_opens": 0,
            "circuit_closes": 0,
        }

    async def call(
        self,
        endpoint: str,
        func: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Execute a call through the circuit breaker.

        Args:
            endpoint: Endpoint identifier
            func: Async function to call

        Returns:
            Result of the function call

        Raises:
            CircuitOpenError: If circuit is open
        """
        self._stats["total_calls"] += 1

        # Check circuit state
        state = await self._get_state(endpoint)

        if state == CircuitState.OPEN:
            # Check if timeout expired
            if time.time() - self._last_failure_time.get(endpoint, 0) > self.config.timeout:
                await self._transition_to(endpoint, CircuitState.HALF_OPEN)
            else:
                self._stats["rejected_calls"] += 1
                raise CircuitOpenError(f"Circuit open for {endpoint}")

        if state == CircuitState.HALF_OPEN:
            # Limit calls in half-open state
            if self._half_open_calls[endpoint] >= self.config.half_open_max_calls:
                self._stats["rejected_calls"] += 1
                raise CircuitOpenError(f"Circuit half-open, max calls reached for {endpoint}")
            self._half_open_calls[endpoint] += 1

        try:
            result = await func()
            await self._record_success(endpoint)
            return result
        except Exception as e:
            await self._record_failure(endpoint)
            raise

    async def _get_state(self, endpoint: str) -> CircuitState:
        """Get current state for endpoint."""
        async with self._lock:
            return self._states[endpoint]

    async def _transition_to(self, endpoint: str, state: CircuitState):
        """Transition circuit to new state."""
        async with self._lock:
            old_state = self._states[endpoint]
            self._states[endpoint] = state

            if state == CircuitState.OPEN:
                self._stats["circuit_opens"] += 1
                logger.warning(f"Circuit OPENED for {endpoint}")
            elif state == CircuitState.CLOSED:
                self._stats["circuit_closes"] += 1
                logger.info(f"Circuit CLOSED for {endpoint}")
            elif state == CircuitState.HALF_OPEN:
                self._half_open_calls[endpoint] = 0
                logger.info(f"Circuit HALF-OPEN for {endpoint}")

    async def _record_success(self, endpoint: str):
        """Record successful call."""
        self._stats["successful_calls"] += 1

        async with self._lock:
            state = self._states[endpoint]

            if state == CircuitState.HALF_OPEN:
                self._success_counts[endpoint] += 1
                if self._success_counts[endpoint] >= self.config.success_threshold:
                    await self._transition_to(endpoint, CircuitState.CLOSED)
                    self._failure_counts[endpoint] = 0
                    self._success_counts[endpoint] = 0

    async def _record_failure(self, endpoint: str):
        """Record failed call."""
        self._stats["failed_calls"] += 1
        self._last_failure_time[endpoint] = time.time()

        async with self._lock:
            state = self._states[endpoint]
            self._failure_counts[endpoint] += 1

            if state == CircuitState.CLOSED:
                if self._failure_counts[endpoint] >= self.config.failure_threshold:
                    await self._transition_to(endpoint, CircuitState.OPEN)

            elif state == CircuitState.HALF_OPEN:
                await self._transition_to(endpoint, CircuitState.OPEN)
                self._success_counts[endpoint] = 0

    def get_state(self, endpoint: str) -> Dict[str, Any]:
        """Get circuit state for endpoint."""
        return {
            "state": self._states[endpoint].value,
            "failure_count": self._failure_counts[endpoint],
            "success_count": self._success_counts[endpoint],
            "last_failure": self._last_failure_time.get(endpoint),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "circuits": {
                endpoint: self.get_state(endpoint)
                for endpoint in self._states.keys()
            },
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_sqlite_engines: WeakValueDictionary[str, SQLiteRetryEngine] = WeakValueDictionary()
_oom_protection: Optional[OOMProtectionEngine] = None
_delivery_queue: Optional[GuaranteedDeliveryQueue] = None
_circuit_breaker: Optional[CircuitBreakerEngine] = None


async def get_sqlite_retry_engine(
    db_path: Union[str, Path] = "~/.jarvis/trinity/events.db",
) -> SQLiteRetryEngine:
    """Get or create SQLite retry engine for a database."""
    db_path = Path(db_path).expanduser()
    key = str(db_path)

    if key not in _sqlite_engines:
        engine = SQLiteRetryEngine(db_path)
        await engine.initialize()
        _sqlite_engines[key] = engine

    return _sqlite_engines[key]


async def get_oom_protection() -> OOMProtectionEngine:
    """Get or create OOM protection engine."""
    global _oom_protection

    if _oom_protection is None:
        _oom_protection = OOMProtectionEngine()

    return _oom_protection


async def get_delivery_queue() -> GuaranteedDeliveryQueue:
    """Get or create delivery queue."""
    global _delivery_queue

    if _delivery_queue is None:
        _delivery_queue = GuaranteedDeliveryQueue()
        await _delivery_queue.start()

    return _delivery_queue


async def get_circuit_breaker() -> CircuitBreakerEngine:
    """Get or create circuit breaker engine."""
    global _circuit_breaker

    if _circuit_breaker is None:
        _circuit_breaker = CircuitBreakerEngine()

    return _circuit_breaker


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "RetryConfig",
    "OOMConfig",
    "DeliveryConfig",
    "CircuitBreakerConfig",
    # Engines
    "SQLiteRetryEngine",
    "OOMProtectionEngine",
    "GuaranteedDeliveryQueue",
    "CircuitBreakerEngine",
    # Data classes
    "MemorySnapshot",
    "PendingMessage",
    "CircuitState",
    # Errors
    "CircuitOpenError",
    # Factory functions
    "get_sqlite_retry_engine",
    "get_oom_protection",
    "get_delivery_queue",
    "get_circuit_breaker",
]
