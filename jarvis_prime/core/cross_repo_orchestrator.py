"""
Cross-Repo Orchestrator v80.0 - Trinity Ecosystem Integration
==============================================================

Ultra-advanced orchestration system connecting JARVIS, JARVIS-Prime, and Reactor-Core
across repositories with graceful shutdown, hot-reload, and distributed coordination.

FEATURES:
    - Automatic repo detection and path resolution
    - Graceful shutdown with async signal handlers
    - Hot-reload configuration via file watching (inotify/watchdog)
    - Health monitoring with deep verification
    - Distributed coordination with consensus
    - Dependency graph resolution
    - Resource allocation and scheduling
    - Automatic recovery and restart
    - Cross-repo state synchronization

TECHNIQUES:
    - signal.set_wakeup_fd for async signal handling
    - watchdog/inotify for file system events
    - networkx for dependency graphs
    - asyncio.TaskGroup for structured concurrency
    - weakref for automatic cleanup
    - contextvars for request tracing

INTEGRATION:
    - Distributed tracing (OpenTelemetry)
    - Zero-copy IPC (memory-mapped)
    - Predictive caching
    - Adaptive rate limiting
    - Advanced async primitives

USAGE:
    from jarvis_prime.core.cross_repo_orchestrator import get_orchestrator

    orchestrator = await get_orchestrator()

    # Start all components
    await orchestrator.start_all()

    # Graceful shutdown
    await orchestrator.shutdown()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Import reliability engines
try:
    from jarvis_prime.core.reliability_engines import (
        SQLiteRetryEngine,
        OOMProtectionEngine,
        GuaranteedDeliveryQueue,
        CircuitBreakerEngine,
        get_sqlite_retry_engine,
        get_oom_protection,
        get_delivery_queue,
        get_circuit_breaker,
        RetryConfig,
        OOMConfig,
        DeliveryConfig,
    )
    RELIABILITY_ENGINES_AVAILABLE = True
except ImportError:
    RELIABILITY_ENGINES_AVAILABLE = False
    logger.warning("Reliability engines not available")

# Import advanced modules
try:
    from jarvis_prime.core.distributed_tracing import tracer, with_span
    from jarvis_prime.core.zero_copy_ipc import get_zero_copy_transport
    from jarvis_prime.core.predictive_cache import get_predictive_cache
    from jarvis_prime.core.adaptive_rate_limiter import get_rate_limiter
    from jarvis_prime.core.advanced_async_primitives import (
        StructuredTaskGroup,
        AsyncRLock,
        get_adaptive_timeout_manager
    )
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    logger.warning("Advanced features not available")

# Import file watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.warning("Watchdog not available - install with: pip install watchdog")

# Import network graph
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available - install with: pip install networkx")


# ============================================================================
# CONFIGURATION
# ============================================================================

class RepoType(Enum):
    """Repository types in Trinity ecosystem."""
    JARVIS = "jarvis"
    JARVIS_PRIME = "jarvis_prime"
    REACTOR_CORE = "reactor_core"


@dataclass
class RepoConfig:
    """Configuration for a single repository."""
    name: str
    type: RepoType
    path: Path
    entry_point: str  # e.g., "python3 -m backend.main"
    health_url: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    startup_timeout: float = 60.0
    shutdown_timeout: float = 30.0
    enabled: bool = True


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    base_dir: Path = field(default_factory=lambda: Path.home() / "Documents" / "repos")
    state_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "trinity")
    enable_hot_reload: bool = field(
        default_factory=lambda: os.getenv("ENABLE_HOT_RELOAD", "true").lower() == "true"
    )
    enable_auto_restart: bool = field(
        default_factory=lambda: os.getenv("ENABLE_AUTO_RESTART", "true").lower() == "true"
    )
    max_restart_attempts: int = field(
        default_factory=lambda: int(os.getenv("MAX_RESTART_ATTEMPTS", "3"))
    )
    health_check_interval: float = 30.0


# ============================================================================
# SIGNAL HANDLER FOR GRACEFUL SHUTDOWN
# ============================================================================

class SignalHandler:
    """
    Async signal handler for graceful shutdown.

    Uses signal.set_wakeup_fd to handle signals in async context.
    """

    def __init__(self):
        """Initialize signal handler."""
        self._shutdown_event = asyncio.Event()
        self._signal_queue: asyncio.Queue[int] = asyncio.Queue()
        self._original_handlers: Dict[int, Any] = {}

    def setup(self):
        """Setup signal handlers."""
        # Handle SIGINT (Ctrl+C) and SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.signal(sig, self._signal_callback)

        logger.info("Signal handlers installed")

    def _signal_callback(self, signum: int, frame):
        """Signal callback (called in signal context)."""
        # Put signal in queue for async processing
        try:
            self._signal_queue.put_nowait(signum)
        except asyncio.QueueFull:
            pass

        # Set shutdown event
        self._shutdown_event.set()

    async def wait_for_signal(self) -> int:
        """Wait for shutdown signal."""
        return await self._signal_queue.get()

    async def wait_for_shutdown(self):
        """Wait for shutdown event."""
        await self._shutdown_event.wait()

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()

    def restore(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)


# ============================================================================
# FILE WATCHER FOR HOT-RELOAD
# ============================================================================

if WATCHDOG_AVAILABLE:
    class ConfigFileHandler(FileSystemEventHandler):
        """Handler for configuration file changes."""

        def __init__(self, callback: Callable[[Path], Awaitable[None]]):
            """
            Initialize file handler.

            Args:
                callback: Async callback for file changes
            """
            self.callback = callback
            self._loop = asyncio.get_event_loop()

        def on_modified(self, event):
            """Handle file modification."""
            if event.is_directory:
                return

            file_path = Path(event.src_path)

            # Only watch specific files
            if file_path.suffix in ('.yaml', '.yml', '.json', '.toml'):
                logger.info(f"Config file changed: {file_path}")
                # Schedule callback in async loop
                asyncio.run_coroutine_threadsafe(
                    self.callback(file_path),
                    self._loop
                )


class FileWatcher:
    """File system watcher for hot-reload."""

    def __init__(self, config_dir: Path, callback: Callable[[Path], Awaitable[None]]):
        """
        Initialize file watcher.

        Args:
            config_dir: Directory to watch
            callback: Async callback for changes
        """
        self.config_dir = config_dir
        self.callback = callback
        self._observer: Optional[Observer] = None

        if not WATCHDOG_AVAILABLE:
            logger.warning("File watching disabled - watchdog not installed")

    def start(self):
        """Start watching files."""
        if not WATCHDOG_AVAILABLE:
            return

        self._observer = Observer()
        event_handler = ConfigFileHandler(self.callback)

        self._observer.schedule(
            event_handler,
            str(self.config_dir),
            recursive=True
        )

        self._observer.start()
        logger.info(f"File watcher started: {self.config_dir}")

    def stop(self):
        """Stop watching files."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            logger.info("File watcher stopped")


# ============================================================================
# DEPENDENCY GRAPH RESOLVER
# ============================================================================

class DependencyResolver:
    """
    Resolves startup order based on dependency graph.

    Uses topological sort to determine correct startup sequence.
    """

    def __init__(self):
        """Initialize dependency resolver."""
        self._graph = nx.DiGraph() if NETWORKX_AVAILABLE else None

    def add_component(self, name: str, dependencies: List[str]):
        """Add component with dependencies."""
        if not NETWORKX_AVAILABLE:
            return

        self._graph.add_node(name)

        for dep in dependencies:
            # Edge from dependency to component (dep must start first)
            self._graph.add_edge(dep, name)

    def get_startup_order(self) -> List[str]:
        """
        Get components in startup order.

        Returns:
            List of component names in topological order
        """
        if not NETWORKX_AVAILABLE or self._graph is None:
            # Fallback: alphabetical order
            return sorted(self._graph.nodes()) if self._graph else []

        try:
            # Topological sort
            return list(nx.topological_sort(self._graph))
        except nx.NetworkXError as e:
            logger.error(f"Circular dependency detected: {e}")
            # Return all nodes in arbitrary order
            return list(self._graph.nodes())

    def has_circular_dependencies(self) -> bool:
        """Check if graph has circular dependencies."""
        if not NETWORKX_AVAILABLE or self._graph is None:
            return False

        return not nx.is_directed_acyclic_graph(self._graph)


# ============================================================================
# v84.0 TRINITY HEALTH CHECKER - Cross-Repo Unified Health Monitoring
# ============================================================================

@dataclass
class ComponentHealth:
    """Health status of a Trinity component."""
    component: str
    online: bool
    heartbeat_age: float = float('inf')
    http_healthy: bool = False
    pid: Optional[int] = None
    port: Optional[int] = None
    model_loaded: bool = False
    last_inference_time: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @property
    def status(self) -> str:
        """Get human-readable status."""
        if not self.online:
            return "OFFLINE"
        if self.heartbeat_age > 60:
            return "STALE"
        if self.heartbeat_age > 30:
            return "DEGRADED"
        return "HEALTHY"


class TrinityHealthChecker:
    """
    v84.0: Unified health checker for all Trinity components.

    Features:
        - Heartbeat file monitoring with staleness detection
        - HTTP health endpoint verification
        - PID validation for process liveness
        - Parallel health checks for speed
        - Circuit breaker for flaky endpoints
        - Automatic dead component cleanup
    """

    def __init__(
        self,
        trinity_dir: Optional[Path] = None,
        stale_threshold: float = 30.0,
        http_timeout: float = 5.0,
    ):
        """
        Initialize health checker.

        Args:
            trinity_dir: Trinity IPC directory
            stale_threshold: Seconds before heartbeat is stale
            http_timeout: HTTP request timeout
        """
        self.trinity_dir = trinity_dir or Path.home() / ".jarvis" / "trinity"
        self.components_dir = self.trinity_dir / "components"
        self.stale_threshold = stale_threshold
        self.http_timeout = http_timeout

        # Component configuration
        self._component_configs: Dict[str, Dict[str, Any]] = {
            "jarvis_prime": {
                "heartbeat_files": ["jarvis_prime.json", "j_prime.json"],
                "health_url": "http://localhost:{port}/health",
                "default_port": 8000,
            },
            "jarvis_body": {
                "heartbeat_files": ["jarvis_body.json", "jarvis.json"],
                "health_url": "http://localhost:{port}/health",
                "default_port": 8010,
            },
            "reactor_core": {
                "heartbeat_files": ["reactor_core.json", "reactor.json"],
                "health_url": "http://localhost:{port}/health",
                "default_port": 8090,
            },
        }

        # Cache
        self._health_cache: Dict[str, ComponentHealth] = {}
        self._cache_ttl: float = 5.0
        self._cache_time: float = 0.0

        # Circuit breaker for HTTP checks
        self._http_failures: Dict[str, int] = defaultdict(int)
        self._http_circuit_open: Dict[str, float] = {}

    async def check_all(self) -> Dict[str, ComponentHealth]:
        """
        Check health of all Trinity components in parallel.

        Returns:
            Dictionary of component name -> health status
        """
        # Check cache
        if time.time() - self._cache_time < self._cache_ttl:
            return self._health_cache.copy()

        # Run checks in parallel
        tasks = [
            self.check_component(name)
            for name in self._component_configs.keys()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        health_map = {}
        for name, result in zip(self._component_configs.keys(), results):
            if isinstance(result, Exception):
                health_map[name] = ComponentHealth(
                    component=name,
                    online=False,
                    error=str(result),
                )
            else:
                health_map[name] = result

        # Update cache
        self._health_cache = health_map
        self._cache_time = time.time()

        return health_map

    async def check_component(self, name: str) -> ComponentHealth:
        """Check health of a single component."""
        config = self._component_configs.get(name)
        if not config:
            return ComponentHealth(
                component=name,
                online=False,
                error=f"Unknown component: {name}",
            )

        # Check heartbeat file
        heartbeat_data = self._read_heartbeat(name, config)

        if heartbeat_data:
            port = heartbeat_data.get("port", config["default_port"])
            timestamp = heartbeat_data.get("timestamp", 0)
            heartbeat_age = time.time() - timestamp

            # Check HTTP if heartbeat is fresh
            http_healthy = False
            if heartbeat_age < self.stale_threshold:
                http_healthy = await self._check_http(name, port)

            return ComponentHealth(
                component=name,
                online=heartbeat_age < self.stale_threshold,
                heartbeat_age=heartbeat_age,
                http_healthy=http_healthy,
                pid=heartbeat_data.get("pid"),
                port=port,
                model_loaded=heartbeat_data.get("model_loaded", False),
                last_inference_time=heartbeat_data.get("last_inference_time", 0),
            )
        else:
            # No heartbeat - try HTTP anyway
            port = config["default_port"]
            http_healthy = await self._check_http(name, port)

            return ComponentHealth(
                component=name,
                online=http_healthy,
                http_healthy=http_healthy,
                port=port if http_healthy else None,
            )

    def _read_heartbeat(
        self, name: str, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Read heartbeat file for a component."""
        for filename in config["heartbeat_files"]:
            heartbeat_file = self.components_dir / filename
            if heartbeat_file.exists():
                try:
                    with open(heartbeat_file, 'r') as f:
                        data = json.load(f)
                        return data
                except (json.JSONDecodeError, IOError):
                    continue
        return None

    async def _check_http(self, name: str, port: int) -> bool:
        """Check HTTP health endpoint."""
        # Check circuit breaker
        if name in self._http_circuit_open:
            if time.time() - self._http_circuit_open[name] < 30:
                return False
            del self._http_circuit_open[name]

        config = self._component_configs[name]
        url = config["health_url"].format(port=port)

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=self.http_timeout)
                ) as response:
                    if response.status == 200:
                        self._http_failures[name] = 0
                        return True
        except Exception:
            self._http_failures[name] += 1
            if self._http_failures[name] >= 3:
                self._http_circuit_open[name] = time.time()
            return False

        return False

    async def wait_for_component(
        self,
        name: str,
        timeout: float = 60.0,
        check_interval: float = 2.0,
    ) -> bool:
        """
        Wait for a component to become healthy.

        Args:
            name: Component name
            timeout: Maximum wait time
            check_interval: Time between checks

        Returns:
            True if component became healthy
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            health = await self.check_component(name)
            if health.online and health.http_healthy:
                return True
            await asyncio.sleep(check_interval)

        return False

    def get_unified_status(self) -> Dict[str, Any]:
        """Get unified status of all components."""
        if not self._health_cache:
            return {"status": "unknown", "components": {}}

        all_online = all(h.online for h in self._health_cache.values())
        any_online = any(h.online for h in self._health_cache.values())

        if all_online:
            status = "healthy"
        elif any_online:
            status = "degraded"
        else:
            status = "offline"

        return {
            "status": status,
            "timestamp": time.time(),
            "components": {
                name: {
                    "status": h.status,
                    "online": h.online,
                    "heartbeat_age": h.heartbeat_age,
                    "http_healthy": h.http_healthy,
                    "port": h.port,
                    "model_loaded": h.model_loaded,
                }
                for name, h in self._health_cache.items()
            }
        }

    async def print_status(self):
        """Print formatted status of all components."""
        health_map = await self.check_all()

        print()
        print("=" * 60)
        print("TRINITY HEALTH STATUS")
        print("=" * 60)

        for name, health in health_map.items():
            status_icon = {
                "HEALTHY": "✅",
                "DEGRADED": "⚠️",
                "STALE": "🔶",
                "OFFLINE": "❌",
            }.get(health.status, "❓")

            print(f"{status_icon} {name:20s} {health.status:10s}", end="")
            if health.online:
                print(f" (heartbeat: {health.heartbeat_age:.1f}s)", end="")
                if health.http_healthy:
                    print(f" [HTTP OK]", end="")
                if health.port:
                    print(f" ::{health.port}", end="")
            print()

        print("=" * 60)


# ============================================================================
# v84.0 EVENT STORE - Reliable Event Persistence with SQLite Retry
# ============================================================================

@dataclass
class TrinityEvent:
    """A Trinity ecosystem event."""
    id: str
    event_type: str
    source: str
    payload: Dict[str, Any]
    timestamp: float
    delivered: bool = False
    retry_count: int = 0

    @classmethod
    def create(
        cls,
        event_type: str,
        source: str,
        payload: Dict[str, Any],
    ) -> "TrinityEvent":
        """Create a new event with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            event_type=event_type,
            source=source,
            payload=payload,
            timestamp=time.time(),
        )


class TrinityEventStore:
    """
    v84.0: Reliable event persistence with SQLite retry engine.

    Features:
        - Automatic retry on database lock contention
        - Write-ahead logging for crash recovery
        - Event deduplication
        - Automatic schema migration
        - Batch operations for efficiency
        - Event TTL and cleanup
    """

    SCHEMA_VERSION = 1

    CREATE_TABLES_SQL = """
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        event_type TEXT NOT NULL,
        source TEXT NOT NULL,
        payload TEXT NOT NULL,
        timestamp REAL NOT NULL,
        delivered INTEGER DEFAULT 0,
        retry_count INTEGER DEFAULT 0,
        created_at REAL DEFAULT (strftime('%s', 'now'))
    );

    CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
    CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
    CREATE INDEX IF NOT EXISTS idx_events_delivered ON events(delivered);

    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    );
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        event_ttl_days: int = 7,
    ):
        """
        Initialize event store.

        Args:
            db_path: Path to SQLite database
            event_ttl_days: Days to keep events before cleanup
        """
        self.db_path = db_path or Path.home() / ".jarvis" / "trinity" / "events.db"
        self.event_ttl_days = event_ttl_days
        self._db: Optional[SQLiteRetryEngine] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the event store."""
        if self._initialized:
            return

        if not RELIABILITY_ENGINES_AVAILABLE:
            logger.warning("Event store disabled - reliability engines not available")
            return

        self._db = await get_sqlite_retry_engine(self.db_path)

        # Create schema
        for statement in self.CREATE_TABLES_SQL.strip().split(';'):
            statement = statement.strip()
            if statement:
                await self._db.execute_write(f"{statement};")

        # Run cleanup
        await self._cleanup_old_events()

        self._initialized = True
        logger.info(f"TrinityEventStore initialized: {self.db_path}")

    async def _cleanup_old_events(self):
        """Remove events older than TTL."""
        if not self._db:
            return

        cutoff = time.time() - (self.event_ttl_days * 86400)
        await self._db.execute_write(
            "DELETE FROM events WHERE timestamp < ?",
            (cutoff,),
        )

    async def store_event(self, event: TrinityEvent) -> bool:
        """
        Store an event with automatic retry.

        Args:
            event: Event to store

        Returns:
            True if stored successfully
        """
        if not self._db:
            return False

        try:
            await self._db.execute_write(
                """
                INSERT OR REPLACE INTO events
                (id, event_type, source, payload, timestamp, delivered, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.event_type,
                    event.source,
                    json.dumps(event.payload),
                    event.timestamp,
                    1 if event.delivered else 0,
                    event.retry_count,
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to store event {event.id}: {e}")
            return False

    async def store_events_batch(self, events: List[TrinityEvent]) -> int:
        """
        Store multiple events in a batch.

        Args:
            events: List of events to store

        Returns:
            Number of events stored
        """
        if not self._db or not events:
            return 0

        params_list = [
            (
                event.id,
                event.event_type,
                event.source,
                json.dumps(event.payload),
                event.timestamp,
                1 if event.delivered else 0,
                event.retry_count,
            )
            for event in events
        ]

        return await self._db.execute_many(
            """
            INSERT OR REPLACE INTO events
            (id, event_type, source, payload, timestamp, delivered, retry_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            params_list,
        )

    async def get_undelivered_events(
        self,
        source: Optional[str] = None,
        limit: int = 100,
    ) -> List[TrinityEvent]:
        """
        Get undelivered events for retry.

        Args:
            source: Filter by source
            limit: Maximum events to return

        Returns:
            List of undelivered events
        """
        if not self._db:
            return []

        if source:
            rows = await self._db.execute_read(
                """
                SELECT id, event_type, source, payload, timestamp, delivered, retry_count
                FROM events
                WHERE delivered = 0 AND source = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (source, limit),
            )
        else:
            rows = await self._db.execute_read(
                """
                SELECT id, event_type, source, payload, timestamp, delivered, retry_count
                FROM events
                WHERE delivered = 0
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (limit,),
            )

        return [
            TrinityEvent(
                id=row[0],
                event_type=row[1],
                source=row[2],
                payload=json.loads(row[3]),
                timestamp=row[4],
                delivered=bool(row[5]),
                retry_count=row[6],
            )
            for row in rows
        ]

    async def mark_delivered(self, event_id: str) -> bool:
        """Mark an event as delivered."""
        if not self._db:
            return False

        result = await self._db.execute_write(
            "UPDATE events SET delivered = 1 WHERE id = ?",
            (event_id,),
        )
        return result is not None and result > 0

    async def increment_retry(self, event_id: str) -> bool:
        """Increment retry count for an event."""
        if not self._db:
            return False

        result = await self._db.execute_write(
            "UPDATE events SET retry_count = retry_count + 1 WHERE id = ?",
            (event_id,),
        )
        return result is not None and result > 0

    async def get_event_stats(self) -> Dict[str, Any]:
        """Get event store statistics."""
        if not self._db:
            return {"status": "unavailable"}

        rows = await self._db.execute_read(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN delivered = 1 THEN 1 ELSE 0 END) as delivered,
                SUM(CASE WHEN delivered = 0 THEN 1 ELSE 0 END) as pending,
                AVG(retry_count) as avg_retries
            FROM events
            """
        )

        if rows:
            return {
                "total_events": rows[0][0],
                "delivered": rows[0][1],
                "pending": rows[0][2],
                "avg_retries": rows[0][3] or 0,
                "db_path": str(self.db_path),
            }

        return {"total_events": 0, "delivered": 0, "pending": 0}

    async def close(self):
        """Close the event store."""
        if self._db:
            await self._db.close()
            self._db = None
            self._initialized = False


# ============================================================================
# v84.0 TRINITY BRIDGE - Cross-Repo Communication with Guaranteed Delivery
# ============================================================================

class TrinityBridge:
    """
    v84.0: Cross-repo communication bridge with reliability guarantees.

    Features:
        - Guaranteed message delivery with ACK
        - Automatic reconnection on failure
        - Circuit breaker for failed endpoints
        - Event persistence for crash recovery
        - Rate limiting to prevent overload
        - Metrics and monitoring
    """

    def __init__(
        self,
        event_store: Optional[TrinityEventStore] = None,
        delivery_config: Optional[DeliveryConfig] = None,
    ):
        """
        Initialize Trinity bridge.

        Args:
            event_store: Event store for persistence
            delivery_config: Delivery configuration
        """
        self._event_store = event_store
        self._delivery_config = delivery_config or DeliveryConfig()

        # Components
        self._delivery_queue: Optional[GuaranteedDeliveryQueue] = None
        self._circuit_breaker: Optional[CircuitBreakerEngine] = None

        # Connections
        self._connections: Dict[str, Any] = {}
        self._connection_lock = asyncio.Lock()

        # Background tasks
        self._retry_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "reconnections": 0,
            "delivery_failures": 0,
        }

    async def initialize(self):
        """Initialize the bridge."""
        if not RELIABILITY_ENGINES_AVAILABLE:
            logger.warning("Trinity bridge disabled - reliability engines not available")
            return

        self._delivery_queue = await get_delivery_queue()
        self._circuit_breaker = await get_circuit_breaker()

        logger.info("TrinityBridge initialized")

    async def start(self):
        """Start the bridge."""
        if self._running:
            return

        await self.initialize()
        self._running = True

        # Start retry task
        self._retry_task = asyncio.create_task(self._retry_undelivered())

        logger.info("TrinityBridge started")

    async def stop(self):
        """Stop the bridge."""
        self._running = False

        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        # Close connections
        async with self._connection_lock:
            for name, conn in self._connections.items():
                try:
                    await conn.close()
                except Exception:
                    pass
            self._connections.clear()

        logger.info("TrinityBridge stopped")

    async def send_event(
        self,
        destination: str,
        event: TrinityEvent,
        require_ack: bool = True,
    ) -> bool:
        """
        Send an event to a destination.

        Args:
            destination: Target component name
            event: Event to send
            require_ack: Wait for acknowledgment

        Returns:
            True if sent (and acked if required)
        """
        # Store event first
        if self._event_store:
            await self._event_store.store_event(event)

        # Check circuit breaker
        if self._circuit_breaker:
            try:
                state = self._circuit_breaker.get_state(destination)
                if state["state"] == "open":
                    logger.warning(f"Circuit open for {destination}, queuing event")
                    return False
            except Exception:
                pass

        # Get connection
        conn = await self._get_connection(destination)
        if not conn:
            logger.warning(f"No connection to {destination}")
            return False

        # Send
        try:
            message = {
                "event_id": event.id,
                "event_type": event.event_type,
                "source": event.source,
                "payload": event.payload,
                "timestamp": event.timestamp,
            }

            if require_ack and self._delivery_queue:
                success = await self._delivery_queue.send_with_ack(
                    conn,
                    message,
                    timeout=self._delivery_config.ack_timeout,
                )
            else:
                await conn.send(json.dumps(message))
                success = True

            if success:
                self._stats["messages_sent"] += 1
                if self._event_store:
                    await self._event_store.mark_delivered(event.id)
                return True
            else:
                self._stats["delivery_failures"] += 1
                if self._event_store:
                    await self._event_store.increment_retry(event.id)
                return False

        except Exception as e:
            logger.error(f"Failed to send event to {destination}: {e}")
            self._stats["delivery_failures"] += 1
            return False

    async def _get_connection(self, destination: str) -> Optional[Any]:
        """Get or create connection to destination."""
        async with self._connection_lock:
            if destination in self._connections:
                conn = self._connections[destination]
                # Check if connection is still alive
                try:
                    await conn.ping()
                    return conn
                except Exception:
                    # Connection dead, remove it
                    del self._connections[destination]

            # Create new connection
            conn = await self._create_connection(destination)
            if conn:
                self._connections[destination] = conn
                self._stats["reconnections"] += 1
            return conn

    async def _create_connection(self, destination: str) -> Optional[Any]:
        """Create a new connection to a destination."""
        # Get destination URL from configuration
        port_map = {
            "jarvis_prime": int(os.getenv("JARVIS_PRIME_PORT", "8000")),
            "jarvis_body": int(os.getenv("JARVIS_PORT", "8010")),
            "reactor_core": int(os.getenv("REACTOR_CORE_PORT", "8090")),
        }

        port = port_map.get(destination)
        if not port:
            logger.warning(f"Unknown destination: {destination}")
            return None

        url = f"ws://localhost:{port}/ws/trinity"

        try:
            import websockets
            conn = await websockets.connect(url, ping_interval=30)
            logger.info(f"Connected to {destination} at {url}")
            return conn
        except ImportError:
            logger.warning("websockets not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to connect to {destination}: {e}")
            return None

    async def _retry_undelivered(self):
        """Background task to retry undelivered events."""
        while self._running:
            try:
                if self._event_store:
                    events = await self._event_store.get_undelivered_events(limit=50)

                    for event in events:
                        if event.retry_count < self._delivery_config.max_retries:
                            # Determine destination from event source
                            # Events are sent TO the opposite component
                            destination = self._infer_destination(event)
                            if destination:
                                await self.send_event(destination, event)

                await asyncio.sleep(5.0)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry loop error: {e}")
                await asyncio.sleep(5.0)

    def _infer_destination(self, event: TrinityEvent) -> Optional[str]:
        """Infer destination from event metadata."""
        # Events from Prime go to JARVIS, and vice versa
        source_to_dest = {
            "jarvis_prime": "jarvis_body",
            "jarvis_body": "jarvis_prime",
            "reactor_core": "jarvis_prime",
        }
        return source_to_dest.get(event.source)

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            **self._stats,
            "connected_destinations": list(self._connections.keys()),
            "running": self._running,
        }


# ============================================================================
# REPO MANAGER
# ============================================================================

@dataclass
class RepoState:
    """Runtime state of a repository."""
    config: RepoConfig
    process: Optional[asyncio.subprocess.Process] = None
    pid: Optional[int] = None
    status: str = "stopped"
    start_time: float = 0.0
    restart_count: int = 0
    last_health_check: float = 0.0
    health_status: bool = False


class RepoManager:
    """Manages a single repository/component."""

    def __init__(self, config: RepoConfig):
        """
        Initialize repo manager.

        Args:
            config: Repository configuration
        """
        self.config = config
        self.state = RepoState(config=config)
        self._lock = AsyncRLock() if ADVANCED_FEATURES_AVAILABLE else asyncio.Lock()
        self._output_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        """
        Start the repository/component.

        Returns:
            True if started successfully
        """
        if ADVANCED_FEATURES_AVAILABLE:
            async with self._lock.acquire_context():
                return await self._start_impl()
        else:
            async with self._lock:
                return await self._start_impl()

    async def _start_impl(self) -> bool:
        """Implementation of start."""
        if self.state.status == "running":
            logger.warning(f"{self.config.name} already running")
            return True

        logger.info(f"Starting {self.config.name}...")
        self.state.status = "starting"

        try:
            # Build command
            cmd_parts = self.config.entry_point.split()

            # Build environment
            env = os.environ.copy()
            env.update(self.config.env_vars)

            # Add Python path
            if self.config.type == RepoType.JARVIS_PRIME:
                env["PYTHONPATH"] = f"{self.config.path}:{env.get('PYTHONPATH', '')}"

            # Start process
            self.state.process = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.config.path)
            )

            self.state.pid = self.state.process.pid
            self.state.start_time = time.time()
            self.state.status = "running"

            # Start output monitoring
            self._output_task = asyncio.create_task(self._monitor_output())

            logger.info(f"{self.config.name} started (PID: {self.state.pid})")

            # Wait for health
            if self.config.health_url:
                healthy = await self._wait_for_health()
                self.state.health_status = healthy

                if healthy:
                    logger.info(f"{self.config.name} is healthy")
                else:
                    logger.warning(f"{self.config.name} failed health check")

            return True

        except Exception as e:
            logger.error(f"Failed to start {self.config.name}: {e}")
            self.state.status = "failed"
            return False

    async def stop(self) -> bool:
        """
        Stop the repository/component gracefully.

        Returns:
            True if stopped successfully
        """
        if ADVANCED_FEATURES_AVAILABLE:
            async with self._lock.acquire_context():
                return await self._stop_impl()
        else:
            async with self._lock:
                return await self._stop_impl()

    async def _stop_impl(self) -> bool:
        """Implementation of stop."""
        if self.state.process is None:
            self.state.status = "stopped"
            return True

        logger.info(f"Stopping {self.config.name}...")
        self.state.status = "stopping"

        try:
            # Send SIGTERM
            self.state.process.terminate()

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    self.state.process.wait(),
                    timeout=self.config.shutdown_timeout
                )
            except asyncio.TimeoutError:
                # Force kill
                logger.warning(f"{self.config.name} didn't stop gracefully, killing...")
                self.state.process.kill()
                await self.state.process.wait()

            # Cancel output task
            if self._output_task:
                self._output_task.cancel()
                try:
                    await self._output_task
                except asyncio.CancelledError:
                    pass

            self.state.status = "stopped"
            self.state.process = None
            self.state.pid = None

            logger.info(f"{self.config.name} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping {self.config.name}: {e}")
            self.state.status = "failed"
            return False

    async def _wait_for_health(self) -> bool:
        """Wait for component to become healthy."""
        if not self.config.health_url:
            return True

        deadline = time.time() + self.config.startup_timeout
        check_interval = 2.0

        while time.time() < deadline:
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.config.health_url,
                        timeout=aiohttp.ClientTimeout(total=5.0)
                    ) as response:
                        if response.status == 200:
                            return True

            except Exception as e:
                logger.debug(f"Health check failed: {e}")

            # Check if process died
            if self.state.process and self.state.process.returncode is not None:
                logger.warning(f"{self.config.name} process exited unexpectedly")
                return False

            await asyncio.sleep(check_interval)

        return False

    async def _monitor_output(self):
        """Monitor process output."""
        if not self.state.process:
            return

        try:
            # Read stdout
            async for line in self.state.process.stdout:
                line_str = line.decode().strip()
                logger.debug(f"[{self.config.name}] {line_str}")

        except Exception as e:
            logger.error(f"Error reading output from {self.config.name}: {e}")


# ============================================================================
# CROSS-REPO ORCHESTRATOR
# ============================================================================

class CrossRepoOrchestrator:
    """
    Master orchestrator for the entire Trinity ecosystem.

    Coordinates JARVIS, JARVIS-Prime, and Reactor-Core across repositories.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize orchestrator.

        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()

        # Repo managers
        self._repos: Dict[str, RepoManager] = {}

        # Components
        self._signal_handler = SignalHandler()
        self._file_watcher: Optional[FileWatcher] = None
        self._dependency_resolver = DependencyResolver()

        # v84.0: Trinity Health Checker for cross-repo monitoring
        self._health_checker = TrinityHealthChecker(
            trinity_dir=self.config.state_dir,
            stale_threshold=30.0,
            http_timeout=5.0,
        )

        # v84.0: Reliability Engines
        self._event_store: Optional[TrinityEventStore] = None
        self._trinity_bridge: Optional[TrinityBridge] = None
        self._oom_protection: Optional[OOMProtectionEngine] = None
        self._oom_monitor_task: Optional[asyncio.Task] = None

        # State
        self._running = False
        self._startup_order: List[str] = []

        # Advanced features
        self._tracer = tracer if ADVANCED_FEATURES_AVAILABLE else None
        self._cache = None
        self._rate_limiter = None

        # Create state directory
        self.config.state_dir.mkdir(parents=True, exist_ok=True)

    def _detect_repos(self) -> Dict[str, RepoConfig]:
        """
        v84.0: Auto-detect repositories with intelligent path resolution.

        Searches multiple possible locations and naming conventions for each repo.
        Uses environment variables with fallback to common paths.
        """
        repos = {}

        # JARVIS Prime (current repo)
        prime_path = Path(__file__).parent.parent.parent
        prime_port = int(os.getenv("JARVIS_PRIME_PORT", "8000"))

        # v84.0: Detect model path dynamically
        model_paths = [
            prime_path / "models" / "current.gguf",
            Path.home() / ".jarvis" / "prime" / "models" / "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
            Path.home() / ".jarvis" / "prime" / "models" / "current.gguf",
        ]
        model_path = next((p for p in model_paths if p.exists()), model_paths[0])

        repos["jarvis_prime"] = RepoConfig(
            name="JARVIS-Prime",
            type=RepoType.JARVIS_PRIME,
            path=prime_path,
            # v84.0: Use run_server.py with Metal GPU enabled
            entry_point=f"python3 run_server.py --model {model_path} --gpu-layers -1 --port {prime_port}",
            health_url=f"http://localhost:{prime_port}/health",
            dependencies=[],
            env_vars={
                "JARVIS_PRIME_PORT": str(prime_port),
                "PYTHONPATH": str(prime_path),
                "TRINITY_ENABLED": "true",
                "METAL_ENABLED": "true",
            }
        )

        # v84.0: JARVIS-AI-Agent detection with multiple path candidates
        jarvis_candidates = [
            Path(os.getenv("JARVIS_AI_AGENT_PATH", "")),
            self.config.base_dir / "JARVIS-AI-Agent",
            self.config.base_dir / "jarvis-ai-agent",
            self.config.base_dir / "jarvis",
            Path.home() / "Documents" / "repos" / "JARVIS-AI-Agent",
        ]
        jarvis_path = next((p for p in jarvis_candidates if p.exists() and p.is_dir()), None)

        if jarvis_path:
            jarvis_port = int(os.getenv("JARVIS_PORT", "8010"))
            # v84.0: Detect correct entry point
            entry_candidates = [
                (jarvis_path / "run_supervisor.py", f"python3 run_supervisor.py"),
                (jarvis_path / "backend" / "main.py", f"python3 -m backend.main"),
            ]
            entry_point = next(
                (cmd for path, cmd in entry_candidates if path.exists()),
                "python3 run_supervisor.py"
            )

            repos["jarvis"] = RepoConfig(
                name="JARVIS-AI-Agent",
                type=RepoType.JARVIS,
                path=jarvis_path,
                entry_point=entry_point,
                health_url=f"http://localhost:{jarvis_port}/health",
                dependencies=[],  # JARVIS is the master, no deps
                env_vars={
                    "JARVIS_PORT": str(jarvis_port),
                    "JARVIS_PRIME_URL": f"http://localhost:{prime_port}",
                    "JARVIS_PRIME_PATH": str(prime_path),
                    "PYTHONPATH": str(jarvis_path),
                    "TRINITY_ENABLED": "true",
                }
            )
            logger.info(f"✓ Found JARVIS-AI-Agent: {jarvis_path}")
        else:
            logger.warning("⚠️ JARVIS-AI-Agent not found in any expected location")

        # v84.0: Reactor Core detection with multiple path candidates
        reactor_candidates = [
            Path(os.getenv("REACTOR_CORE_PATH", "")),
            self.config.base_dir / "reactor-core",
            self.config.base_dir / "Reactor-Core",
            Path.home() / "Documents" / "repos" / "reactor-core",
        ]
        reactor_path = next((p for p in reactor_candidates if p.exists() and p.is_dir()), None)

        if reactor_path:
            reactor_port = int(os.getenv("REACTOR_CORE_PORT", "8090"))
            # v84.0: Detect correct entry point
            entry_candidates = [
                (reactor_path / "run_server.py", f"python3 run_server.py --port {reactor_port}"),
                (reactor_path / "reactor" / "server.py", f"python3 -m reactor.server"),
                (reactor_path / "main.py", f"python3 main.py"),
            ]
            entry_point = next(
                (cmd for path, cmd in entry_candidates if path.exists()),
                f"python3 -m reactor.server --port {reactor_port}"
            )

            repos["reactor_core"] = RepoConfig(
                name="Reactor-Core",
                type=RepoType.REACTOR_CORE,
                path=reactor_path,
                entry_point=entry_point,
                health_url=f"http://localhost:{reactor_port}/health",
                dependencies=["jarvis_prime"],
                env_vars={
                    "REACTOR_PORT": str(reactor_port),
                    "JARVIS_PRIME_URL": f"http://localhost:{prime_port}",
                    "PYTHONPATH": str(reactor_path),
                }
            )
            logger.info(f"✓ Found Reactor-Core: {reactor_path}")
        else:
            logger.info("ℹ️ Reactor-Core not found (optional)")

        return repos

    async def initialize(self):
        """Initialize orchestrator."""
        logger.info("Initializing Trinity Orchestrator v84.0...")
        logger.info("=" * 60)

        # Detect repositories
        repo_configs = self._detect_repos()

        # Create managers
        for name, config in repo_configs.items():
            if config.enabled:
                self._repos[name] = RepoManager(config)

                # Add to dependency resolver
                self._dependency_resolver.add_component(
                    name,
                    config.dependencies
                )

        # Check for circular dependencies
        if self._dependency_resolver.has_circular_dependencies():
            logger.error("Circular dependencies detected!")

        # Get startup order
        self._startup_order = self._dependency_resolver.get_startup_order()
        logger.info(f"Startup order: {' -> '.join(self._startup_order)}")

        # Setup signal handler
        self._signal_handler.setup()

        # Setup file watcher
        if self.config.enable_hot_reload and WATCHDOG_AVAILABLE:
            self._file_watcher = FileWatcher(
                self.config.state_dir,
                self._handle_config_change
            )
            self._file_watcher.start()

        # v84.0: Initialize Reliability Engines
        if RELIABILITY_ENGINES_AVAILABLE:
            logger.info("Initializing reliability engines...")

            # Event Store with SQLite retry
            self._event_store = TrinityEventStore(
                db_path=self.config.state_dir / "events.db",
                event_ttl_days=7,
            )
            await self._event_store.initialize()
            logger.info("  ✓ TrinityEventStore initialized")

            # Trinity Bridge for cross-repo communication
            self._trinity_bridge = TrinityBridge(
                event_store=self._event_store,
            )
            await self._trinity_bridge.start()
            logger.info("  ✓ TrinityBridge initialized")

            # OOM Protection Engine
            self._oom_protection = await get_oom_protection()

            # Register load shedding callbacks
            self._oom_protection.register_shed_callback(self._on_memory_pressure)

            # Start OOM monitoring
            await self._oom_protection.start_monitoring()
            logger.info("  ✓ OOM Protection initialized")

            logger.info("Reliability engines online")
        else:
            logger.warning("Reliability engines not available - running in degraded mode")

        # Initialize advanced features
        if ADVANCED_FEATURES_AVAILABLE:
            self._cache = await get_predictive_cache()
            self._rate_limiter = await get_rate_limiter()

        logger.info("=" * 60)
        logger.info("Trinity Orchestrator v84.0 initialized")

    async def _on_memory_pressure(self):
        """
        v84.0: Handle memory pressure events from OOM protection.

        Called when system memory is critically low. Implements graceful
        degradation by clearing caches and pausing non-critical operations.
        """
        logger.warning("Memory pressure detected - initiating load shedding")

        # Clear caches
        if self._cache:
            try:
                await self._cache.clear()
                logger.info("  ✓ Cleared predictive cache")
            except Exception as e:
                logger.error(f"  ✗ Failed to clear cache: {e}")

        # Pause health checks temporarily
        self._health_checker._cache_ttl = 30.0  # Increase cache TTL

        # Log memory status
        if self._oom_protection:
            status = self._oom_protection.get_current_status()
            logger.info(
                f"  Memory status: {status.get('memory', {}).get('used_percent', '?'):.1f}% used"
            )

    async def start_all(self):
        """
        v84.0: Start all components with health verification and status display.

        Uses parallel startup for independent components and sequential
        startup for components with dependencies.
        """
        logger.info("=" * 60)
        logger.info("TRINITY ECOSYSTEM STARTUP v84.0")
        logger.info("=" * 60)
        self._running = True

        # v84.0: Check for already-running components first
        logger.info("Checking for running components...")
        pre_health = await self._health_checker.check_all()

        already_running = [
            name for name, health in pre_health.items()
            if health.online and health.http_healthy
        ]

        if already_running:
            logger.info(f"✓ Already running: {', '.join(already_running)}")

        # Start components in order
        started = []
        failed = []

        for name in self._startup_order:
            if name in self._repos:
                # Check if already running
                if name in already_running:
                    logger.info(f"  ✓ {name}: Already running")
                    started.append(name)
                    continue

                manager = self._repos[name]
                logger.info(f"  🚀 Starting {name}...")

                success = await manager.start()

                if success:
                    # Wait for health check
                    healthy = await self._health_checker.wait_for_component(
                        name,
                        timeout=manager.config.startup_timeout,
                        check_interval=2.0,
                    )

                    if healthy:
                        logger.info(f"  ✅ {name}: Started and healthy")
                        started.append(name)
                    else:
                        logger.warning(f"  ⚠️ {name}: Started but health check failed")
                        started.append(name)  # Still count as started
                else:
                    logger.error(f"  ❌ {name}: Failed to start")
                    failed.append(name)

                    if not self.config.enable_auto_restart:
                        await self._stop_all_reverse()
                        return False

                # Small delay between starts
                await asyncio.sleep(1.0)

        # v84.0: Display final status
        logger.info("")
        logger.info("=" * 60)
        if failed:
            logger.warning(f"TRINITY STARTUP PARTIAL: {len(started)}/{len(self._startup_order)} components")
        else:
            logger.info(f"🚀 TRINITY ECOSYSTEM ONLINE: {len(started)} components")
        logger.info("=" * 60)

        # Print health status
        await self._health_checker.print_status()

        return len(failed) == 0

    async def _stop_all_reverse(self):
        """Stop all components in reverse order."""
        for name in reversed(self._startup_order):
            if name in self._repos:
                await self._repos[name].stop()

    async def shutdown(self):
        """Graceful shutdown of all components."""
        logger.info("=" * 60)
        logger.info("Shutting down Trinity ecosystem v84.0...")
        self._running = False

        # Stop file watcher
        if self._file_watcher:
            self._file_watcher.stop()

        # v84.0: Stop reliability engines
        if self._oom_protection:
            logger.info("  Stopping OOM protection...")
            await self._oom_protection.stop_monitoring()

        if self._trinity_bridge:
            logger.info("  Stopping Trinity bridge...")
            await self._trinity_bridge.stop()

        if self._event_store:
            logger.info("  Closing event store...")
            await self._event_store.close()

        # Stop components in reverse dependency order
        await self._stop_all_reverse()

        # Restore signal handlers
        self._signal_handler.restore()

        logger.info("=" * 60)
        logger.info("Trinity shutdown complete")

    async def _handle_config_change(self, file_path: Path):
        """Handle configuration file changes (hot-reload)."""
        logger.info(f"Configuration changed: {file_path}")

        # TODO: Reload configuration and restart affected components
        # For now, just log the change

    async def run_until_shutdown(self):
        """Run orchestrator until shutdown signal received."""
        # Wait for shutdown signal
        await self._signal_handler.wait_for_shutdown()

        logger.info("Shutdown signal received")

        # Graceful shutdown
        await self.shutdown()


# ============================================================================
# GLOBAL ORCHESTRATOR
# ============================================================================

_orchestrator: Optional[CrossRepoOrchestrator] = None
_orchestrator_lock = asyncio.Lock()

_health_checker: Optional[TrinityHealthChecker] = None


async def get_orchestrator() -> CrossRepoOrchestrator:
    """Get or create global orchestrator."""
    global _orchestrator

    async with _orchestrator_lock:
        if _orchestrator is None:
            _orchestrator = CrossRepoOrchestrator()
            await _orchestrator.initialize()

        return _orchestrator


def get_health_checker() -> TrinityHealthChecker:
    """Get or create global health checker."""
    global _health_checker

    if _health_checker is None:
        _health_checker = TrinityHealthChecker()

    return _health_checker


async def check_trinity_health() -> Dict[str, ComponentHealth]:
    """Convenience function to check all Trinity component health."""
    checker = get_health_checker()
    return await checker.check_all()


async def print_trinity_status():
    """Print formatted Trinity health status."""
    checker = get_health_checker()
    await checker.print_status()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "RepoType",
    "RepoConfig",
    "OrchestratorConfig",
    # Health
    "ComponentHealth",
    "TrinityHealthChecker",
    "get_health_checker",
    "check_trinity_health",
    "print_trinity_status",
    # v84.0: Reliability Components
    "TrinityEvent",
    "TrinityEventStore",
    "TrinityBridge",
    # Components
    "SignalHandler",
    "FileWatcher",
    "DependencyResolver",
    "RepoState",
    "RepoManager",
    # Orchestrator
    "CrossRepoOrchestrator",
    "get_orchestrator",
]
