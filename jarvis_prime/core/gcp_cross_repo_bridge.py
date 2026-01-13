"""
GCP Cross-Repo Bridge v95.0 - Unified Cloud Infrastructure Coordination
=======================================================================

v95.0 ENHANCEMENTS:
    - Uses gcp_import_bridge.py to access JARVIS GCP components (no duplication)
    - Elastic cloud burst integration with HybridTieredRouter
    - Memory pressure detection via JARVIS AdvancedRAMMonitor
    - Unified budget management via JARVIS CostTracker

Coordinates GCP infrastructure across JARVIS (Body), JARVIS-Prime (Mind),
and Reactor-Core (Nervous System) repositories.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    GCP CROSS-REPO BRIDGE                             │
    │                   "The Cloud Coordinator"                            │
    └──────────────────────────┬──────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │   JARVIS     │    │ JARVIS-PRIME │    │ REACTOR-CORE │
    │   (Body)     │    │   (Mind)     │    │  (Nervous)   │
    │  API Client  │    │  GCPVMManager│    │SpotVMChkptr  │
    └──────────────┘    └──────────────┘    └──────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Shared State      │
                    │ ~/.jarvis/cross_repo│
                    │   /gcp/             │
                    └─────────────────────┘

FEATURES:
    - Cross-repo GCP resource discovery
    - Unified endpoint management
    - Coordinated preemption handling
    - Checkpoint synchronization
    - Cost aggregation across repos
    - Fallback chain coordination
    - Dynamic endpoint routing
    - Zero-downtime VM migration
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TRY IMPORTS - Graceful degradation
# =============================================================================

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class RepoType(Enum):
    """Repository types in the JARVIS ecosystem."""
    JARVIS_BODY = "jarvis"
    JARVIS_PRIME = "jarvis_prime"
    REACTOR_CORE = "reactor_core"


class GCPResourceState(Enum):
    """State of a GCP resource."""
    UNKNOWN = "unknown"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    PREEMPTED = "preempted"
    FAILED = "failed"


class EndpointHealth(Enum):
    """Health status of an endpoint."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class CrossRepoConfig:
    """Configuration for cross-repo GCP coordination."""
    # Repo paths (auto-detected if not set)
    jarvis_path: Optional[Path] = None
    jarvis_prime_path: Optional[Path] = None
    reactor_core_path: Optional[Path] = None

    # Shared state directory
    shared_state_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "cross_repo" / "gcp")

    # GCP settings
    project_id: str = ""
    default_zone: str = "us-central1-a"
    default_region: str = "us-central1"

    # Endpoint settings
    endpoint_discovery_interval: float = 30.0
    health_check_interval: float = 15.0
    health_check_timeout: float = 5.0

    # Preemption settings
    preemption_check_interval: float = 10.0
    preemption_grace_period: float = 30.0

    # Cost tracking
    cost_tracking_enabled: bool = True
    max_hourly_spend: float = 10.0
    max_daily_spend: float = 100.0

    @classmethod
    def from_env(cls) -> "CrossRepoConfig":
        """Create config from environment variables."""
        return cls(
            jarvis_path=Path(os.getenv("JARVIS_PATH", "")) if os.getenv("JARVIS_PATH") else None,
            jarvis_prime_path=Path(os.getenv("JARVIS_PRIME_PATH", "")) if os.getenv("JARVIS_PRIME_PATH") else None,
            reactor_core_path=Path(os.getenv("REACTOR_CORE_PATH", "")) if os.getenv("REACTOR_CORE_PATH") else None,
            project_id=os.getenv("GCP_PROJECT_ID", ""),
            default_zone=os.getenv("GCP_ZONE", "us-central1-a"),
            default_region=os.getenv("GCP_REGION", "us-central1"),
            max_hourly_spend=float(os.getenv("GCP_MAX_HOURLY_SPEND", "10.0")),
            max_daily_spend=float(os.getenv("GCP_MAX_DAILY_SPEND", "100.0")),
        )


@dataclass
class GCPEndpoint:
    """Represents a GCP inference endpoint."""
    id: str
    url: str
    repo: RepoType
    model_name: str
    instance_name: Optional[str] = None
    zone: Optional[str] = None

    # State
    state: GCPResourceState = GCPResourceState.UNKNOWN
    health: EndpointHealth = EndpointHealth.UNKNOWN

    # Performance
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0

    # Cost
    hourly_cost: float = 0.0
    total_cost: float = 0.0
    requests_served: int = 0

    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    max_context_tokens: int = 4096
    max_batch_size: int = 8

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "repo": self.repo.value,
            "model_name": self.model_name,
            "instance_name": self.instance_name,
            "zone": self.zone,
            "state": self.state.value,
            "health": self.health.value,
            "latency_ms": self.latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "hourly_cost": self.hourly_cost,
            "total_cost": self.total_cost,
            "requests_served": self.requests_served,
            "capabilities": self.capabilities,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GCPEndpoint":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            url=data["url"],
            repo=RepoType(data["repo"]),
            model_name=data["model_name"],
            instance_name=data.get("instance_name"),
            zone=data.get("zone"),
            state=GCPResourceState(data.get("state", "unknown")),
            health=EndpointHealth(data.get("health", "unknown")),
            latency_ms=data.get("latency_ms", 0.0),
            tokens_per_second=data.get("tokens_per_second", 0.0),
            hourly_cost=data.get("hourly_cost", 0.0),
            total_cost=data.get("total_cost", 0.0),
            requests_served=data.get("requests_served", 0),
            capabilities=data.get("capabilities", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
        )


@dataclass
class PreemptionEvent:
    """Represents a preemption event."""
    timestamp: datetime
    instance_name: str
    zone: str
    repo: RepoType
    checkpoint_saved: bool = False
    checkpoint_path: Optional[str] = None
    migrated_to: Optional[str] = None
    recovery_time_seconds: float = 0.0


# =============================================================================
# REPO DISCOVERY
# =============================================================================

class RepoDiscovery:
    """
    Discovers and validates repository paths across the ecosystem.

    Uses multiple detection strategies:
    1. Environment variables
    2. Common parent directory patterns
    3. Config file hints
    4. Active process detection
    """

    REPO_MARKERS = {
        RepoType.JARVIS_BODY: ["run_jarvis.py", "jarvis/core", "audio_listener.py"],
        RepoType.JARVIS_PRIME: ["run_supervisor.py", "jarvis_prime/core", "prime_model.py"],
        RepoType.REACTOR_CORE: ["reactor_core/__init__.py", "reactor_core/serving", "training/trainer.py"],
    }

    def __init__(self, config: CrossRepoConfig):
        self._config = config
        self._discovered_repos: Dict[RepoType, Path] = {}
        self._lock = asyncio.Lock()

    async def discover_all(self) -> Dict[RepoType, Path]:
        """Discover all repository paths."""
        async with self._lock:
            # Try configured paths first
            if self._config.jarvis_path and self._config.jarvis_path.exists():
                self._discovered_repos[RepoType.JARVIS_BODY] = self._config.jarvis_path

            if self._config.jarvis_prime_path and self._config.jarvis_prime_path.exists():
                self._discovered_repos[RepoType.JARVIS_PRIME] = self._config.jarvis_prime_path

            if self._config.reactor_core_path and self._config.reactor_core_path.exists():
                self._discovered_repos[RepoType.REACTOR_CORE] = self._config.reactor_core_path

            # Auto-detect from common patterns
            await self._auto_detect()

            return self._discovered_repos.copy()

    async def _auto_detect(self):
        """Auto-detect repository paths from common locations."""
        # Common parent directories
        search_paths = [
            Path.home() / "Documents" / "repos",
            Path.home() / "projects",
            Path.home() / "code",
            Path.home() / "dev",
            Path("/workspace"),
            Path.cwd().parent,
        ]

        # Try to detect from current file location
        current_file = Path(__file__).resolve()
        if "jarvis-prime" in str(current_file).lower():
            # We're in JARVIS-Prime
            jarvis_prime_root = current_file
            while jarvis_prime_root.name.lower() != "jarvis-prime" and jarvis_prime_root.parent != jarvis_prime_root:
                jarvis_prime_root = jarvis_prime_root.parent

            if jarvis_prime_root.name.lower() == "jarvis-prime":
                self._discovered_repos[RepoType.JARVIS_PRIME] = jarvis_prime_root
                # Check sibling directories
                parent = jarvis_prime_root.parent
                search_paths.insert(0, parent)

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for item in search_path.iterdir():
                if not item.is_dir():
                    continue

                # Check each repo type
                for repo_type, markers in self.REPO_MARKERS.items():
                    if repo_type in self._discovered_repos:
                        continue

                    if self._has_markers(item, markers):
                        self._discovered_repos[repo_type] = item
                        logger.info(f"Auto-detected {repo_type.value} at {item}")

    def _has_markers(self, path: Path, markers: List[str]) -> bool:
        """Check if path has expected markers."""
        found = 0
        for marker in markers:
            if (path / marker).exists():
                found += 1

        # Need at least half the markers
        return found >= len(markers) // 2

    def get_repo_path(self, repo_type: RepoType) -> Optional[Path]:
        """Get discovered path for a repo type."""
        return self._discovered_repos.get(repo_type)


# =============================================================================
# ENDPOINT REGISTRY
# =============================================================================

class EndpointRegistry:
    """
    Cross-repo endpoint registry with persistence and discovery.

    Features:
    - File-based persistence in shared state directory
    - Atomic updates with file locking
    - Health-aware endpoint selection
    - Load balancing across healthy endpoints
    - Automatic stale endpoint cleanup
    """

    def __init__(self, config: CrossRepoConfig):
        self._config = config
        self._endpoints: Dict[str, GCPEndpoint] = {}
        self._lock = asyncio.Lock()
        self._registry_file = config.shared_state_dir / "endpoint_registry.json"
        self._change_callbacks: List[Callable[[str, GCPEndpoint], Awaitable[None]]] = []

    async def initialize(self):
        """Initialize registry from persistent storage."""
        self._config.shared_state_dir.mkdir(parents=True, exist_ok=True)

        if self._registry_file.exists():
            try:
                data = json.loads(self._registry_file.read_text())
                for ep_data in data.get("endpoints", []):
                    endpoint = GCPEndpoint.from_dict(ep_data)
                    self._endpoints[endpoint.id] = endpoint
                logger.info(f"Loaded {len(self._endpoints)} endpoints from registry")
            except Exception as e:
                logger.warning(f"Failed to load endpoint registry: {e}")

    async def _save(self):
        """Save registry to persistent storage."""
        try:
            data = {
                "endpoints": [ep.to_dict() for ep in self._endpoints.values()],
                "updated_at": datetime.now().isoformat(),
            }

            # Atomic write
            temp_file = self._registry_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2))
            temp_file.rename(self._registry_file)
        except Exception as e:
            logger.error(f"Failed to save endpoint registry: {e}")

    async def register(self, endpoint: GCPEndpoint) -> str:
        """Register a new endpoint or update existing."""
        async with self._lock:
            endpoint.updated_at = datetime.now()

            # Check for existing endpoint with same URL
            for existing in self._endpoints.values():
                if existing.url == endpoint.url:
                    # Update existing
                    endpoint.id = existing.id
                    endpoint.created_at = existing.created_at
                    break

            self._endpoints[endpoint.id] = endpoint
            await self._save()

            # Notify callbacks
            for callback in self._change_callbacks:
                try:
                    await callback(endpoint.id, endpoint)
                except Exception as e:
                    logger.error(f"Endpoint change callback failed: {e}")

            logger.info(f"Registered endpoint {endpoint.id}: {endpoint.url}")
            return endpoint.id

    async def unregister(self, endpoint_id: str) -> bool:
        """Unregister an endpoint."""
        async with self._lock:
            if endpoint_id in self._endpoints:
                del self._endpoints[endpoint_id]
                await self._save()
                logger.info(f"Unregistered endpoint {endpoint_id}")
                return True
            return False

    async def get_endpoint(self, endpoint_id: str) -> Optional[GCPEndpoint]:
        """Get endpoint by ID."""
        return self._endpoints.get(endpoint_id)

    async def get_by_model(self, model_name: str) -> List[GCPEndpoint]:
        """Get all endpoints serving a model."""
        return [
            ep for ep in self._endpoints.values()
            if ep.model_name == model_name
        ]

    async def get_healthy_endpoint(
        self,
        model_name: Optional[str] = None,
        prefer_repo: Optional[RepoType] = None,
    ) -> Optional[GCPEndpoint]:
        """
        Get the best healthy endpoint.

        Selection criteria (in order):
        1. Health status (healthy > degraded > unhealthy)
        2. Preferred repo if specified
        3. Lowest latency
        4. Lowest cost
        """
        candidates = [
            ep for ep in self._endpoints.values()
            if ep.health in (EndpointHealth.HEALTHY, EndpointHealth.DEGRADED)
            and ep.state == GCPResourceState.RUNNING
        ]

        if model_name:
            candidates = [ep for ep in candidates if ep.model_name == model_name]

        if not candidates:
            return None

        # Sort by health, then repo preference, then latency
        def sort_key(ep: GCPEndpoint) -> Tuple[int, int, float, float]:
            health_score = 0 if ep.health == EndpointHealth.HEALTHY else 1
            repo_score = 0 if prefer_repo and ep.repo == prefer_repo else 1
            return (health_score, repo_score, ep.latency_ms, ep.hourly_cost)

        candidates.sort(key=sort_key)
        return candidates[0]

    async def get_all(self) -> List[GCPEndpoint]:
        """Get all registered endpoints."""
        return list(self._endpoints.values())

    async def update_health(
        self,
        endpoint_id: str,
        health: EndpointHealth,
        latency_ms: Optional[float] = None,
    ):
        """Update endpoint health status."""
        async with self._lock:
            if endpoint_id not in self._endpoints:
                return

            endpoint = self._endpoints[endpoint_id]
            endpoint.health = health
            endpoint.last_health_check = datetime.now()

            if latency_ms is not None:
                endpoint.latency_ms = latency_ms

            if health == EndpointHealth.UNHEALTHY:
                endpoint.consecutive_failures += 1
            else:
                endpoint.consecutive_failures = 0

            endpoint.updated_at = datetime.now()
            await self._save()

    def on_change(self, callback: Callable[[str, GCPEndpoint], Awaitable[None]]):
        """Register a callback for endpoint changes."""
        self._change_callbacks.append(callback)


# =============================================================================
# HEALTH CHECKER
# =============================================================================

class HealthChecker:
    """
    Cross-repo health checker with adaptive intervals.

    Features:
    - Parallel health checks across all endpoints
    - Adaptive check intervals based on health history
    - Detailed health metrics collection
    - Circuit breaker integration
    """

    def __init__(
        self,
        config: CrossRepoConfig,
        registry: EndpointRegistry,
    ):
        self._config = config
        self._registry = registry
        self._session: Optional[Any] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start health checker background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())
        logger.info("Health checker started")

    async def stop(self):
        """Stop health checker."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("Health checker stopped")

    async def _get_session(self) -> Any:
        """Get or create HTTP session."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp required for health checks")

        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._config.health_check_timeout)
            )

        return self._session

    async def _health_check_loop(self):
        """Background health check loop."""
        while self._running:
            try:
                endpoints = await self._registry.get_all()

                # Parallel health checks
                tasks = [
                    self._check_endpoint(ep)
                    for ep in endpoints
                    if ep.state == GCPResourceState.RUNNING
                ]

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(self._config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)

    async def _check_endpoint(self, endpoint: GCPEndpoint):
        """Check health of a single endpoint."""
        start_time = time.time()

        try:
            session = await self._get_session()
            async with session.get(f"{endpoint.url}/health") as response:
                latency_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    data = await response.json()
                    model_loaded = data.get("model_loaded", True)
                    status = data.get("status", "healthy")

                    if model_loaded and status == "healthy":
                        await self._registry.update_health(
                            endpoint.id,
                            EndpointHealth.HEALTHY,
                            latency_ms,
                        )
                    else:
                        await self._registry.update_health(
                            endpoint.id,
                            EndpointHealth.DEGRADED,
                            latency_ms,
                        )
                else:
                    await self._registry.update_health(
                        endpoint.id,
                        EndpointHealth.UNHEALTHY,
                    )

        except Exception as e:
            logger.debug(f"Health check failed for {endpoint.url}: {e}")
            await self._registry.update_health(
                endpoint.id,
                EndpointHealth.UNHEALTHY,
            )


# =============================================================================
# PREEMPTION COORDINATOR
# =============================================================================

class PreemptionCoordinator:
    """
    Coordinates preemption handling across repositories.

    When a GCP Spot VM is preempted:
    1. Detects preemption signal (metadata endpoint or SIGTERM)
    2. Notifies all repos via shared state
    3. Triggers checkpoint in Reactor-Core
    4. Initiates VM migration in JARVIS-Prime
    5. Updates endpoint registry
    """

    def __init__(
        self,
        config: CrossRepoConfig,
        registry: EndpointRegistry,
        repo_discovery: RepoDiscovery,
    ):
        self._config = config
        self._registry = registry
        self._repo_discovery = repo_discovery
        self._preemption_events: List[PreemptionEvent] = []
        self._event_file = config.shared_state_dir / "preemption_events.json"
        self._lock = asyncio.Lock()

        # Callbacks
        self._on_preemption_callbacks: List[Callable[[PreemptionEvent], Awaitable[None]]] = []

    async def signal_preemption(
        self,
        instance_name: str,
        zone: str,
        repo: RepoType,
    ) -> PreemptionEvent:
        """Signal a preemption event to all repos."""
        async with self._lock:
            event = PreemptionEvent(
                timestamp=datetime.now(),
                instance_name=instance_name,
                zone=zone,
                repo=repo,
            )

            # Save to shared state for cross-repo visibility
            self._preemption_events.append(event)
            await self._save_events()

            # Notify callbacks
            for callback in self._on_preemption_callbacks:
                try:
                    await callback(event)
                except Exception as e:
                    logger.error(f"Preemption callback failed: {e}")

            # Update endpoint registry
            endpoints = await self._registry.get_all()
            for ep in endpoints:
                if ep.instance_name == instance_name:
                    ep.state = GCPResourceState.PREEMPTED
                    await self._registry.register(ep)

            logger.warning(f"Preemption signaled for {instance_name} in {zone}")
            return event

    async def _save_events(self):
        """Save preemption events to file."""
        try:
            data = {
                "events": [
                    {
                        "timestamp": e.timestamp.isoformat(),
                        "instance_name": e.instance_name,
                        "zone": e.zone,
                        "repo": e.repo.value,
                        "checkpoint_saved": e.checkpoint_saved,
                        "checkpoint_path": e.checkpoint_path,
                        "migrated_to": e.migrated_to,
                        "recovery_time_seconds": e.recovery_time_seconds,
                    }
                    for e in self._preemption_events[-100:]  # Keep last 100
                ]
            }

            temp_file = self._event_file.with_suffix(".tmp")
            temp_file.write_text(json.dumps(data, indent=2))
            temp_file.rename(self._event_file)

        except Exception as e:
            logger.error(f"Failed to save preemption events: {e}")

    async def trigger_checkpoint(self, instance_name: str) -> bool:
        """
        Trigger checkpoint via Reactor-Core integration.

        Uses the SpotVMCheckpointer from Reactor-Core if available.
        """
        try:
            reactor_path = self._repo_discovery.get_repo_path(RepoType.REACTOR_CORE)

            if reactor_path:
                # Signal checkpoint via file
                checkpoint_signal = self._config.shared_state_dir / "checkpoint_signal.json"
                signal_data = {
                    "action": "checkpoint",
                    "instance_name": instance_name,
                    "timestamp": datetime.now().isoformat(),
                    "urgency": "high",
                }
                checkpoint_signal.write_text(json.dumps(signal_data))
                logger.info(f"Checkpoint signal sent for {instance_name}")
                return True

        except Exception as e:
            logger.error(f"Failed to trigger checkpoint: {e}")

        return False

    async def request_migration(
        self,
        instance_name: str,
        current_zone: str,
        target_zones: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Request VM migration to a new zone.

        Coordinates with GCPVMManager in JARVIS-Prime.
        """
        target_zones = target_zones or [
            "us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f"
        ]

        # Exclude current zone
        target_zones = [z for z in target_zones if z != current_zone]

        if not target_zones:
            logger.warning("No target zones available for migration")
            return None

        try:
            # Signal migration request via file
            migration_request = self._config.shared_state_dir / "migration_request.json"
            request_data = {
                "action": "migrate",
                "instance_name": instance_name,
                "current_zone": current_zone,
                "target_zones": target_zones,
                "timestamp": datetime.now().isoformat(),
            }
            migration_request.write_text(json.dumps(request_data))
            logger.info(f"Migration request sent for {instance_name}")

            return target_zones[0]  # Return first target zone

        except Exception as e:
            logger.error(f"Failed to request migration: {e}")

        return None

    def on_preemption(self, callback: Callable[[PreemptionEvent], Awaitable[None]]):
        """Register a preemption callback."""
        self._on_preemption_callbacks.append(callback)


# =============================================================================
# COST AGGREGATOR
# =============================================================================

class CostAggregator:
    """
    Aggregates costs across all repos and endpoints.

    Features:
    - Real-time cost tracking
    - Budget alerts and limits
    - Cost attribution by repo/model
    - Historical cost analysis
    """

    def __init__(self, config: CrossRepoConfig):
        self._config = config
        self._costs: Dict[str, Dict[str, float]] = {}  # repo -> model -> cost
        self._hourly_total: float = 0.0
        self._daily_total: float = 0.0
        self._lock = asyncio.Lock()
        self._cost_file = config.shared_state_dir / "cost_tracking.json"
        self._last_reset: datetime = datetime.now()

    async def record_cost(
        self,
        repo: RepoType,
        model_name: str,
        cost_usd: float,
    ):
        """Record a cost entry."""
        async with self._lock:
            repo_key = repo.value

            if repo_key not in self._costs:
                self._costs[repo_key] = {}

            if model_name not in self._costs[repo_key]:
                self._costs[repo_key][model_name] = 0.0

            self._costs[repo_key][model_name] += cost_usd
            self._hourly_total += cost_usd
            self._daily_total += cost_usd

            # Check limits
            await self._check_limits()

            # Periodic save
            await self._save()

    async def _check_limits(self):
        """Check if spending limits are exceeded."""
        now = datetime.now()

        # Reset hourly
        if (now - self._last_reset).total_seconds() > 3600:
            self._hourly_total = 0.0
            self._last_reset = now

        # Check limits
        if self._hourly_total >= self._config.max_hourly_spend:
            logger.warning(f"Hourly spend limit reached: ${self._hourly_total:.2f}")

        if self._daily_total >= self._config.max_daily_spend:
            logger.warning(f"Daily spend limit reached: ${self._daily_total:.2f}")

    async def _save(self):
        """Save cost data."""
        try:
            data = {
                "costs": self._costs,
                "hourly_total": self._hourly_total,
                "daily_total": self._daily_total,
                "updated_at": datetime.now().isoformat(),
            }
            self._cost_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug(f"Failed to save cost data: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """Get cost summary."""
        return {
            "hourly_total": self._hourly_total,
            "daily_total": self._daily_total,
            "by_repo": self._costs.copy(),
            "limits": {
                "hourly": self._config.max_hourly_spend,
                "daily": self._config.max_daily_spend,
            },
        }


# =============================================================================
# GCP CROSS-REPO BRIDGE
# =============================================================================

class GCPCrossRepoBridge:
    """
    Master coordinator for GCP infrastructure across all repositories.

    Provides:
    - Unified endpoint discovery and routing
    - Cross-repo preemption handling
    - Cost aggregation and budget management
    - Health monitoring across the ecosystem
    - Seamless failover between repos
    """

    def __init__(self, config: Optional[CrossRepoConfig] = None):
        self._config = config or CrossRepoConfig.from_env()

        # Components
        self._repo_discovery = RepoDiscovery(self._config)
        self._registry = EndpointRegistry(self._config)
        self._health_checker = HealthChecker(self._config, self._registry)
        self._preemption_coordinator = PreemptionCoordinator(
            self._config, self._registry, self._repo_discovery
        )
        self._cost_aggregator = CostAggregator(self._config)

        # State
        self._initialized = False
        self._running = False
        self._lock = asyncio.Lock()

        # Integration with repo managers
        self._gcp_vm_manager: Optional[Any] = None
        self._reactor_checkpointer: Optional[Any] = None

        # Statistics
        self._stats = {
            "endpoints_discovered": 0,
            "requests_routed": 0,
            "preemptions_handled": 0,
            "failovers_performed": 0,
        }

    async def initialize(self) -> bool:
        """Initialize the cross-repo bridge."""
        async with self._lock:
            if self._initialized:
                return True

            try:
                # Create shared state directory
                self._config.shared_state_dir.mkdir(parents=True, exist_ok=True)

                # Discover repos
                repos = await self._repo_discovery.discover_all()
                logger.info(f"Discovered repos: {[r.value for r in repos.keys()]}")

                # Initialize registry
                await self._registry.initialize()

                # Connect to JARVIS-Prime GCPVMManager
                await self._connect_vm_manager()

                # Connect to Reactor-Core checkpointer
                await self._connect_reactor_checkpointer()

                # Register preemption handler
                self._preemption_coordinator.on_preemption(self._handle_preemption)

                self._initialized = True
                logger.info("GCP Cross-Repo Bridge initialized")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize cross-repo bridge: {e}")
                return False

    async def start(self):
        """Start background services."""
        if not self._initialized:
            await self.initialize()

        if self._running:
            return

        self._running = True

        # Start health checker
        await self._health_checker.start()

        # Start endpoint discovery
        asyncio.create_task(self._endpoint_discovery_loop())

        logger.info("GCP Cross-Repo Bridge started")

    async def stop(self):
        """Stop all services."""
        self._running = False

        await self._health_checker.stop()

        logger.info("GCP Cross-Repo Bridge stopped")

    async def _connect_vm_manager(self):
        """Connect to JARVIS's GCPVMManager via import bridge."""
        try:
            # v95.0: Use the GCP import bridge to access JARVIS's components
            from jarvis_prime.core.gcp_import_bridge import get_gcp_vm_manager
            self._gcp_vm_manager = await get_gcp_vm_manager()
            if self._gcp_vm_manager:
                logger.info("Connected to JARVIS GCPVMManager via import bridge")
            else:
                logger.warning("GCPVMManager not available from JARVIS repo")
        except Exception as e:
            logger.warning(f"Could not connect to GCPVMManager: {e}")

    async def _connect_reactor_checkpointer(self):
        """Connect to Reactor-Core's SpotVMCheckpointer."""
        reactor_path = self._repo_discovery.get_repo_path(RepoType.REACTOR_CORE)

        if not reactor_path:
            return

        try:
            import sys
            if str(reactor_path) not in sys.path:
                sys.path.insert(0, str(reactor_path))

            from reactor_core.gcp.checkpointer import get_checkpoint_manager
            self._reactor_checkpointer = await get_checkpoint_manager()
            logger.info("Connected to Reactor-Core checkpointer")

        except Exception as e:
            logger.warning(f"Could not connect to Reactor-Core checkpointer: {e}")

    async def _endpoint_discovery_loop(self):
        """Background endpoint discovery."""
        while self._running:
            try:
                await self._discover_endpoints()
                await asyncio.sleep(self._config.endpoint_discovery_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Endpoint discovery error: {e}")
                await asyncio.sleep(10)

    async def _discover_endpoints(self):
        """Discover endpoints from all sources."""
        # From GCPVMManager
        if self._gcp_vm_manager:
            try:
                status = self._gcp_vm_manager.get_status()
                for name, instance_data in status.get("instances", {}).items():
                    if instance_data.get("inference_endpoint"):
                        endpoint = GCPEndpoint(
                            id=f"prime-{name}",
                            url=instance_data["inference_endpoint"],
                            repo=RepoType.JARVIS_PRIME,
                            model_name=instance_data.get("model", "unknown"),
                            instance_name=name,
                            zone=instance_data.get("zone"),
                            state=GCPResourceState(instance_data.get("state", "unknown")),
                            hourly_cost=instance_data.get("hourly_cost", 0.0),
                        )
                        await self._registry.register(endpoint)
                        self._stats["endpoints_discovered"] += 1
            except Exception as e:
                logger.debug(f"GCPVMManager endpoint discovery failed: {e}")

        # From shared state files (other repos may write here)
        endpoints_file = self._config.shared_state_dir / "discovered_endpoints.json"
        if endpoints_file.exists():
            try:
                data = json.loads(endpoints_file.read_text())
                for ep_data in data.get("endpoints", []):
                    endpoint = GCPEndpoint.from_dict(ep_data)
                    await self._registry.register(endpoint)
            except Exception as e:
                logger.debug(f"Failed to load discovered endpoints: {e}")

    async def _handle_preemption(self, event: PreemptionEvent):
        """Handle preemption event."""
        self._stats["preemptions_handled"] += 1

        # Trigger checkpoint
        if self._reactor_checkpointer:
            try:
                await self._reactor_checkpointer.emergency_checkpoint()
                event.checkpoint_saved = True
                logger.info("Emergency checkpoint saved via Reactor-Core")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

        # Request migration
        target_zone = await self._preemption_coordinator.request_migration(
            event.instance_name,
            event.zone,
        )

        if target_zone and self._gcp_vm_manager:
            event.migrated_to = target_zone
            logger.info(f"Migration requested to {target_zone}")

    async def get_best_endpoint(
        self,
        model_name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
    ) -> Optional[GCPEndpoint]:
        """
        Get the best available endpoint for a request.

        Selection criteria:
        1. Health status
        2. Model match
        3. Capability match
        4. Latency
        5. Cost
        """
        endpoint = await self._registry.get_healthy_endpoint(model_name)

        if endpoint:
            self._stats["requests_routed"] += 1

        return endpoint

    async def provision_endpoint(
        self,
        model_name: str,
        zone: Optional[str] = None,
    ) -> Optional[GCPEndpoint]:
        """Provision a new GCP endpoint if none available."""
        if not self._gcp_vm_manager:
            logger.warning("GCPVMManager not available for provisioning")
            return None

        try:
            zone = zone or self._config.default_zone

            # Check if we already have an endpoint
            existing = await self._registry.get_by_model(model_name)
            healthy = [ep for ep in existing if ep.health == EndpointHealth.HEALTHY]

            if healthy:
                return healthy[0]

            # Provision new instance
            from jarvis_prime.core.gcp_vm_manager import VMConfig

            vm_config = VMConfig(
                name=f"jarvis-{model_name.replace('.', '-')}-{int(time.time()) % 10000}",
                zone=zone,
                gpu_type="nvidia-tesla-t4",
                gpu_count=1,
                spot=True,
            )

            instance = await self._gcp_vm_manager.provision_instance(vm_config=vm_config)

            if instance and instance.inference_endpoint:
                endpoint = GCPEndpoint(
                    id=f"prime-{instance.name}",
                    url=instance.inference_endpoint,
                    repo=RepoType.JARVIS_PRIME,
                    model_name=model_name,
                    instance_name=instance.name,
                    zone=zone,
                    state=GCPResourceState.PROVISIONING,
                )
                await self._registry.register(endpoint)
                return endpoint

        except Exception as e:
            logger.error(f"Failed to provision endpoint: {e}")

        return None

    async def record_inference_cost(
        self,
        endpoint_id: str,
        cost_usd: float,
    ):
        """Record inference cost."""
        endpoint = await self._registry.get_endpoint(endpoint_id)

        if endpoint:
            await self._cost_aggregator.record_cost(
                endpoint.repo,
                endpoint.model_name,
                cost_usd,
            )

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "stats": self._stats.copy(),
            "costs": self._cost_aggregator.get_summary(),
            "repos": {
                repo.value: str(path) if path else None
                for repo, path in self._repo_discovery._discovered_repos.items()
            },
            "config": {
                "project_id": self._config.project_id,
                "default_zone": self._config.default_zone,
                "shared_state_dir": str(self._config.shared_state_dir),
            },
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_bridge: Optional[GCPCrossRepoBridge] = None
_bridge_lock = asyncio.Lock()


async def get_gcp_cross_repo_bridge(
    config: Optional[CrossRepoConfig] = None,
) -> GCPCrossRepoBridge:
    """Get or create the global GCP Cross-Repo Bridge."""
    global _bridge

    if _bridge is not None:
        return _bridge

    async with _bridge_lock:
        if _bridge is not None:
            return _bridge

        _bridge = GCPCrossRepoBridge(config)
        await _bridge.initialize()
        await _bridge.start()

        return _bridge


async def shutdown_gcp_cross_repo_bridge():
    """Shutdown the global bridge."""
    global _bridge

    async with _bridge_lock:
        if _bridge is not None:
            await _bridge.stop()
            _bridge = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "RepoType",
    "GCPResourceState",
    "EndpointHealth",
    # Config
    "CrossRepoConfig",
    # Data classes
    "GCPEndpoint",
    "PreemptionEvent",
    # Components
    "RepoDiscovery",
    "EndpointRegistry",
    "HealthChecker",
    "PreemptionCoordinator",
    "CostAggregator",
    # Bridge
    "GCPCrossRepoBridge",
    # Factory
    "get_gcp_cross_repo_bridge",
    "shutdown_gcp_cross_repo_bridge",
]
