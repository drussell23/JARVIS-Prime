"""
Service Mesh v1.0 - Dynamic Service Discovery & Mesh Networking
================================================================

Provides service mesh capabilities for the Trinity ecosystem:
- Dynamic service registration and discovery
- Health checking with circuit breakers
- Load balancing across endpoints
- Connection pooling
- Request routing and retries

ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────┐
    │                       SERVICE MESH                              │
    └──────────────────────────┬─────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │   Service    │    │   Health     │    │    Load      │
    │   Registry   │    │   Checker    │    │   Balancer   │
    │              │    │              │    │              │
    └──────────────┘    └──────────────┘    └──────────────┘
         │                     │                     │
         └─────────────────────┼─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Connection Pool   │
                    │   (Per-Service)     │
                    └─────────────────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌────────┐              ┌────────────┐              ┌──────────┐
│ JARVIS │              │JARVIS-Prime│              │Reactor   │
│ (Body) │              │  (Mind)    │              │Core      │
└────────┘              └────────────┘              └──────────┘

FEATURES:
    - File-based and memory registry backends
    - Heartbeat-based liveness detection
    - Multiple load balancing strategies
    - Circuit breaker per endpoint
    - Retry with exponential backoff
    - Request tracing and metrics
    - Zero configuration (auto-discovery)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
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

# =============================================================================
# TRY IMPORTS
# =============================================================================

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


# =============================================================================
# CONFIGURATION
# =============================================================================

class ServiceStatus(Enum):
    """Service health status."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    WEIGHTED = "weighted"
    CONSISTENT_HASH = "consistent_hash"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject calls
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class ServiceEndpoint:
    """Represents a service endpoint."""
    service_name: str
    instance_id: str
    host: str
    port: int
    protocol: str = "http"

    # Metadata
    version: str = "1.0.0"
    labels: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)

    # Health
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    # Load
    active_connections: int = 0
    total_requests: int = 0
    weight: float = 1.0

    # Timing
    registered_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    avg_response_time_ms: float = 0.0

    @property
    def url(self) -> str:
        """Get endpoint URL."""
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        """Get health check URL."""
        return f"{self.url}/health"

    @property
    def is_healthy(self) -> bool:
        """Check if endpoint is healthy."""
        return self.status in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED)

    @property
    def is_stale(self) -> bool:
        """Check if endpoint is stale (no heartbeat)."""
        if not self.last_seen:
            return True
        stale_threshold = timedelta(seconds=30)
        return datetime.now() - self.last_seen > stale_threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_name": self.service_name,
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "version": self.version,
            "labels": self.labels,
            "capabilities": self.capabilities,
            "status": self.status.value,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "active_connections": self.active_connections,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceEndpoint":
        return cls(
            service_name=data["service_name"],
            instance_id=data["instance_id"],
            host=data["host"],
            port=data["port"],
            protocol=data.get("protocol", "http"),
            version=data.get("version", "1.0.0"),
            labels=data.get("labels", {}),
            capabilities=data.get("capabilities", []),
            status=ServiceStatus(data.get("status", "unknown")),
            last_seen=datetime.fromisoformat(data["last_seen"]) if data.get("last_seen") else None,
            active_connections=data.get("active_connections", 0),
            weight=data.get("weight", 1.0),
        )


@dataclass
class ServiceMeshConfig:
    """Configuration for service mesh."""
    # Registry
    registry_type: str = "file"  # file, memory
    registry_path: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "trinity" / "service_registry.json")
    refresh_interval_seconds: float = 5.0
    stale_threshold_seconds: float = 30.0

    # Health checking
    health_check_interval_seconds: float = 10.0
    health_check_timeout_seconds: float = 5.0
    failure_threshold: int = 3
    success_threshold: int = 2

    # Circuit breaker
    circuit_enabled: bool = True
    circuit_failure_threshold: int = 5
    circuit_success_threshold: int = 2
    circuit_timeout_seconds: float = 30.0
    circuit_half_open_max_calls: int = 3

    # Load balancing
    lb_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    sticky_sessions: bool = False

    # Connection pool
    pool_enabled: bool = True
    pool_max_connections_per_host: int = 10
    pool_max_keepalive_connections: int = 5
    pool_keepalive_expiry_seconds: float = 300.0
    pool_connection_timeout_seconds: float = 5.0
    pool_read_timeout_seconds: float = 60.0

    # Retry
    retry_max_attempts: int = 3
    retry_initial_delay_seconds: float = 0.5
    retry_max_delay_seconds: float = 10.0
    retry_exponential_base: float = 2.0
    retry_jitter_factor: float = 0.1

    @classmethod
    def from_env(cls) -> "ServiceMeshConfig":
        """Create config from environment variables."""
        return cls(
            registry_type=os.getenv("SERVICE_MESH_REGISTRY", "file"),
            registry_path=Path(os.getenv(
                "SERVICE_MESH_REGISTRY_PATH",
                str(Path.home() / ".jarvis" / "trinity" / "service_registry.json")
            )),
            health_check_interval_seconds=float(os.getenv("SERVICE_MESH_HEALTH_INTERVAL", "10")),
            lb_strategy=LoadBalancingStrategy(os.getenv("SERVICE_MESH_LB_STRATEGY", "round_robin")),
        )


# =============================================================================
# SERVICE REGISTRY
# =============================================================================

class ServiceRegistry(ABC):
    """Abstract service registry interface."""

    @abstractmethod
    async def register(self, endpoint: ServiceEndpoint) -> bool:
        """Register a service endpoint."""
        ...

    @abstractmethod
    async def deregister(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service endpoint."""
        ...

    @abstractmethod
    async def get_endpoints(self, service_name: str) -> List[ServiceEndpoint]:
        """Get all endpoints for a service."""
        ...

    @abstractmethod
    async def get_all_services(self) -> Dict[str, List[ServiceEndpoint]]:
        """Get all registered services."""
        ...

    @abstractmethod
    async def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Send heartbeat for an endpoint."""
        ...

    @abstractmethod
    async def update_status(
        self,
        service_name: str,
        instance_id: str,
        status: ServiceStatus,
    ) -> bool:
        """Update endpoint status."""
        ...


class MemoryServiceRegistry(ServiceRegistry):
    """In-memory service registry."""

    def __init__(self):
        self._services: Dict[str, Dict[str, ServiceEndpoint]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def register(self, endpoint: ServiceEndpoint) -> bool:
        async with self._lock:
            self._services[endpoint.service_name][endpoint.instance_id] = endpoint
            logger.info(f"Registered {endpoint.service_name}/{endpoint.instance_id}")
            return True

    async def deregister(self, service_name: str, instance_id: str) -> bool:
        async with self._lock:
            if service_name in self._services and instance_id in self._services[service_name]:
                del self._services[service_name][instance_id]
                logger.info(f"Deregistered {service_name}/{instance_id}")
                return True
            return False

    async def get_endpoints(self, service_name: str) -> List[ServiceEndpoint]:
        async with self._lock:
            return list(self._services.get(service_name, {}).values())

    async def get_all_services(self) -> Dict[str, List[ServiceEndpoint]]:
        async with self._lock:
            return {
                name: list(endpoints.values())
                for name, endpoints in self._services.items()
            }

    async def heartbeat(self, service_name: str, instance_id: str) -> bool:
        async with self._lock:
            if service_name in self._services and instance_id in self._services[service_name]:
                self._services[service_name][instance_id].last_seen = datetime.now()
                return True
            return False

    async def update_status(
        self,
        service_name: str,
        instance_id: str,
        status: ServiceStatus,
    ) -> bool:
        async with self._lock:
            if service_name in self._services and instance_id in self._services[service_name]:
                self._services[service_name][instance_id].status = status
                return True
            return False


class FileServiceRegistry(ServiceRegistry):
    """File-based service registry with atomic writes."""

    def __init__(self, registry_path: Path):
        self._registry_path = registry_path
        self._lock = asyncio.Lock()
        self._cache: Optional[Dict[str, List[ServiceEndpoint]]] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 1.0

        # Ensure directory exists
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)

    async def _load(self) -> Dict[str, Dict[str, ServiceEndpoint]]:
        """Load registry from file."""
        now = time.time()

        # Use cache if fresh
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return {
                name: {ep.instance_id: ep for ep in endpoints}
                for name, endpoints in self._cache.items()
            }

        if not self._registry_path.exists():
            return {}

        try:
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(self._registry_path, 'r') as f:
                    content = await f.read()
                    data = json.loads(content)
            else:
                with open(self._registry_path, 'r') as f:
                    data = json.load(f)

            result: Dict[str, Dict[str, ServiceEndpoint]] = {}
            for service_name, endpoints in data.items():
                result[service_name] = {}
                for ep_data in endpoints:
                    ep = ServiceEndpoint.from_dict(ep_data)
                    result[service_name][ep.instance_id] = ep

            # Update cache
            self._cache = {
                name: list(endpoints.values())
                for name, endpoints in result.items()
            }
            self._cache_time = now

            return result

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return {}

    async def _save(self, services: Dict[str, Dict[str, ServiceEndpoint]]):
        """Save registry to file with atomic write."""
        try:
            data = {
                name: [ep.to_dict() for ep in endpoints.values()]
                for name, endpoints in services.items()
            }

            # Write to temp file first
            temp_path = self._registry_path.with_suffix('.tmp')

            if AIOFILES_AVAILABLE:
                async with aiofiles.open(temp_path, 'w') as f:
                    await f.write(json.dumps(data, indent=2))
            else:
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)

            # Atomic rename
            temp_path.rename(self._registry_path)

            # Invalidate cache
            self._cache = None

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise

    async def register(self, endpoint: ServiceEndpoint) -> bool:
        async with self._lock:
            services = await self._load()
            if endpoint.service_name not in services:
                services[endpoint.service_name] = {}
            services[endpoint.service_name][endpoint.instance_id] = endpoint
            await self._save(services)
            logger.info(f"Registered {endpoint.service_name}/{endpoint.instance_id}")
            return True

    async def deregister(self, service_name: str, instance_id: str) -> bool:
        async with self._lock:
            services = await self._load()
            if service_name in services and instance_id in services[service_name]:
                del services[service_name][instance_id]
                await self._save(services)
                logger.info(f"Deregistered {service_name}/{instance_id}")
                return True
            return False

    async def get_endpoints(self, service_name: str) -> List[ServiceEndpoint]:
        async with self._lock:
            services = await self._load()
            return list(services.get(service_name, {}).values())

    async def get_all_services(self) -> Dict[str, List[ServiceEndpoint]]:
        async with self._lock:
            services = await self._load()
            return {
                name: list(endpoints.values())
                for name, endpoints in services.items()
            }

    async def heartbeat(self, service_name: str, instance_id: str) -> bool:
        async with self._lock:
            services = await self._load()
            if service_name in services and instance_id in services[service_name]:
                services[service_name][instance_id].last_seen = datetime.now()
                await self._save(services)
                return True
            return False

    async def update_status(
        self,
        service_name: str,
        instance_id: str,
        status: ServiceStatus,
    ) -> bool:
        async with self._lock:
            services = await self._load()
            if service_name in services and instance_id in services[service_name]:
                services[service_name][instance_id].status = status
                await self._save(services)
                return True
            return False


# =============================================================================
# HEALTH CHECKER
# =============================================================================

class HealthChecker:
    """
    Health checking with configurable thresholds.

    Features:
        - HTTP health endpoint checking
        - Configurable failure/success thresholds
        - Status transitions with hysteresis
        - Parallel health checks
    """

    def __init__(
        self,
        config: ServiceMeshConfig,
        registry: ServiceRegistry,
    ):
        self._config = config
        self._registry = registry
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Start health checking."""
        if self._running:
            return

        self._running = True

        if AIOHTTP_AVAILABLE:
            timeout = aiohttp.ClientTimeout(total=self._config.health_check_timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

        self._check_task = asyncio.create_task(self._check_loop())
        logger.info("Health checker started")

    async def stop(self):
        """Stop health checking."""
        self._running = False

        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Health checker stopped")

    async def _check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                services = await self._registry.get_all_services()

                # Check all endpoints in parallel
                tasks = []
                for service_name, endpoints in services.items():
                    for endpoint in endpoints:
                        tasks.append(self._check_endpoint(endpoint))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                await asyncio.sleep(self._config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)

    async def _check_endpoint(self, endpoint: ServiceEndpoint):
        """Check health of a single endpoint."""
        if not self._session:
            return

        try:
            async with self._session.get(endpoint.health_url) as response:
                if response.status == 200:
                    endpoint.consecutive_successes += 1
                    endpoint.consecutive_failures = 0

                    if endpoint.consecutive_successes >= self._config.success_threshold:
                        if endpoint.status != ServiceStatus.HEALTHY:
                            logger.info(f"Endpoint {endpoint.url} is now HEALTHY")
                        endpoint.status = ServiceStatus.HEALTHY

                else:
                    await self._record_failure(endpoint, f"HTTP {response.status}")

        except asyncio.TimeoutError:
            await self._record_failure(endpoint, "timeout")
        except Exception as e:
            await self._record_failure(endpoint, str(e))

        endpoint.last_health_check = datetime.now()

        # Update registry
        await self._registry.update_status(
            endpoint.service_name,
            endpoint.instance_id,
            endpoint.status,
        )

    async def _record_failure(self, endpoint: ServiceEndpoint, reason: str):
        """Record a health check failure."""
        endpoint.consecutive_failures += 1
        endpoint.consecutive_successes = 0

        if endpoint.consecutive_failures >= self._config.failure_threshold:
            if endpoint.status != ServiceStatus.UNHEALTHY:
                logger.warning(f"Endpoint {endpoint.url} is now UNHEALTHY: {reason}")
            endpoint.status = ServiceStatus.UNHEALTHY
        elif endpoint.consecutive_failures >= self._config.failure_threshold // 2:
            endpoint.status = ServiceStatus.DEGRADED


# =============================================================================
# LOAD BALANCER
# =============================================================================

class LoadBalancer:
    """
    Load balancing across service endpoints.

    Strategies:
        - Round Robin: Rotate through endpoints
        - Least Connections: Choose endpoint with fewest active connections
        - Random: Random selection
        - Weighted: Weight-based selection
        - Consistent Hash: Hash-based routing for sticky sessions
    """

    def __init__(self, config: ServiceMeshConfig):
        self._config = config
        self._round_robin_index: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def select(
        self,
        endpoints: List[ServiceEndpoint],
        key: Optional[str] = None,
    ) -> Optional[ServiceEndpoint]:
        """Select an endpoint using configured strategy."""
        # Filter healthy endpoints
        healthy = [ep for ep in endpoints if ep.is_healthy and not ep.is_stale]

        if not healthy:
            return None

        strategy = self._config.lb_strategy

        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin(healthy)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections(healthy)
        elif strategy == LoadBalancingStrategy.RANDOM:
            return self._random(healthy)
        elif strategy == LoadBalancingStrategy.WEIGHTED:
            return self._weighted(healthy)
        elif strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash(healthy, key)
        else:
            return healthy[0]

    async def _round_robin(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round robin selection."""
        async with self._lock:
            service_name = endpoints[0].service_name
            index = self._round_robin_index[service_name]
            endpoint = endpoints[index % len(endpoints)]
            self._round_robin_index[service_name] = index + 1
            return endpoint

    async def _least_connections(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint with least active connections."""
        return min(endpoints, key=lambda ep: ep.active_connections)

    def _random(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Random selection."""
        return random.choice(endpoints)

    def _weighted(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted random selection."""
        total_weight = sum(ep.weight for ep in endpoints)
        r = random.uniform(0, total_weight)
        cumulative = 0
        for ep in endpoints:
            cumulative += ep.weight
            if r <= cumulative:
                return ep
        return endpoints[-1]

    def _consistent_hash(
        self,
        endpoints: List[ServiceEndpoint],
        key: Optional[str],
    ) -> ServiceEndpoint:
        """Consistent hash selection for sticky sessions."""
        if not key:
            return self._random(endpoints)

        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return endpoints[hash_value % len(endpoints)]


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class EndpointCircuitBreaker:
    """
    Circuit breaker for a single endpoint.

    States:
        - CLOSED: Normal operation, requests pass through
        - OPEN: Failing, requests rejected immediately
        - HALF_OPEN: Testing recovery, limited requests allowed
    """

    def __init__(self, endpoint: ServiceEndpoint, config: ServiceMeshConfig):
        self._endpoint = endpoint
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

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
                    if elapsed >= self._config.circuit_timeout_seconds:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        logger.info(f"Circuit half-open for {self._endpoint.url}")
                        return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._config.circuit_half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self):
        """Record successful request."""
        async with self._lock:
            self._success_count += 1
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self._config.circuit_success_threshold:
                    self._state = CircuitState.CLOSED
                    self._success_count = 0
                    logger.info(f"Circuit closed for {self._endpoint.url}")

    async def record_failure(self):
        """Record failed request."""
        async with self._lock:
            self._failure_count += 1
            self._success_count = 0
            self._last_failure_time = time.time()

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.circuit_failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit opened for {self._endpoint.url}")

            elif self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit re-opened for {self._endpoint.url}")

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
        }


class CircuitBreakerManager:
    """Manages circuit breakers for all endpoints."""

    def __init__(self, config: ServiceMeshConfig):
        self._config = config
        self._breakers: Dict[str, EndpointCircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def _endpoint_key(self, endpoint: ServiceEndpoint) -> str:
        return f"{endpoint.service_name}:{endpoint.instance_id}"

    async def get_breaker(self, endpoint: ServiceEndpoint) -> EndpointCircuitBreaker:
        """Get or create circuit breaker for endpoint."""
        key = self._endpoint_key(endpoint)

        async with self._lock:
            if key not in self._breakers:
                self._breakers[key] = EndpointCircuitBreaker(endpoint, self._config)
            return self._breakers[key]

    async def allow_request(self, endpoint: ServiceEndpoint) -> bool:
        """Check if request to endpoint should be allowed."""
        if not self._config.circuit_enabled:
            return True
        breaker = await self.get_breaker(endpoint)
        return await breaker.allow_request()

    async def record_success(self, endpoint: ServiceEndpoint):
        """Record successful request."""
        breaker = await self.get_breaker(endpoint)
        await breaker.record_success()

    async def record_failure(self, endpoint: ServiceEndpoint):
        """Record failed request."""
        breaker = await self.get_breaker(endpoint)
        await breaker.record_failure()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all circuit breaker states."""
        return {key: breaker.get_state() for key, breaker in self._breakers.items()}


# =============================================================================
# CONNECTION POOL
# =============================================================================

class ConnectionPool:
    """
    HTTP connection pool with per-host limits.

    Features:
        - Connection reuse
        - Keep-alive support
        - Connection limits per host
        - Automatic cleanup
    """

    def __init__(self, config: ServiceMeshConfig):
        self._config = config
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._lock = asyncio.Lock()

    async def get_session(self, host: str) -> aiohttp.ClientSession:
        """Get or create session for host."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")

        async with self._lock:
            if host not in self._sessions or self._sessions[host].closed:
                connector = aiohttp.TCPConnector(
                    limit=self._config.pool_max_connections_per_host,
                    limit_per_host=self._config.pool_max_connections_per_host,
                    ttl_dns_cache=300,
                    keepalive_timeout=self._config.pool_keepalive_expiry_seconds,
                )
                timeout = aiohttp.ClientTimeout(
                    total=self._config.pool_read_timeout_seconds,
                    connect=self._config.pool_connection_timeout_seconds,
                )
                self._sessions[host] = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                )
            return self._sessions[host]

    async def close_all(self):
        """Close all sessions."""
        async with self._lock:
            for session in self._sessions.values():
                if not session.closed:
                    await session.close()
            self._sessions.clear()


# =============================================================================
# SERVICE MESH CLIENT
# =============================================================================

class ServiceMeshClient:
    """
    Client for making requests through the service mesh.

    Features:
        - Automatic endpoint discovery
        - Load balancing
        - Circuit breaker protection
        - Retry with backoff
        - Connection pooling
    """

    def __init__(
        self,
        config: ServiceMeshConfig,
        registry: ServiceRegistry,
        load_balancer: LoadBalancer,
        circuit_manager: CircuitBreakerManager,
        connection_pool: ConnectionPool,
    ):
        self._config = config
        self._registry = registry
        self._load_balancer = load_balancer
        self._circuit_manager = circuit_manager
        self._connection_pool = connection_pool

    async def request(
        self,
        service_name: str,
        method: str,
        path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a request to a service.

        Args:
            service_name: Name of the service
            method: HTTP method
            path: Request path
            **kwargs: Additional request arguments

        Returns:
            Response data
        """
        # Get endpoints
        endpoints = await self._registry.get_endpoints(service_name)

        if not endpoints:
            raise ServiceNotFoundError(f"No endpoints for service: {service_name}")

        # Retry loop
        last_error: Optional[Exception] = None

        for attempt in range(self._config.retry_max_attempts):
            # Select endpoint
            sticky_key = kwargs.pop("sticky_key", None)
            endpoint = await self._load_balancer.select(endpoints, sticky_key)

            if not endpoint:
                raise NoHealthyEndpointError(f"No healthy endpoints for: {service_name}")

            # Check circuit breaker
            if not await self._circuit_manager.allow_request(endpoint):
                logger.debug(f"Circuit open for {endpoint.url}, trying next")
                continue

            try:
                # Make request
                endpoint.active_connections += 1
                start_time = time.time()

                session = await self._connection_pool.get_session(f"{endpoint.host}:{endpoint.port}")
                url = f"{endpoint.url}{path}"

                async with session.request(method, url, **kwargs) as response:
                    data = await response.json()

                    # Update metrics
                    response_time = (time.time() - start_time) * 1000
                    endpoint.avg_response_time_ms = (
                        endpoint.avg_response_time_ms * 0.9 + response_time * 0.1
                    )
                    endpoint.total_requests += 1
                    endpoint.active_connections -= 1

                    # Record success
                    await self._circuit_manager.record_success(endpoint)

                    return data

            except Exception as e:
                endpoint.active_connections -= 1
                last_error = e

                # Record failure
                await self._circuit_manager.record_failure(endpoint)

                # Calculate backoff
                if attempt < self._config.retry_max_attempts - 1:
                    delay = self._config.retry_initial_delay_seconds * (
                        self._config.retry_exponential_base ** attempt
                    )
                    delay = min(delay, self._config.retry_max_delay_seconds)
                    jitter = delay * self._config.retry_jitter_factor * random.random()
                    delay += jitter

                    logger.debug(f"Request failed, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)

        raise RequestFailedError(f"Request failed after {self._config.retry_max_attempts} attempts: {last_error}")

    async def get(self, service_name: str, path: str, **kwargs) -> Dict[str, Any]:
        """Make GET request."""
        return await self.request(service_name, "GET", path, **kwargs)

    async def post(self, service_name: str, path: str, **kwargs) -> Dict[str, Any]:
        """Make POST request."""
        return await self.request(service_name, "POST", path, **kwargs)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ServiceMeshError(Exception):
    """Base service mesh error."""
    pass


class ServiceNotFoundError(ServiceMeshError):
    """Service not found in registry."""
    pass


class NoHealthyEndpointError(ServiceMeshError):
    """No healthy endpoints available."""
    pass


class RequestFailedError(ServiceMeshError):
    """Request failed after retries."""
    pass


class CircuitOpenError(ServiceMeshError):
    """Circuit breaker is open."""
    pass


# =============================================================================
# SERVICE MESH (MAIN CLASS)
# =============================================================================

class ServiceMesh:
    """
    Main service mesh class that ties everything together.

    Usage:
        mesh = ServiceMesh()
        await mesh.start()

        # Register a service
        await mesh.register_service(
            "jarvis-prime",
            host="localhost",
            port=8000,
        )

        # Make a request
        response = await mesh.client.get("jarvis-prime", "/health")

        await mesh.stop()
    """

    def __init__(self, config: Optional[ServiceMeshConfig] = None):
        self._config = config or ServiceMeshConfig.from_env()

        # Create components
        if self._config.registry_type == "file":
            self._registry = FileServiceRegistry(self._config.registry_path)
        else:
            self._registry = MemoryServiceRegistry()

        self._health_checker = HealthChecker(self._config, self._registry)
        self._load_balancer = LoadBalancer(self._config)
        self._circuit_manager = CircuitBreakerManager(self._config)
        self._connection_pool = ConnectionPool(self._config)

        self._client = ServiceMeshClient(
            self._config,
            self._registry,
            self._load_balancer,
            self._circuit_manager,
            self._connection_pool,
        )

        # State
        self._running = False
        self._instance_id = str(uuid.uuid4())[:8]

        # Registered services (for cleanup)
        self._registered: List[Tuple[str, str]] = []

        logger.info("Service mesh initialized")

    @property
    def client(self) -> ServiceMeshClient:
        """Get the mesh client."""
        return self._client

    @property
    def registry(self) -> ServiceRegistry:
        """Get the service registry."""
        return self._registry

    async def start(self):
        """Start the service mesh."""
        if self._running:
            return

        self._running = True

        # Start health checker
        await self._health_checker.start()

        logger.info("Service mesh started")

    async def stop(self):
        """Stop the service mesh."""
        self._running = False

        # Deregister our services
        for service_name, instance_id in self._registered:
            await self._registry.deregister(service_name, instance_id)

        # Stop health checker
        await self._health_checker.stop()

        # Close connection pool
        await self._connection_pool.close_all()

        logger.info("Service mesh stopped")

    async def register_service(
        self,
        service_name: str,
        host: str,
        port: int,
        protocol: str = "http",
        version: str = "1.0.0",
        labels: Optional[Dict[str, str]] = None,
        capabilities: Optional[List[str]] = None,
        instance_id: Optional[str] = None,
    ) -> ServiceEndpoint:
        """Register a service with the mesh."""
        endpoint = ServiceEndpoint(
            service_name=service_name,
            instance_id=instance_id or f"{service_name}-{self._instance_id}",
            host=host,
            port=port,
            protocol=protocol,
            version=version,
            labels=labels or {},
            capabilities=capabilities or [],
            status=ServiceStatus.HEALTHY,
        )

        await self._registry.register(endpoint)
        self._registered.append((service_name, endpoint.instance_id))

        logger.info(f"Registered service: {service_name} at {endpoint.url}")
        return endpoint

    async def deregister_service(self, service_name: str, instance_id: str):
        """Deregister a service."""
        await self._registry.deregister(service_name, instance_id)
        self._registered = [
            (s, i) for s, i in self._registered
            if not (s == service_name and i == instance_id)
        ]

    async def get_endpoint(self, service_name: str) -> Optional[ServiceEndpoint]:
        """Get a healthy endpoint for a service."""
        endpoints = await self._registry.get_endpoints(service_name)
        return await self._load_balancer.select(endpoints)

    async def get_all_endpoints(self, service_name: str) -> List[ServiceEndpoint]:
        """Get all endpoints for a service."""
        return await self._registry.get_endpoints(service_name)

    def get_status(self) -> Dict[str, Any]:
        """Get mesh status."""
        return {
            "running": self._running,
            "instance_id": self._instance_id,
            "config": {
                "registry_type": self._config.registry_type,
                "lb_strategy": self._config.lb_strategy.value,
                "circuit_enabled": self._config.circuit_enabled,
            },
            "circuit_breakers": self._circuit_manager.get_all_states(),
            "registered_services": self._registered,
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_mesh: Optional[ServiceMesh] = None
_mesh_lock = asyncio.Lock()


async def get_service_mesh(
    config: Optional[ServiceMeshConfig] = None,
) -> ServiceMesh:
    """Get or create the global ServiceMesh."""
    global _mesh

    if _mesh is not None:
        return _mesh

    async with _mesh_lock:
        if _mesh is not None:
            return _mesh

        _mesh = ServiceMesh(config)
        await _mesh.start()

        return _mesh


async def shutdown_service_mesh():
    """Shutdown the global mesh."""
    global _mesh

    async with _mesh_lock:
        if _mesh is not None:
            await _mesh.stop()
            _mesh = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ServiceStatus",
    "LoadBalancingStrategy",
    "CircuitState",
    # Data classes
    "ServiceEndpoint",
    "ServiceMeshConfig",
    # Registry
    "ServiceRegistry",
    "MemoryServiceRegistry",
    "FileServiceRegistry",
    # Components
    "HealthChecker",
    "LoadBalancer",
    "EndpointCircuitBreaker",
    "CircuitBreakerManager",
    "ConnectionPool",
    "ServiceMeshClient",
    # Exceptions
    "ServiceMeshError",
    "ServiceNotFoundError",
    "NoHealthyEndpointError",
    "RequestFailedError",
    "CircuitOpenError",
    # Main class
    "ServiceMesh",
    # Factory
    "get_service_mesh",
    "shutdown_service_mesh",
]
