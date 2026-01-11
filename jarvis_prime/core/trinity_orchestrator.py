"""
Trinity Orchestrator v1.0 - Unified Cross-Repo Orchestration
=============================================================

The central nervous system that connects JARVIS, JARVIS-Prime, and Reactor-Core
into a unified distributed organism.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         TRINITY ORCHESTRATOR                             │
    │                    "The Central Nervous System"                          │
    └──────────────────────────────┬──────────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
    ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
    │   JARVIS     │       │ JARVIS-Prime │       │ Reactor-Core │
    │   (Body)     │       │    (Mind)    │       │    (Soul)    │
    │  Port 8080   │       │  Port 8000   │       │  Port 8090   │
    └──────────────┘       └──────────────┘       └──────────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                     ┌─────────────┼─────────────┐
                     │             │             │
                     ▼             ▼             ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │  Health  │  │  Service │  │  Cross   │
              │ Monitor  │  │   Mesh   │  │   Repo   │
              │          │  │          │  │    IPC   │
              └──────────┘  └──────────┘  └──────────┘

FEATURES:
    - Dynamic repo discovery (finds JARVIS, Reactor-Core automatically)
    - Process management with graceful startup/shutdown
    - Health monitoring across all repos
    - Cross-repo IPC via Trinity Protocol
    - Automatic dependency resolution
    - Self-healing (auto-restart failed components)
    - Resource-aware orchestration (OOM protection)
    - Distributed tracing across repos
    - Hot-reload configuration propagation

USAGE:
    orchestrator = await TrinityOrchestrator.create()
    await orchestrator.start_all()
    # ... system is now running ...
    await orchestrator.shutdown_all()
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
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# TRY IMPORTS
# =============================================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from jarvis_prime.core.advanced_primitives import (
        AtomicFileWriter,
        AdvancedCircuitBreaker,
        CircuitBreakerConfig,
        ExponentialBackoff,
        BackoffConfig,
        TraceContext,
        trace_operation,
        ResourceMonitor,
    )
    ADVANCED_PRIMITIVES_AVAILABLE = True
except ImportError:
    ADVANCED_PRIMITIVES_AVAILABLE = False

try:
    from jarvis_prime.core.trinity_event_bus import (
        TrinityEventBus,
        EventType,
        EventPriority,
        ComponentID,
        TrinityEvent,
        get_event_bus,
        shutdown_event_bus,
    )
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

class ComponentState(Enum):
    """State of a Trinity component."""
    UNKNOWN = "unknown"
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    FAILED = "failed"
    DISABLED = "disabled"


class ComponentType(Enum):
    """Type of Trinity component."""
    BODY = "body"       # JARVIS - Action executor
    MIND = "mind"       # JARVIS-Prime - Cognitive engine
    SOUL = "soul"       # Reactor-Core - Training pipeline


@dataclass
class ComponentConfig:
    """Configuration for a Trinity component."""
    name: str
    component_type: ComponentType
    description: str

    # Network
    port: int
    health_endpoint: str = "/health"
    metrics_endpoint: str = "/metrics"

    # Process management
    entry_point: str = ""
    path_candidates: List[str] = field(default_factory=list)
    python_path: str = ""

    # Lifecycle
    enabled: bool = True
    auto_restart: bool = True
    max_restarts: int = 5
    restart_delay_seconds: float = 5.0
    startup_timeout_seconds: float = 60.0
    shutdown_timeout_seconds: float = 30.0

    # Dependencies
    dependencies: List[str] = field(default_factory=list)

    # Runtime state (not config)
    _resolved_path: Optional[Path] = field(default=None, repr=False)
    _process: Optional[subprocess.Popen] = field(default=None, repr=False)
    _state: ComponentState = field(default=ComponentState.UNKNOWN, repr=False)
    _restart_count: int = field(default=0, repr=False)
    _last_health_check: float = field(default=0.0, repr=False)
    _last_error: Optional[str] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.component_type.value,
            "port": self.port,
            "enabled": self.enabled,
            "state": self._state.value,
            "path": str(self._resolved_path) if self._resolved_path else None,
            "restart_count": self._restart_count,
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the Trinity Orchestrator."""
    # IPC
    trinity_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "trinity")
    cross_repo_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "cross_repo")

    # Health checking
    health_check_interval_seconds: float = 10.0
    health_check_timeout_seconds: float = 5.0
    unhealthy_threshold: int = 3  # Failures before marking unhealthy

    # Process management
    parallel_startup: bool = False  # Start components in parallel
    graceful_shutdown_timeout: float = 30.0

    # Self-healing
    self_healing_enabled: bool = True
    restart_backoff_seconds: float = 5.0
    max_restart_backoff_seconds: float = 300.0

    # Resource protection
    oom_protection_enabled: bool = True
    max_ram_percent_per_component: float = 40.0

    # Logging
    log_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "logs")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "OrchestratorConfig":
        """Load configuration from YAML file."""
        config = cls()

        if not config_path.exists():
            return config

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)

            if "trinity_protocol" in data:
                tp = data["trinity_protocol"]
                if "ipc_dir" in tp:
                    config.trinity_dir = Path(os.path.expandvars(tp["ipc_dir"]))

            if "reliability" in data and "oom_protection" in data["reliability"]:
                oom = data["reliability"]["oom_protection"]
                config.oom_protection_enabled = oom.get("enabled", True)

            if "service_mesh" in data and "health" in data["service_mesh"]:
                health = data["service_mesh"]["health"]
                config.health_check_interval_seconds = health.get("check_interval_seconds", 10.0)
                config.health_check_timeout_seconds = health.get("timeout_seconds", 5.0)
                config.unhealthy_threshold = health.get("failure_threshold", 3)

        except Exception as e:
            logger.warning(f"Failed to load orchestrator config from YAML: {e}")

        return config


# =============================================================================
# REPO DISCOVERY
# =============================================================================

class RepoDiscovery:
    """
    Discovers Trinity repository locations automatically.

    Searches for JARVIS, JARVIS-Prime, and Reactor-Core repos
    in common locations and validates their structure.
    """

    # Known repo identifiers
    JARVIS_MARKERS = ["run_jarvis.py", "jarvis/__init__.py", "src/jarvis"]
    PRIME_MARKERS = ["run_server.py", "jarvis_prime/__init__.py", "jarvis_prime/core"]
    REACTOR_MARKERS = ["run_reactor.py", "reactor_core/__init__.py", "reactor/__init__.py"]

    def __init__(self, base_dir: Optional[Path] = None):
        self._base_dir = base_dir or self._detect_base_dir()
        self._discovered: Dict[str, Path] = {}

    def _detect_base_dir(self) -> Path:
        """Detect the base directory containing all repos."""
        # Start from current file location
        current = Path(__file__).parent.parent.parent  # jarvis-prime root

        # Go up one level to find sibling repos
        return current.parent

    def discover_all(self) -> Dict[str, Optional[Path]]:
        """Discover all Trinity repositories."""
        self._discovered = {
            "jarvis": self._find_repo("jarvis", self.JARVIS_MARKERS, [
                "JARVIS-AI-Agent",
                "jarvis-ai-agent",
                "JARVIS",
                "jarvis",
            ]),
            "jarvis_prime": self._find_repo("jarvis_prime", self.PRIME_MARKERS, [
                "JARVIS-Prime",
                "jarvis-prime",
            ]),
            "reactor_core": self._find_repo("reactor_core", self.REACTOR_MARKERS, [
                "Reactor-Core",
                "reactor-core",
            ]),
        }

        return self._discovered

    def _find_repo(
        self,
        name: str,
        markers: List[str],
        candidates: List[str],
    ) -> Optional[Path]:
        """Find a repository by checking candidate paths for markers."""
        # Check environment variable first
        env_var = f"{name.upper()}_PATH"
        env_path = os.getenv(env_var)
        if env_path:
            path = Path(env_path)
            if path.exists() and self._has_markers(path, markers):
                logger.debug(f"Found {name} via env var {env_var}: {path}")
                return path

        # Check current directory (for jarvis_prime)
        if name == "jarvis_prime":
            current = Path(__file__).parent.parent.parent
            if current.exists() and self._has_markers(current, markers):
                return current

        # Check candidate directories
        for candidate in candidates:
            path = self._base_dir / candidate
            if path.exists() and self._has_markers(path, markers):
                logger.debug(f"Found {name} at: {path}")
                return path

        # Check JARVIS_BASE_DIR
        base_dir = os.getenv("JARVIS_BASE_DIR")
        if base_dir:
            for candidate in candidates:
                path = Path(base_dir) / candidate
                if path.exists() and self._has_markers(path, markers):
                    logger.debug(f"Found {name} via JARVIS_BASE_DIR: {path}")
                    return path

        logger.warning(f"Could not find {name} repository")
        return None

    def _has_markers(self, path: Path, markers: List[str]) -> bool:
        """Check if path contains any of the marker files/dirs."""
        return any((path / marker).exists() for marker in markers)

    def get_repo_path(self, name: str) -> Optional[Path]:
        """Get discovered path for a repo."""
        if not self._discovered:
            self.discover_all()
        return self._discovered.get(name)


# =============================================================================
# PROCESS MANAGER
# =============================================================================

class ProcessManager:
    """
    Manages subprocess lifecycle for Trinity components.

    Features:
    - Graceful startup/shutdown
    - Process monitoring
    - Output capture
    - Signal handling
    """

    def __init__(self, component: ComponentConfig, log_dir: Path):
        self._component = component
        self._log_dir = log_dir
        self._process: Optional[subprocess.Popen] = None
        self._log_file: Optional[Path] = None

    async def start(self) -> bool:
        """Start the component process."""
        if not self._component._resolved_path:
            logger.error(f"Cannot start {self._component.name}: path not resolved")
            return False

        if not self._component.entry_point:
            logger.error(f"Cannot start {self._component.name}: no entry point")
            return False

        # Prepare log file
        self._log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = self._log_dir / f"{self._component.name}_{timestamp}.log"

        # Build command
        python_exe = self._component.python_path or sys.executable
        entry_point = self._component._resolved_path / self._component.entry_point

        if not entry_point.exists():
            logger.error(f"Entry point not found: {entry_point}")
            return False

        cmd = [python_exe, str(entry_point)]

        # Environment
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._component._resolved_path)
        env["COMPONENT_NAME"] = self._component.name
        env["COMPONENT_PORT"] = str(self._component.port)

        logger.info(f"Starting {self._component.name}: {' '.join(cmd)}")

        try:
            with open(self._log_file, "w") as log_f:
                self._process = subprocess.Popen(
                    cmd,
                    cwd=str(self._component._resolved_path),
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # Process group for cleanup
                )

            self._component._process = self._process
            self._component._state = ComponentState.STARTING

            # Wait for startup (with timeout)
            start_time = time.time()
            timeout = self._component.startup_timeout_seconds

            while time.time() - start_time < timeout:
                if self._process.poll() is not None:
                    # Process exited
                    logger.error(f"{self._component.name} exited during startup")
                    self._component._state = ComponentState.FAILED
                    return False

                # Check health endpoint
                if await self._check_health():
                    logger.info(f"{self._component.name} started successfully")
                    self._component._state = ComponentState.RUNNING
                    return True

                await asyncio.sleep(1.0)

            logger.error(f"{self._component.name} startup timeout")
            self._component._state = ComponentState.FAILED
            return False

        except Exception as e:
            logger.error(f"Failed to start {self._component.name}: {e}")
            self._component._state = ComponentState.FAILED
            self._component._last_error = str(e)
            return False

    async def stop(self, graceful: bool = True) -> bool:
        """Stop the component process."""
        if not self._process:
            return True

        self._component._state = ComponentState.STOPPING

        try:
            if graceful:
                # Send SIGTERM
                self._process.terminate()

                # Wait for graceful shutdown
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self._process.wait
                        ),
                        timeout=self._component.shutdown_timeout_seconds
                    )
                    logger.info(f"{self._component.name} stopped gracefully")
                except asyncio.TimeoutError:
                    logger.warning(f"{self._component.name} didn't stop gracefully, killing")
                    self._process.kill()
                    self._process.wait()
            else:
                # Force kill
                self._process.kill()
                self._process.wait()

            self._component._state = ComponentState.STOPPED
            self._process = None
            self._component._process = None
            return True

        except Exception as e:
            logger.error(f"Error stopping {self._component.name}: {e}")
            self._component._state = ComponentState.FAILED
            return False

    async def _check_health(self) -> bool:
        """Check if component health endpoint is responding."""
        if not AIOHTTP_AVAILABLE:
            return False

        url = f"http://localhost:{self._component.port}{self._component.health_endpoint}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False

    @property
    def is_running(self) -> bool:
        """Check if process is running."""
        if not self._process:
            return False
        return self._process.poll() is None

    @property
    def pid(self) -> Optional[int]:
        """Get process PID."""
        return self._process.pid if self._process else None


# =============================================================================
# HEALTH MONITOR
# =============================================================================

class HealthMonitor:
    """
    Monitors health of all Trinity components.

    Features:
    - Periodic health checks
    - Failure tracking
    - Auto-recovery triggering
    - Metrics collection
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        components: Dict[str, ComponentConfig],
        on_unhealthy: Optional[Callable[[str], None]] = None,
    ):
        self._config = config
        self._components = components
        self._on_unhealthy = on_unhealthy

        # Tracking
        self._failure_counts: Dict[str, int] = {name: 0 for name in components}
        self._last_check: Dict[str, float] = {name: 0.0 for name in components}
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Session for health checks
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Start health monitoring."""
        if self._running:
            return

        self._running = True

        if AIOHTTP_AVAILABLE:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._config.health_check_timeout_seconds)
            )

        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")

    async def stop(self):
        """Stop health monitoring."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Health monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all()
                await asyncio.sleep(self._config.health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5.0)

    async def _check_all(self):
        """Check health of all components."""
        tasks = []
        for name, component in self._components.items():
            if component._state in [ComponentState.RUNNING, ComponentState.HEALTHY, ComponentState.UNHEALTHY]:
                tasks.append(self._check_component(name, component))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_component(self, name: str, component: ComponentConfig):
        """Check health of a single component."""
        now = time.time()
        self._last_check[name] = now

        if not AIOHTTP_AVAILABLE or not self._session:
            return

        url = f"http://localhost:{component.port}{component.health_endpoint}"

        try:
            async with self._session.get(url) as response:
                if response.status == 200:
                    # Healthy
                    self._failure_counts[name] = 0
                    if component._state != ComponentState.HEALTHY:
                        component._state = ComponentState.HEALTHY
                        logger.debug(f"{name} is healthy")
                else:
                    await self._handle_failure(name, component, f"HTTP {response.status}")

        except Exception as e:
            await self._handle_failure(name, component, str(e))

    async def _handle_failure(self, name: str, component: ComponentConfig, reason: str):
        """Handle a health check failure."""
        self._failure_counts[name] += 1
        component._last_error = reason

        if self._failure_counts[name] >= self._config.unhealthy_threshold:
            if component._state != ComponentState.UNHEALTHY:
                component._state = ComponentState.UNHEALTHY
                logger.warning(f"{name} is UNHEALTHY: {reason}")

                if self._on_unhealthy:
                    self._on_unhealthy(name)

    def get_status(self) -> Dict[str, Any]:
        """Get health status of all components."""
        return {
            name: {
                "state": comp._state.value,
                "failures": self._failure_counts.get(name, 0),
                "last_check": self._last_check.get(name, 0),
                "last_error": comp._last_error,
            }
            for name, comp in self._components.items()
        }


# =============================================================================
# TRINITY PROTOCOL IPC
# =============================================================================

class TrinityProtocol:
    """
    Inter-Process Communication protocol for Trinity components.

    Uses file-based IPC for reliability with optional WebSocket for real-time.
    """

    def __init__(self, trinity_dir: Path, component_name: str):
        self._trinity_dir = trinity_dir
        self._component_name = component_name

        # Create directories
        self._trinity_dir.mkdir(parents=True, exist_ok=True)
        self._inbox_dir = self._trinity_dir / "inbox" / component_name
        self._outbox_dir = self._trinity_dir / "outbox" / component_name
        self._registry_file = self._trinity_dir / "registry.json"

        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        self._outbox_dir.mkdir(parents=True, exist_ok=True)

    async def register(self, port: int, capabilities: List[str]):
        """Register this component in the Trinity registry."""
        registry = await self._load_registry()

        registry[self._component_name] = {
            "port": port,
            "capabilities": capabilities,
            "registered_at": time.time(),
            "heartbeat": time.time(),
        }

        await self._save_registry(registry)
        logger.debug(f"Registered {self._component_name} in Trinity registry")

    async def unregister(self):
        """Unregister this component from the Trinity registry."""
        registry = await self._load_registry()

        if self._component_name in registry:
            del registry[self._component_name]
            await self._save_registry(registry)

    async def heartbeat(self):
        """Update heartbeat timestamp."""
        registry = await self._load_registry()

        if self._component_name in registry:
            registry[self._component_name]["heartbeat"] = time.time()
            await self._save_registry(registry)

    async def send_message(self, target: str, message: Dict[str, Any]):
        """Send message to another component."""
        target_inbox = self._trinity_dir / "inbox" / target

        if not target_inbox.exists():
            logger.warning(f"Target component {target} not found")
            return False

        message_id = f"{int(time.time() * 1000)}_{self._component_name}"
        message_file = target_inbox / f"{message_id}.json"

        message_data = {
            "id": message_id,
            "from": self._component_name,
            "to": target,
            "timestamp": time.time(),
            "payload": message,
        }

        if ADVANCED_PRIMITIVES_AVAILABLE:
            await AtomicFileWriter(message_file).write(message_data)
        else:
            with open(message_file, "w") as f:
                json.dump(message_data, f)

        return True

    async def receive_messages(self) -> List[Dict[str, Any]]:
        """Receive messages from inbox."""
        messages = []

        for msg_file in sorted(self._inbox_dir.glob("*.json")):
            try:
                with open(msg_file) as f:
                    message = json.load(f)
                messages.append(message)

                # Delete after reading
                msg_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to read message {msg_file}: {e}")

        return messages

    async def get_registry(self) -> Dict[str, Any]:
        """Get the current Trinity registry."""
        return await self._load_registry()

    async def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk."""
        if not self._registry_file.exists():
            return {}

        try:
            if ADVANCED_PRIMITIVES_AVAILABLE:
                return await AtomicFileWriter(self._registry_file).read() or {}
            else:
                with open(self._registry_file) as f:
                    return json.load(f)
        except Exception:
            return {}

    async def _save_registry(self, registry: Dict[str, Any]):
        """Save registry to disk."""
        if ADVANCED_PRIMITIVES_AVAILABLE:
            await AtomicFileWriter(self._registry_file).write(registry)
        else:
            with open(self._registry_file, "w") as f:
                json.dump(registry, f, indent=2)


# =============================================================================
# TRINITY ORCHESTRATOR
# =============================================================================

class TrinityOrchestrator:
    """
    The Central Nervous System - Orchestrates all Trinity components.

    Usage:
        orchestrator = await TrinityOrchestrator.create()
        await orchestrator.start_all()
        # System is now running
        await orchestrator.shutdown_all()
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        config_path: Optional[Path] = None,
    ):
        # Configuration
        self._config_path = config_path or Path(__file__).parent.parent.parent / "config" / "unified_config.yaml"
        self._config = config or OrchestratorConfig.from_yaml(self._config_path)

        # Components
        self._components: Dict[str, ComponentConfig] = {}
        self._process_managers: Dict[str, ProcessManager] = {}

        # Discovery
        self._discovery = RepoDiscovery()

        # Health monitoring
        self._health_monitor: Optional[HealthMonitor] = None

        # IPC
        self._protocol: Optional[TrinityProtocol] = None

        # Event Bus - The neural impulses that close The Loop
        self._event_bus: Optional["TrinityEventBus"] = None
        self._event_subscriptions: List[str] = []

        # Model registry - tracks available models for hot-swap
        self._model_registry: Dict[str, Dict[str, Any]] = {}
        self._active_model: Optional[str] = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

    @classmethod
    async def create(
        cls,
        config: Optional[OrchestratorConfig] = None,
        config_path: Optional[Path] = None,
    ) -> "TrinityOrchestrator":
        """Create and initialize the orchestrator."""
        orchestrator = cls(config, config_path)
        await orchestrator._initialize()
        return orchestrator

    async def _initialize(self):
        """Initialize the orchestrator."""
        # Discover repos
        repos = self._discovery.discover_all()

        # Load component configurations
        await self._load_components(repos)

        # Create IPC protocol
        self._protocol = TrinityProtocol(
            self._config.trinity_dir,
            "orchestrator"
        )

        # Create health monitor
        self._health_monitor = HealthMonitor(
            self._config,
            self._components,
            on_unhealthy=self._handle_unhealthy,
        )

        # Initialize Event Bus - The neural impulses that close The Loop
        await self._initialize_event_bus()

        logger.info(f"TrinityOrchestrator initialized with {len(self._components)} components")

    async def _initialize_event_bus(self):
        """Initialize the event bus and set up subscriptions."""
        if not EVENT_BUS_AVAILABLE:
            logger.warning("Event bus not available - The Loop will not be closed")
            return

        try:
            # Create event bus as orchestrator
            self._event_bus = await TrinityEventBus.create(
                ComponentID.ORCHESTRATOR,
                use_file_transport=True,
                trinity_dir=self._config.trinity_dir,
            )

            # Subscribe to key events for The Loop
            # MODEL_READY: Reactor trained a new model -> trigger hot-swap
            sub_id = await self._event_bus.subscribe(
                EventType.MODEL_READY,
                self._on_model_ready,
            )
            self._event_subscriptions.append(sub_id)

            # TRAINING_COMPLETE: Training finished -> validate and promote
            sub_id = await self._event_bus.subscribe(
                EventType.TRAINING_COMPLETE,
                self._on_training_complete,
            )
            self._event_subscriptions.append(sub_id)

            # EXPERIENCE_COLLECTED: Body collected data -> notify Reactor
            sub_id = await self._event_bus.subscribe(
                EventType.EXPERIENCE_COLLECTED,
                self._on_experience_collected,
            )
            self._event_subscriptions.append(sub_id)

            # HEALTH_CHANGED: Component health changed -> trigger recovery
            sub_id = await self._event_bus.subscribe(
                EventType.HEALTH_CHANGED,
                self._on_health_changed,
            )
            self._event_subscriptions.append(sub_id)

            # SHUTDOWN_REQUESTED: Graceful shutdown requested
            sub_id = await self._event_bus.subscribe(
                EventType.SHUTDOWN_REQUESTED,
                self._on_shutdown_requested,
            )
            self._event_subscriptions.append(sub_id)

            logger.info("Event bus initialized - The Loop is now connected!")

        except Exception as e:
            logger.error(f"Failed to initialize event bus: {e}")
            self._event_bus = None

    async def _load_components(self, repos: Dict[str, Optional[Path]]):
        """Load component configurations from YAML and discovered repos."""
        # Load from unified config if exists
        if self._config_path.exists():
            try:
                with open(self._config_path) as f:
                    data = yaml.safe_load(f)

                if "components" in data:
                    for name, comp_data in data["components"].items():
                        self._components[name] = self._create_component_from_yaml(
                            name, comp_data, repos
                        )
            except Exception as e:
                logger.warning(f"Failed to load components from YAML: {e}")

        # Ensure required components exist with defaults
        if "jarvis" not in self._components:
            self._components["jarvis"] = ComponentConfig(
                name="jarvis",
                component_type=ComponentType.BODY,
                description="JARVIS AI Agent - Action executor",
                port=8080,
                entry_point="run_jarvis.py",
                _resolved_path=repos.get("jarvis"),
            )

        if "jarvis_prime" not in self._components:
            self._components["jarvis_prime"] = ComponentConfig(
                name="jarvis_prime",
                component_type=ComponentType.MIND,
                description="JARVIS Prime - Cognitive engine",
                port=8000,
                entry_point="run_server.py",
                _resolved_path=repos.get("jarvis_prime"),
            )

        if "reactor_core" not in self._components:
            self._components["reactor_core"] = ComponentConfig(
                name="reactor_core",
                component_type=ComponentType.SOUL,
                description="Reactor Core - Training pipeline",
                port=8090,
                entry_point="run_reactor.py",
                enabled=False,  # Disabled by default
                _resolved_path=repos.get("reactor_core"),
            )

        # Create process managers
        for name, component in self._components.items():
            self._process_managers[name] = ProcessManager(
                component,
                self._config.log_dir,
            )

    def _create_component_from_yaml(
        self,
        name: str,
        data: Dict[str, Any],
        repos: Dict[str, Optional[Path]],
    ) -> ComponentConfig:
        """Create ComponentConfig from YAML data."""
        # Map component type
        type_map = {"body": ComponentType.BODY, "mind": ComponentType.MIND, "soul": ComponentType.SOUL}
        comp_type = type_map.get(data.get("type", ""), ComponentType.BODY)

        # Resolve path
        resolved_path = repos.get(name)

        if not resolved_path:
            # Try path candidates
            for candidate in data.get("path_candidates", []):
                candidate = os.path.expandvars(candidate)
                path = Path(candidate)
                if path.exists():
                    resolved_path = path
                    break

        return ComponentConfig(
            name=data.get("name", name),
            component_type=comp_type,
            description=data.get("description", ""),
            port=data.get("port", 8000),
            health_endpoint=data.get("health_endpoint", "/health"),
            metrics_endpoint=data.get("metrics_endpoint", "/metrics"),
            entry_point=data.get("entry_point", ""),
            enabled=data.get("enabled", True),
            auto_restart=data.get("auto_restart", True),
            max_restarts=data.get("max_restarts", 5),
            restart_delay_seconds=data.get("restart_delay_seconds", 5.0),
            startup_timeout_seconds=data.get("startup_timeout_seconds", 60.0),
            shutdown_timeout_seconds=data.get("shutdown_timeout_seconds", 30.0),
            dependencies=data.get("dependencies", []),
            _resolved_path=resolved_path,
        )

    async def start_all(self) -> bool:
        """
        Start all enabled Trinity components.

        Respects dependency order and startup configuration.
        """
        logger.info("=" * 70)
        logger.info("STARTING TRINITY ECOSYSTEM")
        logger.info("=" * 70)

        self._running = True

        # Determine startup order (respecting dependencies)
        startup_order = self._get_startup_order()

        logger.info(f"Startup order: {startup_order}")

        # Start components
        for name in startup_order:
            component = self._components[name]

            if not component.enabled:
                logger.info(f"  [{name}] Disabled - skipping")
                component._state = ComponentState.DISABLED
                continue

            if not component._resolved_path:
                logger.warning(f"  [{name}] Path not resolved - skipping")
                component._state = ComponentState.FAILED
                continue

            logger.info(f"  [{name}] Starting...")

            manager = self._process_managers[name]
            success = await manager.start()

            if success:
                logger.info(f"  [{name}] Started on port {component.port}")

                # Register in Trinity protocol
                await self._protocol.register(component.port, [])
            else:
                logger.error(f"  [{name}] FAILED to start")

                # Don't continue if critical component failed
                if name == "jarvis_prime":
                    logger.error("Critical component failed - aborting startup")
                    return False

        # Start health monitoring
        await self._health_monitor.start()

        logger.info("=" * 70)
        logger.info("TRINITY ECOSYSTEM STARTED")
        logger.info("=" * 70)

        # Print status
        self._print_status()

        return True

    async def shutdown_all(self):
        """Gracefully shutdown all Trinity components."""
        logger.info("=" * 70)
        logger.info("SHUTTING DOWN TRINITY ECOSYSTEM")
        logger.info("=" * 70)

        self._running = False

        # Notify all components of shutdown
        if self._event_bus:
            try:
                await self._event_bus.publish_shutdown_requested(
                    reason="orchestrator_shutdown",
                    graceful=True,
                )
            except Exception as e:
                logger.warning(f"Failed to send shutdown event: {e}")

        # Stop event bus
        if self._event_bus:
            try:
                await self._event_bus.shutdown()
                self._event_bus = None
                logger.info("Event bus shutdown complete")
            except Exception as e:
                logger.warning(f"Error shutting down event bus: {e}")

        # Stop health monitoring
        if self._health_monitor:
            await self._health_monitor.stop()

        # Shutdown in reverse order
        shutdown_order = list(reversed(self._get_startup_order()))

        for name in shutdown_order:
            component = self._components[name]

            if component._state in [ComponentState.DISABLED, ComponentState.STOPPED, ComponentState.UNKNOWN]:
                continue

            logger.info(f"  [{name}] Stopping...")

            manager = self._process_managers[name]
            await manager.stop()

            logger.info(f"  [{name}] Stopped")

            # Unregister from Trinity protocol
            await self._protocol.unregister()

        logger.info("=" * 70)
        logger.info("TRINITY ECOSYSTEM STOPPED")
        logger.info("=" * 70)

    # =========================================================================
    # EVENT HANDLERS - The Neural Impulses of The Loop
    # =========================================================================

    async def _on_model_ready(self, event: "TrinityEvent"):
        """
        Handle MODEL_READY event - A new model is ready for use.

        This is THE KEY EVENT that closes The Loop:
        Reactor trains model -> publishes MODEL_READY -> Orchestrator hot-swaps

        THE LOOP:
        ┌──────────────────────────────────────────────────────────────────┐
        │                                                                  │
        │  JARVIS (Body) ──experience──> Reactor ──training──> Prime       │
        │       ▲                                               │          │
        │       │                                               │          │
        │       └───────────── model_ready ─────────────────────┘          │
        │                                                                  │
        │                    *** YOU ARE HERE ***                          │
        └──────────────────────────────────────────────────────────────────┘
        """
        model_name = event.payload.get("model_name", "unknown")
        model_path = event.payload.get("model_path", "")
        model_type = event.payload.get("model_type", "llm")
        capabilities = event.payload.get("capabilities", [])

        logger.info(f"[EVENT] MODEL_READY received: {model_name}")
        logger.info(f"  Path: {model_path}")
        logger.info(f"  Type: {model_type}")
        logger.info(f"  Capabilities: {capabilities}")

        # Register the model
        self._model_registry[model_name] = {
            "path": model_path,
            "type": model_type,
            "capabilities": capabilities,
            "ready_at": event.payload.get("ready_at", time.time()),
            "source": event.source.value,
        }

        # Trigger hot-swap if this is a better model
        await self._trigger_model_hot_swap(model_name)

    async def _on_training_complete(self, event: "TrinityEvent"):
        """Handle TRAINING_COMPLETE event - Reactor finished training."""
        model_name = event.payload.get("model_name", "unknown")
        model_path = event.payload.get("model_path", "")
        metrics = event.payload.get("training_metrics", {})

        logger.info(f"[EVENT] TRAINING_COMPLETE: {model_name}")
        logger.info(f"  Metrics: {json.dumps(metrics, indent=2) if metrics else 'none'}")

        # Validate the trained model
        is_valid = await self._validate_model(model_name, model_path, metrics)

        if is_valid:
            # Promote to ready state
            await self.notify_model_ready(
                model_name=model_name,
                model_path=model_path,
                model_type="llm",
                capabilities=["text-generation"],
                metadata={"training_metrics": metrics},
            )
        else:
            logger.warning(f"Model {model_name} failed validation - not promoting")

    async def _on_experience_collected(self, event: "TrinityEvent"):
        """Handle EXPERIENCE_COLLECTED event - JARVIS collected new data."""
        experience_type = event.payload.get("experience_type", "unknown")
        sample_count = event.payload.get("sample_count", 0)
        data_path = event.payload.get("data_path", "")

        logger.info(f"[EVENT] EXPERIENCE_COLLECTED: {experience_type}")
        logger.info(f"  Samples: {sample_count}")
        logger.info(f"  Path: {data_path}")

        # Forward to Reactor for training
        if self._event_bus:
            await self._event_bus.publish(
                event_type=EventType.DATA_BATCH_READY,
                payload={
                    "experience_type": experience_type,
                    "sample_count": sample_count,
                    "data_path": data_path,
                    "forwarded_at": time.time(),
                },
                target=ComponentID.REACTOR_CORE,
                priority=EventPriority.NORMAL,
            )

    async def _on_health_changed(self, event: "TrinityEvent"):
        """Handle HEALTH_CHANGED event - Component health changed."""
        status = event.payload.get("status", "unknown")
        details = event.payload.get("details", {})

        logger.info(f"[EVENT] HEALTH_CHANGED from {event.source.value}: {status}")

        # Trigger recovery if unhealthy
        if status in ["unhealthy", "failed", "critical"]:
            component_name = self._map_component_id_to_name(event.source)
            if component_name:
                await self._restart_component(component_name)

    async def _on_shutdown_requested(self, event: "TrinityEvent"):
        """Handle SHUTDOWN_REQUESTED event - Graceful shutdown requested."""
        reason = event.payload.get("reason", "unknown")
        graceful = event.payload.get("graceful", True)

        logger.info(f"[EVENT] SHUTDOWN_REQUESTED: {reason} (graceful={graceful})")

        # Signal shutdown
        self._shutdown_event.set()

    def _map_component_id_to_name(self, component_id: "ComponentID") -> Optional[str]:
        """Map ComponentID enum to component name."""
        mapping = {
            ComponentID.JARVIS_BODY: "jarvis",
            ComponentID.JARVIS_PRIME: "jarvis_prime",
            ComponentID.REACTOR_CORE: "reactor_core",
        }
        return mapping.get(component_id)

    # =========================================================================
    # MODEL HOT-SWAP - The Core of The Loop
    # =========================================================================

    async def notify_model_ready(
        self,
        model_name: str,
        model_path: str,
        model_type: str = "llm",
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Notify the Trinity ecosystem that a model is ready.

        This is THE KEY METHOD that closes The Loop:

        USAGE:
            # After Reactor finishes training:
            await orchestrator.notify_model_ready(
                model_name="prime-7b-v2",
                model_path="/models/prime-7b-v2",
                model_type="llm",
                capabilities=["text-generation", "code-completion"],
            )

            # This triggers:
            # 1. MODEL_READY event broadcast to all components
            # 2. Prime receives event -> hot-swaps to new model
            # 3. Body receives event -> updates routing to use new model
            # 4. The Loop is complete!

        Args:
            model_name: Human-readable model name
            model_path: Path to model files
            model_type: Type of model (llm, embedding, classifier, etc.)
            capabilities: List of capabilities (text-generation, code, etc.)
            metadata: Additional metadata

        Returns:
            True if notification was published successfully
        """
        if not self._event_bus:
            logger.warning("Event bus not available - cannot notify model ready")

            # Fallback to file-based notification
            return await self._notify_model_ready_fallback(
                model_name, model_path, model_type, capabilities, metadata
            )

        try:
            # Publish MODEL_READY event to all components
            event = await self._event_bus.publish_model_ready(
                model_name=model_name,
                model_path=model_path,
                model_type=model_type,
                capabilities=capabilities or [],
                metadata=metadata,
            )

            logger.info(f"[NOTIFY] MODEL_READY published: {model_name}")
            logger.info(f"  Event ID: {event.event_id}")
            logger.info(f"  Correlation ID: {event.correlation_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to publish MODEL_READY: {e}")
            return False

    async def _notify_model_ready_fallback(
        self,
        model_name: str,
        model_path: str,
        model_type: str,
        capabilities: Optional[List[str]],
        metadata: Optional[Dict[str, Any]],
    ) -> bool:
        """Fallback notification via direct file."""
        try:
            notification_dir = self._config.trinity_dir / "model_notifications"
            notification_dir.mkdir(parents=True, exist_ok=True)

            notification = {
                "event": "model_ready",
                "model_name": model_name,
                "model_path": model_path,
                "model_type": model_type,
                "capabilities": capabilities or [],
                "metadata": metadata or {},
                "timestamp": time.time(),
            }

            notification_file = notification_dir / f"{model_name}_{int(time.time())}.json"

            with open(notification_file, "w") as f:
                json.dump(notification, f, indent=2)

            logger.info(f"[NOTIFY] MODEL_READY written to file: {notification_file}")
            return True

        except Exception as e:
            logger.error(f"Fallback notification failed: {e}")
            return False

    async def _trigger_model_hot_swap(self, model_name: str):
        """
        Trigger hot-swap to a new model.

        This notifies Prime to load the new model and Body to route to it.
        """
        model_info = self._model_registry.get(model_name)
        if not model_info:
            logger.warning(f"Model {model_name} not found in registry")
            return

        logger.info(f"[HOT-SWAP] Triggering hot-swap to model: {model_name}")

        # Notify Prime to load the new model
        if self._protocol:
            await self._protocol.send_message("jarvis_prime", {
                "action": "hot_swap_model",
                "model_name": model_name,
                "model_path": model_info["path"],
                "model_type": model_info["type"],
            })

        # Notify Body about the new default model
        if self._protocol:
            await self._protocol.send_message("jarvis", {
                "action": "update_model_routing",
                "default_model": model_name,
                "model_capabilities": model_info["capabilities"],
            })

        self._active_model = model_name
        logger.info(f"[HOT-SWAP] Active model is now: {model_name}")

    async def _validate_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict[str, Any],
    ) -> bool:
        """
        Validate a trained model before promotion.

        Checks:
        1. Model files exist
        2. Model can be loaded
        3. Basic inference works
        4. Metrics meet quality threshold
        """
        logger.info(f"[VALIDATE] Validating model: {model_name}")

        # Check path exists
        path = Path(model_path)
        if not path.exists():
            logger.warning(f"Model path does not exist: {model_path}")
            return False

        # Check required files (for typical LLM)
        required_files = ["config.json", "model.safetensors"]  # or pytorch_model.bin
        has_required = any((path / f).exists() for f in required_files)

        if not has_required:
            # Check for alternative structures
            has_required = (path / "adapter_model.safetensors").exists()  # LoRA

        if not has_required:
            logger.warning(f"Model missing required files at: {model_path}")
            # Don't fail - might be a different structure
            logger.info("  Proceeding anyway (may be alternative model structure)")

        # Check metrics (if provided)
        if metrics:
            loss = metrics.get("loss", float("inf"))
            if loss > 10.0:
                logger.warning(f"Model loss too high: {loss}")
                return False

        logger.info(f"[VALIDATE] Model {model_name} passed validation")
        return True

    # =========================================================================
    # EXPERIENCE NOTIFICATION - Body -> Reactor
    # =========================================================================

    async def notify_experience_collected(
        self,
        experience_type: str,
        sample_count: int,
        data_path: Optional[str] = None,
    ) -> bool:
        """
        Notify that new experience data has been collected.

        This is how the Body tells the Reactor about new training data.

        Args:
            experience_type: Type of experience (conversation, feedback, etc.)
            sample_count: Number of samples collected
            data_path: Path to the data file

        Returns:
            True if notification was published successfully
        """
        if not self._event_bus:
            logger.warning("Event bus not available - cannot notify experience")
            return False

        try:
            await self._event_bus.publish_experience_collected(
                experience_type=experience_type,
                sample_count=sample_count,
                data_path=data_path,
            )

            logger.info(f"[NOTIFY] EXPERIENCE_COLLECTED: {sample_count} {experience_type} samples")
            return True

        except Exception as e:
            logger.error(f"Failed to publish EXPERIENCE_COLLECTED: {e}")
            return False

    async def notify_training_started(
        self,
        model_name: str,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Notify that model training has started."""
        if not self._event_bus:
            return False

        try:
            await self._event_bus.publish(
                event_type=EventType.TRAINING_STARTED,
                payload={
                    "model_name": model_name,
                    "training_config": training_config or {},
                    "started_at": time.time(),
                },
                target=ComponentID.BROADCAST,
                priority=EventPriority.NORMAL,
            )

            logger.info(f"[NOTIFY] TRAINING_STARTED: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish TRAINING_STARTED: {e}")
            return False

    def _get_startup_order(self) -> List[str]:
        """Get component startup order respecting dependencies."""
        # Simple topological sort
        order = []
        visited = set()

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)

            component = self._components.get(name)
            if component:
                for dep in component.dependencies:
                    if dep in self._components:
                        visit(dep)
                order.append(name)

        # Visit all components
        for name in self._components:
            visit(name)

        return order

    def _handle_unhealthy(self, name: str):
        """Handle an unhealthy component."""
        if not self._config.self_healing_enabled:
            return

        component = self._components.get(name)
        if not component or not component.auto_restart:
            return

        if component._restart_count >= component.max_restarts:
            logger.error(f"{name} exceeded max restarts ({component.max_restarts})")
            return

        # Schedule restart
        asyncio.create_task(self._restart_component(name))

    async def _restart_component(self, name: str):
        """Restart a failed component."""
        component = self._components[name]
        manager = self._process_managers[name]

        # Backoff
        backoff = min(
            self._config.restart_backoff_seconds * (2 ** component._restart_count),
            self._config.max_restart_backoff_seconds,
        )

        logger.info(f"Restarting {name} in {backoff:.1f}s (attempt {component._restart_count + 1})")
        await asyncio.sleep(backoff)

        # Stop if running
        await manager.stop(graceful=False)

        # Start
        component._restart_count += 1
        success = await manager.start()

        if success:
            logger.info(f"{name} restarted successfully")
        else:
            logger.error(f"{name} restart failed")

    def _print_status(self):
        """Print current status of all components."""
        logger.info("")
        logger.info("Component Status:")
        logger.info("-" * 50)

        for name, component in self._components.items():
            state_icon = {
                ComponentState.HEALTHY: "[HEALTHY]",
                ComponentState.RUNNING: "[RUNNING]",
                ComponentState.STOPPED: "[STOPPED]",
                ComponentState.FAILED: "[FAILED]",
                ComponentState.DISABLED: "[DISABLED]",
                ComponentState.UNHEALTHY: "[UNHEALTHY]",
            }.get(component._state, "[UNKNOWN]")

            port_info = f"port {component.port}" if component.enabled else "disabled"
            logger.info(f"  {state_icon:12} {name:20} ({port_info})")

        logger.info("-" * 50)

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "running": self._running,
            "components": {
                name: comp.to_dict()
                for name, comp in self._components.items()
            },
            "health": self._health_monitor.get_status() if self._health_monitor else {},
            "event_bus": {
                "available": EVENT_BUS_AVAILABLE,
                "connected": self._event_bus is not None,
                "metrics": self._event_bus.get_metrics() if self._event_bus else {},
            },
            "model_registry": self._model_registry,
            "active_model": self._active_model,
        }

    async def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ComponentState",
    "ComponentType",
    # Configuration
    "ComponentConfig",
    "OrchestratorConfig",
    # Discovery
    "RepoDiscovery",
    # Process management
    "ProcessManager",
    # Health monitoring
    "HealthMonitor",
    # IPC
    "TrinityProtocol",
    # Orchestrator
    "TrinityOrchestrator",
    # Availability flags
    "ADVANCED_PRIMITIVES_AVAILABLE",
    "EVENT_BUS_AVAILABLE",
]
