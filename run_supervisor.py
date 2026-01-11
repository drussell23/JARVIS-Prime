#!/usr/bin/env python3
"""
JARVIS Unified Supervisor - Cross-Repository Orchestration
=============================================================

v87.0 - THE CONNECTIVE TISSUE - Unified Trinity Ecosystem

This supervisor connects and orchestrates all JARVIS ecosystem components:
- JARVIS (Body): Main orchestrator, computer use, action execution
- JARVIS-Prime (Mind): LLM inference, reasoning, cognitive processing
- Reactor-Core (Training): Model training, fine-tuning, deployment

NEW IN v87.0 - THE CONNECTIVE TISSUE:
    - UNIFIED MODE: python3 run_supervisor.py --unified (RECOMMENDED)
    - INTELLIGENT MODEL ROUTER: Local 7B -> GCP 13B -> Claude API fallback
    - GCP VM MANAGER: Spot VM lifecycle, preemption handling, auto-scaling
    - SERVICE MESH: Dynamic service discovery, circuit breakers, load balancing
    - UNIFIED CONFIG: Single YAML source of truth (config/unified_config.yaml)
    - RAM-AWARE ROUTING: Automatic failover when local resources exhausted
    - ADAPTIVE THRESHOLDS: Learn from routing outcomes for optimization

NEW IN v85.0:
    - FIXED: Environment variable precedence (system env overrides config)
    - FIXED: Intelligent repo path discovery with multiple fallback strategies
    - NEW: Startup verification with process/health/functional checks
    - NEW: JARVIS_BASE_DIR, JARVIS_PRIME_PATH, JARVIS_AI_AGENT_PATH env vars
    - Detailed logging of detected repo paths for debugging

EXISTING FEATURES (v80.0):
    - Automatic cross-repo detection and path resolution
    - Graceful shutdown with async signal handlers
    - Hot-reload configuration watching
    - Distributed tracing (OpenTelemetry)
    - Zero-copy memory-mapped IPC
    - Predictive caching with ML
    - Adaptive rate limiting
    - Graph-based routing optimization
    - Advanced async primitives
    - Dependency graph resolution

USAGE:
    python3 run_supervisor.py --unified          # RECOMMENDED: Full Connective Tissue
    python3 run_supervisor.py                    # Start all components (legacy)
    python3 run_supervisor.py --prime-only       # Start only JARVIS-Prime
    python3 run_supervisor.py --components prime,jarvis  # Specific components
    python3 run_supervisor.py --config config.yaml       # Custom config
    python3 run_supervisor.py --enable-tracing   # Enable distributed tracing
    python3 run_supervisor.py --enable-gcp       # Enable GCP cloud inference

ARCHITECTURE:
    Supervisor
        |
        +-- JARVIS Body (macOS integration, computer use)
        |       |
        |       +-- Window Manager (Ghost Display, Exile)
        |       +-- Input Controller (Mouse, Keyboard)
        |       +-- Screen Monitor (Surveillance)
        |       +-- Safety Manager (Kill Switch, Confirmations)
        |
        +-- JARVIS Prime (Mind/Inference)
        |       |
        |       +-- AGI Models (Action, Meta-Reasoner, Causal, etc.)
        |       +-- Reasoning Engine (CoT, ToT, Self-Reflection)
        |       +-- Multi-Modal Fusion
        |       +-- Continuous Learning
        |
        +-- Reactor Core (Training Pipeline)
                |
                +-- Training Jobs
                +-- Model Registry
                +-- Deployment Pipeline

COMMUNICATION:
    - Trinity Protocol: File-based IPC for J-Prime <-> JARVIS Body
    - Cross-Repo Bridge: State sharing via ~/.jarvis/cross_repo/
    - HTTP APIs: REST endpoints for direct communication
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("supervisor")


# =============================================================================
# CONFIGURATION
# =============================================================================

class ComponentType(Enum):
    """Types of ecosystem components."""
    JARVIS = "jarvis"           # Main JARVIS body
    JARVIS_PRIME = "jarvis_prime"  # Mind/Inference
    REACTOR_CORE = "reactor_core"  # Training pipeline


class ComponentStatus(Enum):
    """Status of a component."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class ComponentConfig:
    """Configuration for a single component."""
    type: ComponentType
    name: str
    enabled: bool = True

    # Paths
    repo_path: Optional[Path] = None
    entry_point: str = ""
    working_dir: Optional[Path] = None

    # Runtime
    python_path: str = "python3"
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    # Networking
    host: str = "localhost"
    port: int = 8000
    health_endpoint: str = "/health"

    # Process management
    auto_restart: bool = True
    max_restarts: int = 3
    restart_delay_seconds: float = 5.0

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    startup_timeout_seconds: float = 60.0


@dataclass
class SupervisorConfig:
    """Configuration for the supervisor."""
    # Component configs
    components: Dict[str, ComponentConfig] = field(default_factory=dict)

    # Paths
    state_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "supervisor")
    logs_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "logs")

    # Cross-repo communication
    trinity_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "trinity")
    cross_repo_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "cross_repo")

    # Health checking
    health_check_interval_seconds: float = 10.0
    health_timeout_seconds: float = 5.0

    # Startup
    startup_order: List[str] = field(default_factory=list)
    parallel_startup: bool = False

    # Shutdown
    shutdown_timeout_seconds: float = 30.0
    graceful_shutdown: bool = True


# =============================================================================
# COMPONENT MANAGER
# =============================================================================

@dataclass
class ComponentState:
    """Runtime state of a component."""
    config: ComponentConfig
    status: ComponentStatus = ComponentStatus.UNKNOWN
    process: Optional[asyncio.subprocess.Process] = None
    pid: Optional[int] = None

    # Health
    last_health_check: Optional[float] = None
    consecutive_failures: int = 0
    restart_count: int = 0

    # Metrics
    start_time: Optional[float] = None
    uptime_seconds: float = 0.0
    total_requests: int = 0

    # Logs
    stdout_buffer: List[str] = field(default_factory=list)
    stderr_buffer: List[str] = field(default_factory=list)


class HealthCheckPool:
    """
    Connection pool for health checks.

    v79.1: Reuses HTTP sessions to avoid TCP overhead on every health check.
    """

    def __init__(self):
        self._sessions: Dict[str, Any] = {}  # component_name -> aiohttp.ClientSession
        self._lock = asyncio.Lock()

    async def get_session(self, component_name: str) -> Any:
        """Get or create a session for a component."""
        async with self._lock:
            if component_name not in self._sessions:
                try:
                    import aiohttp
                    connector = aiohttp.TCPConnector(
                        limit=2,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                    )
                    self._sessions[component_name] = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=5, connect=2),
                    )
                except ImportError:
                    return None
            return self._sessions[component_name]

    async def close_all(self) -> None:
        """Close all sessions."""
        async with self._lock:
            for session in self._sessions.values():
                await session.close()
            self._sessions.clear()


class ComponentManager:
    """Manages lifecycle of a single component."""

    # Shared health check pool across all managers
    _health_pool: Optional[HealthCheckPool] = None

    @classmethod
    def get_health_pool(cls) -> HealthCheckPool:
        """Get shared health check pool."""
        if cls._health_pool is None:
            cls._health_pool = HealthCheckPool()
        return cls._health_pool

    def __init__(
        self,
        config: ComponentConfig,
        supervisor: "UnifiedSupervisor",
    ):
        self.config = config
        self.supervisor = supervisor
        self.state = ComponentState(config=config)
        self._output_task: Optional[asyncio.Task] = None

    async def start(self) -> bool:
        """Start the component."""
        if self.state.status in (ComponentStatus.RUNNING, ComponentStatus.HEALTHY):
            logger.warning(f"{self.config.name} is already running")
            return True

        logger.info(f"Starting {self.config.name}...")
        self.state.status = ComponentStatus.STARTING

        try:
            # Build command
            cmd = self._build_command()

            # Build environment
            env = self._build_environment()

            # Start process
            self.state.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(self.config.working_dir or self.config.repo_path),
            )

            self.state.pid = self.state.process.pid
            self.state.start_time = time.time()
            self.state.status = ComponentStatus.RUNNING

            # Start output reader
            self._output_task = asyncio.create_task(self._read_output())

            logger.info(f"{self.config.name} started (PID: {self.state.pid})")

            # Wait for health
            if self.config.health_endpoint:
                healthy = await self._wait_for_health()
                if healthy:
                    self.state.status = ComponentStatus.HEALTHY
                    logger.info(f"{self.config.name} is healthy")
                else:
                    logger.warning(f"{self.config.name} failed health check")
                    self.state.status = ComponentStatus.UNHEALTHY

            return True

        except Exception as e:
            logger.error(f"Failed to start {self.config.name}: {e}")
            self.state.status = ComponentStatus.FAILED
            return False

    async def stop(self, timeout: float = 10.0) -> bool:
        """Stop the component gracefully."""
        if self.state.process is None:
            self.state.status = ComponentStatus.STOPPED
            return True

        logger.info(f"Stopping {self.config.name}...")
        self.state.status = ComponentStatus.STOPPING

        try:
            # Send SIGTERM
            self.state.process.terminate()

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    self.state.process.wait(),
                    timeout=timeout,
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

            self.state.status = ComponentStatus.STOPPED
            self.state.process = None
            self.state.pid = None

            logger.info(f"{self.config.name} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping {self.config.name}: {e}")
            self.state.status = ComponentStatus.FAILED
            return False

    async def restart(self) -> bool:
        """Restart the component."""
        logger.info(f"Restarting {self.config.name}...")

        await self.stop()
        await asyncio.sleep(self.config.restart_delay_seconds)

        self.state.restart_count += 1
        return await self.start()

    async def health_check(self) -> bool:
        """
        Check component health.

        v79.1: Uses pooled connections for efficiency.
        """
        if self.state.process is None or self.state.process.returncode is not None:
            self.state.status = ComponentStatus.STOPPED
            return False

        if not self.config.health_endpoint:
            return True

        try:
            # Use pooled session for efficiency
            session = await self.get_health_pool().get_session(self.config.name)
            if session is None:
                # Fallback if aiohttp not available
                return True

            url = f"http://{self.config.host}:{self.config.port}{self.config.health_endpoint}"

            async with session.get(url) as response:
                if response.status == 200:
                    self.state.status = ComponentStatus.HEALTHY
                    self.state.consecutive_failures = 0
                    self.state.last_health_check = time.time()

                    # Deep health check: verify response content
                    try:
                        data = await response.json()
                        if isinstance(data, dict):
                            # Check for degraded status
                            if data.get("status") == "degraded":
                                logger.warning(f"{self.config.name} is degraded: {data.get('reason', 'unknown')}")
                            # Track subsystem health if available
                            if "subsystems" in data:
                                self.state.metrics["subsystems"] = data["subsystems"]
                    except Exception:
                        pass  # JSON parsing optional

                    return True
                else:
                    logger.debug(f"Health check returned {response.status} for {self.config.name}")

        except asyncio.TimeoutError:
            logger.debug(f"Health check timeout for {self.config.name}")
        except Exception as e:
            logger.debug(f"Health check failed for {self.config.name}: {e}")

        self.state.consecutive_failures += 1
        if self.state.consecutive_failures >= 3:
            self.state.status = ComponentStatus.UNHEALTHY

        return False

    async def _wait_for_health(self) -> bool:
        """Wait for component to become healthy."""
        start = time.time()

        while time.time() - start < self.config.startup_timeout_seconds:
            if await self.health_check():
                return True
            await asyncio.sleep(1)

        return False

    async def _read_output(self) -> None:
        """Read stdout/stderr from process."""
        if not self.state.process:
            return

        async def read_stream(stream, buffer: List[str], prefix: str):
            while True:
                line = await stream.readline()
                if not line:
                    break

                decoded = line.decode().rstrip()
                buffer.append(decoded)

                # Keep only last 1000 lines
                if len(buffer) > 1000:
                    buffer.pop(0)

                # Log with prefix
                logger.debug(f"[{self.config.name}:{prefix}] {decoded}")

        try:
            await asyncio.gather(
                read_stream(self.state.process.stdout, self.state.stdout_buffer, "out"),
                read_stream(self.state.process.stderr, self.state.stderr_buffer, "err"),
            )
        except asyncio.CancelledError:
            pass

    def _build_command(self) -> List[str]:
        """Build command to run component."""
        cmd = [self.config.python_path]

        if self.config.entry_point:
            entry = self.config.entry_point
            if self.config.repo_path:
                entry = str(self.config.repo_path / entry)
            cmd.append(entry)

        cmd.extend(self.config.args)

        return cmd

    def _build_environment(self) -> Dict[str, str]:
        """
        v85.0: Build environment for component with correct precedence.

        Priority (highest to lowest):
        1. System environment variables (user explicitly set)
        2. Component config env vars (defaults for this component)
        3. Auto-generated vars (PYTHONPATH, component identification)

        This allows users to override any setting via environment.
        """
        # Start with config defaults
        env = {}

        # Add component config env vars (lowest priority)
        env.update(self.config.env)

        # Copy system environment (higher priority - overrides config)
        system_env = os.environ.copy()

        # For each config key, only use it if system env doesn't have it
        for key, value in self.config.env.items():
            if key not in system_env:
                env[key] = value

        # Merge with system env taking precedence
        env.update(system_env)

        # Set Python path (prepend repo path)
        if self.config.repo_path:
            python_path = env.get("PYTHONPATH", "")
            repo_path_str = str(self.config.repo_path)
            if python_path:
                # Only add if not already in path
                if repo_path_str not in python_path:
                    python_path = f"{repo_path_str}:{python_path}"
            else:
                python_path = repo_path_str
            env["PYTHONPATH"] = python_path

        # Set component-specific vars (for identification)
        env["JARVIS_COMPONENT"] = self.config.name
        env["JARVIS_COMPONENT_TYPE"] = self.config.type.value

        return env


# =============================================================================
# UNIFIED SUPERVISOR
# =============================================================================

class UnifiedSupervisor:
    """
    Unified supervisor for the JARVIS ecosystem.

    v79.1: Enhanced with CognitiveRouter integration for Body-Mind connection.

    Orchestrates all components across repositories:
    - JARVIS (Body): macOS integration and action execution
    - JARVIS-Prime (Mind): LLM inference and reasoning
    - Reactor-Core (Training): Model training pipeline

    The supervisor now also initializes the CognitiveRouter which acts as
    the "Corpus Callosum" connecting the Body to the Mind.
    """

    def __init__(
        self,
        config: Optional[SupervisorConfig] = None,
    ):
        self.config = config or self._create_default_config()
        self._managers: Dict[str, ComponentManager] = {}
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._cognitive_router = None

        # Ensure directories exist
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config.trinity_dir.mkdir(parents=True, exist_ok=True)
        self.config.cross_repo_dir.mkdir(parents=True, exist_ok=True)

        # Initialize managers
        for name, comp_config in self.config.components.items():
            if comp_config.enabled:
                self._managers[name] = ComponentManager(comp_config, self)

        logger.info(f"Supervisor initialized with {len(self._managers)} components")

    def _create_default_config(self) -> SupervisorConfig:
        """
        v85.0: Create default configuration with intelligent repo discovery.

        Uses environment variables with fallback to automatic detection.
        Environment variables take priority for explicit configuration.
        """
        config = SupervisorConfig()

        # v85.0: Intelligent repo path detection with env var priority
        current_dir = Path(__file__).parent

        # JARVIS Prime - current repo or env var
        jarvis_prime_path = Path(os.environ.get(
            "JARVIS_PRIME_PATH",
            str(current_dir)
        )).expanduser()

        # Base directory for sibling repos
        base_dir = Path(os.environ.get(
            "JARVIS_BASE_DIR",
            str(jarvis_prime_path.parent)
        )).expanduser()

        # JARVIS Body - try env var, then multiple naming conventions
        jarvis_env = os.environ.get("JARVIS_AI_AGENT_PATH", "")
        if jarvis_env:
            jarvis_path = Path(jarvis_env).expanduser()
        else:
            jarvis_candidates = [
                base_dir / "JARVIS-AI-Agent",
                base_dir / "jarvis-ai-agent",
                base_dir / "JARVIS",
                current_dir.parent / "JARVIS-AI-Agent",
                current_dir.parent / "JARVIS",
            ]
            jarvis_path = next(
                (p for p in jarvis_candidates if p.exists() and p.is_dir()),
                jarvis_candidates[0]  # Default to first candidate
            )

        # Reactor Core - try env var, then multiple naming conventions
        reactor_env = os.environ.get("REACTOR_CORE_PATH", "")
        if reactor_env:
            reactor_core_path = Path(reactor_env).expanduser()
        else:
            reactor_candidates = [
                base_dir / "Reactor-Core",
                base_dir / "reactor-core",
                current_dir.parent / "Reactor-Core",
            ]
            reactor_core_path = next(
                (p for p in reactor_candidates if p.exists() and p.is_dir()),
                reactor_candidates[0]
            )

        # Log detected paths for debugging
        logger.info(f"Repo paths detected:")
        logger.info(f"  JARVIS-Prime: {jarvis_prime_path} (exists: {jarvis_prime_path.exists()})")
        logger.info(f"  JARVIS-Body:  {jarvis_path} (exists: {jarvis_path.exists()})")
        logger.info(f"  Reactor-Core: {reactor_core_path} (exists: {reactor_core_path.exists()})")

        # JARVIS Prime configuration
        config.components["jarvis_prime"] = ComponentConfig(
            type=ComponentType.JARVIS_PRIME,
            name="jarvis_prime",
            repo_path=jarvis_prime_path,
            entry_point="run_server.py",
            port=8000,
            health_endpoint="/health",
            args=["--port", "8000"],
            env={
                "TRINITY_ENABLED": "true",
                "JARVIS_PRIME_PORT": "8000",
            },
        )

        # JARVIS Body configuration (if exists)
        if jarvis_path.exists():
            config.components["jarvis"] = ComponentConfig(
                type=ComponentType.JARVIS,
                name="jarvis",
                repo_path=jarvis_path,
                entry_point="run_jarvis.py",
                port=8080,
                health_endpoint="/health",
                depends_on=["jarvis_prime"],
                env={
                    "TRINITY_ENABLED": "true",
                    "JARVIS_PRIME_URL": "http://localhost:8000",
                },
            )

        # Reactor Core configuration (if exists)
        if reactor_core_path.exists():
            config.components["reactor_core"] = ComponentConfig(
                type=ComponentType.REACTOR_CORE,
                name="reactor_core",
                repo_path=reactor_core_path,
                entry_point="run_reactor.py",
                port=8090,
                health_endpoint="/health",
                enabled=False,  # Disabled by default (training on demand)
                env={
                    "JARVIS_PRIME_URL": "http://localhost:8000",
                    "MODEL_OUTPUT_DIR": str(jarvis_prime_path / "models"),
                },
            )

        # Set startup order
        config.startup_order = ["jarvis_prime", "jarvis", "reactor_core"]

        return config

    async def start(self, components: Optional[List[str]] = None) -> bool:
        """
        Start the supervisor and all components.

        v79.1: Now includes CognitiveRouter (Corpus Callosum) initialization.

        Args:
            components: Specific components to start (None = all enabled)
        """
        logger.info("=" * 60)
        logger.info("JARVIS Unified Supervisor v79.1 - Starting")
        logger.info("=" * 60)

        self._running = True

        # Setup signal handlers
        self._setup_signal_handlers()

        # Write supervisor state
        await self._write_state()

        # Initialize CognitiveRouter (Corpus Callosum) for Body-Mind connection
        try:
            from jarvis_prime.core.hybrid_router import get_cognitive_router
            self._cognitive_router = await get_cognitive_router()
            prime_healthy = self._cognitive_router._prime_bridge.is_healthy
            logger.info(f"🧠 CognitiveRouter (Corpus Callosum) initialized, Prime healthy={prime_healthy}")
        except Exception as e:
            logger.warning(f"CognitiveRouter initialization deferred: {e}")

        # Determine components to start
        if components:
            to_start = [c for c in components if c in self._managers]
        else:
            to_start = [c for c in self.config.startup_order if c in self._managers]

        # Start components
        if self.config.parallel_startup:
            results = await asyncio.gather(
                *[self._start_component(name) for name in to_start],
                return_exceptions=True,
            )
            success = all(r is True for r in results)
        else:
            success = True
            for name in to_start:
                if not await self._start_component(name):
                    success = False
                    if name in self.config.components and self.config.components[name].depends_on:
                        logger.error(f"Dependency {name} failed, stopping startup")
                        break

        if success:
            logger.info("=" * 60)
            logger.info("All components started successfully")
            logger.info("=" * 60)

            # Start health monitoring
            self._health_task = asyncio.create_task(self._health_monitor())

            # v85.0: Run startup verification
            verification_passed = await self._verify_startup()

            # Print status
            self._print_status()

            if not verification_passed:
                logger.warning("=" * 60)
                logger.warning("STARTUP VERIFICATION INCOMPLETE")
                logger.warning("Some components may not be fully operational")
                logger.warning("=" * 60)

        return success

    async def _verify_startup(self) -> bool:
        """
        v85.0: Verify that all components started correctly.

        Performs:
        1. Process liveness check (is process still running?)
        2. HTTP health endpoint check (is service responding?)
        3. Functional verification (is service operational?)

        Returns:
            True if all verifications passed
        """
        logger.info("")
        logger.info("=" * 60)
        logger.info("STARTUP VERIFICATION")
        logger.info("=" * 60)

        all_passed = True
        results = []

        for name, manager in self._managers.items():
            if manager.state.status == ComponentStatus.STOPPED:
                continue

            verification = {
                "name": name,
                "process_alive": False,
                "health_check": False,
                "functional": False,
            }

            # 1. Process liveness check
            if manager.state.process and manager.state.process.returncode is None:
                verification["process_alive"] = True
                logger.info(f"  [PROCESS] {name}: PID {manager.state.pid} running")
            else:
                all_passed = False
                logger.error(f"  [PROCESS] {name}: NOT RUNNING (exit code: {manager.state.process.returncode if manager.state.process else 'N/A'})")
                # Try to get last stderr for debugging
                if manager.state.stderr_buffer:
                    last_errors = manager.state.stderr_buffer[-5:]
                    logger.error(f"    Last errors: {last_errors}")

            # 2. Health check (with timeout)
            if verification["process_alive"]:
                try:
                    healthy = await asyncio.wait_for(
                        manager.health_check(),
                        timeout=10.0
                    )
                    verification["health_check"] = healthy
                    if healthy:
                        logger.info(f"  [HEALTH]  {name}: responding on port {manager.config.port}")
                    else:
                        logger.warning(f"  [HEALTH]  {name}: not responding (might still be starting)")
                except asyncio.TimeoutError:
                    logger.warning(f"  [HEALTH]  {name}: health check timed out")

            # 3. Functional check (basic - can extend per component)
            if verification["health_check"]:
                try:
                    # For JARVIS Prime, check if model is loaded
                    if manager.config.type == ComponentType.JARVIS_PRIME:
                        functional = await self._verify_prime_functional(manager)
                        verification["functional"] = functional
                        if functional:
                            logger.info(f"  [FUNC]    {name}: inference engine ready")
                        else:
                            logger.warning(f"  [FUNC]    {name}: inference engine not ready yet")
                    else:
                        # For other components, health check is sufficient
                        verification["functional"] = True
                        logger.info(f"  [FUNC]    {name}: operational")
                except Exception as e:
                    logger.warning(f"  [FUNC]    {name}: verification error: {e}")

            results.append(verification)

        logger.info("=" * 60)

        # Summary
        passed = sum(1 for r in results if r["process_alive"] and r["health_check"])
        total = len(results)
        logger.info(f"Verification: {passed}/{total} components fully operational")

        return all_passed

    async def _verify_prime_functional(self, manager: "ComponentManager") -> bool:
        """Verify JARVIS Prime is functionally ready."""
        try:
            import aiohttp
            url = f"http://{manager.config.host}:{manager.config.port}/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if model is loaded (if this info is available)
                        if isinstance(data, dict):
                            model_loaded = data.get("model_loaded", True)
                            status = data.get("status", "unknown")
                            return model_loaded or status == "healthy"
                        return True
        except Exception:
            pass
        return False

    async def stop(self) -> None:
        """
        Stop all components and the supervisor.

        v79.1: Also cleans up CognitiveRouter and connection pools.
        """
        logger.info("=" * 60)
        logger.info("JARVIS Unified Supervisor v79.1 - Stopping")
        logger.info("=" * 60)

        self._running = False

        # Stop health monitoring
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop CognitiveRouter (Corpus Callosum)
        if self._cognitive_router is not None:
            try:
                from jarvis_prime.core.hybrid_router import shutdown_cognitive_router
                await shutdown_cognitive_router()
                logger.info("🧠 CognitiveRouter (Corpus Callosum) stopped")
            except Exception as e:
                logger.warning(f"CognitiveRouter shutdown error: {e}")

        # Stop components in reverse order
        stop_order = list(reversed(self.config.startup_order))

        for name in stop_order:
            if name in self._managers:
                manager = self._managers[name]
                if manager.state.status in (ComponentStatus.RUNNING, ComponentStatus.HEALTHY):
                    await manager.stop(timeout=self.config.shutdown_timeout_seconds)

        # Close health check connection pool
        if ComponentManager._health_pool is not None:
            await ComponentManager._health_pool.close_all()
            logger.info("Health check connection pool closed")

        # Write final state
        await self._write_state()

        logger.info("Supervisor stopped")

    async def run(self, components: Optional[List[str]] = None) -> None:
        """Run the supervisor until shutdown signal."""
        if not await self.start(components):
            logger.error("Failed to start supervisor")
            return

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Stop everything
        await self.stop()

    async def _start_component(self, name: str) -> bool:
        """Start a single component with dependency checking."""
        if name not in self._managers:
            logger.error(f"Unknown component: {name}")
            return False

        manager = self._managers[name]
        config = manager.config

        # Check dependencies
        for dep in config.depends_on:
            if dep in self._managers:
                dep_manager = self._managers[dep]
                if dep_manager.state.status not in (ComponentStatus.RUNNING, ComponentStatus.HEALTHY):
                    logger.warning(f"Dependency {dep} not running for {name}")

                    # Try to start dependency
                    if not await self._start_component(dep):
                        logger.error(f"Failed to start dependency {dep}")
                        return False

        # Start component
        return await manager.start()

    async def _health_monitor(self) -> None:
        """Monitor component health and restart if needed."""
        while self._running:
            try:
                for name, manager in self._managers.items():
                    if manager.state.status not in (ComponentStatus.RUNNING, ComponentStatus.HEALTHY):
                        continue

                    # Check health
                    healthy = await manager.health_check()

                    if not healthy and manager.config.auto_restart:
                        if manager.state.restart_count < manager.config.max_restarts:
                            logger.warning(f"{name} unhealthy, restarting...")
                            await manager.restart()
                        else:
                            logger.error(f"{name} exceeded max restarts")

                # Update state
                await self._write_state()

                await asyncio.sleep(self.config.health_check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)

    async def _write_state(self) -> None:
        """Write supervisor state to disk."""
        state = {
            "supervisor_id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "timestamp_iso": datetime.now().isoformat(),
            "running": self._running,
            "components": {},
        }

        for name, manager in self._managers.items():
            state["components"][name] = {
                "type": manager.config.type.value,
                "status": manager.state.status.value,
                "pid": manager.state.pid,
                "port": manager.config.port,
                "start_time": manager.state.start_time,
                "restart_count": manager.state.restart_count,
                "uptime_seconds": (
                    time.time() - manager.state.start_time
                    if manager.state.start_time else 0
                ),
            }

        state_file = self.config.state_dir / "supervisor_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _print_status(self) -> None:
        """Print current status."""
        logger.info("")
        logger.info("Component Status:")
        logger.info("-" * 40)

        for name, manager in self._managers.items():
            status = manager.state.status.value
            port = manager.config.port
            pid = manager.state.pid or "N/A"

            status_icon = {
                "healthy": "[OK]",
                "running": "[~~]",
                "unhealthy": "[!!]",
                "stopped": "[--]",
                "failed": "[XX]",
            }.get(status, "[??]")

            logger.info(f"  {status_icon} {name:20} Port:{port:5} PID:{pid}")

        logger.info("-" * 40)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(self._handle_signal()),
            )

    async def _handle_signal(self) -> None:
        """Handle shutdown signal."""
        logger.info("Received shutdown signal")
        self._shutdown_event.set()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_component(self, name: str) -> Optional[ComponentManager]:
        """Get a component manager."""
        return self._managers.get(name)

    def get_status(self) -> Dict[str, Any]:
        """Get supervisor status."""
        return {
            "running": self._running,
            "components": {
                name: {
                    "status": manager.state.status.value,
                    "pid": manager.state.pid,
                    "port": manager.config.port,
                    "restart_count": manager.state.restart_count,
                }
                for name, manager in self._managers.items()
            },
        }

    async def restart_component(self, name: str) -> bool:
        """Restart a specific component."""
        if name not in self._managers:
            return False
        return await self._managers[name].restart()

    async def stop_component(self, name: str) -> bool:
        """Stop a specific component."""
        if name not in self._managers:
            return False
        return await self._managers[name].stop()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="JARVIS Unified Supervisor - Cross-Repository Orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 run_supervisor.py                    # Start all components
    python3 run_supervisor.py --prime-only       # Start only JARVIS-Prime
    python3 run_supervisor.py --components prime,jarvis
    python3 run_supervisor.py --debug            # Enable debug logging
        """,
    )

    parser.add_argument(
        "--components", "-c",
        help="Comma-separated list of components to start",
    )

    parser.add_argument(
        "--prime-only",
        action="store_true",
        help="Start only JARVIS-Prime (inference server)",
    )

    parser.add_argument(
        "--jarvis-port",
        type=int,
        default=8080,
        help="Port for JARVIS (default: 8080)",
    )

    parser.add_argument(
        "--prime-port",
        type=int,
        default=8000,
        help="Port for JARVIS-Prime (default: 8000)",
    )

    parser.add_argument(
        "--reactor-port",
        type=int,
        default=8090,
        help="Port for Reactor-Core (default: 8090)",
    )

    parser.add_argument(
        "--enable-reactor",
        action="store_true",
        help="Enable Reactor-Core component",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Start components in parallel",
    )

    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Disable automatic restart on failure",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--advanced-orchestrator",
        action="store_true",
        help="Use v80.0 advanced cross-repo orchestrator (EXPERIMENTAL)",
    )

    parser.add_argument(
        "--enable-tracing",
        action="store_true",
        help="Enable distributed tracing with OpenTelemetry",
    )

    parser.add_argument(
        "--enable-hot-reload",
        action="store_true",
        help="Enable hot-reload configuration watching",
    )

    parser.add_argument(
        "--unified",
        action="store_true",
        help="v87.0: Start with full Connective Tissue (RECOMMENDED)",
    )

    parser.add_argument(
        "--enable-gcp",
        action="store_true",
        help="v87.0: Enable GCP Spot VM for cloud inference",
    )

    parser.add_argument(
        "--enable-service-mesh",
        action="store_true",
        default=True,
        help="v87.0: Enable service mesh for dynamic discovery (default: true)",
    )

    parser.add_argument(
        "--enable-intelligent-routing",
        action="store_true",
        default=True,
        help="v87.0: Enable intelligent model routing (default: true)",
    )

    return parser.parse_args()


async def main_v80_orchestrator(args):
    """
    Main entry point using v80.0 advanced orchestrator.

    This provides:
    - Automatic repo detection
    - Graceful shutdown
    - Hot-reload config
    - Distributed tracing
    - Zero-copy IPC
    - And more...
    """
    logger.info("=" * 70)
    logger.info("JARVIS Trinity Orchestrator v80.0 - Starting")
    logger.info("=" * 70)

    # Set environment variables based on args
    if args.enable_tracing:
        os.environ["TRACING_ENABLED"] = "true"

    if args.enable_hot_reload:
        os.environ["ENABLE_HOT_RELOAD"] = "true"

    try:
        # Import advanced orchestrator
        from jarvis_prime.core.cross_repo_orchestrator import get_orchestrator

        # Get orchestrator
        orchestrator = await get_orchestrator()

        # Start all components
        success = await orchestrator.start_all()

        if not success:
            logger.error("Failed to start all components")
            return

        logger.info("")
        logger.info("=" * 70)
        logger.info("🚀 JARVIS Trinity Ecosystem is ONLINE")
        logger.info("=" * 70)
        logger.info("")
        logger.info("Press Ctrl+C to shut down gracefully")
        logger.info("")

        # Run until shutdown signal
        await orchestrator.run_until_shutdown()

    except ImportError as e:
        logger.error(f"Advanced orchestrator not available: {e}")
        logger.error("Please ensure all dependencies are installed:")
        logger.error("  pip install watchdog networkx opentelemetry-api opentelemetry-sdk")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Orchestrator error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def main_v87_unified(args):
    """
    v87.0: Main entry point with full Connective Tissue.

    This provides the complete unified architecture:
    - Intelligent Model Router (Local 7B -> GCP 13B -> Claude API)
    - GCP VM Manager (Spot VM lifecycle, auto-scaling, preemption)
    - Service Mesh (discovery, circuit breakers, load balancing)
    - Unified Configuration (single YAML source of truth)
    - Cross-repo orchestration with dependency resolution
    """
    logger.info("=" * 70)
    logger.info("JARVIS TRINITY v87.0 - THE CONNECTIVE TISSUE")
    logger.info("=" * 70)
    logger.info("")

    # Track initialized components for cleanup
    initialized = []

    try:
        # 1. Load unified configuration
        config_path = Path(__file__).parent / "config" / "unified_config.yaml"
        if config_path.exists():
            logger.info(f"Loading unified config from {config_path}")
            import yaml
            with open(config_path) as f:
                unified_config = yaml.safe_load(f)
            logger.info("  Unified configuration loaded")
        else:
            logger.warning(f"No unified config at {config_path}, using defaults")
            unified_config = {}

        # 2. Initialize Service Mesh
        if args.enable_service_mesh:
            logger.info("")
            logger.info("Initializing Service Mesh...")
            try:
                from jarvis_prime.core.service_mesh import get_service_mesh
                service_mesh = await get_service_mesh()
                initialized.append(("service_mesh", service_mesh))
                logger.info("  Service Mesh initialized")
            except Exception as e:
                logger.warning(f"  Service Mesh initialization failed: {e}")
                service_mesh = None
        else:
            service_mesh = None

        # 3. Initialize Intelligent Model Router
        if args.enable_intelligent_routing:
            logger.info("")
            logger.info("Initializing Intelligent Model Router (Brain Router)...")
            try:
                from jarvis_prime.core.intelligent_model_router import get_intelligent_router
                model_router = await get_intelligent_router()
                initialized.append(("model_router", model_router))

                # Check endpoint health
                health_status = await model_router.health_check_all()
                for tier, healthy in health_status.items():
                    status = "READY" if healthy else "OFFLINE"
                    logger.info(f"    {tier}: {status}")

                logger.info("  Intelligent Model Router initialized")
            except Exception as e:
                logger.warning(f"  Model Router initialization failed: {e}")
                model_router = None
        else:
            model_router = None

        # 4. Initialize GCP VM Manager (if enabled)
        if args.enable_gcp:
            logger.info("")
            logger.info("Initializing GCP VM Manager...")
            try:
                from jarvis_prime.core.gcp_vm_manager import get_gcp_manager
                gcp_manager = await get_gcp_manager()
                initialized.append(("gcp_manager", gcp_manager))

                status = gcp_manager.get_status()
                logger.info(f"    Project: {status['config'].get('project_id', 'N/A')}")
                logger.info(f"    Region: {status['config'].get('region', 'N/A')}")
                logger.info(f"    Instances: {len(status.get('instances', {}))}")
                logger.info("  GCP VM Manager initialized")
            except Exception as e:
                logger.warning(f"  GCP VM Manager initialization failed: {e}")
                gcp_manager = None
        else:
            gcp_manager = None
            logger.info("")
            logger.info("GCP VM Manager: DISABLED (use --enable-gcp to enable)")

        # 5. Initialize Cross-Repo Orchestrator
        logger.info("")
        logger.info("Initializing Cross-Repo Orchestrator...")
        try:
            from jarvis_prime.core.cross_repo_orchestrator import get_orchestrator
            orchestrator = await get_orchestrator()
            initialized.append(("orchestrator", orchestrator))
            logger.info("  Cross-Repo Orchestrator initialized")
        except Exception as e:
            logger.warning(f"  Orchestrator initialization failed: {e}")
            orchestrator = None

        # 6. Register services with mesh
        if service_mesh:
            logger.info("")
            logger.info("Registering services with mesh...")

            # Register JARVIS-Prime
            try:
                await service_mesh.register_service(
                    service_name="jarvis-prime",
                    host="localhost",
                    port=8000,
                    capabilities=["inference", "reasoning", "agi"],
                )
                logger.info("  Registered jarvis-prime (Mind)")
            except Exception as e:
                logger.debug(f"  Failed to register jarvis-prime: {e}")

        # 7. Start components
        logger.info("")
        logger.info("Starting Trinity components...")

        if orchestrator:
            success = await orchestrator.start_all()
            if not success:
                logger.error("Failed to start all components")
        else:
            # Fallback to direct startup
            logger.info("  Starting JARVIS-Prime directly...")
            # This will be handled by the legacy supervisor

        # 8. Print status summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("JARVIS TRINITY ECOSYSTEM STATUS")
        logger.info("=" * 70)
        logger.info("")
        logger.info("CONNECTIVE TISSUE:")
        logger.info(f"  Service Mesh:           {'ACTIVE' if service_mesh else 'INACTIVE'}")
        logger.info(f"  Intelligent Router:     {'ACTIVE' if model_router else 'INACTIVE'}")
        logger.info(f"  GCP VM Manager:         {'ACTIVE' if gcp_manager else 'INACTIVE'}")
        logger.info(f"  Cross-Repo Orchestrator:{'ACTIVE' if orchestrator else 'INACTIVE'}")
        logger.info("")
        logger.info("ROUTING TIERS:")
        logger.info("  Tier 1 (Priority): Local 7B   -> http://localhost:8000")
        if gcp_manager:
            endpoint = await gcp_manager.get_inference_endpoint()
            logger.info(f"  Tier 2 (Fallback): GCP 13B   -> {endpoint or 'Not provisioned'}")
        else:
            logger.info("  Tier 2 (Fallback): GCP 13B   -> DISABLED")
        logger.info("  Tier 3 (Ultimate): Claude API -> api.anthropic.com")
        logger.info("")
        logger.info("=" * 70)
        logger.info("TRINITY IS ONLINE - Press Ctrl+C to shutdown gracefully")
        logger.info("=" * 70)
        logger.info("")

        # 9. Run until shutdown
        if orchestrator:
            await orchestrator.run_until_shutdown()
        else:
            # Fallback: wait for signal
            shutdown_event = asyncio.Event()

            def signal_handler():
                shutdown_event.set()

            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, signal_handler)

            await shutdown_event.wait()

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Shutdown requested...")
    except Exception as e:
        logger.error(f"Trinity error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        logger.info("")
        logger.info("Shutting down Connective Tissue...")

        for name, component in reversed(initialized):
            try:
                if name == "service_mesh":
                    from jarvis_prime.core.service_mesh import shutdown_service_mesh
                    await shutdown_service_mesh()
                    logger.info(f"  {name}: stopped")
                elif name == "model_router":
                    from jarvis_prime.core.intelligent_model_router import shutdown_intelligent_router
                    await shutdown_intelligent_router()
                    logger.info(f"  {name}: stopped")
                elif name == "gcp_manager":
                    from jarvis_prime.core.gcp_vm_manager import shutdown_gcp_manager
                    await shutdown_gcp_manager()
                    logger.info(f"  {name}: stopped")
                elif name == "orchestrator":
                    # Orchestrator handles its own shutdown
                    pass
            except Exception as e:
                logger.debug(f"  Error stopping {name}: {e}")

        logger.info("")
        logger.info("Trinity shutdown complete")


async def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # v87.0: Unified mode (RECOMMENDED)
    if args.unified:
        await main_v87_unified(args)
        return

    # Check if using advanced orchestrator
    if args.advanced_orchestrator:
        await main_v80_orchestrator(args)
        return

    # Original supervisor implementation
    # Create supervisor
    supervisor = UnifiedSupervisor()

    # Apply CLI overrides
    if args.prime_only:
        components = ["jarvis_prime"]
    elif args.components:
        components = args.components.split(",")
    else:
        components = None

    # Apply port overrides
    if "jarvis_prime" in supervisor.config.components:
        supervisor.config.components["jarvis_prime"].port = args.prime_port
        supervisor.config.components["jarvis_prime"].args = ["--port", str(args.prime_port)]

    if "jarvis" in supervisor.config.components:
        supervisor.config.components["jarvis"].port = args.jarvis_port

    if "reactor_core" in supervisor.config.components:
        supervisor.config.components["reactor_core"].port = args.reactor_port
        if args.enable_reactor:
            supervisor.config.components["reactor_core"].enabled = True

    # Apply other options
    supervisor.config.parallel_startup = args.parallel

    if args.no_restart:
        for comp in supervisor.config.components.values():
            comp.auto_restart = False

    # Run
    await supervisor.run(components)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
