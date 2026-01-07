#!/usr/bin/env python3
"""
JARVIS Unified Supervisor - Cross-Repository Orchestration
=============================================================

v76.0 - Advanced AGI System Orchestrator

This supervisor connects and orchestrates all JARVIS ecosystem components:
- JARVIS (Body): Main orchestrator, computer use, action execution
- JARVIS-Prime (Mind): LLM inference, reasoning, cognitive processing
- Reactor-Core (Training): Model training, fine-tuning, deployment

USAGE:
    python3 run_supervisor.py                    # Start all components
    python3 run_supervisor.py --prime-only       # Start only JARVIS-Prime
    python3 run_supervisor.py --components prime,jarvis  # Specific components
    python3 run_supervisor.py --config config.yaml       # Custom config

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


class ComponentManager:
    """Manages lifecycle of a single component."""

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
        """Check component health."""
        if self.state.process is None or self.state.process.returncode is not None:
            self.state.status = ComponentStatus.STOPPED
            return False

        if not self.config.health_endpoint:
            return True

        try:
            import aiohttp

            url = f"http://{self.config.host}:{self.config.port}{self.config.health_endpoint}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        self.state.status = ComponentStatus.HEALTHY
                        self.state.consecutive_failures = 0
                        self.state.last_health_check = time.time()
                        return True

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
        """Build environment for component."""
        env = os.environ.copy()

        # Add custom env vars
        env.update(self.config.env)

        # Set Python path
        if self.config.repo_path:
            python_path = env.get("PYTHONPATH", "")
            if python_path:
                python_path = f"{self.config.repo_path}:{python_path}"
            else:
                python_path = str(self.config.repo_path)
            env["PYTHONPATH"] = python_path

        # Set component-specific vars
        env["JARVIS_COMPONENT"] = self.config.name
        env["JARVIS_COMPONENT_TYPE"] = self.config.type.value

        return env


# =============================================================================
# UNIFIED SUPERVISOR
# =============================================================================

class UnifiedSupervisor:
    """
    Unified supervisor for the JARVIS ecosystem.

    Orchestrates all components across repositories:
    - JARVIS (Body): macOS integration and action execution
    - JARVIS-Prime (Mind): LLM inference and reasoning
    - Reactor-Core (Training): Model training pipeline
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
        """Create default configuration."""
        config = SupervisorConfig()

        # Detect repository paths
        current_dir = Path(__file__).parent
        jarvis_prime_path = current_dir
        jarvis_path = current_dir.parent / "JARVIS"
        reactor_core_path = current_dir.parent / "Reactor-Core"

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

        Args:
            components: Specific components to start (None = all enabled)
        """
        logger.info("=" * 60)
        logger.info("JARVIS Unified Supervisor - Starting")
        logger.info("=" * 60)

        self._running = True

        # Setup signal handlers
        self._setup_signal_handlers()

        # Write supervisor state
        await self._write_state()

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

            # Print status
            self._print_status()

        return success

    async def stop(self) -> None:
        """Stop all components and the supervisor."""
        logger.info("=" * 60)
        logger.info("JARVIS Unified Supervisor - Stopping")
        logger.info("=" * 60)

        self._running = False

        # Stop health monitoring
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Stop components in reverse order
        stop_order = list(reversed(self.config.startup_order))

        for name in stop_order:
            if name in self._managers:
                manager = self._managers[name]
                if manager.state.status in (ComponentStatus.RUNNING, ComponentStatus.HEALTHY):
                    await manager.stop(timeout=self.config.shutdown_timeout_seconds)

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

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

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
