"""
JARVIS-Prime Cross-Repo Bridge
==============================

Connects JARVIS-Prime to the main JARVIS infrastructure tracking system.
Enables unified cost tracking, resource monitoring, and coordinated shutdowns.

Features:
- Reads/writes bridge state from shared ~/.jarvis/cross_repo/
- Reports inference metrics to main JARVIS instance
- Tracks local inference costs
- Coordinates with infrastructure orchestrator
"""

import asyncio
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from contextlib import asynccontextmanager

logger = logging.getLogger("jarvis-prime.bridge")

# ============================================================================
# Constants
# ============================================================================

BRIDGE_STATE_DIR = Path.home() / ".jarvis" / "cross_repo"
BRIDGE_STATE_FILE = BRIDGE_STATE_DIR / "bridge_state.json"
PRIME_STATE_FILE = BRIDGE_STATE_DIR / "jarvis_prime_state.json"
HEARTBEAT_INTERVAL = 30  # seconds
STALE_THRESHOLD = 120  # seconds - consider stale if no heartbeat

# v93.3: Intelligent Cross-Repo Connection Constants
STARTUP_GRACE_PERIOD = float(os.environ.get("JARVIS_STARTUP_GRACE_PERIOD", "120.0"))  # 2 min grace
INITIAL_RETRY_DELAY = float(os.environ.get("JARVIS_INITIAL_RETRY_DELAY", "2.0"))  # Start with 2s
MAX_RETRY_DELAY = float(os.environ.get("JARVIS_MAX_RETRY_DELAY", "30.0"))  # Cap at 30s
RETRY_BACKOFF_MULTIPLIER = float(os.environ.get("JARVIS_RETRY_BACKOFF", "1.5"))  # 1.5x each retry
MAX_STARTUP_RETRIES = int(os.environ.get("JARVIS_MAX_STARTUP_RETRIES", "10"))  # 10 retries during startup
AGGRESSIVE_RETRY_INTERVAL = float(os.environ.get("JARVIS_AGGRESSIVE_RETRY", "5.0"))  # 5s during startup

# PROJECT TRINITY: Unified command routing
TRINITY_DIR = Path.home() / ".jarvis" / "trinity"
TRINITY_COMMANDS_DIR = TRINITY_DIR / "commands"

# v93.0: Predictive Memory Defense - cross-repo signaling
MEMORY_PRESSURE_FILE = BRIDGE_STATE_DIR / "memory_pressure.json"
MEMORY_PRESSURE_CHECK_INTERVAL = 2.0  # Check every 2 seconds when active

# v100.0: Docker State Integration - coordinate with JARVIS Docker Manager
DOCKER_STATE_DIR = Path.home() / ".jarvis" / "trinity" / "docker"
DOCKER_STATE_FILE = DOCKER_STATE_DIR / "state.json"
DOCKER_EVENTS_FILE = DOCKER_STATE_DIR / "events.json"
DOCKER_CHECK_INTERVAL = 10.0  # Check Docker state every 10 seconds


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class InferenceMetrics:
    """Tracks inference statistics for cost estimation."""
    total_requests: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    model_name: str = "jarvis-prime"

    # Cost estimation (local inference is essentially free, but track for comparison)
    estimated_cost_usd: float = 0.0  # If using cloud API instead
    savings_vs_cloud_usd: float = 0.0  # Savings from using local model

    def record_inference(
        self,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
    ) -> None:
        """Record a single inference request."""
        self.total_requests += 1
        self.total_tokens_in += tokens_in
        self.total_tokens_out += tokens_out
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.total_requests

        # Estimate savings vs cloud API
        # Using approximate Claude API pricing as baseline
        # Input: $0.008/1K tokens, Output: $0.024/1K tokens (Claude 3 Haiku)
        cloud_input_cost = (tokens_in / 1000) * 0.008
        cloud_output_cost = (tokens_out / 1000) * 0.024
        self.estimated_cost_usd += cloud_input_cost + cloud_output_cost
        self.savings_vs_cloud_usd = self.estimated_cost_usd  # Local = $0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceMetrics":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PrimeState:
    """State of the JARVIS-Prime instance."""
    instance_id: str = ""
    started_at: str = ""
    last_heartbeat: str = ""
    status: str = "initializing"  # initializing, ready, busy, shutting_down, stopped
    model_loaded: bool = False
    model_path: str = ""
    endpoint: str = ""
    port: int = 8000
    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)

    # Cross-repo coordination
    connected_to_jarvis: bool = False
    jarvis_session_id: str = ""

    # v93.3: Enhanced Connection State Tracking
    connection_state: str = "initializing"  # initializing, connecting, connected, standalone, reconnecting
    connection_attempts: int = 0
    last_connection_attempt: str = ""
    connection_error: str = ""
    startup_phase: bool = True  # True during initial startup grace period

    # v93.0: Predictive Memory Defense integration
    memory_pressure_status: str = "normal"  # normal, elevated, critical, offload_active
    paused_by_memory_defense: bool = False  # True if SIGSTOP'd by main JARVIS
    last_memory_pressure_check: str = ""

    # v100.0: Docker State Integration
    docker_available: bool = False  # True if Docker is running and healthy
    docker_status: str = "unknown"  # unknown, starting, running, stopped, error
    docker_health_score: float = 0.0  # 0.0-1.0 health score
    last_docker_check: str = ""
    use_docker_inference: bool = False  # True if using Docker-based inference

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimeState":
        """Create from dictionary."""
        metrics_data = data.pop("metrics", {})
        metrics = InferenceMetrics.from_dict(metrics_data) if metrics_data else InferenceMetrics()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}, metrics=metrics)


# ============================================================================
# Cross-Repo Bridge
# ============================================================================

class CrossRepoBridge:
    """
    Manages cross-repo communication between JARVIS-Prime and main JARVIS.

    Features:
    - Heartbeat system for liveness detection
    - Metrics sharing for unified cost tracking
    - Coordinated startup/shutdown
    - Model status reporting
    """

    def __init__(
        self,
        instance_id: Optional[str] = None,
        port: int = 8000,
        auto_heartbeat: bool = True,
    ):
        """
        Initialize the cross-repo bridge.

        Args:
            instance_id: Unique ID for this instance (auto-generated if not provided)
            port: Port this instance is running on
            auto_heartbeat: Whether to start automatic heartbeat loop
        """
        self.instance_id = instance_id or f"prime-{os.getpid()}-{int(time.time())}"
        self.port = port
        self.auto_heartbeat = auto_heartbeat

        self.state = PrimeState(
            instance_id=self.instance_id,
            started_at=datetime.now().isoformat(),
            last_heartbeat=datetime.now().isoformat(),
            port=port,
            endpoint=f"http://localhost:{port}",
            connection_state="initializing",
            startup_phase=True,
        )

        self._heartbeat_task: Optional[asyncio.Task] = None
        self._connection_retry_task: Optional[asyncio.Task] = None
        self._startup_grace_task: Optional[asyncio.Task] = None
        self._shutdown_callbacks: List[Callable] = []
        self._initialized = False

        # v93.3: Intelligent Retry State
        self._startup_time = time.time()
        self._retry_delay = INITIAL_RETRY_DELAY
        self._connection_lock = asyncio.Lock()

        # Ensure state directory exists
        BRIDGE_STATE_DIR.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """
        v93.3: Initialize the bridge with intelligent retry logic.

        CRITICAL FIX: Uses startup-aware connection with:
        1. Immediate state write (even before JARVIS connection)
        2. Parallel startup grace period monitoring
        3. Intelligent exponential backoff retry
        4. Graceful degradation to standalone mode
        """
        if self._initialized:
            return

        logger.info(f"[v93.3] Initializing cross-repo bridge (instance={self.instance_id})")

        # Write initial state FIRST - ensures we're discoverable immediately
        self.state.connection_state = "connecting"
        await self._write_state()

        # Start startup grace period monitor (runs in background)
        self._startup_grace_task = asyncio.create_task(self._startup_grace_monitor())

        # Attempt initial JARVIS connection with retry
        connected = await self._connect_with_retry()

        if not connected:
            # Not connected yet - startup grace period will continue retrying
            logger.info(
                f"[v93.3] JARVIS not available yet, entering startup grace period "
                f"({STARTUP_GRACE_PERIOD}s) - will continue retrying in background"
            )
            self.state.connection_state = "standalone"
            await self._write_state()

        # Start heartbeat loop (this will also retry connection periodically)
        if self.auto_heartbeat:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._initialized = True
        logger.info(
            f"[v93.3] Cross-repo bridge initialized "
            f"(connected={self.state.connected_to_jarvis}, "
            f"state={self.state.connection_state})"
        )

    async def _connect_with_retry(self, max_retries: Optional[int] = None) -> bool:
        """
        v93.3: Attempt JARVIS connection with intelligent exponential backoff.

        Args:
            max_retries: Max retry attempts (None = use MAX_STARTUP_RETRIES during startup)

        Returns:
            True if connected, False if all retries exhausted
        """
        max_retries = max_retries or MAX_STARTUP_RETRIES
        retry_delay = INITIAL_RETRY_DELAY

        for attempt in range(1, max_retries + 1):
            async with self._connection_lock:
                self.state.connection_attempts += 1
                self.state.last_connection_attempt = datetime.now().isoformat()

                try:
                    connected = await self._try_jarvis_connection()
                    if connected:
                        self.state.connection_state = "connected"
                        self.state.connection_error = ""
                        await self._write_state()
                        logger.info(
                            f"[v93.3] Connected to JARVIS on attempt {attempt} "
                            f"(session={self.state.jarvis_session_id[:8] if self.state.jarvis_session_id else 'N/A'}...)"
                        )
                        return True

                except Exception as e:
                    self.state.connection_error = str(e)
                    logger.debug(f"[v93.3] Connection attempt {attempt} failed: {e}")

                # Not connected - wait before retry with exponential backoff
                if attempt < max_retries:
                    logger.debug(
                        f"[v93.3] JARVIS not found, retry {attempt}/{max_retries} "
                        f"in {retry_delay:.1f}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * RETRY_BACKOFF_MULTIPLIER, MAX_RETRY_DELAY)

        return False

    async def _try_jarvis_connection(self) -> bool:
        """
        v93.3: Single attempt to connect to JARVIS.

        Returns:
            True if JARVIS state exists and is fresh, False otherwise
        """
        jarvis_state = await self.get_jarvis_state()

        if not jarvis_state:
            return False

        # Check if JARVIS state is recent (not stale)
        last_update = jarvis_state.get("last_update", "")
        if not last_update:
            return False

        try:
            update_time = datetime.fromisoformat(last_update)
            age_seconds = (datetime.now() - update_time).total_seconds()

            # During startup grace period, be more lenient with stale threshold
            effective_threshold = STALE_THRESHOLD
            if self.state.startup_phase:
                # Allow 5x longer stale threshold during startup
                effective_threshold = STALE_THRESHOLD * 5

            if age_seconds < effective_threshold:
                self.state.connected_to_jarvis = True
                self.state.jarvis_session_id = jarvis_state.get("session_id", "")
                await self.notify_jarvis("connected", {
                    "port": self.port,
                    "model_loaded": self.state.model_loaded,
                    "connection_attempt": self.state.connection_attempts,
                })
                return True

        except (ValueError, TypeError) as e:
            logger.debug(f"[v93.3] Failed to parse JARVIS state timestamp: {e}")

        return False

    async def _startup_grace_monitor(self) -> None:
        """
        v93.3: Monitor startup grace period and transition to normal mode.

        During startup grace period:
        - More aggressive retry interval (AGGRESSIVE_RETRY_INTERVAL)
        - Continues attempting connection in background
        - After grace period, transitions to normal retry interval
        """
        grace_end_time = self._startup_time + STARTUP_GRACE_PERIOD
        aggressive_retries = 0

        while True:
            try:
                current_time = time.time()
                remaining_grace = grace_end_time - current_time

                if remaining_grace <= 0:
                    # Grace period ended
                    self.state.startup_phase = False
                    await self._write_state()
                    logger.info(
                        f"[v93.3] Startup grace period ended - "
                        f"connected={self.state.connected_to_jarvis}, "
                        f"total_attempts={self.state.connection_attempts}"
                    )
                    break

                # During grace period, aggressively retry if not connected
                if not self.state.connected_to_jarvis:
                    aggressive_retries += 1
                    logger.debug(
                        f"[v93.3] Startup grace retry {aggressive_retries} "
                        f"(remaining={remaining_grace:.1f}s)"
                    )
                    connected = await self._try_jarvis_connection()
                    if connected:
                        self.state.connection_state = "connected"
                        await self._write_state()
                        logger.info(
                            f"[v93.3] Connected to JARVIS during startup grace "
                            f"(after {aggressive_retries} aggressive retries)"
                        )
                        # Continue monitoring in case we need to reconnect
                    else:
                        self.state.connection_state = "standalone"
                        await self._write_state()

                await asyncio.sleep(AGGRESSIVE_RETRY_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[v93.3] Startup grace monitor error: {e}")
                await asyncio.sleep(AGGRESSIVE_RETRY_INTERVAL)

    async def shutdown(self) -> None:
        """
        v93.3: Gracefully shutdown the bridge with proper task cleanup.
        """
        logger.info("[v93.3] Shutting down cross-repo bridge...")

        # Update state
        self.state.status = "shutting_down"
        self.state.connection_state = "disconnecting"
        await self._write_state()

        # v93.3: Cancel all background tasks
        tasks_to_cancel = [
            ("heartbeat", self._heartbeat_task),
            ("startup_grace", self._startup_grace_task),
            ("connection_retry", self._connection_retry_task),
        ]

        for task_name, task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as e:
                    logger.debug(f"[v93.3] Error cancelling {task_name} task: {e}")

        # Run shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.warning(f"Shutdown callback error: {e}")

        # Notify JARVIS of shutdown if connected
        if self.state.connected_to_jarvis:
            try:
                await self.notify_jarvis("disconnected", {
                    "reason": "shutdown",
                    "uptime_seconds": time.time() - self._startup_time,
                })
            except Exception as e:
                logger.debug(f"[v93.3] Failed to notify JARVIS of shutdown: {e}")

        # Mark as stopped
        self.state.status = "stopped"
        self.state.connection_state = "disconnected"
        await self._write_state()

        logger.info("[v93.3] Cross-repo bridge shutdown complete")

    def on_shutdown(self, callback: Callable) -> None:
        """Register a callback to run on shutdown."""
        self._shutdown_callbacks.append(callback)

    def update_model_status(
        self,
        loaded: bool,
        model_path: str = "",
    ) -> None:
        """Update model loading status."""
        self.state.model_loaded = loaded
        self.state.model_path = model_path
        self.state.status = "ready" if loaded else "initializing"
        asyncio.create_task(self._write_state())

    def record_inference(
        self,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
    ) -> None:
        """Record an inference request for metrics tracking."""
        self.state.metrics.record_inference(tokens_in, tokens_out, latency_ms)
        # Don't write state on every inference - too expensive
        # Heartbeat will sync periodically

    def get_metrics(self) -> Dict[str, Any]:
        """Get current inference metrics."""
        return self.state.metrics.to_dict()

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary."""
        metrics = self.state.metrics
        return {
            "instance_id": self.instance_id,
            "total_requests": metrics.total_requests,
            "total_tokens": metrics.total_tokens_in + metrics.total_tokens_out,
            "local_cost_usd": 0.0,  # Local inference is free
            "cloud_equivalent_cost_usd": metrics.estimated_cost_usd,
            "savings_usd": metrics.savings_vs_cloud_usd,
            "savings_percent": 100.0 if metrics.total_requests > 0 else 0.0,
        }

    async def get_jarvis_state(self) -> Optional[Dict[str, Any]]:
        """Read main JARVIS bridge state."""
        try:
            if BRIDGE_STATE_FILE.exists():
                content = BRIDGE_STATE_FILE.read_text()
                return json.loads(content)
        except Exception as e:
            logger.warning(f"Failed to read JARVIS state: {e}")
        return None

    async def notify_jarvis(
        self,
        event: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send an event notification to main JARVIS.

        Args:
            event: Event type (e.g., "startup", "ready", "inference", "shutdown")
            data: Optional event data

        Returns:
            True if notification was written successfully
        """
        try:
            event_file = BRIDGE_STATE_DIR / "prime_events.json"

            events = []
            if event_file.exists():
                try:
                    events = json.loads(event_file.read_text())
                except json.JSONDecodeError:
                    events = []

            # Add new event
            events.append({
                "timestamp": datetime.now().isoformat(),
                "instance_id": self.instance_id,
                "event": event,
                "data": data or {},
            })

            # Keep only last 100 events
            events = events[-100:]

            event_file.write_text(json.dumps(events, indent=2))
            return True

        except Exception as e:
            logger.warning(f"Failed to notify JARVIS: {e}")
            return False

    # =========================================================================
    # PROJECT TRINITY: Direct command routing to JARVIS Body
    # =========================================================================

    async def send_trinity_command(
        self,
        intent: str,
        payload: Dict[str, Any],
        priority: int = 5,
        requires_ack: bool = True,
        ttl_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Send a Trinity command directly to JARVIS Body.

        This is the J-Prime (Mind) -> JARVIS (Body) command pathway through
        the Trinity file-based transport system.

        Args:
            intent: Command intent (e.g., "start_surveillance", "bring_back_window")
            payload: Command payload data
            priority: Command priority (1-10, lower is higher)
            requires_ack: Whether to expect ACK response
            ttl_seconds: Time-to-live for command

        Returns:
            Dict with success status and command_id
        """
        import uuid

        try:
            # Ensure Trinity directory exists
            TRINITY_COMMANDS_DIR.mkdir(parents=True, exist_ok=True)

            command_id = str(uuid.uuid4())
            timestamp = time.time()

            command = {
                "id": command_id,
                "timestamp": timestamp,
                "source": "j_prime",
                "intent": intent,
                "payload": payload,
                "metadata": {
                    "prime_instance_id": self.instance_id,
                    "model_loaded": self.state.model_loaded,
                },
                "target": "jarvis_body",
                "priority": priority,
                "requires_ack": requires_ack,
                "ttl_seconds": ttl_seconds,
            }

            # Write command file
            filename = f"{int(timestamp * 1000)}_{command_id}.json"
            filepath = TRINITY_COMMANDS_DIR / filename

            with open(filepath, "w") as f:
                json.dump(command, f, indent=2)

            logger.info(f"[Trinity] J-Prime sent command: {intent} (id={command_id[:8]})")
            return {"success": True, "command_id": command_id}

        except Exception as e:
            logger.error(f"[Trinity] Failed to send command: {e}")
            return {"success": False, "error": str(e)}

    async def start_surveillance(
        self,
        app_name: str,
        trigger_text: str,
        all_spaces: bool = True,
        max_duration: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Request JARVIS to start surveillance on an app.

        This is a cognitive decision from J-Prime (Mind) to activate
        the Ghost Monitor Protocol in JARVIS (Body).
        """
        return await self.send_trinity_command(
            intent="start_surveillance",
            payload={
                "app_name": app_name,
                "trigger_text": trigger_text,
                "all_spaces": all_spaces,
                "max_duration": max_duration,
            },
        )

    async def stop_surveillance(self, app_name: Optional[str] = None) -> Dict[str, Any]:
        """Request JARVIS to stop surveillance."""
        return await self.send_trinity_command(
            intent="stop_surveillance",
            payload={"app_name": app_name},
        )

    async def bring_back_windows(self, app_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Request JARVIS to bring back windows from Ghost Display.

        This is a cognitive decision from J-Prime to restore user visibility.
        """
        return await self.send_trinity_command(
            intent="bring_back_window",
            payload={"app_name": app_name},
        )

    async def exile_window(
        self,
        app_name: str,
        window_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Request JARVIS to exile a window to Ghost Display."""
        return await self.send_trinity_command(
            intent="exile_window",
            payload={
                "app_name": app_name,
                "window_title": window_title,
            },
        )

    async def freeze_app(self, app_name: str, reason: str = "") -> Dict[str, Any]:
        """Request JARVIS to freeze an app (SIGSTOP)."""
        return await self.send_trinity_command(
            intent="freeze_app",
            payload={"app_name": app_name, "reason": reason},
        )

    async def thaw_app(self, app_name: str) -> Dict[str, Any]:
        """Request JARVIS to thaw a frozen app (SIGCONT)."""
        return await self.send_trinity_command(
            intent="thaw_app",
            payload={"app_name": app_name},
        )

    # =========================================================================
    # v93.0: Predictive Memory Defense Integration
    # =========================================================================

    async def check_memory_pressure(self) -> Optional[Dict[str, Any]]:
        """
        v93.0: Check if main JARVIS has signaled memory pressure.

        Reads from ~/.jarvis/cross_repo/memory_pressure.json for signals.
        Returns pressure info if active, None otherwise.
        """
        try:
            if not MEMORY_PRESSURE_FILE.exists():
                self.state.memory_pressure_status = "normal"
                return None

            content = MEMORY_PRESSURE_FILE.read_text()
            pressure_data = json.loads(content)

            # Check if signal is still valid (not expired)
            timestamp = pressure_data.get("timestamp", 0)
            if time.time() - timestamp > 30:  # 30s TTL
                self.state.memory_pressure_status = "normal"
                return None

            status = pressure_data.get("status", "normal")
            self.state.memory_pressure_status = status
            self.state.last_memory_pressure_check = datetime.now().isoformat()

            return pressure_data

        except Exception as e:
            logger.warning(f"[v93.0] Error checking memory pressure: {e}")
            return None

    async def report_memory_defense_status(
        self,
        is_paused: bool,
        reason: str = "",
    ) -> None:
        """
        v93.0: Report our status back to main JARVIS.

        Called when we're paused/resumed by memory defense.
        """
        self.state.paused_by_memory_defense = is_paused
        await self.notify_jarvis(
            event="memory_defense_status",
            data={
                "is_paused": is_paused,
                "reason": reason,
                "model_loaded": self.state.model_loaded,
                "instance_id": self.instance_id,
            },
        )
        await self._write_state()

    async def start_memory_pressure_monitor(self) -> asyncio.Task:
        """
        v93.0: Start background task to monitor memory pressure signals.

        Returns the task so it can be cancelled on shutdown.
        """
        return asyncio.create_task(self._memory_pressure_monitor_loop())

    async def _memory_pressure_monitor_loop(self) -> None:
        """
        v93.0: Background loop to respond to memory pressure signals.

        Checks for:
        - Emergency offload requests (pause self via SIGSTOP awareness)
        - Memory pressure elevation warnings
        - Offload complete notifications
        """
        while True:
            try:
                pressure_data = await self.check_memory_pressure()

                if pressure_data:
                    status = pressure_data.get("status", "normal")
                    action = pressure_data.get("action")

                    if status == "offload_active" and action == "pause":
                        # Main JARVIS is pausing local processes
                        if not self.state.paused_by_memory_defense:
                            logger.warning(
                                "[v93.0] Memory pressure EMERGENCY - main JARVIS pausing processes"
                            )
                            await self.report_memory_defense_status(
                                is_paused=True,
                                reason="emergency_offload_active",
                            )

                    elif status == "offload_active" and action == "terminate":
                        # We're being asked to terminate (GCP ready)
                        logger.warning(
                            "[v93.0] Memory pressure TERMINATION request - GCP VM ready"
                        )
                        await self.notify_jarvis(
                            event="memory_defense_terminating",
                            data={"instance_id": self.instance_id},
                        )
                        # Initiate graceful shutdown
                        await self.shutdown()
                        return

                    elif status == "normal" and self.state.paused_by_memory_defense:
                        # We were paused but pressure has normalized
                        logger.info(
                            "[v93.0] Memory pressure normalized - resuming operations"
                        )
                        await self.report_memory_defense_status(
                            is_paused=False,
                            reason="pressure_normalized",
                        )

                    elif status == "elevated":
                        # Warning: memory getting high
                        if self.state.memory_pressure_status != "elevated":
                            logger.warning(
                                f"[v93.0] Memory pressure elevated: "
                                f"{pressure_data.get('used_percent', 0):.1f}%"
                            )

                # Check more frequently if in elevated/critical state
                if self.state.memory_pressure_status in ("elevated", "critical", "offload_active"):
                    await asyncio.sleep(MEMORY_PRESSURE_CHECK_INTERVAL)
                else:
                    await asyncio.sleep(HEARTBEAT_INTERVAL / 2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[v93.0] Memory pressure monitor error: {e}")
                await asyncio.sleep(5.0)

    def get_memory_defense_status(self) -> Dict[str, Any]:
        """v93.0: Get current memory defense status."""
        return {
            "memory_pressure_status": self.state.memory_pressure_status,
            "paused_by_memory_defense": self.state.paused_by_memory_defense,
            "last_check": self.state.last_memory_pressure_check,
            "connected_to_jarvis": self.state.connected_to_jarvis,
        }

    # =========================================================================
    # Docker State Integration (v100.0)
    # =========================================================================

    async def check_docker_state(self) -> Optional[Dict[str, Any]]:
        """
        v100.0: Check Docker state from JARVIS Docker Manager.

        Reads state from ~/.jarvis/trinity/docker/state.json which is
        maintained by the Docker Daemon Manager in JARVIS-AI-Agent.

        Returns:
            Docker state dict or None if unavailable
        """
        try:
            if not DOCKER_STATE_FILE.exists():
                return None

            state_data = json.loads(DOCKER_STATE_FILE.read_text())
            docker_state = state_data.get("state", {})

            # Update our local tracking
            self.state.docker_available = docker_state.get("healthy", False)
            self.state.docker_status = docker_state.get("status", "unknown")
            self.state.docker_health_score = docker_state.get("health_score", 0.0)
            self.state.last_docker_check = datetime.now().isoformat()

            return docker_state

        except Exception as e:
            logger.debug(f"[v100.0] Failed to read Docker state: {e}")
            return None

    async def is_docker_available(self) -> bool:
        """
        v100.0: Check if Docker is available for container-based operations.

        Used to determine if we can use Docker-based inference or need
        to fall back to other methods.
        """
        docker_state = await self.check_docker_state()
        return docker_state is not None and docker_state.get("healthy", False)

    async def get_docker_events(self, since_timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        v100.0: Get recent Docker events from the event log.

        Args:
            since_timestamp: Only get events after this timestamp

        Returns:
            List of Docker events
        """
        try:
            if not DOCKER_EVENTS_FILE.exists():
                return []

            events = json.loads(DOCKER_EVENTS_FILE.read_text())
            if not isinstance(events, list):
                return []

            if since_timestamp:
                events = [e for e in events if e.get("timestamp", 0) > since_timestamp]

            return events

        except Exception as e:
            logger.debug(f"[v100.0] Failed to read Docker events: {e}")
            return []

    async def request_docker_start(self) -> bool:
        """
        v100.0: Request JARVIS to start Docker daemon.

        Sends a request through the Trinity protocol for Docker to be started.
        """
        try:
            # Write request to requests directory
            request_dir = DOCKER_STATE_DIR / "requests"
            request_dir.mkdir(parents=True, exist_ok=True)

            import uuid
            request_id = str(uuid.uuid4())
            request_file = request_dir / f"{request_id}.json"

            request = {
                "request_id": request_id,
                "request_type": "docker.request.start",
                "source": "jarvis_prime",
                "timestamp": time.time(),
                "payload": {},
            }

            request_file.write_text(json.dumps(request, indent=2))

            # Wait for response
            response_dir = DOCKER_STATE_DIR / "responses"
            response_file = response_dir / f"{request_id}.json"

            start_time = time.time()
            while (time.time() - start_time) < 60.0:  # 60s timeout
                if response_file.exists():
                    response = json.loads(response_file.read_text())
                    response_file.unlink()
                    return response.get("success", False)
                await asyncio.sleep(1.0)

            logger.warning("[v100.0] Docker start request timed out")
            return False

        except Exception as e:
            logger.error(f"[v100.0] Failed to request Docker start: {e}")
            return False

    async def start_docker_monitor(self) -> asyncio.Task:
        """
        v100.0: Start background task to monitor Docker state.

        Returns the task so it can be cancelled on shutdown.
        """
        return asyncio.create_task(self._docker_monitor_loop())

    async def _docker_monitor_loop(self) -> None:
        """
        v100.0: Background loop to monitor Docker state changes.

        Updates local state and can trigger routing tier changes
        based on Docker availability.
        """
        last_docker_available = None

        while True:
            try:
                docker_state = await self.check_docker_state()
                current_available = docker_state.get("healthy", False) if docker_state else False

                # Detect state changes
                if last_docker_available is not None and current_available != last_docker_available:
                    if current_available:
                        logger.info("[v100.0] Docker became available - container inference enabled")
                        await self.notify_jarvis(
                            event="docker_available",
                            data={"instance_id": self.instance_id},
                        )
                    else:
                        logger.warning("[v100.0] Docker became unavailable - falling back to local inference")
                        await self.notify_jarvis(
                            event="docker_unavailable",
                            data={"instance_id": self.instance_id},
                        )

                last_docker_available = current_available

                await asyncio.sleep(DOCKER_CHECK_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[v100.0] Docker monitor error: {e}")
                await asyncio.sleep(10.0)

    # =========================================================================
    # Readiness State Integration (v2.0)
    # =========================================================================

    async def check_jarvis_body_readiness(self) -> Optional[Dict[str, Any]]:
        """
        v2.0: Check jarvis-body readiness state from Trinity Protocol.

        Reads state from ~/.jarvis/trinity/state/jarvis-body_readiness.json
        which is maintained by the ReadinessStateManager in JARVIS-AI-Agent.

        Returns:
            Readiness state dict or None if unavailable
        """
        try:
            readiness_dir = Path.home() / ".jarvis" / "trinity" / "state"
            readiness_file = readiness_dir / "jarvis-body_readiness.json"

            if not readiness_file.exists():
                return None

            state_data = json.loads(readiness_file.read_text())

            # Check staleness (>30s is stale)
            timestamp = state_data.get("timestamp", 0)
            age = time.time() - timestamp
            if age > 30:
                logger.debug(f"[v2.0] jarvis-body readiness state is stale ({age:.0f}s)")
                return None

            return state_data

        except Exception as e:
            logger.debug(f"[v2.0] Failed to read jarvis-body readiness: {e}")
            return None

    async def is_jarvis_body_ready(self) -> bool:
        """
        v2.0: Check if jarvis-body is ready to accept requests.

        This enables JARVIS-Prime to wait for jarvis-body readiness
        before attempting to communicate with it.
        """
        state = await self.check_jarvis_body_readiness()
        if state is None:
            return False
        return state.get("is_ready", False)

    async def wait_for_jarvis_body(
        self,
        timeout: float = 120.0,
        check_interval: float = 2.0,
    ) -> bool:
        """
        v2.0: Wait for jarvis-body to become ready.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: How often to check

        Returns:
            True if jarvis-body became ready, False on timeout
        """
        start_time = time.time()
        logged_waiting = False

        while (time.time() - start_time) < timeout:
            if await self.is_jarvis_body_ready():
                if logged_waiting:
                    elapsed = time.time() - start_time
                    logger.info(f"[v2.0] jarvis-body is now ready ({elapsed:.1f}s wait)")
                return True

            if not logged_waiting:
                logger.info("[v2.0] Waiting for jarvis-body to become ready...")
                logged_waiting = True

            await asyncio.sleep(check_interval)

        logger.warning(f"[v2.0] Timeout waiting for jarvis-body readiness ({timeout}s)")
        return False

    def get_docker_status(self) -> Dict[str, Any]:
        """v100.0: Get current Docker status."""
        return {
            "docker_available": self.state.docker_available,
            "docker_status": self.state.docker_status,
            "docker_health_score": self.state.docker_health_score,
            "last_check": self.state.last_docker_check,
            "use_docker_inference": self.state.use_docker_inference,
        }

    async def create_ghost_display(self) -> Dict[str, Any]:
        """Request JARVIS to create a Ghost Display (virtual display)."""
        return await self.send_trinity_command(
            intent="create_ghost_display",
            payload={},
        )

    async def execute_plan(
        self,
        plan_id: str,
        steps: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Request JARVIS to execute a multi-step plan.

        This is the primary cognitive output from J-Prime - a reasoned
        plan of actions for JARVIS to execute.
        """
        return await self.send_trinity_command(
            intent="execute_plan",
            payload={
                "plan_id": plan_id,
                "steps": steps,
                "context": context or {},
            },
            priority=2,  # High priority for plans
        )

    async def _check_jarvis_connection(self) -> None:
        """
        v93.3: DEPRECATED - Use _try_jarvis_connection instead.
        Kept for backward compatibility.
        """
        connected = await self._try_jarvis_connection()
        if not connected:
            self.state.connected_to_jarvis = False
            if not self.state.startup_phase:
                # Only log standalone message after startup grace period
                logger.info("[v93.3] Main JARVIS not detected - running standalone")

    async def _write_state(self) -> None:
        """Write current state to shared file with atomic write."""
        try:
            self.state.last_heartbeat = datetime.now().isoformat()
            # v93.3: Atomic write to prevent partial state reads
            temp_file = PRIME_STATE_FILE.with_suffix(".tmp")
            temp_file.write_text(json.dumps(self.state.to_dict(), indent=2))
            temp_file.replace(PRIME_STATE_FILE)
        except Exception as e:
            logger.warning(f"Failed to write state: {e}")

    async def _send_registry_heartbeat(self) -> None:
        """
        v93.14: Send heartbeat to JARVIS service registry for cross-repo discovery.

        This writes directly to the shared service registry file at
        ~/.jarvis/registry/services.json to keep jarvis-prime alive in the
        registry without requiring the JARVIS supervisor to do health checks.

        The service registry uses file-based locking (fcntl) for thread safety.
        """
        import fcntl

        registry_dir = Path.home() / ".jarvis" / "registry"
        registry_file = registry_dir / "services.json"

        try:
            # Ensure directory exists
            registry_dir.mkdir(parents=True, exist_ok=True)

            # Read current registry
            services = {}
            if registry_file.exists():
                try:
                    # Open with shared lock for reading
                    with open(registry_file, 'r') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        try:
                            services = json.load(f)
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except (json.JSONDecodeError, IOError) as e:
                    logger.debug(f"[v93.14] Registry read error: {e}")
                    services = {}

            # Update or add jarvis-prime entry
            service_name = "jarvis-prime"
            current_time = time.time()

            if service_name in services:
                # Update existing entry
                services[service_name]["last_heartbeat"] = current_time
                services[service_name]["status"] = "healthy" if self.state.model_loaded else "starting"
                services[service_name]["metadata"] = {
                    "model_loaded": self.state.model_loaded,
                    "model_path": self.state.model_path,
                    "connection_state": self.state.connection_state,
                    "instance_id": self.instance_id,
                }
            else:
                # Register new entry
                # v93.16: Include process_start_time for PID reuse detection
                try:
                    import psutil
                    process_start_time = psutil.Process(os.getpid()).create_time()
                except Exception:
                    process_start_time = time.time()

                services[service_name] = {
                    "service_name": service_name,
                    "pid": os.getpid(),
                    "port": self.port,
                    "host": "localhost",
                    "health_endpoint": "/health",
                    "status": "healthy" if self.state.model_loaded else "starting",
                    "registered_at": current_time,
                    "last_heartbeat": current_time,
                    "process_start_time": process_start_time,  # v93.16: PID reuse detection
                    "metadata": {
                        "model_loaded": self.state.model_loaded,
                        "model_path": self.state.model_path,
                        "connection_state": self.state.connection_state,
                        "instance_id": self.instance_id,
                        "registered_by": "cross_repo_bridge",
                    },
                }
                logger.info(f"[v93.14] Registered jarvis-prime with JARVIS service registry")

            # Write back with exclusive lock (atomic)
            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(services, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Atomic rename
            temp_file.replace(registry_file)

        except Exception as e:
            # Non-fatal - log and continue
            logger.debug(f"[v93.14] Registry heartbeat error: {e}")

    async def _heartbeat_loop(self) -> None:
        """
        v95.0: Enterprise-grade heartbeat loop with fire-and-forget pattern.

        Features:
        - Fire-and-forget heartbeats (non-blocking)
        - Timeout protection on all I/O operations
        - Intelligent JARVIS reconnection with adaptive backoff
        - Connection state tracking
        - Handles both connected and standalone modes
        - Cross-repo service registry discovery
        - Self-healing on persistent failures
        """
        reconnect_delay = HEARTBEAT_INTERVAL
        consecutive_failures = 0
        heartbeat_timeout = 5.0  # Max time for heartbeat operations
        total_heartbeats = 0

        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                total_heartbeats += 1

                # v95.0: Fire-and-forget state write with timeout protection
                try:
                    await asyncio.wait_for(
                        self._write_state(),
                        timeout=heartbeat_timeout
                    )
                except asyncio.TimeoutError:
                    logger.debug("[v95.0] State write timed out (non-fatal)")
                except Exception as e:
                    logger.debug(f"[v95.0] State write error (non-fatal): {e}")

                # v95.0: Fire-and-forget registry heartbeat with timeout
                try:
                    await asyncio.wait_for(
                        self._send_registry_heartbeat(),
                        timeout=heartbeat_timeout
                    )
                except asyncio.TimeoutError:
                    logger.debug("[v95.0] Registry heartbeat timed out (non-fatal)")
                except Exception as e:
                    logger.debug(f"[v95.0] Registry heartbeat error (non-fatal): {e}")

                # v93.3: Intelligent reconnection logic
                if not self.state.connected_to_jarvis:
                    # Attempt reconnection
                    self.state.connection_state = "reconnecting"

                    try:
                        connected = await asyncio.wait_for(
                            self._try_jarvis_connection(),
                            timeout=10.0  # Connection attempt timeout
                        )
                    except asyncio.TimeoutError:
                        connected = False
                        logger.debug("[v95.0] JARVIS connection attempt timed out")

                    if connected:
                        self.state.connection_state = "connected"
                        consecutive_failures = 0
                        reconnect_delay = HEARTBEAT_INTERVAL
                        logger.info(
                            f"[v93.3] Reconnected to JARVIS "
                            f"(session={self.state.jarvis_session_id[:8] if self.state.jarvis_session_id else 'N/A'}...)"
                        )
                    else:
                        consecutive_failures += 1
                        self.state.connection_state = "standalone"

                        # v95.0: Adaptive exponential backoff with jitter
                        if consecutive_failures > 3:
                            jitter = (hash(time.time()) % 100) / 100.0 * 0.4 + 0.8  # 0.8-1.2
                            reconnect_delay = min(
                                reconnect_delay * RETRY_BACKOFF_MULTIPLIER * jitter,
                                MAX_RETRY_DELAY * 2
                            )

                        if consecutive_failures % 5 == 0:
                            logger.debug(
                                f"[v93.3] JARVIS reconnection attempt {consecutive_failures} failed, "
                                f"next retry in {reconnect_delay:.1f}s"
                            )
                else:
                    # Verify JARVIS is still alive with timeout
                    try:
                        jarvis_state = await asyncio.wait_for(
                            self.get_jarvis_state(),
                            timeout=heartbeat_timeout
                        )
                    except asyncio.TimeoutError:
                        jarvis_state = None
                        logger.debug("[v95.0] JARVIS state check timed out")

                    if jarvis_state:
                        last_update = jarvis_state.get("last_update", "")
                        if last_update:
                            try:
                                update_time = datetime.fromisoformat(last_update)
                                age_seconds = (datetime.now() - update_time).total_seconds()

                                if age_seconds > STALE_THRESHOLD:
                                    # JARVIS went stale
                                    logger.warning(
                                        f"[v93.3] JARVIS connection lost "
                                        f"(last heartbeat {age_seconds:.0f}s ago)"
                                    )
                                    self.state.connected_to_jarvis = False
                                    self.state.connection_state = "reconnecting"
                            except (ValueError, TypeError):
                                pass
                    else:
                        # JARVIS state file disappeared
                        logger.warning("[v93.3] JARVIS state file not found - connection lost")
                        self.state.connected_to_jarvis = False
                        self.state.connection_state = "reconnecting"

                # v95.0: Log periodic heartbeat summary
                if total_heartbeats % 100 == 0:
                    logger.debug(
                        f"[v95.0] CrossRepoBridge heartbeat #{total_heartbeats} "
                        f"(connected={self.state.connected_to_jarvis})"
                    )

            except asyncio.CancelledError:
                logger.debug("[v95.0] CrossRepoBridge heartbeat loop cancelled")
                break
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures <= 3 or consecutive_failures % 10 == 0:
                    logger.warning(f"[v95.0] Heartbeat error (failure {consecutive_failures}): {e}")


# ============================================================================
# Global Instance & Helpers
# ============================================================================

_bridge_instance: Optional[CrossRepoBridge] = None


def get_bridge() -> Optional[CrossRepoBridge]:
    """Get the global bridge instance."""
    return _bridge_instance


async def initialize_bridge(
    port: int = 8000,
    auto_heartbeat: bool = True,
) -> CrossRepoBridge:
    """
    Initialize the global cross-repo bridge.

    Args:
        port: Port this instance is running on
        auto_heartbeat: Whether to start automatic heartbeat

    Returns:
        The initialized bridge instance
    """
    global _bridge_instance

    if _bridge_instance is None:
        _bridge_instance = CrossRepoBridge(port=port, auto_heartbeat=auto_heartbeat)
        await _bridge_instance.initialize()

    return _bridge_instance


async def shutdown_bridge() -> None:
    """
    Shutdown the global bridge instance.

    v95.0: Enhanced with enterprise-grade error handling:
    - Timeout protection to prevent hanging
    - Exception isolation
    - Guaranteed cleanup of global state
    """
    global _bridge_instance

    if _bridge_instance:
        bridge_to_shutdown = _bridge_instance
        logger.info("[v95.0] Initiating cross-repo bridge shutdown...")

        try:
            # v95.0: Timeout protection - don't let shutdown hang forever
            await asyncio.wait_for(
                bridge_to_shutdown.shutdown(),
                timeout=30.0
            )
            logger.info("[v95.0] Cross-repo bridge shutdown completed")
        except asyncio.TimeoutError:
            logger.warning("[v95.0] Bridge shutdown timed out after 30s - forcing cleanup")
        except asyncio.CancelledError:
            logger.warning("[v95.0] Bridge shutdown was cancelled - forcing cleanup")
        except Exception as e:
            logger.error(f"[v95.0] Bridge shutdown error: {e}")
        finally:
            # v95.0: ALWAYS clean up global state, even on error
            _bridge_instance = None


@asynccontextmanager
async def bridge_context(port: int = 8000):
    """
    Context manager for bridge lifecycle.

    Usage:
        async with bridge_context(port=8000) as bridge:
            # Use bridge
            bridge.record_inference(...)
    """
    bridge = await initialize_bridge(port=port)
    try:
        yield bridge
    finally:
        await shutdown_bridge()


# ============================================================================
# Convenience Functions
# ============================================================================

def record_inference(
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
) -> None:
    """Record inference metrics if bridge is active."""
    if _bridge_instance:
        _bridge_instance.record_inference(tokens_in, tokens_out, latency_ms)


def get_cost_summary() -> Dict[str, Any]:
    """Get cost summary if bridge is active."""
    if _bridge_instance:
        return _bridge_instance.get_cost_summary()
    return {}


def update_model_status(loaded: bool, model_path: str = "") -> None:
    """Update model status if bridge is active."""
    if _bridge_instance:
        _bridge_instance.update_model_status(loaded, model_path)
