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

# PROJECT TRINITY: Unified command routing
TRINITY_DIR = Path.home() / ".jarvis" / "trinity"
TRINITY_COMMANDS_DIR = TRINITY_DIR / "commands"


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
        )

        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown_callbacks: List[Callable] = []
        self._initialized = False

        # Ensure state directory exists
        BRIDGE_STATE_DIR.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the bridge and connect to main JARVIS."""
        if self._initialized:
            return

        logger.info(f"Initializing cross-repo bridge (instance={self.instance_id})")

        # Check for main JARVIS instance
        await self._check_jarvis_connection()

        # Write initial state
        await self._write_state()

        # Start heartbeat loop
        if self.auto_heartbeat:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._initialized = True
        logger.info("Cross-repo bridge initialized successfully")

    async def shutdown(self) -> None:
        """Gracefully shutdown the bridge."""
        logger.info("Shutting down cross-repo bridge...")

        # Update state
        self.state.status = "shutting_down"
        await self._write_state()

        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Run shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.warning(f"Shutdown callback error: {e}")

        # Mark as stopped
        self.state.status = "stopped"
        await self._write_state()

        logger.info("Cross-repo bridge shutdown complete")

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
        """Check if main JARVIS is running and get session info."""
        jarvis_state = await self.get_jarvis_state()

        if jarvis_state:
            # Check if JARVIS state is recent (not stale)
            last_update = jarvis_state.get("last_update", "")
            if last_update:
                try:
                    update_time = datetime.fromisoformat(last_update)
                    age_seconds = (datetime.now() - update_time).total_seconds()

                    if age_seconds < STALE_THRESHOLD:
                        self.state.connected_to_jarvis = True
                        self.state.jarvis_session_id = jarvis_state.get("session_id", "")
                        logger.info(f"Connected to JARVIS session: {self.state.jarvis_session_id}")
                        await self.notify_jarvis("connected", {
                            "port": self.port,
                            "model_loaded": self.state.model_loaded,
                        })
                        return
                except (ValueError, TypeError):
                    pass

        self.state.connected_to_jarvis = False
        logger.info("Main JARVIS not detected - running standalone")

    async def _write_state(self) -> None:
        """Write current state to shared file."""
        try:
            self.state.last_heartbeat = datetime.now().isoformat()
            PRIME_STATE_FILE.write_text(json.dumps(self.state.to_dict(), indent=2))
        except Exception as e:
            logger.warning(f"Failed to write state: {e}")

    async def _heartbeat_loop(self) -> None:
        """Background loop to maintain heartbeat and sync state."""
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)

                # Update heartbeat
                await self._write_state()

                # Re-check JARVIS connection periodically
                if not self.state.connected_to_jarvis:
                    await self._check_jarvis_connection()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")


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
    """Shutdown the global bridge instance."""
    global _bridge_instance

    if _bridge_instance:
        await _bridge_instance.shutdown()
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
