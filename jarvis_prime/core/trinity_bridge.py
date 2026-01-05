"""
PROJECT TRINITY Phase 3: J-Prime Trinity Bridge

This module provides automatic Trinity integration for J-Prime (the Mind).
It enables J-Prime to:
- Connect to the Trinity network
- Send cognitive commands to JARVIS Body
- Receive state updates from the orchestrator
- Broadcast model status heartbeats

ARCHITECTURE:
J-Prime generates high-level cognitive decisions (plans, reasoning, analysis)
and sends them through Trinity to JARVIS Body for execution.

USAGE:
    from jarvis_prime.core.trinity_bridge import (
        initialize_trinity,
        shutdown_trinity,
        send_to_jarvis,
    )

    # Initialize during startup
    await initialize_trinity(port=8000)

    # Send a command to JARVIS
    result = await send_to_jarvis(
        intent="start_surveillance",
        payload={"app_name": "Chrome", "trigger_text": "bouncing ball"},
    )
"""

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("jarvis-prime.trinity")


# =============================================================================
# CONFIGURATION
# =============================================================================

TRINITY_ENABLED = os.getenv("TRINITY_ENABLED", "true").lower() == "true"
TRINITY_HEARTBEAT_INTERVAL = float(os.getenv("TRINITY_HEARTBEAT_INTERVAL", "5.0"))
TRINITY_DIR = Path.home() / ".jarvis" / "trinity"

# Instance identification
JPRIME_INSTANCE_ID = os.getenv(
    "JPRIME_INSTANCE_ID",
    f"jprime-{os.getpid()}-{int(time.time())}"
)


# =============================================================================
# GLOBAL STATE
# =============================================================================

_trinity_initialized = False
_heartbeat_task: Optional[asyncio.Task] = None
_state_watcher_task: Optional[asyncio.Task] = None
_start_time = time.time()
_model_loaded = False
_model_path = ""
_port = 8000

# JARVIS state cache (from heartbeats)
_jarvis_state: Dict[str, Any] = {}


# =============================================================================
# INITIALIZATION
# =============================================================================

async def initialize_trinity(
    port: int = 8000,
    model_path: str = "",
    model_loaded: bool = False,
) -> bool:
    """
    Initialize Trinity for J-Prime.

    Args:
        port: Port J-Prime is running on
        model_path: Path to loaded model
        model_loaded: Whether model is loaded

    Returns:
        True if initialization succeeded
    """
    global _trinity_initialized, _heartbeat_task, _state_watcher_task
    global _port, _model_path, _model_loaded

    if not TRINITY_ENABLED:
        logger.info("[Trinity] Trinity is disabled (TRINITY_ENABLED=false)")
        return False

    if _trinity_initialized:
        logger.debug("[Trinity] Already initialized")
        return True

    logger.info("=" * 60)
    logger.info("PROJECT TRINITY: Initializing J-Prime (Mind) Connection")
    logger.info("=" * 60)

    _port = port
    _model_path = model_path
    _model_loaded = model_loaded

    try:
        # Ensure directories exist
        (TRINITY_DIR / "commands").mkdir(parents=True, exist_ok=True)
        (TRINITY_DIR / "heartbeats").mkdir(parents=True, exist_ok=True)
        (TRINITY_DIR / "components").mkdir(parents=True, exist_ok=True)

        # Start heartbeat broadcast
        _heartbeat_task = asyncio.create_task(_heartbeat_loop())
        logger.info(f"[Trinity] ✓ Heartbeat started (interval={TRINITY_HEARTBEAT_INTERVAL}s)")

        # Start state watcher (to receive JARVIS state)
        _state_watcher_task = asyncio.create_task(_state_watcher_loop())
        logger.info("[Trinity] ✓ State watcher started")

        _trinity_initialized = True

        logger.info("=" * 60)
        logger.info(f"PROJECT TRINITY: J-Prime Online (ID: {JPRIME_INSTANCE_ID[:16]})")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"[Trinity] Initialization failed: {e}")
        return False


async def shutdown_trinity() -> None:
    """Shutdown Trinity connection."""
    global _trinity_initialized, _heartbeat_task, _state_watcher_task

    if not _trinity_initialized:
        return

    logger.info("[Trinity] Shutting down J-Prime connection...")

    # Stop background tasks
    for task in [_heartbeat_task, _state_watcher_task]:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    _heartbeat_task = None
    _state_watcher_task = None
    _trinity_initialized = False

    logger.info("[Trinity] J-Prime disconnected")


def update_model_status(loaded: bool, model_path: str = "") -> None:
    """Update model loading status for heartbeat."""
    global _model_loaded, _model_path
    _model_loaded = loaded
    _model_path = model_path


# =============================================================================
# COMMAND SENDING
# =============================================================================

async def send_to_jarvis(
    intent: str,
    payload: Dict[str, Any],
    priority: int = 5,
    timeout: float = 30.0,
    requires_ack: bool = True,
) -> Dict[str, Any]:
    """
    Send a command to JARVIS Body.

    This is the primary method for J-Prime to trigger actions in JARVIS.

    Args:
        intent: Command intent (e.g., "start_surveillance", "bring_back_window")
        payload: Command payload
        priority: Priority (1-10, lower is higher)
        timeout: Command timeout
        requires_ack: Whether to wait for acknowledgment

    Returns:
        Result dict with success status
    """
    if not _trinity_initialized:
        return {"success": False, "error": "Trinity not initialized"}

    command_id = str(uuid.uuid4())

    try:
        command = {
            "id": command_id,
            "timestamp": time.time(),
            "source": "j_prime",
            "intent": intent,
            "payload": payload,
            "metadata": {
                "jprime_instance_id": JPRIME_INSTANCE_ID,
                "model_loaded": _model_loaded,
                "model_path": _model_path,
            },
            "target": "jarvis_body",
            "priority": priority,
            "requires_ack": requires_ack,
            "ttl_seconds": timeout,
        }

        # Write command file
        commands_dir = TRINITY_DIR / "commands"
        filename = f"{int(time.time() * 1000)}_{command_id}.json"
        filepath = commands_dir / filename

        with open(filepath, "w") as f:
            json.dump(command, f, indent=2)

        logger.info(f"[Trinity] J-Prime sent command: {intent} (id={command_id[:8]})")
        return {"success": True, "command_id": command_id}

    except Exception as e:
        logger.error(f"[Trinity] Failed to send command: {e}")
        return {"success": False, "error": str(e)}


async def send_plan_to_jarvis(
    plan_id: str,
    steps: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Send a multi-step cognitive plan to JARVIS.

    This is the primary output of J-Prime's reasoning - a sequence of
    actions for JARVIS to execute.

    Args:
        plan_id: Unique plan identifier
        steps: List of plan steps, each with intent and payload
        context: Optional context for the plan

    Returns:
        Result dict
    """
    return await send_to_jarvis(
        intent="execute_plan",
        payload={
            "plan_id": plan_id,
            "steps": steps,
            "context": context or {},
        },
        priority=2,  # High priority for plans
    )


# =============================================================================
# CONVENIENCE METHODS
# =============================================================================

async def start_surveillance(
    app_name: str,
    trigger_text: str,
    all_spaces: bool = True,
    max_duration: Optional[int] = None,
) -> Dict[str, Any]:
    """Request JARVIS to start surveillance."""
    return await send_to_jarvis(
        intent="start_surveillance",
        payload={
            "app_name": app_name,
            "trigger_text": trigger_text,
            "all_spaces": all_spaces,
            "max_duration": max_duration,
        },
    )


async def stop_surveillance(app_name: Optional[str] = None) -> Dict[str, Any]:
    """Request JARVIS to stop surveillance."""
    return await send_to_jarvis(
        intent="stop_surveillance",
        payload={"app_name": app_name},
    )


async def bring_back_windows(app_name: Optional[str] = None) -> Dict[str, Any]:
    """Request JARVIS to bring back windows from Ghost Display."""
    return await send_to_jarvis(
        intent="bring_back_window",
        payload={"app_name": app_name},
    )


async def exile_window(
    app_name: str,
    window_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Request JARVIS to exile a window to Ghost Display."""
    return await send_to_jarvis(
        intent="exile_window",
        payload={
            "app_name": app_name,
            "window_title": window_title,
        },
    )


async def freeze_app(app_name: str, reason: str = "") -> Dict[str, Any]:
    """Request JARVIS to freeze an app."""
    return await send_to_jarvis(
        intent="freeze_app",
        payload={"app_name": app_name, "reason": reason},
    )


async def thaw_app(app_name: str) -> Dict[str, Any]:
    """Request JARVIS to thaw a frozen app."""
    return await send_to_jarvis(
        intent="thaw_app",
        payload={"app_name": app_name},
    )


# =============================================================================
# HEARTBEAT
# =============================================================================

async def _heartbeat_loop() -> None:
    """Background task to broadcast heartbeats."""
    while _trinity_initialized:
        try:
            await _broadcast_heartbeat()
            await asyncio.sleep(TRINITY_HEARTBEAT_INTERVAL)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"[Trinity] Heartbeat error: {e}")
            await asyncio.sleep(TRINITY_HEARTBEAT_INTERVAL)


async def _broadcast_heartbeat() -> None:
    """Broadcast J-Prime state as heartbeat."""
    state = {
        "instance_id": JPRIME_INSTANCE_ID,
        "component_type": "j_prime",
        "timestamp": time.time(),
        "uptime_seconds": time.time() - _start_time,
        "model_loaded": _model_loaded,
        "model_path": _model_path,
        "port": _port,
        "endpoint": f"http://localhost:{_port}",
    }

    # System metrics
    try:
        import psutil
        state["system_cpu_percent"] = psutil.cpu_percent()
        state["system_memory_percent"] = psutil.virtual_memory().percent
    except ImportError:
        pass

    # Write to components directory
    try:
        components_dir = TRINITY_DIR / "components"
        state_file = components_dir / "j_prime.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.debug(f"[Trinity] Could not write state: {e}")

    # Also write heartbeat file for bridge compatibility
    try:
        heartbeats_dir = TRINITY_DIR / "heartbeats"
        heartbeat_file = heartbeats_dir / f"jprime_{int(time.time() * 1000)}.json"
        with open(heartbeat_file, "w") as f:
            json.dump({
                "id": str(uuid.uuid4()),
                "timestamp": time.time(),
                "source": "j_prime",
                "intent": "heartbeat",
                "payload": state,
            }, f, indent=2)

        # Cleanup old heartbeats (keep last 10)
        heartbeats = sorted(heartbeats_dir.glob("jprime_*.json"))
        for old_file in heartbeats[:-10]:
            old_file.unlink(missing_ok=True)

    except Exception as e:
        logger.debug(f"[Trinity] Could not write heartbeat: {e}")


# =============================================================================
# STATE WATCHING
# =============================================================================

async def _state_watcher_loop() -> None:
    """Watch for JARVIS state updates."""
    global _jarvis_state

    while _trinity_initialized:
        try:
            # Read JARVIS component state
            jarvis_file = TRINITY_DIR / "components" / "jarvis_body.json"
            if jarvis_file.exists():
                with open(jarvis_file) as f:
                    data = json.load(f)
                _jarvis_state = data.get("metrics", {})

            await asyncio.sleep(2.0)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"[Trinity] State watcher error: {e}")
            await asyncio.sleep(2.0)


def get_jarvis_state() -> Dict[str, Any]:
    """Get cached JARVIS state."""
    return _jarvis_state.copy()


def is_jarvis_online() -> bool:
    """Check if JARVIS is online based on cached state."""
    if not _jarvis_state:
        return False

    timestamp = _jarvis_state.get("timestamp", 0)
    age = time.time() - timestamp
    return age < 15.0  # Consider online if heartbeat < 15s old


# =============================================================================
# STATUS
# =============================================================================

def is_trinity_initialized() -> bool:
    """Check if Trinity is initialized."""
    return _trinity_initialized


def get_trinity_status() -> Dict[str, Any]:
    """Get current Trinity status."""
    return {
        "enabled": TRINITY_ENABLED,
        "initialized": _trinity_initialized,
        "instance_id": JPRIME_INSTANCE_ID,
        "uptime_seconds": time.time() - _start_time if _trinity_initialized else 0,
        "model_loaded": _model_loaded,
        "model_path": _model_path,
        "jarvis_online": is_jarvis_online(),
        "jarvis_state": _jarvis_state,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Initialization
    "initialize_trinity",
    "shutdown_trinity",
    "update_model_status",
    "is_trinity_initialized",
    "get_trinity_status",
    # Commands
    "send_to_jarvis",
    "send_plan_to_jarvis",
    # Convenience
    "start_surveillance",
    "stop_surveillance",
    "bring_back_windows",
    "exile_window",
    "freeze_app",
    "thaw_app",
    # State
    "get_jarvis_state",
    "is_jarvis_online",
    # Constants
    "JPRIME_INSTANCE_ID",
]
