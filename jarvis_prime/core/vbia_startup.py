"""
JARVIS Prime VBIA Startup Integration
======================================

Initializes JARVIS Prime's connection to the JARVIS cross-repo VBIA system
during startup.

Features:
- Cross-repo state directory monitoring
- JARVIS Prime state file initialization
- Event consumer for VBIA visual security events
- Heartbeat registration
- Async startup integration

Author: JARVIS Prime Team
Version: 6.2.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

logger = logging.getLogger("jarvis-prime.vbia.startup")


# =============================================================================
# Configuration
# =============================================================================

def _get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    return _get_env(key, str(default)).lower() in ("true", "1", "yes", "on")


@dataclass
class VBIAStartupConfig:
    """Configuration for VBIA startup integration."""
    # Cross-repo directory
    cross_repo_dir: Path = field(
        default_factory=lambda: Path(os.path.expanduser(
            _get_env("JARVIS_CROSS_REPO_DIR", "~/.jarvis/cross_repo")
        ))
    )

    # Event consumption settings
    enable_event_consumption: bool = field(
        default_factory=lambda: _get_env_bool("JARVIS_PRIME_CONSUME_VBIA_EVENTS", True)
    )
    event_poll_interval: float = field(
        default_factory=lambda: float(_get_env("JARVIS_PRIME_EVENT_POLL_INTERVAL", "2.0"))
    )

    # State update settings
    heartbeat_interval: float = field(
        default_factory=lambda: float(_get_env("JARVIS_PRIME_HEARTBEAT_INTERVAL", "10.0"))
    )
    state_update_interval: float = field(
        default_factory=lambda: float(_get_env("JARVIS_PRIME_STATE_UPDATE_INTERVAL", "5.0"))
    )

    # Capabilities
    vbia_delegation_enabled: bool = field(
        default_factory=lambda: _get_env_bool("JARVIS_PRIME_VBIA_DELEGATION", True)
    )
    visual_security_aware: bool = field(
        default_factory=lambda: _get_env_bool("JARVIS_PRIME_VISUAL_SECURITY_AWARE", True)
    )


# =============================================================================
# Enums
# =============================================================================

class PrimeStatus(str, Enum):
    """JARVIS Prime status."""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    DEGRADED = "degraded"
    ERROR = "error"
    OFFLINE = "offline"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class PrimeState:
    """JARVIS Prime state for cross-repo system."""
    repo_type: str = "jarvis_prime"
    status: PrimeStatus = PrimeStatus.INITIALIZING
    last_update: str = field(default_factory=lambda: datetime.now().isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "6.2.0"
    capabilities: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class PrimeHeartbeat:
    """JARVIS Prime heartbeat."""
    repo_type: str = "jarvis_prime"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: PrimeStatus = PrimeStatus.READY
    uptime_seconds: float = 0.0
    active_sessions: int = 0


# =============================================================================
# JARVIS Prime VBIA Startup Integrator
# =============================================================================

class JARVISPrimeVBIAStartup:
    """
    Manages JARVIS Prime's startup integration with the cross-repo VBIA system.
    """

    def __init__(self, config: Optional[VBIAStartupConfig] = None):
        self.config = config or VBIAStartupConfig()
        self._initialized = False
        self._start_time = time.time()
        self._running = False

        # State files
        self._prime_state_file = self.config.cross_repo_dir / "prime_state.json"
        self._heartbeat_file = self.config.cross_repo_dir / "heartbeat.json"
        self._vbia_events_file = self.config.cross_repo_dir / "vbia_events.json"

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._state_update_task: Optional[asyncio.Task] = None
        self._event_consumer_task: Optional[asyncio.Task] = None

        # State
        self._prime_state = PrimeState(
            capabilities={
                "vbia_delegation": self.config.vbia_delegation_enabled,
                "visual_security_aware": self.config.visual_security_aware,
                "event_consumption": self.config.enable_event_consumption,
            }
        )

        # Event handlers (can be registered by application)
        self._event_handlers: Dict[str, List[callable]] = {}

    async def initialize(self) -> bool:
        """
        Initialize JARVIS Prime's cross-repo VBIA connection.

        Returns:
            True if initialization succeeded, False otherwise
        """
        if self._initialized:
            logger.info("[VBIA Startup] Already initialized")
            return True

        try:
            logger.info("[VBIA Startup] Starting initialization...")

            # Ensure cross-repo directory exists
            await self._ensure_cross_repo_directory()

            # Initialize JARVIS Prime state file
            await self._initialize_prime_state()

            # Update heartbeat
            await self._update_heartbeat()

            # Start background tasks
            self._running = True
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._state_update_task = asyncio.create_task(self._state_update_loop())

            if self.config.enable_event_consumption:
                self._event_consumer_task = asyncio.create_task(self._event_consumer_loop())

            # Update state to ready
            self._prime_state.status = PrimeStatus.READY
            await self._write_prime_state()

            self._initialized = True
            logger.info("[VBIA Startup] ✅ Initialization complete")
            logger.info(f"[VBIA Startup]    Cross-repo dir: {self.config.cross_repo_dir}")
            logger.info(f"[VBIA Startup]    VBIA delegation: {self.config.vbia_delegation_enabled}")
            logger.info(f"[VBIA Startup]    Visual security aware: {self.config.visual_security_aware}")
            logger.info(f"[VBIA Startup]    Event consumption: {self.config.enable_event_consumption}")

            return True

        except Exception as e:
            logger.error(f"[VBIA Startup] ❌ Initialization failed: {e}", exc_info=True)
            self._prime_state.status = PrimeStatus.ERROR
            self._prime_state.errors.append(str(e))
            return False

    async def shutdown(self) -> None:
        """Shutdown the VBIA startup integration."""
        logger.info("[VBIA Startup] Shutting down...")

        self._running = False

        # Cancel background tasks
        for task in [self._heartbeat_task, self._state_update_task, self._event_consumer_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Update state to offline
        self._prime_state.status = PrimeStatus.OFFLINE
        await self._write_prime_state()

        logger.info("[VBIA Startup] ✅ Shutdown complete")

    # =========================================================================
    # Initialization Helpers
    # =========================================================================

    async def _ensure_cross_repo_directory(self) -> None:
        """Ensure cross-repo directory exists."""
        try:
            self.config.cross_repo_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[VBIA Startup] Cross-repo directory ready: {self.config.cross_repo_dir}")
        except Exception as e:
            logger.error(f"[VBIA Startup] Failed to create cross-repo directory: {e}")
            raise

    async def _initialize_prime_state(self) -> None:
        """Initialize JARVIS Prime state file."""
        await self._write_prime_state()
        logger.info("[VBIA Startup] ✓ prime_state.json initialized")

    async def _update_heartbeat(self) -> None:
        """Update heartbeat in cross-repo system."""
        try:
            # Read existing heartbeats
            heartbeats = await self._read_json_file(self._heartbeat_file, default={})

            # Update JARVIS Prime heartbeat
            heartbeats["jarvis_prime"] = asdict(PrimeHeartbeat(
                status=self._prime_state.status,
                uptime_seconds=time.time() - self._start_time,
                active_sessions=self._prime_state.metrics.get("active_sessions", 0),
            ))

            # Write back
            await self._write_json_file(self._heartbeat_file, heartbeats)

        except Exception as e:
            logger.error(f"[VBIA Startup] Failed to update heartbeat: {e}")

    # =========================================================================
    # Event Handling
    # =========================================================================

    def register_event_handler(self, event_type: str, handler: callable) -> None:
        """
        Register an event handler for VBIA events.

        Args:
            event_type: Event type to handle (e.g., "vbia_visual_threat")
            handler: Async callable to handle the event
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.info(f"[VBIA Startup] Event handler registered for: {event_type}")

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """
        Handle a VBIA event from the cross-repo system.

        Args:
            event: Event dictionary
        """
        event_type = event.get("event_type")
        if not event_type:
            return

        # Call registered handlers
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"[VBIA Startup] Event handler error for {event_type}: {e}")

    # =========================================================================
    # Background Tasks
    # =========================================================================

    async def _heartbeat_loop(self) -> None:
        """Background task that updates heartbeat."""
        logger.info("[VBIA Startup] Heartbeat loop started")

        while self._running:
            try:
                await self._update_heartbeat()
                self._prime_state.last_heartbeat = datetime.now().isoformat()
                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VBIA Startup] Heartbeat loop error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)

    async def _state_update_loop(self) -> None:
        """Background task that updates JARVIS Prime state."""
        logger.info("[VBIA Startup] State update loop started")

        while self._running:
            try:
                await self._write_prime_state()
                await asyncio.sleep(self.config.state_update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VBIA Startup] State update loop error: {e}")
                await asyncio.sleep(self.config.state_update_interval)

    async def _event_consumer_loop(self) -> None:
        """Background task that consumes VBIA events."""
        logger.info("[VBIA Startup] Event consumer loop started")

        last_event_index = 0

        while self._running:
            try:
                # Read events file
                events = await self._read_json_file(self._vbia_events_file, default=[])

                # Process new events
                if len(events) > last_event_index:
                    new_events = events[last_event_index:]
                    for event in new_events:
                        await self._handle_event(event)
                    last_event_index = len(events)

                await asyncio.sleep(self.config.event_poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VBIA Startup] Event consumer error: {e}")
                await asyncio.sleep(self.config.event_poll_interval)

    # =========================================================================
    # State Management
    # =========================================================================

    async def _write_prime_state(self) -> None:
        """Write JARVIS Prime state to file."""
        self._prime_state.last_update = datetime.now().isoformat()
        self._prime_state.metrics["uptime_seconds"] = time.time() - self._start_time
        await self._write_json_file(self._prime_state_file, asdict(self._prime_state))

    async def update_status(self, status: PrimeStatus, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update JARVIS Prime status.

        Args:
            status: New status
            metrics: Optional metrics to update
        """
        self._prime_state.status = status
        if metrics:
            self._prime_state.metrics.update(metrics)
        await self._write_prime_state()

    # =========================================================================
    # File I/O
    # =========================================================================

    async def _read_json_file(self, file_path: Path, default: Any = None) -> Any:
        """Read JSON file asynchronously."""
        try:
            if not file_path.exists():
                return default

            async with aiofiles.open(file_path, "r") as f:
                content = await f.read()
                return json.loads(content) if content else default

        except Exception as e:
            logger.error(f"[VBIA Startup] Failed to read {file_path}: {e}")
            return default

    async def _write_json_file(self, file_path: Path, data: Any) -> None:
        """Write JSON file asynchronously."""
        try:
            async with aiofiles.open(file_path, "w") as f:
                await f.write(json.dumps(data, indent=2))

        except Exception as e:
            logger.error(f"[VBIA Startup] Failed to write {file_path}: {e}")
            raise


# =============================================================================
# Global Singleton
# =============================================================================

_vbia_startup: Optional[JARVISPrimeVBIAStartup] = None


async def get_vbia_startup(
    config: Optional[VBIAStartupConfig] = None
) -> JARVISPrimeVBIAStartup:
    """
    Get or create the global VBIA startup instance.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        The VBIA startup instance
    """
    global _vbia_startup

    if _vbia_startup is None:
        _vbia_startup = JARVISPrimeVBIAStartup(config)

    return _vbia_startup


async def initialize_vbia_startup(
    config: Optional[VBIAStartupConfig] = None
) -> bool:
    """
    Initialize JARVIS Prime's VBIA cross-repo connection.

    This is the main entry point for JARVIS Prime startup integration.

    Args:
        config: Optional configuration

    Returns:
        True if initialization succeeded, False otherwise
    """
    startup = await get_vbia_startup(config)
    return await startup.initialize()


async def shutdown_vbia_startup() -> None:
    """Shutdown JARVIS Prime's VBIA cross-repo connection."""
    global _vbia_startup

    if _vbia_startup:
        await _vbia_startup.shutdown()
        _vbia_startup = None
