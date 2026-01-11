"""
Trinity Bridge Adapter v1.0 - Unified Event Translation Layer
==============================================================

This module bridges THREE different event systems into ONE unified stream:

1. JARVIS-Prime TrinityEventBus (trinity_event_bus.py)
   - FILE-based transport in ~/.jarvis/trinity/events
   - EventType enum (MODEL_READY, TRAINING_COMPLETE, etc.)

2. Reactor-Core EventBridge (event_bridge.py)
   - FILE-based transport in ~/.jarvis/events
   - CrossRepoEvent with EventType enum

3. JARVIS-AI-Agent (via file watcher)
   - Legacy file-based in ~/.jarvis/cross_repo
   - JSON files with event data

THE PROBLEM:
    Each repo has its own event system with different:
    - File locations
    - Event formats
    - Enum types
    - Transport mechanisms

THE SOLUTION:
    This adapter creates a unified translation layer that:
    - Watches ALL event directories
    - Translates events between formats
    - Forwards events to appropriate destinations
    - Handles race conditions and deduplication
    - Provides async-safe operation

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    TRINITY BRIDGE ADAPTER                               │
    │                                                                         │
    │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐             │
    │  │   JARVIS     │     │    Prime     │     │   Reactor    │             │
    │  │ EventBridge  │     │ TrinityBus   │     │  EventBridge │             │
    │  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘             │
    │         │                    │                    │                     │
    │         └────────────────────┼────────────────────┘                     │
    │                              │                                          │
    │                    ┌─────────▼─────────┐                                │
    │                    │     ADAPTER       │                                │
    │                    │  - Translation    │                                │ 
    │                    │  - Deduplication  │                                │
    │                    │  - Routing        │                                │
    │                    └───────────────────┘                                │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

USAGE:
    adapter = await TrinityBridgeAdapter.create()
    await adapter.start()
    # Events now flow between all repos automatically
    await adapter.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# TRY IMPORTS
# =============================================================================

try:
    from jarvis_prime.core.trinity_event_bus import (
        TrinityEventBus,
        EventType as PrimeEventType,
        ComponentID,
        TrinityEvent,
        get_event_bus,
    )
    PRIME_EVENT_BUS_AVAILABLE = True
except ImportError:
    PRIME_EVENT_BUS_AVAILABLE = False
    PrimeEventType = None
    ComponentID = None


# =============================================================================
# UNIFIED EVENT TYPE MAPPING
# =============================================================================

class UnifiedEventType(Enum):
    """Unified event types that work across all repos."""
    # Model lifecycle
    MODEL_READY = "model_ready"
    MODEL_LOADING = "model_loading"
    MODEL_UNLOADED = "model_unloaded"
    MODEL_FAILED = "model_failed"

    # Training pipeline
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_FAILED = "training_failed"

    # Data collection
    EXPERIENCE_COLLECTED = "experience_collected"
    DATA_BATCH_READY = "data_batch_ready"

    # Health & Status
    HEALTH_CHANGED = "health_changed"
    HEARTBEAT = "heartbeat"
    COMPONENT_STARTED = "component_started"
    COMPONENT_STOPPED = "component_stopped"

    # System
    SHUTDOWN_REQUESTED = "shutdown_requested"
    CONFIG_UPDATED = "config_updated"


# Mapping from Reactor-Core EventType values to UnifiedEventType
REACTOR_TO_UNIFIED = {
    "training_start": UnifiedEventType.TRAINING_STARTED,
    "training_progress": UnifiedEventType.TRAINING_PROGRESS,
    "training_complete": UnifiedEventType.TRAINING_COMPLETE,
    "training_failed": UnifiedEventType.TRAINING_FAILED,
    "model_updated": UnifiedEventType.MODEL_READY,
    "service_up": UnifiedEventType.COMPONENT_STARTED,
    "service_down": UnifiedEventType.COMPONENT_STOPPED,
    "config_changed": UnifiedEventType.CONFIG_UPDATED,
}

# Mapping from Prime EventType values to UnifiedEventType
PRIME_TO_UNIFIED = {
    "model_ready": UnifiedEventType.MODEL_READY,
    "model_loading": UnifiedEventType.MODEL_LOADING,
    "training_started": UnifiedEventType.TRAINING_STARTED,
    "training_progress": UnifiedEventType.TRAINING_PROGRESS,
    "training_complete": UnifiedEventType.TRAINING_COMPLETE,
    "training_failed": UnifiedEventType.TRAINING_FAILED,
    "experience_collected": UnifiedEventType.EXPERIENCE_COLLECTED,
    "health_changed": UnifiedEventType.HEALTH_CHANGED,
    "heartbeat": UnifiedEventType.HEARTBEAT,
    "shutdown_requested": UnifiedEventType.SHUTDOWN_REQUESTED,
}


# =============================================================================
# UNIFIED EVENT
# =============================================================================

@dataclass
class UnifiedEvent:
    """A unified event that can be shared across all repos."""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: UnifiedEventType = UnifiedEventType.HEARTBEAT
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # For deduplication
    _hash: str = ""

    def __post_init__(self):
        if not self._hash:
            self._hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash for deduplication."""
        content = f"{self.event_type.value}:{self.source}:{json.dumps(self.payload, sort_keys=True, default=str)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "metadata": self.metadata,
            "hash": self._hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedEvent":
        event_type_str = data.get("event_type", "heartbeat")
        try:
            event_type = UnifiedEventType(event_type_str)
        except ValueError:
            event_type = UnifiedEventType.HEARTBEAT

        return cls(
            event_id=data.get("event_id", uuid.uuid4().hex[:12]),
            event_type=event_type,
            source=data.get("source", "unknown"),
            timestamp=data.get("timestamp", time.time()),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# FILE WATCHER
# =============================================================================

class DirectoryWatcher:
    """Async directory watcher for event files."""

    def __init__(
        self,
        directory: Path,
        callback: Callable[[Path], Any],
        poll_interval: float = 0.5,
        file_pattern: str = "*.json",
    ):
        self._directory = directory
        self._callback = callback
        self._poll_interval = poll_interval
        self._file_pattern = file_pattern
        self._seen_files: Set[str] = set()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start watching the directory."""
        if self._running:
            return

        self._directory.mkdir(parents=True, exist_ok=True)

        # Pre-populate seen files to avoid processing old events
        for f in self._directory.glob(self._file_pattern):
            self._seen_files.add(f.name)

        self._running = True
        self._task = asyncio.create_task(self._watch_loop())
        logger.debug(f"DirectoryWatcher started for {self._directory}")

    async def stop(self):
        """Stop watching."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _watch_loop(self):
        """Main watch loop."""
        while self._running:
            try:
                await self._check_for_new_files()
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DirectoryWatcher error: {e}")
                await asyncio.sleep(1.0)

    async def _check_for_new_files(self):
        """Check for new event files."""
        if not self._directory.exists():
            return

        for event_file in sorted(self._directory.glob(self._file_pattern)):
            if event_file.name not in self._seen_files:
                self._seen_files.add(event_file.name)
                try:
                    # Call async or sync callback
                    result = self._callback(event_file)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Error processing {event_file}: {e}")


# =============================================================================
# EVENT TRANSLATORS
# =============================================================================

class ReactorEventTranslator:
    """Translates Reactor-Core events to unified format."""

    @staticmethod
    def translate(file_path: Path) -> Optional[UnifiedEvent]:
        """Translate a Reactor event file to UnifiedEvent."""
        try:
            data = json.loads(file_path.read_text())

            # Reactor events have: event_type, source, payload
            event_type_str = data.get("event_type", "")
            unified_type = REACTOR_TO_UNIFIED.get(event_type_str)

            if not unified_type:
                logger.debug(f"Unknown Reactor event type: {event_type_str}")
                return None

            return UnifiedEvent(
                event_id=data.get("event_id", uuid.uuid4().hex[:12]),
                event_type=unified_type,
                source=data.get("source", "reactor_core"),
                timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())).timestamp()
                         if isinstance(data.get("timestamp"), str) else data.get("timestamp", time.time()),
                payload=data.get("payload", {}),
                metadata=data.get("metadata", {}),
            )

        except Exception as e:
            logger.error(f"Failed to translate Reactor event: {e}")
            return None


class PrimeEventTranslator:
    """Translates Prime events to unified format."""

    @staticmethod
    def translate(file_path: Path) -> Optional[UnifiedEvent]:
        """Translate a Prime event file to UnifiedEvent."""
        try:
            data = json.loads(file_path.read_text())

            event_type_str = data.get("event_type", "")
            unified_type = PRIME_TO_UNIFIED.get(event_type_str)

            if not unified_type:
                logger.debug(f"Unknown Prime event type: {event_type_str}")
                return None

            return UnifiedEvent(
                event_id=data.get("event_id", uuid.uuid4().hex[:12]),
                event_type=unified_type,
                source=data.get("source", "jarvis_prime"),
                timestamp=data.get("timestamp", time.time()),
                payload=data.get("payload", {}),
                metadata=data.get("metadata", {}),
            )

        except Exception as e:
            logger.error(f"Failed to translate Prime event: {e}")
            return None


# =============================================================================
# TRINITY BRIDGE ADAPTER
# =============================================================================

class TrinityBridgeAdapter:
    """
    The unified bridge that connects all event systems.

    This adapter:
    1. Watches all event directories
    2. Translates events between formats
    3. Forwards events to appropriate destinations
    4. Handles deduplication
    """

    _instance: Optional["TrinityBridgeAdapter"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        # Event directories
        self._base_dir = Path.home() / ".jarvis"
        self._prime_events_dir = self._base_dir / "trinity" / "events"
        self._reactor_events_dir = self._base_dir / "reactor" / "events"
        self._jarvis_events_dir = self._base_dir / "cross_repo"
        self._unified_events_dir = self._base_dir / "unified_events"

        # Watchers
        self._watchers: List[DirectoryWatcher] = []

        # Deduplication
        self._processed_hashes: deque = deque(maxlen=1000)

        # Event handlers
        self._handlers: Dict[UnifiedEventType, List[Callable]] = {}

        # State
        self._running = False

        # Prime event bus integration
        self._prime_bus: Optional["TrinityEventBus"] = None

        # Metrics
        self._events_received = 0
        self._events_forwarded = 0
        self._events_deduplicated = 0

    @classmethod
    async def create(cls) -> "TrinityBridgeAdapter":
        """Create or get the singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance._initialize()
            return cls._instance

    async def _initialize(self):
        """Initialize the adapter."""
        # Create directories
        for d in [self._prime_events_dir, self._reactor_events_dir,
                  self._jarvis_events_dir, self._unified_events_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Connect to Prime event bus if available
        if PRIME_EVENT_BUS_AVAILABLE:
            try:
                self._prime_bus = await get_event_bus(ComponentID.ORCHESTRATOR)
                logger.info("Connected to Prime TrinityEventBus")
            except Exception as e:
                logger.warning(f"Could not connect to Prime event bus: {e}")

        logger.info("TrinityBridgeAdapter initialized")

    async def start(self):
        """Start the bridge adapter."""
        if self._running:
            return

        self._running = True

        # Create watchers for each event directory
        self._watchers = [
            DirectoryWatcher(
                self._reactor_events_dir,
                self._on_reactor_event,
                poll_interval=0.3,
            ),
            DirectoryWatcher(
                self._jarvis_events_dir,
                self._on_jarvis_event,
                poll_interval=0.3,
            ),
        ]

        # Start all watchers
        for watcher in self._watchers:
            await watcher.start()

        logger.info("TrinityBridgeAdapter started - watching all event directories")

    async def stop(self):
        """Stop the bridge adapter."""
        self._running = False

        for watcher in self._watchers:
            await watcher.stop()

        self._watchers.clear()

        async with self._lock:
            TrinityBridgeAdapter._instance = None

        logger.info("TrinityBridgeAdapter stopped")

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    async def _on_reactor_event(self, file_path: Path):
        """Handle events from Reactor-Core."""
        event = ReactorEventTranslator.translate(file_path)
        if event:
            await self._process_event(event, source_system="reactor")
            # Clean up the file
            try:
                file_path.unlink()
            except Exception:
                pass

    async def _on_jarvis_event(self, file_path: Path):
        """Handle events from JARVIS-AI-Agent."""
        try:
            data = json.loads(file_path.read_text())

            # Try to determine event type from payload
            event_type_str = data.get("event_type", data.get("type", ""))

            # Map common JARVIS events
            if "experience" in event_type_str.lower() or "interaction" in event_type_str.lower():
                unified_type = UnifiedEventType.EXPERIENCE_COLLECTED
            elif "health" in event_type_str.lower():
                unified_type = UnifiedEventType.HEALTH_CHANGED
            else:
                unified_type = UnifiedEventType.HEARTBEAT

            event = UnifiedEvent(
                event_type=unified_type,
                source="jarvis_agent",
                payload=data.get("payload", data),
                metadata=data.get("metadata", {}),
            )

            await self._process_event(event, source_system="jarvis")

            # Clean up the file
            try:
                file_path.unlink()
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Failed to process JARVIS event: {e}")

    async def _process_event(self, event: UnifiedEvent, source_system: str):
        """Process and forward a unified event."""
        self._events_received += 1

        # Deduplication
        if event._hash in self._processed_hashes:
            self._events_deduplicated += 1
            return

        self._processed_hashes.append(event._hash)

        logger.info(f"[BRIDGE] {source_system} -> {event.event_type.value}: {event.payload.get('model_name', 'N/A')}")

        # Forward to Prime event bus
        if self._prime_bus and source_system != "prime":
            await self._forward_to_prime(event)

        # Forward to JARVIS (write to cross_repo directory)
        if source_system != "jarvis":
            await self._forward_to_jarvis(event)

        # Write to unified events directory for any consumer
        await self._write_unified_event(event)

        # Call registered handlers
        await self._call_handlers(event)

        self._events_forwarded += 1

    async def _forward_to_prime(self, event: UnifiedEvent):
        """Forward event to Prime's TrinityEventBus."""
        if not self._prime_bus or not PRIME_EVENT_BUS_AVAILABLE:
            return

        try:
            # Map UnifiedEventType to Prime's EventType
            prime_type_map = {
                UnifiedEventType.MODEL_READY: PrimeEventType.MODEL_READY,
                UnifiedEventType.TRAINING_STARTED: PrimeEventType.TRAINING_STARTED,
                UnifiedEventType.TRAINING_COMPLETE: PrimeEventType.TRAINING_COMPLETE,
                UnifiedEventType.TRAINING_FAILED: PrimeEventType.TRAINING_FAILED,
                UnifiedEventType.EXPERIENCE_COLLECTED: PrimeEventType.EXPERIENCE_COLLECTED,
                UnifiedEventType.HEALTH_CHANGED: PrimeEventType.HEALTH_CHANGED,
                UnifiedEventType.HEARTBEAT: PrimeEventType.HEARTBEAT,
            }

            prime_type = prime_type_map.get(event.event_type)
            if prime_type:
                await self._prime_bus.publish(
                    event_type=prime_type,
                    payload=event.payload,
                )
                logger.debug(f"Forwarded to Prime: {event.event_type.value}")

        except Exception as e:
            logger.error(f"Failed to forward to Prime: {e}")

    async def _forward_to_jarvis(self, event: UnifiedEvent):
        """Forward event to JARVIS by writing to cross_repo directory."""
        try:
            event_file = self._jarvis_events_dir / f"unified_{event.event_id}.json"

            # Write event in JARVIS-compatible format
            data = {
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "source": event.source,
                "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
                "payload": event.payload,
                "metadata": event.metadata,
            }

            with open(event_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Forwarded to JARVIS: {event.event_type.value}")

        except Exception as e:
            logger.error(f"Failed to forward to JARVIS: {e}")

    async def _write_unified_event(self, event: UnifiedEvent):
        """Write event to unified events directory."""
        try:
            timestamp_prefix = f"{int(event.timestamp * 1000):015d}"
            event_file = self._unified_events_dir / f"{timestamp_prefix}_{event.event_id}.json"

            with open(event_file, "w") as f:
                json.dump(event.to_dict(), f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to write unified event: {e}")

    async def _call_handlers(self, event: UnifiedEvent):
        """Call registered handlers for the event type."""
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Handler error for {event.event_type.value}: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def subscribe(
        self,
        event_type: UnifiedEventType,
        handler: Callable[[UnifiedEvent], Any],
    ) -> str:
        """Subscribe to events of a specific type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        return f"{event_type.value}_{len(self._handlers[event_type])}"

    async def publish(
        self,
        event_type: UnifiedEventType,
        payload: Dict[str, Any],
        source: str = "trinity_adapter",
    ) -> bool:
        """Publish an event to all systems."""
        event = UnifiedEvent(
            event_type=event_type,
            source=source,
            payload=payload,
        )

        await self._process_event(event, source_system="adapter")
        return True

    async def publish_model_ready(
        self,
        model_name: str,
        model_path: str,
        capabilities: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Convenience method to publish MODEL_READY event."""
        return await self.publish(
            event_type=UnifiedEventType.MODEL_READY,
            payload={
                "model_name": model_name,
                "model_path": model_path,
                "capabilities": capabilities or [],
                "metrics": metrics or {},
                "ready_at": time.time(),
            },
        )

    async def publish_training_complete(
        self,
        model_name: str,
        model_path: str,
        training_metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Convenience method to publish TRAINING_COMPLETE event."""
        return await self.publish(
            event_type=UnifiedEventType.TRAINING_COMPLETE,
            payload={
                "model_name": model_name,
                "model_path": model_path,
                "training_metrics": training_metrics or {},
                "completed_at": time.time(),
            },
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get adapter metrics."""
        return {
            "events_received": self._events_received,
            "events_forwarded": self._events_forwarded,
            "events_deduplicated": self._events_deduplicated,
            "watchers_active": len(self._watchers),
            "running": self._running,
            "prime_bus_connected": self._prime_bus is not None,
        }


# =============================================================================
# GLOBAL ACCESS
# =============================================================================

_adapter: Optional[TrinityBridgeAdapter] = None


async def get_bridge_adapter() -> TrinityBridgeAdapter:
    """Get or create the bridge adapter."""
    global _adapter
    if _adapter is None:
        _adapter = await TrinityBridgeAdapter.create()
    return _adapter


async def start_bridge() -> TrinityBridgeAdapter:
    """Start the bridge adapter."""
    adapter = await get_bridge_adapter()
    await adapter.start()
    return adapter


async def stop_bridge():
    """Stop the bridge adapter."""
    global _adapter
    if _adapter is not None:
        await _adapter.stop()
        _adapter = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "UnifiedEventType",
    "UnifiedEvent",
    # Adapter
    "TrinityBridgeAdapter",
    # Functions
    "get_bridge_adapter",
    "start_bridge",
    "stop_bridge",
]
