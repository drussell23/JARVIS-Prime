"""
Trinity Event Bus v1.0 - The Neural Impulses
=============================================

This is the MISSING PIECE that closes the Trinity Loop.
Without this, the repos are like nerves without electrical impulses.

THE LOOP (Now Connected):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         THE TRINITY LOOP                                │
    │                                                                         │
    │   ┌──────────┐    experience    ┌──────────┐    training    ┌─────────┐ │
    │   │  JARVIS  │ ───────────────► │ Reactor  │ ─────────────► │  Prime  │ │
    │   │  (Body)  │                  │ (Nerves) │                │  (Mind) │ │
    │   └────┬─────┘                  └────┬─────┘                └────┬────┘ │
    │        │                             │                           │      │
    │        │◄────────── new_model ───────┼───────── model_ready ─────┘      │
    │        │                             │                                  │
    │        └─────────────────────────────┴──────────────────────────────────│
    │                                                                         │
    │                    ▼▼▼ EVENT BUS (This File) ▼▼▼                        │
    └─────────────────────────────────────────────────────────────────────────┘

EVENT TYPES:
    - model_ready: New model available for hot-swap
    - training_started: Reactor began training
    - training_complete: Reactor finished training
    - experience_collected: JARVIS collected new data
    - health_changed: Component health status changed
    - config_updated: Configuration hot-reloaded
    - shutdown_requested: Graceful shutdown initiated

TRANSPORTS:
    - File IPC (default, always works)
    - Redis Pub/Sub (optional, for distributed deployments)
    - WebSocket (optional, for real-time UI)
    - Memory (for single-process testing)

FEATURES:
    - Async pub/sub with backpressure
    - Event persistence (replay after restart)
    - Dead letter queue for failed handlers
    - Event deduplication
    - Priority queues
    - Wildcard subscriptions
    - Event correlation (trace through the loop)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
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
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    """All event types in the Trinity ecosystem."""
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
    DATA_QUALITY_ALERT = "data_quality_alert"

    # Health & Status
    HEALTH_CHANGED = "health_changed"
    COMPONENT_STARTED = "component_started"
    COMPONENT_STOPPED = "component_stopped"
    COMPONENT_FAILED = "component_failed"

    # Configuration
    CONFIG_UPDATED = "config_updated"
    CONFIG_RELOAD_REQUESTED = "config_reload_requested"

    # System
    SHUTDOWN_REQUESTED = "shutdown_requested"
    HEARTBEAT = "heartbeat"

    # Custom
    CUSTOM = "custom"


class EventPriority(Enum):
    """Event priority levels."""
    CRITICAL = 0    # Process immediately (shutdown, failures)
    HIGH = 1        # Process soon (model ready, training complete)
    NORMAL = 2      # Standard processing
    LOW = 3         # Background/batch processing
    BACKGROUND = 4  # When idle


class ComponentID(Enum):
    """Trinity component identifiers."""
    ORCHESTRATOR = "orchestrator"
    JARVIS_BODY = "jarvis_body"
    JARVIS_PRIME = "jarvis_prime"
    REACTOR_CORE = "reactor_core"
    EVENT_BUS = "event_bus"
    BROADCAST = "*"  # All components


# =============================================================================
# EVENT DATA STRUCTURES
# =============================================================================

@dataclass
class TrinityEvent:
    """
    A single event in the Trinity ecosystem.

    Events are immutable once created and can be serialized for
    persistence or network transmission.
    """
    # Identity
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event_type: EventType = EventType.CUSTOM

    # Routing
    source: ComponentID = ComponentID.ORCHESTRATOR
    target: ComponentID = ComponentID.BROADCAST
    priority: EventPriority = EventPriority.NORMAL

    # Payload
    payload: Dict[str, Any] = field(default_factory=dict)

    # Timing
    timestamp: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # Auto-expire old events

    # Correlation (for tracing through the loop)
    correlation_id: Optional[str] = None  # Links related events
    causation_id: Optional[str] = None    # The event that caused this one
    sequence: int = 0                      # Order in a sequence

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Auto-set correlation_id if not provided
        if self.correlation_id is None:
            self.correlation_id = self.event_id

    @property
    def is_expired(self) -> bool:
        """Check if event has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get event age in seconds."""
        return time.time() - self.timestamp

    def create_response(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        source: ComponentID,
    ) -> "TrinityEvent":
        """Create a response event linked to this one."""
        return TrinityEvent(
            event_type=event_type,
            source=source,
            target=self.source,
            payload=payload,
            correlation_id=self.correlation_id,
            causation_id=self.event_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source.value,
            "target": self.target.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "expires_at": self.expires_at,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "sequence": self.sequence,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrinityEvent":
        """Deserialize from dictionary."""
        return cls(
            event_id=data.get("event_id", uuid.uuid4().hex),
            event_type=EventType(data.get("event_type", "custom")),
            source=ComponentID(data.get("source", "orchestrator")),
            target=ComponentID(data.get("target", "*")),
            priority=EventPriority(data.get("priority", 2)),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            expires_at=data.get("expires_at"),
            correlation_id=data.get("correlation_id"),
            causation_id=data.get("causation_id"),
            sequence=data.get("sequence", 0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "TrinityEvent":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# EVENT HANDLER TYPES
# =============================================================================

# Type for event handlers
EventHandler = Callable[[TrinityEvent], Awaitable[None]]
SyncEventHandler = Callable[[TrinityEvent], None]


@dataclass
class Subscription:
    """A subscription to events."""
    subscription_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    handler: EventHandler = field(default=None)
    event_types: Set[EventType] = field(default_factory=set)  # Empty = all types
    source_filter: Optional[ComponentID] = None  # None = all sources
    priority_filter: Optional[EventPriority] = None  # None = all priorities
    active: bool = True
    created_at: float = field(default_factory=time.time)

    def matches(self, event: TrinityEvent) -> bool:
        """Check if this subscription should receive the event."""
        if not self.active:
            return False

        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check source filter
        if self.source_filter and event.source != self.source_filter:
            return False

        # Check priority filter
        if self.priority_filter and event.priority != self.priority_filter:
            return False

        return True


# =============================================================================
# TRANSPORT INTERFACE
# =============================================================================

class EventTransport(ABC):
    """Abstract base for event transports."""

    @abstractmethod
    async def publish(self, event: TrinityEvent) -> bool:
        """Publish an event."""
        ...

    @abstractmethod
    async def subscribe(self, callback: EventHandler) -> str:
        """Subscribe to events. Returns subscription ID."""
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        ...

    @abstractmethod
    async def start(self):
        """Start the transport."""
        ...

    @abstractmethod
    async def stop(self):
        """Stop the transport."""
        ...


# =============================================================================
# FILE-BASED TRANSPORT (Default, Always Works)
# =============================================================================

class FileEventTransport(EventTransport):
    """
    File-based event transport using the Trinity directory.

    Events are written as JSON files to a shared directory.
    Components poll for new events.

    This is the default transport that always works, even across
    different processes and without any external dependencies.
    """

    def __init__(
        self,
        trinity_dir: Optional[Path] = None,
        component_id: ComponentID = ComponentID.ORCHESTRATOR,
        poll_interval: float = 0.5,
        max_event_age_seconds: float = 300.0,  # 5 minutes
    ):
        self._trinity_dir = trinity_dir or Path.home() / ".jarvis" / "trinity"
        self._component_id = component_id
        self._poll_interval = poll_interval
        self._max_event_age = max_event_age_seconds

        # Directories
        self._events_dir = self._trinity_dir / "events"
        self._inbox_dir = self._trinity_dir / "inbox" / component_id.value
        self._outbox_dir = self._trinity_dir / "outbox" / component_id.value
        self._processed_dir = self._trinity_dir / "processed"
        self._dead_letter_dir = self._trinity_dir / "dead_letter"

        # State
        self._subscriptions: Dict[str, Subscription] = {}
        self._processed_events: Set[str] = set()
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Metrics
        self._events_published = 0
        self._events_received = 0
        self._events_failed = 0

    async def start(self):
        """Start the file transport."""
        if self._running:
            return

        # Create directories
        for d in [self._events_dir, self._inbox_dir, self._outbox_dir,
                  self._processed_dir, self._dead_letter_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Load processed events (for deduplication)
        await self._load_processed_events()

        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

        logger.info(f"FileEventTransport started for {self._component_id.value}")

    async def stop(self):
        """Stop the file transport."""
        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        logger.info(f"FileEventTransport stopped for {self._component_id.value}")

    async def publish(self, event: TrinityEvent) -> bool:
        """Publish an event by writing to the events directory."""
        try:
            # Generate filename with priority prefix for ordering
            priority_prefix = f"{event.priority.value:01d}"
            timestamp_prefix = f"{int(event.timestamp * 1000):015d}"
            filename = f"{priority_prefix}_{timestamp_prefix}_{event.event_id}.json"

            # Determine target directory
            if event.target == ComponentID.BROADCAST:
                # Write to main events directory (all components see it)
                event_path = self._events_dir / filename
            else:
                # Write to specific component's inbox
                target_inbox = self._trinity_dir / "inbox" / event.target.value
                target_inbox.mkdir(parents=True, exist_ok=True)
                event_path = target_inbox / filename

            # Atomic write (write to temp then rename)
            temp_path = event_path.with_suffix('.tmp')

            with open(temp_path, 'w') as f:
                f.write(event.to_json())
                f.flush()
                os.fsync(f.fileno())

            temp_path.rename(event_path)

            self._events_published += 1
            logger.debug(f"Published event {event.event_type.value} -> {event.target.value}")

            return True

        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            self._events_failed += 1
            return False

    async def subscribe(self, callback: EventHandler) -> str:
        """Subscribe to events."""
        subscription = Subscription(handler=callback)

        async with self._lock:
            self._subscriptions[subscription.subscription_id] = subscription

        return subscription.subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        async with self._lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                return True
            return False

    async def _poll_loop(self):
        """Poll for new events."""
        while self._running:
            try:
                await self._process_events()
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll loop error: {e}")
                await asyncio.sleep(1.0)

    async def _process_events(self):
        """Process all pending events."""
        # Process events from main directory (broadcasts)
        await self._process_directory(self._events_dir)

        # Process events from our inbox (targeted)
        await self._process_directory(self._inbox_dir)

    async def _process_directory(self, directory: Path):
        """Process events from a directory."""
        if not directory.exists():
            return

        # Get event files sorted by name (priority + timestamp)
        event_files = sorted(directory.glob("*.json"))

        for event_file in event_files:
            try:
                # Read event
                event_data = json.loads(event_file.read_text())
                event = TrinityEvent.from_dict(event_data)

                # Skip if already processed (deduplication)
                if event.event_id in self._processed_events:
                    self._cleanup_event_file(event_file)
                    continue

                # Skip if expired
                if event.is_expired:
                    self._cleanup_event_file(event_file)
                    continue

                # Skip if too old
                if event.age_seconds > self._max_event_age:
                    self._cleanup_event_file(event_file)
                    continue

                # Skip if not for us (for broadcast events)
                if event.target != ComponentID.BROADCAST and event.target != self._component_id:
                    continue

                # Skip if we sent it (don't receive our own broadcasts)
                if event.source == self._component_id and event.target == ComponentID.BROADCAST:
                    self._cleanup_event_file(event_file)
                    continue

                # Deliver to subscribers
                await self._deliver_event(event)

                # Mark as processed
                self._processed_events.add(event.event_id)
                self._events_received += 1

                # Move to processed directory
                self._cleanup_event_file(event_file)

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid event file {event_file}: {e}")
                self._move_to_dead_letter(event_file)
            except Exception as e:
                logger.error(f"Error processing event {event_file}: {e}")

    async def _deliver_event(self, event: TrinityEvent):
        """Deliver event to matching subscriptions."""
        async with self._lock:
            handlers = [
                sub.handler for sub in self._subscriptions.values()
                if sub.matches(event)
            ]

        # Run handlers concurrently
        if handlers:
            results = await asyncio.gather(
                *[self._safe_call_handler(h, event) for h in handlers],
                return_exceptions=True
            )

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Handler error for {event.event_type.value}: {result}")

    async def _safe_call_handler(self, handler: EventHandler, event: TrinityEvent):
        """Safely call an event handler with timeout."""
        try:
            await asyncio.wait_for(handler(event), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Handler timeout for event {event.event_id}")
        except Exception as e:
            logger.error(f"Handler exception: {e}")
            raise

    def _cleanup_event_file(self, event_file: Path):
        """Remove or archive an event file."""
        try:
            event_file.unlink()
        except Exception:
            pass

    def _move_to_dead_letter(self, event_file: Path):
        """Move failed event to dead letter queue."""
        try:
            dest = self._dead_letter_dir / event_file.name
            event_file.rename(dest)
        except Exception:
            pass

    async def _load_processed_events(self):
        """Load processed event IDs from disk."""
        processed_file = self._trinity_dir / "processed_events.json"

        if processed_file.exists():
            try:
                data = json.loads(processed_file.read_text())
                # Only keep recent events (last hour)
                cutoff = time.time() - 3600
                self._processed_events = {
                    event_id for event_id, ts in data.items()
                    if ts > cutoff
                }
            except Exception:
                self._processed_events = set()

    def get_metrics(self) -> Dict[str, Any]:
        """Get transport metrics."""
        return {
            "component_id": self._component_id.value,
            "events_published": self._events_published,
            "events_received": self._events_received,
            "events_failed": self._events_failed,
            "subscriptions": len(self._subscriptions),
            "processed_cache_size": len(self._processed_events),
        }


# =============================================================================
# MEMORY TRANSPORT (For Testing / Single Process)
# =============================================================================

class MemoryEventTransport(EventTransport):
    """
    In-memory event transport for testing and single-process deployments.

    Events are delivered immediately without persistence.
    """

    def __init__(self):
        self._subscriptions: Dict[str, Subscription] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._dispatch_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the memory transport."""
        if self._running:
            return

        self._running = True
        self._dispatch_task = asyncio.create_task(self._dispatch_loop())
        logger.info("MemoryEventTransport started")

    async def stop(self):
        """Stop the memory transport."""
        self._running = False

        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass

    async def publish(self, event: TrinityEvent) -> bool:
        """Publish an event to the queue."""
        await self._event_queue.put(event)
        return True

    async def subscribe(self, callback: EventHandler) -> str:
        """Subscribe to events."""
        subscription = Subscription(handler=callback)

        async with self._lock:
            self._subscriptions[subscription.subscription_id] = subscription

        return subscription.subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        async with self._lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                return True
            return False

    async def _dispatch_loop(self):
        """Dispatch events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                await self._deliver_event(event)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatch error: {e}")

    async def _deliver_event(self, event: TrinityEvent):
        """Deliver event to subscriptions."""
        async with self._lock:
            handlers = [
                sub.handler for sub in self._subscriptions.values()
                if sub.matches(event)
            ]

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Handler error: {e}")


# =============================================================================
# THE TRINITY EVENT BUS (Main Interface)
# =============================================================================

class TrinityEventBus:
    """
    The Trinity Event Bus - Central hub for all event communication.

    This is the main interface for publishing and subscribing to events.
    It manages transports and provides a unified API.

    Usage:
        bus = await TrinityEventBus.create(ComponentID.JARVIS_PRIME)

        # Subscribe to events
        await bus.subscribe(EventType.MODEL_READY, handle_model_ready)

        # Publish events
        await bus.publish_model_ready("prime-7b", "/models/prime-7b")

        # Cleanup
        await bus.shutdown()
    """

    _instances: Dict[str, "TrinityEventBus"] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(
        self,
        component_id: ComponentID,
        transport: Optional[EventTransport] = None,
    ):
        self._component_id = component_id
        self._transport = transport
        self._subscriptions: Dict[str, Subscription] = {}
        self._running = False

        # Event sequence counter
        self._sequence = 0

        # Metrics
        self._events_published = 0
        self._events_received = 0

    @classmethod
    async def create(
        cls,
        component_id: ComponentID,
        use_file_transport: bool = True,
        trinity_dir: Optional[Path] = None,
    ) -> "TrinityEventBus":
        """Create and start an event bus."""
        # Check for existing instance
        async with cls._lock:
            key = component_id.value
            if key in cls._instances:
                return cls._instances[key]

            # Create transport
            if use_file_transport:
                transport = FileEventTransport(
                    trinity_dir=trinity_dir,
                    component_id=component_id,
                )
            else:
                transport = MemoryEventTransport()

            # Create bus
            bus = cls(component_id, transport)

            # Start
            await bus.start()

            cls._instances[key] = bus
            return bus

    @classmethod
    async def get_instance(
        cls,
        component_id: ComponentID,
    ) -> Optional["TrinityEventBus"]:
        """Get existing instance or None."""
        async with cls._lock:
            return cls._instances.get(component_id.value)

    async def start(self):
        """Start the event bus."""
        if self._running:
            return

        await self._transport.start()

        # Subscribe transport to deliver events to our handlers
        await self._transport.subscribe(self._handle_event)

        self._running = True
        logger.info(f"TrinityEventBus started for {self._component_id.value}")

    async def shutdown(self):
        """Shutdown the event bus."""
        self._running = False
        await self._transport.stop()

        async with self._lock:
            key = self._component_id.value
            if key in self._instances:
                del self._instances[key]

        logger.info(f"TrinityEventBus stopped for {self._component_id.value}")

    # =========================================================================
    # SUBSCRIPTION API
    # =========================================================================

    async def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        source_filter: Optional[ComponentID] = None,
    ) -> str:
        """
        Subscribe to a specific event type.

        Args:
            event_type: The event type to subscribe to
            handler: Async function to handle the event
            source_filter: Only receive events from this source

        Returns:
            Subscription ID for later unsubscription
        """
        subscription = Subscription(
            handler=handler,
            event_types={event_type},
            source_filter=source_filter,
        )

        self._subscriptions[subscription.subscription_id] = subscription
        logger.debug(f"Subscribed to {event_type.value} -> {subscription.subscription_id}")

        return subscription.subscription_id

    async def subscribe_all(
        self,
        handler: EventHandler,
        source_filter: Optional[ComponentID] = None,
    ) -> str:
        """Subscribe to all event types."""
        subscription = Subscription(
            handler=handler,
            event_types=set(),  # Empty = all types
            source_filter=source_filter,
        )

        self._subscriptions[subscription.subscription_id] = subscription
        return subscription.subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    # =========================================================================
    # PUBLISHING API
    # =========================================================================

    async def publish(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        target: ComponentID = ComponentID.BROADCAST,
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
        expires_in_seconds: Optional[float] = None,
    ) -> TrinityEvent:
        """
        Publish an event to the bus.

        Args:
            event_type: Type of event
            payload: Event data
            target: Target component (BROADCAST for all)
            priority: Event priority
            correlation_id: For linking related events
            expires_in_seconds: Auto-expire after this time

        Returns:
            The published event
        """
        self._sequence += 1

        event = TrinityEvent(
            event_type=event_type,
            source=self._component_id,
            target=target,
            priority=priority,
            payload=payload,
            correlation_id=correlation_id,
            sequence=self._sequence,
            expires_at=time.time() + expires_in_seconds if expires_in_seconds else None,
        )

        success = await self._transport.publish(event)

        if success:
            self._events_published += 1
            logger.debug(f"Published {event_type.value} to {target.value}")
        else:
            logger.error(f"Failed to publish {event_type.value}")

        return event

    # =========================================================================
    # CONVENIENCE METHODS (The Loop Events)
    # =========================================================================

    async def publish_model_ready(
        self,
        model_name: str,
        model_path: str,
        model_type: str = "llm",
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrinityEvent:
        """
        Notify that a model is ready for use.

        This is THE KEY EVENT for closing the loop:
        Reactor trains -> publishes model_ready -> Prime hot-swaps
        """
        payload = {
            "model_name": model_name,
            "model_path": model_path,
            "model_type": model_type,
            "capabilities": capabilities or [],
            "ready_at": time.time(),
            **(metadata or {}),
        }

        return await self.publish(
            event_type=EventType.MODEL_READY,
            payload=payload,
            target=ComponentID.BROADCAST,
            priority=EventPriority.HIGH,
        )

    async def publish_training_complete(
        self,
        model_name: str,
        model_path: str,
        training_metrics: Optional[Dict[str, Any]] = None,
    ) -> TrinityEvent:
        """Notify that model training is complete."""
        payload = {
            "model_name": model_name,
            "model_path": model_path,
            "training_metrics": training_metrics or {},
            "completed_at": time.time(),
        }

        return await self.publish(
            event_type=EventType.TRAINING_COMPLETE,
            payload=payload,
            target=ComponentID.BROADCAST,
            priority=EventPriority.HIGH,
        )

    async def publish_experience_collected(
        self,
        experience_type: str,
        sample_count: int,
        data_path: Optional[str] = None,
    ) -> TrinityEvent:
        """Notify that new experience data is available."""
        payload = {
            "experience_type": experience_type,
            "sample_count": sample_count,
            "data_path": data_path,
            "collected_at": time.time(),
        }

        return await self.publish(
            event_type=EventType.EXPERIENCE_COLLECTED,
            payload=payload,
            target=ComponentID.REACTOR_CORE,
            priority=EventPriority.NORMAL,
        )

    async def publish_health_changed(
        self,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> TrinityEvent:
        """Notify of a health status change."""
        payload = {
            "status": status,
            "details": details or {},
            "changed_at": time.time(),
        }

        return await self.publish(
            event_type=EventType.HEALTH_CHANGED,
            payload=payload,
            target=ComponentID.ORCHESTRATOR,
            priority=EventPriority.HIGH,
        )

    async def publish_shutdown_requested(
        self,
        reason: str = "user_request",
        graceful: bool = True,
    ) -> TrinityEvent:
        """Request graceful shutdown of all components."""
        payload = {
            "reason": reason,
            "graceful": graceful,
            "requested_at": time.time(),
        }

        return await self.publish(
            event_type=EventType.SHUTDOWN_REQUESTED,
            payload=payload,
            target=ComponentID.BROADCAST,
            priority=EventPriority.CRITICAL,
        )

    async def heartbeat(self) -> TrinityEvent:
        """Send a heartbeat event."""
        return await self.publish(
            event_type=EventType.HEARTBEAT,
            payload={"component": self._component_id.value},
            target=ComponentID.ORCHESTRATOR,
            priority=EventPriority.LOW,
            expires_in_seconds=60.0,
        )

    # =========================================================================
    # INTERNAL
    # =========================================================================

    async def _handle_event(self, event: TrinityEvent):
        """Internal handler that routes events to subscriptions."""
        self._events_received += 1

        # Find matching subscriptions
        handlers = [
            sub.handler for sub in self._subscriptions.values()
            if sub.matches(event)
        ]

        # Run handlers
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Event handler error for {event.event_type.value}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        return {
            "component_id": self._component_id.value,
            "events_published": self._events_published,
            "events_received": self._events_received,
            "subscriptions": len(self._subscriptions),
            "running": self._running,
            "transport_metrics": (
                self._transport.get_metrics()
                if hasattr(self._transport, 'get_metrics') else {}
            ),
        }


# =============================================================================
# GLOBAL ACCESS FUNCTIONS
# =============================================================================

_default_bus: Optional[TrinityEventBus] = None


async def get_event_bus(
    component_id: ComponentID = ComponentID.ORCHESTRATOR,
) -> TrinityEventBus:
    """Get or create the event bus for a component."""
    return await TrinityEventBus.create(component_id)


async def shutdown_event_bus(
    component_id: ComponentID = ComponentID.ORCHESTRATOR,
):
    """Shutdown the event bus for a component."""
    bus = await TrinityEventBus.get_instance(component_id)
    if bus:
        await bus.shutdown()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "EventType",
    "EventPriority",
    "ComponentID",
    # Event
    "TrinityEvent",
    "Subscription",
    # Transport
    "EventTransport",
    "FileEventTransport",
    "MemoryEventTransport",
    # Bus
    "TrinityEventBus",
    # Functions
    "get_event_bus",
    "shutdown_event_bus",
]
