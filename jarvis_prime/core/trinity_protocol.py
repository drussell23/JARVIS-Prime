"""
Trinity Protocol - Cross-Repository IPC Communication
======================================================

v78.0 - Advanced Inter-Process Communication Protocol

The Trinity Protocol enables seamless communication between:
- JARVIS (Body): macOS LAM, action execution, UI control
- JARVIS-Prime (Mind): LLM inference, reasoning, cognition
- Reactor-Core (Soul): Training, fine-tuning, deployment

PROTOCOL LAYERS:
    1. Transport Layer: WebSocket, HTTP/2, Unix Domain Sockets
    2. Message Layer: Protocol Buffers / MessagePack / JSON
    3. Routing Layer: Service discovery, load balancing
    4. Security Layer: mTLS, token auth, encryption

COMMUNICATION PATTERNS:
    - Request/Response: Synchronous calls with timeout
    - Fire-and-Forget: Async notifications
    - Streaming: Real-time data flow
    - Pub/Sub: Event-driven messaging

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     TRINITY PROTOCOL BUS                        │
    │  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐ │
    │  │  JARVIS  │◄──►│ Message Bus  │◄──►│     JARVIS-Prime      │ │
    │  │  (Body)  │    │  (Router)    │    │       (Mind)          │ │
    │  └──────────┘    └──────────────┘    └───────────────────────┘ │
    │        ▲                │                       ▲              │
    │        │                ▼                       │              │
    │        │         ┌─────────────┐                │              │
    │        └────────►│ Reactor-Core │◄──────────────┘              │
    │                  │   (Soul)     │                              │
    │                  └─────────────┘                               │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import struct
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
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
    Set,
    Tuple,
    TypeVar,
    Union,
)
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOL CONSTANTS AND ENUMS
# =============================================================================


class TrinityNode(Enum):
    """Trinity ecosystem nodes."""

    JARVIS = "jarvis"           # Main LAM (Body)
    PRIME = "prime"             # LLM (Mind)
    REACTOR = "reactor"         # Training (Soul)
    SUPERVISOR = "supervisor"   # Orchestrator
    UNKNOWN = "unknown"


class MessageType(IntEnum):
    """Protocol message types."""

    # Control messages (0x00-0x0F)
    HEARTBEAT = 0x01
    HANDSHAKE = 0x02
    HANDSHAKE_ACK = 0x03
    DISCONNECT = 0x04
    ERROR = 0x0F

    # Request/Response (0x10-0x2F)
    REQUEST = 0x10
    RESPONSE = 0x11
    STREAMING_START = 0x12
    STREAMING_CHUNK = 0x13
    STREAMING_END = 0x14

    # Events (0x30-0x4F)
    EVENT = 0x30
    BROADCAST = 0x31
    SUBSCRIBE = 0x32
    UNSUBSCRIBE = 0x33

    # AGI specific (0x50-0x6F)
    INFERENCE_REQUEST = 0x50
    INFERENCE_RESPONSE = 0x51
    REASONING_REQUEST = 0x52
    REASONING_RESPONSE = 0x53
    LEARNING_UPDATE = 0x54
    MODEL_STATUS = 0x55

    # Action execution (0x70-0x8F)
    ACTION_REQUEST = 0x70
    ACTION_RESPONSE = 0x71
    ACTION_STREAM = 0x72
    SAFETY_CHECK = 0x73
    SAFETY_RESPONSE = 0x74


class MessagePriority(IntEnum):
    """Message priority levels."""

    CRITICAL = 0    # Safety, kill switch
    HIGH = 1        # User commands
    NORMAL = 2      # Regular operations
    LOW = 3         # Background tasks
    BULK = 4        # Batch operations


class TransportType(Enum):
    """Supported transport mechanisms."""

    WEBSOCKET = "websocket"
    HTTP2 = "http2"
    UNIX_SOCKET = "unix_socket"
    FILE_IPC = "file_ipc"       # Fallback for cross-process
    MEMORY = "memory"           # In-process (testing)


class ConnectionState(Enum):
    """Connection state machine."""

    DISCONNECTED = auto()
    CONNECTING = auto()
    HANDSHAKING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    CLOSING = auto()
    CLOSED = auto()


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TrinityMessage:
    """
    Core protocol message structure.

    Wire format:
    ┌────────────────────────────────────────────────────────────────┐
    │ Magic (4) │ Version (1) │ Type (1) │ Flags (2) │ Length (4)   │
    ├────────────────────────────────────────────────────────────────┤
    │ Message ID (16 bytes UUID)                                     │
    ├────────────────────────────────────────────────────────────────┤
    │ Correlation ID (16 bytes UUID, for responses)                  │
    ├────────────────────────────────────────────────────────────────┤
    │ Source Node (1) │ Target Node (1) │ Priority (1) │ Reserved   │
    ├────────────────────────────────────────────────────────────────┤
    │ Timestamp (8 bytes)                                            │
    ├────────────────────────────────────────────────────────────────┤
    │ Payload (variable length)                                      │
    ├────────────────────────────────────────────────────────────────┤
    │ HMAC Signature (32 bytes, optional)                            │
    └────────────────────────────────────────────────────────────────┘
    """

    # Header fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    message_type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL

    # Routing
    source: TrinityNode = TrinityNode.UNKNOWN
    target: TrinityNode = TrinityNode.UNKNOWN

    # Timing
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 30.0

    # Payload
    action: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Security
    signature: Optional[bytes] = None
    encrypted: bool = False

    # Protocol version
    version: int = 1

    # Flags
    requires_ack: bool = False
    is_streaming: bool = False

    def is_expired(self) -> bool:
        """Check if message has expired."""
        return (time.time() - self.timestamp) > self.ttl_seconds

    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        payload_json = json.dumps({
            "action": self.action,
            "payload": self.payload,
            "metadata": self.metadata,
        }).encode("utf-8")

        # Pack header
        magic = b"TRTY"
        header = struct.pack(
            "!4sBBHI",
            magic,
            self.version,
            self.message_type,
            0,  # flags
            len(payload_json),
        )

        # Pack IDs
        id_bytes = uuid.UUID(self.id).bytes
        corr_bytes = uuid.UUID(self.correlation_id).bytes if self.correlation_id else b"\x00" * 16

        # Pack routing
        routing = struct.pack(
            "!BBBxd",
            list(TrinityNode).index(self.source),
            list(TrinityNode).index(self.target),
            self.priority,
            self.timestamp,
        )

        return header + id_bytes + corr_bytes + routing + payload_json

    @classmethod
    def from_bytes(cls, data: bytes) -> "TrinityMessage":
        """Deserialize message from bytes."""
        # Unpack header
        magic, version, msg_type, flags, length = struct.unpack("!4sBBHI", data[:12])

        if magic != b"TRTY":
            raise ValueError("Invalid magic bytes")

        # Unpack IDs
        id_bytes = data[12:28]
        corr_bytes = data[28:44]

        # Unpack routing
        src, tgt, priority, _, timestamp = struct.unpack("!BBBxd", data[44:56])

        # Unpack payload
        payload_json = data[56:56 + length]
        payload_data = json.loads(payload_json.decode("utf-8"))

        return cls(
            id=str(uuid.UUID(bytes=id_bytes)),
            correlation_id=str(uuid.UUID(bytes=corr_bytes)) if corr_bytes != b"\x00" * 16 else None,
            message_type=MessageType(msg_type),
            priority=MessagePriority(priority),
            source=list(TrinityNode)[src],
            target=list(TrinityNode)[tgt],
            timestamp=timestamp,
            action=payload_data.get("action", ""),
            payload=payload_data.get("payload", {}),
            metadata=payload_data.get("metadata", {}),
            version=version,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "message_type": self.message_type.name,
            "priority": self.priority.name,
            "source": self.source.value,
            "target": self.target.value,
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "action": self.action,
            "payload": self.payload,
            "metadata": self.metadata,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrinityMessage":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            correlation_id=data.get("correlation_id"),
            message_type=MessageType[data.get("message_type", "REQUEST")],
            priority=MessagePriority[data.get("priority", "NORMAL")],
            source=TrinityNode(data.get("source", "unknown")),
            target=TrinityNode(data.get("target", "unknown")),
            timestamp=data.get("timestamp", time.time()),
            ttl_seconds=data.get("ttl_seconds", 30.0),
            action=data.get("action", ""),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
            version=data.get("version", 1),
        )


@dataclass
class TrinityResponse:
    """Response wrapper for Trinity messages."""

    success: bool
    message: TrinityMessage
    error: Optional[str] = None
    latency_ms: float = 0.0

    @classmethod
    def success_response(
        cls,
        request: TrinityMessage,
        payload: Dict[str, Any],
        source: TrinityNode,
    ) -> "TrinityResponse":
        """Create a success response."""
        response_msg = TrinityMessage(
            correlation_id=request.id,
            message_type=MessageType.RESPONSE,
            priority=request.priority,
            source=source,
            target=request.source,
            action=f"{request.action}_response",
            payload=payload,
        )
        return cls(success=True, message=response_msg)

    @classmethod
    def error_response(
        cls,
        request: TrinityMessage,
        error: str,
        source: TrinityNode,
    ) -> "TrinityResponse":
        """Create an error response."""
        response_msg = TrinityMessage(
            correlation_id=request.id,
            message_type=MessageType.ERROR,
            priority=MessagePriority.HIGH,
            source=source,
            target=request.source,
            action="error",
            payload={"error": error},
        )
        return cls(success=False, message=response_msg, error=error)


@dataclass
class NodeInfo:
    """Information about a Trinity node."""

    node: TrinityNode
    host: str = "localhost"
    port: int = 0
    transport: TransportType = TransportType.WEBSOCKET
    capabilities: Set[str] = field(default_factory=set)
    last_heartbeat: float = 0.0
    is_healthy: bool = False
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrinityConfig:
    """Configuration for Trinity Protocol."""

    # Identity
    node_type: TrinityNode = TrinityNode.PRIME
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Transport
    transport: TransportType = TransportType.FILE_IPC
    host: str = "localhost"
    port: int = 9100

    # File IPC paths
    ipc_base_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "trinity")

    # Security
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))
    enable_encryption: bool = False
    enable_signatures: bool = True

    # Timing
    heartbeat_interval: float = 5.0
    connection_timeout: float = 10.0
    request_timeout: float = 30.0
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0

    # Queue settings
    max_queue_size: int = 10000
    max_message_size: int = 10 * 1024 * 1024  # 10MB

    # Retry settings
    max_retries: int = 3
    retry_backoff: float = 1.5


# =============================================================================
# TRANSPORT LAYER
# =============================================================================


class Transport(ABC):
    """Abstract base for transport implementations."""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        ...

    @abstractmethod
    async def send(self, message: TrinityMessage) -> bool:
        """Send a message."""
        ...

    @abstractmethod
    async def receive(self) -> AsyncIterator[TrinityMessage]:
        """Receive messages."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        ...


class FileIPCTransport(Transport):
    """
    File-based IPC transport for cross-process communication.

    v79.1: Fixed race conditions and improved reliability:
    - Uses OrderedDict for proper FIFO eviction of processed IDs
    - Bounded message queue to prevent memory exhaustion
    - File locking to prevent concurrent read conflicts
    - Adaptive polling interval to reduce CPU when idle

    Directory structure:
    ~/.jarvis/trinity/
    ├── jarvis/
    │   ├── inbox/
    │   └── outbox/
    ├── prime/
    │   ├── inbox/
    │   └── outbox/
    └── reactor/
        ├── inbox/
        └── outbox/
    """

    # Configurable limits (can be overridden via environment)
    MAX_QUEUE_SIZE = int(os.getenv("TRINITY_MAX_QUEUE_SIZE", "10000"))
    MAX_PROCESSED_IDS = int(os.getenv("TRINITY_MAX_PROCESSED_IDS", "10000"))
    MIN_POLL_INTERVAL = float(os.getenv("TRINITY_MIN_POLL_INTERVAL", "0.05"))
    MAX_POLL_INTERVAL = float(os.getenv("TRINITY_MAX_POLL_INTERVAL", "1.0"))

    def __init__(self, config: TrinityConfig) -> None:
        self._config = config
        self._node = config.node_type
        self._base_dir = config.ipc_base_dir
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None
        # v79.1: Bounded queue to prevent memory exhaustion
        self._message_queue: asyncio.Queue[TrinityMessage] = asyncio.Queue(
            maxsize=self.MAX_QUEUE_SIZE
        )
        self._shutdown = asyncio.Event()
        # v79.1: Use OrderedDict for proper FIFO eviction (preserves insertion order)
        from collections import OrderedDict
        self._processed_ids: OrderedDict[str, float] = OrderedDict()  # id -> timestamp
        self._max_processed_ids = self.MAX_PROCESSED_IDS
        # v79.1: Adaptive polling interval
        self._current_poll_interval = self.MIN_POLL_INTERVAL
        # v79.1: File lock for inbox processing (platform-specific)
        self._inbox_lock = asyncio.Lock()

    @property
    def inbox_dir(self) -> Path:
        """Get inbox directory for this node."""
        return self._base_dir / self._node.value / "inbox"

    @property
    def outbox_dir(self) -> Path:
        """Get outbox directory for this node."""
        return self._base_dir / self._node.value / "outbox"

    def _get_target_inbox(self, target: TrinityNode) -> Path:
        """Get inbox directory for target node."""
        return self._base_dir / target.value / "inbox"

    async def connect(self) -> bool:
        """Initialize IPC directories."""
        try:
            # Create directories
            for node in TrinityNode:
                if node == TrinityNode.UNKNOWN:
                    continue
                inbox = self._base_dir / node.value / "inbox"
                outbox = self._base_dir / node.value / "outbox"
                inbox.mkdir(parents=True, exist_ok=True)
                outbox.mkdir(parents=True, exist_ok=True)

            # Start receive loop
            self._shutdown.clear()
            self._receive_task = asyncio.create_task(self._receive_loop())

            self._connected = True
            logger.info(f"FileIPC transport connected for {self._node.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize FileIPC: {e}")
            return False

    async def disconnect(self) -> None:
        """Cleanup IPC resources."""
        self._shutdown.set()

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        self._connected = False
        logger.info(f"FileIPC transport disconnected for {self._node.value}")

    async def send(self, message: TrinityMessage) -> bool:
        """Send message to target node via file."""
        try:
            target_inbox = self._get_target_inbox(message.target)

            # Create unique filename
            filename = f"{message.timestamp:.6f}_{message.id}.msg"
            filepath = target_inbox / filename

            # Write atomically (write to temp, then rename)
            temp_path = filepath.with_suffix(".tmp")

            content = json.dumps(message.to_dict())

            # Write in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._atomic_write, temp_path, filepath, content)

            logger.debug(f"Sent message {message.id} to {message.target.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    def _atomic_write(self, temp_path: Path, final_path: Path, content: str) -> None:
        """Atomically write file."""
        temp_path.write_text(content)
        temp_path.rename(final_path)

    async def receive(self) -> AsyncIterator[TrinityMessage]:
        """Yield received messages."""
        while not self._shutdown.is_set():
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message
            except asyncio.TimeoutError:
                continue

    async def _receive_loop(self) -> None:
        """
        Background loop to check inbox for messages.

        v79.1: Uses adaptive polling interval to reduce CPU when idle.
        """
        while not self._shutdown.is_set():
            try:
                had_messages = await self._check_inbox()

                # Adaptive polling: speed up when busy, slow down when idle
                if had_messages:
                    self._current_poll_interval = self.MIN_POLL_INTERVAL
                else:
                    # Exponential backoff when idle (max 1 second)
                    self._current_poll_interval = min(
                        self._current_poll_interval * 1.5,
                        self.MAX_POLL_INTERVAL
                    )

                await asyncio.sleep(self._current_poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                await asyncio.sleep(1.0)

    async def _check_inbox(self) -> bool:
        """
        Check inbox for new messages.

        v79.1: Added file locking to prevent concurrent read conflicts.

        Returns:
            True if any messages were processed, False otherwise.
        """
        # Use lock to prevent concurrent inbox processing
        async with self._inbox_lock:
            loop = asyncio.get_event_loop()
            processed_any = False

            # Get sorted list of message files
            try:
                msg_files = sorted(self.inbox_dir.glob("*.msg"))
            except Exception as e:
                logger.debug(f"Error listing inbox: {e}")
                return False

            for filepath in msg_files:
                try:
                    # v79.1: Try to acquire file lock before reading
                    # This prevents race conditions when multiple readers exist
                    content = await loop.run_in_executor(
                        None,
                        self._read_with_lock,
                        filepath
                    )

                    if content is None:
                        # Another process is reading this file, skip it
                        continue

                    data = json.loads(content)
                    message = TrinityMessage.from_dict(data)

                    # Check if already processed (using OrderedDict for O(1) lookup)
                    if message.id in self._processed_ids:
                        await loop.run_in_executor(None, self._safe_unlink, filepath)
                        continue

                    # Check if expired
                    if message.is_expired():
                        logger.warning(f"Dropping expired message {message.id}")
                        await loop.run_in_executor(None, self._safe_unlink, filepath)
                        continue

                    # Queue message (with timeout to handle full queue)
                    try:
                        await asyncio.wait_for(
                            self._message_queue.put(message),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Message queue full, dropping message {message.id}")
                        continue

                    # Mark as processed with timestamp (OrderedDict preserves order)
                    self._processed_ids[message.id] = time.time()

                    # v79.1: FIFO eviction - remove oldest entries when over limit
                    while len(self._processed_ids) > self._max_processed_ids:
                        self._processed_ids.popitem(last=False)  # Remove oldest (FIFO)

                    # Delete file
                    await loop.run_in_executor(None, self._safe_unlink, filepath)
                    processed_any = True

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in message file {filepath}: {e}")
                    # Move corrupted file to error directory
                    await loop.run_in_executor(None, self._quarantine_file, filepath)
                except Exception as e:
                    logger.error(f"Error processing message file {filepath}: {e}")

            return processed_any

    def _read_with_lock(self, filepath: Path) -> Optional[str]:
        """
        Read file with advisory file lock.

        v79.1: Prevents race conditions when multiple processes read.
        """
        import fcntl

        try:
            with open(filepath, 'r') as f:
                # Try to acquire exclusive lock (non-blocking)
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    # Another process has the lock
                    return None

                try:
                    content = f.read()
                    return content
                finally:
                    # Release lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except FileNotFoundError:
            # File was deleted by another process
            return None
        except Exception as e:
            logger.debug(f"Error reading {filepath}: {e}")
            return None

    def _safe_unlink(self, filepath: Path) -> None:
        """Safely delete a file, ignoring if already deleted."""
        try:
            filepath.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Error unlinking {filepath}: {e}")

    def _quarantine_file(self, filepath: Path) -> None:
        """Move corrupted file to error directory for debugging."""
        try:
            error_dir = self._base_dir / "errors"
            error_dir.mkdir(parents=True, exist_ok=True)
            dest = error_dir / f"{time.time():.6f}_{filepath.name}"
            filepath.rename(dest)
            logger.info(f"Quarantined corrupted file to {dest}")
        except Exception as e:
            logger.debug(f"Error quarantining {filepath}: {e}")
            self._safe_unlink(filepath)

    @property
    def is_connected(self) -> bool:
        return self._connected


class WebSocketTransport(Transport):
    """WebSocket-based transport for real-time communication."""

    def __init__(self, config: TrinityConfig) -> None:
        self._config = config
        self._connected = False
        self._ws = None
        self._receive_task: Optional[asyncio.Task] = None
        self._message_queue: asyncio.Queue[TrinityMessage] = asyncio.Queue()

    async def connect(self) -> bool:
        """Connect via WebSocket."""
        try:
            import websockets

            uri = f"ws://{self._config.host}:{self._config.port}/trinity"
            self._ws = await websockets.connect(
                uri,
                ping_interval=self._config.heartbeat_interval,
                ping_timeout=self._config.connection_timeout,
            )

            self._connected = True
            self._receive_task = asyncio.create_task(self._receive_loop())

            logger.info(f"WebSocket connected to {uri}")
            return True

        except ImportError:
            logger.warning("websockets not installed, falling back to FileIPC")
            return False
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect WebSocket."""
        if self._receive_task:
            self._receive_task.cancel()

        if self._ws:
            await self._ws.close()

        self._connected = False

    async def send(self, message: TrinityMessage) -> bool:
        """Send message via WebSocket."""
        if not self._ws:
            return False

        try:
            await self._ws.send(json.dumps(message.to_dict()))
            return True
        except Exception as e:
            logger.error(f"WebSocket send failed: {e}")
            return False

    async def receive(self) -> AsyncIterator[TrinityMessage]:
        """Receive messages from WebSocket."""
        while self._connected:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=1.0,
                )
                yield message
            except asyncio.TimeoutError:
                continue

    async def _receive_loop(self) -> None:
        """Background WebSocket receive loop."""
        if not self._ws:
            return

        try:
            async for raw_msg in self._ws:
                data = json.loads(raw_msg)
                message = TrinityMessage.from_dict(data)
                await self._message_queue.put(message)
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected


# =============================================================================
# MESSAGE HANDLERS
# =============================================================================


MessageHandler = Callable[[TrinityMessage], Awaitable[Optional[TrinityResponse]]]


class HandlerRegistry:
    """Registry for message handlers."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[MessageHandler]] = {}
        self._type_handlers: Dict[MessageType, List[MessageHandler]] = {}

    def register(
        self,
        action: Optional[str] = None,
        message_type: Optional[MessageType] = None,
    ) -> Callable[[MessageHandler], MessageHandler]:
        """Decorator to register a message handler."""
        def decorator(handler: MessageHandler) -> MessageHandler:
            if action:
                if action not in self._handlers:
                    self._handlers[action] = []
                self._handlers[action].append(handler)

            if message_type:
                if message_type not in self._type_handlers:
                    self._type_handlers[message_type] = []
                self._type_handlers[message_type].append(handler)

            return handler
        return decorator

    async def dispatch(self, message: TrinityMessage) -> Optional[TrinityResponse]:
        """Dispatch message to appropriate handlers."""
        handlers: List[MessageHandler] = []

        # Get action-specific handlers
        if message.action in self._handlers:
            handlers.extend(self._handlers[message.action])

        # Get type-specific handlers
        if message.message_type in self._type_handlers:
            handlers.extend(self._type_handlers[message.message_type])

        if not handlers:
            logger.warning(f"No handler for action={message.action}, type={message.message_type}")
            return None

        # Execute handlers (first response wins)
        for handler in handlers:
            try:
                response = await handler(message)
                if response:
                    return response
            except Exception as e:
                logger.error(f"Handler error for {message.action}: {e}")

        return None


# =============================================================================
# TRINITY CLIENT
# =============================================================================


class TrinityClient:
    """
    Client for Trinity Protocol communication.

    Provides high-level API for sending/receiving messages between nodes.

    Usage:
        client = TrinityClient(TrinityConfig(node_type=TrinityNode.PRIME))
        await client.connect()

        # Send request and wait for response
        response = await client.request(
            target=TrinityNode.JARVIS,
            action="execute_action",
            payload={"action_type": "click", "x": 100, "y": 200},
        )

        # Subscribe to events
        @client.on("screen_captured")
        async def handle_screen(message):
            process_screen(message.payload["image"])
    """

    def __init__(self, config: Optional[TrinityConfig] = None) -> None:
        self._config = config or TrinityConfig()
        self._transport: Optional[Transport] = None
        self._state = ConnectionState.DISCONNECTED
        self._handlers = HandlerRegistry()

        # Pending requests waiting for response
        self._pending: Dict[str, asyncio.Future] = {}

        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Node registry
        self._nodes: Dict[TrinityNode, NodeInfo] = {}

        # Metrics
        self._messages_sent = 0
        self._messages_received = 0
        self._errors = 0

        # Shutdown event
        self._shutdown = asyncio.Event()

    async def connect(self) -> bool:
        """Connect to Trinity network."""
        if self._state == ConnectionState.CONNECTED:
            return True

        self._state = ConnectionState.CONNECTING

        # Create transport based on config
        if self._config.transport == TransportType.WEBSOCKET:
            self._transport = WebSocketTransport(self._config)
        else:
            self._transport = FileIPCTransport(self._config)

        # Connect transport
        if not await self._transport.connect():
            self._state = ConnectionState.DISCONNECTED
            return False

        # Start background tasks
        self._shutdown.clear()
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        self._state = ConnectionState.CONNECTED
        logger.info(f"Trinity client connected as {self._config.node_type.value}")

        # Broadcast presence
        await self._broadcast_presence()

        return True

    async def disconnect(self) -> None:
        """Disconnect from Trinity network."""
        self._state = ConnectionState.CLOSING
        self._shutdown.set()

        # Cancel background tasks
        for task in [self._receive_task, self._heartbeat_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Disconnect transport
        if self._transport:
            await self._transport.disconnect()

        # Cancel pending requests
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()

        self._state = ConnectionState.DISCONNECTED
        logger.info("Trinity client disconnected")

    async def request(
        self,
        target: TrinityNode,
        action: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrinityResponse:
        """
        Send request and wait for response.

        Args:
            target: Target node
            action: Action name
            payload: Request payload
            timeout: Optional timeout override
            priority: Message priority
            metadata: Optional metadata

        Returns:
            TrinityResponse with result
        """
        timeout = timeout or self._config.request_timeout

        message = TrinityMessage(
            message_type=MessageType.REQUEST,
            priority=priority,
            source=self._config.node_type,
            target=target,
            action=action,
            payload=payload,
            metadata=metadata or {},
            requires_ack=True,
        )

        # Create future for response
        future: asyncio.Future[TrinityResponse] = asyncio.Future()
        self._pending[message.id] = future

        try:
            # Send message
            if not await self._send(message):
                return TrinityResponse(
                    success=False,
                    message=message,
                    error="Failed to send message",
                )

            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            self._pending.pop(message.id, None)
            return TrinityResponse(
                success=False,
                message=message,
                error=f"Request timed out after {timeout}s",
            )
        except Exception as e:
            self._pending.pop(message.id, None)
            return TrinityResponse(
                success=False,
                message=message,
                error=str(e),
            )

    async def send(
        self,
        target: TrinityNode,
        action: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.EVENT,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> bool:
        """
        Send message without waiting for response.

        Fire-and-forget pattern for events/notifications.
        """
        message = TrinityMessage(
            message_type=message_type,
            priority=priority,
            source=self._config.node_type,
            target=target,
            action=action,
            payload=payload,
        )

        return await self._send(message)

    async def broadcast(
        self,
        action: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> Dict[TrinityNode, bool]:
        """
        Broadcast message to all nodes.

        Returns dict of node -> success status.
        """
        results = {}

        for node in [TrinityNode.JARVIS, TrinityNode.PRIME, TrinityNode.REACTOR]:
            if node == self._config.node_type:
                continue

            success = await self.send(
                target=node,
                action=action,
                payload=payload,
                message_type=MessageType.BROADCAST,
                priority=priority,
            )
            results[node] = success

        return results

    def on(self, action: str) -> Callable[[MessageHandler], MessageHandler]:
        """Decorator to register action handler."""
        return self._handlers.register(action=action)

    def on_type(self, message_type: MessageType) -> Callable[[MessageHandler], MessageHandler]:
        """Decorator to register message type handler."""
        return self._handlers.register(message_type=message_type)

    async def _send(self, message: TrinityMessage) -> bool:
        """Internal send with signature."""
        if not self._transport or not self._transport.is_connected:
            logger.error("Transport not connected")
            return False

        # Sign message if enabled
        if self._config.enable_signatures:
            message.signature = self._sign_message(message)

        success = await self._transport.send(message)

        if success:
            self._messages_sent += 1
        else:
            self._errors += 1

        return success

    async def _receive_loop(self) -> None:
        """Background loop to process incoming messages."""
        if not self._transport:
            return

        try:
            async for message in self._transport.receive():
                if self._shutdown.is_set():
                    break

                await self._handle_message(message)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receive loop error: {e}")

    async def _handle_message(self, message: TrinityMessage) -> None:
        """Handle incoming message."""
        self._messages_received += 1

        # Verify signature if enabled
        if self._config.enable_signatures and message.signature:
            if not self._verify_signature(message):
                logger.warning(f"Invalid signature for message {message.id}")
                return

        # Check if response to pending request
        if message.correlation_id and message.correlation_id in self._pending:
            future = self._pending.pop(message.correlation_id)
            response = TrinityResponse(
                success=message.message_type != MessageType.ERROR,
                message=message,
                error=message.payload.get("error") if message.message_type == MessageType.ERROR else None,
            )
            future.set_result(response)
            return

        # Handle heartbeat
        if message.message_type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(message)
            return

        # Dispatch to handlers
        response = await self._handlers.dispatch(message)

        # Send response if needed
        if response and message.requires_ack:
            await self._send(response.message)

    async def _handle_heartbeat(self, message: TrinityMessage) -> None:
        """Handle heartbeat message."""
        source = message.source

        if source not in self._nodes:
            self._nodes[source] = NodeInfo(
                node=source,
                capabilities=set(message.payload.get("capabilities", [])),
                version=message.payload.get("version", "1.0.0"),
            )

        self._nodes[source].last_heartbeat = time.time()
        self._nodes[source].is_healthy = True

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while not self._shutdown.is_set():
            try:
                await self._broadcast_heartbeat()
                await asyncio.sleep(self._config.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _broadcast_heartbeat(self) -> None:
        """Broadcast heartbeat to all nodes."""
        await self.broadcast(
            action="heartbeat",
            payload={
                "node": self._config.node_type.value,
                "node_id": self._config.node_id,
                "capabilities": ["agi", "inference", "reasoning"],
                "version": "77.0",
                "timestamp": time.time(),
            },
            priority=MessagePriority.LOW,
        )

    async def _broadcast_presence(self) -> None:
        """Announce presence to network."""
        await self.broadcast(
            action="node_online",
            payload={
                "node": self._config.node_type.value,
                "node_id": self._config.node_id,
                "transport": self._config.transport.value,
                "host": self._config.host,
                "port": self._config.port,
            },
            priority=MessagePriority.HIGH,
        )

    def _sign_message(self, message: TrinityMessage) -> bytes:
        """Sign message with HMAC."""
        content = json.dumps(message.to_dict(), sort_keys=True).encode()
        return hmac.new(
            self._config.secret_key.encode(),
            content,
            hashlib.sha256,
        ).digest()

    def _verify_signature(self, message: TrinityMessage) -> bool:
        """Verify message signature."""
        if not message.signature:
            return False

        expected = self._sign_message(message)
        return hmac.compare_digest(expected, message.signature)

    def get_node_status(self, node: TrinityNode) -> Optional[NodeInfo]:
        """Get status of a specific node."""
        return self._nodes.get(node)

    def get_healthy_nodes(self) -> List[TrinityNode]:
        """Get list of healthy nodes."""
        now = time.time()
        healthy = []

        for node, info in self._nodes.items():
            if info.is_healthy and (now - info.last_heartbeat) < self._config.heartbeat_interval * 3:
                healthy.append(node)

        return healthy

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return {
            "state": self._state.name,
            "node_type": self._config.node_type.value,
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "errors": self._errors,
            "pending_requests": len(self._pending),
            "known_nodes": len(self._nodes),
            "healthy_nodes": len(self.get_healthy_nodes()),
        }

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================


_trinity_client: Optional[TrinityClient] = None
_trinity_lock = asyncio.Lock()


async def get_trinity_client(config: Optional[TrinityConfig] = None) -> TrinityClient:
    """Get or create global Trinity client."""
    global _trinity_client

    async with _trinity_lock:
        if _trinity_client is None:
            _trinity_client = TrinityClient(config)
            await _trinity_client.connect()

        return _trinity_client


async def shutdown_trinity() -> None:
    """Shutdown global Trinity client."""
    global _trinity_client

    if _trinity_client:
        await _trinity_client.disconnect()
        _trinity_client = None


# =============================================================================
# CONVENIENCE WRAPPERS
# =============================================================================


async def send_to_jarvis(action: str, payload: Dict[str, Any]) -> TrinityResponse:
    """Send request to JARVIS (Body)."""
    client = await get_trinity_client()
    return await client.request(TrinityNode.JARVIS, action, payload)


async def send_to_reactor(action: str, payload: Dict[str, Any]) -> TrinityResponse:
    """Send request to Reactor Core."""
    client = await get_trinity_client()
    return await client.request(TrinityNode.REACTOR, action, payload)


async def notify_all(action: str, payload: Dict[str, Any]) -> Dict[TrinityNode, bool]:
    """Broadcast notification to all nodes."""
    client = await get_trinity_client()
    return await client.broadcast(action, payload)
