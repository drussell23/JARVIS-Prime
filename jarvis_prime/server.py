"""
JARVIS-Prime Server - Tier-0 Muscle Memory Brain
=================================================

v84.0 - Trinity-Integrated with Advanced Async Patterns

Entry point for running JARVIS-Prime as an OpenAI-compatible API server.
Uses llama-cpp-python with GGUF models for efficient local inference.

TRINITY INTEGRATION:
    - Automatic connection to Trinity network on startup
    - Guaranteed event delivery with ACK and retry
    - OOM protection for parallel inference
    - Network partition detection
    - Graceful shutdown with Trinity notification

Usage:
    # Start server (auto-detects Metal GPU, connects to Trinity)
    python -m jarvis_prime.server

    # With custom settings
    python -m jarvis_prime.server --port 8000 --models-dir ./models

    # Disable Trinity (standalone mode)
    TRINITY_ENABLED=false python -m jarvis_prime.server

    # CPU-only mode (no GPU)
    python -m jarvis_prime.server --cpu-only

    # Test endpoint
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

Hardware Detection:
    - Apple Silicon (M1/M2/M3/M4): Full Metal GPU acceleration
    - NVIDIA GPU: CUDA acceleration
    - CPU: Optimized multi-threaded inference
"""

import argparse
import asyncio
import functools
import importlib.util
import json
import logging
import os
import signal
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
from enum import Enum, auto

# v96.0: Process fingerprinting for enhanced service registry
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# v93.0: Suppress known benign compatibility warnings from external libraries
# These are informational warnings about version compatibility, not errors
warnings.filterwarnings('ignore', message='.*scikit-learn version.*not supported.*')
warnings.filterwarnings('ignore', message='.*Torch version.*has not been tested.*')
warnings.filterwarnings('ignore', message='.*coremltools.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='coremltools.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='coremltools.*')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# v96.0: PROCESS FINGERPRINTING FOR SERVICE REGISTRY
# =============================================================================

def _get_process_fingerprint() -> Dict[str, Any]:
    """
    v96.0: Capture process fingerprint for enhanced service registry.

    This enables PID reuse detection and process identity validation.
    If the same PID is reused by a different process (after a crash),
    the fingerprint mismatch will detect this.

    Returns:
        Dictionary with process fingerprint data
    """
    fingerprint = {
        "pid": os.getpid(),
        "process_name": "",
        "process_cmdline": "",
        "process_exe_path": "",
        "process_cwd": "",
        "parent_pid": 0,
        "parent_name": "",
        "process_start_time": 0.0,
    }

    if PSUTIL_AVAILABLE:
        try:
            proc = psutil.Process()
            fingerprint["process_name"] = proc.name()
            fingerprint["process_cmdline"] = " ".join(proc.cmdline())
            fingerprint["process_exe_path"] = proc.exe()
            fingerprint["process_cwd"] = proc.cwd()
            fingerprint["process_start_time"] = proc.create_time()

            # Parent process info
            parent = proc.parent()
            if parent:
                fingerprint["parent_pid"] = parent.pid
                fingerprint["parent_name"] = parent.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
            logger.debug(f"[v96.0] Process fingerprint capture partial: {e}")

    return fingerprint


def _get_machine_id() -> str:
    """
    v96.0: Get unique machine identifier for distributed environments.

    This helps identify which machine owns a lock/registration when
    multiple machines may have the same PID.
    """
    # Try various methods to get machine ID
    machine_id_paths = [
        "/etc/machine-id",           # Linux
        "/var/lib/dbus/machine-id",  # Older Linux
    ]

    for path in machine_id_paths:
        try:
            if Path(path).exists():
                return Path(path).read_text().strip()
        except Exception:
            pass

    # Fallback: use hostname + some system info
    try:
        import platform
        return f"{platform.node()}-{uuid.getnode()}"
    except Exception:
        return f"unknown-{uuid.uuid4().hex[:8]}"


# =============================================================================
# v96.0: ATOMIC SHARED REGISTRY FOR CROSS-PROCESS COORDINATION
# =============================================================================

import fcntl
from contextlib import contextmanager

# Registry lock configuration
_REGISTRY_LOCK_TIMEOUT = 30.0  # Max seconds to wait for lock
_REGISTRY_LOCK_POLL_INTERVAL = 0.05  # Poll interval when waiting


@contextmanager
def _acquire_registry_lock(lock_path: Path, timeout: float = _REGISTRY_LOCK_TIMEOUT):
    """
    v96.0: Acquire exclusive lock on the registry using a separate lock file.

    This is the CRITICAL fix for race conditions - we lock a separate file
    and hold it for the entire read-modify-write cycle.

    Args:
        lock_path: Path to the lock file
        timeout: Max seconds to wait for lock

    Yields:
        The lock file handle

    Raises:
        TimeoutError: If lock cannot be acquired within timeout
    """
    # Ensure directory exists
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    # Create lock file if it doesn't exist
    lock_path.touch(exist_ok=True)

    start_time = time.time()
    lock_fd = None

    try:
        # Open lock file
        lock_fd = open(lock_path, 'r+')

        # Try to acquire lock with timeout
        while True:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Lock acquired!
                break
            except (IOError, OSError):
                # Lock is held by another process
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    lock_fd.close()
                    raise TimeoutError(
                        f"Could not acquire registry lock within {timeout}s. "
                        f"Another process may be holding the lock."
                    )
                time.sleep(_REGISTRY_LOCK_POLL_INTERVAL)

        yield lock_fd

    finally:
        # Release lock
        if lock_fd:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                lock_fd.close()
            except Exception:
                pass


def _atomic_register_service(
    service_name: str,
    service_data: Dict[str, Any],
    alternate_names: Optional[List[str]] = None,
    timeout: float = _REGISTRY_LOCK_TIMEOUT,
    cleanup_stale: bool = True,
    max_stale_age: float = 300.0,
) -> bool:
    """
    v96.0: Register a service with the shared registry atomically.

    This function ensures proper cross-process synchronization by:
    1. Acquiring an exclusive lock on a separate .lock file
    2. Cleaning up stale entries from previous runs
    3. Reading the current registry state
    4. Modifying it
    5. Writing it back atomically
    6. Releasing the lock

    Args:
        service_name: Primary service name
        service_data: Service data dict (pid, port, host, etc.)
        alternate_names: Optional list of alternate names to also register
        timeout: Max seconds to wait for lock
        cleanup_stale: Whether to clean up stale entries first
        max_stale_age: Max age in seconds before entry is considered stale

    Returns:
        True if registration succeeded, False otherwise
    """
    registry_dir = Path(os.getenv(
        "JARVIS_REGISTRY_DIR",
        str(Path.home() / ".jarvis" / "registry")
    ))
    registry_file = registry_dir / "services.json"
    lock_file = registry_dir / "services.json.lock"

    try:
        with _acquire_registry_lock(lock_file, timeout):
            # Read current registry (inside lock)
            existing_services = {}
            if registry_file.exists():
                try:
                    content = registry_file.read_text()
                    if content.strip():
                        existing_services = json.loads(content)
                        if not isinstance(existing_services, dict):
                            existing_services = {}
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"[v96.0] Registry read error (will overwrite): {e}")
                    existing_services = {}

            # v96.0: Clean up stale entries before registration
            if cleanup_stale and existing_services:
                current_time = time.time()
                entries_to_remove = []

                for name, data in list(existing_services.items()):
                    if not isinstance(data, dict):
                        entries_to_remove.append(name)
                        continue

                    # Check age
                    last_activity = max(
                        data.get("last_heartbeat", 0),
                        data.get("registered_at", 0)
                    )
                    age = current_time - last_activity

                    if age > max_stale_age:
                        entries_to_remove.append(name)
                        logger.debug(f"[v96.0] Removing stale: {name} (age: {age:.0f}s)")
                    else:
                        # Check if process is alive
                        pid = data.get("pid")
                        if pid and PSUTIL_AVAILABLE:
                            try:
                                if not psutil.pid_exists(pid):
                                    entries_to_remove.append(name)
                                    logger.debug(f"[v96.0] Removing dead: {name} (pid {pid})")
                            except Exception:
                                pass

                for name in entries_to_remove:
                    existing_services.pop(name, None)

                if entries_to_remove:
                    logger.info(f"[v96.0] Cleaned {len(entries_to_remove)} stale entries before registration")

            # Update registry
            existing_services[service_name] = service_data

            # Register alternate names
            if alternate_names:
                for alt_name in alternate_names:
                    existing_services[alt_name] = service_data

            # Write atomically (inside lock)
            registry_dir.mkdir(parents=True, exist_ok=True)
            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(existing_services, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            # Atomic rename
            temp_file.replace(registry_file)

        return True

    except TimeoutError as e:
        logger.error(f"[v96.0] Registry lock timeout for {service_name}: {e}")
        return False
    except Exception as e:
        logger.error(f"[v96.0] Failed to register {service_name}: {e}")
        return False


def _atomic_update_heartbeat(
    service_names: List[str],
    status: str = "running",
    timeout: float = _REGISTRY_LOCK_TIMEOUT,
) -> bool:
    """
    v97.0: Atomically update heartbeat timestamp in the service registry.

    This function ensures external services (like jarvis-prime) maintain
    fresh heartbeats in the shared registry so they don't get cleaned up
    as stale by the JARVIS service registry cleanup process.

    Args:
        service_names: List of service names to update heartbeat for
        status: Service status (running, starting, stopping, etc.)
        timeout: Max seconds to wait for lock

    Returns:
        True if heartbeat update succeeded, False otherwise
    """
    registry_dir = Path(os.getenv(
        "JARVIS_REGISTRY_DIR",
        str(Path.home() / ".jarvis" / "registry")
    ))
    registry_file = registry_dir / "services.json"
    lock_file = registry_dir / "services.json.lock"

    try:
        with _acquire_registry_lock(lock_file, timeout):
            # Read current registry
            existing_services = {}
            if registry_file.exists():
                try:
                    content = registry_file.read_text()
                    if content.strip():
                        existing_services = json.loads(content)
                        if not isinstance(existing_services, dict):
                            existing_services = {}
                except (json.JSONDecodeError, OSError):
                    return False

            # Update heartbeat for each service
            current_time = time.time()
            updated = False
            for name in service_names:
                if name in existing_services:
                    existing_services[name]["last_heartbeat"] = current_time
                    existing_services[name]["status"] = status
                    updated = True

            if not updated:
                return False  # Services not found in registry

            # Write back atomically
            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(existing_services, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            temp_file.replace(registry_file)

        return True

    except Exception as e:
        logger.debug(f"[v97.0] Heartbeat update error: {e}")
        return False


def _atomic_deregister_service(
    service_names: List[str],
    timeout: float = _REGISTRY_LOCK_TIMEOUT,
) -> bool:
    """
    v96.0: Deregister services from the shared registry atomically.

    Args:
        service_names: List of service names to remove
        timeout: Max seconds to wait for lock

    Returns:
        True if deregistration succeeded, False otherwise
    """
    registry_dir = Path(os.getenv(
        "JARVIS_REGISTRY_DIR",
        str(Path.home() / ".jarvis" / "registry")
    ))
    registry_file = registry_dir / "services.json"
    lock_file = registry_dir / "services.json.lock"

    try:
        with _acquire_registry_lock(lock_file, timeout):
            # Read current registry
            existing_services = {}
            if registry_file.exists():
                try:
                    content = registry_file.read_text()
                    if content.strip():
                        existing_services = json.loads(content)
                        if not isinstance(existing_services, dict):
                            return True  # Nothing to deregister
                except (json.JSONDecodeError, OSError):
                    return True  # Nothing to deregister

            # Remove services
            for name in service_names:
                existing_services.pop(name, None)

            # Write back atomically
            temp_file = registry_file.with_suffix(".tmp")
            with open(temp_file, 'w') as f:
                json.dump(existing_services, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            temp_file.replace(registry_file)

        return True

    except Exception as e:
        logger.debug(f"[v96.0] Deregistration error: {e}")
        return False


# =============================================================================
# v84.0: ADVANCED TRINITY INTEGRATION
# =============================================================================

class TrinityConnectionState(Enum):
    """Trinity connection states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    PARTITIONED = auto()
    RECONNECTING = auto()


@dataclass
class TrinityIntegration:
    """
    v95.0: Advanced Trinity integration for J-Prime with CONTINUOUS RECONNECTION.

    Features:
        - Automatic connection on startup
        - Network partition detection
        - OOM protection before inference
        - Graceful shutdown coordination
        - Heartbeat with guaranteed delivery
        - v95.0: CONTINUOUS reconnection (never gives up)
        - v95.0: Exponential backoff with configurable ceiling
        - v95.0: Lazy reconnection mode for extended disconnections

    CRITICAL FIX (v95.0):
        Previous versions gave up after max_reconnect_attempts and stayed in
        PARTITIONED state permanently. Now uses continuous retry with exponential
        backoff, capping at max_reconnect_interval but NEVER stopping.
    """
    # Configuration from environment (zero hardcoding)
    enabled: bool = field(default_factory=lambda: os.getenv("TRINITY_ENABLED", "true").lower() == "true")
    heartbeat_interval: float = field(default_factory=lambda: float(os.getenv("TRINITY_HEARTBEAT_INTERVAL", "5.0")))
    partition_threshold: float = field(default_factory=lambda: float(os.getenv("TRINITY_PARTITION_THRESHOLD", "30.0")))
    oom_memory_limit_mb: float = field(default_factory=lambda: float(os.getenv("OOM_MEMORY_LIMIT_MB", "8192")))
    oom_warning_threshold: float = field(default_factory=lambda: float(os.getenv("OOM_WARNING_THRESHOLD", "0.75")))
    reconnect_interval: float = field(default_factory=lambda: float(os.getenv("TRINITY_RECONNECT_INTERVAL", "5.0")))
    max_reconnect_attempts: int = field(default_factory=lambda: int(os.getenv("TRINITY_MAX_RECONNECT_ATTEMPTS", "10")))

    # v95.0: NEW - Continuous reconnection settings (CRITICAL FIX)
    # These ensure we NEVER give up trying to reconnect
    continuous_reconnect: bool = field(default_factory=lambda: os.getenv("TRINITY_CONTINUOUS_RECONNECT", "true").lower() == "true")
    max_reconnect_interval: float = field(default_factory=lambda: float(os.getenv("TRINITY_MAX_RECONNECT_INTERVAL", "60.0")))
    lazy_reconnect_threshold: float = field(default_factory=lambda: float(os.getenv("TRINITY_LAZY_RECONNECT_THRESHOLD", "300.0")))
    lazy_reconnect_interval: float = field(default_factory=lambda: float(os.getenv("TRINITY_LAZY_RECONNECT_INTERVAL", "120.0")))

    # State tracking
    state: TrinityConnectionState = TrinityConnectionState.DISCONNECTED
    last_heartbeat_time: float = 0.0
    last_jarvis_heartbeat: float = 0.0
    reconnect_attempts: int = 0
    start_time: float = field(default_factory=time.time)

    # Background tasks
    _heartbeat_task: Optional[asyncio.Task] = None
    _partition_detector_task: Optional[asyncio.Task] = None
    _reconnect_task: Optional[asyncio.Task] = None

    # Callbacks
    _partition_callbacks: List[Callable] = field(default_factory=list)
    _connection_callbacks: List[Callable] = field(default_factory=list)

    # Trinity bridge reference
    _trinity_bridge = None

    async def initialize(self, port: int, model_path: str = "", model_loaded: bool = False) -> bool:
        """
        Initialize Trinity integration.

        Args:
            port: Port J-Prime is running on
            model_path: Path to loaded model
            model_loaded: Whether model is loaded

        Returns:
            True if initialization succeeded
        """
        if not self.enabled:
            logger.info("[Trinity] Integration disabled (TRINITY_ENABLED=false)")
            return False

        self.state = TrinityConnectionState.CONNECTING
        logger.info("=" * 60)
        logger.info("v84.0 TRINITY INTEGRATION: Initializing J-Prime Connection")
        logger.info("=" * 60)

        try:
            # Import trinity_bridge
            from jarvis_prime.core.trinity_bridge import (
                initialize_trinity,
                update_model_status,
                set_model_health_callback,
                record_inference,
                TRINITY_ENABLED,
            )

            self._trinity_bridge = {
                "initialize": initialize_trinity,
                "update_model_status": update_model_status,
                "set_model_health_callback": set_model_health_callback,
                "record_inference": record_inference,
            }

            # Initialize Trinity connection
            success = await initialize_trinity(
                port=port,
                model_path=model_path,
                model_loaded=model_loaded,
            )

            if success:
                self.state = TrinityConnectionState.CONNECTED
                self.reconnect_attempts = 0
                logger.info("[Trinity] ✓ Connected to Trinity network")

                # Start background tasks
                self._heartbeat_task = asyncio.create_task(self._enhanced_heartbeat_loop())
                self._partition_detector_task = asyncio.create_task(self._partition_detection_loop())

                # Notify callbacks
                for callback in self._connection_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(True)
                        else:
                            callback(True)
                    except Exception as e:
                        logger.warning(f"[Trinity] Connection callback error: {e}")

                return True
            else:
                self.state = TrinityConnectionState.DISCONNECTED
                logger.warning("[Trinity] Failed to connect to Trinity network")
                return False

        except ImportError as e:
            logger.warning(f"[Trinity] Trinity bridge not available: {e}")
            self.state = TrinityConnectionState.DISCONNECTED
            return False
        except Exception as e:
            logger.error(f"[Trinity] Initialization error: {e}")
            self.state = TrinityConnectionState.DISCONNECTED
            return False

    async def shutdown(self) -> None:
        """Graceful shutdown of Trinity integration."""
        if self.state == TrinityConnectionState.DISCONNECTED:
            return

        logger.info("[Trinity] Shutting down integration...")

        # Cancel background tasks
        for task in [self._heartbeat_task, self._partition_detector_task, self._reconnect_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown trinity bridge
        try:
            from jarvis_prime.core.trinity_bridge import shutdown_trinity
            await shutdown_trinity()
        except Exception as e:
            logger.warning(f"[Trinity] Shutdown error: {e}")

        self.state = TrinityConnectionState.DISCONNECTED
        logger.info("[Trinity] Disconnected from Trinity network")

    async def _enhanced_heartbeat_loop(self) -> None:
        """Enhanced heartbeat with OOM check and guaranteed delivery."""
        while self.state in (TrinityConnectionState.CONNECTED, TrinityConnectionState.RECONNECTING):
            try:
                # Check memory before operations (OOM protection)
                if not await self._check_memory_safe():
                    logger.warning("[Trinity] Memory pressure detected, skipping heartbeat")
                    await asyncio.sleep(self.heartbeat_interval)
                    continue

                self.last_heartbeat_time = time.time()
                await asyncio.sleep(self.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[Trinity] Heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _partition_detection_loop(self) -> None:
        """
        v95.0: Continuous partition detection - runs in ALL active states.

        CRITICAL FIX: Previous versions only ran while CONNECTED or RECONNECTING,
        meaning the loop would exit when state became PARTITIONED, and would
        never restart the reconnection task even if it finished.

        Now runs in all states except DISCONNECTED, and ensures reconnection
        task is always active when partitioned.
        """
        # v95.0: Run in ALL states except DISCONNECTED (the shutdown state)
        while self.state != TrinityConnectionState.DISCONNECTED:
            try:
                # Check JARVIS heartbeat freshness
                jarvis_state_file = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_body.json"

                if jarvis_state_file.exists():
                    try:
                        import json
                        with open(jarvis_state_file, 'r') as f:
                            data = json.load(f)
                            jarvis_timestamp = data.get("timestamp", 0)
                            heartbeat_age = time.time() - jarvis_timestamp

                            if heartbeat_age > self.partition_threshold:
                                # Partition detected or continuing
                                if self.state == TrinityConnectionState.CONNECTED:
                                    logger.warning(
                                        f"[Trinity] JARVIS heartbeat stale ({heartbeat_age:.1f}s), "
                                        f"initiating partition recovery"
                                    )
                                    self.state = TrinityConnectionState.PARTITIONED
                                    await self._handle_partition()

                                # v95.0: CRITICAL - Ensure reconnection task is ALWAYS running
                                # when partitioned, even if it somehow finished
                                elif self.state in (
                                    TrinityConnectionState.PARTITIONED,
                                    TrinityConnectionState.RECONNECTING
                                ):
                                    if self._reconnect_task is None or self._reconnect_task.done():
                                        logger.warning(
                                            "[Trinity] Reconnection task not active - restarting"
                                        )
                                        self._reconnect_task = asyncio.create_task(
                                            self._reconnection_loop()
                                        )
                            else:
                                # Heartbeat is fresh - connection recovered
                                if self.state in (
                                    TrinityConnectionState.PARTITIONED,
                                    TrinityConnectionState.RECONNECTING
                                ):
                                    logger.info(
                                        "[Trinity] ✅ JARVIS heartbeat recovered, "
                                        f"partition resolved (age: {heartbeat_age:.1f}s)"
                                    )
                                    self.state = TrinityConnectionState.CONNECTED

                                    # Cancel reconnection task if running
                                    if self._reconnect_task and not self._reconnect_task.done():
                                        self._reconnect_task.cancel()
                                        try:
                                            await self._reconnect_task
                                        except asyncio.CancelledError:
                                            pass

                                self.last_jarvis_heartbeat = jarvis_timestamp

                    except json.JSONDecodeError:
                        pass  # File being written atomically
                else:
                    # No heartbeat file - JARVIS might not be running
                    if self.state == TrinityConnectionState.CONNECTED:
                        logger.warning(
                            "[Trinity] JARVIS heartbeat file missing - initiating recovery"
                        )
                        self.state = TrinityConnectionState.PARTITIONED
                        await self._handle_partition()

                # v95.0: Adaptive check interval
                # Check more frequently when partitioned/reconnecting
                if self.state in (
                    TrinityConnectionState.PARTITIONED,
                    TrinityConnectionState.RECONNECTING
                ):
                    await asyncio.sleep(2.0)  # Faster checks during recovery
                else:
                    await asyncio.sleep(5.0)  # Normal interval when connected

            except asyncio.CancelledError:
                logger.debug("[Trinity] Partition detection loop cancelled")
                break
            except Exception as e:
                logger.debug(f"[Trinity] Partition detection error: {e}")
                await asyncio.sleep(5.0)

    async def _handle_partition(self) -> None:
        """
        v95.0: Handle detected network partition with GUARANTEED reconnection.

        Key changes from v84.0:
        - Always ensures reconnection task is running
        - Doesn't check if already PARTITIONED (reconnection loop handles that)
        - Logs partition event for debugging
        """
        logger.warning("[Trinity] Network partition detected - initiating continuous recovery")

        # Notify callbacks
        for callback in self._partition_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.warning(f"[Trinity] Partition callback error: {e}")

        # v95.0: ALWAYS ensure reconnection task is running
        # This is the critical fix - previous versions could leave state stuck
        if self._reconnect_task is None or self._reconnect_task.done():
            logger.info("[Trinity] Starting continuous reconnection loop")
            self._reconnect_task = asyncio.create_task(self._reconnection_loop())
        else:
            logger.debug("[Trinity] Reconnection loop already active")

    async def _reconnection_loop(self) -> None:
        """
        v95.0: CONTINUOUS reconnection after partition - NEVER gives up.

        Previous versions gave up after max_reconnect_attempts, leaving J-Prime
        permanently disconnected. This version implements:

        1. Exponential backoff with ceiling (caps at max_reconnect_interval)
        2. Lazy mode after extended disconnection (uses longer intervals)
        3. NEVER stops trying - continuous retry until success or shutdown
        4. Immediate retry reset on successful connection

        The key insight: network partitions can be temporary (JARVIS restart,
        network blip) or extended (JARVIS not running). We handle both:
        - Short partitions: aggressive retry with exponential backoff
        - Long partitions: lazy retry to conserve resources
        """
        self.state = TrinityConnectionState.RECONNECTING
        partition_start_time = time.time()
        consecutive_failures = 0

        # v95.0: Continuous loop - NEVER exits except on success or shutdown
        while True:
            # Check for shutdown signal
            if self.state == TrinityConnectionState.DISCONNECTED:
                logger.info("[Trinity] Reconnection loop stopped - shutdown requested")
                return

            consecutive_failures += 1
            partition_duration = time.time() - partition_start_time

            # v95.0: Adaptive delay calculation
            # Phase 1: Exponential backoff (first max_reconnect_attempts)
            # Phase 2: Capped at max_reconnect_interval
            # Phase 3: Lazy mode after lazy_reconnect_threshold
            if partition_duration > self.lazy_reconnect_threshold:
                # Lazy mode: extended disconnection, use longer intervals
                base_delay = self.lazy_reconnect_interval
                mode = "lazy"
            elif consecutive_failures <= self.max_reconnect_attempts:
                # Aggressive mode: exponential backoff
                base_delay = min(
                    self.reconnect_interval * (2 ** (consecutive_failures - 1)),
                    self.max_reconnect_interval
                )
                mode = "aggressive"
            else:
                # Sustained mode: capped at max interval
                base_delay = self.max_reconnect_interval
                mode = "sustained"

            # Add jitter to prevent thundering herd (±10%)
            jitter = base_delay * 0.1 * ((hash(str(time.time())) % 20) - 10) / 10
            actual_delay = max(1.0, base_delay + jitter)  # Minimum 1 second

            # Log with appropriate verbosity (less frequent logging in lazy mode)
            if mode == "lazy":
                if consecutive_failures % 5 == 1:  # Log every 5th attempt in lazy mode
                    logger.info(
                        f"[Trinity] Reconnect attempt #{consecutive_failures} ({mode} mode) "
                        f"in {actual_delay:.1f}s (partition: {partition_duration:.0f}s)"
                    )
            else:
                logger.info(
                    f"[Trinity] Reconnect attempt #{consecutive_failures} ({mode} mode) "
                    f"in {actual_delay:.1f}s"
                )

            try:
                await asyncio.sleep(actual_delay)
            except asyncio.CancelledError:
                logger.info("[Trinity] Reconnection loop cancelled")
                return

            # v95.0: Multi-source connection check
            connected = await self._check_jarvis_connection()

            if connected:
                logger.info(
                    f"[Trinity] ✅ Reconnection successful after {consecutive_failures} attempts "
                    f"(partition duration: {partition_duration:.1f}s)"
                )
                self.state = TrinityConnectionState.CONNECTED
                self.reconnect_attempts = 0

                # Notify connection callbacks
                for callback in self._connection_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(True)
                        else:
                            callback(True)
                    except Exception as e:
                        logger.debug(f"[Trinity] Connection callback error: {e}")

                return  # Successfully reconnected

            # v95.0: Still partitioned, continue loop (NEVER give up)
            self.state = TrinityConnectionState.RECONNECTING

    async def _check_jarvis_connection(self) -> bool:
        """
        v95.0: Multi-source JARVIS connection verification.

        Checks multiple sources to determine if JARVIS Body is reachable:
        1. Heartbeat file freshness (primary)
        2. HTTP health endpoint (secondary)
        3. Process liveness via PID file (tertiary)

        Returns:
            True if JARVIS Body is confirmed reachable
        """
        jarvis_state_file = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_body.json"

        # Method 1: Heartbeat file check
        if jarvis_state_file.exists():
            try:
                import json
                with open(jarvis_state_file, 'r') as f:
                    data = json.load(f)
                    heartbeat_age = time.time() - data.get("timestamp", 0)

                    if heartbeat_age < self.partition_threshold:
                        self.last_jarvis_heartbeat = data.get("timestamp", 0)
                        return True
            except (json.JSONDecodeError, IOError):
                pass  # File being written or corrupted

        # Method 2: Direct HTTP health check (fallback)
        try:
            import aiohttp
            jarvis_port = int(os.getenv("JARVIS_PORT", "8010"))

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as session:
                async with session.get(f"http://localhost:{jarvis_port}/health") as resp:
                    if resp.status == 200:
                        logger.debug("[Trinity] JARVIS reachable via HTTP health check")
                        return True
        except Exception:
            pass  # Connection failed

        # Method 3: PID file check (last resort)
        pid_file = Path.home() / ".jarvis" / "jarvis.pid"
        if pid_file.exists():
            try:
                import psutil
                pid = int(pid_file.read_text().strip())
                if psutil.pid_exists(pid):
                    proc = psutil.Process(pid)
                    if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
                        logger.debug("[Trinity] JARVIS process alive via PID check")
                        # Process exists but heartbeat stale - might be starting
                        return False  # Don't confirm yet, wait for heartbeat
            except Exception:
                pass

        return False

    async def _check_memory_safe(self) -> bool:
        """Check if memory usage is safe for operations."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            used_percent = memory.percent / 100.0

            if used_percent > self.oom_warning_threshold:
                logger.warning(f"[Trinity] Memory usage at {used_percent*100:.1f}% (threshold: {self.oom_warning_threshold*100:.0f}%)")
                return False
            return True
        except ImportError:
            return True  # Can't check, assume safe

    def record_inference(self, latency_ms: float, success: bool = True) -> None:
        """Record inference metrics for Trinity heartbeat."""
        if self._trinity_bridge and "record_inference" in self._trinity_bridge:
            self._trinity_bridge["record_inference"](latency_ms, success)

    def update_model_status(self, loaded: bool, model_path: str = "") -> None:
        """Update model status in Trinity heartbeat."""
        if self._trinity_bridge and "update_model_status" in self._trinity_bridge:
            self._trinity_bridge["update_model_status"](loaded, model_path)

    def register_partition_callback(self, callback: Callable) -> None:
        """Register callback for partition events."""
        self._partition_callbacks.append(callback)

    def register_connection_callback(self, callback: Callable) -> None:
        """Register callback for connection events."""
        self._connection_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get current Trinity integration status."""
        return {
            "enabled": self.enabled,
            "state": self.state.name,
            "connected": self.state == TrinityConnectionState.CONNECTED,
            "partitioned": self.state == TrinityConnectionState.PARTITIONED,
            "uptime_seconds": time.time() - self.start_time,
            "last_heartbeat_time": self.last_heartbeat_time,
            "last_jarvis_heartbeat": self.last_jarvis_heartbeat,
            "reconnect_attempts": self.reconnect_attempts,
        }


# Global Trinity integration instance
_trinity_integration: Optional[TrinityIntegration] = None


def get_trinity_integration() -> Optional[TrinityIntegration]:
    """Get the global Trinity integration instance."""
    return _trinity_integration


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="JARVIS-Prime Tier-0 Brain Server (M1/Metal Optimized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        # v192.2: Changed from 8000 to 8001 to avoid conflict with jarvis-body/unified_supervisor
        default=int(os.getenv("JARVIS_PRIME_PORT", "8001")),
        help="Port to listen on",
    )

    # Model settings
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing model files (default: ~/.jarvis/prime/models)",
    )
    parser.add_argument(
        "--initial-model",
        type=str,
        default=None,
        help="Path to initial model to load (default: auto-detect)",
    )

    # Executor settings
    parser.add_argument(
        "--executor",
        type=str,
        choices=["llama-cpp", "transformers", "auto"],
        default="llama-cpp",
        help="Model executor backend",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Disable GPU acceleration (CPU only)",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all)",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=4096,
        help="Context window size",
    )

    # Monitoring
    parser.add_argument(
        "--telemetry-dir",
        type=str,
        default="./telemetry",
        help="Directory for telemetry logs",
    )
    parser.add_argument(
        "--reactor-core-dir",
        type=str,
        default=None,
        help="Directory to watch for reactor-core model updates",
    )

    # Server options
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    # Auto-download
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Auto-download recommended model if not found",
    )

    return parser.parse_args()


class StartupState:
    """
    v93.1: Track server startup state for immediate health checks.
    v221.0: Enhanced with progress percentage for cross-repo coordination.
    v223.0: Enterprise-Grade Async Startup - Granular stages and step tracking
            for SmartWatchdog integration.

    This allows the HTTP server to start IMMEDIATELY and respond to health
    checks while heavy initialization (ML imports, model loading) happens
    in the background.
    
    STAGES (v223.0):
    - booting: Server process starting, minimal initialization
    - importing: Heavy ML library imports (torch, etc.)
    - hardware_detect: Detecting GPU/CPU/Memory
    - downloading: Model download in progress (if needed)
    - loading_tensors: Loading model weights into memory
    - initializing: Model initialization and warmup
    - ready: Fully operational
    - error: Startup failed
    """
    # v221.0: Expected model load time for progress estimation
    DEFAULT_MODEL_LOAD_TIMEOUT = float(os.environ.get("JARVIS_PRIME_MODEL_TIMEOUT", "720"))  # 12 minutes default
    
    # v223.0: Stage weights for progress calculation (must sum to 100)
    STAGE_WEIGHTS = {
        "booting": 2,         # 0-2%
        "importing": 8,       # 2-10%
        "hardware_detect": 5, # 10-15%
        "downloading": 15,    # 15-30% (if applicable)
        "loading_tensors": 55,# 30-85% (bulk of time for large models)
        "initializing": 10,   # 85-95%
        "ready": 5,           # 95-100%
    }
    
    # v223.0: Stage descriptions for human-readable output
    STAGE_DESCRIPTIONS = {
        "booting": "Initializing server process",
        "importing": "Loading ML libraries",
        "hardware_detect": "Detecting hardware capabilities",
        "downloading": "Downloading model files",
        "loading_tensors": "Loading model tensors into memory",
        "initializing": "Initializing model and warmup",
        "ready": "Server operational",
        "error": "Startup failed",
    }
    
    def __init__(self):
        self.phase = "starting"  # starting -> initializing -> loading_model -> ready | error
        # v223.0: Granular stage tracking
        self.stage = "booting"   # Detailed stage within phase
        self.current_step = "Server starting..."  # Human-readable step description
        self.start_time = time.time()
        self.error: Optional[str] = None
        self.error_details: Optional[str] = None  # v223.0: Stack trace or additional details
        self.manager = None
        self.init_elapsed: Optional[float] = None
        self.model_load_start: Optional[float] = None
        self.model_load_elapsed: Optional[float] = None
        # v221.0: Track model loading timeout for progress calculation
        self.model_load_timeout: float = self.DEFAULT_MODEL_LOAD_TIMEOUT
        
        # v223.0: Detailed progress tracking
        self._stage_progress: float = 0.0  # Progress within current stage (0-1)
        self._stages_completed: List[str] = []
        self._last_progress_update: float = time.time()
        self._model_info: Dict[str, Any] = {}  # Model name, size, etc.
        
        # v223.0: Thread-safe progress updates
        import threading
        self._lock = threading.Lock()

        # v235.2: Optional APARS file integration (for golden-image startup)
        self._apars_file: Optional[str] = os.environ.get("JARVIS_APARS_FILE")
        self._apars_cache: Optional[Dict[str, Any]] = None
        self._apars_cache_mtime: float = 0.0

    def _read_apars_file(self) -> Optional[Dict[str, Any]]:
        """Load APARS payload from file if configured."""
        if not self._apars_file:
            self._apars_file = os.environ.get("JARVIS_APARS_FILE")
        if not self._apars_file:
            return None
        try:
            mtime = os.path.getmtime(self._apars_file)
        except FileNotFoundError:
            return None
        except Exception:
            return self._apars_cache
        if self._apars_cache is not None and mtime == self._apars_cache_mtime:
            return self._apars_cache
        try:
            with open(self._apars_file, "r") as f:
                payload = json.load(f)
            self._apars_cache = payload
            self._apars_cache_mtime = mtime
            return payload
        except Exception:
            return self._apars_cache
    
    def set_stage(self, stage: str, current_step: str = "", stage_progress: float = 0.0) -> None:
        """
        v223.0: Update the current stage with thread safety.
        
        Args:
            stage: One of the STAGE_* constants
            current_step: Human-readable description of current action
            stage_progress: Progress within this stage (0.0 to 1.0)
        """
        with self._lock:
            if self.stage != stage:
                if self.stage and self.stage not in self._stages_completed:
                    self._stages_completed.append(self.stage)
                self.stage = stage
                self._stage_progress = stage_progress
            else:
                self._stage_progress = stage_progress
            
            if current_step:
                self.current_step = current_step
            elif stage in self.STAGE_DESCRIPTIONS:
                self.current_step = self.STAGE_DESCRIPTIONS[stage]
            
            self._last_progress_update = time.time()
    
    def update_progress(self, stage_progress: float, current_step: Optional[str] = None) -> None:
        """
        v223.0: Update progress within current stage.
        
        Args:
            stage_progress: Progress within this stage (0.0 to 1.0)
            current_step: Optional updated step description
        """
        with self._lock:
            self._stage_progress = min(1.0, max(0.0, stage_progress))
            if current_step:
                self.current_step = current_step
            self._last_progress_update = time.time()
    
    def set_model_info(self, name: str = "", size_gb: float = 0.0, path: str = "") -> None:
        """v223.0: Set model information for progress display."""
        with self._lock:
            self._model_info = {
                "name": name,
                "size_gb": size_gb,
                "path": path,
            }
    
    def set_error(self, error: str, details: Optional[str] = None) -> None:
        """v223.0: Set error state with optional details."""
        with self._lock:
            self.stage = "error"
            self.phase = "error"
            self.error = error
            self.error_details = details
            self._last_progress_update = time.time()
    
    def _calculate_progress_pct(self) -> int:
        """
        v223.0: Calculate overall progress percentage based on completed stages
        and progress within current stage.
        
        Returns:
            Progress percentage (0-100)
        """
        if self.stage == "ready":
            return 100
        if self.stage == "error":
            # Return progress at failure point
            return self._calculate_base_progress()
        
        # Calculate base progress from completed stages
        base_progress = self._calculate_base_progress()
        
        # Add progress within current stage
        current_weight = self.STAGE_WEIGHTS.get(self.stage, 0)
        stage_contribution = int(current_weight * self._stage_progress)
        
        return min(99, base_progress + stage_contribution)  # Cap at 99 until ready
    
    def _calculate_base_progress(self) -> int:
        """Calculate progress from completed stages."""
        progress = 0
        for stage in self._stages_completed:
            progress += self.STAGE_WEIGHTS.get(stage, 0)
        return progress
    
    def get_startup_status(self) -> Dict[str, Any]:
        """
        v223.0: Get detailed startup status for /health/startup endpoint.
        
        This is the SINGLE SOURCE OF TRUTH for startup progress, designed for
        enterprise-grade SmartWatchdog integration.
        
        Returns:
            Structured JSON with status, progress, stage, step, and error info
        """
        with self._lock:
            elapsed = time.time() - self.start_time
            progress_pct = self._calculate_progress_pct()
            
            result = {
                # Primary status fields for SmartWatchdog
                "status": "error" if self.error else ("ready" if self.stage == "ready" else "loading"),
                "progress_pct": progress_pct,
                "current_step": self.current_step,
                
                # Detailed stage information
                "stage": self.stage,
                "stage_description": self.STAGE_DESCRIPTIONS.get(self.stage, self.stage),
                "stage_progress": round(self._stage_progress * 100, 1),
                
                # Timing information
                "elapsed_seconds": round(elapsed, 1),
                "last_progress_update": round(time.time() - self._last_progress_update, 1),
                
                # Process information
                "pid": os.getpid(),
                "timestamp": time.time(),
            }
            
            # Add model info if available
            if self._model_info:
                result["model"] = self._model_info
            
            # Add error details if in error state
            if self.error:
                result["error_details"] = self.error
                if self.error_details:
                    result["error_trace"] = self.error_details
            
            # Add ETA estimation
            if progress_pct > 5 and elapsed > 10 and progress_pct < 100:
                rate = progress_pct / elapsed
                remaining = (100 - progress_pct) / rate if rate > 0 else 600
                result["estimated_remaining_seconds"] = round(remaining, 0)
                result["estimated_total_seconds"] = round(elapsed + remaining, 0)
            
            return result

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status for /health endpoint.
        
        v221.0: Added model_load_progress_pct for cross-repo coordination with
        unified_supervisor. This allows the supervisor to track loading progress
        during Early Prime → Trinity handoff without progress regression.
        
        v223.0: Now uses granular stage tracking for accurate progress.
        """
        with self._lock:
            elapsed = time.time() - self.start_time
            progress_pct = self._calculate_progress_pct()
            
            result = {
                "status": "error" if self.error else ("healthy" if self.phase == "ready" else "starting"),
                "phase": self.phase,
                "startup_elapsed_seconds": round(elapsed, 1),
                "pid": os.getpid(),
            }
            if self.init_elapsed:
                result["init_elapsed_seconds"] = round(self.init_elapsed, 1)
            if self.model_load_elapsed:
                result["model_load_elapsed_seconds"] = round(self.model_load_elapsed, 1)
            if self.error:
                result["error"] = self.error
            
            # v223.0: Use granular progress calculation
            if self.phase in ("loading_model", "initializing") or self.stage in ("downloading", "loading_tensors", "initializing"):
                result["model_load_progress_pct"] = progress_pct
                result["model_loading_in_progress"] = True
                if self.model_load_start:
                    result["model_load_elapsed_seconds"] = round(time.time() - self.model_load_start, 1)
                result["model_load_timeout_seconds"] = self.model_load_timeout
                # v223.0: Add stage info for SmartWatchdog
                result["stage"] = self.stage
                result["current_step"] = self.current_step
            elif self.phase == "ready":
                result["model_load_progress_pct"] = 100
                result["model_loading_in_progress"] = False
                result["stage"] = "ready"
            else:
                # v223.0: Even in early phases, report progress
                result["model_load_progress_pct"] = progress_pct
                result["model_loading_in_progress"] = self.stage not in ("booting", "ready", "error")
                result["stage"] = self.stage
                result["current_step"] = self.current_step

            # v270.1: Always include ``ready_for_inference`` so PrimeClient's
            # health schema validation (v276.0) doesn't block hot-swap.
            # Previously this field was only present when APARS had it,
            # causing the schema check to always fail during normal operation.
            result.setdefault("ready_for_inference", self.phase == "ready")
            result.setdefault("model_loaded", self.phase == "ready")

            # v300.0: Phase 2 — Autonomy contract versions for boot check
            result["autonomy_schema_version"] = "1.0"
            result["contract_version"] = "1.0"

            # v235.2: Attach APARS payload if available (golden-image startup)
            apars_payload = self._read_apars_file()
            if apars_payload:
                result["apars"] = apars_payload
                # APARS values override the defaults set above.
                # v303.0: Coerce to bool — the progress file can contain
                # null when update_apars() is called without explicit
                # model_loaded/ready args.  None would fail PrimeClient's
                # v276.0 health schema validation ("expected bool got NoneType").
                if "ready_for_inference" in apars_payload:
                    _val = apars_payload["ready_for_inference"]
                    result["ready_for_inference"] = bool(_val) if _val is not None else result["ready_for_inference"]
                if "model_loaded" in apars_payload:
                    _val = apars_payload["model_loaded"]
                    result["model_loaded"] = bool(_val) if _val is not None else result["model_loaded"]

            return result


# Global startup state for immediate health checks
_startup_state: Optional[StartupState] = None

# v235.3: Module-level app reference for ASGI middleware import
# The APARS enrichment middleware (generated by gcp_vm_manager startup script)
# imports this to wrap /health responses with progress data during model loading.
# Assigned inside main() via `global app`.
app = None


async def main():
    """
    v93.1: Main entry point with IMMEDIATE HTTP server startup.

    CRITICAL FIX: The HTTP server now starts FIRST before any heavy imports
    or model loading. This ensures health checks succeed immediately while
    initialization happens in the background.

    Startup Flow:
    1. Start HTTP server immediately (responds with "starting" status)
    2. Run heavy initialization in background task
    3. Update status to "healthy" when ready
    """
    global _trinity_integration, _startup_state, app

    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default models directory
    models_dir = args.models_dir
    if models_dir is None:
        models_dir = str(Path.home() / ".jarvis" / "prime" / "models")

    # Print banner
    logger.info("=" * 70)
    logger.info("JARVIS-Prime Tier-0 Brain Server")
    logger.info("v93.1 - Immediate Startup with Background Initialization")
    logger.info("=" * 70)
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info("-" * 70)

    # Initialize startup state FIRST
    _startup_state = StartupState()

    try:
        # =====================================================================
        # STEP 1: Create minimal FastAPI app that responds to health checks
        # =====================================================================
        # v93.2: Extra warning suppression to ensure no ML warnings during startup
        import warnings
        warnings.filterwarnings('ignore', message='.*scikit-learn.*')
        warnings.filterwarnings('ignore', message='.*Torch.*')
        warnings.filterwarnings('ignore', message='.*coremltools.*')
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        from fastapi import FastAPI, HTTPException, Request

        app = FastAPI(
            title="JARVIS-Prime API",
            description="OpenAI-compatible API for JARVIS-Prime Tier-0 Brain",
            version="1.0.0",
        )

        @app.get("/health")
        async def health():
            """
            v93.1: Immediate health endpoint.

            Returns status IMMEDIATELY - even during heavy initialization.
            """
            status = _startup_state.get_status() if _startup_state else {"status": "starting"}

            if status.get("status") == "error":
                raise HTTPException(status_code=503, detail=status)

            return status

        @app.get("/health/startup")
        async def health_startup():
            """
            v223.0: Enterprise-Grade Startup Progress Endpoint.
            
            This is the SINGLE SOURCE OF TRUTH for startup progress, designed for
            SmartWatchdog integration in unified_supervisor.py.
            
            Returns a structured JSON with:
            - status: "booting" | "loading" | "ready" | "error"
            - progress_pct: 0-100 overall progress
            - current_step: Human-readable description (e.g., "Loading layer 12/32")
            - stage: Detailed stage name
            - error_details: Error info if status == "error"
            
            The SmartWatchdog uses this to:
            1. LIVENESS RULE: If progress_pct > last_seen, reset kill timer
            2. STALL RULE: If no progress for STALL_THRESHOLD, kill
            3. FAIL FAST RULE: If status == "error", kill immediately
            
            Example response during loading:
            {
                "status": "loading",
                "progress_pct": 45,
                "current_step": "Loading layer 12/32",
                "stage": "loading_tensors",
                "stage_description": "Loading model tensors into memory",
                "stage_progress": 52.5,
                "elapsed_seconds": 180.5,
                "estimated_remaining_seconds": 220.0,
                "pid": 12345,
                "model": {"name": "Mistral-7B", "size_gb": 14.5}
            }
            
            Example response when ready:
            {
                "status": "ready",
                "progress_pct": 100,
                "current_step": "Server operational",
                "stage": "ready",
                "elapsed_seconds": 420.0,
                "pid": 12345
            }
            """
            if not _startup_state:
                return {
                    "status": "booting",
                    "progress_pct": 0,
                    "current_step": "Server initializing...",
                    "stage": "booting",
                    "pid": os.getpid(),
                    "timestamp": time.time(),
                }
            
            return _startup_state.get_startup_status()

        @app.get("/v1/capability")
        async def capability():
            """
            Dedicated capability contract endpoint (NOT /health overload).

            Returns the server's static capability contract for boot-time
            assertion by unified_supervisor.  Callers assert contract_version
            is within [min_runtime_contract_version, max_runtime_contract_version]
            declared in brain_selection_policy.yaml.

            Response shape (contract-stable):
                schema_version       : semver string  e.g. "1.0"
                contract_version     : semver string  e.g. "1.0.0"
                model_loaded         : bool
                context_window       : int   (maximum prompt tokens this server accepts)
                host_id              : str   (stable machine identifier)
                host_binding         : str   "host:port"
                generated_at_epoch_s : int   (unix timestamp of this response)
                compute_class        : str   one of cpu|gpu_t4|gpu_l4|gpu_v100|gpu_a100
                model_id             : str   human-readable model name (no quant suffix)
                model_artifact       : str   basename of the loaded GGUF file
                gpu_layers           : int   n_gpu_layers on the loaded model
                tok_s_estimate       : float tokens-per-second estimate (0 if unknown)
                host                 : str   socket.gethostname()
            """
            import platform
            import re
            import socket

            model_loaded: bool = False
            if _startup_state:
                status = _startup_state.get_status()
                model_loaded = bool(status.get("model_loaded", False))

            # context_window: honour env override, else use server default 8192
            context_window = int(os.environ.get("JARVIS_PRIME_CONTEXT_WINDOW", "8192"))

            try:
                _hostname = socket.gethostname()
            except Exception:
                _hostname = "unknown"

            # Stable host identifier: prefer machine-id, fall back to hostname
            host_id: str = ""
            try:
                _mid = Path("/etc/machine-id")
                if _mid.exists():
                    host_id = _mid.read_text().strip()
            except Exception:
                pass
            if not host_id:
                try:
                    host_id = _hostname or platform.node() or "unknown"
                except Exception:
                    host_id = "unknown"

            # Derive host_binding from listen address
            host_binding = f"{args.host}:{args.port}"

            # --- compute_class / model fields -----------------------------------
            # The executor is a LlamaCppExecutor stored inside the hot-swap
            # manager's loader.  We access it defensively so that early
            # capability checks (before model loads) still return valid data.
            gpu_layers: int = 0
            model_artifact: str = ""
            model_id: str = ""
            compute_class: str = "cpu"

            _manager = getattr(_startup_state, "manager", None) if _startup_state else None
            _executor = None
            if _manager is not None:
                try:
                    _executor = _manager._hot_swap._loader._executor
                except Exception:
                    _executor = None

            if _executor is not None:
                try:
                    gpu_layers = int(_executor.config.n_gpu_layers)
                except Exception:
                    gpu_layers = 0

                try:
                    _mp = _executor._model_path  # Path | None
                    if _mp is not None:
                        model_artifact = Path(_mp).name
                        # Strip quantisation suffix and .gguf extension to get a
                        # clean human-readable model_id, e.g.:
                        #   "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
                        #   → "qwen2.5-coder-7b-instruct"
                        model_id = re.sub(
                            r"[-_]Q\d.*$",
                            "",
                            model_artifact.replace(".gguf", "").replace(".GGUF", ""),
                            flags=re.IGNORECASE,
                        )
                except Exception:
                    pass

                # Derive compute_class from gpu_layers and JARVIS_GPU_TYPE env var
                if gpu_layers == -1 or gpu_layers > 0:
                    _gpu_type = os.environ.get("JARVIS_GPU_TYPE", "").lower().strip()
                    _gpu_type_map = {
                        "l4": "gpu_l4",
                        "v100": "gpu_v100",
                        "a100": "gpu_a100",
                    }
                    compute_class = _gpu_type_map.get(_gpu_type, "gpu_t4")
                else:
                    compute_class = "cpu"

            # tok_s_estimate: optional throughput hint stored on the app object
            tok_s_estimate: float = float(
                getattr(app, "_tok_s_estimate", 0) or 0
            )

            return {
                "schema_version": "1.0",
                "contract_version": "1.0.0",
                "model_loaded": model_loaded,
                "context_window": context_window,
                "host_id": host_id,
                "host_binding": host_binding,
                "generated_at_epoch_s": int(time.time()),
                # --- Ouroboros compute-admission fields (schema_version 1.0) ---
                "compute_class": compute_class,
                "model_id": model_id,
                "model_artifact": model_artifact,
                "gpu_layers": gpu_layers,
                "tok_s_estimate": tok_s_estimate,
                "host": _hostname,
            }

        @app.get("/trinity/status")
        async def trinity_status():
            """Get Trinity integration status."""
            if _trinity_integration:
                return _trinity_integration.get_status()
            return {"enabled": False, "state": "DISABLED"}

        # Placeholder routes that will be replaced once manager is ready
        _completion_routes_ready = False

        @app.post("/v1/chat/completions")
        async def chat_completions_placeholder(request: Request):
            """Placeholder until model manager is ready.
            v153.0: Changed param from `dict` to `Request` — FastAPI treated `dict`
            as a query parameter, causing 422 on POST. `Request` accepts any body."""
            if not _completion_routes_ready:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Model still loading",
                        "status": _startup_state.get_status() if _startup_state else {}
                    }
                )
            return {"error": "Not implemented"}

        @app.post("/v1/completions")
        async def completions_placeholder(request: Request):
            """Placeholder until model manager is ready.
            v153.0: Changed param from `dict` to `Request` — see chat_completions_placeholder."""
            if not _completion_routes_ready:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "Model still loading",
                        "status": _startup_state.get_status() if _startup_state else {}
                    }
                )
            return {"error": "Not implemented"}

        # =====================================================================
        # STEP 2: Define background initialization (runs AFTER server starts)
        # =====================================================================
        # Handle shutdown
        shutdown_event = asyncio.Event()
        init_task: Optional[asyncio.Task] = None
        # v97.0: Registry heartbeat task to prevent stale cleanup
        registry_heartbeat_task: Optional[asyncio.Task] = None

        async def _registry_heartbeat_loop() -> None:
            """
            v97.0: Periodically update heartbeat in the shared service registry.

            This is CRITICAL for external services like jarvis-prime to prevent
            being marked as stale by the JARVIS service registry cleanup process.
            """
            service_names = ["jarvis_prime", "jarvis-prime", "jprime"]
            heartbeat_interval = 15.0
            consecutive_failures = 0

            logger.info(f"[v97.0] Registry heartbeat loop started for: {service_names}")

            while not shutdown_event.is_set():
                try:
                    await asyncio.sleep(heartbeat_interval)

                    if shutdown_event.is_set():
                        break

                    # Update heartbeat in service registry
                    success = _atomic_update_heartbeat(
                        service_names=service_names,
                        status="running"
                    )

                    if success:
                        consecutive_failures = 0
                        logger.debug(f"[v97.0] Registry heartbeat updated for {service_names[0]}")
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 5:
                            logger.warning(
                                f"[v97.0] {consecutive_failures} consecutive heartbeat failures"
                            )

                except asyncio.CancelledError:
                    logger.info("[v97.0] Registry heartbeat loop cancelled")
                    break
                except Exception as e:
                    logger.debug(f"[v97.0] Registry heartbeat error: {e}")
                    consecutive_failures += 1
                    await asyncio.sleep(1.0)

            logger.info("[v97.0] Registry heartbeat loop stopped")

        async def background_initialization():
            """
            v93.1: Run all heavy initialization in background.
            v192.0: Added hollow client mode support for slim hardware.
            v223.0: Enterprise-grade granular stage tracking for SmartWatchdog.

            This includes:
            - Heavy ML library imports (torch, scikit-learn, etc.)
            - Hardware detection
            - Model manager creation
            - Model loading (SKIPPED in hollow client mode)
            - Trinity integration
            - Heartbeat writer

            v223.0 Stages:
            - booting → importing → hardware_detect → [downloading] → loading_tensors → initializing → ready

            Hollow Client Mode (v192.0):
            - Activated by JARVIS_SKIP_LOCAL_MODEL_LOAD=true
            - Skips model loading entirely to prevent OOM on slim hardware
            - Server starts healthy but with model_loaded=false
            - Inference requests will be routed to GCP by JARVIS Body
            """
            nonlocal _completion_routes_ready
            global _trinity_integration  # v192.0: Declare early for hollow client mode

            try:
                _startup_state.phase = "initializing"
                _startup_state.set_stage("booting", "Background initialization starting...")
                init_start = time.time()

                # v192.0: Check for hollow client mode
                skip_local_model = os.environ.get("JARVIS_SKIP_LOCAL_MODEL_LOAD", "").lower() in ("true", "1", "yes", "on")
                hollow_client_mode = os.environ.get("JARVIS_HOLLOW_CLIENT_MODE", "").lower() in ("true", "1", "yes", "on")
                gcp_endpoint = os.environ.get("GCP_PRIME_ENDPOINT", "") or os.environ.get("JARVIS_GCP_PRIME_ENDPOINT", "")

                if skip_local_model or hollow_client_mode:
                    logger.info("=" * 70)
                    logger.info("[v192.0] 🔌 HOLLOW CLIENT MODE ACTIVATED")
                    logger.info("=" * 70)
                    logger.info(f"[v192.0] JARVIS_SKIP_LOCAL_MODEL_LOAD={skip_local_model}")
                    logger.info(f"[v192.0] JARVIS_HOLLOW_CLIENT_MODE={hollow_client_mode}")
                    if gcp_endpoint:
                        logger.info(f"[v192.0] GCP_PRIME_ENDPOINT={gcp_endpoint}")
                        logger.info("[v192.0] Inference will be routed to GCP VM")
                    else:
                        logger.info("[v192.0] No GCP endpoint - inference requests will fail")
                        logger.info("[v192.0] Waiting for GCP VM to become available...")
                    logger.info("=" * 70)

                    # Skip heavy model loading - just set up minimal server
                    _startup_state.phase = "ready"
                    _startup_state.set_stage("ready", "Hollow client mode - no local model")
                    _startup_state.init_elapsed = time.time() - init_start

                    # Initialize Trinity (without loaded model)
                    # Note: _trinity_integration is declared global later in the function
                    _trinity_integration = TrinityIntegration()
                    await _trinity_integration.initialize(
                        port=args.port,
                        model_path="",  # No local model
                        model_loaded=False,  # Explicitly mark as no model
                    )

                    logger.info(f"[v192.0] Hollow client ready in {_startup_state.init_elapsed:.1f}s")
                    logger.info("[v192.0] Health endpoint: http://127.0.0.1:{args.port}/health")
                    return  # Exit early - no model loading needed

                logger.info("[Background] Starting heavy initialization...")

                # v223.0: Stage - importing ML libraries
                _startup_state.set_stage("importing", "Importing ML libraries (torch, transformers)...")
                logger.info("[Background] Importing ML libraries...")
                from jarvis_prime.core.model_manager import PrimeModelManager, create_api_app
                _startup_state.update_progress(0.5, "ML libraries loaded, importing executors...")

                # v223.0: Stage - hardware detection
                _startup_state.set_stage("hardware_detect", "Detecting hardware capabilities...")
                try:
                    from jarvis_prime.core.llama_cpp_executor import HardwareDetector, LlamaCppConfig
                    hw = HardwareDetector.detect()
                    logger.info(f"[Background] Hardware: {hw.gpu_name or 'CPU'}")
                    logger.info(f"[Background] Backend: {hw.backend.name}")
                    logger.info(f"[Background] Memory: {hw.total_memory_gb:.1f} GB")
                    _startup_state.update_progress(1.0, f"Detected: {hw.gpu_name or 'CPU'}, {hw.total_memory_gb:.1f}GB RAM")
                except Exception as e:
                    logger.warning(f"[Background] Hardware detection failed: {e}")
                    _startup_state.update_progress(1.0, "Hardware detection skipped")

                # Select executor
                executor_class = None
                initial_model = args.initial_model
                model_path = None

                if args.executor == "llama-cpp":
                    from jarvis_prime.core.llama_cpp_executor import (
                        LlamaCppExecutor,
                        LlamaCppConfig,
                        GGUFModelDownloader,
                        get_default_model_path,
                    )

                    # Configure for hardware
                    if args.cpu_only:
                        executor_config = LlamaCppConfig.for_cpu(args.context_size)
                    else:
                        executor_config = LlamaCppConfig.auto_detect(args.context_size)
                        executor_config.n_gpu_layers = args.n_gpu_layers

                    executor_class = LlamaCppExecutor
                    logger.info(f"[Background] Using LlamaCppExecutor (n_gpu_layers={executor_config.n_gpu_layers})")

                    # Find model
                    if initial_model is None:
                        model_path_obj = get_default_model_path()
                        if model_path_obj.exists():
                            initial_model = str(model_path_obj)
                            logger.info(f"[Background] Found model: {model_path_obj.name}")
                        elif args.auto_download:
                            # v223.0: Stage - downloading model
                            _startup_state.set_stage("downloading", "Downloading model files...")
                            logger.info("[Background] Downloading model...")
                            downloader = GGUFModelDownloader(models_dir=Path(models_dir))
                            recommended = downloader.get_recommended_model()
                            if recommended:
                                _startup_state.update_progress(0.1, f"Downloading {recommended.filename}...")
                                model_path_obj = await downloader.download(
                                    f"{recommended.repo_id}/{recommended.filename}"
                                )
                                initial_model = str(model_path_obj)
                                _startup_state.update_progress(1.0, f"Downloaded {recommended.filename}")

                _startup_state.init_elapsed = time.time() - init_start
                logger.info(f"[Background] Initialization complete ({_startup_state.init_elapsed:.1f}s)")

                # Create manager
                manager = PrimeModelManager(
                    models_dir=models_dir,
                    telemetry_dir=args.telemetry_dir,
                    reactor_core_watch_dir=args.reactor_core_dir,
                    executor_class=executor_class,
                )
                _startup_state.manager = manager

                # v223.0: Stage - loading model tensors
                _startup_state.phase = "loading_model"
                _startup_state.model_load_start = time.time()
                model_path = Path(initial_model) if initial_model else None
                
                if model_path:
                    # Set model info for progress display
                    model_name = model_path.stem
                    try:
                        model_size_gb = model_path.stat().st_size / (1024 ** 3)
                    except Exception:
                        model_size_gb = 7.0  # Default estimate
                    _startup_state.set_model_info(name=model_name, size_gb=model_size_gb, path=str(model_path))
                    _startup_state.set_stage("loading_tensors", f"Loading {model_name} into memory...")
                    logger.info(f"[Background] Loading model: {model_name} ({model_size_gb:.1f}GB)")
                
                await manager.start(initial_model_path=model_path, background_model_load=True)

                # Wait for model to load with progress tracking
                if model_path:
                    logger.info(f"[Background] Waiting for model to load: {model_path.name}")
                    load_start = time.time()
                    estimated_load_time = _startup_state.model_load_timeout
                    poll_count = 0
                    
                    # v223.0: Progress tracking loop with granular updates
                    while manager.get_health_status() == "starting":
                        await asyncio.sleep(1.0)
                        if shutdown_event.is_set():
                            return
                        
                        poll_count += 1
                        elapsed = time.time() - load_start
                        
                        # Calculate progress based on elapsed time and estimated total
                        # Use sigmoid-like curve for more realistic progress
                        # (slower start, faster middle, slower end)
                        linear_progress = min(0.99, elapsed / estimated_load_time)
                        # Sigmoid adjustment: steeper in middle
                        adjusted_progress = linear_progress ** 0.8  # Slightly faster early progress
                        
                        # Update stage progress
                        _startup_state.update_progress(
                            adjusted_progress,
                            f"Loading model tensors... ({elapsed:.0f}s elapsed)"
                        )
                        
                        # Log progress periodically (every 30s)
                        if poll_count % 30 == 0:
                            progress_pct = int(adjusted_progress * 100)
                            logger.info(f"[Background] Model loading: {progress_pct}% ({elapsed:.0f}s)")
                    
                    _startup_state.model_load_elapsed = time.time() - _startup_state.model_load_start
                    logger.info(f"[Background] Model loaded ({_startup_state.model_load_elapsed:.1f}s)")
                    
                    # v223.0: Transition to initializing stage
                    _startup_state.set_stage("initializing", "Model loaded, warming up...")

                # Initialize Trinity
                # Note: global _trinity_integration is declared at function start (v192.0)
                _trinity_integration = TrinityIntegration()
                trinity_model_path = str(model_path) if model_path else ""
                await _trinity_integration.initialize(
                    port=args.port,
                    model_path=trinity_model_path,
                    model_loaded=model_path is not None and model_path.exists(),
                )

                # v109.2: Replace routes with real API - with defensive error handling
                logger.debug("[v109.2] Creating real API app...")
                try:
                    real_app = create_api_app(manager)
                    logger.debug(f"[v109.2] Real API app created with {len(real_app.routes)} routes")
                except Exception as e:
                    logger.error(f"[v109.2] Failed to create real API app: {e}")
                    import traceback
                    logger.error(f"[v109.2] create_api_app traceback:\n{traceback.format_exc()}")
                    raise

                # Copy real routes to main app
                # v109.2: Use proper route mounting that works across FastAPI versions
                # CRITICAL FIX: In newer FastAPI/Starlette, app.routes is a property
                # that cannot be directly assigned. We use include_router or mount instead.
                logger.debug("[v109.2] Copying routes to main app...")
                try:
                    paths_to_skip = {'/health', '/trinity/status'}

                    for route in real_app.routes:
                        route_path = getattr(route, 'path', None)
                        if route_path and route_path not in paths_to_skip:
                            # v109.2: Use add_api_route for proper route registration
                            # This works across all FastAPI/Starlette versions
                            if hasattr(route, 'endpoint') and hasattr(route, 'methods'):
                                # APIRoute - has endpoint and methods
                                try:
                                    # Remove existing route with same path if any
                                    # Use router's internal list which IS mutable
                                    app.router.routes = [
                                        r for r in app.router.routes
                                        if getattr(r, 'path', None) != route_path
                                    ]
                                    # Add the new route
                                    app.router.routes.append(route)
                                except AttributeError:
                                    # Fallback: try direct route list modification via internal
                                    logger.debug(f"[v109.2] Fallback route addition for {route_path}")
                                    if hasattr(app, '_routes'):
                                        app._routes = [
                                            r for r in app._routes
                                            if getattr(r, 'path', None) != route_path
                                        ]
                                        app._routes.append(route)
                                    else:
                                        # Last resort: use include_router pattern
                                        logger.warning(f"[v109.2] Cannot add route {route_path} - no mutable route list")
                            else:
                                # Other route types (Mount, etc.)
                                logger.debug(f"[v109.2] Skipping non-API route: {route_path}")

                    logger.debug(f"[v109.2] Route replacement complete: {len(app.router.routes)} total routes")
                except Exception as e:
                    logger.error(f"[v109.2] Failed to replace routes: {e}")
                    import traceback
                    logger.error(f"[v109.2] Route replacement traceback:\n{traceback.format_exc()}")
                    raise

                _completion_routes_ready = True

                # Start heartbeat writer
                await start_heartbeat_writer(args, manager, model_path)

                # v223.0: Mark as ready with proper stage tracking
                _startup_state.phase = "ready"
                _startup_state.set_stage("ready", "Server operational - accepting requests")
                total_time = time.time() - _startup_state.start_time
                logger.info("=" * 70)
                logger.info(f"[v223.0] JARVIS-Prime fully ready in {total_time:.1f}s")
                logger.info(f"[v223.0] /health/startup endpoint: http://{args.host}:{args.port}/health/startup")
                logger.info(f"API endpoint: http://{args.host}:{args.port}/v1/chat/completions")
                logger.info("=" * 70)

            except Exception as e:
                # v223.0: Enhanced error logging with full traceback for SmartWatchdog
                import traceback
                full_traceback = traceback.format_exc()
                logger.error(f"[Background] Initialization failed: {e}")
                logger.error(f"[Background] Full traceback:\n{full_traceback}")
                _startup_state.phase = "error"
                _startup_state.set_error(str(e), full_traceback)

        async def start_heartbeat_writer(args, manager, model_path):
            """Start the heartbeat writer task."""
            import tempfile

            heartbeat_file = Path.home() / ".jarvis" / "trinity" / "components" / "jarvis_prime.json"
            trinity_dir = heartbeat_file.parent
            trinity_dir.mkdir(parents=True, exist_ok=True)

            heartbeat_interval = float(os.getenv("JARVIS_PRIME_HEARTBEAT_INTERVAL", "5.0"))

            async def writer():
                while not shutdown_event.is_set():
                    try:
                        heartbeat_data = {
                            "component": "jarvis_prime",
                            "component_type": "j_prime",
                            "instance_id": f"jprime-{os.getpid()}-{int(time.time())}",
                            "pid": os.getpid(),
                            "port": args.port,
                            "host": args.host,
                            "endpoint": f"http://localhost:{args.port}",
                            "api_format": "openai",
                            "model_loaded": manager.current_model is not None if hasattr(manager, 'current_model') else False,
                            "model_name": str(manager.current_model.name) if hasattr(manager, 'current_model') and manager.current_model else None,
                            "model_path": str(model_path) if model_path else None,
                            "status": _startup_state.phase if _startup_state else "unknown",
                            "healthy": _startup_state.phase == "ready" if _startup_state else False,
                            "timestamp": time.time(),
                            "uptime_seconds": time.time() - _startup_state.start_time if _startup_state else 0,
                        }

                        try:
                            import psutil
                            proc = psutil.Process()
                            heartbeat_data["cpu_percent"] = proc.cpu_percent()
                            heartbeat_data["memory_mb"] = proc.memory_info().rss / (1024 * 1024)
                        except ImportError:
                            pass

                        if _trinity_integration:
                            trinity_status = _trinity_integration.get_status()
                            heartbeat_data["trinity_connected"] = trinity_status.get("connected", False)
                            heartbeat_data["trinity_state"] = trinity_status.get("state", "UNKNOWN")

                        # Atomic write
                        tmp_fd, tmp_name = tempfile.mkstemp(dir=trinity_dir, prefix=".jprime.", suffix=".tmp")
                        try:
                            with os.fdopen(tmp_fd, 'w') as f:
                                json.dump(heartbeat_data, f, indent=2)
                                f.flush()
                                os.fsync(f.fileno())
                            os.replace(tmp_name, heartbeat_file)
                        except Exception:
                            if os.path.exists(tmp_name):
                                os.unlink(tmp_name)

                        await asyncio.sleep(heartbeat_interval)

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.debug(f"[Heartbeat] Error: {e}")
                        await asyncio.sleep(heartbeat_interval)

            asyncio.create_task(writer())
            logger.info(f"[Heartbeat] Writer started (file={heartbeat_file})")

        # =====================================================================
        # STEP 3: Use FastAPI startup event to trigger background initialization
        # =====================================================================
        # CRITICAL: This ensures heavy initialization ONLY starts AFTER
        # uvicorn has bound to the port and is accepting connections.
        # This is the key fix for the 61.9s timeout issue.
        @app.on_event("startup")
        async def on_startup():
            """
            v95.0: Trigger background initialization and SERVICE REGISTRATION.

            This is the critical fix - the startup event fires AFTER uvicorn
            has successfully bound to the port and is ready to accept connections.
            Health checks will succeed immediately while heavy init runs.

            v95.0 Enhancement:
            - Registers JARVIS Prime with the shared service registry
            - This enables TrinityIntegrator to verify registration (not just HTTP discovery)
            - Fixes the timing issue where components were marked "pending"
            """
            nonlocal init_task
            logger.info("=" * 70)
            logger.info("[v95.0] Server is now LISTENING - starting background initialization")
            logger.info(f"Health endpoint ready: http://{args.host}:{args.port}/health")
            logger.info("=" * 70)

            # v96.0: Register with shared service registry using ATOMIC locking
            # This is CRITICAL for TrinityIntegrator's registration-aware verification
            # Uses file locking to prevent race conditions between multiple services

            # Capture process fingerprint for PID reuse detection
            fingerprint = _get_process_fingerprint()
            machine_id = _get_machine_id()

            # Determine configured port vs actual port
            configured_port = int(os.getenv("JARVIS_PRIME_PORT", "8000"))
            actual_port = args.port
            is_fallback = actual_port != configured_port

            # Build service data
            service_data = {
                "service_name": "jarvis_prime",
                "pid": os.getpid(),
                "port": actual_port,
                "host": args.host,
                "health_endpoint": "/health",
                "status": "starting",
                "registered_at": time.time(),
                "last_heartbeat": time.time(),
                "metadata": {
                    "version": "4.0.0",
                    "role": "inference",
                    "gpu": os.getenv("JARVIS_PRIME_GPU", "auto"),
                },
                # v96.0: Port fallback tracking (Problem 17 fix)
                "primary_port": configured_port,
                "is_fallback_port": is_fallback,
                "fallback_reason": f"Port {configured_port} unavailable" if is_fallback else "",
                "ports_tried": [configured_port] if is_fallback else [],
                "port_allocation_time": time.time(),
                # v96.0: Process fingerprint for identity validation (Problem 18 fix)
                "process_name": fingerprint["process_name"],
                "process_cmdline": fingerprint["process_cmdline"],
                "process_exe_path": fingerprint["process_exe_path"],
                "process_cwd": fingerprint["process_cwd"],
                "parent_pid": fingerprint["parent_pid"],
                "parent_name": fingerprint["parent_name"],
                "process_start_time": fingerprint["process_start_time"],
                # v96.0: Machine ID for distributed environments
                "machine_id": machine_id,
            }

            # Use atomic registration with file locking
            if _atomic_register_service(
                service_name="jarvis_prime",
                service_data=service_data,
                alternate_names=["jarvis-prime", "jprime"],
            ):
                fallback_msg = f" (fallback from {configured_port})" if is_fallback else ""
                logger.info(
                    f"[v96.0] ✅ Registered with service registry using atomic lock "
                    f"(port={actual_port}{fallback_msg}, pid={os.getpid()})"
                )

                # v97.0: Start registry heartbeat loop to prevent stale cleanup
                nonlocal registry_heartbeat_task
                registry_heartbeat_task = asyncio.create_task(
                    _registry_heartbeat_loop(),
                    name="jarvis_prime_registry_heartbeat"
                )
                logger.info("[v97.0] ✅ Registry heartbeat started (interval: 15s)")
            else:
                logger.warning("[v96.0] Service registry registration failed (non-fatal)")

            init_task = asyncio.create_task(background_initialization())

        @app.on_event("shutdown")
        async def on_shutdown():
            """
            v97.0: Clean up on server shutdown with service deregistration.

            Ensures JARVIS Prime is removed from the shared service registry
            so TrinityIntegrator knows it's no longer available.
            """
            nonlocal init_task, registry_heartbeat_task

            # v97.0: Stop registry heartbeat loop first
            if registry_heartbeat_task and not registry_heartbeat_task.done():
                registry_heartbeat_task.cancel()
                try:
                    await asyncio.wait_for(registry_heartbeat_task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                logger.info("[v97.0] Registry heartbeat stopped")

            # v96.0: Deregister from shared service registry using atomic locking
            if _atomic_deregister_service(["jarvis_prime", "jarvis-prime", "jprime"]):
                logger.info("[v96.0] ✅ Deregistered from service registry")
            else:
                logger.debug("[v96.0] Service deregistration completed (may have already been removed)")

            if init_task and not init_task.done():
                init_task.cancel()
                try:
                    await init_task
                except asyncio.CancelledError:
                    pass

            if _startup_state and _startup_state.manager:
                try:
                    await _startup_state.manager.stop()
                except Exception as e:
                    logger.debug(f"Manager cleanup error: {e}")

            if _trinity_integration:
                try:
                    await _trinity_integration.shutdown()
                except Exception as e:
                    logger.debug(f"Trinity cleanup error: {e}")

            logger.info("Server shutdown complete")

        # =====================================================================
        # STEP 4: Start uvicorn server
        # =====================================================================
        import uvicorn

        config = uvicorn.Config(
            app,
            host=args.host,
            port=args.port,
            workers=1,  # Single worker for startup
            log_level="debug" if args.debug else "info",
        )

        server = uvicorn.Server(config)

        async def handle_shutdown():
            """Handle graceful shutdown."""
            logger.info("Shutdown signal received")
            shutdown_event.set()
            server.should_exit = True

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(handle_shutdown()))

        logger.info("=" * 70)
        logger.info("[v93.2] Starting uvicorn - server will listen IMMEDIATELY")
        logger.info("Heavy initialization will start AFTER server is ready")
        logger.info("=" * 70)

        # Run server (this blocks until shutdown)
        await server.serve()

    except ImportError as e:
        # v197.1: Enhanced error handling with auto-installation attempt
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install fastapi uvicorn llama-cpp-python")
        
        # v197.1: Attempt to identify and install the specific missing package
        missing_pkg = str(e).split("'")[1] if "'" in str(e) else None
        if missing_pkg:
            logger.info(f"[v197.1] Attempting to install missing package: {missing_pkg}")
            try:
                import subprocess
                # Map common module names to pip package names
                pkg_map = {
                    "fastapi": "fastapi",
                    "uvicorn": "uvicorn",
                    "pydantic": "pydantic",
                    "llama_cpp": "llama-cpp-python",
                    "aiohttp": "aiohttp",
                    "httpx": "httpx",
                }
                pip_pkg = pkg_map.get(missing_pkg, missing_pkg)
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--quiet", pip_pkg],
                    capture_output=True,
                    text=True,
                    timeout=180  # llama-cpp-python can take a while to compile
                )
                if result.returncode == 0:
                    logger.info(f"[v197.1] ✅ Installed {pip_pkg}. Please restart the server.")
                else:
                    logger.error(f"[v197.1] ❌ Failed to install {pip_pkg}: {result.stderr[:200]}")
            except Exception as install_err:
                logger.error(f"[v197.1] ❌ Auto-install failed: {install_err}")
        
        logger.error("[v197.1] CRITICAL: Cannot start due to missing dependencies.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


# v93.1: Legacy shutdown function (deprecated - cleanup is now handled inline in main())
async def shutdown(
    manager,
    server,
    heartbeat_task: Optional[asyncio.Task] = None,
    heartbeat_file: Optional[Path] = None,
):
    """
    Graceful shutdown with Trinity coordination and heartbeat cleanup.

    DEPRECATED: v93.1 handles cleanup inline in main() for better control.
    This function is kept for backwards compatibility only.
    """
    logger.info("Shutting down...")

    # v84.0: Cancel heartbeat task first
    if heartbeat_task and not heartbeat_task.done():
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        logger.info("[Heartbeat] ✓ Writer stopped")

    # Cleanup heartbeat file
    if heartbeat_file:
        try:
            if heartbeat_file.exists():
                heartbeat_file.unlink()
                logger.info("[Heartbeat] ✓ File cleaned up")
        except Exception as e:
            logger.debug(f"[Heartbeat] Cleanup error: {e}")

    # v84.0: Shutdown Trinity integration (notify network)
    if _trinity_integration:
        try:
            await _trinity_integration.shutdown()
            logger.info("[Trinity] ✓ Trinity integration shutdown complete")
        except Exception as e:
            logger.warning(f"[Trinity] Shutdown error: {e}")

    # Then stop the model manager
    if manager:
        await manager.stop()
    if server:
        server.should_exit = True
    logger.info("Shutdown complete")


# =============================================================================
# ENTRYPOINT UNIFICATION
# =============================================================================
#
# Root-cause hardening:
# - run_server.py is the authoritative full-featured Prime server surface.
# - jarvis_prime.server must resolve to the same runtime behavior when started
#   as `python -m jarvis_prime.server` (or when `main()` is called directly).
#
# We keep the in-package implementation as a fallback only for environments
# where repository-level run_server.py is unavailable.

_legacy_main = main


@functools.lru_cache(maxsize=1)
def _load_authoritative_main() -> Optional[Callable[[], Awaitable[Any]]]:
    """Load authoritative run_server.main from repository root if available."""
    run_server_path = Path(__file__).resolve().parents[1] / "run_server.py"
    if not run_server_path.exists():
        return None

    try:
        spec = importlib.util.spec_from_file_location(
            "jarvis_prime_authoritative_run_server",
            run_server_path,
        )
        if spec is None or spec.loader is None:
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("jarvis_prime_authoritative_run_server", module)
        spec.loader.exec_module(module)

        candidate = getattr(module, "main", None)
        if asyncio.iscoroutinefunction(candidate):
            return candidate
    except Exception as exc:
        logger.warning(
            "[Entrypoint] Failed to load authoritative run_server.main: %s",
            exc,
        )

    return None


async def main() -> Any:
    """
    Unified Prime entrypoint.

    Delegates to the authoritative full-server implementation in run_server.py
    when present, guaranteeing capability parity across startup paths.
    """
    authoritative_main = _load_authoritative_main()
    if authoritative_main is not None:
        logger.info("[Entrypoint] Delegating to authoritative run_server.main")
        return await authoritative_main()

    allow_legacy = os.getenv(
        "JARVIS_PRIME_ALLOW_LEGACY_SERVER_FALLBACK",
        "false",
    ).strip().lower() in ("1", "true", "yes", "on")
    if allow_legacy:
        logger.warning(
            "[Entrypoint] run_server.py not found; using legacy in-package server implementation "
            "(JARVIS_PRIME_ALLOW_LEGACY_SERVER_FALLBACK=true)",
        )
        return await _legacy_main()

    message = (
        "Authoritative Prime entrypoint unavailable: run_server.py not found. "
        "Refusing to start degraded jarvis_prime.server path. "
        "Set JARVIS_PRIME_ALLOW_LEGACY_SERVER_FALLBACK=true to bypass."
    )
    logger.error("[Entrypoint] %s", message)
    raise RuntimeError(message)


if __name__ == "__main__":
    asyncio.run(main())
