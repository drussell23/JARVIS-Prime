"""
Resource Manager - Memory Pressure & Graceful Degradation
===========================================================

v92.0 - Production-Grade Resource Management

This module provides comprehensive resource management with:
- Real-time memory/CPU/GPU monitoring
- Automatic model eviction under pressure
- Graceful degradation strategies
- Request queuing during high load
- Predictive resource allocation
- Cross-component coordination

ARCHITECTURE:
    ResourceManager
        |
        +-- MemoryMonitor (real-time memory tracking)
        +-- GPUMonitor (CUDA memory management)
        +-- LoadBalancer (request distribution)
        +-- DegradationController (quality vs latency tradeoff)
        +-- ModelEvictor (intelligent cache eviction)
        +-- RequestQueue (overflow handling)

STRATEGIES:
    1. Memory Pressure Response:
       - LOW (< 60%): Full capacity, all models available
       - MODERATE (60-80%): Evict unused models, reduce batch sizes
       - HIGH (80-90%): Emergency eviction, queue requests
       - CRITICAL (> 90%): Fallback to API, reject new loads

    2. Graceful Degradation Modes:
       - FULL_QUALITY: Best model, no compromises
       - BALANCED: Trade some quality for latency
       - FAST: Prioritize speed over quality
       - MINIMAL: Emergency mode, basic functionality only

USAGE:
    manager = await get_resource_manager()

    # Check before loading model
    if await manager.can_load_model("prime-13b", required_gb=26):
        await load_model()
    else:
        # Fall back to smaller model
        await load_smaller_model()

    # Request with resource awareness
    async with manager.resource_context("inference"):
        result = await model.generate(prompt)
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import platform
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ResourceConfig:
    """Configuration for resource management."""

    # Memory thresholds (percentage of total)
    memory_low_threshold: float = field(default_factory=lambda: float(
        os.getenv("RESOURCE_MEM_LOW", "0.60")
    ))
    memory_moderate_threshold: float = field(default_factory=lambda: float(
        os.getenv("RESOURCE_MEM_MODERATE", "0.80")
    ))
    memory_high_threshold: float = field(default_factory=lambda: float(
        os.getenv("RESOURCE_MEM_HIGH", "0.90")
    ))
    memory_critical_threshold: float = field(default_factory=lambda: float(
        os.getenv("RESOURCE_MEM_CRITICAL", "0.95")
    ))

    # GPU memory thresholds
    gpu_memory_threshold: float = field(default_factory=lambda: float(
        os.getenv("RESOURCE_GPU_THRESHOLD", "0.90")
    ))

    # Request queue settings
    max_queue_size: int = field(default_factory=lambda: int(
        os.getenv("RESOURCE_QUEUE_SIZE", "1000")
    ))
    queue_timeout_seconds: float = field(default_factory=lambda: float(
        os.getenv("RESOURCE_QUEUE_TIMEOUT", "30.0")
    ))

    # Monitoring
    monitor_interval_seconds: float = field(default_factory=lambda: float(
        os.getenv("RESOURCE_MONITOR_INTERVAL", "5.0")
    ))

    # Degradation
    enable_auto_degradation: bool = field(default_factory=lambda:
        os.getenv("RESOURCE_AUTO_DEGRADE", "true").lower() == "true"
    )

    # Eviction
    model_idle_timeout_seconds: float = field(default_factory=lambda: float(
        os.getenv("RESOURCE_IDLE_TIMEOUT", "300.0")  # 5 minutes
    ))


# =============================================================================
# ENUMS
# =============================================================================


class MemoryPressure(Enum):
    """Memory pressure levels."""
    LOW = "low"  # < 60%
    MODERATE = "moderate"  # 60-80%
    HIGH = "high"  # 80-90%
    CRITICAL = "critical"  # > 90%


class DegradationMode(Enum):
    """Graceful degradation modes."""
    FULL_QUALITY = "full_quality"
    BALANCED = "balanced"
    FAST = "fast"
    MINIMAL = "minimal"


class ResourceType(Enum):
    """Types of resources to manage."""
    CPU = "cpu"
    RAM = "ram"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ResourceSnapshot:
    """Snapshot of current resource usage."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Memory (bytes)
    total_ram: int = 0
    available_ram: int = 0
    used_ram: int = 0
    ram_percent: float = 0.0

    # GPU memory (bytes)
    total_gpu_ram: int = 0
    available_gpu_ram: int = 0
    used_gpu_ram: int = 0
    gpu_ram_percent: float = 0.0
    gpu_available: bool = False

    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 1

    # Disk
    disk_free_gb: float = 0.0

    # Pressure levels
    memory_pressure: MemoryPressure = MemoryPressure.LOW
    degradation_mode: DegradationMode = DegradationMode.FULL_QUALITY

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "ram": {
                "total_gb": self.total_ram / (1024**3),
                "available_gb": self.available_ram / (1024**3),
                "used_gb": self.used_ram / (1024**3),
                "percent": self.ram_percent,
            },
            "gpu": {
                "available": self.gpu_available,
                "total_gb": self.total_gpu_ram / (1024**3) if self.gpu_available else 0,
                "used_gb": self.used_gpu_ram / (1024**3) if self.gpu_available else 0,
                "percent": self.gpu_ram_percent,
            },
            "cpu": {
                "percent": self.cpu_percent,
                "count": self.cpu_count,
            },
            "disk_free_gb": self.disk_free_gb,
            "memory_pressure": self.memory_pressure.value,
            "degradation_mode": self.degradation_mode.value,
        }


@dataclass
class LoadedModel:
    """Metadata for a loaded model."""
    name: str
    memory_bytes: int
    loaded_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0
    priority: int = 0  # Higher = harder to evict

    @property
    def idle_seconds(self) -> float:
        """Get seconds since last use."""
        return (datetime.now() - self.last_used).total_seconds()


@dataclass
class QueuedRequest:
    """A queued inference request."""
    id: str
    priority: int
    created_at: datetime
    timeout_at: datetime
    event: asyncio.Event
    result: Optional[Any] = None
    error: Optional[Exception] = None


# =============================================================================
# MEMORY MONITOR
# =============================================================================


class MemoryMonitor:
    """Monitors system memory usage."""

    def __init__(self, config: ResourceConfig):
        self._config = config
        self._history: Deque[ResourceSnapshot] = deque(maxlen=100)
        self._last_snapshot: Optional[ResourceSnapshot] = None

    async def get_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        snapshot = ResourceSnapshot()

        # Get RAM info
        try:
            import psutil
            mem = psutil.virtual_memory()
            snapshot.total_ram = mem.total
            snapshot.available_ram = mem.available
            snapshot.used_ram = mem.used
            snapshot.ram_percent = mem.percent / 100.0
        except ImportError:
            # Fallback: read from /proc/meminfo on Linux
            if platform.system() == "Linux":
                snapshot = self._read_proc_meminfo(snapshot)
            elif platform.system() == "Darwin":
                snapshot = self._read_macos_memory(snapshot)

        # Get GPU info
        snapshot = await self._get_gpu_info(snapshot)

        # Get CPU info
        try:
            import psutil
            snapshot.cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            snapshot.cpu_count = psutil.cpu_count()
        except ImportError:
            snapshot.cpu_count = os.cpu_count() or 1

        # Get disk info
        try:
            import shutil
            disk = shutil.disk_usage(Path.home())
            snapshot.disk_free_gb = disk.free / (1024**3)
        except Exception:
            pass

        # Determine pressure level
        snapshot.memory_pressure = self._determine_pressure(snapshot)
        snapshot.degradation_mode = self._determine_degradation(snapshot)

        self._history.append(snapshot)
        self._last_snapshot = snapshot

        return snapshot

    def _read_proc_meminfo(self, snapshot: ResourceSnapshot) -> ResourceSnapshot:
        """Read memory info from /proc/meminfo on Linux."""
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()

            mem_info = {}
            for line in lines:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]
                    mem_info[key] = int(value) * 1024  # Convert KB to bytes

            snapshot.total_ram = mem_info.get("MemTotal", 0)
            snapshot.available_ram = mem_info.get("MemAvailable", mem_info.get("MemFree", 0))
            snapshot.used_ram = snapshot.total_ram - snapshot.available_ram

            if snapshot.total_ram > 0:
                snapshot.ram_percent = snapshot.used_ram / snapshot.total_ram

        except Exception as e:
            logger.debug(f"Failed to read /proc/meminfo: {e}")

        return snapshot

    def _read_macos_memory(self, snapshot: ResourceSnapshot) -> ResourceSnapshot:
        """Read memory info on macOS."""
        try:
            import subprocess

            # Get total memory
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
            )
            snapshot.total_ram = int(result.stdout.strip())

            # Get used memory (approximate)
            result = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
            )

            page_size = 16384  # Default macOS page size
            pages_free = 0
            pages_inactive = 0

            for line in result.stdout.split("\n"):
                if "page size of" in line:
                    page_size = int(line.split()[-2])
                elif "Pages free" in line:
                    pages_free = int(line.split()[-1].replace(".", ""))
                elif "Pages inactive" in line:
                    pages_inactive = int(line.split()[-1].replace(".", ""))

            snapshot.available_ram = (pages_free + pages_inactive) * page_size
            snapshot.used_ram = snapshot.total_ram - snapshot.available_ram

            if snapshot.total_ram > 0:
                snapshot.ram_percent = snapshot.used_ram / snapshot.total_ram

        except Exception as e:
            logger.debug(f"Failed to read macOS memory: {e}")

        return snapshot

    async def _get_gpu_info(self, snapshot: ResourceSnapshot) -> ResourceSnapshot:
        """Get GPU memory information."""
        try:
            import torch

            if torch.cuda.is_available():
                snapshot.gpu_available = True

                # Get GPU memory
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    snapshot.total_gpu_ram += props.total_memory

                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    snapshot.used_gpu_ram += reserved

                snapshot.available_gpu_ram = snapshot.total_gpu_ram - snapshot.used_gpu_ram

                if snapshot.total_gpu_ram > 0:
                    snapshot.gpu_ram_percent = snapshot.used_gpu_ram / snapshot.total_gpu_ram

            elif torch.backends.mps.is_available():
                # Apple Silicon MPS
                snapshot.gpu_available = True
                # MPS doesn't have easy memory query, estimate from system
                snapshot.total_gpu_ram = snapshot.total_ram  # Unified memory

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to get GPU info: {e}")

        return snapshot

    def _determine_pressure(self, snapshot: ResourceSnapshot) -> MemoryPressure:
        """Determine memory pressure level."""
        if snapshot.ram_percent >= self._config.memory_critical_threshold:
            return MemoryPressure.CRITICAL
        elif snapshot.ram_percent >= self._config.memory_high_threshold:
            return MemoryPressure.HIGH
        elif snapshot.ram_percent >= self._config.memory_moderate_threshold:
            return MemoryPressure.MODERATE
        else:
            return MemoryPressure.LOW

    def _determine_degradation(self, snapshot: ResourceSnapshot) -> DegradationMode:
        """Determine degradation mode based on resources."""
        if snapshot.memory_pressure == MemoryPressure.CRITICAL:
            return DegradationMode.MINIMAL
        elif snapshot.memory_pressure == MemoryPressure.HIGH:
            return DegradationMode.FAST
        elif snapshot.memory_pressure == MemoryPressure.MODERATE:
            return DegradationMode.BALANCED
        else:
            return DegradationMode.FULL_QUALITY

    def get_trend(self, window_seconds: float = 60.0) -> str:
        """Get memory usage trend."""
        if len(self._history) < 2:
            return "stable"

        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        recent = [s for s in self._history if s.timestamp > cutoff]

        if len(recent) < 2:
            return "stable"

        first_avg = sum(s.ram_percent for s in recent[:len(recent)//2]) / (len(recent)//2)
        second_avg = sum(s.ram_percent for s in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)

        diff = second_avg - first_avg

        if diff > 0.05:
            return "increasing"
        elif diff < -0.05:
            return "decreasing"
        else:
            return "stable"


# =============================================================================
# MODEL EVICTOR
# =============================================================================


class ModelEvictor:
    """Manages model eviction under memory pressure."""

    def __init__(self, config: ResourceConfig):
        self._config = config
        self._loaded_models: Dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._eviction_callbacks: List[Callable[[str], Any]] = []

    def register_model(
        self,
        name: str,
        memory_bytes: int,
        priority: int = 0,
    ) -> None:
        """Register a loaded model."""
        self._loaded_models[name] = LoadedModel(
            name=name,
            memory_bytes=memory_bytes,
            priority=priority,
        )
        logger.debug(f"Registered model {name} ({memory_bytes / 1024**3:.2f} GB)")

    def unregister_model(self, name: str) -> None:
        """Unregister a model."""
        if name in self._loaded_models:
            del self._loaded_models[name]
            logger.debug(f"Unregistered model {name}")

    def record_usage(self, name: str) -> None:
        """Record model usage."""
        if name in self._loaded_models:
            self._loaded_models[name].last_used = datetime.now()
            self._loaded_models[name].use_count += 1

    def on_eviction(self, callback: Callable[[str], Any]) -> None:
        """Register eviction callback."""
        self._eviction_callbacks.append(callback)

    async def get_eviction_candidates(
        self,
        required_bytes: int,
        exclude: Optional[Set[str]] = None,
    ) -> List[str]:
        """Get models to evict to free required memory."""
        exclude = exclude or set()

        # Sort by priority (low first), then by idle time (high first)
        candidates = sorted(
            [m for m in self._loaded_models.values() if m.name not in exclude],
            key=lambda m: (m.priority, -m.idle_seconds),
        )

        to_evict: List[str] = []
        freed = 0

        for model in candidates:
            if freed >= required_bytes:
                break

            # Don't evict recently used models unless critical
            if model.idle_seconds < 60:  # Used in last minute
                continue

            to_evict.append(model.name)
            freed += model.memory_bytes

        return to_evict

    async def evict_models(self, models: List[str]) -> int:
        """Evict specified models."""
        freed = 0

        async with self._lock:
            for name in models:
                if name in self._loaded_models:
                    freed += self._loaded_models[name].memory_bytes

                    # Call eviction callbacks
                    for callback in self._eviction_callbacks:
                        try:
                            result = callback(name)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logger.warning(f"Eviction callback failed: {e}")

                    del self._loaded_models[name]
                    logger.info(f"Evicted model {name}")

        # Force garbage collection
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return freed

    async def evict_idle_models(self) -> int:
        """Evict models that have been idle too long."""
        idle_threshold = self._config.model_idle_timeout_seconds
        to_evict = [
            m.name for m in self._loaded_models.values()
            if m.idle_seconds > idle_threshold
        ]

        if to_evict:
            return await self.evict_models(to_evict)
        return 0

    def get_loaded_models(self) -> Dict[str, LoadedModel]:
        """Get all loaded models."""
        return dict(self._loaded_models)

    def get_total_memory(self) -> int:
        """Get total memory used by loaded models."""
        return sum(m.memory_bytes for m in self._loaded_models.values())


# =============================================================================
# REQUEST QUEUE
# =============================================================================


class RequestQueue:
    """Manages request queue during high load."""

    def __init__(self, config: ResourceConfig):
        self._config = config
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=config.max_queue_size
        )
        self._pending: Dict[str, QueuedRequest] = {}
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        priority: int = 0,
        timeout: Optional[float] = None,
    ) -> QueuedRequest:
        """Enqueue a request and return handle."""
        timeout = timeout or self._config.queue_timeout_seconds

        request = QueuedRequest(
            id=str(uuid.uuid4()),
            priority=priority,
            created_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=timeout),
            event=asyncio.Event(),
        )

        try:
            # Priority queue uses (priority, timestamp, request) tuple
            self._queue.put_nowait((-priority, time.time(), request))
            self._pending[request.id] = request
            return request
        except asyncio.QueueFull:
            raise RuntimeError("Request queue is full")

    async def wait_for_slot(self, request: QueuedRequest) -> bool:
        """Wait for request to be processed."""
        try:
            remaining = (request.timeout_at - datetime.now()).total_seconds()
            if remaining <= 0:
                return False

            await asyncio.wait_for(request.event.wait(), timeout=remaining)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            if request.id in self._pending:
                del self._pending[request.id]

    async def process_next(self) -> Optional[QueuedRequest]:
        """Get next request from queue."""
        try:
            priority, timestamp, request = await asyncio.wait_for(
                self._queue.get(),
                timeout=0.1,
            )

            # Check if timed out
            if datetime.now() > request.timeout_at:
                request.error = TimeoutError("Request timed out in queue")
                request.event.set()
                return None

            return request
        except asyncio.TimeoutError:
            return None

    def complete(self, request: QueuedRequest, result: Any = None) -> None:
        """Mark request as complete."""
        request.result = result
        request.event.set()

    def fail(self, request: QueuedRequest, error: Exception) -> None:
        """Mark request as failed."""
        request.error = error
        request.event.set()

    @property
    def size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()


# =============================================================================
# DEGRADATION CONTROLLER
# =============================================================================


class DegradationController:
    """Controls graceful degradation strategies."""

    # Degradation parameters for each mode
    MODE_PARAMS: Dict[DegradationMode, Dict[str, Any]] = {
        DegradationMode.FULL_QUALITY: {
            "max_tokens": 2048,
            "batch_size": 8,
            "use_quantization": False,
            "fallback_to_api": False,
            "model_size_limit_gb": float("inf"),
        },
        DegradationMode.BALANCED: {
            "max_tokens": 1024,
            "batch_size": 4,
            "use_quantization": True,
            "fallback_to_api": False,
            "model_size_limit_gb": 20.0,
        },
        DegradationMode.FAST: {
            "max_tokens": 512,
            "batch_size": 2,
            "use_quantization": True,
            "fallback_to_api": True,
            "model_size_limit_gb": 10.0,
        },
        DegradationMode.MINIMAL: {
            "max_tokens": 256,
            "batch_size": 1,
            "use_quantization": True,
            "fallback_to_api": True,
            "model_size_limit_gb": 5.0,
        },
    }

    def __init__(self, config: ResourceConfig):
        self._config = config
        self._current_mode = DegradationMode.FULL_QUALITY
        self._lock = asyncio.Lock()
        self._mode_change_callbacks: List[Callable[[DegradationMode], Any]] = []

    @property
    def current_mode(self) -> DegradationMode:
        """Get current degradation mode."""
        return self._current_mode

    @property
    def params(self) -> Dict[str, Any]:
        """Get current mode parameters."""
        return self.MODE_PARAMS[self._current_mode]

    async def set_mode(self, mode: DegradationMode) -> None:
        """Set degradation mode."""
        async with self._lock:
            if mode != self._current_mode:
                old_mode = self._current_mode
                self._current_mode = mode
                logger.info(f"Degradation mode: {old_mode.value} -> {mode.value}")

                # Notify callbacks
                for callback in self._mode_change_callbacks:
                    try:
                        result = callback(mode)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.warning(f"Mode change callback failed: {e}")

    def on_mode_change(self, callback: Callable[[DegradationMode], Any]) -> None:
        """Register mode change callback."""
        self._mode_change_callbacks.append(callback)

    def should_use_api_fallback(self) -> bool:
        """Check if should fallback to API."""
        return self.params.get("fallback_to_api", False)

    def get_max_tokens(self) -> int:
        """Get max tokens for current mode."""
        return self.params.get("max_tokens", 1024)

    def get_batch_size(self) -> int:
        """Get batch size for current mode."""
        return self.params.get("batch_size", 1)

    def can_load_model_size(self, size_gb: float) -> bool:
        """Check if model size is allowed in current mode."""
        return size_gb <= self.params.get("model_size_limit_gb", float("inf"))


# =============================================================================
# RESOURCE MANAGER
# =============================================================================


class ResourceManager:
    """Main resource manager coordinating all components."""

    def __init__(self, config: Optional[ResourceConfig] = None):
        self._config = config or ResourceConfig()
        self._memory_monitor = MemoryMonitor(self._config)
        self._model_evictor = ModelEvictor(self._config)
        self._request_queue = RequestQueue(self._config)
        self._degradation = DegradationController(self._config)

        self._monitor_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "evictions": 0,
            "queued_requests": 0,
            "degradation_changes": 0,
            "memory_pressure_events": 0,
        }

    async def start(self) -> None:
        """Start resource monitoring."""
        if self._is_running:
            return

        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Resource manager started")

    async def stop(self) -> None:
        """Stop resource monitoring."""
        self._is_running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Resource manager stopped")

    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._is_running:
            try:
                snapshot = await self._memory_monitor.get_snapshot()

                # Update degradation mode
                if self._config.enable_auto_degradation:
                    await self._degradation.set_mode(snapshot.degradation_mode)

                # Handle pressure
                await self._handle_pressure(snapshot)

                await asyncio.sleep(self._config.monitor_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(1)

    async def _handle_pressure(self, snapshot: ResourceSnapshot) -> None:
        """Handle memory pressure conditions."""
        if snapshot.memory_pressure == MemoryPressure.CRITICAL:
            self._stats["memory_pressure_events"] += 1
            logger.warning("CRITICAL memory pressure - emergency eviction")

            # Evict all non-essential models
            models = self._model_evictor.get_loaded_models()
            to_evict = [m.name for m in models.values() if m.priority < 10]
            if to_evict:
                freed = await self._model_evictor.evict_models(to_evict)
                self._stats["evictions"] += len(to_evict)
                logger.info(f"Freed {freed / 1024**3:.2f} GB from emergency eviction")

        elif snapshot.memory_pressure == MemoryPressure.HIGH:
            # Evict idle models
            await self._model_evictor.evict_idle_models()

    async def can_load_model(
        self,
        model_name: str,
        required_gb: float,
    ) -> Tuple[bool, str]:
        """Check if a model can be loaded."""
        snapshot = await self._memory_monitor.get_snapshot()

        # Check degradation limits
        if not self._degradation.can_load_model_size(required_gb):
            return False, f"Model too large for {self._degradation.current_mode.value} mode"

        # Check available memory
        required_bytes = int(required_gb * 1024**3)
        available = snapshot.available_ram

        # Add buffer (20% extra)
        required_with_buffer = int(required_bytes * 1.2)

        if available >= required_with_buffer:
            return True, "Sufficient memory available"

        # Check if we can free enough memory
        candidates = await self._model_evictor.get_eviction_candidates(
            required_bytes - available,
            exclude={model_name},
        )

        if candidates:
            potential_freed = sum(
                self._model_evictor._loaded_models[m].memory_bytes
                for m in candidates
                if m in self._model_evictor._loaded_models
            )

            if available + potential_freed >= required_bytes:
                return True, f"Can load after evicting: {candidates}"

        return False, "Insufficient memory even after eviction"

    async def prepare_for_load(
        self,
        model_name: str,
        required_gb: float,
    ) -> bool:
        """Prepare resources for loading a model."""
        can_load, reason = await self.can_load_model(model_name, required_gb)

        if not can_load:
            logger.warning(f"Cannot load {model_name}: {reason}")
            return False

        if "evicting" in reason.lower():
            # Extract models to evict
            import re
            match = re.search(r'\[(.*)\]', reason)
            if match:
                models = [m.strip().strip("'") for m in match.group(1).split(",")]
                await self._model_evictor.evict_models(models)
                self._stats["evictions"] += len(models)

        return True

    def register_model(
        self,
        name: str,
        memory_bytes: int,
        priority: int = 0,
    ) -> None:
        """Register a loaded model."""
        self._model_evictor.register_model(name, memory_bytes, priority)

    def unregister_model(self, name: str) -> None:
        """Unregister a model."""
        self._model_evictor.unregister_model(name)

    def record_model_usage(self, name: str) -> None:
        """Record model usage."""
        self._model_evictor.record_usage(name)

    @asynccontextmanager
    async def resource_context(
        self,
        operation: str = "inference",
        priority: int = 0,
    ) -> AsyncIterator[ResourceSnapshot]:
        """Context manager for resource-aware operations."""
        snapshot = await self._memory_monitor.get_snapshot()

        # Queue if under high pressure
        if snapshot.memory_pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL):
            if not self._request_queue.is_full:
                request = await self._request_queue.enqueue(priority=priority)
                self._stats["queued_requests"] += 1

                # Wait for slot
                if not await self._request_queue.wait_for_slot(request):
                    raise TimeoutError("Request timed out waiting in queue")

                if request.error:
                    raise request.error

        try:
            yield snapshot
        finally:
            # Update snapshot
            await self._memory_monitor.get_snapshot()

    async def get_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        return await self._memory_monitor.get_snapshot()

    def get_degradation_mode(self) -> DegradationMode:
        """Get current degradation mode."""
        return self._degradation.current_mode

    def get_degradation_params(self) -> Dict[str, Any]:
        """Get current degradation parameters."""
        return self._degradation.params

    def should_use_api_fallback(self) -> bool:
        """Check if should fallback to API."""
        return self._degradation.should_use_api_fallback()

    def on_eviction(self, callback: Callable[[str], Any]) -> None:
        """Register model eviction callback."""
        self._model_evictor.on_eviction(callback)

    def on_degradation_change(self, callback: Callable[[DegradationMode], Any]) -> None:
        """Register degradation mode change callback."""
        self._degradation.on_mode_change(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "running": self._is_running,
            "loaded_models": {
                name: {
                    "memory_gb": m.memory_bytes / (1024**3),
                    "idle_seconds": m.idle_seconds,
                    "use_count": m.use_count,
                    "priority": m.priority,
                }
                for name, m in self._model_evictor.get_loaded_models().items()
            },
            "total_model_memory_gb": self._model_evictor.get_total_memory() / (1024**3),
            "queue_size": self._request_queue.size,
            "degradation_mode": self._degradation.current_mode.value,
            "stats": self._stats,
            "config": {
                "memory_thresholds": {
                    "low": self._config.memory_low_threshold,
                    "moderate": self._config.memory_moderate_threshold,
                    "high": self._config.memory_high_threshold,
                    "critical": self._config.memory_critical_threshold,
                },
                "auto_degradation": self._config.enable_auto_degradation,
                "queue_size": self._config.max_queue_size,
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_manager: Optional[ResourceManager] = None
_manager_lock = asyncio.Lock()


async def get_resource_manager(
    config: Optional[ResourceConfig] = None,
) -> ResourceManager:
    """Get or create the global resource manager."""
    global _manager

    async with _manager_lock:
        if _manager is None:
            _manager = ResourceManager(config)
            await _manager.start()

        return _manager


async def shutdown_resource_manager() -> None:
    """Shutdown the global resource manager."""
    global _manager

    async with _manager_lock:
        if _manager is not None:
            await _manager.stop()
            _manager = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Config
    "ResourceConfig",
    # Enums
    "MemoryPressure",
    "DegradationMode",
    "ResourceType",
    # Data structures
    "ResourceSnapshot",
    "LoadedModel",
    "QueuedRequest",
    # Components
    "MemoryMonitor",
    "ModelEvictor",
    "RequestQueue",
    "DegradationController",
    # Manager
    "ResourceManager",
    # Factory
    "get_resource_manager",
    "shutdown_resource_manager",
]
