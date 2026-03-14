"""
VRAM Pressure Monitor — GPU Memory Watchdog
============================================

Read-only module that monitors GPU VRAM usage and emits pressure events.
Never triggers swaps directly — events are consumed by ModelTransitionManager.

Backends: pynvml (preferred), nvidia-smi (fallback), mock (testing).
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


# =============================================================================
# DATA MODELS
# =============================================================================

class VRAMPressureZone(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    CRITICAL = "critical"


@dataclass(frozen=True)
class VRAMPressureEvent:
    """Emitted when pressure zone changes. Advisory only."""
    zone: VRAMPressureZone
    previous_zone: VRAMPressureZone
    total_bytes: int
    used_bytes: int
    free_bytes: int
    fragmentation_estimate: float
    model_resident_bytes: int
    kv_cache_bytes: int
    timestamp: float
    sustained_seconds: float
    node_id: str


@dataclass
class VRAMMonitorConfig:
    poll_interval_s: float = field(default_factory=lambda: _env_float("JARVIS_VRAM_POLL_INTERVAL_S", 5.0))
    critical_poll_interval_s: float = 1.0
    zone_thresholds: dict = field(default_factory=lambda: {
        "yellow": _env_float("JARVIS_VRAM_YELLOW_THRESHOLD", 0.70),
        "red": _env_float("JARVIS_VRAM_RED_THRESHOLD", 0.85),
        "critical": _env_float("JARVIS_VRAM_CRITICAL_THRESHOLD", 0.92),
    })
    sustained_threshold_s: float = 10.0
    backend: str = "pynvml"


@dataclass(frozen=True)
class VRAMSnapshot:
    """Point-in-time VRAM reading."""
    zone: VRAMPressureZone
    total_bytes: int
    used_bytes: int
    free_bytes: int
    utilization: float
    timestamp: float


# =============================================================================
# HELPERS
# =============================================================================

def estimate_effective_free(free_bytes: int, model_loaded: bool) -> int:
    """Conservative estimate of allocatable VRAM."""
    if not model_loaded:
        return int(free_bytes * 0.95)
    return int(free_bytes * 0.80)


# =============================================================================
# MONITOR
# =============================================================================

class VRAMPressureMonitor:
    """GPU memory watchdog. Emits VRAMPressureEvents on zone transitions."""

    def __init__(
        self,
        config: Optional[VRAMMonitorConfig] = None,
        node_id: str = "gcp-jarvis-prime-stable",
    ):
        self._config = config or VRAMMonitorConfig()
        self._node_id = node_id
        self._current_zone = VRAMPressureZone.GREEN
        self._zone_entered_at: float = time.monotonic()
        self._callbacks: List[Callable[[VRAMPressureEvent], None]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Mock backend state
        self._mock_total: int = 0
        self._mock_used: int = 0
        self._initialized = False

    def on_pressure_change(self, callback: Callable[[VRAMPressureEvent], None]) -> None:
        self._callbacks.append(callback)

    def set_mock_vram(self, total: int, used: int) -> None:
        self._mock_total = total
        self._mock_used = used

    @staticmethod
    def _zone_for_utilization(util: float, config: VRAMMonitorConfig) -> VRAMPressureZone:
        if util >= config.zone_thresholds["critical"]:
            return VRAMPressureZone.CRITICAL
        if util >= config.zone_thresholds["red"]:
            return VRAMPressureZone.RED
        if util >= config.zone_thresholds["yellow"]:
            return VRAMPressureZone.YELLOW
        return VRAMPressureZone.GREEN

    async def _read_vram(self) -> tuple:
        """Read (total, used) VRAM bytes from configured backend."""
        backend = self._config.backend

        if backend == "mock":
            return self._mock_total, self._mock_used

        if backend == "pynvml":
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return info.total, info.used
            except Exception:
                logger.warning("[VRAMMonitor] pynvml failed, falling back to nvidia-smi")
                backend = "nvidia_smi"

        if backend == "nvidia_smi":
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    total_mib = float(parts[0].strip())
                    used_mib = float(parts[1].strip())
                    return int(total_mib * 1024 * 1024), int(used_mib * 1024 * 1024)
            except Exception:
                pass

        # Safe default
        logger.warning("[VRAMMonitor] All backends failed, assuming YELLOW")
        return 23_034 * 1024 * 1024, int(23_034 * 1024 * 1024 * 0.75)

    async def sample(self) -> VRAMSnapshot:
        """Take a single VRAM sample and emit events if zone changed."""
        total, used = await self._read_vram()
        free = total - used
        util = used / total if total > 0 else 0.0
        now = time.monotonic()

        new_zone = self._zone_for_utilization(util, self._config)
        snapshot = VRAMSnapshot(
            zone=new_zone, total_bytes=total, used_bytes=used,
            free_bytes=free, utilization=util, timestamp=time.time(),
        )

        if new_zone != self._current_zone:
            sustained = now - self._zone_entered_at
            if not self._initialized or sustained >= self._config.sustained_threshold_s:
                event = VRAMPressureEvent(
                    zone=new_zone,
                    previous_zone=self._current_zone,
                    total_bytes=total,
                    used_bytes=used,
                    free_bytes=free,
                    fragmentation_estimate=0.05 if used > 0 else 0.0,
                    model_resident_bytes=0,
                    kv_cache_bytes=0,
                    timestamp=time.time(),
                    sustained_seconds=sustained,
                    node_id=self._node_id,
                )
                for cb in self._callbacks:
                    try:
                        cb(event)
                    except Exception:
                        logger.exception("[VRAMMonitor] Callback error")
                self._current_zone = new_zone
                self._zone_entered_at = now
                self._initialized = True

        return snapshot

    async def start(self) -> None:
        """Start background monitoring loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"[VRAMMonitor] Started on {self._node_id}")

    async def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self.sample()
            except Exception:
                logger.exception("[VRAMMonitor] Poll error")
            interval = (
                self._config.critical_poll_interval_s
                if self._current_zone in (VRAMPressureZone.RED, VRAMPressureZone.CRITICAL)
                else self._config.poll_interval_s
            )
            await asyncio.sleep(interval)

    @property
    def current_zone(self) -> VRAMPressureZone:
        return self._current_zone
