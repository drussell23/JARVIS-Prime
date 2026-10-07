"""Training lease -- J-Prime hands the GPU to a trainer, exclusively, and takes it back.

One RTX 5090 cannot hold the 30B for inference and train its LoRA at the same
time (inference ~21 GiB resident, a GRPO step ~28 GiB). The handoff is a
state machine owned HERE, by the process that owns the inference engines,
so no other actor can reload a model while a trainer holds the card:

    SERVING --acquire--> DRAINING --(in-flight done, engines stopped,
                                     VRAM release VERIFIED)--> RELEASED
    RELEASED --release / expiry--> RESTORING --(preload ok)--> SERVING

* DRAINING refuses new generations (503 + Retry-After) and waits for the
  in-flight ones to finish -- a request is never cut off mid-stream.
* RELEASED is reached only when ``nvidia-smi`` shows the memory the engines
  held is actually free. "We sent terminate" is not evidence; the card is.
* The holder must renew. If it vanishes, the lease EXPIRES -- but restoring
  is still subject to the pool's measured VRAM admission, so an orphaned
  trainer that still holds the card keeps J-Prime from loading on top of it
  (it retries until the card is free). Expiry can never cause an OOM.
* Exactly one lease at a time; the token is required to renew or release.

Never raises out of its public coroutines except ``LeaseConflict`` /
``LeaseTokenMismatch``, which the HTTP layer maps to 409.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import gpu

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


class LeaseState(str, enum.Enum):
    SERVING = "serving"
    DRAINING = "draining"
    RELEASED = "released"
    RESTORING = "restoring"


class LeaseConflict(RuntimeError):
    pass


class LeaseTokenMismatch(RuntimeError):
    pass


@dataclass
class Lease:
    holder: str
    purpose: str
    token: str
    ttl_s: float
    acquired_at: float = field(default_factory=time.time)
    renewed_at: float = field(default_factory=time.time)
    restore_models: List[str] = field(default_factory=list)
    freed_mib: int = 0
    note: str = ""

    @property
    def expires_at(self) -> float:
        return self.renewed_at + self.ttl_s

    def public(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("token")
        d["expires_at"] = self.expires_at
        return d


class TrainingLease:
    """The handoff state machine over an :class:`EnginePool`."""

    def __init__(self, pool: Any, *, gpu_probe: Callable[[], Optional[gpu.GpuMemory]] = gpu.query,
                 clock: Callable[[], float] = time.time) -> None:
        self.pool = pool
        self._gpu = gpu_probe
        self._clock = clock
        self.state = LeaseState.SERVING
        self.lease: Optional[Lease] = None
        # The latest failure of the CURRENT lease cycle, and when. Cleared when
        # a new lease is acquired: an error from an earlier cycle (a restore
        # that failed hours ago and was since recovered) is history, and
        # reporting it beside a live lease would describe the wrong cycle.
        self.last_error = ""
        self.last_error_at: Optional[float] = None
        self._lock = asyncio.Lock()
        self._watch: Optional[asyncio.Task] = None

    # ------------------------------------------------------------- queries
    @property
    def admitting(self) -> bool:
        """May the pool load models / serve generations right now?"""
        return self.state == LeaseState.SERVING

    def retry_after_s(self) -> int:
        if self.lease is None:
            return int(_env_float("JPRIME_LEASE_RETRY_AFTER_S", 30.0))
        return max(1, int(self.lease.expires_at - self._clock()))

    def status(self) -> Dict[str, Any]:
        return {"state": self.state.value, "lease": self.lease.public() if self.lease else None,
                "last_error": self.last_error, "last_error_at": self.last_error_at}

    def _fail(self, message: str) -> None:
        self.last_error, self.last_error_at = message, self._clock()

    # ------------------------------------------------------------- acquire
    async def acquire(self, *, holder: str, purpose: str, ttl_s: float,
                      drain_timeout_s: Optional[float] = None,
                      release_timeout_s: Optional[float] = None) -> Dict[str, Any]:
        """Drain, stop every engine, and VERIFY the card is free. Returns the
        token and what was freed. A failed verification restores serving and
        raises -- the caller must never start a trainer on an unverified card."""
        async with self._lock:
            if self.state != LeaseState.SERVING:
                raise LeaseConflict(f"lease busy: state={self.state.value}")
            self.last_error, self.last_error_at = "", None     # a new cycle
            drain_timeout = drain_timeout_s if drain_timeout_s is not None else _env_float(
                "JPRIME_LEASE_DRAIN_TIMEOUT_S", 600.0)
            release_timeout = release_timeout_s if release_timeout_s is not None else _env_float(
                "JPRIME_LEASE_RELEASE_TIMEOUT_S", 120.0)
            self.lease = Lease(holder=holder, purpose=purpose, token=secrets.token_hex(16),
                               ttl_s=max(30.0, float(ttl_s)), acquired_at=self._clock(),
                               renewed_at=self._clock(),
                               restore_models=[e.spec.name for e in self.pool.resident()])
            self.state = LeaseState.DRAINING
            logger.warning("[Lease] DRAINING for %s (%s); resident=%s", holder, purpose,
                           self.lease.restore_models)
        try:
            await self._drain(drain_timeout)
            held_mib = sum(e.size_vram_bytes for e in self.pool.resident()) // (1024 * 1024)
            before = await asyncio.to_thread(self._gpu)
            for name in list(self.pool.engines):
                await self.pool.unload(name)
            freed = await self._verify_released(before, held_mib, release_timeout)
        except Exception as exc:
            self._fail(f"acquire failed: {exc}")
            logger.error("[Lease] %s -- restoring service", self.last_error)
            await self._restore()
            raise
        async with self._lock:
            self.lease.freed_mib = freed
            self.state = LeaseState.RELEASED
            self._arm_watch()
        mem = await asyncio.to_thread(self._gpu)
        logger.warning("[Lease] RELEASED to %s: freed %d MiB, card free %s MiB", holder, freed,
                       mem.free_mib if mem else "?")
        return {"token": self.lease.token, "freed_mib": freed,
                "gpu_free_mib": mem.free_mib if mem else None, "lease": self.lease.public()}

    async def _drain(self, timeout_s: float) -> None:
        deadline = self._clock() + timeout_s
        while any(e.inflight for e in self.pool.resident()):
            if self._clock() > deadline:
                busy = {e.spec.name: e.inflight for e in self.pool.resident() if e.inflight}
                raise TimeoutError(f"in-flight generations did not finish in {timeout_s:.0f}s: {busy}")
            await asyncio.sleep(0.5)

    async def _verify_released(self, before: Optional[gpu.GpuMemory], held_mib: int,
                               timeout_s: float) -> int:
        """Poll the card until the engines' memory is measurably gone."""
        if before is None:
            # No instrument: say so rather than claim a release we cannot see.
            raise RuntimeError("cannot verify VRAM release: nvidia-smi unavailable")
        if held_mib <= 0:
            return 0
        # Released when used memory has dropped by what the engines held, less
        # a tolerance for allocator slack and other processes' jitter.
        tolerance = int(_env_float("JPRIME_LEASE_RELEASE_TOLERANCE_MIB", 512.0))
        target_used = before.used_mib - held_mib + tolerance
        deadline = self._clock() + timeout_s
        while True:
            now = await asyncio.to_thread(self._gpu)
            if now is not None and now.used_mib <= target_used:
                return max(0, before.used_mib - now.used_mib)
            if self._clock() > deadline:
                raise TimeoutError(f"VRAM not released: used {now.used_mib if now else '?'} MiB, "
                                   f"needed <= {target_used} MiB within {timeout_s:.0f}s")
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------- renew / release
    def _check(self, token: str) -> None:
        if self.lease is None or not secrets.compare_digest(self.lease.token, token or ""):
            raise LeaseTokenMismatch("no such lease")

    async def renew(self, token: str, ttl_s: Optional[float] = None) -> Dict[str, Any]:
        async with self._lock:
            self._check(token)
            if ttl_s:
                self.lease.ttl_s = max(30.0, float(ttl_s))
            self.lease.renewed_at = self._clock()
            return self.lease.public()

    async def release(self, token: str, *, restore_models: Optional[List[str]] = None) -> Dict[str, Any]:
        async with self._lock:
            self._check(token)
            if restore_models is not None:
                self.lease.restore_models = list(restore_models)
        return await self._restore()

    async def _restore(self) -> Dict[str, Any]:
        """Return to SERVING, reloading what was resident (or what the holder named).
        Load failures are reported, not fatal: J-Prime serves on demand regardless."""
        async with self._lock:
            names = list(self.lease.restore_models) if self.lease else []
            self.state = LeaseState.RESTORING
            if self._watch is not None:
                self._watch.cancel()
                self._watch = None
        loaded, failed = [], {}
        for name in names:
            try:
                async with self.pool.acquire(name, keep_alive=-1):
                    pass
                loaded.append(name)
            except Exception as exc:  # noqa: BLE001 -- reported, never fatal
                failed[name] = str(exc)
        async with self._lock:
            self.state = LeaseState.SERVING
            self.lease = None
        if failed:
            self._fail(f"restore: {failed}")
        logger.warning("[Lease] SERVING again; reloaded=%s failed=%s", loaded, failed or "none")
        return {"state": self.state.value, "reloaded": loaded, "failed": failed}

    # ------------------------------------------------------------- expiry
    def _arm_watch(self) -> None:
        if self._watch is None:
            self._watch = asyncio.get_running_loop().create_task(self._watch_expiry())

    async def _watch_expiry(self) -> None:
        """An abandoned lease restores service -- but only through the pool's
        measured VRAM admission, so a trainer that outlived its holder keeps
        the card. Retries until it can load."""
        poll = _env_float("JPRIME_LEASE_EXPIRY_POLL_S", 5.0)
        while True:
            await asyncio.sleep(poll)
            lease = self.lease
            if lease is None or self.state != LeaseState.RELEASED:
                return
            if self._clock() < lease.expires_at:
                continue
            mem = await asyncio.to_thread(self._gpu)
            need = max((self.pool.estimate_mib(s, self.pool._wanted_ctx(s, None))
                        for s in filter(None, (self.pool.store.resolve(n) for n in lease.restore_models))),
                       default=0)
            if mem is not None and mem.free_mib < need + self.pool.config.vram_headroom_mib:
                self._fail(f"lease of {lease.holder} expired but the card is still occupied "
                           f"({mem.free_mib} MiB free, need ~{need}) -- waiting")
                logger.error("[Lease] %s", self.last_error)
                continue
            logger.error("[Lease] lease of %s expired; restoring service", lease.holder)
            self._watch = None
            await self._restore()
            return
