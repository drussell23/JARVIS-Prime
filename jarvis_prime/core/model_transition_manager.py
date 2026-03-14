"""
Model Transition Manager — The Single Executor
================================================

THE executor for all model changes. Serialized FSM with epoch-based
consistency, drain protocol, and VRAMBudgetAuthority integration.

States: IDLE → PREPARE → DRAIN → CUTOVER → VERIFY → COMMIT/ROLLBACK
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


# =============================================================================
# VRAM BUDGET AUTHORITY
# =============================================================================

class LeaseState(Enum):
    GRANTED = "granted"
    ACTIVE = "active"
    RELEASED = "released"
    ROLLED_BACK = "rolled_back"


class VRAMPriority(Enum):
    CRITICAL = 0
    NORMAL = 1
    BACKGROUND = 2


class VRAMGrant:
    """A VRAM lease with lifecycle management."""

    def __init__(
        self,
        grant_id: str,
        component: str,
        granted_bytes: int,
        ttl_seconds: float,
        authority: VRAMBudgetAuthority,
    ):
        self.grant_id = grant_id
        self.component = component
        self.granted_bytes = granted_bytes
        self.actual_bytes: int = 0
        self.state = LeaseState.GRANTED
        self.ttl_seconds = ttl_seconds
        self.created_at = time.monotonic()
        self._authority = authority

    async def commit(self, actual_bytes: int) -> None:
        if self.state != LeaseState.GRANTED:
            raise RuntimeError(f"Cannot commit grant in state {self.state}")
        self.actual_bytes = actual_bytes
        self.state = LeaseState.ACTIVE
        self._authority._update_grant(self)

    async def rollback(self, reason: str = "") -> None:
        if self.state in (LeaseState.RELEASED, LeaseState.ROLLED_BACK):
            return
        prev = self.state
        self.state = LeaseState.ROLLED_BACK
        self._authority._release_grant(self)
        logger.info(f"[VRAMAuthority] Grant {self.grant_id} rolled back from {prev.value}: {reason}")

    async def release(self) -> None:
        if self.state in (LeaseState.RELEASED, LeaseState.ROLLED_BACK):
            return
        self.state = LeaseState.RELEASED
        self._authority._release_grant(self)

    async def heartbeat(self) -> None:
        self.created_at = time.monotonic()


class VRAMBudgetAuthority:
    """Lightweight VRAM admission controller for GCP VM."""

    def __init__(self, total_vram_bytes: int):
        self._total = total_vram_bytes
        self._grants: Dict[str, VRAMGrant] = {}
        self._lock = asyncio.Lock()

    @property
    def total_vram_bytes(self) -> int:
        return self._total

    @property
    def allocated_bytes(self) -> int:
        return sum(
            g.granted_bytes for g in self._grants.values()
            if g.state in (LeaseState.GRANTED, LeaseState.ACTIVE)
        )

    @property
    def available_bytes(self) -> int:
        return self._total - self.allocated_bytes

    async def request(
        self,
        component: str,
        bytes_requested: int,
        priority: VRAMPriority,
        *,
        ttl_seconds: float = 300.0,
        releasing_grant_id: Optional[str] = None,
    ) -> Optional[VRAMGrant]:
        """Issue or deny a VRAM grant.

        Args:
            releasing_grant_id: If set, the budget calculation assumes this
                grant will be released during CUTOVER (two-grant swap
                reservation). Without this, same-size model swaps would
                always be denied because both grants count against budget.
        """
        async with self._lock:
            available = self.available_bytes
            # Account for grant being released during swap
            if releasing_grant_id and releasing_grant_id in self._grants:
                releasing = self._grants[releasing_grant_id]
                if releasing.state in (LeaseState.GRANTED, LeaseState.ACTIVE):
                    available += releasing.granted_bytes
            if bytes_requested > available:
                logger.warning(
                    f"[VRAMAuthority] Denied {component}: "
                    f"requested {bytes_requested:,} > available {available:,}"
                )
                return None
            grant = VRAMGrant(
                grant_id=f"vram-{uuid.uuid4().hex[:8]}",
                component=component,
                granted_bytes=bytes_requested,
                ttl_seconds=ttl_seconds,
                authority=self,
            )
            self._grants[grant.grant_id] = grant
            logger.info(
                f"[VRAMAuthority] Granted {grant.grant_id} to {component}: "
                f"{bytes_requested:,} bytes"
            )
            return grant

    def _update_grant(self, grant: VRAMGrant) -> None:
        pass  # Grant already tracked

    def _release_grant(self, grant: VRAMGrant) -> None:
        self._grants.pop(grant.grant_id, None)


# =============================================================================
# TRANSITION STATE MACHINE
# =============================================================================

class TransitionState(Enum):
    IDLE = "idle"
    PREPARE = "prepare"
    DRAIN = "drain"
    CUTOVER = "cutover"
    VERIFY = "verify"
    COMMIT = "commit"
    ROLLBACK = "rollback"


@dataclass
class TransitionEpoch:
    model_epoch: int = 0
    cache_epoch: int = 0
    inventory_epoch: int = 0

    def advance_model(self) -> int:
        self.model_epoch += 1
        return self.model_epoch

    def advance_cache(self) -> int:
        self.cache_epoch += 1
        return self.cache_epoch


@dataclass
class TransitionPolicy:
    min_cooldown_s: float = field(default_factory=lambda: _env_float("JARVIS_MODEL_SWAP_COOLDOWN_S", 90.0))
    max_swaps_per_hour: int = field(default_factory=lambda: _env_int("JARVIS_MODEL_MAX_SWAPS_PER_HOUR", 4))
    quality_dead_zone: float = field(default_factory=lambda: _env_float("JARVIS_MODEL_QUALITY_DEAD_ZONE", 0.05))
    upgrade_sustained_s: float = 30.0
    downgrade_sustained_s: float = 10.0
    cold_start_lockout_s: float = 120.0
    backoff_base_s: float = 90.0
    backoff_multiplier: float = 2.0
    backoff_max_s: float = 600.0


@dataclass(frozen=True)
class ModelTransitionEvent:
    event_type: str
    transition_id: str
    trigger: str
    from_model: Optional[str]
    to_model: str
    from_quant: Optional[str]
    to_quant: str
    model_epoch: int
    duration_ms: Optional[float]
    outcome: str
    reason: str
    timestamp: float


class ModelTransitionManager:
    """THE single executor for all model transitions."""

    def __init__(
        self,
        executor: Any,  # LlamaCppExecutor (not typed to avoid circular import)
        vram_authority: VRAMBudgetAuthority,
        model_dir: Path,
        policy: Optional[TransitionPolicy] = None,
    ):
        self._executor = executor
        self._vram_authority = vram_authority
        self._model_dir = model_dir
        self._policy = policy or TransitionPolicy()
        self._state = TransitionState.IDLE
        self._epoch = TransitionEpoch()
        self._lock = asyncio.Lock()
        self._started_at = time.monotonic()
        self._last_swap_time: float = 0.0
        self._swap_times: List[float] = []
        self._current_model_path: Optional[Path] = None
        self._current_grant: Optional[VRAMGrant] = None
        self._active_requests: int = 0
        self._drain_event: Optional[asyncio.Event] = None
        self._event_callbacks: List[Callable[[ModelTransitionEvent], None]] = []
        self._current_fitness: Optional[float] = None
        self._draining: bool = False
        self._degraded: bool = False

    @property
    def state(self) -> TransitionState:
        return self._state

    @property
    def epoch(self) -> TransitionEpoch:
        return self._epoch

    @property
    def current_model_path(self) -> Optional[Path]:
        return self._current_model_path

    def on_transition_event(self, cb: Callable[[ModelTransitionEvent], None]) -> None:
        self._event_callbacks.append(cb)

    def _emit_event(self, event: ModelTransitionEvent) -> None:
        for cb in self._event_callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception("[TransitionManager] Event callback error")

    def _check_cooldown(self) -> Optional[str]:
        """Check if swap is allowed by policy. Returns rejection reason or None."""
        now = time.monotonic()

        # Cold start lockout
        if now - self._started_at < self._policy.cold_start_lockout_s:
            remaining = self._policy.cold_start_lockout_s - (now - self._started_at)
            return f"Cold start lockout: {remaining:.0f}s remaining"

        # Cooldown
        if self._last_swap_time > 0:
            elapsed = now - self._last_swap_time
            cooldown = min(
                self._policy.backoff_base_s * (self._policy.backoff_multiplier ** max(0, len(self._swap_times) - 1)),
                self._policy.backoff_max_s,
            )
            if elapsed < cooldown:
                return f"Cooldown: {cooldown - elapsed:.0f}s remaining"

        # Hourly cap
        hour_ago = now - 3600
        recent = [t for t in self._swap_times if t > hour_ago]
        if len(recent) >= self._policy.max_swaps_per_hour:
            return f"Hourly cap reached: {len(recent)}/{self._policy.max_swaps_per_hour}"

        return None

    def admit_request(self) -> bool:
        """Check if new requests should be admitted. Returns False during DRAIN."""
        if self._draining:
            return False
        self.request_started()
        return True

    async def accept(self, proposal: Any) -> bool:
        """Accept a ModelSelectionProposal and execute the transition."""
        async with self._lock:
            if self._state != TransitionState.IDLE:
                logger.warning(f"[TransitionManager] Rejected: state is {self._state.value}")
                return False

            # Check cooldown (skip for startup trigger)
            if proposal.trigger != "startup":
                rejection = self._check_cooldown()
                if rejection:
                    logger.info(f"[TransitionManager] Rejected: {rejection}")
                    return False

                # Quality dead zone: don't swap if <5% fitness improvement
                if (self._current_fitness is not None and
                    hasattr(proposal, 'quality_score') and
                    abs(proposal.quality_score.fitness_score - self._current_fitness) < self._policy.quality_dead_zone):
                    logger.info("[TransitionManager] Rejected: within quality dead zone")
                    return False

            # Revalidate inventory digest -- reject stale proposals
            try:
                from jarvis_prime.core.adaptive_model_selector import _compute_inventory_digest
                current_digest = _compute_inventory_digest(self._model_dir)
                if (hasattr(proposal, 'inventory_digest') and
                    proposal.inventory_digest != "unknown" and
                    proposal.inventory_digest != current_digest):
                    logger.info(
                        f"[TransitionManager] Rejected: inventory changed "
                        f"(proposal={proposal.inventory_digest}, current={current_digest})"
                    )
                    return False
            except ImportError:
                pass

            transition_id = f"trans-{uuid.uuid4().hex[:8]}"
            start_time = time.monotonic()
            target_path = proposal.selected_variant.path
            target_size = proposal.selected_variant.size_bytes
            old_path = self._current_model_path
            old_grant = self._current_grant

            try:
                # PREPARE — use releasing_grant_id for two-grant swap reservation
                self._state = TransitionState.PREPARE
                releasing_id = self._current_grant.grant_id if self._current_grant else None
                new_grant = await self._vram_authority.request(
                    f"model-{proposal.selected_variant.quant_name}",
                    target_size,
                    VRAMPriority.NORMAL,
                    releasing_grant_id=releasing_id,
                )
                if new_grant is None:
                    self._state = TransitionState.IDLE
                    self._emit_event(ModelTransitionEvent(
                        event_type="transition_failed", transition_id=transition_id,
                        trigger=proposal.trigger, from_model=str(old_path),
                        to_model=str(target_path), from_quant=None,
                        to_quant=proposal.selected_variant.quant_name,
                        model_epoch=self._epoch.model_epoch,
                        duration_ms=(time.monotonic() - start_time) * 1000,
                        outcome="denied", reason="VRAM budget denied",
                        timestamp=time.time(),
                    ))
                    return False

                # DRAIN
                self._state = TransitionState.DRAIN
                self._draining = True
                if self._active_requests > 0:
                    self._drain_event = asyncio.Event()
                    try:
                        await asyncio.wait_for(self._drain_event.wait(), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning("[TransitionManager] Drain timeout, proceeding with rollback")
                        await new_grant.rollback("drain timeout")
                        self._state = TransitionState.IDLE
                        self._draining = False
                        return False
                self._draining = False

                # CUTOVER
                self._state = TransitionState.CUTOVER
                if self._executor.is_loaded():
                    await self._executor.unload()
                if old_grant:
                    await old_grant.release()

                await self._executor.load(target_path)
                await new_grant.commit(target_size)

                # VERIFY
                self._state = TransitionState.VERIFY
                valid = await self._executor.validate()
                if not valid:
                    raise RuntimeError("Post-swap validation failed")

                # COMMIT
                self._state = TransitionState.COMMIT
                new_epoch = self._epoch.advance_model()
                self._current_model_path = target_path
                self._current_grant = new_grant
                self._current_fitness = proposal.quality_score.fitness_score
                self._last_swap_time = time.monotonic()
                self._swap_times.append(time.monotonic())

                duration_ms = (time.monotonic() - start_time) * 1000
                self._emit_event(ModelTransitionEvent(
                    event_type="transition_completed", transition_id=transition_id,
                    trigger=proposal.trigger, from_model=str(old_path),
                    to_model=str(target_path), from_quant=None,
                    to_quant=proposal.selected_variant.quant_name,
                    model_epoch=new_epoch, duration_ms=duration_ms,
                    outcome="commit",
                    reason=proposal.reason, timestamp=time.time(),
                ))
                logger.info(
                    f"[TransitionManager] COMMIT epoch={new_epoch} "
                    f"model={target_path.name} in {duration_ms:.0f}ms"
                )
                self._state = TransitionState.IDLE
                return True

            except Exception as e:
                # ROLLBACK
                logger.error(f"[TransitionManager] ROLLBACK: {e}")
                self._state = TransitionState.ROLLBACK
                self._draining = False
                try:
                    if 'new_grant' in locals() and new_grant:
                        await new_grant.rollback(str(e))
                    if old_path and old_path.exists():
                        await self._executor.load(old_path)
                        # Re-acquire grant for old model
                        restored_grant = await self._vram_authority.request(
                            "model-restored", old_path.stat().st_size, VRAMPriority.CRITICAL,
                        )
                        self._current_grant = restored_grant
                except Exception as rollback_err:
                    logger.critical(
                        f"[TransitionManager] ROLLBACK FAILED: {rollback_err}. "
                        f"Entering DEGRADED state -- no model loaded, manual intervention required."
                    )
                    self._degraded = True
                    self._current_model_path = None
                    self._current_grant = None
                    self._emit_event(ModelTransitionEvent(
                        event_type="transition_degraded",
                        transition_id=transition_id,
                        trigger=proposal.trigger,
                        from_model=str(old_path),
                        to_model="NONE",
                        from_quant=None, to_quant="NONE",
                        model_epoch=self._epoch.model_epoch,
                        duration_ms=(time.monotonic() - start_time) * 1000,
                        outcome="degraded",
                        reason=f"Rollback failed: {rollback_err}",
                        timestamp=time.time(),
                    ))

                self._emit_event(ModelTransitionEvent(
                    event_type="transition_failed", transition_id=transition_id,
                    trigger=proposal.trigger, from_model=str(old_path),
                    to_model=str(target_path), from_quant=None,
                    to_quant=proposal.selected_variant.quant_name,
                    model_epoch=self._epoch.model_epoch,
                    duration_ms=(time.monotonic() - start_time) * 1000,
                    outcome="rollback", reason=str(e), timestamp=time.time(),
                ))
                self._state = TransitionState.IDLE
                return False

    def request_started(self) -> None:
        self._active_requests += 1

    def request_completed(self) -> None:
        self._active_requests = max(0, self._active_requests - 1)
        if self._active_requests == 0 and self._drain_event:
            self._drain_event.set()

    def status(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "model_epoch": self._epoch.model_epoch,
            "cache_epoch": self._epoch.cache_epoch,
            "current_model": str(self._current_model_path) if self._current_model_path else None,
            "active_requests": self._active_requests,
            "swap_count": len(self._swap_times),
        }
