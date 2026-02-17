"""
Reactor-Core Watcher - Auto-Deployment Pipeline
================================================

Watches for new models from reactor-core training pipeline and
automatically deploys them to JARVIS-Prime with:
- File system watching for new GGUF models
- Validation before deployment
- Automatic hot-swap with rollback capability
- Telemetry reporting for training feedback loop

Integration Flow:
1. reactor-core completes training and converts to GGUF
2. GGUF file dropped into watch directory
3. This watcher detects new file
4. Validates model integrity
5. Registers with ModelRegistry
6. Triggers hot-swap deployment
7. Reports deployment status back for reactor-core feedback

Directory Structure:
    /app/reactor-core-output/
        ├── pending/          # New models waiting to deploy
        ├── deployed/         # Successfully deployed models
        ├── failed/           # Models that failed validation
        └── manifest.json     # Model metadata from reactor-core
"""

from __future__ import annotations

import asyncio
import enum
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

logger = logging.getLogger(__name__)


# ============================================================================
# Pipeline Event Logger (v1.1)
# ============================================================================
# Simple JSONL event logger for cross-repo event tracing.
# TrinityEventBus is not importable from jarvis-prime, so we write
# structured events to a shared JSONL file.

import uuid as _uuid_module

_PIPELINE_EVENTS_DIR = Path(
    os.environ.get("JARVIS_PIPELINE_EVENTS_DIR",
                    str(Path.home() / ".jarvis" / "reactor" / "events"))
)
_PIPELINE_EVENTS_FILE = _PIPELINE_EVENTS_DIR / "pipeline_events.jsonl"


def emit_pipeline_event(
    topic: str,
    payload: Optional[Dict[str, Any]] = None,
    correlation_id: str = "",
    causation_id: str = "",
) -> Optional[str]:
    """
    Write a structured pipeline event to the shared JSONL log.

    Args:
        topic: Event topic (e.g. "model.deployed", "probation.committed").
        payload: Event payload data.
        correlation_id: Correlation ID for distributed tracing.
        causation_id: ID of the event that caused this one.

    Returns:
        The event_id string, or None on failure.
    """
    event_id = str(_uuid_module.uuid4())
    event = {
        "event_id": event_id,
        "topic": topic,
        "source": "prime",
        "timestamp": datetime.now().isoformat(),
        "correlation_id": correlation_id or event_id,
        "causation_id": causation_id,
        "payload": payload or {},
    }
    try:
        _PIPELINE_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_PIPELINE_EVENTS_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        logger.debug(f"[PipelineEvent] {topic} (id={event_id[:8]})")
        return event_id
    except Exception as e:
        logger.debug(f"[PipelineEvent] Failed to write event: {e}")
        return None


# ============================================================================
# Cross-Repo Feedback
# ============================================================================

def resolve_cross_repo_dir() -> Path:
    """
    Resolve the cross-repo directory for inter-repo communication.

    Checks JARVIS_CROSS_REPO_DIR env var first, falls back to
    ~/.jarvis/cross_repo.
    """
    env_dir = os.environ.get("JARVIS_CROSS_REPO_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".jarvis" / "cross_repo"


def write_deployment_feedback(
    cross_repo_dir: Path,
    model_id: str,
    status: str,  # "success", "failed", "rollback"
    previous_model: Optional[str] = None,
    reactor_job_id: Optional[str] = None,
    first_inference_latency_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Write deployment feedback to cross-repo directory for Reactor-Core.

    This closes the deployment feedback loop: Reactor-Core trains a model,
    Prime deploys it, and this function reports the result back so
    Reactor-Core knows whether its trained model deployed successfully.

    The file is written atomically via a temporary file to prevent
    partial reads by Reactor-Core.

    Args:
        cross_repo_dir: Path to the shared cross-repo directory.
        model_id: Identifier of the deployed model.
        status: One of "success", "failed", or "rollback".
        previous_model: The model that was active before this deployment.
        reactor_job_id: Reactor-Core's training job ID (for tracing).
        first_inference_latency_ms: Latency of the first inference after deploy.
        error: Error message if deployment failed.
    """
    # Ensure directory exists
    cross_repo_dir = Path(cross_repo_dir)
    cross_repo_dir.mkdir(parents=True, exist_ok=True)

    feedback = {
        "model_id": model_id,
        "deployment_status": status,
        "deployed_at": datetime.now().isoformat(),
        "previous_model": previous_model,
        "health_check_passed": status == "success",
        "first_inference_latency_ms": first_inference_latency_ms,
        "error": error,
        "reactor_job_id": reactor_job_id,
    }

    status_file = cross_repo_dir / "deployment_status.json"
    # Atomic write via temp file to prevent partial reads
    tmp_file = status_file.with_suffix(".tmp")
    try:
        tmp_file.write_text(json.dumps(feedback, indent=2, default=str))
        tmp_file.rename(status_file)
    except Exception:
        # Clean up temp file on failure
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    logger.info(f"[ReactorWatcher] Deployment feedback written: {status} for {model_id}")


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ReactorCoreModelManifest:
    """Manifest for a model produced by reactor-core."""
    model_id: str
    version: str
    training_run_id: str
    base_model: str
    quantization_method: str
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    file_path: str
    sha256: str
    created_at: datetime
    lineage: str = "reactor-core"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "training_run_id": self.training_run_id,
            "base_model": self.base_model,
            "quantization_method": self.quantization_method,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "file_path": self.file_path,
            "sha256": self.sha256,
            "created_at": self.created_at.isoformat(),
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReactorCoreModelManifest":
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            training_run_id=data["training_run_id"],
            base_model=data["base_model"],
            quantization_method=data["quantization_method"],
            training_config=data.get("training_config", {}),
            metrics=data.get("metrics", {}),
            file_path=data["file_path"],
            sha256=data["sha256"],
            created_at=datetime.fromisoformat(data["created_at"]),
            lineage=data.get("lineage", "reactor-core"),
        )


@dataclass
class DeploymentResult:
    """Result of a model deployment attempt."""
    success: bool
    model_id: str
    version: str
    deployed_at: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "model_id": self.model_id,
            "version": self.version,
            "deployed_at": self.deployed_at.isoformat(),
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


# ============================================================================
# Post-Deployment Probation
# ============================================================================

class ProbationStatus(enum.Enum):
    """Status of a post-deployment probation window."""
    MONITORING = "monitoring"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


@dataclass
class ProbationConfig:
    """Configuration for the post-deployment probation window.

    All thresholds are configurable via constructor parameters.
    Defaults: 30 minute window, probe every 60 seconds.
    """
    probation_duration_s: float = 1800.0
    probe_interval_s: float = 60.0
    commit_threshold: float = 0.8
    rollback_threshold: float = 0.5
    emergency_error_rate_multiplier: float = 5.0
    min_probes_before_decision: int = 3
    # Health score weights
    error_weight: float = 0.5
    latency_weight: float = 0.3
    status_weight: float = 0.2
    # Latency normalization: latencies above this (seconds) get max penalty
    latency_ceiling_s: float = 10.0
    # HTTP timeout for health probes
    probe_timeout_s: float = 10.0


class ProbationMonitor:
    """
    Post-deployment probation monitor.

    After a successful model deployment, starts a probation window that
    periodically probes the deployed model's health endpoint. Based on
    accumulated health data, decides whether to commit (keep) or roll back
    the deployment.

    Health score formula:
        1.0 - (error_weight * error_rate) - (latency_weight * normalized_latency)
        with status-based adjustments for non-healthy states.

    Emergency rollback: triggered when the current error rate exceeds
    the baseline error rate by more than emergency_error_rate_multiplier.

    State is persisted to disk so probation can resume after a crash/restart.
    """

    def __init__(
        self,
        config: ProbationConfig,
        health_endpoint: str,
        cross_repo_dir: Path,
        previous_model_dir: Optional[Path] = None,
        state_file: Optional[Path] = None,
        hot_swap_callback: Optional[Callable] = None,
    ):
        self._config = config
        self._health_endpoint = health_endpoint
        self._cross_repo_dir = Path(cross_repo_dir)
        self._previous_model_dir = Path(previous_model_dir) if previous_model_dir else (
            Path.home() / ".jarvis" / "reactor" / "models" / "previous"
        )
        self._state_file = Path(state_file) if state_file else (
            Path.home() / ".jarvis" / "reactor" / "probation_state.json"
        )
        self._hot_swap_callback = hot_swap_callback

        # Runtime state
        self._status: ProbationStatus = ProbationStatus.MONITORING
        self._model_id: Optional[str] = None
        self._started_at: Optional[float] = None
        self._probe_history: List[Dict[str, Any]] = []
        self._baseline_error_rate: float = 0.0
        self._deployed_model_path: Optional[Path] = None
        self._previous_model_path: Optional[Path] = None
        self._probation_task: Optional[asyncio.Task] = None

    @property
    def config(self) -> ProbationConfig:
        return self._config

    @property
    def status(self) -> ProbationStatus:
        return self._status

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start(
        self,
        model_id: str,
        deployed_model_path: Optional[Path],
        previous_model_path: Optional[Path] = None,
        baseline_error_rate: float = 0.0,
    ) -> None:
        """
        Begin the probation window for a newly deployed model.

        Args:
            model_id: Identifier of the deployed model.
            deployed_model_path: Path to the deployed GGUF file.
            previous_model_path: Path to the model that was active before.
                If provided, it will be backed up to previous_model_dir.
            baseline_error_rate: The error rate of the previous model
                (used for emergency rollback threshold).
        """
        self._model_id = model_id
        self._deployed_model_path = deployed_model_path
        self._previous_model_path = previous_model_path
        self._baseline_error_rate = baseline_error_rate
        self._status = ProbationStatus.MONITORING
        self._started_at = time.time()
        self._probe_history = []

        # Back up the previous model if one was provided
        if previous_model_path and previous_model_path.exists():
            await self.backup_previous_model(previous_model_path)

        # Persist initial state
        await self.persist_state()

        # v1.1: Emit probation.started pipeline event
        emit_pipeline_event(
            topic="probation.started",
            payload={
                "model_id": model_id,
                "duration_s": self._config.probation_duration_s,
                "probe_interval_s": self._config.probe_interval_s,
            },
        )

        logger.info(
            f"[Probation] Started probation for {model_id}. "
            f"Duration: {self._config.probation_duration_s}s, "
            f"probe interval: {self._config.probe_interval_s}s"
        )

    # ------------------------------------------------------------------
    # Health Probing
    # ------------------------------------------------------------------

    async def probe(self) -> Dict[str, Any]:
        """
        Probe the deployed model's health endpoint.

        Returns a dict with:
            health_score: float [0.0, 1.0]
            reachable: bool
            latency_s: float
            error_rate: float (derived from HTTP status)
            http_status: int or None
            response_data: dict or None
            timestamp: float
        """
        start = time.monotonic()
        http_status = None
        response_data = None
        reachable = False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self._health_endpoint,
                    timeout=aiohttp.ClientTimeout(total=self._config.probe_timeout_s),
                ) as resp:
                    http_status = resp.status
                    try:
                        response_data = await resp.json()
                    except Exception:
                        response_data = {}
                    reachable = True
        except Exception as e:
            logger.warning(f"[Probation] Health probe failed: {e}")

        latency_s = time.monotonic() - start

        # Calculate health score
        if reachable and http_status is not None and response_data is not None:
            health_score = self.calculate_health_score(
                http_status=http_status,
                response_data=response_data,
                latency_s=latency_s,
            )
            error_rate = 0.0 if 200 <= http_status < 300 else 1.0
        else:
            health_score = 0.0
            error_rate = 1.0

        result = {
            "health_score": health_score,
            "reachable": reachable,
            "latency_s": latency_s,
            "error_rate": error_rate,
            "http_status": http_status,
            "response_data": response_data,
            "timestamp": time.time(),
        }

        self._probe_history.append(result)
        return result

    def calculate_health_score(
        self,
        http_status: int,
        response_data: Dict[str, Any],
        latency_s: float,
    ) -> float:
        """
        Calculate a health score in [0.0, 1.0].

        Formula:
            base = 1.0
            - error_weight * error_rate
            - latency_weight * min(latency_s / latency_ceiling, 1.0)
            - status_weight * status_penalty

        Where:
            error_rate = 1.0 if HTTP status >= 400 else 0.0
            status_penalty = 0.0 for "healthy"/"ready", 0.5 for "starting", 1.0 for "error"
        """
        cfg = self._config

        # Error component
        error_rate = 1.0 if http_status >= 400 else 0.0

        # Latency component (normalized to [0, 1])
        normalized_latency = min(latency_s / cfg.latency_ceiling_s, 1.0)

        # Status component
        status_str = response_data.get("status", "unknown")
        if status_str in ("healthy", "ready"):
            status_penalty = 0.0
        elif status_str == "starting":
            status_penalty = 0.5
        else:
            status_penalty = 1.0

        score = 1.0 - (
            cfg.error_weight * error_rate
            + cfg.latency_weight * normalized_latency
            + cfg.status_weight * status_penalty
        )

        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> ProbationStatus:
        """
        Evaluate accumulated probe data and decide: COMMITTED, ROLLING_BACK, or MONITORING.

        Decision logic:
        1. If fewer than min_probes_before_decision probes, stay MONITORING.
        2. Emergency: if latest error rate > baseline * multiplier, ROLLING_BACK.
        3. Average health_score >= commit_threshold -> COMMITTED.
        4. Average health_score < rollback_threshold -> ROLLING_BACK.
        5. Otherwise, MONITORING (continue probing).
        """
        if len(self._probe_history) < self._config.min_probes_before_decision:
            return ProbationStatus.MONITORING

        # Emergency rollback: check if current error rate massively exceeds baseline
        if self._baseline_error_rate > 0:
            recent_error_rates = [p["error_rate"] for p in self._probe_history[-3:]]
            avg_recent_error = sum(recent_error_rates) / len(recent_error_rates)
            if avg_recent_error > self._baseline_error_rate * self._config.emergency_error_rate_multiplier:
                logger.warning(
                    f"[Probation] EMERGENCY: error rate {avg_recent_error:.3f} "
                    f"> {self._config.emergency_error_rate_multiplier}x baseline "
                    f"({self._baseline_error_rate:.3f})"
                )
                return ProbationStatus.ROLLING_BACK

        # Average health score across all probes
        scores = [p["health_score"] for p in self._probe_history]
        avg_score = sum(scores) / len(scores)

        if avg_score >= self._config.commit_threshold:
            return ProbationStatus.COMMITTED
        elif avg_score < self._config.rollback_threshold:
            return ProbationStatus.ROLLING_BACK
        else:
            return ProbationStatus.MONITORING

    # ------------------------------------------------------------------
    # Commit
    # ------------------------------------------------------------------

    async def commit(self) -> None:
        """
        Commit the deployment: mark as successful, update feedback, clean state.
        """
        self._status = ProbationStatus.COMMITTED

        # Calculate summary stats
        scores = [p["health_score"] for p in self._probe_history]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        latencies = [p["latency_s"] for p in self._probe_history]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Write deployment feedback with probation details
        try:
            feedback = {
                "model_id": self._model_id,
                "deployment_status": "success",
                "deployed_at": datetime.now().isoformat(),
                "previous_model": (
                    str(self._previous_model_path) if self._previous_model_path else None
                ),
                "health_check_passed": True,
                "first_inference_latency_ms": (
                    latencies[0] * 1000 if latencies else None
                ),
                "error": None,
                "reactor_job_id": None,
                "probation": {
                    "result": "committed",
                    "duration_s": time.time() - self._started_at if self._started_at else 0,
                    "probes_count": len(self._probe_history),
                    "avg_health_score": round(avg_score, 4),
                    "avg_latency_s": round(avg_latency, 4),
                },
            }
            self._cross_repo_dir.mkdir(parents=True, exist_ok=True)
            status_file = self._cross_repo_dir / "deployment_status.json"
            tmp_file = status_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(feedback, indent=2, default=str))
            tmp_file.rename(status_file)
        except Exception as e:
            logger.error(f"[Probation] Failed to write commit feedback: {e}")

        # Clean up state file
        await self._cleanup_state_file()

        # v1.1: Emit probation.committed pipeline event
        emit_pipeline_event(
            topic="probation.committed",
            payload={
                "model_id": self._model_id,
                "avg_health_score": round(avg_score, 4),
                "avg_latency_s": round(avg_latency, 4),
                "probes_count": len(self._probe_history),
            },
        )

        logger.info(
            f"[Probation] COMMITTED {self._model_id}. "
            f"avg_score={avg_score:.3f}, probes={len(self._probe_history)}"
        )

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    async def rollback(self) -> None:
        """
        Roll back the deployment: restore previous model, update feedback, clean state.
        """
        self._status = ProbationStatus.ROLLING_BACK

        # Restore previous model if available
        if self._previous_model_path and self._previous_model_path.exists():
            try:
                if self._deployed_model_path:
                    # Copy previous model to where the deployed model is
                    dest = self._deployed_model_path.parent / self._previous_model_path.name
                    shutil.copy2(str(self._previous_model_path), str(dest))
                    logger.info(f"[Probation] Restored previous model: {dest}")

                    # Trigger hot-swap if callback provided
                    if self._hot_swap_callback:
                        try:
                            await self._hot_swap_callback(dest, "rollback")
                        except Exception as e:
                            logger.error(f"[Probation] Hot-swap callback failed during rollback: {e}")
            except Exception as e:
                logger.error(f"[Probation] Failed to restore previous model: {e}")
        else:
            logger.warning("[Probation] No previous model available to restore")

        # Calculate summary stats
        scores = [p["health_score"] for p in self._probe_history]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Write deployment feedback
        try:
            feedback = {
                "model_id": self._model_id,
                "deployment_status": "rollback",
                "deployed_at": datetime.now().isoformat(),
                "previous_model": (
                    str(self._previous_model_path) if self._previous_model_path else None
                ),
                "health_check_passed": False,
                "first_inference_latency_ms": None,
                "error": f"Probation failed: avg_score={avg_score:.3f}",
                "reactor_job_id": None,
                "probation": {
                    "result": "rolled_back",
                    "duration_s": time.time() - self._started_at if self._started_at else 0,
                    "probes_count": len(self._probe_history),
                    "avg_health_score": round(avg_score, 4),
                },
            }
            self._cross_repo_dir.mkdir(parents=True, exist_ok=True)
            status_file = self._cross_repo_dir / "deployment_status.json"
            tmp_file = status_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(feedback, indent=2, default=str))
            tmp_file.rename(status_file)
        except Exception as e:
            logger.error(f"[Probation] Failed to write rollback feedback: {e}")

        self._status = ProbationStatus.ROLLED_BACK

        # Clean up state file
        await self._cleanup_state_file()

        # v1.1: Emit probation.rollback pipeline event
        emit_pipeline_event(
            topic="probation.rollback",
            payload={
                "model_id": self._model_id,
                "avg_health_score": round(avg_score, 4),
                "probes_count": len(self._probe_history),
            },
        )

        logger.info(
            f"[Probation] ROLLED BACK {self._model_id}. "
            f"avg_score={avg_score:.3f}, probes={len(self._probe_history)}"
        )

    # ------------------------------------------------------------------
    # Model Backup
    # ------------------------------------------------------------------

    async def backup_previous_model(self, model_path: Path) -> None:
        """
        Back up the previous model to the previous_model_dir.

        Uses copy (not move) so the current deployment directory is untouched.

        Args:
            model_path: Path to the model file to back up.
        """
        self._previous_model_dir.mkdir(parents=True, exist_ok=True)
        dest = self._previous_model_dir / model_path.name
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, shutil.copy2, str(model_path), str(dest)
            )
            logger.info(f"[Probation] Backed up previous model to {dest}")
        except Exception as e:
            logger.error(f"[Probation] Failed to back up model: {e}")

    # ------------------------------------------------------------------
    # State Persistence
    # ------------------------------------------------------------------

    async def persist_state(self) -> None:
        """Persist probation state to disk for crash recovery."""
        state = {
            "model_id": self._model_id,
            "status": self._status.value,
            "started_at": self._started_at,
            "probe_history": self._probe_history,
            "baseline_error_rate": self._baseline_error_rate,
            "deployed_model_path": str(self._deployed_model_path) if self._deployed_model_path else None,
            "previous_model_path": str(self._previous_model_path) if self._previous_model_path else None,
        }
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self._state_file.with_suffix(".tmp")
        try:
            tmp_file.write_text(json.dumps(state, indent=2, default=str))
            tmp_file.rename(self._state_file)
        except Exception as e:
            logger.error(f"[Probation] Failed to persist state: {e}")
            if tmp_file.exists():
                tmp_file.unlink()

    async def restore_state(self) -> bool:
        """
        Restore probation state from disk.

        Returns True if state was restored, False if no state file exists.
        """
        if not self._state_file.exists():
            return False

        try:
            data = json.loads(self._state_file.read_text())
            self._model_id = data.get("model_id")
            self._status = ProbationStatus(data.get("status", "monitoring"))
            self._started_at = data.get("started_at")
            self._probe_history = data.get("probe_history", [])
            self._baseline_error_rate = data.get("baseline_error_rate", 0.0)

            deployed_path = data.get("deployed_model_path")
            self._deployed_model_path = Path(deployed_path) if deployed_path else None

            previous_path = data.get("previous_model_path")
            self._previous_model_path = Path(previous_path) if previous_path else None

            logger.info(
                f"[Probation] Restored state for {self._model_id}: "
                f"status={self._status.value}, probes={len(self._probe_history)}"
            )
            return True
        except Exception as e:
            logger.error(f"[Probation] Failed to restore state: {e}")
            return False

    async def _cleanup_state_file(self) -> None:
        """Remove the probation state file."""
        try:
            if self._state_file.exists():
                self._state_file.unlink()
        except Exception as e:
            logger.warning(f"[Probation] Failed to clean up state file: {e}")

    # ------------------------------------------------------------------
    # Background Probation Loop
    # ------------------------------------------------------------------

    async def run_probation_loop(self) -> ProbationStatus:
        """
        Run the full probation loop: probe periodically, evaluate, commit or rollback.

        This is meant to be run as an asyncio task. It returns the final status.
        """
        if self._status != ProbationStatus.MONITORING:
            return self._status

        deadline = (self._started_at or time.time()) + self._config.probation_duration_s

        while self._status == ProbationStatus.MONITORING and time.time() < deadline:
            try:
                await self.probe()
                await self.persist_state()

                decision = self.evaluate()
                if decision == ProbationStatus.COMMITTED:
                    await self.commit()
                    return self._status
                elif decision == ProbationStatus.ROLLING_BACK:
                    await self.rollback()
                    return self._status

                # Wait for next probe interval
                remaining = deadline - time.time()
                wait_time = min(self._config.probe_interval_s, remaining)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            except asyncio.CancelledError:
                logger.info("[Probation] Probation loop cancelled")
                await self.persist_state()
                raise
            except Exception as e:
                logger.error(f"[Probation] Error in probation loop: {e}")
                await asyncio.sleep(self._config.probe_interval_s)

        # Time expired without clear decision - evaluate one final time
        if self._status == ProbationStatus.MONITORING:
            decision = self.evaluate()
            if decision == ProbationStatus.ROLLING_BACK:
                await self.rollback()
            else:
                # Default to commit if not clearly bad after full probation window
                await self.commit()

        return self._status


# ============================================================================
# File System Handler
# ============================================================================

class ModelFileHandler(FileSystemEventHandler):
    """Watches for new GGUF files in the watch directory."""

    def __init__(self, callback: Callable[[Path], None]):
        self.callback = callback
        self._debounce: Dict[str, datetime] = {}
        self._debounce_seconds = 5.0

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            self._handle_file(Path(event.src_path))

    def on_moved(self, event):
        if isinstance(event, FileMovedEvent):
            self._handle_file(Path(event.dest_path))

    def _handle_file(self, path: Path):
        # Only handle GGUF files
        if not path.suffix == ".gguf":
            return

        # Only handle files in pending/ directory
        if path.parent.name != "pending":
            return

        # Debounce (wait for file to finish writing)
        now = datetime.now()
        last = self._debounce.get(str(path))
        if last and (now - last).total_seconds() < self._debounce_seconds:
            return
        self._debounce[str(path)] = now

        # Trigger callback
        logger.info(f"Detected new model: {path}")
        self.callback(path)


# ============================================================================
# Reactor-Core Watcher
# ============================================================================

class ReactorCoreWatcher:
    """
    Watches for models from reactor-core and auto-deploys them.

    Usage:
        watcher = ReactorCoreWatcher(
            watch_dir="/app/reactor-core-output",
            models_dir="/app/models",
            on_deploy_success=lambda result: print(f"Deployed {result.model_id}"),
            on_deploy_failure=lambda result: print(f"Failed {result.model_id}"),
        )
        await watcher.start()

        # ... runs in background ...

        await watcher.stop()
    """

    def __init__(
        self,
        watch_dir: str | Path,
        models_dir: str | Path,
        hot_swap_callback: Optional[Callable[[Path, str], asyncio.Future]] = None,
        on_deploy_success: Optional[Callable[[DeploymentResult], None]] = None,
        on_deploy_failure: Optional[Callable[[DeploymentResult], None]] = None,
        auto_deploy: bool = True,
        validation_enabled: bool = True,
        cross_repo_dir: Optional[Path] = None,
        probation_config: Optional[ProbationConfig] = None,
        health_endpoint: str = "http://localhost:8000/health",
    ):
        self.watch_dir = Path(watch_dir)
        self.models_dir = Path(models_dir)
        self.hot_swap_callback = hot_swap_callback
        self.on_deploy_success = on_deploy_success
        self.on_deploy_failure = on_deploy_failure
        self.auto_deploy = auto_deploy
        self.validation_enabled = validation_enabled
        self.cross_repo_dir = Path(cross_repo_dir) if cross_repo_dir else resolve_cross_repo_dir()
        self._probation_config = probation_config
        self._health_endpoint = health_endpoint

        # State
        self._observer: Optional[Observer] = None
        self._running = False
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._process_task: Optional[asyncio.Task] = None
        self._probation_monitor: Optional[ProbationMonitor] = None
        self._probation_task: Optional[asyncio.Task] = None

        # Statistics
        self._deployments_success = 0
        self._deployments_failed = 0
        self._last_deployment: Optional[datetime] = None

        # Ensure directories exist
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create required directories."""
        (self.watch_dir / "pending").mkdir(parents=True, exist_ok=True)
        (self.watch_dir / "deployed").mkdir(parents=True, exist_ok=True)
        (self.watch_dir / "failed").mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start the watcher."""
        if self._running:
            return

        logger.info(f"Starting ReactorCoreWatcher: {self.watch_dir}")

        # Start file observer
        handler = ModelFileHandler(self._on_new_model)
        self._observer = Observer()
        self._observer.schedule(
            handler,
            str(self.watch_dir / "pending"),
            recursive=False,
        )
        self._observer.start()

        # Start processing task
        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())

        # Check for any existing pending models
        await self._check_pending()

        logger.info("ReactorCoreWatcher started")

    async def stop(self) -> None:
        """Stop the watcher."""
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None

        logger.info("ReactorCoreWatcher stopped")

    def _on_new_model(self, path: Path) -> None:
        """Called when a new model file is detected."""
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: self._processing_queue.put_nowait(path)
            )
        except Exception as e:
            logger.error(f"Error queuing model: {e}")

    async def _check_pending(self) -> None:
        """Check for existing pending models on startup."""
        pending_dir = self.watch_dir / "pending"
        for gguf_file in pending_dir.glob("*.gguf"):
            logger.info(f"Found pending model: {gguf_file}")
            await self._processing_queue.put(gguf_file)

    async def _process_loop(self) -> None:
        """Background loop to process queued models."""
        while self._running:
            try:
                # Wait for next model
                path = await asyncio.wait_for(
                    self._processing_queue.get(),
                    timeout=30.0,
                )

                # Process the model
                await self._process_model(path)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process loop: {e}")

    async def _process_model(self, model_path: Path) -> None:
        """Process a new model file."""
        logger.info(f"Processing model: {model_path}")

        # Load manifest if exists
        manifest = await self._load_manifest(model_path)

        # Validate model
        if self.validation_enabled:
            is_valid = await self._validate_model(model_path, manifest)
            if not is_valid:
                await self._handle_failed_deployment(
                    model_path,
                    manifest,
                    "Model validation failed",
                )
                return

        # Deploy if auto_deploy enabled
        if self.auto_deploy:
            await self._deploy_model(model_path, manifest)
        else:
            logger.info(f"Auto-deploy disabled, model ready: {model_path}")

    async def _load_manifest(self, model_path: Path) -> Optional[ReactorCoreModelManifest]:
        """Load manifest for a model."""
        # Look for accompanying manifest file
        manifest_path = model_path.with_suffix(".json")
        if not manifest_path.exists():
            # Try global manifest
            manifest_path = self.watch_dir / "manifest.json"

        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                    # Handle single model or model list
                    if isinstance(data, dict) and "model_id" in data:
                        return ReactorCoreModelManifest.from_dict(data)
                    elif isinstance(data, dict) and "models" in data:
                        # Find matching model in list
                        for m in data["models"]:
                            if Path(m["file_path"]).name == model_path.name:
                                return ReactorCoreModelManifest.from_dict(m)
            except Exception as e:
                logger.warning(f"Error loading manifest: {e}")

        return None

    async def _validate_model(
        self,
        model_path: Path,
        manifest: Optional[ReactorCoreModelManifest],
    ) -> bool:
        """Validate model file."""
        # Check file exists and is not empty
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        if model_path.stat().st_size < 1024:  # At least 1KB
            logger.error(f"Model file too small: {model_path}")
            return False

        # Verify checksum if available
        if manifest and manifest.sha256:
            actual_sha256 = await self._calculate_sha256(model_path)
            if actual_sha256 != manifest.sha256:
                logger.error(f"SHA256 mismatch for {model_path}")
                return False

        # Basic GGUF header validation
        try:
            with open(model_path, "rb") as f:
                magic = f.read(4)
                if magic != b"GGUF":
                    logger.error(f"Invalid GGUF magic: {magic}")
                    return False
        except Exception as e:
            logger.error(f"Error reading model: {e}")
            return False

        logger.info(f"Model validation passed: {model_path}")
        return True

    async def _calculate_sha256(self, path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        loop = asyncio.get_event_loop()

        def _hash():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()

        return await loop.run_in_executor(None, _hash)

    async def _deploy_model(
        self,
        model_path: Path,
        manifest: Optional[ReactorCoreModelManifest],
    ) -> None:
        """Deploy a validated model."""
        model_id = manifest.model_id if manifest else model_path.stem
        version = manifest.version if manifest else datetime.now().strftime("v%Y%m%d-%H%M%S")

        try:
            # Copy to models directory
            dest_path = self.models_dir / f"{model_id}-{version}.gguf"
            shutil.copy2(model_path, dest_path)

            # Trigger hot-swap if callback provided
            if self.hot_swap_callback:
                await self.hot_swap_callback(dest_path, version)

            # Move original to deployed/
            deployed_path = self.watch_dir / "deployed" / model_path.name
            shutil.move(str(model_path), str(deployed_path))

            # Move manifest too
            manifest_src = model_path.with_suffix(".json")
            if manifest_src.exists():
                shutil.move(
                    str(manifest_src),
                    str(self.watch_dir / "deployed" / manifest_src.name),
                )

            # Create result
            result = DeploymentResult(
                success=True,
                model_id=model_id,
                version=version,
                deployed_at=datetime.now(),
                metrics={
                    "file_size_mb": dest_path.stat().st_size / (1024 * 1024),
                },
            )

            # Update statistics
            self._deployments_success += 1
            self._last_deployment = datetime.now()

            # Notify callback
            if self.on_deploy_success:
                self.on_deploy_success(result)

            # Start probation if configured, otherwise write immediate success feedback
            if self._probation_config is not None:
                await self._start_probation(
                    model_id=model_id,
                    deployed_model_path=dest_path,
                    previous_model_path=None,  # TODO: track previous model path
                    manifest=manifest,
                )
            else:
                # Write deployment feedback for Reactor-Core (no probation)
                try:
                    previous_model_id = None
                    write_deployment_feedback(
                        cross_repo_dir=self.cross_repo_dir,
                        model_id=model_id,
                        status="success",
                        previous_model=previous_model_id,
                        reactor_job_id=manifest.training_run_id if manifest else None,
                    )
                except Exception as fb_err:
                    logger.warning(f"Failed to write deployment feedback: {fb_err}")

            # v1.1: Emit model.deployed pipeline event
            emit_pipeline_event(
                topic="model.deployed",
                payload={
                    "model_id": model_id,
                    "version": version,
                    "file_size_mb": dest_path.stat().st_size / (1024 * 1024),
                    "reactor_job_id": manifest.training_run_id if manifest else None,
                },
                correlation_id=manifest.training_run_id if manifest else "",
            )

            logger.info(f"Successfully deployed: {model_id} v{version}")

        except Exception as e:
            await self._handle_failed_deployment(model_path, manifest, str(e))

    async def _handle_failed_deployment(
        self,
        model_path: Path,
        manifest: Optional[ReactorCoreModelManifest],
        error_message: str,
    ) -> None:
        """Handle a failed deployment."""
        model_id = manifest.model_id if manifest else model_path.stem
        version = manifest.version if manifest else "unknown"

        # Move to failed/
        try:
            failed_path = self.watch_dir / "failed" / model_path.name
            if model_path.exists():
                shutil.move(str(model_path), str(failed_path))
        except Exception as e:
            logger.error(f"Error moving failed model: {e}")

        # Create result
        result = DeploymentResult(
            success=False,
            model_id=model_id,
            version=version,
            deployed_at=datetime.now(),
            error_message=error_message,
        )

        # Update statistics
        self._deployments_failed += 1

        # Notify callback
        if self.on_deploy_failure:
            self.on_deploy_failure(result)

        # Write deployment feedback for Reactor-Core
        try:
            write_deployment_feedback(
                cross_repo_dir=self.cross_repo_dir,
                model_id=model_id,
                status="failed",
                reactor_job_id=manifest.training_run_id if manifest else None,
                error=error_message,
            )
        except Exception as fb_err:
            logger.warning(f"Failed to write deployment feedback: {fb_err}")

        logger.error(f"Deployment failed for {model_id}: {error_message}")

    async def _start_probation(
        self,
        model_id: str,
        deployed_model_path: Path,
        previous_model_path: Optional[Path],
        manifest: Optional[ReactorCoreModelManifest] = None,
    ) -> None:
        """Start post-deployment probation for a newly deployed model."""
        if not self._probation_config:
            return

        # Cancel any existing probation
        if self._probation_task and not self._probation_task.done():
            self._probation_task.cancel()
            try:
                await self._probation_task
            except asyncio.CancelledError:
                pass

        self._probation_monitor = ProbationMonitor(
            config=self._probation_config,
            health_endpoint=self._health_endpoint,
            cross_repo_dir=self.cross_repo_dir,
            hot_swap_callback=self.hot_swap_callback,
        )

        await self._probation_monitor.start(
            model_id=model_id,
            deployed_model_path=deployed_model_path,
            previous_model_path=previous_model_path,
        )

        # Run probation in background
        self._probation_task = asyncio.create_task(
            self._run_probation(manifest)
        )

        logger.info(f"[ReactorWatcher] Started probation for {model_id}")

    async def _run_probation(
        self, manifest: Optional[ReactorCoreModelManifest]
    ) -> None:
        """Background task that runs the probation loop and handles the result."""
        if not self._probation_monitor:
            return

        try:
            final_status = await self._probation_monitor.run_probation_loop()
            logger.info(
                f"[ReactorWatcher] Probation finished: {final_status.value} "
                f"for {self._probation_monitor._model_id}"
            )
        except asyncio.CancelledError:
            logger.info("[ReactorWatcher] Probation task cancelled")
        except Exception as e:
            logger.error(f"[ReactorWatcher] Probation loop error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get watcher status, including probation state if active."""
        pending_count = len(list((self.watch_dir / "pending").glob("*.gguf")))
        deployed_count = len(list((self.watch_dir / "deployed").glob("*.gguf")))
        failed_count = len(list((self.watch_dir / "failed").glob("*.gguf")))

        status = {
            "running": self._running,
            "watch_dir": str(self.watch_dir),
            "auto_deploy": self.auto_deploy,
            "pending_models": pending_count,
            "deployed_models": deployed_count,
            "failed_models": failed_count,
            "deployments_success": self._deployments_success,
            "deployments_failed": self._deployments_failed,
            "last_deployment": self._last_deployment.isoformat() if self._last_deployment else None,
        }

        if self._probation_monitor:
            status["probation"] = {
                "status": self._probation_monitor.status.value,
                "model_id": self._probation_monitor._model_id,
                "probes_count": len(self._probation_monitor._probe_history),
            }

        return status


# ============================================================================
# Reactor-Core Output Writer (for reactor-core side)
# ============================================================================

async def push_model_to_jarvis_prime(
    model_path: Path,
    watch_dir: Path,
    manifest: ReactorCoreModelManifest,
) -> None:
    """
    Push a trained model to JARVIS-Prime's watch directory.

    This function should be called from reactor-core after training completes.

    Args:
        model_path: Path to the trained GGUF model
        watch_dir: JARVIS-Prime watch directory (pending/)
        manifest: Model manifest with metadata
    """
    pending_dir = watch_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Copy model to pending
    dest_path = pending_dir / model_path.name
    shutil.copy2(model_path, dest_path)

    # Write manifest
    manifest_path = dest_path.with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info(f"Pushed model to JARVIS-Prime: {dest_path}")
