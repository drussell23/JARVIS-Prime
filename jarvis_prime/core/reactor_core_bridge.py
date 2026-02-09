"""
Reactor Core Bridge - Training Pipeline Integration
====================================================

v91.0 - Comprehensive Training Pipeline Integration with Reactor Core

This module provides integration between JARVIS Prime and Reactor Core:
- Training job submission and tracking
- Fine-tuning request management
- Model deployment pipeline
- Training data export from experiences
- Bidirectional communication

ARCHITECTURE:
    JARVIS Prime                     Reactor Core
    ============                     ============
    Experiences → TrainingDataExporter → Training Pipeline
    ↓                                    ↓
    TrainingJobManager ← ←           ← Model Training
    ↓                                    ↓
    Model Deployment ← ReactorCoreWatcher ← GGUF Output

COMMUNICATION:
    - Message Queue (Redis/RabbitMQ)
    - File System Watchers
    - REST API (optional)
    - WebSocket (real-time updates)

USAGE:
    bridge = await get_reactor_core_bridge()

    # Submit fine-tuning job
    job_id = await bridge.submit_finetuning_job(
        base_model="prime-7b-chat-v1",
        training_data=experiences,
        config=FineTuningConfig(epochs=3),
    )

    # Track job status
    status = await bridge.get_job_status(job_id)

    # Get training metrics
    metrics = await bridge.get_training_metrics(job_id)

INTEGRATION:
    - ContinuousLearning: Exports experiences for training
    - RLHF Pipeline: Preference data for alignment
    - PrimeModel: Receives deployed models
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Deque,
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
# ENUMS & TYPES
# =============================================================================


class JobStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Types of training jobs."""
    FINE_TUNING = "fine_tuning"
    RLHF = "rlhf"
    CONTINUAL_LEARNING = "continual_learning"
    DISTILLATION = "distillation"
    QUANTIZATION = "quantization"


class CommunicationMethod(Enum):
    """Communication methods with Reactor Core."""
    FILE_SYSTEM = "file_system"
    MESSAGE_QUEUE = "message_queue"
    REST_API = "rest_api"
    WEBSOCKET = "websocket"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning job."""
    # Training
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_steps: Optional[int] = None

    # Model
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    use_8bit: bool = True
    gradient_checkpointing: bool = True

    # Data
    max_seq_length: int = 2048
    train_split: float = 0.9

    # Output
    output_format: str = "gguf"
    quantization: str = "q4_k_m"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FineTuningConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingData:
    """Training data for fine-tuning."""
    # Data sources
    experiences: List[Dict[str, Any]] = field(default_factory=list)
    preferences: List[Dict[str, Any]] = field(default_factory=list)
    demonstrations: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    total_samples: int = 0
    source: str = "jarvis_prime"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiences": self.experiences,
            "preferences": self.preferences,
            "demonstrations": self.demonstrations,
            "total_samples": self.total_samples,
            "source": self.source,
            "created_at": self.created_at,
        }


@dataclass
class TrainingJob:
    """A training job submitted to Reactor Core."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    type: JobType = JobType.FINE_TUNING
    status: JobStatus = JobStatus.PENDING

    # Configuration
    base_model: str = ""
    config: Optional[FineTuningConfig] = None
    training_data_path: Optional[str] = None

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    # Progress
    current_step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[str] = field(default_factory=list)

    # Output
    output_model_path: Optional[str] = None
    output_model_id: Optional[str] = None

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "base_model": self.base_model,
            "config": self.config.to_dict() if self.config else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": self.progress_percent,
            "metrics": self.metrics,
            "output_model_id": self.output_model_id,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingJob":
        job = cls(
            id=data.get("id", str(uuid.uuid4())[:12]),
            type=JobType(data.get("type", "fine_tuning")),
            status=JobStatus(data.get("status", "pending")),
            base_model=data.get("base_model", ""),
            created_at=data.get("created_at", time.time()),
        )
        if data.get("config"):
            job.config = FineTuningConfig.from_dict(data["config"])
        job.metrics = data.get("metrics", {})
        job.current_step = data.get("current_step", 0)
        job.total_steps = data.get("total_steps", 0)
        job.progress_percent = data.get("progress_percent", 0.0)
        job.error_message = data.get("error_message")
        return job


@dataclass
class ReactorCoreConfig:
    """Configuration for Reactor Core bridge."""
    # Communication
    communication_method: CommunicationMethod = field(default_factory=lambda: CommunicationMethod(
        os.getenv("REACTOR_COMM_METHOD", "file_system")
    ))

    # Paths (file system communication)
    jobs_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "REACTOR_JOBS_DIR", str(Path.home() / ".jarvis" / "reactor" / "jobs")
    )))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "REACTOR_DATA_DIR", str(Path.home() / ".jarvis" / "reactor" / "data")
    )))
    models_output_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "REACTOR_MODELS_DIR", str(Path.home() / ".jarvis" / "reactor" / "models")
    )))

    # API (REST communication)
    api_url: str = field(default_factory=lambda: os.getenv("REACTOR_API_URL", "http://localhost:8080"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("REACTOR_API_KEY"))

    # Message queue
    queue_url: str = field(default_factory=lambda: os.getenv("REACTOR_QUEUE_URL", "redis://localhost:6379"))
    queue_name: str = field(default_factory=lambda: os.getenv("REACTOR_QUEUE_NAME", "reactor_jobs"))

    # Polling
    poll_interval_seconds: float = field(default_factory=lambda: float(os.getenv("REACTOR_POLL_INTERVAL", "30.0")))

    # Limits
    max_concurrent_jobs: int = field(default_factory=lambda: int(os.getenv("REACTOR_MAX_JOBS", "3")))
    job_timeout_hours: float = field(default_factory=lambda: float(os.getenv("REACTOR_JOB_TIMEOUT_HOURS", "24.0")))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "communication_method": self.communication_method.value,
            "jobs_dir": str(self.jobs_dir),
            "api_url": self.api_url,
            "max_concurrent_jobs": self.max_concurrent_jobs,
        }


# =============================================================================
# TRAINING DATA EXPORTER
# =============================================================================


class TrainingDataExporter:
    """
    Exports training data from JARVIS Prime for Reactor Core.

    Converts experiences, preferences, and demonstrations into
    formats suitable for fine-tuning.
    """

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def export_experiences(
        self,
        experiences: List[Any],
        format: str = "jsonl",
    ) -> Path:
        """
        Export experiences to training format.

        Args:
            experiences: List of Experience objects
            format: Output format (jsonl, parquet, etc.)

        Returns:
            Path to exported file
        """
        output_path = self._output_dir / f"experiences_{int(time.time())}.{format}"

        if format == "jsonl":
            await self._export_jsonl(experiences, output_path, self._format_experience)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(experiences)} experiences to {output_path}")
        return output_path

    async def export_preferences(
        self,
        preferences: List[Any],
        format: str = "jsonl",
    ) -> Path:
        """Export preferences for RLHF training."""
        output_path = self._output_dir / f"preferences_{int(time.time())}.{format}"

        if format == "jsonl":
            await self._export_jsonl(preferences, output_path, self._format_preference)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(preferences)} preferences to {output_path}")
        return output_path

    async def _export_jsonl(
        self,
        items: List[Any],
        path: Path,
        formatter: Callable[[Any], Dict[str, Any]],
    ) -> None:
        """Export items to JSONL format."""
        with open(path, "w") as f:
            for item in items:
                formatted = formatter(item)
                f.write(json.dumps(formatted) + "\n")

    def _format_experience(self, exp: Any) -> Dict[str, Any]:
        """Format an experience for training."""
        # Handle different experience types
        if hasattr(exp, "to_dict"):
            exp_dict = exp.to_dict()
        elif isinstance(exp, dict):
            exp_dict = exp
        else:
            exp_dict = {"input": str(exp)}

        return {
            "instruction": exp_dict.get("input_text", exp_dict.get("input", "")),
            "output": exp_dict.get("output_text", exp_dict.get("output", "")),
            "metadata": {
                "id": exp_dict.get("id"),
                "timestamp": exp_dict.get("timestamp"),
                "feedback_score": exp_dict.get("feedback_score"),
            },
        }

    def _format_preference(self, pref: Any) -> Dict[str, Any]:
        """Format a preference for RLHF training."""
        if hasattr(pref, "to_dict"):
            pref_dict = pref.to_dict()
        elif isinstance(pref, dict):
            pref_dict = pref
        else:
            return {}

        return {
            "prompt": pref_dict.get("prompt", ""),
            "chosen": pref_dict.get("response_a" if pref_dict.get("preference") == "a" else "response_b", ""),
            "rejected": pref_dict.get("response_b" if pref_dict.get("preference") == "a" else "response_a", ""),
            "metadata": {
                "id": pref_dict.get("id"),
                "preference": pref_dict.get("preference"),
                "confidence": pref_dict.get("confidence"),
            },
        }


# =============================================================================
# JOB MANAGER
# =============================================================================


class JobManager:
    """Manages training jobs submitted to Reactor Core."""

    def __init__(self, config: ReactorCoreConfig):
        self._config = config
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = asyncio.Lock()

        # Ensure directories exist
        config.jobs_dir.mkdir(parents=True, exist_ok=True)

    async def create_job(
        self,
        job_type: JobType,
        base_model: str,
        config: FineTuningConfig,
        training_data_path: str,
    ) -> TrainingJob:
        """Create a new training job."""
        job = TrainingJob(
            type=job_type,
            base_model=base_model,
            config=config,
            training_data_path=training_data_path,
        )

        async with self._lock:
            self._jobs[job.id] = job

            # Write job file
            job_file = self._config.jobs_dir / f"{job.id}.json"
            with open(job_file, "w") as f:
                json.dump(job.to_dict(), f, indent=2)

        logger.info(f"Created training job: {job.id}")
        return job

    async def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        async with self._lock:
            # Check memory first
            if job_id in self._jobs:
                return self._jobs[job_id]

            # Check file system
            job_file = self._config.jobs_dir / f"{job_id}.json"
            if job_file.exists():
                with open(job_file) as f:
                    data = json.load(f)
                job = TrainingJob.from_dict(data)
                self._jobs[job_id] = job
                return job

            return None

    async def update_job(self, job: TrainingJob) -> None:
        """Update a job."""
        async with self._lock:
            self._jobs[job.id] = job

            # Update file
            job_file = self._config.jobs_dir / f"{job.id}.json"
            with open(job_file, "w") as f:
                json.dump(job.to_dict(), f, indent=2)

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
    ) -> List[TrainingJob]:
        """List jobs, optionally filtered by status."""
        jobs = []

        # Load from files
        for job_file in self._config.jobs_dir.glob("*.json"):
            try:
                with open(job_file) as f:
                    data = json.load(f)
                job = TrainingJob.from_dict(data)

                if status is None or job.status == status:
                    jobs.append(job)
            except Exception as e:
                logger.warning(f"Error loading job {job_file}: {e}")

        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        job = await self.get_job(job_id)
        if not job:
            return False

        if job.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.RUNNING):
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            await self.update_job(job)
            return True

        return False


# =============================================================================
# REACTOR CORE BRIDGE
# =============================================================================


class ReactorCoreBridge:
    """
    Bridge for communication with Reactor Core training system.

    Handles:
    - Job submission
    - Status polling
    - Training data export
    - Model deployment triggers
    """

    def __init__(self, config: Optional[ReactorCoreConfig] = None):
        self._config = config or ReactorCoreConfig()

        # Components
        self._job_manager = JobManager(self._config)
        self._data_exporter = TrainingDataExporter(self._config.data_dir)

        # State
        self._is_running = False
        self._poll_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_job_complete: List[Callable[[TrainingJob], None]] = []
        self._on_job_failed: List[Callable[[TrainingJob], None]] = []

        # Statistics
        self._jobs_submitted = 0
        self._jobs_completed = 0
        self._jobs_failed = 0

    async def start(self) -> None:
        """Start the bridge."""
        self._is_running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("ReactorCoreBridge started")

    async def stop(self) -> None:
        """Stop the bridge."""
        self._is_running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        logger.info("ReactorCoreBridge stopped")

    async def submit_finetuning_job(
        self,
        base_model: str,
        experiences: Optional[List[Any]] = None,
        preferences: Optional[List[Any]] = None,
        config: Optional[FineTuningConfig] = None,
    ) -> str:
        """
        Submit a fine-tuning job to Reactor Core.

        Args:
            base_model: Base model to fine-tune
            experiences: Training experiences
            preferences: RLHF preferences
            config: Fine-tuning configuration

        Returns:
            Job ID
        """
        config = config or FineTuningConfig()

        # Export training data
        data_paths = []

        if experiences:
            exp_path = await self._data_exporter.export_experiences(experiences)
            data_paths.append(str(exp_path))

        if preferences:
            pref_path = await self._data_exporter.export_preferences(preferences)
            data_paths.append(str(pref_path))

        # Create job
        job = await self._job_manager.create_job(
            job_type=JobType.FINE_TUNING,
            base_model=base_model,
            config=config,
            training_data_path=",".join(data_paths),
        )

        # Submit to Reactor Core
        await self._submit_job(job)

        self._jobs_submitted += 1
        return job.id

    async def submit_rlhf_job(
        self,
        base_model: str,
        preferences: List[Any],
        config: Optional[FineTuningConfig] = None,
    ) -> str:
        """Submit an RLHF training job."""
        config = config or FineTuningConfig()

        # Export preferences
        pref_path = await self._data_exporter.export_preferences(preferences)

        # Create job
        job = await self._job_manager.create_job(
            job_type=JobType.RLHF,
            base_model=base_model,
            config=config,
            training_data_path=str(pref_path),
        )

        await self._submit_job(job)
        self._jobs_submitted += 1
        return job.id

    async def upload_training_data(
        self,
        data: List[Dict[str, Any]],
        data_type: str = "instructions",
    ) -> bool:
        """
        v242.0: Upload training data to Reactor Core.

        Called by TrainingDataPipeline._sync_to_reactor() to send
        collected and annotated training data.

        Args:
            data: List of training examples (instruction/output dicts)
            data_type: Type of data ("instructions", "preferences", "conversations")

        Returns:
            True if upload succeeded, False otherwise.
        """
        if not data:
            logger.info("[Bridge] No data to upload")
            return True

        try:
            # v242.2: Convert to canonical experience format (repo-local first, fallback)
            _has_schema = False
            try:
                from jarvis_prime.schemas.experience_schema import (
                    ExperienceEvent,
                    ExperienceSource,
                    ExperienceType,
                    SCHEMA_VERSION,
                )
                _has_schema = True
            except ImportError:
                try:
                    import sys as _sys
                    _jarvis_home = str(Path.home() / ".jarvis")
                    if _jarvis_home not in _sys.path:
                        _sys.path.insert(0, _jarvis_home)
                    from schemas.experience_schema import (
                        ExperienceEvent,
                        ExperienceSource,
                        ExperienceType,
                        SCHEMA_VERSION,
                    )
                    _has_schema = True
                except ImportError:
                    pass

            uploaded = 0

            if self._config.communication_method == CommunicationMethod.FILE_SYSTEM:
                # File-based upload: write to shared directory
                data_dir = Path(self._config.data_dir)
                data_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = data_dir / f"{data_type}_{timestamp}.jsonl"

                with open(output_file, "w") as f:
                    for item in data:
                        if _has_schema:
                            event = ExperienceEvent(
                                event_type=ExperienceType.INTERACTION,
                                source=ExperienceSource.JARVIS_PRIME,
                                user_input=item.get("instruction", item.get("user_input", "")),
                                assistant_output=item.get("output", item.get("assistant_output", "")),
                                system_context=item.get("input", item.get("system_context")),
                                confidence=item.get("metadata", {}).get("feedback_score", 1.0),
                                model_id=item.get("metadata", {}).get("model_id"),
                                task_type=item.get("metadata", {}).get("task_type"),
                                metadata=item.get("metadata", {}),
                            )
                            f.write(json.dumps(event.to_reactor_core_format()) + "\n")
                        else:
                            f.write(json.dumps(item) + "\n")
                        uploaded += 1

                logger.info(f"[Bridge] Uploaded {uploaded} {data_type} to {output_file}")

            elif self._config.communication_method == CommunicationMethod.REST_API:
                # v242.1: HTTP upload via batch endpoint (single request for all items)
                # Falls back to per-item POST if batch endpoint unavailable
                import aiohttp

                # Convert all items to canonical format
                batch_items = []
                for item in data:
                    if _has_schema:
                        event = ExperienceEvent(
                            event_type=ExperienceType.INTERACTION,
                            source=ExperienceSource.JARVIS_PRIME,
                            user_input=item.get("instruction", item.get("user_input", "")),
                            assistant_output=item.get("output", item.get("assistant_output", "")),
                            confidence=item.get("metadata", {}).get("feedback_score", 1.0),
                            metadata=item.get("metadata", {}),
                        )
                        batch_items.append(event.to_reactor_core_format())
                    else:
                        batch_items.append(item)

                async with aiohttp.ClientSession() as session:
                    # Try batch endpoint first (v242.1)
                    batch_url = f"{self._config.api_url}/api/v1/experiences/batch"
                    try:
                        payload = {"experiences": batch_items, "source": "jarvis_prime"}
                        async with session.post(batch_url, json=payload) as resp:
                            if resp.status == 200:
                                result = await resp.json()
                                uploaded = result.get("accepted", 0)
                                logger.info(f"[Bridge] Batch uploaded {uploaded}/{len(data)} {data_type}")
                            elif resp.status == 404:
                                # Batch endpoint not available — fall back to per-item
                                logger.debug("[Bridge] Batch endpoint not available, falling back to per-item")
                                url = f"{self._config.api_url}/api/v1/experiences/stream"
                                for exp_item in batch_items:
                                    try:
                                        item_payload = {"experience": exp_item, "source": "jarvis_prime"}
                                        async with session.post(url, json=item_payload) as item_resp:
                                            if item_resp.status == 200:
                                                uploaded += 1
                                    except Exception:
                                        pass
                                logger.info(f"[Bridge] Per-item uploaded {uploaded}/{len(data)} {data_type}")
                            else:
                                text = await resp.text()
                                logger.warning(f"[Bridge] Batch upload failed ({resp.status}): {text[:200]}")
                    except Exception as e:
                        logger.warning(f"[Bridge] Upload request failed: {e}")

            return uploaded > 0

        except Exception as e:
            logger.error(f"[Bridge] upload_training_data failed: {e}")
            return False

    async def _submit_job(self, job: TrainingJob) -> None:
        """Submit job to Reactor Core."""
        if self._config.communication_method == CommunicationMethod.FILE_SYSTEM:
            # Job already written to file by JobManager
            job.status = JobStatus.QUEUED
            await self._job_manager.update_job(job)
            logger.info(f"Job {job.id} queued (file system)")

        elif self._config.communication_method == CommunicationMethod.REST_API:
            # Submit via API
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self._config.api_url}/jobs",
                        json=job.to_dict(),
                        headers={"Authorization": f"Bearer {self._config.api_key}"} if self._config.api_key else {},
                    )
                    response.raise_for_status()
                    job.status = JobStatus.QUEUED
                    await self._job_manager.update_job(job)
            except Exception as e:
                logger.error(f"Failed to submit job via API: {e}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                await self._job_manager.update_job(job)

        elif self._config.communication_method == CommunicationMethod.MESSAGE_QUEUE:
            # Submit via message queue
            try:
                import redis.asyncio as redis
                r = redis.from_url(self._config.queue_url)
                await r.lpush(self._config.queue_name, json.dumps(job.to_dict()))
                job.status = JobStatus.QUEUED
                await self._job_manager.update_job(job)
            except Exception as e:
                logger.error(f"Failed to submit job via queue: {e}")
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                await self._job_manager.update_job(job)

    async def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get current status of a job."""
        return await self._job_manager.get_job(job_id)

    async def get_training_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get training metrics for a job."""
        job = await self._job_manager.get_job(job_id)
        if not job:
            return {}

        return {
            "job_id": job_id,
            "status": job.status.value,
            "progress": job.progress_percent,
            "current_step": job.current_step,
            "total_steps": job.total_steps,
            "metrics": job.metrics,
        }

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        return await self._job_manager.cancel_job(job_id)

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
    ) -> List[TrainingJob]:
        """List all jobs."""
        return await self._job_manager.list_jobs(status)

    def on_job_complete(self, callback: Callable[[TrainingJob], None]) -> None:
        """Register callback for job completion."""
        self._on_job_complete.append(callback)

    def on_job_failed(self, callback: Callable[[TrainingJob], None]) -> None:
        """Register callback for job failure."""
        self._on_job_failed.append(callback)

    async def _poll_loop(self) -> None:
        """Background polling for job updates."""
        while self._is_running:
            try:
                await asyncio.sleep(self._config.poll_interval_seconds)

                # Poll for updates
                await self._check_job_updates()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll loop error: {e}")

    async def _check_job_updates(self) -> None:
        """Check for job status updates."""
        # Get all active jobs
        active_jobs = await self._job_manager.list_jobs(JobStatus.RUNNING)
        active_jobs.extend(await self._job_manager.list_jobs(JobStatus.QUEUED))

        for job in active_jobs:
            try:
                updated = await self._fetch_job_update(job)
                if updated:
                    await self._job_manager.update_job(updated)

                    if updated.status == JobStatus.COMPLETED:
                        self._jobs_completed += 1
                        for callback in self._on_job_complete:
                            try:
                                callback(updated)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

                    elif updated.status == JobStatus.FAILED:
                        self._jobs_failed += 1
                        for callback in self._on_job_failed:
                            try:
                                callback(updated)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.warning(f"Error checking job {job.id}: {e}")

    async def _fetch_job_update(self, job: TrainingJob) -> Optional[TrainingJob]:
        """Fetch job update from Reactor Core."""
        if self._config.communication_method == CommunicationMethod.FILE_SYSTEM:
            # Check for status file from Reactor Core
            status_file = self._config.jobs_dir / f"{job.id}_status.json"
            if status_file.exists():
                with open(status_file) as f:
                    data = json.load(f)
                job.status = JobStatus(data.get("status", job.status.value))
                job.current_step = data.get("current_step", job.current_step)
                job.total_steps = data.get("total_steps", job.total_steps)
                job.progress_percent = data.get("progress_percent", job.progress_percent)
                job.metrics = data.get("metrics", job.metrics)
                job.output_model_path = data.get("output_model_path")
                job.error_message = data.get("error_message")
                return job

        elif self._config.communication_method == CommunicationMethod.REST_API:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self._config.api_url}/jobs/{job.id}",
                        headers={"Authorization": f"Bearer {self._config.api_key}"} if self._config.api_key else {},
                    )
                    response.raise_for_status()
                    data = response.json()
                    return TrainingJob.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to fetch job update: {e}")

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "jobs_submitted": self._jobs_submitted,
            "jobs_completed": self._jobs_completed,
            "jobs_failed": self._jobs_failed,
            "is_running": self._is_running,
            "config": self._config.to_dict(),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_bridge: Optional[ReactorCoreBridge] = None
_bridge_lock = asyncio.Lock()


async def get_reactor_core_bridge(
    config: Optional[ReactorCoreConfig] = None,
) -> ReactorCoreBridge:
    """Get or create the global Reactor Core bridge."""
    global _bridge

    async with _bridge_lock:
        if _bridge is None:
            _bridge = ReactorCoreBridge(config)
            await _bridge.start()

        return _bridge


async def shutdown_reactor_core_bridge() -> None:
    """Shutdown the global bridge."""
    global _bridge

    async with _bridge_lock:
        if _bridge:
            await _bridge.stop()
            _bridge = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "JobStatus",
    "JobType",
    "CommunicationMethod",
    # Data structures
    "FineTuningConfig",
    "TrainingData",
    "TrainingJob",
    "ReactorCoreConfig",
    # Components
    "TrainingDataExporter",
    "JobManager",
    # Bridge
    "ReactorCoreBridge",
    "get_reactor_core_bridge",
    "shutdown_reactor_core_bridge",
]
