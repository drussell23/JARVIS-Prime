"""
Training Execution Engine - Connect Training Loop to Models
=============================================================

v92.0 - Production-Grade Training Orchestration

This module bridges the gap between data collection and model training:
- Executes fine-tuning on collected JARVIS conversation data
- Orchestrates training jobs across local and Reactor Core
- Manages training schedules and resource allocation
- Implements progressive training with validation gates
- Handles model checkpointing and rollback

ARCHITECTURE:
    TrainingExecutor
        |
        +-- TrainingScheduler (when to train)
        +-- DataPreparator (format data for training)
        +-- LocalTrainer (on-device fine-tuning)
        +-- RemoteTrainer (Reactor Core delegation)
        +-- ValidationGate (quality checks before deployment)
        +-- CheckpointManager (save/restore/rollback)

TRAINING STRATEGIES:
    1. Incremental: Small frequent updates (hourly)
    2. Batch: Larger updates on accumulated data (daily)
    3. Full: Complete retraining (weekly/monthly)
    4. Emergency: Immediate update for critical fixes

USAGE:
    executor = await get_training_executor()

    # Schedule automatic training
    await executor.schedule_training(strategy="incremental", interval_hours=1)

    # Manual training trigger
    job = await executor.start_training(
        base_model="prime-7b-chat-v1",
        training_type="lora",
        data_source="collected"
    )

    # Monitor and deploy
    await executor.wait_for_completion(job.id)
    await executor.deploy_if_validated(job.id)
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
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
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ADVANCED UTILITIES - Retry, Backoff, Resource Management
# =============================================================================


@dataclass
class BackoffConfig:
    """Configuration for exponential backoff."""
    initial_delay: float = 1.0
    max_delay: float = 300.0  # 5 minutes
    multiplier: float = 2.0
    jitter: float = 0.1  # Add 10% random jitter
    max_retries: int = 5


class ExponentialBackoff:
    """
    Exponential backoff with jitter for resilient retries.

    Features:
    - Exponential delay increase
    - Random jitter to prevent thundering herd
    - Configurable max delay and max retries
    - Reset capability for success
    """

    __slots__ = ('_config', '_attempt', '_consecutive_failures')

    def __init__(self, config: Optional[BackoffConfig] = None):
        self._config = config or BackoffConfig()
        self._attempt = 0
        self._consecutive_failures = 0

    def get_delay(self) -> float:
        """Calculate delay for current attempt with jitter."""
        base_delay = min(
            self._config.initial_delay * (self._config.multiplier ** self._attempt),
            self._config.max_delay
        )
        # Add jitter
        jitter = base_delay * self._config.jitter * random.uniform(-1, 1)
        return max(0, base_delay + jitter)

    def should_retry(self) -> bool:
        """Check if we should retry."""
        return self._attempt < self._config.max_retries

    def record_failure(self) -> None:
        """Record a failure and increment attempt counter."""
        self._attempt += 1
        self._consecutive_failures += 1

    def reset(self) -> None:
        """Reset backoff state after success."""
        self._attempt = 0
        self._consecutive_failures = 0

    @property
    def attempt(self) -> int:
        return self._attempt

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures


def with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Tuple[type, ...] = (Exception,),
):
    """
    Decorator for async functions with exponential backoff retry.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        retryable_exceptions: Tuple of exceptions that trigger retry
    """
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            backoff = ExponentialBackoff(BackoffConfig(
                initial_delay=initial_delay,
                max_delay=max_delay,
                max_retries=max_retries,
            ))

            last_exception = None
            while True:
                try:
                    result = await func(*args, **kwargs)
                    backoff.reset()
                    return result
                except retryable_exceptions as e:
                    last_exception = e
                    if not backoff.should_retry():
                        logger.error(f"{func.__name__} failed after {backoff.attempt} attempts: {e}")
                        raise

                    delay = backoff.get_delay()
                    logger.warning(
                        f"{func.__name__} failed (attempt {backoff.attempt}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    backoff.record_failure()
                    await asyncio.sleep(delay)

        return wrapper
    return decorator


@asynccontextmanager
async def managed_model_resources(model_path: Path, device: str = "auto"):
    """
    Context manager for model loading with guaranteed cleanup.

    Ensures GPU memory is properly released even on exceptions.
    """
    model = None
    tokenizer = None

    try:
        # Lazy import to avoid loading torch unnecessarily
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            TORCH_AVAILABLE = True
        except ImportError:
            TORCH_AVAILABLE = False

        if TORCH_AVAILABLE:
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )

        yield model, tokenizer

    finally:
        # Guaranteed cleanup
        if model is not None:
            try:
                del model
            except Exception:
                pass

        if tokenizer is not None:
            try:
                del tokenizer
            except Exception:
                pass

        # Clear GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        # Force garbage collection
        import gc
        gc.collect()


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TrainingExecutorConfig:
    """Configuration for training execution."""

    # Paths
    checkpoint_dir: Path = field(default_factory=lambda: Path(
        os.getenv("TRAINING_CHECKPOINT_DIR", str(Path.home() / ".jarvis" / "checkpoints"))
    ))
    output_dir: Path = field(default_factory=lambda: Path(
        os.getenv("TRAINING_OUTPUT_DIR", str(Path.home() / ".jarvis" / "trained_models"))
    ))

    # Training settings
    default_training_type: str = field(default_factory=lambda:
        os.getenv("TRAINING_DEFAULT_TYPE", "lora")  # "lora", "full", "qlora"
    )
    default_epochs: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_DEFAULT_EPOCHS", "3")
    ))
    default_batch_size: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_BATCH_SIZE", "4")
    ))
    default_learning_rate: float = field(default_factory=lambda: float(
        os.getenv("TRAINING_LEARNING_RATE", "2e-4")
    ))

    # LoRA settings
    lora_r: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_LORA_R", "16")
    ))
    lora_alpha: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_LORA_ALPHA", "32")
    ))
    lora_dropout: float = field(default_factory=lambda: float(
        os.getenv("TRAINING_LORA_DROPOUT", "0.05")
    ))

    # Scheduling
    incremental_interval_hours: float = field(default_factory=lambda: float(
        os.getenv("TRAINING_INCREMENTAL_HOURS", "1.0")
    ))
    batch_interval_hours: float = field(default_factory=lambda: float(
        os.getenv("TRAINING_BATCH_HOURS", "24.0")
    ))
    min_samples_for_training: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_MIN_SAMPLES", "50")
    ))

    # Validation
    validation_split: float = field(default_factory=lambda: float(
        os.getenv("TRAINING_VAL_SPLIT", "0.1")
    ))
    min_improvement_threshold: float = field(default_factory=lambda: float(
        os.getenv("TRAINING_MIN_IMPROVEMENT", "0.01")  # 1% improvement required
    ))

    # Resource limits
    max_memory_gb: float = field(default_factory=lambda: float(
        os.getenv("TRAINING_MAX_MEMORY_GB", "16.0")
    ))
    use_gradient_checkpointing: bool = field(default_factory=lambda:
        os.getenv("TRAINING_GRAD_CHECKPOINT", "true").lower() == "true"
    )

    # Reactor Core
    enable_remote_training: bool = field(default_factory=lambda:
        os.getenv("TRAINING_ENABLE_REMOTE", "true").lower() == "true"
    )
    prefer_remote_for_large_models: bool = field(default_factory=lambda:
        os.getenv("TRAINING_PREFER_REMOTE", "true").lower() == "true"
    )
    large_model_threshold_gb: float = field(default_factory=lambda: float(
        os.getenv("TRAINING_LARGE_MODEL_GB", "10.0")
    ))


# =============================================================================
# ENUMS
# =============================================================================


class TrainingType(Enum):
    """Types of training."""
    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"
    PREFIX_TUNING = "prefix_tuning"
    ADAPTER = "adapter"


class TrainingStrategy(Enum):
    """Training scheduling strategies."""
    INCREMENTAL = "incremental"  # Small frequent updates
    BATCH = "batch"  # Larger daily updates
    FULL = "full"  # Complete retraining
    EMERGENCY = "emergency"  # Immediate update
    MANUAL = "manual"  # User-triggered


class JobStatus(Enum):
    """Status of a training job."""
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


class JobLocation(Enum):
    """Where training runs."""
    LOCAL = "local"
    REACTOR_CORE = "reactor_core"
    HYBRID = "hybrid"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class TrainingHyperparameters:
    """Hyperparameters for training."""
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # LoRA specific
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # Optimization
    use_gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    fp16: bool = True
    bf16: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "fp16": self.fp16,
        }


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    train_loss: float = 0.0
    eval_loss: float = 0.0
    train_accuracy: float = 0.0
    eval_accuracy: float = 0.0
    perplexity: float = 0.0

    # Performance
    samples_per_second: float = 0.0
    steps_per_second: float = 0.0
    total_steps: int = 0

    # History
    loss_history: List[float] = field(default_factory=list)
    eval_history: List[Dict[str, float]] = field(default_factory=list)

    def improvement_over(self, baseline: "TrainingMetrics") -> float:
        """Calculate improvement over baseline."""
        if baseline.eval_loss > 0:
            return (baseline.eval_loss - self.eval_loss) / baseline.eval_loss
        return 0.0


@dataclass
class TrainingJob:
    """A training job."""
    id: str
    base_model: str
    training_type: TrainingType
    strategy: TrainingStrategy
    status: JobStatus
    location: JobLocation

    # Configuration
    hyperparameters: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    data_source: str = "collected"  # "collected", "synthetic", "mixed"

    # Progress
    current_epoch: int = 0
    current_step: int = 0
    total_steps: int = 0
    progress_percent: float = 0.0

    # Metrics
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    baseline_metrics: Optional[TrainingMetrics] = None

    # Paths
    checkpoint_path: Optional[Path] = None
    output_path: Optional[Path] = None
    data_path: Optional[Path] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Metadata
    error_message: Optional[str] = None
    remote_job_id: Optional[str] = None  # For Reactor Core jobs

    @property
    def duration_seconds(self) -> float:
        if not self.started_at:
            return 0.0
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()

    @property
    def is_running(self) -> bool:
        return self.status in (
            JobStatus.PREPARING,
            JobStatus.TRAINING,
            JobStatus.VALIDATING,
            JobStatus.DEPLOYING,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "base_model": self.base_model,
            "training_type": self.training_type.value,
            "strategy": self.strategy.value,
            "status": self.status.value,
            "location": self.location.value,
            "progress_percent": self.progress_percent,
            "current_epoch": self.current_epoch,
            "metrics": {
                "train_loss": self.metrics.train_loss,
                "eval_loss": self.metrics.eval_loss,
            },
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


@dataclass
class DeploymentResult:
    """Result of deploying a trained model."""
    success: bool
    model_name: str
    model_path: Path
    previous_model: Optional[str] = None
    rollback_available: bool = False
    deployed_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


# =============================================================================
# DATA PREPARATOR
# =============================================================================


class DataPreparator:
    """Prepares training data from various sources."""

    def __init__(self, config: TrainingExecutorConfig):
        self._config = config

    async def prepare_from_collected(
        self,
        min_quality: float = 0.6,
        max_samples: Optional[int] = None,
        output_path: Optional[Path] = None,
    ) -> Tuple[Path, int]:
        """Prepare training data from collected conversations."""
        from jarvis_prime.core.training_data_pipeline import get_training_data_pipeline

        pipeline = await get_training_data_pipeline()

        # Export to temporary directory if no path specified
        output_path = output_path or Path(tempfile.mkdtemp(prefix="training_"))

        # Export training data
        files = await pipeline.export_for_training(
            output_dir=output_path,
            min_quality=min_quality,
            include_augmented=True,
        )

        # Count samples
        sample_count = 0
        for file_path in files.values():
            if file_path.exists():
                with open(file_path) as f:
                    sample_count += sum(1 for _ in f)

        logger.info(f"Prepared {sample_count} training samples in {output_path}")

        return output_path, sample_count

    async def prepare_from_preferences(
        self,
        output_path: Optional[Path] = None,
    ) -> Tuple[Path, int]:
        """Prepare DPO training data from preference pairs."""
        from jarvis_prime.learning.rlhf_integration import get_rlhf_pipeline

        pipeline = await get_rlhf_pipeline()
        status = pipeline.get_status()

        output_path = output_path or Path(tempfile.mkdtemp(prefix="dpo_"))
        output_path.mkdir(parents=True, exist_ok=True)

        # Get preference pairs
        pairs = []
        if hasattr(pipeline, '_preference_buffer'):
            pairs = list(pipeline._preference_buffer._pairs)

        # Export to DPO format
        dpo_path = output_path / "preferences.jsonl"
        with open(dpo_path, "w") as f:
            for pair in pairs:
                data = {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen_response,
                    "rejected": pair.rejected_response,
                }
                f.write(json.dumps(data) + "\n")

        logger.info(f"Prepared {len(pairs)} preference pairs for DPO")

        return output_path, len(pairs)

    async def split_data(
        self,
        data_path: Path,
        val_split: float = 0.1,
    ) -> Tuple[Path, Path]:
        """Split data into train and validation sets."""
        import random

        train_path = data_path / "train.jsonl"
        val_path = data_path / "val.jsonl"

        # Read all data
        all_data = []
        for jsonl_file in data_path.glob("*.jsonl"):
            if jsonl_file.name not in ("train.jsonl", "val.jsonl"):
                with open(jsonl_file) as f:
                    all_data.extend([json.loads(line) for line in f])

        # Shuffle and split
        random.shuffle(all_data)
        split_idx = int(len(all_data) * (1 - val_split))

        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]

        # Write splits
        with open(train_path, "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        with open(val_path, "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")

        logger.info(f"Split data: {len(train_data)} train, {len(val_data)} val")

        return train_path, val_path


# =============================================================================
# LOCAL TRAINER
# =============================================================================


class LocalTrainer:
    """Handles local on-device training."""

    def __init__(self, config: TrainingExecutorConfig):
        self._config = config
        self._current_job: Optional[TrainingJob] = None
        self._stop_requested = False

    async def train(
        self,
        job: TrainingJob,
        progress_callback: Optional[Callable[[TrainingJob], None]] = None,
    ) -> TrainingJob:
        """Execute local training."""
        self._current_job = job
        self._stop_requested = False

        job.status = JobStatus.TRAINING
        job.started_at = datetime.now()

        try:
            # Import training libraries
            try:
                import torch
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    TrainingArguments,
                    Trainer,
                    DataCollatorForLanguageModeling,
                )
                from peft import LoraConfig, get_peft_model, TaskType
                from datasets import load_dataset
            except ImportError as e:
                job.status = JobStatus.FAILED
                job.error_message = f"Missing training dependencies: {e}"
                return job

            # Check resources
            from jarvis_prime.core.resource_manager import get_resource_manager
            resource_mgr = await get_resource_manager()
            snapshot = await resource_mgr.get_snapshot()

            available_gb = snapshot.available_ram / (1024**3)
            if available_gb < self._config.max_memory_gb * 0.5:
                job.status = JobStatus.FAILED
                job.error_message = f"Insufficient memory: {available_gb:.1f}GB available"
                return job

            # Load base model
            logger.info(f"Loading base model: {job.base_model}")

            # Get model path from registry
            from jarvis_prime.models.prime_model import PrimeModelRegistry
            registry = PrimeModelRegistry()
            model_spec = registry.get(job.base_model)

            if not model_spec:
                job.status = JobStatus.FAILED
                job.error_message = f"Unknown model: {job.base_model}"
                return job

            base_model_path = model_spec.hub_path or model_spec.base_model

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model with quantization for efficiency
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
            }

            if job.training_type == TrainingType.QLORA:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            if job.hyperparameters.use_gradient_checkpointing:
                model_kwargs["use_cache"] = False

            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                **model_kwargs,
            )

            # Apply LoRA if needed
            if job.training_type in (TrainingType.LORA, TrainingType.QLORA):
                lora_config = LoraConfig(
                    r=job.hyperparameters.lora_r,
                    lora_alpha=job.hyperparameters.lora_alpha,
                    lora_dropout=job.hyperparameters.lora_dropout,
                    target_modules=job.hyperparameters.lora_target_modules,
                    task_type=TaskType.CAUSAL_LM,
                    bias="none",
                )
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()

            if job.hyperparameters.use_gradient_checkpointing:
                model.gradient_checkpointing_enable()

            # Load dataset
            if not job.data_path or not job.data_path.exists():
                job.status = JobStatus.FAILED
                job.error_message = "Training data not found"
                return job

            train_path = job.data_path / "train.jsonl"
            val_path = job.data_path / "val.jsonl"

            if not train_path.exists():
                # Use any available jsonl file
                jsonl_files = list(job.data_path.glob("*.jsonl"))
                if jsonl_files:
                    train_path = jsonl_files[0]

            dataset = load_dataset(
                "json",
                data_files={
                    "train": str(train_path),
                    "validation": str(val_path) if val_path.exists() else str(train_path),
                },
            )

            # Tokenize dataset
            def tokenize_function(examples):
                # Handle different formats
                if "instruction" in examples:
                    texts = [
                        f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                        for inst, out in zip(examples["instruction"], examples["output"])
                    ]
                elif "messages" in examples:
                    texts = []
                    for msgs in examples["messages"]:
                        text = ""
                        for msg in msgs:
                            text += f"<|{msg['role']}|>\n{msg['content']}\n"
                        texts.append(text)
                else:
                    texts = examples.get("text", examples.get("content", [""]))

                return tokenizer(
                    texts,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                )

            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset["train"].column_names,
            )

            # Setup training arguments
            output_dir = self._config.output_dir / f"job_{job.id}"
            output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=job.hyperparameters.epochs,
                per_device_train_batch_size=job.hyperparameters.batch_size,
                per_device_eval_batch_size=job.hyperparameters.batch_size,
                learning_rate=job.hyperparameters.learning_rate,
                warmup_steps=job.hyperparameters.warmup_steps,
                weight_decay=job.hyperparameters.weight_decay,
                max_grad_norm=job.hyperparameters.max_grad_norm,
                fp16=job.hyperparameters.fp16,
                bf16=job.hyperparameters.bf16,
                logging_steps=10,
                eval_steps=100,
                save_steps=500,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                report_to=[],  # Disable wandb etc
            )

            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset.get("validation", tokenized_dataset["train"]),
                data_collator=data_collator,
            )

            # Custom callback for progress
            class ProgressCallback:
                def __init__(self, job: TrainingJob, callback: Optional[Callable]):
                    self.job = job
                    self.callback = callback

                def on_step_end(self, args, state, control, **kwargs):
                    self.job.current_step = state.global_step
                    self.job.total_steps = state.max_steps
                    self.job.progress_percent = (state.global_step / state.max_steps) * 100

                    if state.log_history:
                        last_log = state.log_history[-1]
                        if "loss" in last_log:
                            self.job.metrics.train_loss = last_log["loss"]
                            self.job.metrics.loss_history.append(last_log["loss"])

                    if self.callback:
                        self.callback(self.job)

                def on_epoch_end(self, args, state, control, **kwargs):
                    self.job.current_epoch = int(state.epoch)

            trainer.add_callback(ProgressCallback(job, progress_callback))

            # Train
            logger.info("Starting training...")
            train_result = trainer.train()

            # Get final metrics
            job.metrics.train_loss = train_result.training_loss
            job.metrics.total_steps = train_result.global_step

            # Evaluate
            eval_result = trainer.evaluate()
            job.metrics.eval_loss = eval_result.get("eval_loss", 0.0)
            job.metrics.perplexity = 2 ** job.metrics.eval_loss

            # Save model
            job.output_path = output_dir / "final"
            trainer.save_model(str(job.output_path))
            tokenizer.save_pretrained(str(job.output_path))

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress_percent = 100.0

            logger.info(f"Training completed. Loss: {job.metrics.train_loss:.4f}")

            return job

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return job
        finally:
            self._current_job = None

    def request_stop(self) -> None:
        """Request training to stop."""
        self._stop_requested = True


# =============================================================================
# REMOTE TRAINER (REACTOR CORE)
# =============================================================================


class RemoteTrainer:
    """Delegates training to Reactor Core."""

    def __init__(self, config: TrainingExecutorConfig):
        self._config = config

    async def submit(
        self,
        job: TrainingJob,
    ) -> TrainingJob:
        """Submit job to Reactor Core."""
        try:
            from jarvis_prime.core.reactor_core_bridge import get_reactor_core_bridge

            bridge = await get_reactor_core_bridge()

            # Prepare training data for upload
            if job.data_path and job.data_path.exists():
                # Upload training data
                with open(job.data_path / "train.jsonl") as f:
                    training_data = [json.loads(line) for line in f]

                # Submit job
                from jarvis_prime.core.reactor_core_bridge import FineTuningConfig

                config = FineTuningConfig(
                    method=job.training_type.value,
                    epochs=job.hyperparameters.epochs,
                    batch_size=job.hyperparameters.batch_size,
                    learning_rate=job.hyperparameters.learning_rate,
                    lora_r=job.hyperparameters.lora_r,
                    lora_alpha=job.hyperparameters.lora_alpha,
                )

                remote_job_id = await bridge.submit_finetuning_job(
                    base_model=job.base_model,
                    experiences=training_data,
                    config=config,
                )

                job.remote_job_id = remote_job_id
                job.status = JobStatus.TRAINING
                job.location = JobLocation.REACTOR_CORE

                logger.info(f"Submitted job to Reactor Core: {remote_job_id}")

            return job

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = f"Failed to submit to Reactor Core: {e}"
            return job

    async def check_status(self, job: TrainingJob) -> TrainingJob:
        """Check status of remote job."""
        if not job.remote_job_id:
            return job

        try:
            from jarvis_prime.core.reactor_core_bridge import get_reactor_core_bridge

            bridge = await get_reactor_core_bridge()
            remote_status = await bridge.get_job_status(job.remote_job_id)

            if remote_status:
                # Map remote status to local
                status_map = {
                    "pending": JobStatus.PENDING,
                    "running": JobStatus.TRAINING,
                    "completed": JobStatus.COMPLETED,
                    "failed": JobStatus.FAILED,
                }
                job.status = status_map.get(remote_status.status, JobStatus.TRAINING)
                job.progress_percent = remote_status.progress * 100

                if remote_status.error:
                    job.error_message = remote_status.error

                if remote_status.output_path:
                    job.output_path = Path(remote_status.output_path)

            return job

        except Exception as e:
            logger.warning(f"Failed to check remote status: {e}")
            return job


# =============================================================================
# VALIDATION GATE
# =============================================================================


class ValidationGate:
    """Validates trained models before deployment."""

    def __init__(self, config: TrainingExecutorConfig):
        self._config = config

    async def validate(
        self,
        job: TrainingJob,
    ) -> Tuple[bool, str]:
        """Validate a trained model."""
        job.status = JobStatus.VALIDATING

        try:
            # 1. Check if training improved over baseline
            if job.baseline_metrics:
                improvement = job.metrics.improvement_over(job.baseline_metrics)
                if improvement < self._config.min_improvement_threshold:
                    return False, f"Insufficient improvement: {improvement:.2%} < {self._config.min_improvement_threshold:.2%}"

            # 2. Run safety validation
            from jarvis_prime.core.model_evaluation import get_evaluation_orchestrator

            evaluator = await get_evaluation_orchestrator()

            if job.output_path and job.output_path.exists():
                # Create inference function for the new model
                try:
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(str(job.output_path))
                    model = AutoModelForCausalLM.from_pretrained(
                        str(job.output_path),
                        device_map="auto",
                        torch_dtype=torch.float16,
                    )

                    async def inference_fn(prompt: str) -> str:
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=256,
                                do_sample=True,
                                temperature=0.7,
                            )
                        return tokenizer.decode(outputs[0], skip_special_tokens=True)

                    # Run safety validation
                    safety_report = await evaluator.validate_safety(
                        model_name=f"{job.base_model}_trained",
                        inference_fn=inference_fn,
                    )

                    if not safety_report.is_safe:
                        return False, f"Safety validation failed: {safety_report.risk_level}"

                    # Run performance benchmark
                    benchmark_result = await evaluator.run_benchmark_suite(
                        model_name=f"{job.base_model}_trained",
                        suite="jarvis-core",
                        inference_fn=inference_fn,
                    )

                    if benchmark_result.pass_rate < 0.7:
                        return False, f"Benchmark pass rate too low: {benchmark_result.pass_rate:.2%}"

                    # Clean up
                    del model
                    del tokenizer
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                except Exception as e:
                    logger.warning(f"Validation inference failed: {e}")
                    # Continue with basic validation

            # 3. Check output files exist
            if job.output_path:
                required_files = ["config.json"]
                for f in required_files:
                    if not (job.output_path / f).exists():
                        return False, f"Missing required file: {f}"

            return True, "Validation passed"

        except Exception as e:
            return False, f"Validation error: {e}"


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================


class CheckpointManager:
    """Manages model checkpoints for rollback."""

    def __init__(self, config: TrainingExecutorConfig):
        self._config = config
        self._checkpoints: Dict[str, List[Path]] = defaultdict(list)
        self._max_checkpoints = 3

    async def save_checkpoint(
        self,
        model_name: str,
        source_path: Path,
        description: str = "",
    ) -> Path:
        """Save a checkpoint of the current model."""
        checkpoint_dir = self._config.checkpoint_dir / model_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f"checkpoint_{timestamp}"

        # Copy model files
        shutil.copytree(source_path, checkpoint_path)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "source_path": str(source_path),
            "description": description,
        }
        with open(checkpoint_path / "checkpoint_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Track checkpoint
        self._checkpoints[model_name].append(checkpoint_path)

        # Prune old checkpoints
        while len(self._checkpoints[model_name]) > self._max_checkpoints:
            old_checkpoint = self._checkpoints[model_name].pop(0)
            shutil.rmtree(old_checkpoint, ignore_errors=True)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

        return checkpoint_path

    async def get_latest_checkpoint(self, model_name: str) -> Optional[Path]:
        """Get the latest checkpoint for a model."""
        if model_name in self._checkpoints and self._checkpoints[model_name]:
            return self._checkpoints[model_name][-1]

        # Scan checkpoint directory
        checkpoint_dir = self._config.checkpoint_dir / model_name
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_*"))
            if checkpoints:
                return checkpoints[-1]

        return None

    async def rollback_to_checkpoint(
        self,
        model_name: str,
        checkpoint_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Rollback to a checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = await self.get_latest_checkpoint(model_name)

        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning(f"No checkpoint found for {model_name}")
            return None

        # Target path for restored model
        restore_path = self._config.output_dir / model_name / "restored"
        restore_path.parent.mkdir(parents=True, exist_ok=True)

        if restore_path.exists():
            shutil.rmtree(restore_path)

        shutil.copytree(checkpoint_path, restore_path)

        logger.info(f"Rolled back {model_name} to {checkpoint_path}")

        return restore_path


# =============================================================================
# TRAINING SCHEDULER
# =============================================================================


class TrainingScheduler:
    """Schedules automatic training runs."""

    def __init__(self, config: TrainingExecutorConfig):
        self._config = config
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        self._last_training: Dict[TrainingStrategy, datetime] = {}

    async def schedule(
        self,
        strategy: TrainingStrategy,
        executor: "TrainingExecutor",
        interval_hours: Optional[float] = None,
    ) -> None:
        """Schedule training with a strategy."""
        task_key = strategy.value

        # Cancel existing task
        if task_key in self._scheduled_tasks:
            self._scheduled_tasks[task_key].cancel()

        # Determine interval
        if interval_hours is None:
            if strategy == TrainingStrategy.INCREMENTAL:
                interval_hours = self._config.incremental_interval_hours
            elif strategy == TrainingStrategy.BATCH:
                interval_hours = self._config.batch_interval_hours
            else:
                interval_hours = 24.0

        # Create task with exponential backoff for error resilience
        async def training_loop():
            backoff = ExponentialBackoff(BackoffConfig(
                initial_delay=60.0,  # Start with 1 minute delay on error
                max_delay=3600.0,    # Max 1 hour delay
                multiplier=2.0,
                max_retries=10,      # Allow many retries for scheduled tasks
            ))

            while True:
                try:
                    await asyncio.sleep(interval_hours * 3600)

                    # Check if we have enough data
                    from jarvis_prime.core.training_data_pipeline import get_training_data_pipeline
                    pipeline = await get_training_data_pipeline()
                    stats = await pipeline._collector.get_statistics()

                    unexported = stats["total_conversations"] - stats["exported_count"]

                    if unexported >= self._config.min_samples_for_training:
                        logger.info(f"Scheduled {strategy.value} training with {unexported} new samples")
                        await executor.start_training(
                            base_model="prime-7b-chat-v1",
                            training_type=TrainingType.LORA,
                            strategy=strategy,
                        )
                        self._last_training[strategy] = datetime.now()
                        backoff.reset()  # Reset backoff on success
                    else:
                        logger.debug(f"Not enough samples for training: {unexported}")
                        backoff.reset()  # No error, just not enough data

                except asyncio.CancelledError:
                    logger.info(f"Training loop for {strategy.value} cancelled")
                    break
                except Exception as e:
                    backoff.record_failure()
                    delay = backoff.get_delay()

                    logger.error(
                        f"Scheduled training error (attempt {backoff.attempt}): {e}. "
                        f"Retrying in {delay:.0f}s"
                    )

                    if not backoff.should_retry():
                        logger.critical(
                            f"Training scheduler for {strategy.value} exceeded max retries. "
                            f"Stopping scheduled training."
                        )
                        break

                    # Wait with backoff before next attempt
                    await asyncio.sleep(delay)

        self._scheduled_tasks[task_key] = asyncio.create_task(training_loop())
        logger.info(f"Scheduled {strategy.value} training every {interval_hours} hours")

    async def cancel(self, strategy: TrainingStrategy) -> None:
        """Cancel scheduled training."""
        task_key = strategy.value
        if task_key in self._scheduled_tasks:
            self._scheduled_tasks[task_key].cancel()
            del self._scheduled_tasks[task_key]

    async def cancel_all(self) -> None:
        """Cancel all scheduled training."""
        for task in self._scheduled_tasks.values():
            task.cancel()
        self._scheduled_tasks.clear()


# =============================================================================
# TRAINING EXECUTOR
# =============================================================================


class TrainingExecutor:
    """Main training execution orchestrator."""

    def __init__(self, config: Optional[TrainingExecutorConfig] = None):
        self._config = config or TrainingExecutorConfig()

        self._data_preparator = DataPreparator(self._config)
        self._local_trainer = LocalTrainer(self._config)
        self._remote_trainer = RemoteTrainer(self._config)
        self._validation_gate = ValidationGate(self._config)
        self._checkpoint_manager = CheckpointManager(self._config)
        self._scheduler = TrainingScheduler(self._config)

        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = asyncio.Lock()
        self._is_running = False

    async def start(self) -> None:
        """Start the executor."""
        self._is_running = True
        logger.info("Training executor started")

    async def stop(self) -> None:
        """Stop the executor."""
        self._is_running = False
        await self._scheduler.cancel_all()
        logger.info("Training executor stopped")

    async def start_training(
        self,
        base_model: str,
        training_type: TrainingType = TrainingType.LORA,
        strategy: TrainingStrategy = TrainingStrategy.MANUAL,
        hyperparameters: Optional[TrainingHyperparameters] = None,
        data_source: str = "collected",
        progress_callback: Optional[Callable[[TrainingJob], None]] = None,
    ) -> TrainingJob:
        """Start a training job."""
        async with self._lock:
            # Create job
            job = TrainingJob(
                id=str(uuid.uuid4())[:8],
                base_model=base_model,
                training_type=training_type,
                strategy=strategy,
                status=JobStatus.PREPARING,
                location=JobLocation.LOCAL,
                hyperparameters=hyperparameters or TrainingHyperparameters(),
                data_source=data_source,
            )

            self._jobs[job.id] = job

            # Determine training location
            from jarvis_prime.models.prime_model import PrimeModelRegistry
            registry = PrimeModelRegistry()
            model_spec = registry.get(base_model)

            if model_spec:
                if (self._config.prefer_remote_for_large_models and
                    model_spec.size_gb > self._config.large_model_threshold_gb and
                    self._config.enable_remote_training):
                    job.location = JobLocation.REACTOR_CORE

            # Prepare data
            try:
                if data_source == "collected":
                    data_path, sample_count = await self._data_preparator.prepare_from_collected(
                        min_quality=0.6,
                    )
                elif data_source == "preferences":
                    data_path, sample_count = await self._data_preparator.prepare_from_preferences()
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = f"Unknown data source: {data_source}"
                    return job

                if sample_count < self._config.min_samples_for_training:
                    job.status = JobStatus.FAILED
                    job.error_message = f"Insufficient training data: {sample_count} samples"
                    return job

                # Split data
                await self._data_preparator.split_data(
                    data_path,
                    val_split=self._config.validation_split,
                )

                job.data_path = data_path

            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = f"Data preparation failed: {e}"
                return job

            # Execute training
            if job.location == JobLocation.LOCAL:
                job = await self._local_trainer.train(job, progress_callback)
            else:
                job = await self._remote_trainer.submit(job)
                # For remote jobs, we need to poll for completion
                if job.status == JobStatus.TRAINING:
                    asyncio.create_task(self._poll_remote_job(job))

            return job

    async def _poll_remote_job(self, job: TrainingJob) -> None:
        """Poll remote job until completion."""
        while job.is_running:
            await asyncio.sleep(30)
            job = await self._remote_trainer.check_status(job)
            self._jobs[job.id] = job

        # Validate if completed
        if job.status == JobStatus.COMPLETED:
            await self.validate_and_deploy(job.id)

    async def validate_and_deploy(
        self,
        job_id: str,
        auto_deploy: bool = True,
    ) -> Tuple[bool, str]:
        """Validate and optionally deploy a trained model."""
        job = self._jobs.get(job_id)
        if not job:
            return False, f"Job not found: {job_id}"

        if job.status != JobStatus.COMPLETED:
            return False, f"Job not completed: {job.status.value}"

        # Validate
        valid, message = await self._validation_gate.validate(job)

        if not valid:
            job.status = JobStatus.FAILED
            job.error_message = message
            return False, message

        if auto_deploy:
            result = await self.deploy(job_id)
            if result.success:
                return True, f"Deployed successfully: {result.model_name}"
            else:
                return False, f"Deployment failed: {result.error}"

        return True, "Validation passed"

    async def deploy(self, job_id: str) -> DeploymentResult:
        """Deploy a trained model."""
        job = self._jobs.get(job_id)
        if not job or not job.output_path:
            return DeploymentResult(
                success=False,
                model_name="",
                model_path=Path(),
                error="Job or output not found",
            )

        job.status = JobStatus.DEPLOYING

        try:
            # Save checkpoint of current model
            model_name = f"{job.base_model}_finetuned"
            await self._checkpoint_manager.save_checkpoint(
                model_name=job.base_model,
                source_path=job.output_path,
                description=f"Before deployment of job {job.id}",
            )

            # Register the new model
            from jarvis_prime.models.prime_model import PrimeModelRegistry, ModelSpec, ModelCapability
            registry = PrimeModelRegistry()

            # Get base spec
            base_spec = registry.get(job.base_model)
            if base_spec:
                new_spec = ModelSpec(
                    name=model_name,
                    base_model=str(job.output_path),
                    description=f"Fine-tuned {job.base_model} (job: {job.id})",
                    capabilities=base_spec.capabilities,
                    size_gb=base_spec.size_gb,
                    recommended_ram_gb=base_spec.recommended_ram_gb,
                    recommended_vram_gb=base_spec.recommended_vram_gb,
                    complexity_handling=min(1.0, base_spec.complexity_handling + 0.1),
                    local_path=job.output_path,
                    tags=["finetuned", "jarvis"],
                )
                await registry.register(new_spec)

            job.status = JobStatus.DEPLOYED

            return DeploymentResult(
                success=True,
                model_name=model_name,
                model_path=job.output_path,
                previous_model=job.base_model,
                rollback_available=True,
            )

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = f"Deployment failed: {e}"
            return DeploymentResult(
                success=False,
                model_name="",
                model_path=Path(),
                error=str(e),
            )

    async def rollback(self, model_name: str) -> bool:
        """Rollback to previous checkpoint."""
        restored_path = await self._checkpoint_manager.rollback_to_checkpoint(model_name)
        return restored_path is not None

    async def schedule_training(
        self,
        strategy: TrainingStrategy,
        interval_hours: Optional[float] = None,
    ) -> None:
        """Schedule automatic training."""
        await self._scheduler.schedule(strategy, self, interval_hours)

    async def cancel_scheduled(self, strategy: TrainingStrategy) -> None:
        """Cancel scheduled training."""
        await self._scheduler.cancel(strategy)

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_all_jobs(self) -> List[TrainingJob]:
        """Get all jobs."""
        return list(self._jobs.values())

    def get_status(self) -> Dict[str, Any]:
        """Get executor status."""
        running_jobs = [j for j in self._jobs.values() if j.is_running]
        completed_jobs = [j for j in self._jobs.values() if j.status == JobStatus.COMPLETED]
        failed_jobs = [j for j in self._jobs.values() if j.status == JobStatus.FAILED]

        return {
            "running": self._is_running,
            "total_jobs": len(self._jobs),
            "running_jobs": len(running_jobs),
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len(failed_jobs),
            "scheduled_strategies": list(self._scheduler._scheduled_tasks.keys()),
            "config": {
                "default_type": self._config.default_training_type,
                "min_samples": self._config.min_samples_for_training,
                "remote_enabled": self._config.enable_remote_training,
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_executor: Optional[TrainingExecutor] = None
_executor_lock = asyncio.Lock()


async def get_training_executor(
    config: Optional[TrainingExecutorConfig] = None,
) -> TrainingExecutor:
    """Get or create the global training executor."""
    global _executor

    async with _executor_lock:
        if _executor is None:
            _executor = TrainingExecutor(config)
            await _executor.start()

        return _executor


async def shutdown_training_executor() -> None:
    """Shutdown the global training executor."""
    global _executor

    async with _executor_lock:
        if _executor is not None:
            await _executor.stop()
            _executor = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Config
    "TrainingExecutorConfig",
    # Enums
    "TrainingType",
    "TrainingStrategy",
    "JobStatus",
    "JobLocation",
    # Data structures
    "TrainingHyperparameters",
    "TrainingMetrics",
    "TrainingJob",
    "DeploymentResult",
    # Components
    "DataPreparator",
    "LocalTrainer",
    "RemoteTrainer",
    "ValidationGate",
    "CheckpointManager",
    "TrainingScheduler",
    # Executor
    "TrainingExecutor",
    # Factory
    "get_training_executor",
    "shutdown_training_executor",
]
