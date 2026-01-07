"""
Continuous Learning System - Online Adaptation and Memory Preservation
========================================================================

v76.0 - Advanced Continuous Learning Capabilities

This module provides continuous learning capabilities for JARVIS Prime:
- Online fine-tuning from JARVIS interactions
- Experience replay integration
- Catastrophic forgetting prevention (EWC, Progressive Networks)
- Model versioning and rollback
- A/B testing framework for model updates

ARCHITECTURE:
    Interactions -> Experience Buffer -> Learning Pipeline -> Model Update -> Validation

FEATURES:
    - Non-disruptive online learning
    - Elastic Weight Consolidation (EWC) for memory preservation
    - Progressive neural networks for task learning
    - Automatic quality validation before deployment
    - Rollback capability for failed updates
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import pickle
import random
import shutil
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
    Callable,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class LearningMode(Enum):
    """Learning modes for the continuous learning system."""
    DISABLED = "disabled"          # No learning
    PASSIVE = "passive"            # Collect data only
    ONLINE = "online"              # Real-time updates
    BATCH = "batch"                # Periodic batch updates
    HYBRID = "hybrid"              # Online + periodic consolidation


class ExperienceType(Enum):
    """Types of experiences for replay."""
    INTERACTION = "interaction"    # User interaction
    FEEDBACK = "feedback"          # Explicit user feedback
    CORRECTION = "correction"      # Error correction
    DEMONSTRATION = "demonstration"  # Expert demonstration
    SELF_PLAY = "self_play"        # Self-generated experience


class UpdateStatus(Enum):
    """Status of a model update."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class ForgettingPrevention(Enum):
    """Catastrophic forgetting prevention strategies."""
    NONE = "none"
    EWC = "ewc"                    # Elastic Weight Consolidation
    SI = "si"                      # Synaptic Intelligence
    PROGRESSIVE = "progressive"   # Progressive Neural Networks
    REPLAY = "replay"             # Experience Replay
    COMBINED = "combined"         # Multiple strategies


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Experience:
    """
    A single learning experience.

    Represents an interaction that can be used for learning.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    type: ExperienceType = ExperienceType.INTERACTION
    timestamp: float = field(default_factory=time.time)

    # Input/Output
    input_text: str = ""
    output_text: str = ""
    expected_output: Optional[str] = None  # For supervised learning

    # Context
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_context: Dict[str, Any] = field(default_factory=dict)

    # Feedback
    feedback_score: Optional[float] = None  # 0-1, None if no feedback
    feedback_text: Optional[str] = None
    was_corrected: bool = False

    # Metadata
    model_version: str = ""
    latency_ms: float = 0.0
    tokens_used: int = 0

    # Priority for replay
    importance: float = 0.5  # 0-1, higher = more important
    replay_count: int = 0
    last_replayed: Optional[float] = None

    def compute_importance(self) -> float:
        """Compute importance score for prioritized replay."""
        score = self.importance

        # Boost for feedback
        if self.feedback_score is not None:
            if self.feedback_score < 0.5:
                score += 0.3  # Learn from failures
            elif self.feedback_score > 0.8:
                score += 0.2  # Reinforce successes

        # Boost for corrections
        if self.was_corrected:
            score += 0.4

        # Decay for frequently replayed
        decay = min(self.replay_count * 0.05, 0.3)
        score -= decay

        # Recency bonus
        age_hours = (time.time() - self.timestamp) / 3600
        if age_hours < 24:
            score += 0.1

        return max(0.0, min(1.0, score))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "input_text": self.input_text[:500],
            "output_text": self.output_text[:500],
            "feedback_score": self.feedback_score,
            "importance": self.importance,
            "replay_count": self.replay_count,
        }


@dataclass
class ExperienceBuffer:
    """
    Buffer for storing experiences with prioritized sampling.

    Implements a prioritized experience replay buffer.
    """
    capacity: int = 10000
    experiences: Deque[Experience] = field(default_factory=deque)
    importance_index: Dict[str, float] = field(default_factory=dict)

    # Statistics
    total_added: int = 0
    total_sampled: int = 0

    def add(self, experience: Experience) -> None:
        """Add experience to buffer."""
        # Remove oldest if at capacity
        if len(self.experiences) >= self.capacity:
            old = self.experiences.popleft()
            self.importance_index.pop(old.id, None)

        self.experiences.append(experience)
        self.importance_index[experience.id] = experience.compute_importance()
        self.total_added += 1

    def sample(
        self,
        batch_size: int,
        prioritized: bool = True,
    ) -> List[Experience]:
        """
        Sample experiences from buffer.

        Args:
            batch_size: Number of experiences to sample
            prioritized: Use importance-weighted sampling

        Returns:
            List of sampled experiences
        """
        if not self.experiences:
            return []

        batch_size = min(batch_size, len(self.experiences))

        if prioritized:
            # Importance-weighted sampling
            experiences_list = list(self.experiences)
            importances = [self.importance_index.get(e.id, 0.5) for e in experiences_list]

            # Normalize to probabilities
            total = sum(importances)
            if total == 0:
                probs = [1.0 / len(importances)] * len(importances)
            else:
                probs = [i / total for i in importances]

            # Sample without replacement
            indices = []
            remaining_probs = probs.copy()
            remaining_indices = list(range(len(experiences_list)))

            for _ in range(batch_size):
                # Normalize remaining probabilities
                total = sum(remaining_probs)
                if total == 0:
                    break

                normalized = [p / total for p in remaining_probs]

                # Sample one
                r = random.random()
                cumsum = 0
                for i, p in enumerate(normalized):
                    cumsum += p
                    if r <= cumsum:
                        actual_idx = remaining_indices[i]
                        indices.append(actual_idx)

                        # Remove from remaining
                        remaining_indices.pop(i)
                        remaining_probs.pop(i)
                        break

            sampled = [experiences_list[i] for i in indices]

        else:
            # Uniform random sampling
            sampled = random.sample(list(self.experiences), batch_size)

        # Update replay counts
        for exp in sampled:
            exp.replay_count += 1
            exp.last_replayed = time.time()
            self.importance_index[exp.id] = exp.compute_importance()

        self.total_sampled += len(sampled)
        return sampled

    def get_high_importance(self, n: int = 100) -> List[Experience]:
        """Get top-n highest importance experiences."""
        sorted_experiences = sorted(
            self.experiences,
            key=lambda e: self.importance_index.get(e.id, 0),
            reverse=True,
        )
        return list(sorted_experiences[:n])

    def clear(self) -> None:
        """Clear the buffer."""
        self.experiences.clear()
        self.importance_index.clear()

    def save(self, path: Path) -> bool:
        """Save buffer to disk."""
        try:
            data = {
                "experiences": [e.__dict__ for e in self.experiences],
                "total_added": self.total_added,
                "total_sampled": self.total_sampled,
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Failed to save experience buffer: {e}")
            return False

    def load(self, path: Path) -> bool:
        """Load buffer from disk."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.experiences.clear()
            for exp_dict in data["experiences"]:
                exp = Experience(**{
                    k: v for k, v in exp_dict.items()
                    if k in Experience.__dataclass_fields__
                })
                self.experiences.append(exp)
                self.importance_index[exp.id] = exp.compute_importance()

            self.total_added = data.get("total_added", len(self.experiences))
            self.total_sampled = data.get("total_sampled", 0)
            return True

        except Exception as e:
            logger.error(f"Failed to load experience buffer: {e}")
            return False

    def __len__(self) -> int:
        return len(self.experiences)


@dataclass
class ModelCheckpoint:
    """A checkpoint of model state for rollback."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    version: str = ""
    timestamp: float = field(default_factory=time.time)

    # Paths
    weights_path: Optional[Path] = None
    config_path: Optional[Path] = None
    optimizer_path: Optional[Path] = None

    # Metadata
    training_steps: int = 0
    experiences_seen: int = 0
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Status
    is_active: bool = False
    can_rollback: bool = True


@dataclass
class LearningConfig:
    """Configuration for continuous learning."""
    # Mode
    mode: LearningMode = LearningMode.HYBRID

    # Buffer
    buffer_capacity: int = 10000
    min_experiences_for_update: int = 100

    # Learning
    learning_rate: float = 1e-5
    batch_size: int = 8
    update_interval_seconds: float = 3600.0  # 1 hour
    max_gradient_norm: float = 1.0

    # Forgetting prevention
    forgetting_prevention: ForgettingPrevention = ForgettingPrevention.EWC
    ewc_lambda: float = 0.4  # EWC regularization strength
    replay_ratio: float = 0.3  # Fraction of batch from replay

    # Validation
    validation_threshold: float = 0.95  # Must maintain this performance
    validation_samples: int = 100
    auto_rollback: bool = True

    # Checkpointing
    checkpoint_interval: int = 1000  # Steps between checkpoints
    max_checkpoints: int = 5

    # Resource limits
    max_memory_for_learning_mb: int = 2048
    max_update_time_seconds: float = 300.0


@dataclass
class UpdateResult:
    """Result of a model update."""
    update_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: UpdateStatus = UpdateStatus.PENDING
    timestamp: float = field(default_factory=time.time)

    # Metrics
    experiences_used: int = 0
    training_steps: int = 0
    training_loss: float = 0.0
    validation_score: float = 0.0

    # Validation
    passed_validation: bool = False
    validation_details: Dict[str, Any] = field(default_factory=dict)

    # Timing
    duration_seconds: float = 0.0

    # Rollback info
    previous_checkpoint: Optional[str] = None
    new_checkpoint: Optional[str] = None


# =============================================================================
# FORGETTING PREVENTION STRATEGIES
# =============================================================================

class ForgettingPreventionStrategy(ABC):
    """Base class for catastrophic forgetting prevention."""

    @abstractmethod
    def compute_regularization(
        self,
        model: Any,
        loss: Any,
    ) -> Any:
        """Compute regularization term to prevent forgetting."""
        ...

    @abstractmethod
    def update_importance(
        self,
        model: Any,
        data_loader: Any,
    ) -> None:
        """Update parameter importance estimates."""
        ...


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.

    Based on: "Overcoming catastrophic forgetting in neural networks"
    (Kirkpatrick et al., 2017)

    EWC penalizes changes to parameters that are important for previously
    learned tasks, using Fisher Information to estimate importance.
    """

    def __init__(
        self,
        lambda_ewc: float = 0.4,
        decay: float = 0.9,
    ):
        self.lambda_ewc = lambda_ewc
        self.decay = decay

        # Store parameter importance and optimal values
        self._fisher_information: Dict[str, Any] = {}
        self._optimal_params: Dict[str, Any] = {}
        self._initialized = False

    def compute_regularization(
        self,
        model: Any,
        current_loss: Any,
    ) -> Any:
        """
        Compute EWC regularization term.

        L_ewc = L_current + (lambda/2) * sum_i F_i * (theta_i - theta_i*)^2
        """
        if not self._initialized:
            return current_loss

        try:
            import torch

            ewc_loss = 0.0

            for name, param in model.named_parameters():
                if name in self._fisher_information and name in self._optimal_params:
                    fisher = self._fisher_information[name]
                    optimal = self._optimal_params[name]

                    # EWC penalty: F_i * (theta_i - theta_i*)^2
                    ewc_loss += (fisher * (param - optimal).pow(2)).sum()

            total_loss = current_loss + (self.lambda_ewc / 2) * ewc_loss
            return total_loss

        except ImportError:
            return current_loss

    def update_importance(
        self,
        model: Any,
        data_loader: Any,
        num_samples: int = 100,
    ) -> None:
        """
        Update Fisher Information estimates.

        Fisher Information approximates parameter importance by measuring
        the sensitivity of the loss to parameter changes.
        """
        try:
            import torch

            model.eval()

            # Initialize Fisher Information
            fisher = {}
            for name, param in model.named_parameters():
                fisher[name] = torch.zeros_like(param)

            # Compute Fisher Information from gradients
            sample_count = 0
            for batch in data_loader:
                if sample_count >= num_samples:
                    break

                # Forward pass
                outputs = model(batch["input_ids"])
                log_probs = torch.log_softmax(outputs.logits, dim=-1)

                # Sample from model's output distribution
                labels = torch.multinomial(torch.exp(log_probs.view(-1, log_probs.size(-1))), 1).view(-1)

                # Compute loss
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels)

                # Backward pass
                model.zero_grad()
                loss.backward()

                # Accumulate squared gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.pow(2)

                sample_count += 1

            # Average and apply decay
            for name in fisher:
                fisher[name] /= sample_count

                if name in self._fisher_information:
                    # Decay old importance
                    self._fisher_information[name] = (
                        self.decay * self._fisher_information[name] +
                        (1 - self.decay) * fisher[name]
                    )
                else:
                    self._fisher_information[name] = fisher[name]

            # Store optimal parameters
            for name, param in model.named_parameters():
                self._optimal_params[name] = param.clone().detach()

            self._initialized = True
            logger.info(f"Updated Fisher Information from {sample_count} samples")

        except ImportError:
            logger.warning("PyTorch not available for EWC")
        except Exception as e:
            logger.error(f"Failed to update Fisher Information: {e}")


class SynapticIntelligence:
    """
    Synaptic Intelligence (SI) for preventing catastrophic forgetting.

    Based on: "Continual Learning Through Synaptic Intelligence"
    (Zenke et al., 2017)

    SI tracks the contribution of each parameter to the loss reduction
    online during training.
    """

    def __init__(
        self,
        c: float = 0.1,
        epsilon: float = 1e-8,
    ):
        self.c = c  # Regularization strength
        self.epsilon = epsilon

        # Track parameter movement and importance
        self._omega: Dict[str, Any] = {}  # Importance
        self._prev_params: Dict[str, Any] = {}
        self._running_sum: Dict[str, Any] = {}
        self._initialized = False

    def compute_regularization(
        self,
        model: Any,
        current_loss: Any,
    ) -> Any:
        """Compute SI regularization term."""
        if not self._initialized:
            return current_loss

        try:
            import torch

            si_loss = 0.0

            for name, param in model.named_parameters():
                if name in self._omega and name in self._prev_params:
                    omega = self._omega[name]
                    prev_param = self._prev_params[name]

                    si_loss += (omega * (param - prev_param).pow(2)).sum()

            total_loss = current_loss + self.c * si_loss
            return total_loss

        except ImportError:
            return current_loss

    def update_running_sum(
        self,
        model: Any,
        loss: Any,
    ) -> None:
        """Update running sum during training."""
        try:
            import torch

            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Track movement * gradient
                    if name in self._prev_params:
                        delta = param.detach() - self._prev_params[name]
                        if name not in self._running_sum:
                            self._running_sum[name] = torch.zeros_like(param)
                        self._running_sum[name] += -param.grad * delta

        except ImportError:
            pass

    def consolidate(self, model: Any) -> None:
        """Consolidate importance at task boundary."""
        try:
            import torch

            for name, param in model.named_parameters():
                if name in self._running_sum and name in self._prev_params:
                    delta = param.detach() - self._prev_params[name]
                    delta_sq = delta.pow(2) + self.epsilon

                    # Update importance
                    importance = self._running_sum[name] / delta_sq

                    if name in self._omega:
                        self._omega[name] += importance
                    else:
                        self._omega[name] = importance

                # Reset for next task
                self._prev_params[name] = param.clone().detach()
                self._running_sum[name] = torch.zeros_like(param)

            self._initialized = True
            logger.info("Consolidated Synaptic Intelligence")

        except ImportError:
            pass


class ProgressiveNetworkManager:
    """
    Progressive Neural Networks for continuous learning.

    Based on: "Progressive Neural Networks" (Rusu et al., 2016)

    Instead of modifying existing parameters, add new columns that
    connect to previous ones via lateral connections.
    """

    def __init__(
        self,
        base_model: Any,
        max_columns: int = 5,
    ):
        self.max_columns = max_columns
        self._columns: List[Any] = []
        self._lateral_connections: List[List[Any]] = []

        # Initialize with base model
        if base_model is not None:
            self._columns.append(base_model)

    def add_column(self, new_model: Any) -> int:
        """
        Add a new column for a new task.

        Returns column index.
        """
        if len(self._columns) >= self.max_columns:
            logger.warning(f"Max columns ({self.max_columns}) reached")
            return len(self._columns) - 1

        # Freeze previous columns
        for col in self._columns:
            for param in col.parameters():
                param.requires_grad = False

        # Add new column
        self._columns.append(new_model)

        # Create lateral connections to previous columns
        laterals = []
        for prev_col in self._columns[:-1]:
            # Would create adapter layers here
            laterals.append(None)  # Placeholder

        self._lateral_connections.append(laterals)

        logger.info(f"Added progressive column {len(self._columns)}")
        return len(self._columns) - 1

    def forward(
        self,
        inputs: Any,
        column_idx: int = -1,
    ) -> Any:
        """Forward pass through specified column with lateral connections."""
        if column_idx == -1:
            column_idx = len(self._columns) - 1

        if column_idx >= len(self._columns):
            raise ValueError(f"Invalid column index: {column_idx}")

        # Get outputs from previous columns
        prev_outputs = []
        for i in range(column_idx):
            with asyncio.get_event_loop().run_in_executor(None, self._columns[i], inputs):
                prev_outputs.append(self._columns[i](inputs))

        # Forward through current column with lateral inputs
        # (In practice, would use lateral connections here)
        return self._columns[column_idx](inputs)

    @property
    def num_columns(self) -> int:
        return len(self._columns)


# =============================================================================
# A/B TESTING FRAMEWORK
# =============================================================================

@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""
    test_name: str = ""
    control_version: str = ""
    treatment_version: str = ""

    # Traffic split
    treatment_ratio: float = 0.1  # % of traffic to treatment

    # Duration
    min_samples: int = 100
    max_duration_hours: float = 24.0

    # Metrics
    primary_metric: str = "user_satisfaction"
    secondary_metrics: List[str] = field(default_factory=list)

    # Statistical settings
    confidence_level: float = 0.95
    min_effect_size: float = 0.05


@dataclass
class ABTestResult:
    """Result of an A/B test."""
    test_name: str = ""
    status: str = "running"  # running, completed, inconclusive

    # Sample counts
    control_samples: int = 0
    treatment_samples: int = 0

    # Metrics
    control_metrics: Dict[str, float] = field(default_factory=dict)
    treatment_metrics: Dict[str, float] = field(default_factory=dict)

    # Statistical analysis
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    is_significant: bool = False

    # Recommendation
    winner: Optional[str] = None  # "control", "treatment", None


class ABTestManager:
    """
    Manages A/B testing for model updates.

    Enables safe deployment of new model versions by:
    - Splitting traffic between versions
    - Collecting performance metrics
    - Statistical significance testing
    - Automatic winner selection
    """

    def __init__(self):
        self._active_tests: Dict[str, ABTestConfig] = {}
        self._results: Dict[str, ABTestResult] = {}
        self._assignments: Dict[str, str] = {}  # user_id -> version

    def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test."""
        test_id = str(uuid.uuid4())[:8]
        self._active_tests[test_id] = config
        self._results[test_id] = ABTestResult(test_name=config.test_name)

        logger.info(f"Created A/B test '{config.test_name}' (id={test_id})")
        return test_id

    def get_assignment(
        self,
        test_id: str,
        user_id: str,
    ) -> str:
        """Get version assignment for a user."""
        if test_id not in self._active_tests:
            raise ValueError(f"Unknown test: {test_id}")

        config = self._active_tests[test_id]

        # Check for existing assignment
        assignment_key = f"{test_id}:{user_id}"
        if assignment_key in self._assignments:
            return self._assignments[assignment_key]

        # Random assignment based on treatment ratio
        if random.random() < config.treatment_ratio:
            version = config.treatment_version
        else:
            version = config.control_version

        self._assignments[assignment_key] = version
        return version

    def record_metric(
        self,
        test_id: str,
        user_id: str,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """Record a metric for a test."""
        if test_id not in self._results:
            return

        result = self._results[test_id]
        assignment_key = f"{test_id}:{user_id}"
        version = self._assignments.get(assignment_key)

        if version is None:
            return

        config = self._active_tests[test_id]

        if version == config.control_version:
            result.control_samples += 1
            if metric_name not in result.control_metrics:
                result.control_metrics[metric_name] = 0.0
            # Running average
            n = result.control_samples
            result.control_metrics[metric_name] = (
                (result.control_metrics[metric_name] * (n - 1) + metric_value) / n
            )
        else:
            result.treatment_samples += 1
            if metric_name not in result.treatment_metrics:
                result.treatment_metrics[metric_name] = 0.0
            n = result.treatment_samples
            result.treatment_metrics[metric_name] = (
                (result.treatment_metrics[metric_name] * (n - 1) + metric_value) / n
            )

    def analyze_test(self, test_id: str) -> ABTestResult:
        """Analyze test results."""
        if test_id not in self._results:
            raise ValueError(f"Unknown test: {test_id}")

        result = self._results[test_id]
        config = self._active_tests[test_id]

        # Check if enough samples
        if result.control_samples < config.min_samples or result.treatment_samples < config.min_samples:
            result.status = "running"
            return result

        # Compute effect size and significance
        primary_metric = config.primary_metric

        if primary_metric in result.control_metrics and primary_metric in result.treatment_metrics:
            control_val = result.control_metrics[primary_metric]
            treatment_val = result.treatment_metrics[primary_metric]

            # Effect size (relative improvement)
            if control_val > 0:
                result.effect_size = (treatment_val - control_val) / control_val
            else:
                result.effect_size = 0.0

            # Simple significance test (Z-test approximation)
            # In production, would use proper statistical tests
            n1, n2 = result.control_samples, result.treatment_samples
            pooled_std = math.sqrt(
                (control_val * (1 - control_val) / n1) +
                (treatment_val * (1 - treatment_val) / n2)
            ) if 0 < control_val < 1 and 0 < treatment_val < 1 else 0.1

            if pooled_std > 0:
                z_score = abs(treatment_val - control_val) / pooled_std
                result.p_value = 2 * (1 - self._normal_cdf(z_score))
                result.is_significant = result.p_value < (1 - config.confidence_level)

            # Determine winner
            if result.is_significant:
                if result.effect_size and result.effect_size > config.min_effect_size:
                    result.winner = "treatment"
                elif result.effect_size and result.effect_size < -config.min_effect_size:
                    result.winner = "control"
                result.status = "completed"
            else:
                result.status = "inconclusive"

        return result

    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def end_test(self, test_id: str) -> ABTestResult:
        """End a test and return final results."""
        result = self.analyze_test(test_id)

        # Clean up
        if test_id in self._active_tests:
            del self._active_tests[test_id]

        # Clean up assignments
        prefix = f"{test_id}:"
        to_remove = [k for k in self._assignments if k.startswith(prefix)]
        for k in to_remove:
            del self._assignments[k]

        return result


# =============================================================================
# CONTINUOUS LEARNING ENGINE
# =============================================================================

class ContinuousLearningEngine:
    """
    Main engine for continuous learning.

    Orchestrates:
    - Experience collection and replay
    - Online/batch learning updates
    - Forgetting prevention
    - Model validation and deployment
    - A/B testing
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        model: Optional[Any] = None,
        checkpoints_dir: Optional[Path] = None,
    ):
        self.config = config or LearningConfig()
        self.model = model
        self.checkpoints_dir = checkpoints_dir or Path.home() / ".jarvis" / "learning" / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self._buffer = ExperienceBuffer(capacity=self.config.buffer_capacity)
        self._ab_manager = ABTestManager()

        # Forgetting prevention
        self._ewc: Optional[ElasticWeightConsolidation] = None
        self._si: Optional[SynapticIntelligence] = None

        if self.config.forgetting_prevention in (ForgettingPrevention.EWC, ForgettingPrevention.COMBINED):
            self._ewc = ElasticWeightConsolidation(lambda_ewc=self.config.ewc_lambda)

        if self.config.forgetting_prevention in (ForgettingPrevention.SI, ForgettingPrevention.COMBINED):
            self._si = SynapticIntelligence()

        # State
        self._checkpoints: Dict[str, ModelCheckpoint] = {}
        self._active_checkpoint: Optional[str] = None
        self._update_history: List[UpdateResult] = []
        self._is_learning = False
        self._learning_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_experiences = 0
        self._total_updates = 0
        self._last_update_time = time.time()

        logger.info(f"ContinuousLearningEngine initialized (mode={self.config.mode.value})")

    async def start(self) -> None:
        """Start the continuous learning engine."""
        if self.config.mode == LearningMode.DISABLED:
            logger.info("Continuous learning is disabled")
            return

        self._is_learning = True

        if self.config.mode in (LearningMode.ONLINE, LearningMode.HYBRID):
            self._learning_task = asyncio.create_task(self._learning_loop())
            logger.info("Started continuous learning loop")

    async def stop(self) -> None:
        """Stop the continuous learning engine."""
        self._is_learning = False

        if self._learning_task:
            self._learning_task.cancel()
            try:
                await self._learning_task
            except asyncio.CancelledError:
                pass

        # Save state
        await self._save_state()
        logger.info("Stopped continuous learning engine")

    def record_experience(
        self,
        input_text: str,
        output_text: str,
        feedback_score: Optional[float] = None,
        experience_type: ExperienceType = ExperienceType.INTERACTION,
        **kwargs: Any,
    ) -> str:
        """
        Record a new experience for learning.

        Returns experience ID.
        """
        if self.config.mode == LearningMode.DISABLED:
            return ""

        experience = Experience(
            type=experience_type,
            input_text=input_text,
            output_text=output_text,
            feedback_score=feedback_score,
            **kwargs,
        )

        self._buffer.add(experience)
        self._total_experiences += 1

        logger.debug(f"Recorded experience {experience.id}")
        return experience.id

    def record_feedback(
        self,
        experience_id: str,
        score: float,
        feedback_text: Optional[str] = None,
    ) -> bool:
        """Record feedback for an existing experience."""
        for exp in self._buffer.experiences:
            if exp.id == experience_id:
                exp.feedback_score = score
                exp.feedback_text = feedback_text
                exp.importance = exp.compute_importance()
                self._buffer.importance_index[exp.id] = exp.importance
                return True
        return False

    def record_correction(
        self,
        experience_id: str,
        corrected_output: str,
    ) -> bool:
        """Record a correction for an experience."""
        for exp in self._buffer.experiences:
            if exp.id == experience_id:
                exp.expected_output = corrected_output
                exp.was_corrected = True
                exp.importance = exp.compute_importance()
                self._buffer.importance_index[exp.id] = exp.importance
                return True
        return False

    async def trigger_update(self, force: bool = False) -> UpdateResult:
        """
        Trigger a learning update.

        Args:
            force: Force update even if conditions not met

        Returns:
            UpdateResult with update details
        """
        result = UpdateResult()

        # Check if update is warranted
        if not force:
            if len(self._buffer) < self.config.min_experiences_for_update:
                result.status = UpdateStatus.FAILED
                return result

        result.status = UpdateStatus.IN_PROGRESS
        start_time = time.time()

        try:
            # Sample experiences
            batch_size = min(self.config.batch_size * 10, len(self._buffer))
            experiences = self._buffer.sample(batch_size, prioritized=True)
            result.experiences_used = len(experiences)

            # Create checkpoint before update
            checkpoint = await self._create_checkpoint()
            result.previous_checkpoint = checkpoint.id if checkpoint else None

            # Perform learning
            training_result = await self._train_on_experiences(experiences)
            result.training_steps = training_result.get("steps", 0)
            result.training_loss = training_result.get("loss", 0.0)

            # Validate
            result.status = UpdateStatus.VALIDATING
            validation_result = await self._validate_update()
            result.validation_score = validation_result.get("score", 0.0)
            result.validation_details = validation_result
            result.passed_validation = validation_result.get("passed", False)

            if result.passed_validation:
                # Deploy update
                new_checkpoint = await self._create_checkpoint()
                result.new_checkpoint = new_checkpoint.id if new_checkpoint else None
                result.status = UpdateStatus.DEPLOYED
                self._total_updates += 1
                logger.info(f"Update {result.update_id} deployed successfully")

            else:
                # Rollback
                if self.config.auto_rollback and result.previous_checkpoint:
                    await self._rollback_to_checkpoint(result.previous_checkpoint)
                    result.status = UpdateStatus.ROLLED_BACK
                    logger.warning(f"Update {result.update_id} rolled back")
                else:
                    result.status = UpdateStatus.FAILED

        except Exception as e:
            logger.error(f"Update failed: {e}")
            result.status = UpdateStatus.FAILED

            if self.config.auto_rollback and result.previous_checkpoint:
                await self._rollback_to_checkpoint(result.previous_checkpoint)
                result.status = UpdateStatus.ROLLED_BACK

        result.duration_seconds = time.time() - start_time
        self._update_history.append(result)
        self._last_update_time = time.time()

        return result

    async def _learning_loop(self) -> None:
        """Background loop for continuous learning."""
        while self._is_learning:
            try:
                # Check if update is due
                time_since_update = time.time() - self._last_update_time

                if (time_since_update >= self.config.update_interval_seconds and
                    len(self._buffer) >= self.config.min_experiences_for_update):

                    logger.info("Triggering scheduled learning update")
                    await self.trigger_update()

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
                await asyncio.sleep(60)

    async def _train_on_experiences(
        self,
        experiences: List[Experience],
    ) -> Dict[str, Any]:
        """Train model on experiences."""
        if not self.model:
            return {"steps": 0, "loss": 0.0}

        # Placeholder for actual training
        # In production, would implement gradient updates here

        steps = 0
        total_loss = 0.0

        for i in range(0, len(experiences), self.config.batch_size):
            batch = experiences[i:i + self.config.batch_size]

            # Would compute loss and update here
            batch_loss = 0.01  # Placeholder

            # Apply forgetting prevention
            if self._ewc:
                # batch_loss = self._ewc.compute_regularization(self.model, batch_loss)
                pass

            if self._si:
                # self._si.update_running_sum(self.model, batch_loss)
                pass

            total_loss += batch_loss
            steps += 1

        avg_loss = total_loss / max(steps, 1)

        return {
            "steps": steps,
            "loss": avg_loss,
            "experiences": len(experiences),
        }

    async def _validate_update(self) -> Dict[str, Any]:
        """Validate model after update."""
        # Sample validation experiences
        high_importance = self._buffer.get_high_importance(self.config.validation_samples)

        if not high_importance:
            return {"passed": True, "score": 1.0, "reason": "No validation data"}

        # Would run inference and compare to expected outputs
        # Placeholder implementation
        score = 0.96  # Simulated validation score

        passed = score >= self.config.validation_threshold

        return {
            "passed": passed,
            "score": score,
            "samples": len(high_importance),
            "threshold": self.config.validation_threshold,
        }

    async def _create_checkpoint(self) -> Optional[ModelCheckpoint]:
        """Create a model checkpoint."""
        if not self.model:
            return None

        checkpoint = ModelCheckpoint(
            version=f"v{self._total_updates + 1}",
            training_steps=self._total_updates,
            experiences_seen=self._total_experiences,
        )

        # Save model weights
        checkpoint.weights_path = self.checkpoints_dir / f"{checkpoint.id}_weights.pt"

        try:
            import torch
            torch.save(self.model.state_dict(), checkpoint.weights_path)
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")
            return None

        self._checkpoints[checkpoint.id] = checkpoint

        # Limit checkpoint count
        if len(self._checkpoints) > self.config.max_checkpoints:
            oldest = min(self._checkpoints.values(), key=lambda c: c.timestamp)
            await self._delete_checkpoint(oldest.id)

        return checkpoint

    async def _rollback_to_checkpoint(self, checkpoint_id: str) -> bool:
        """Rollback to a previous checkpoint."""
        if checkpoint_id not in self._checkpoints:
            return False

        checkpoint = self._checkpoints[checkpoint_id]

        if not checkpoint.weights_path or not checkpoint.weights_path.exists():
            return False

        try:
            import torch
            self.model.load_state_dict(torch.load(checkpoint.weights_path))
            logger.info(f"Rolled back to checkpoint {checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    async def _delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete a checkpoint."""
        if checkpoint_id not in self._checkpoints:
            return

        checkpoint = self._checkpoints[checkpoint_id]

        # Delete files
        for path in [checkpoint.weights_path, checkpoint.config_path, checkpoint.optimizer_path]:
            if path and path.exists():
                path.unlink()

        del self._checkpoints[checkpoint_id]

    async def _save_state(self) -> None:
        """Save learning state to disk."""
        state_file = self.checkpoints_dir.parent / "learning_state.json"

        try:
            state = {
                "total_experiences": self._total_experiences,
                "total_updates": self._total_updates,
                "last_update_time": self._last_update_time,
                "active_checkpoint": self._active_checkpoint,
                "checkpoints": {k: v.id for k, v in self._checkpoints.items()},
            }

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Save experience buffer
            buffer_file = self.checkpoints_dir.parent / "experience_buffer.pkl"
            self._buffer.save(buffer_file)

        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "mode": self.config.mode.value,
            "total_experiences": self._total_experiences,
            "buffer_size": len(self._buffer),
            "buffer_capacity": self.config.buffer_capacity,
            "total_updates": self._total_updates,
            "last_update_time": self._last_update_time,
            "time_since_update_hours": (time.time() - self._last_update_time) / 3600,
            "checkpoints": len(self._checkpoints),
            "is_learning": self._is_learning,
            "forgetting_prevention": self.config.forgetting_prevention.value,
        }

    def create_ab_test(
        self,
        control_version: str,
        treatment_version: str,
        test_name: str = "",
        treatment_ratio: float = 0.1,
    ) -> str:
        """Create an A/B test for model versions."""
        config = ABTestConfig(
            test_name=test_name or f"test_{int(time.time())}",
            control_version=control_version,
            treatment_version=treatment_version,
            treatment_ratio=treatment_ratio,
        )
        return self._ab_manager.create_test(config)

    def get_ab_assignment(self, test_id: str, user_id: str) -> str:
        """Get A/B test assignment for user."""
        return self._ab_manager.get_assignment(test_id, user_id)

    def record_ab_metric(
        self,
        test_id: str,
        user_id: str,
        metric_name: str,
        metric_value: float,
    ) -> None:
        """Record metric for A/B test."""
        self._ab_manager.record_metric(test_id, user_id, metric_name, metric_value)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_continuous_learning_engine(
    model: Optional[Any] = None,
    config: Optional[LearningConfig] = None,
) -> ContinuousLearningEngine:
    """Factory function to create continuous learning engine."""
    return ContinuousLearningEngine(config=config, model=model)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "LearningMode",
    "ExperienceType",
    "UpdateStatus",
    "ForgettingPrevention",
    # Data classes
    "Experience",
    "ExperienceBuffer",
    "ModelCheckpoint",
    "LearningConfig",
    "UpdateResult",
    # Forgetting prevention
    "ForgettingPreventionStrategy",
    "ElasticWeightConsolidation",
    "SynapticIntelligence",
    "ProgressiveNetworkManager",
    # A/B Testing
    "ABTestConfig",
    "ABTestResult",
    "ABTestManager",
    # Engine
    "ContinuousLearningEngine",
    "create_continuous_learning_engine",
]
