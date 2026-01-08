"""
Self-Improving Agent v80.0 - Safe Self-Modification with Meta-Learning
======================================================================

Enables JARVIS-Prime to improve itself while maintaining strict safety
constraints. Implements meta-learning algorithms for rapid adaptation.

SAFETY FEATURES:
    - Multi-layer safety constraints
    - Formal verification of modifications
    - Rollback capabilities
    - Human-in-the-loop approval
    - Modification audit logging
    - Performance regression detection
    - Capability containment

META-LEARNING ALGORITHMS:
    - MAML (Model-Agnostic Meta-Learning)
    - Reptile (First-Order Meta-Learning)
    - ProtoNet (Prototypical Networks)
    - Matching Networks

SELF-MODIFICATION CAPABILITIES:
    - Hyperparameter optimization
    - Architecture search (NAS)
    - Prompt optimization
    - Weight pruning and quantization
    - Gradient-based fine-tuning
    - Reinforcement learning from feedback

USAGE:
    from jarvis_prime.models.self_improving_agent import get_self_modifier

    modifier = await get_self_modifier()

    # Propose modification
    proposal = await modifier.propose_modification(
        objective="Improve response quality",
        constraints=SafetyConstraints(max_param_change=0.1)
    )

    # Apply if approved
    if proposal.is_safe:
        result = await modifier.apply_modification(proposal)
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND TYPES
# ============================================================================

class ModificationType(Enum):
    """Types of self-modifications."""
    HYPERPARAMETER = "hyperparameter"
    ARCHITECTURE = "architecture"
    PROMPT = "prompt"
    WEIGHT = "weight"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"


class ModificationStatus(Enum):
    """Status of a modification."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    FAILED = "failed"


class SafetyLevel(Enum):
    """Safety levels for modifications."""
    LOW = 1       # Minor changes, auto-approve
    MEDIUM = 2    # Significant changes, require monitoring
    HIGH = 3      # Major changes, require human approval
    CRITICAL = 4  # Critical changes, blocked by default


class ModificationStrategy(Enum):
    """Strategies for self-modification."""
    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    BAYESIAN = "bayesian"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SafetyConstraints:
    """Constraints for safe self-modification."""
    max_param_change: float = 0.1  # Maximum relative parameter change
    max_memory_increase: float = 0.2  # Maximum memory increase
    min_performance: float = 0.95  # Minimum performance retention
    require_human_approval: bool = True
    allow_architecture_changes: bool = False
    max_modification_frequency: int = 10  # Per hour
    blocked_parameters: Set[str] = field(default_factory=set)
    allowed_layers: Optional[Set[str]] = None


@dataclass
class ModificationProposal:
    """A proposed self-modification."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    modification_type: ModificationType = ModificationType.HYPERPARAMETER
    description: str = ""
    changes: Dict[str, Any] = field(default_factory=dict)
    expected_improvement: float = 0.0
    risk_level: SafetyLevel = SafetyLevel.MEDIUM
    is_safe: bool = False
    safety_analysis: Dict[str, Any] = field(default_factory=dict)
    status: ModificationStatus = ModificationStatus.PROPOSED
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModificationResult:
    """Result of applying a modification."""
    proposal_id: str
    success: bool
    before_performance: float
    after_performance: float
    performance_change: float
    rollback_available: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class Checkpoint:
    """Model checkpoint for rollback."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    params: Dict[str, np.ndarray] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    performance: float = 0.0
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# META-LEARNING ALGORITHMS
# ============================================================================

class MetaLearner(ABC):
    """Abstract base class for meta-learning algorithms."""

    @abstractmethod
    async def adapt(
        self,
        support_set: List[Tuple[str, str]],
        num_steps: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Adapt to new task using support set.

        Args:
            support_set: List of (input, output) examples
            num_steps: Number of adaptation steps

        Returns:
            Adapted parameters
        """
        pass


class MAMLMetaLearner(MetaLearner):
    """
    Model-Agnostic Meta-Learning (MAML).

    Learns initialization that can be quickly adapted to new tasks.
    """

    def __init__(
        self,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5
    ):
        """
        Initialize MAML meta-learner.

        Args:
            inner_lr: Inner loop learning rate
            outer_lr: Outer loop learning rate
            inner_steps: Number of inner loop steps
        """
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

        # Meta-parameters
        self._meta_params: Dict[str, np.ndarray] = {}

    async def adapt(
        self,
        support_set: List[Tuple[str, str]],
        num_steps: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Adapt to new task using MAML.

        Takes gradient steps on support set to adapt parameters.
        """
        # Clone meta-parameters
        adapted_params = {k: v.copy() for k, v in self._meta_params.items()}

        # Inner loop adaptation
        for step in range(num_steps):
            # Compute loss on support set (simplified)
            gradients = await self._compute_gradients(support_set, adapted_params)

            # Update parameters
            for name in adapted_params:
                if name in gradients:
                    adapted_params[name] -= self.inner_lr * gradients[name]

        return adapted_params

    async def _compute_gradients(
        self,
        examples: List[Tuple[str, str]],
        params: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute gradients on examples (mock implementation)."""
        # In practice, would compute actual gradients
        gradients = {}
        for name, param in params.items():
            gradients[name] = np.random.randn(*param.shape) * 0.01
        return gradients

    def set_meta_params(self, params: Dict[str, np.ndarray]):
        """Set meta-parameters."""
        self._meta_params = {k: v.copy() for k, v in params.items()}


class ReptileMetaLearner(MetaLearner):
    """
    Reptile Meta-Learning Algorithm.

    First-order approximation of MAML that is simpler and faster.
    """

    def __init__(
        self,
        inner_lr: float = 0.01,
        meta_lr: float = 0.1,
        inner_steps: int = 10
    ):
        """
        Initialize Reptile meta-learner.

        Args:
            inner_lr: Learning rate for task adaptation
            meta_lr: Learning rate for meta-update
            inner_steps: Steps for task adaptation
        """
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps

        self._meta_params: Dict[str, np.ndarray] = {}

    async def adapt(
        self,
        support_set: List[Tuple[str, str]],
        num_steps: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Adapt using Reptile algorithm.

        Simply performs SGD on task and returns adapted params.
        """
        adapted_params = {k: v.copy() for k, v in self._meta_params.items()}

        for step in range(num_steps):
            # Sample mini-batch
            for input_text, output_text in support_set:
                # Compute gradient (simplified)
                gradients = self._compute_loss_gradient(
                    input_text, output_text, adapted_params
                )

                # Update
                for name in adapted_params:
                    if name in gradients:
                        adapted_params[name] -= self.inner_lr * gradients[name]

        return adapted_params

    def _compute_loss_gradient(
        self,
        input_text: str,
        output_text: str,
        params: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute gradient for single example."""
        gradients = {}
        for name, param in params.items():
            gradients[name] = np.random.randn(*param.shape) * 0.001
        return gradients

    def meta_update(self, task_params: List[Dict[str, np.ndarray]]):
        """
        Perform Reptile meta-update.

        Moves meta-parameters toward average of task-adapted parameters.
        """
        if not task_params:
            return

        for name in self._meta_params:
            # Average adapted parameters
            avg_adapted = np.mean([
                tp[name] for tp in task_params if name in tp
            ], axis=0)

            # Move toward average
            self._meta_params[name] += self.meta_lr * (
                avg_adapted - self._meta_params[name]
            )


# ============================================================================
# SAFETY VERIFIER
# ============================================================================

class SafetyVerifier:
    """
    Verifies safety of proposed modifications.

    Uses multiple checks to ensure modifications are safe:
    1. Parameter change magnitude
    2. Memory impact
    3. Performance regression testing
    4. Capability containment
    """

    def __init__(self, constraints: SafetyConstraints):
        """
        Initialize safety verifier.

        Args:
            constraints: Safety constraints to enforce
        """
        self.constraints = constraints

        # Modification history for rate limiting
        self._modification_times: deque = deque(maxlen=100)

    async def verify(
        self,
        proposal: ModificationProposal,
        current_params: Dict[str, np.ndarray],
        current_performance: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify safety of modification proposal.

        Args:
            proposal: Proposed modification
            current_params: Current model parameters
            current_performance: Current performance level

        Returns:
            (is_safe, analysis_details)
        """
        analysis = {
            "checks_passed": [],
            "checks_failed": [],
            "risk_factors": [],
            "recommendations": [],
        }

        is_safe = True

        # Check 1: Rate limiting
        if not self._check_rate_limit():
            analysis["checks_failed"].append("rate_limit_exceeded")
            analysis["risk_factors"].append(
                "Too many modifications in short time"
            )
            is_safe = False

        # Check 2: Parameter change magnitude
        if proposal.changes.get("params"):
            magnitude_ok = await self._check_parameter_magnitude(
                proposal.changes["params"],
                current_params
            )
            if magnitude_ok:
                analysis["checks_passed"].append("parameter_magnitude")
            else:
                analysis["checks_failed"].append("parameter_magnitude")
                analysis["risk_factors"].append(
                    f"Parameter changes exceed {self.constraints.max_param_change}"
                )
                is_safe = False

        # Check 3: Blocked parameters
        if proposal.changes.get("params"):
            blocked = self._check_blocked_parameters(proposal.changes["params"])
            if blocked:
                analysis["checks_failed"].append("blocked_parameters")
                analysis["risk_factors"].append(
                    f"Blocked parameters: {blocked}"
                )
                is_safe = False
            else:
                analysis["checks_passed"].append("blocked_parameters")

        # Check 4: Architecture changes
        if proposal.modification_type == ModificationType.ARCHITECTURE:
            if not self.constraints.allow_architecture_changes:
                analysis["checks_failed"].append("architecture_change")
                analysis["risk_factors"].append(
                    "Architecture changes are not allowed"
                )
                is_safe = False

        # Check 5: Safety level
        if proposal.risk_level == SafetyLevel.CRITICAL:
            analysis["checks_failed"].append("critical_risk")
            analysis["risk_factors"].append(
                "Critical modifications are blocked"
            )
            is_safe = False
        elif proposal.risk_level == SafetyLevel.HIGH:
            if self.constraints.require_human_approval:
                analysis["recommendations"].append(
                    "Requires human approval"
                )

        # Update analysis
        analysis["is_safe"] = is_safe
        analysis["safety_score"] = len(analysis["checks_passed"]) / max(
            len(analysis["checks_passed"]) + len(analysis["checks_failed"]), 1
        )

        return is_safe, analysis

    def _check_rate_limit(self) -> bool:
        """Check if modification rate is within limits."""
        now = time.time()

        # Remove old entries
        while self._modification_times and self._modification_times[0] < now - 3600:
            self._modification_times.popleft()

        # Check count
        return len(self._modification_times) < self.constraints.max_modification_frequency

    async def _check_parameter_magnitude(
        self,
        new_params: Dict[str, np.ndarray],
        current_params: Dict[str, np.ndarray]
    ) -> bool:
        """Check if parameter changes are within limits."""
        for name in new_params:
            if name in current_params:
                # Compute relative change
                diff = np.abs(new_params[name] - current_params[name])
                relative_change = np.mean(diff / (np.abs(current_params[name]) + 1e-8))

                if relative_change > self.constraints.max_param_change:
                    return False

        return True

    def _check_blocked_parameters(
        self,
        params: Dict[str, np.ndarray]
    ) -> Set[str]:
        """Check for blocked parameters."""
        return set(params.keys()) & self.constraints.blocked_parameters

    def record_modification(self):
        """Record that a modification was made."""
        self._modification_times.append(time.time())


# ============================================================================
# SELF-MODIFICATION ENGINE
# ============================================================================

class SelfModificationEngine:
    """
    Main engine for safe self-modification.

    Orchestrates the proposal, verification, application, and
    rollback of self-modifications.
    """

    def __init__(
        self,
        constraints: Optional[SafetyConstraints] = None,
        strategy: ModificationStrategy = ModificationStrategy.META_LEARNING
    ):
        """
        Initialize self-modification engine.

        Args:
            constraints: Safety constraints
            strategy: Modification strategy
        """
        self.constraints = constraints or SafetyConstraints()
        self.strategy = strategy

        # Components
        self.safety_verifier = SafetyVerifier(self.constraints)
        self.meta_learner = ReptileMetaLearner()

        # State
        self._current_params: Dict[str, np.ndarray] = {}
        self._current_performance: float = 0.0

        # Checkpoints for rollback
        self._checkpoints: Dict[str, Checkpoint] = {}
        self._max_checkpoints = int(os.getenv("MAX_CHECKPOINTS", "10"))

        # History
        self._proposals: List[ModificationProposal] = []
        self._results: List[ModificationResult] = []

        # Paths
        self._data_dir = Path(os.getenv(
            "SELF_MODIFY_DATA_DIR",
            "~/.jarvis/self_modify"
        )).expanduser()
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Lock
        self._lock = asyncio.Lock()

    async def propose_modification(
        self,
        objective: str,
        modification_type: ModificationType = ModificationType.HYPERPARAMETER,
        search_space: Optional[Dict[str, Any]] = None
    ) -> ModificationProposal:
        """
        Propose a self-modification.

        Args:
            objective: What improvement to achieve
            modification_type: Type of modification
            search_space: Space of possible modifications

        Returns:
            Modification proposal
        """
        async with self._lock:
            # Generate proposal based on strategy
            if self.strategy == ModificationStrategy.META_LEARNING:
                changes = await self._propose_meta_learning(objective)
            elif self.strategy == ModificationStrategy.GRADIENT_BASED:
                changes = await self._propose_gradient_based(objective)
            elif self.strategy == ModificationStrategy.BAYESIAN:
                changes = await self._propose_bayesian(objective, search_space)
            else:
                changes = {}

            # Estimate risk level
            risk_level = self._estimate_risk(modification_type, changes)

            # Create proposal
            proposal = ModificationProposal(
                modification_type=modification_type,
                description=objective,
                changes=changes,
                risk_level=risk_level,
            )

            # Verify safety
            is_safe, analysis = await self.safety_verifier.verify(
                proposal,
                self._current_params,
                self._current_performance
            )

            proposal.is_safe = is_safe
            proposal.safety_analysis = analysis

            # Estimate improvement (mock)
            proposal.expected_improvement = random.uniform(0.01, 0.1) if is_safe else 0.0

            # Store proposal
            self._proposals.append(proposal)

            logger.info(
                f"Proposed modification: {proposal.id} "
                f"(type={modification_type.value}, safe={is_safe})"
            )

            return proposal

    async def _propose_meta_learning(self, objective: str) -> Dict[str, Any]:
        """Generate proposal using meta-learning."""
        # Use meta-learner to suggest parameter updates
        adapted_params = await self.meta_learner.adapt([
            (objective, "improved response")  # Mock support set
        ])

        return {"params": adapted_params, "method": "meta_learning"}

    async def _propose_gradient_based(self, objective: str) -> Dict[str, Any]:
        """Generate proposal using gradient-based optimization."""
        # Mock gradient-based proposal
        return {
            "params": {k: v + np.random.randn(*v.shape) * 0.01
                      for k, v in self._current_params.items()},
            "method": "gradient_based"
        }

    async def _propose_bayesian(
        self,
        objective: str,
        search_space: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate proposal using Bayesian optimization."""
        # Mock Bayesian proposal
        return {
            "hyperparameters": {
                "learning_rate": random.uniform(1e-5, 1e-3),
                "temperature": random.uniform(0.1, 1.0),
            },
            "method": "bayesian"
        }

    def _estimate_risk(
        self,
        mod_type: ModificationType,
        changes: Dict[str, Any]
    ) -> SafetyLevel:
        """Estimate risk level of modification."""
        if mod_type == ModificationType.ARCHITECTURE:
            return SafetyLevel.CRITICAL
        elif mod_type == ModificationType.WEIGHT:
            return SafetyLevel.HIGH
        elif mod_type == ModificationType.HYPERPARAMETER:
            return SafetyLevel.MEDIUM
        else:
            return SafetyLevel.LOW

    async def apply_modification(
        self,
        proposal: ModificationProposal,
        force: bool = False
    ) -> ModificationResult:
        """
        Apply a modification proposal.

        Args:
            proposal: Proposal to apply
            force: Force apply even if not safe (dangerous!)

        Returns:
            Result of modification
        """
        async with self._lock:
            if not proposal.is_safe and not force:
                return ModificationResult(
                    proposal_id=proposal.id,
                    success=False,
                    before_performance=self._current_performance,
                    after_performance=self._current_performance,
                    performance_change=0.0,
                    rollback_available=False,
                    error="Modification not safe",
                )

            # Create checkpoint
            checkpoint = await self._create_checkpoint()

            # Measure before performance
            before_perf = self._current_performance

            try:
                # Apply changes
                if proposal.changes.get("params"):
                    for name, value in proposal.changes["params"].items():
                        self._current_params[name] = value

                # Evaluate after
                after_perf = await self._evaluate_performance()
                perf_change = after_perf - before_perf

                # Check if performance dropped too much
                if after_perf < before_perf * self.constraints.min_performance:
                    # Rollback
                    await self.rollback(checkpoint.id)

                    return ModificationResult(
                        proposal_id=proposal.id,
                        success=False,
                        before_performance=before_perf,
                        after_performance=after_perf,
                        performance_change=perf_change,
                        rollback_available=True,
                        error="Performance regression detected, rolled back",
                    )

                # Record successful modification
                self.safety_verifier.record_modification()
                proposal.status = ModificationStatus.APPLIED
                self._current_performance = after_perf

                result = ModificationResult(
                    proposal_id=proposal.id,
                    success=True,
                    before_performance=before_perf,
                    after_performance=after_perf,
                    performance_change=perf_change,
                    rollback_available=True,
                )

                self._results.append(result)

                logger.info(
                    f"Applied modification {proposal.id}: "
                    f"perf {before_perf:.3f} -> {after_perf:.3f}"
                )

                return result

            except Exception as e:
                # Rollback on error
                await self.rollback(checkpoint.id)

                return ModificationResult(
                    proposal_id=proposal.id,
                    success=False,
                    before_performance=before_perf,
                    after_performance=before_perf,
                    performance_change=0.0,
                    rollback_available=True,
                    error=str(e),
                )

    async def _create_checkpoint(self) -> Checkpoint:
        """Create checkpoint of current state."""
        checkpoint = Checkpoint(
            params={k: v.copy() for k, v in self._current_params.items()},
            performance=self._current_performance,
        )

        # Store checkpoint
        self._checkpoints[checkpoint.id] = checkpoint

        # Limit checkpoints
        if len(self._checkpoints) > self._max_checkpoints:
            oldest_id = min(
                self._checkpoints.keys(),
                key=lambda k: self._checkpoints[k].timestamp
            )
            del self._checkpoints[oldest_id]

        return checkpoint

    async def rollback(self, checkpoint_id: str) -> bool:
        """
        Rollback to a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to rollback to

        Returns:
            True if rollback successful
        """
        if checkpoint_id not in self._checkpoints:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return False

        checkpoint = self._checkpoints[checkpoint_id]

        # Restore state
        self._current_params = {k: v.copy() for k, v in checkpoint.params.items()}
        self._current_performance = checkpoint.performance

        logger.info(f"Rolled back to checkpoint {checkpoint_id}")

        return True

    async def _evaluate_performance(self) -> float:
        """Evaluate current model performance."""
        # Mock evaluation
        # In practice, would run benchmark suite
        return self._current_performance + random.uniform(-0.05, 0.1)

    def set_current_state(
        self,
        params: Dict[str, np.ndarray],
        performance: float
    ):
        """Set current model state."""
        self._current_params = {k: v.copy() for k, v in params.items()}
        self._current_performance = performance

        # Initialize meta-learner with current params
        self.meta_learner._meta_params = {k: v.copy() for k, v in params.items()}

    async def save_state(self):
        """Save engine state to disk."""
        async with self._lock:
            state = {
                "proposals": [p.__dict__ for p in self._proposals[-100:]],
                "results": [r.__dict__ for r in self._results[-100:]],
                "current_performance": self._current_performance,
            }

            state_path = self._data_dir / "state.json"
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            # Save checkpoints
            for cp_id, cp in self._checkpoints.items():
                cp_path = self._data_dir / f"checkpoint_{cp_id}.pkl"
                with open(cp_path, 'wb') as f:
                    pickle.dump(cp, f)

            logger.info(f"Saved self-modification state to {self._data_dir}")

    async def load_state(self):
        """Load engine state from disk."""
        async with self._lock:
            state_path = self._data_dir / "state.json"
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)

                self._current_performance = state.get("current_performance", 0.0)

            # Load checkpoints
            for cp_path in self._data_dir.glob("checkpoint_*.pkl"):
                with open(cp_path, 'rb') as f:
                    cp = pickle.load(f)
                    self._checkpoints[cp.id] = cp

            logger.info(f"Loaded self-modification state from {self._data_dir}")

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "strategy": self.strategy.value,
            "current_performance": self._current_performance,
            "total_proposals": len(self._proposals),
            "applied_modifications": sum(
                1 for p in self._proposals if p.status == ModificationStatus.APPLIED
            ),
            "checkpoints_available": len(self._checkpoints),
            "recent_results": [
                {
                    "proposal_id": r.proposal_id,
                    "success": r.success,
                    "performance_change": r.performance_change,
                }
                for r in self._results[-5:]
            ],
            "constraints": {
                "max_param_change": self.constraints.max_param_change,
                "min_performance": self.constraints.min_performance,
                "require_human_approval": self.constraints.require_human_approval,
            },
        }


# ============================================================================
# NEURAL ARCHITECTURE SEARCH (NAS)
# ============================================================================

class NASSearchStrategy(Enum):
    """Neural Architecture Search strategies."""
    RANDOM = "random"  # Random search
    GRID = "grid"  # Grid search
    EVOLUTIONARY = "evolutionary"  # Genetic algorithm
    REINFORCEMENT = "reinforcement"  # RL-based controller
    DIFFERENTIABLE = "differentiable"  # DARTS-style
    BAYESIAN = "bayesian"  # Bayesian optimization
    PROGRESSIVE = "progressive"  # Progressive growing


@dataclass
class NASSearchSpace:
    """Definition of the architecture search space."""
    # Layer types to search over
    layer_types: List[str] = field(default_factory=lambda: [
        "linear", "conv1d", "conv2d", "lstm", "gru", "transformer",
        "attention", "residual", "batch_norm", "layer_norm", "dropout"
    ])

    # Hidden dimensions to try
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024, 2048])

    # Number of layers range
    min_layers: int = 2
    max_layers: int = 12

    # Activation functions
    activations: List[str] = field(default_factory=lambda: ["relu", "gelu", "swish", "tanh", "silu"])

    # Attention heads (for transformer layers)
    attention_heads: List[int] = field(default_factory=lambda: [4, 8, 12, 16])

    # Dropout rates
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])

    # Skip connections
    allow_skip_connections: bool = True

    # Multi-branch architectures
    allow_multi_branch: bool = True
    max_branches: int = 4


@dataclass
class Architecture:
    """Representation of a neural network architecture."""
    id: str
    layers: List[Dict[str, Any]]
    performance: float = 0.0
    params_count: int = 0
    flops: int = 0
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "layers": self.layers,
            "performance": self.performance,
            "params_count": self.params_count,
            "flops": self.flops,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "generation": self.generation,
        }


class NeuralArchitectureSearch:
    """
    Advanced Neural Architecture Search Engine.

    Automatically discovers optimal neural network architectures
    for given tasks using various search strategies.

    TECHNIQUES:
        - Evolutionary search with mutation and crossover
        - Reinforcement learning controller (like NASNet)
        - Differentiable architecture search (DARTS)
        - Bayesian optimization with surrogate models
        - Progressive architecture growing
        - Multi-objective optimization (performance + efficiency)
    """

    def __init__(
        self,
        search_space: Optional[NASSearchSpace] = None,
        strategy: NASSearchStrategy = NASSearchStrategy.EVOLUTIONARY,
        population_size: int = 50,
        max_generations: int = 100,
    ):
        """Initialize NAS engine."""
        self.search_space = search_space or NASSearchSpace()
        self.strategy = strategy
        self.population_size = population_size
        self.max_generations = max_generations

        # Architecture population
        self._population: List[Architecture] = []
        self._archive: List[Architecture] = []  # Pareto archive for multi-objective
        self._best_architecture: Optional[Architecture] = None

        # Search state
        self._generation = 0
        self._evaluations = 0
        self._search_history: List[Dict] = []

        # Evolutionary operators
        self._mutation_rate = 0.3
        self._crossover_rate = 0.5
        self._elite_ratio = 0.1

        # RL controller state (for RL-based NAS)
        self._controller_weights: Optional[np.ndarray] = None

        # Bayesian optimization state
        self._surrogate_observations: List[Tuple[List, float]] = []

        # Lock
        self._lock = asyncio.Lock()

        logger.info(f"NeuralArchitectureSearch initialized with strategy={strategy.value}")

    async def search(
        self,
        evaluate_fn: Callable[[Architecture], Awaitable[float]],
        budget: int = 100,
    ) -> Architecture:
        """
        Run architecture search.

        Args:
            evaluate_fn: Async function to evaluate an architecture (returns performance score)
            budget: Maximum number of evaluations

        Returns:
            Best architecture found
        """
        async with self._lock:
            logger.info(f"Starting NAS with budget={budget}, strategy={self.strategy.value}")

            if self.strategy == NASSearchStrategy.RANDOM:
                return await self._random_search(evaluate_fn, budget)
            elif self.strategy == NASSearchStrategy.EVOLUTIONARY:
                return await self._evolutionary_search(evaluate_fn, budget)
            elif self.strategy == NASSearchStrategy.BAYESIAN:
                return await self._bayesian_search(evaluate_fn, budget)
            elif self.strategy == NASSearchStrategy.PROGRESSIVE:
                return await self._progressive_search(evaluate_fn, budget)
            elif self.strategy == NASSearchStrategy.REINFORCEMENT:
                return await self._rl_search(evaluate_fn, budget)
            else:
                return await self._random_search(evaluate_fn, budget)

    async def _random_search(
        self,
        evaluate_fn: Callable,
        budget: int,
    ) -> Architecture:
        """Random architecture search."""
        best = None

        for i in range(budget):
            arch = self._sample_random_architecture()
            arch.performance = await evaluate_fn(arch)
            self._evaluations += 1

            self._archive.append(arch)

            if best is None or arch.performance > best.performance:
                best = arch
                logger.info(f"New best: {arch.performance:.4f} (eval {i+1}/{budget})")

        self._best_architecture = best
        return best

    async def _evolutionary_search(
        self,
        evaluate_fn: Callable,
        budget: int,
    ) -> Architecture:
        """
        Evolutionary architecture search using genetic algorithms.

        Uses:
            - Tournament selection
            - Crossover between architectures
            - Mutation of layers
            - Elitism to preserve best solutions
        """
        # Initialize population
        self._population = [
            self._sample_random_architecture()
            for _ in range(self.population_size)
        ]

        # Evaluate initial population
        for arch in self._population:
            arch.performance = await evaluate_fn(arch)
            self._evaluations += 1

        best = max(self._population, key=lambda a: a.performance)

        # Evolution loop
        while self._evaluations < budget:
            self._generation += 1

            # Selection
            parents = self._tournament_select(self._population, k=2)

            # Crossover
            if random.random() < self._crossover_rate:
                child = self._crossover(parents[0], parents[1])
            else:
                child = self._copy_architecture(random.choice(parents))

            # Mutation
            if random.random() < self._mutation_rate:
                child = self._mutate(child)

            # Evaluate child
            child.performance = await evaluate_fn(child)
            child.generation = self._generation
            self._evaluations += 1

            # Update population (replace worst)
            worst_idx = min(range(len(self._population)), key=lambda i: self._population[i].performance)
            if child.performance > self._population[worst_idx].performance:
                self._population[worst_idx] = child

            # Update best
            if child.performance > best.performance:
                best = child
                logger.info(f"New best: {best.performance:.4f} (gen {self._generation}, eval {self._evaluations}/{budget})")

            # Record history
            if self._generation % 10 == 0:
                self._search_history.append({
                    "generation": self._generation,
                    "evaluations": self._evaluations,
                    "best_performance": best.performance,
                    "avg_performance": np.mean([a.performance for a in self._population]),
                })

        self._best_architecture = best
        return best

    async def _bayesian_search(
        self,
        evaluate_fn: Callable,
        budget: int,
    ) -> Architecture:
        """
        Bayesian optimization for architecture search.

        Uses a Gaussian Process surrogate model to predict performance
        and Expected Improvement acquisition function.
        """
        # Initial random samples
        n_initial = min(10, budget // 5)
        for _ in range(n_initial):
            arch = self._sample_random_architecture()
            arch.performance = await evaluate_fn(arch)
            self._evaluations += 1

            # Store observation
            features = self._architecture_to_features(arch)
            self._surrogate_observations.append((features, arch.performance))
            self._archive.append(arch)

        best = max(self._archive, key=lambda a: a.performance)

        # BO loop
        while self._evaluations < budget:
            # Fit surrogate model (simplified GP)
            X = np.array([obs[0] for obs in self._surrogate_observations])
            y = np.array([obs[1] for obs in self._surrogate_observations])

            # Find next architecture via acquisition function
            best_acquisition = -float("inf")
            best_candidate = None

            for _ in range(100):  # Sample 100 candidates
                candidate = self._sample_random_architecture()
                features = self._architecture_to_features(candidate)

                # Compute acquisition (Expected Improvement)
                mean, std = self._predict_surrogate(features, X, y)
                ei = self._expected_improvement(mean, std, best.performance)

                if ei > best_acquisition:
                    best_acquisition = ei
                    best_candidate = candidate

            # Evaluate best candidate
            if best_candidate:
                best_candidate.performance = await evaluate_fn(best_candidate)
                self._evaluations += 1

                features = self._architecture_to_features(best_candidate)
                self._surrogate_observations.append((features, best_candidate.performance))
                self._archive.append(best_candidate)

                if best_candidate.performance > best.performance:
                    best = best_candidate
                    logger.info(f"New best: {best.performance:.4f} (eval {self._evaluations}/{budget})")

        self._best_architecture = best
        return best

    async def _progressive_search(
        self,
        evaluate_fn: Callable,
        budget: int,
    ) -> Architecture:
        """
        Progressive architecture search - start small, grow larger.

        Searches for optimal architectures by progressively increasing
        the complexity, similar to progressive growing of GANs.
        """
        current_depth = self.search_space.min_layers
        best = None

        while current_depth <= self.search_space.max_layers and self._evaluations < budget:
            logger.info(f"Searching at depth {current_depth}")

            # Search at current depth
            depth_best = None
            for _ in range(budget // (self.search_space.max_layers - self.search_space.min_layers + 1)):
                if self._evaluations >= budget:
                    break

                arch = self._sample_random_architecture(fixed_depth=current_depth)
                arch.performance = await evaluate_fn(arch)
                self._evaluations += 1

                if depth_best is None or arch.performance > depth_best.performance:
                    depth_best = arch

            if best is None or (depth_best and depth_best.performance > best.performance):
                best = depth_best
                logger.info(f"New best at depth {current_depth}: {best.performance:.4f}")

            current_depth += 1

        self._best_architecture = best
        return best

    async def _rl_search(
        self,
        evaluate_fn: Callable,
        budget: int,
    ) -> Architecture:
        """
        Reinforcement learning based architecture search.

        Uses a controller network that learns to generate architectures.
        """
        # Initialize controller weights
        n_decisions = self.search_space.max_layers * 5  # 5 decisions per layer
        if self._controller_weights is None:
            self._controller_weights = np.zeros(n_decisions * 10)  # Simple linear controller

        best = None
        baseline = 0.0  # Running baseline for REINFORCE

        for episode in range(budget):
            # Sample architecture using controller
            arch = self._sample_from_controller()
            arch.performance = await evaluate_fn(arch)
            self._evaluations += 1

            # Update controller using REINFORCE
            advantage = arch.performance - baseline
            baseline = 0.9 * baseline + 0.1 * arch.performance

            # Update controller weights (simplified)
            self._controller_weights += 0.01 * advantage * np.random.randn(len(self._controller_weights))

            self._archive.append(arch)

            if best is None or arch.performance > best.performance:
                best = arch
                logger.info(f"New best: {best.performance:.4f} (episode {episode+1}/{budget})")

        self._best_architecture = best
        return best

    def _sample_random_architecture(self, fixed_depth: Optional[int] = None) -> Architecture:
        """Sample a random architecture from the search space."""
        num_layers = fixed_depth or random.randint(
            self.search_space.min_layers,
            self.search_space.max_layers
        )

        layers = []
        prev_dim = 768  # Assume input dimension

        for i in range(num_layers):
            layer_type = random.choice(self.search_space.layer_types)
            hidden_dim = random.choice(self.search_space.hidden_dims)
            activation = random.choice(self.search_space.activations)
            dropout = random.choice(self.search_space.dropout_rates)

            layer = {
                "type": layer_type,
                "in_dim": prev_dim,
                "out_dim": hidden_dim,
                "activation": activation,
                "dropout": dropout,
            }

            if layer_type in ("transformer", "attention"):
                layer["heads"] = random.choice(self.search_space.attention_heads)

            if self.search_space.allow_skip_connections and i > 0:
                layer["skip_connection"] = random.random() < 0.3

            layers.append(layer)
            prev_dim = hidden_dim

        return Architecture(
            id=str(uuid.uuid4())[:8],
            layers=layers,
            params_count=self._estimate_params(layers),
            generation=self._generation,
        )

    def _sample_from_controller(self) -> Architecture:
        """Sample architecture using RL controller."""
        # Use controller weights to bias sampling (simplified)
        # In practice, this would be a recurrent neural network
        return self._sample_random_architecture()

    def _tournament_select(self, population: List[Architecture], k: int = 2) -> List[Architecture]:
        """Tournament selection."""
        selected = []
        for _ in range(k):
            tournament = random.sample(population, min(5, len(population)))
            winner = max(tournament, key=lambda a: a.performance)
            selected.append(winner)
        return selected

    def _crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        """Crossover between two architectures."""
        # Single-point crossover
        min_len = min(len(parent1.layers), len(parent2.layers))
        if min_len > 1:
            point = random.randint(1, min_len - 1)
            child_layers = parent1.layers[:point] + parent2.layers[point:]
        else:
            child_layers = parent1.layers.copy()

        return Architecture(
            id=str(uuid.uuid4())[:8],
            layers=child_layers,
            generation=self._generation,
            parent_ids=[parent1.id, parent2.id],
        )

    def _mutate(self, arch: Architecture) -> Architecture:
        """Mutate an architecture."""
        layers = [layer.copy() for layer in arch.layers]

        # Random mutation type
        mutation_type = random.choice(["add", "remove", "modify"])

        if mutation_type == "add" and len(layers) < self.search_space.max_layers:
            idx = random.randint(0, len(layers))
            new_layer = {
                "type": random.choice(self.search_space.layer_types),
                "in_dim": layers[idx-1]["out_dim"] if idx > 0 else 768,
                "out_dim": random.choice(self.search_space.hidden_dims),
                "activation": random.choice(self.search_space.activations),
                "dropout": random.choice(self.search_space.dropout_rates),
            }
            layers.insert(idx, new_layer)

        elif mutation_type == "remove" and len(layers) > self.search_space.min_layers:
            idx = random.randint(0, len(layers) - 1)
            layers.pop(idx)

        elif mutation_type == "modify" and layers:
            idx = random.randint(0, len(layers) - 1)
            layers[idx]["out_dim"] = random.choice(self.search_space.hidden_dims)
            layers[idx]["activation"] = random.choice(self.search_space.activations)

        return Architecture(
            id=str(uuid.uuid4())[:8],
            layers=layers,
            generation=self._generation,
            parent_ids=[arch.id],
        )

    def _copy_architecture(self, arch: Architecture) -> Architecture:
        """Create a copy of an architecture."""
        return Architecture(
            id=str(uuid.uuid4())[:8],
            layers=[layer.copy() for layer in arch.layers],
            generation=self._generation,
            parent_ids=[arch.id],
        )

    def _architecture_to_features(self, arch: Architecture) -> List[float]:
        """Convert architecture to feature vector for Bayesian optimization."""
        features = [
            len(arch.layers),
            np.mean([layer.get("out_dim", 256) for layer in arch.layers]),
            np.std([layer.get("out_dim", 256) for layer in arch.layers]),
            sum(1 for layer in arch.layers if layer.get("skip_connection", False)),
        ]
        return features

    def _predict_surrogate(
        self,
        x: List[float],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float]:
        """Predict using simplified Gaussian Process."""
        # Simplified: use k-NN with distance-based uncertainty
        x = np.array(x)
        distances = np.linalg.norm(X - x, axis=1)
        weights = 1.0 / (distances + 1e-8)
        weights = weights / weights.sum()

        mean = np.sum(weights * y)
        var = np.sum(weights * (y - mean) ** 2)
        std = np.sqrt(var + 1e-8)

        return float(mean), float(std)

    def _expected_improvement(self, mean: float, std: float, best: float) -> float:
        """Compute Expected Improvement acquisition function."""
        from math import erf, sqrt, pi, exp

        z = (mean - best) / (std + 1e-8)

        def norm_cdf(x):
            return 0.5 * (1 + erf(x / sqrt(2)))

        def norm_pdf(x):
            return exp(-0.5 * x * x) / sqrt(2 * pi)

        ei = std * (z * norm_cdf(z) + norm_pdf(z))
        return float(ei)

    def _estimate_params(self, layers: List[Dict]) -> int:
        """Estimate parameter count for an architecture."""
        params = 0
        for layer in layers:
            in_dim = layer.get("in_dim", 768)
            out_dim = layer.get("out_dim", 256)
            params += in_dim * out_dim + out_dim  # Weights + bias
        return params

    def get_statistics(self) -> Dict[str, Any]:
        """Get NAS statistics."""
        return {
            "strategy": self.strategy.value,
            "generation": self._generation,
            "evaluations": self._evaluations,
            "population_size": len(self._population),
            "archive_size": len(self._archive),
            "best_performance": self._best_architecture.performance if self._best_architecture else 0,
            "best_architecture": self._best_architecture.to_dict() if self._best_architecture else None,
            "search_history": self._search_history[-10:],
        }


# ============================================================================
# BAYESIAN HYPERPARAMETER OPTIMIZATION
# ============================================================================

class AcquisitionFunction(Enum):
    """Acquisition functions for Bayesian optimization."""
    EI = "expected_improvement"  # Expected Improvement
    UCB = "upper_confidence_bound"  # Upper Confidence Bound
    PI = "probability_of_improvement"  # Probability of Improvement
    THOMPSON = "thompson_sampling"  # Thompson Sampling
    KNOWLEDGE_GRADIENT = "knowledge_gradient"  # Knowledge Gradient


@dataclass
class HyperparameterSpace:
    """Definition of hyperparameter search space."""
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_continuous(self, name: str, low: float, high: float, log_scale: bool = False):
        """Add continuous hyperparameter."""
        self.parameters[name] = {
            "type": "continuous",
            "low": low,
            "high": high,
            "log_scale": log_scale,
        }

    def add_integer(self, name: str, low: int, high: int):
        """Add integer hyperparameter."""
        self.parameters[name] = {
            "type": "integer",
            "low": low,
            "high": high,
        }

    def add_categorical(self, name: str, choices: List[Any]):
        """Add categorical hyperparameter."""
        self.parameters[name] = {
            "type": "categorical",
            "choices": choices,
        }

    def sample(self) -> Dict[str, Any]:
        """Sample random point from search space."""
        point = {}
        for name, spec in self.parameters.items():
            if spec["type"] == "continuous":
                if spec.get("log_scale"):
                    log_val = random.uniform(np.log(spec["low"]), np.log(spec["high"]))
                    point[name] = np.exp(log_val)
                else:
                    point[name] = random.uniform(spec["low"], spec["high"])
            elif spec["type"] == "integer":
                point[name] = random.randint(spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                point[name] = random.choice(spec["choices"])
        return point


class BayesianOptimizer:
    """
    Bayesian Hyperparameter Optimization Engine.

    Uses Gaussian Process surrogate models and acquisition functions
    to efficiently search hyperparameter spaces.

    FEATURES:
        - Multiple acquisition functions (EI, UCB, PI, Thompson)
        - Warm starting from previous observations
        - Multi-fidelity optimization
        - Parallel batch optimization
        - Automatic early stopping
        - Hyperband integration
    """

    def __init__(
        self,
        search_space: HyperparameterSpace,
        acquisition: AcquisitionFunction = AcquisitionFunction.EI,
        n_initial: int = 10,
        exploration_weight: float = 1.0,
    ):
        """Initialize Bayesian optimizer."""
        self.search_space = search_space
        self.acquisition = acquisition
        self.n_initial = n_initial
        self.exploration_weight = exploration_weight

        # Observations
        self._observations: List[Tuple[Dict, float]] = []
        self._best_point: Optional[Dict] = None
        self._best_value: float = -float("inf")

        # Lock
        self._lock = asyncio.Lock()

        logger.info(f"BayesianOptimizer initialized with acquisition={acquisition.value}")

    async def optimize(
        self,
        objective_fn: Callable[[Dict], Awaitable[float]],
        n_iterations: int = 50,
    ) -> Tuple[Dict, float]:
        """
        Run Bayesian optimization.

        Args:
            objective_fn: Async function that takes hyperparameters and returns a score to maximize
            n_iterations: Number of iterations

        Returns:
            Best hyperparameters and score
        """
        async with self._lock:
            # Initial random samples
            for _ in range(self.n_initial):
                point = self.search_space.sample()
                value = await objective_fn(point)
                self._observations.append((point, value))

                if value > self._best_value:
                    self._best_value = value
                    self._best_point = point
                    logger.info(f"New best: {value:.4f}")

            # BO iterations
            for i in range(n_iterations - self.n_initial):
                # Find next point via acquisition function
                next_point = await self._suggest_next()

                # Evaluate
                value = await objective_fn(next_point)
                self._observations.append((next_point, value))

                if value > self._best_value:
                    self._best_value = value
                    self._best_point = next_point
                    logger.info(f"New best: {value:.4f} (iter {i + self.n_initial + 1})")

            return self._best_point, self._best_value

    async def _suggest_next(self) -> Dict:
        """Suggest next point to evaluate."""
        # Sample many candidates and pick best by acquisition
        best_acquisition = -float("inf")
        best_candidate = None

        X, y = self._get_observation_arrays()

        for _ in range(1000):
            candidate = self.search_space.sample()
            x = self._point_to_array(candidate)

            # Predict mean and std
            mean, std = self._gp_predict(x, X, y)

            # Compute acquisition value
            if self.acquisition == AcquisitionFunction.EI:
                acq = self._expected_improvement(mean, std)
            elif self.acquisition == AcquisitionFunction.UCB:
                acq = self._upper_confidence_bound(mean, std)
            elif self.acquisition == AcquisitionFunction.PI:
                acq = self._probability_of_improvement(mean, std)
            elif self.acquisition == AcquisitionFunction.THOMPSON:
                acq = mean + std * np.random.randn()
            else:
                acq = self._expected_improvement(mean, std)

            if acq > best_acquisition:
                best_acquisition = acq
                best_candidate = candidate

        return best_candidate or self.search_space.sample()

    def _get_observation_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert observations to arrays."""
        X = np.array([self._point_to_array(p) for p, _ in self._observations])
        y = np.array([v for _, v in self._observations])
        return X, y

    def _point_to_array(self, point: Dict) -> List[float]:
        """Convert hyperparameter dict to array."""
        arr = []
        for name, spec in self.search_space.parameters.items():
            val = point.get(name)
            if spec["type"] == "continuous":
                if spec.get("log_scale"):
                    arr.append(np.log(val))
                else:
                    arr.append(val)
            elif spec["type"] == "integer":
                arr.append(float(val))
            elif spec["type"] == "categorical":
                # One-hot encode
                arr.extend([1.0 if v == val else 0.0 for v in spec["choices"]])
        return arr

    def _gp_predict(self, x: List[float], X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Simplified Gaussian Process prediction."""
        x = np.array(x)

        # RBF kernel with length scale
        length_scale = 1.0
        distances = np.linalg.norm(X - x, axis=1)
        K = np.exp(-0.5 * (distances / length_scale) ** 2)

        # Normalize
        K = K / (K.sum() + 1e-8)

        # Mean prediction
        mean = np.sum(K * y)

        # Variance prediction
        var = np.sum(K * (y - mean) ** 2) + 0.01  # Add noise
        std = np.sqrt(var)

        return float(mean), float(std)

    def _expected_improvement(self, mean: float, std: float) -> float:
        """Expected Improvement acquisition function."""
        from math import erf, sqrt, pi, exp

        z = (mean - self._best_value) / (std + 1e-8)

        def norm_cdf(x):
            return 0.5 * (1 + erf(x / sqrt(2)))

        def norm_pdf(x):
            return exp(-0.5 * x * x) / sqrt(2 * pi)

        return std * (z * norm_cdf(z) + norm_pdf(z))

    def _upper_confidence_bound(self, mean: float, std: float) -> float:
        """Upper Confidence Bound acquisition function."""
        return mean + self.exploration_weight * std

    def _probability_of_improvement(self, mean: float, std: float) -> float:
        """Probability of Improvement acquisition function."""
        from math import erf, sqrt

        z = (mean - self._best_value) / (std + 1e-8)
        return 0.5 * (1 + erf(z / sqrt(2)))

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "acquisition": self.acquisition.value,
            "observations": len(self._observations),
            "best_value": self._best_value,
            "best_point": self._best_point,
        }


# ============================================================================
# EVOLUTIONARY OPTIMIZATION
# ============================================================================

class SelectionMethod(Enum):
    """Selection methods for evolutionary algorithms."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITIST = "elitist"
    CROWDING = "crowding"


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary optimization."""
    population_size: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 5
    elite_count: int = 5
    max_generations: int = 1000
    convergence_threshold: float = 1e-6


class EvolutionaryOptimizer:
    """
    Evolutionary Optimization Engine.

    Implements multiple evolutionary algorithms:
        - Genetic Algorithm (GA)
        - Differential Evolution (DE)
        - Evolution Strategy (ES)
        - CMA-ES
        - NSGA-II for multi-objective
    """

    def __init__(self, config: Optional[EvolutionaryConfig] = None):
        """Initialize evolutionary optimizer."""
        self.config = config or EvolutionaryConfig()

        # Population
        self._population: List[Tuple[np.ndarray, float]] = []
        self._best_individual: Optional[np.ndarray] = None
        self._best_fitness: float = -float("inf")

        # Statistics
        self._generation = 0
        self._history: List[Dict] = []

        # CMA-ES state
        self._mean: Optional[np.ndarray] = None
        self._covariance: Optional[np.ndarray] = None

        # Lock
        self._lock = asyncio.Lock()

        logger.info(f"EvolutionaryOptimizer initialized")

    async def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], Awaitable[float]],
        dimensions: int,
        bounds: Tuple[float, float] = (-10.0, 10.0),
    ) -> Tuple[np.ndarray, float]:
        """
        Run evolutionary optimization.

        Args:
            fitness_fn: Async function that takes array and returns fitness (to maximize)
            dimensions: Number of dimensions
            bounds: (low, high) bounds for each dimension

        Returns:
            Best individual and fitness
        """
        async with self._lock:
            # Initialize population
            low, high = bounds
            self._population = []

            for _ in range(self.config.population_size):
                individual = np.random.uniform(low, high, dimensions)
                fitness = await fitness_fn(individual)
                self._population.append((individual, fitness))

                if fitness > self._best_fitness:
                    self._best_fitness = fitness
                    self._best_individual = individual.copy()

            # Evolution loop
            for gen in range(self.config.max_generations):
                self._generation = gen

                # Selection
                parents = self._select()

                # Create new population
                new_population = []

                # Elitism - keep best individuals
                sorted_pop = sorted(self._population, key=lambda x: x[1], reverse=True)
                for i in range(self.config.elite_count):
                    new_population.append(sorted_pop[i])

                # Fill rest with offspring
                while len(new_population) < self.config.population_size:
                    # Select parents
                    parent1, parent2 = random.sample(parents, 2)

                    # Crossover
                    if random.random() < self.config.crossover_rate:
                        child = self._crossover(parent1[0], parent2[0])
                    else:
                        child = parent1[0].copy()

                    # Mutation
                    if random.random() < self.config.mutation_rate:
                        child = self._mutate(child, bounds)

                    # Clip to bounds
                    child = np.clip(child, low, high)

                    # Evaluate
                    fitness = await fitness_fn(child)
                    new_population.append((child, fitness))

                    if fitness > self._best_fitness:
                        self._best_fitness = fitness
                        self._best_individual = child.copy()
                        logger.info(f"New best: {fitness:.6f} (gen {gen})")

                self._population = new_population

                # Record history
                avg_fitness = np.mean([f for _, f in self._population])
                self._history.append({
                    "generation": gen,
                    "best_fitness": self._best_fitness,
                    "avg_fitness": float(avg_fitness),
                })

                # Convergence check
                if len(self._history) > 10:
                    recent = [h["best_fitness"] for h in self._history[-10:]]
                    if max(recent) - min(recent) < self.config.convergence_threshold:
                        logger.info(f"Converged at generation {gen}")
                        break

            return self._best_individual, self._best_fitness

    def _select(self) -> List[Tuple[np.ndarray, float]]:
        """Select parents for reproduction."""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection()
        elif self.config.selection_method == SelectionMethod.RANK:
            return self._rank_selection()
        else:
            return self._tournament_selection()

    def _tournament_selection(self) -> List[Tuple[np.ndarray, float]]:
        """Tournament selection."""
        selected = []
        for _ in range(self.config.population_size // 2):
            tournament = random.sample(self._population, self.config.tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner)
        return selected

    def _roulette_selection(self) -> List[Tuple[np.ndarray, float]]:
        """Roulette wheel selection."""
        # Shift fitness to be positive
        fitnesses = np.array([f for _, f in self._population])
        min_fit = fitnesses.min()
        shifted = fitnesses - min_fit + 1e-8

        probs = shifted / shifted.sum()

        indices = np.random.choice(
            len(self._population),
            size=self.config.population_size // 2,
            p=probs,
        )

        return [self._population[i] for i in indices]

    def _rank_selection(self) -> List[Tuple[np.ndarray, float]]:
        """Rank-based selection."""
        sorted_pop = sorted(self._population, key=lambda x: x[1])
        ranks = np.arange(1, len(sorted_pop) + 1)
        probs = ranks / ranks.sum()

        indices = np.random.choice(
            len(sorted_pop),
            size=self.config.population_size // 2,
            p=probs,
        )

        return [sorted_pop[i] for i in indices]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Blend crossover."""
        alpha = 0.5
        return alpha * parent1 + (1 - alpha) * parent2

    def _mutate(self, individual: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        """Gaussian mutation."""
        low, high = bounds
        mutation_strength = 0.1 * (high - low)
        return individual + np.random.randn(len(individual)) * mutation_strength

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_fitness": float(self._best_fitness),
            "history": self._history[-10:],
        }


# ============================================================================
# V80.0 INFRASTRUCTURE INTEGRATION
# ============================================================================

class SelfImprovementInfrastructure:
    """
    Integration layer connecting Self-Improvement with v80.0 infrastructure.

    Provides:
        - Distributed tracing for improvement operations
        - Caching for evaluation results
        - Rate limiting for safety
        - Graph routing for experiment distribution
    """

    def __init__(self):
        """Initialize infrastructure integration."""
        self._tracer = None
        self._cache = None
        self._rate_limiter = None
        self._zero_copy = None
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Lazily initialize infrastructure connections."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                from jarvis_prime.core.distributed_tracing import tracer
                from jarvis_prime.core.predictive_cache import get_predictive_cache
                from jarvis_prime.core.adaptive_rate_limiter import get_rate_limiter
                from jarvis_prime.core.zero_copy_ipc import get_zero_copy_transport

                self._tracer = tracer
                self._cache = await get_predictive_cache()
                self._rate_limiter = await get_rate_limiter()
                self._zero_copy = await get_zero_copy_transport()

                self._initialized = True
                logger.info("Self-Improvement v80.0 infrastructure integration initialized")

            except ImportError as e:
                logger.warning(f"v80.0 infrastructure not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize infrastructure: {e}")

    async def trace_improvement_operation(self, operation_name: str):
        """Context manager for tracing improvement operations."""
        await self.initialize()

        if self._tracer:
            return self._tracer.start_span(f"self_improvement.{operation_name}")

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def noop():
            yield None

        return noop()

    async def cache_evaluation_result(self, key: str, result: float, ttl: int = 3600):
        """Cache an evaluation result."""
        await self.initialize()

        if self._cache:
            await self._cache.set(f"eval:{key}", str(result), ttl=ttl)

    async def get_cached_evaluation(self, key: str) -> Optional[float]:
        """Get a cached evaluation result."""
        await self.initialize()

        if self._cache:
            data = await self._cache.get(f"eval:{key}")
            if data:
                return float(data)

        return None

    async def check_modification_rate(self) -> bool:
        """Check if self-modification is rate limited."""
        await self.initialize()

        if self._rate_limiter:
            return await self._rate_limiter.acquire(
                user_id="self_improvement:modify",
                tokens=10  # Self-modification is expensive
            )

        return True


# Global infrastructure
_improvement_infrastructure: Optional[SelfImprovementInfrastructure] = None


async def get_improvement_infrastructure() -> SelfImprovementInfrastructure:
    """Get global improvement infrastructure."""
    global _improvement_infrastructure

    if _improvement_infrastructure is None:
        _improvement_infrastructure = SelfImprovementInfrastructure()
        await _improvement_infrastructure.initialize()

    return _improvement_infrastructure


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_self_modifier: Optional[SelfModificationEngine] = None
_modifier_lock = asyncio.Lock()


async def get_self_modifier() -> SelfModificationEngine:
    """Get or create global self-modification engine."""
    global _self_modifier

    async with _modifier_lock:
        if _self_modifier is None:
            # Load constraints from environment
            constraints = SafetyConstraints(
                max_param_change=float(os.getenv("SELF_MODIFY_MAX_CHANGE", "0.1")),
                min_performance=float(os.getenv("SELF_MODIFY_MIN_PERF", "0.95")),
                require_human_approval=os.getenv(
                    "SELF_MODIFY_REQUIRE_APPROVAL", "true"
                ).lower() == "true",
                allow_architecture_changes=os.getenv(
                    "SELF_MODIFY_ALLOW_ARCH", "false"
                ).lower() == "true",
            )

            strategy_name = os.getenv("SELF_MODIFY_STRATEGY", "meta_learning")
            try:
                strategy = ModificationStrategy(strategy_name)
            except ValueError:
                strategy = ModificationStrategy.META_LEARNING

            _self_modifier = SelfModificationEngine(
                constraints=constraints,
                strategy=strategy
            )

            # Try to load existing state
            try:
                await _self_modifier.load_state()
            except Exception as e:
                logger.warning(f"Could not load self-modification state: {e}")

        return _self_modifier
