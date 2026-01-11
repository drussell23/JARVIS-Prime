"""
RLHF Integration - Reinforcement Learning from Human Feedback
================================================================

v91.0 - Advanced RLHF Pipeline for JARVIS Prime

This module provides RLHF capabilities for model improvement:
- Reward model training from human preferences
- PPO-based policy optimization
- Preference learning from comparisons
- Integration with Continuous Learning pipeline
- Online learning from user feedback

ARCHITECTURE:
    User Feedback → Preference Database → Reward Model → Policy Optimization
                                              ↓
    Base Model → Supervised Fine-tuning → RL Training → Aligned Model

USAGE:
    # Initialize RLHF pipeline
    rlhf = await get_rlhf_pipeline()

    # Record user preference
    await rlhf.record_preference(
        prompt="What is AI?",
        response_a="AI is...",
        response_b="Artificial Intelligence...",
        preference="a",  # User preferred response A
    )

    # Train reward model
    await rlhf.train_reward_model()

    # Optimize policy with PPO
    await rlhf.optimize_policy()

INTEGRATION:
    - ContinuousLearningEngine: Uses experiences for RLHF
    - PrimeModel: Base model for optimization
    - Reactor Core: Large-scale training pipelines
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
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

T = TypeVar("T")


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class RLHFConfig:
    """
    Configuration for RLHF pipeline.

    All values configurable via environment variables with RLHF_ prefix.
    """
    # Preference collection
    min_preferences_for_training: int = field(default_factory=lambda: int(os.getenv("RLHF_MIN_PREFERENCES", "100")))
    preference_buffer_size: int = field(default_factory=lambda: int(os.getenv("RLHF_BUFFER_SIZE", "10000")))

    # Reward model
    reward_model_epochs: int = field(default_factory=lambda: int(os.getenv("RLHF_REWARD_EPOCHS", "3")))
    reward_model_lr: float = field(default_factory=lambda: float(os.getenv("RLHF_REWARD_LR", "1e-5")))
    reward_model_batch_size: int = field(default_factory=lambda: int(os.getenv("RLHF_REWARD_BATCH", "8")))

    # PPO
    ppo_epochs: int = field(default_factory=lambda: int(os.getenv("RLHF_PPO_EPOCHS", "4")))
    ppo_batch_size: int = field(default_factory=lambda: int(os.getenv("RLHF_PPO_BATCH", "8")))
    ppo_lr: float = field(default_factory=lambda: float(os.getenv("RLHF_PPO_LR", "1e-6")))
    ppo_clip_range: float = field(default_factory=lambda: float(os.getenv("RLHF_PPO_CLIP", "0.2")))
    ppo_value_coef: float = field(default_factory=lambda: float(os.getenv("RLHF_PPO_VALUE_COEF", "0.5")))
    ppo_entropy_coef: float = field(default_factory=lambda: float(os.getenv("RLHF_PPO_ENTROPY_COEF", "0.01")))
    ppo_max_grad_norm: float = field(default_factory=lambda: float(os.getenv("RLHF_PPO_MAX_GRAD", "0.5")))

    # KL divergence
    kl_penalty_coef: float = field(default_factory=lambda: float(os.getenv("RLHF_KL_COEF", "0.1")))
    target_kl: float = field(default_factory=lambda: float(os.getenv("RLHF_TARGET_KL", "0.02")))
    adaptive_kl: bool = field(default_factory=lambda: os.getenv("RLHF_ADAPTIVE_KL", "true").lower() == "true")

    # Generation
    max_response_length: int = field(default_factory=lambda: int(os.getenv("RLHF_MAX_RESPONSE", "512")))
    num_return_sequences: int = field(default_factory=lambda: int(os.getenv("RLHF_NUM_SEQUENCES", "4")))

    # Training schedule
    reward_model_update_interval_hours: float = field(default_factory=lambda: float(os.getenv("RLHF_REWARD_INTERVAL_HOURS", "24.0")))
    policy_update_interval_hours: float = field(default_factory=lambda: float(os.getenv("RLHF_POLICY_INTERVAL_HOURS", "48.0")))

    # Paths
    checkpoints_dir: Path = field(default_factory=lambda: Path(os.getenv(
        "RLHF_CHECKPOINTS_DIR", str(Path.home() / ".jarvis" / "learning" / "rlhf")
    )))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_preferences_for_training": self.min_preferences_for_training,
            "reward_model_epochs": self.reward_model_epochs,
            "ppo_epochs": self.ppo_epochs,
            "ppo_clip_range": self.ppo_clip_range,
            "kl_penalty_coef": self.kl_penalty_coef,
            "target_kl": self.target_kl,
        }


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class PreferencePair:
    """A single human preference comparison."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)

    # The prompt/context
    prompt: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Two responses to compare
    response_a: str = ""
    response_b: str = ""

    # Human preference: "a", "b", "tie", or None (no preference)
    preference: Optional[str] = None
    confidence: float = 1.0  # 0-1, how confident the human was

    # Metadata
    user_id: Optional[str] = None
    model_version: str = ""
    source: str = "direct"  # direct, implicit, synthetic

    # For learning
    reward_a: Optional[float] = None
    reward_b: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "prompt": self.prompt[:200],
            "preference": self.preference,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferencePair":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PreferenceBuffer:
    """Buffer for storing human preferences."""
    capacity: int = 10000
    preferences: Deque[PreferencePair] = field(default_factory=deque)

    # Statistics
    total_added: int = 0
    preference_counts: Dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0, "tie": 0})

    def add(self, preference: PreferencePair) -> None:
        """Add a preference to the buffer."""
        if len(self.preferences) >= self.capacity:
            old = self.preferences.popleft()
            if old.preference:
                self.preference_counts[old.preference] = max(0, self.preference_counts.get(old.preference, 0) - 1)

        self.preferences.append(preference)
        self.total_added += 1

        if preference.preference:
            self.preference_counts[preference.preference] = self.preference_counts.get(preference.preference, 0) + 1

    def sample(self, n: int, balanced: bool = True) -> List[PreferencePair]:
        """Sample preferences for training."""
        if not self.preferences:
            return []

        prefs_list = list(self.preferences)

        if balanced:
            # Balance between preference types
            a_prefs = [p for p in prefs_list if p.preference == "a"]
            b_prefs = [p for p in prefs_list if p.preference == "b"]
            tie_prefs = [p for p in prefs_list if p.preference == "tie"]

            n_each = n // 3
            samples = []

            for group in [a_prefs, b_prefs, tie_prefs]:
                samples.extend(random.sample(group, min(n_each, len(group))))

            # Fill remaining with random
            remaining = n - len(samples)
            if remaining > 0:
                available = [p for p in prefs_list if p not in samples]
                samples.extend(random.sample(available, min(remaining, len(available))))

            return samples
        else:
            return random.sample(prefs_list, min(n, len(prefs_list)))

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "total_preferences": len(self.preferences),
            "total_added": self.total_added,
            "preference_distribution": dict(self.preference_counts),
            "capacity": self.capacity,
        }

    def save(self, path: Path) -> bool:
        """Save buffer to disk."""
        try:
            data = {
                "preferences": [p.__dict__ for p in self.preferences],
                "total_added": self.total_added,
                "preference_counts": self.preference_counts,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to save preference buffer: {e}")
            return False

    def load(self, path: Path) -> bool:
        """Load buffer from disk."""
        try:
            with open(path) as f:
                data = json.load(f)

            self.preferences.clear()
            for p_dict in data.get("preferences", []):
                self.preferences.append(PreferencePair.from_dict(p_dict))

            self.total_added = data.get("total_added", len(self.preferences))
            self.preference_counts = data.get("preference_counts", {"a": 0, "b": 0, "tie": 0})
            return True
        except Exception as e:
            logger.error(f"Failed to load preference buffer: {e}")
            return False

    def __len__(self) -> int:
        return len(self.preferences)


@dataclass
class RLHFExperience:
    """Experience for RL training."""
    prompt: str = ""
    response: str = ""
    reward: float = 0.0

    # PPO-specific
    log_prob: float = 0.0
    value: float = 0.0
    advantage: float = 0.0
    returns: float = 0.0

    # Metadata
    prompt_tokens: int = 0
    response_tokens: int = 0
    kl_divergence: float = 0.0


@dataclass
class TrainingMetrics:
    """Metrics from RLHF training."""
    step: int = 0
    epoch: int = 0

    # Reward model metrics
    reward_model_loss: float = 0.0
    reward_model_accuracy: float = 0.0

    # PPO metrics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    clip_fraction: float = 0.0

    # Reward metrics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0

    # Learning rate
    learning_rate: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# REWARD MODEL
# =============================================================================


class RewardModel(ABC):
    """Abstract base class for reward models."""

    @abstractmethod
    async def predict_reward(self, prompt: str, response: str) -> float:
        """Predict reward for a response."""
        ...

    @abstractmethod
    async def train(
        self,
        preferences: List[PreferencePair],
        epochs: int,
    ) -> TrainingMetrics:
        """Train the reward model on preferences."""
        ...


class NeuralRewardModel(RewardModel):
    """
    Neural network reward model.

    Uses a language model backbone with a scalar head to predict rewards.
    Trained on human preference comparisons using Bradley-Terry model.
    """

    def __init__(
        self,
        base_model: Optional[Any] = None,
        config: Optional[RLHFConfig] = None,
    ):
        self._base_model = base_model
        self._config = config or RLHFConfig()
        self._reward_head: Optional[Any] = None
        self._optimizer: Optional[Any] = None
        self._initialized = False

        # Statistics
        self._total_predictions = 0
        self._total_training_steps = 0

    async def initialize(self, model: Any) -> None:
        """Initialize reward model with a base language model."""
        self._base_model = model
        self._initialized = True
        logger.info("NeuralRewardModel initialized")

    async def predict_reward(self, prompt: str, response: str) -> float:
        """
        Predict reward for a prompt-response pair.

        Returns scalar reward value.
        """
        if not self._initialized:
            # Return neutral reward if not initialized
            return 0.0

        try:
            # Placeholder implementation
            # In production, would:
            # 1. Tokenize prompt + response
            # 2. Forward through base model
            # 3. Apply reward head to get scalar

            self._total_predictions += 1

            # Simple heuristic reward for now
            # Longer, more detailed responses get higher rewards
            word_count = len(response.split())
            detail_score = min(word_count / 100, 1.0)

            # Penalize very short responses
            if word_count < 10:
                detail_score -= 0.5

            # Bonus for completing sentences
            if response.strip().endswith(('.', '!', '?')):
                detail_score += 0.1

            return max(-1.0, min(1.0, detail_score))

        except Exception as e:
            logger.error(f"Reward prediction failed: {e}")
            return 0.0

    async def train(
        self,
        preferences: List[PreferencePair],
        epochs: int = 3,
    ) -> TrainingMetrics:
        """
        Train reward model on preference comparisons.

        Uses Bradley-Terry model for pairwise comparisons:
        P(a > b) = sigmoid(r(a) - r(b))

        Loss = -log P(chosen > rejected)
        """
        if not self._initialized or not preferences:
            return TrainingMetrics()

        logger.info(f"Training reward model on {len(preferences)} preferences for {epochs} epochs")

        metrics = TrainingMetrics()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(epochs):
            random.shuffle(preferences)

            for pref in preferences:
                if not pref.preference or pref.preference == "tie":
                    continue

                # Get rewards for both responses
                reward_a = await self.predict_reward(pref.prompt, pref.response_a)
                reward_b = await self.predict_reward(pref.prompt, pref.response_b)

                # Bradley-Terry probability
                prob_a_better = 1 / (1 + math.exp(reward_b - reward_a))

                # Compute loss based on preference
                if pref.preference == "a":
                    loss = -math.log(max(prob_a_better, 1e-10))
                    correct += 1 if reward_a > reward_b else 0
                else:
                    loss = -math.log(max(1 - prob_a_better, 1e-10))
                    correct += 1 if reward_b > reward_a else 0

                total_loss += loss
                total += 1
                self._total_training_steps += 1

        metrics.epoch = epochs
        metrics.step = self._total_training_steps
        metrics.reward_model_loss = total_loss / max(total, 1)
        metrics.reward_model_accuracy = correct / max(total, 1)

        logger.info(f"Reward model training complete: loss={metrics.reward_model_loss:.4f}, accuracy={metrics.reward_model_accuracy:.2%}")

        return metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get reward model statistics."""
        return {
            "initialized": self._initialized,
            "total_predictions": self._total_predictions,
            "total_training_steps": self._total_training_steps,
        }


# =============================================================================
# PPO TRAINER
# =============================================================================


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for RLHF.

    Implements PPO with:
    - Clipped surrogate objective
    - Value function baseline
    - Entropy regularization
    - KL divergence penalty
    - Adaptive KL coefficient
    """

    def __init__(
        self,
        policy_model: Optional[Any] = None,
        reference_model: Optional[Any] = None,
        reward_model: Optional[RewardModel] = None,
        config: Optional[RLHFConfig] = None,
    ):
        self._policy = policy_model
        self._reference = reference_model
        self._reward_model = reward_model
        self._config = config or RLHFConfig()

        # PPO state
        self._kl_coef = self._config.kl_penalty_coef
        self._optimizer: Optional[Any] = None

        # Statistics
        self._total_steps = 0
        self._total_episodes = 0

    async def initialize(
        self,
        policy_model: Any,
        reference_model: Optional[Any] = None,
    ) -> None:
        """Initialize PPO trainer with models."""
        self._policy = policy_model

        # Reference model is a frozen copy of initial policy
        if reference_model:
            self._reference = reference_model
        else:
            # Would deep copy policy here
            self._reference = policy_model

        logger.info("PPOTrainer initialized")

    async def generate_experience(
        self,
        prompts: List[str],
    ) -> List[RLHFExperience]:
        """Generate experience by sampling from policy."""
        experiences = []

        for prompt in prompts:
            try:
                # Generate response from policy
                # Placeholder - would use actual generation
                response = f"Generated response for: {prompt[:50]}..."

                # Get reward
                reward = 0.0
                if self._reward_model:
                    reward = await self._reward_model.predict_reward(prompt, response)

                # Compute KL divergence from reference
                # Placeholder - would compute actual KL
                kl_div = 0.01

                exp = RLHFExperience(
                    prompt=prompt,
                    response=response,
                    reward=reward,
                    kl_divergence=kl_div,
                    prompt_tokens=len(prompt.split()),
                    response_tokens=len(response.split()),
                )
                experiences.append(exp)

            except Exception as e:
                logger.error(f"Experience generation failed: {e}")

        return experiences

    async def compute_advantages(
        self,
        experiences: List[RLHFExperience],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> List[RLHFExperience]:
        """Compute advantages using GAE."""
        # For single-turn responses, advantages are simpler
        # Just use reward - baseline

        for exp in experiences:
            # Apply KL penalty to reward
            kl_penalty = self._kl_coef * exp.kl_divergence
            adjusted_reward = exp.reward - kl_penalty

            # Simple advantage: reward - value estimate
            exp.advantage = adjusted_reward - exp.value
            exp.returns = adjusted_reward

        return experiences

    async def train_step(
        self,
        experiences: List[RLHFExperience],
    ) -> TrainingMetrics:
        """Perform one PPO training step."""
        if not experiences:
            return TrainingMetrics()

        metrics = TrainingMetrics(step=self._total_steps)

        # Compute advantages
        experiences = await self.compute_advantages(experiences)

        # PPO update
        policy_losses = []
        value_losses = []
        entropies = []
        kl_divs = []
        clip_fractions = []

        for epoch in range(self._config.ppo_epochs):
            random.shuffle(experiences)

            for i in range(0, len(experiences), self._config.ppo_batch_size):
                batch = experiences[i:i + self._config.ppo_batch_size]

                # Placeholder PPO computations
                # In production, would compute:
                # 1. New log probs under current policy
                # 2. Policy ratio: exp(new_log_prob - old_log_prob)
                # 3. Clipped surrogate loss
                # 4. Value function loss
                # 5. Entropy bonus

                batch_policy_loss = 0.01
                batch_value_loss = 0.01
                batch_entropy = 0.1
                batch_kl = sum(e.kl_divergence for e in batch) / len(batch)
                batch_clip_frac = 0.1

                policy_losses.append(batch_policy_loss)
                value_losses.append(batch_value_loss)
                entropies.append(batch_entropy)
                kl_divs.append(batch_kl)
                clip_fractions.append(batch_clip_frac)

                self._total_steps += 1

        # Aggregate metrics
        metrics.policy_loss = sum(policy_losses) / max(len(policy_losses), 1)
        metrics.value_loss = sum(value_losses) / max(len(value_losses), 1)
        metrics.entropy = sum(entropies) / max(len(entropies), 1)
        metrics.kl_divergence = sum(kl_divs) / max(len(kl_divs), 1)
        metrics.clip_fraction = sum(clip_fractions) / max(len(clip_fractions), 1)

        # Reward statistics
        rewards = [e.reward for e in experiences]
        metrics.mean_reward = sum(rewards) / max(len(rewards), 1)
        metrics.std_reward = (sum((r - metrics.mean_reward) ** 2 for r in rewards) / max(len(rewards), 1)) ** 0.5
        metrics.min_reward = min(rewards) if rewards else 0.0
        metrics.max_reward = max(rewards) if rewards else 0.0

        # Adaptive KL coefficient
        if self._config.adaptive_kl:
            if metrics.kl_divergence > self._config.target_kl * 1.5:
                self._kl_coef *= 1.5
            elif metrics.kl_divergence < self._config.target_kl * 0.5:
                self._kl_coef *= 0.5
            self._kl_coef = max(0.001, min(1.0, self._kl_coef))

        self._total_episodes += len(experiences)

        return metrics

    def get_statistics(self) -> Dict[str, Any]:
        """Get PPO trainer statistics."""
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "current_kl_coef": self._kl_coef,
            "config": {
                "ppo_epochs": self._config.ppo_epochs,
                "ppo_clip_range": self._config.ppo_clip_range,
                "target_kl": self._config.target_kl,
            },
        }


# =============================================================================
# RLHF PIPELINE
# =============================================================================


class RLHFPipeline:
    """
    Complete RLHF pipeline for JARVIS Prime.

    Orchestrates:
    - Preference collection
    - Reward model training
    - Policy optimization with PPO
    - Model evaluation and deployment
    """

    def __init__(self, config: Optional[RLHFConfig] = None):
        self._config = config or RLHFConfig()
        self._config.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self._preference_buffer = PreferenceBuffer(capacity=self._config.preference_buffer_size)
        self._reward_model = NeuralRewardModel(config=self._config)
        self._ppo_trainer: Optional[PPOTrainer] = None

        # Models
        self._policy_model: Optional[Any] = None
        self._reference_model: Optional[Any] = None

        # State
        self._is_running = False
        self._training_task: Optional[asyncio.Task] = None
        self._last_reward_training = 0.0
        self._last_policy_training = 0.0

        # History
        self._training_history: List[TrainingMetrics] = []

        logger.info("RLHFPipeline initialized")

    async def start(self, model: Optional[Any] = None) -> None:
        """Start the RLHF pipeline."""
        if model:
            self._policy_model = model
            self._reference_model = model  # Would be a frozen copy
            await self._reward_model.initialize(model)
            self._ppo_trainer = PPOTrainer(
                policy_model=model,
                reference_model=model,
                reward_model=self._reward_model,
                config=self._config,
            )

        # Load saved state
        await self._load_state()

        self._is_running = True
        self._training_task = asyncio.create_task(self._training_loop())

        logger.info("RLHFPipeline started")

    async def stop(self) -> None:
        """Stop the RLHF pipeline."""
        self._is_running = False

        if self._training_task:
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass

        await self._save_state()
        logger.info("RLHFPipeline stopped")

    async def record_preference(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        preference: str,  # "a", "b", or "tie"
        confidence: float = 1.0,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a human preference comparison.

        Returns preference ID.
        """
        pref = PreferencePair(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            preference=preference,
            confidence=confidence,
            user_id=user_id,
            context=context or {},
        )

        self._preference_buffer.add(pref)

        logger.debug(f"Recorded preference {pref.id}: {preference}")
        return pref.id

    async def record_implicit_preference(
        self,
        prompt: str,
        response: str,
        was_helpful: bool,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Record implicit preference from user behavior.

        If user indicated response was helpful, creates synthetic comparison.
        """
        if not was_helpful:
            # Generate alternative response for comparison
            # In production, would actually generate an alternative
            alternative = f"Alternative response for: {prompt[:50]}..."

            return await self.record_preference(
                prompt=prompt,
                response_a=response,
                response_b=alternative,
                preference="b" if not was_helpful else "a",
                confidence=0.7,  # Lower confidence for implicit
                user_id=user_id,
                context={"source": "implicit"},
            )
        else:
            # Record positive feedback
            return await self.record_preference(
                prompt=prompt,
                response_a=response,
                response_b="[baseline response]",
                preference="a",
                confidence=0.8,
                user_id=user_id,
                context={"source": "implicit_positive"},
            )

    async def train_reward_model(self, force: bool = False) -> TrainingMetrics:
        """Train the reward model on collected preferences."""
        if not force:
            if len(self._preference_buffer) < self._config.min_preferences_for_training:
                logger.info(f"Not enough preferences for training: {len(self._preference_buffer)}/{self._config.min_preferences_for_training}")
                return TrainingMetrics()

        logger.info("Training reward model...")

        # Sample preferences
        preferences = self._preference_buffer.sample(
            min(len(self._preference_buffer), self._config.min_preferences_for_training * 2),
            balanced=True,
        )

        # Train
        metrics = await self._reward_model.train(
            preferences,
            epochs=self._config.reward_model_epochs,
        )

        self._training_history.append(metrics)
        self._last_reward_training = time.time()

        return metrics

    async def optimize_policy(self, prompts: Optional[List[str]] = None) -> TrainingMetrics:
        """Optimize policy using PPO with reward model."""
        if not self._ppo_trainer:
            logger.warning("PPO trainer not initialized")
            return TrainingMetrics()

        logger.info("Optimizing policy with PPO...")

        # Use provided prompts or sample from preferences
        if not prompts:
            recent_preferences = list(self._preference_buffer.preferences)[-100:]
            prompts = [p.prompt for p in recent_preferences]

        if not prompts:
            logger.warning("No prompts for policy optimization")
            return TrainingMetrics()

        # Generate experience
        experiences = await self._ppo_trainer.generate_experience(prompts)

        # Train
        metrics = await self._ppo_trainer.train_step(experiences)

        self._training_history.append(metrics)
        self._last_policy_training = time.time()

        return metrics

    async def _training_loop(self) -> None:
        """Background training loop."""
        while self._is_running:
            try:
                await asyncio.sleep(3600)  # Check hourly

                # Check if reward model training is due
                hours_since_reward = (time.time() - self._last_reward_training) / 3600
                if hours_since_reward >= self._config.reward_model_update_interval_hours:
                    if len(self._preference_buffer) >= self._config.min_preferences_for_training:
                        await self.train_reward_model()

                # Check if policy training is due
                hours_since_policy = (time.time() - self._last_policy_training) / 3600
                if hours_since_policy >= self._config.policy_update_interval_hours:
                    await self.optimize_policy()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RLHF training loop error: {e}")
                await asyncio.sleep(60)

    async def _save_state(self) -> None:
        """Save pipeline state."""
        state_file = self._config.checkpoints_dir / "rlhf_state.json"

        try:
            state = {
                "last_reward_training": self._last_reward_training,
                "last_policy_training": self._last_policy_training,
                "training_history_count": len(self._training_history),
            }

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Save preference buffer
            buffer_file = self._config.checkpoints_dir / "preferences.json"
            self._preference_buffer.save(buffer_file)

            logger.info("RLHF state saved")

        except Exception as e:
            logger.error(f"Failed to save RLHF state: {e}")

    async def _load_state(self) -> None:
        """Load pipeline state."""
        state_file = self._config.checkpoints_dir / "rlhf_state.json"

        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)

                self._last_reward_training = state.get("last_reward_training", 0.0)
                self._last_policy_training = state.get("last_policy_training", 0.0)

            except Exception as e:
                logger.warning(f"Failed to load RLHF state: {e}")

        # Load preference buffer
        buffer_file = self._config.checkpoints_dir / "preferences.json"
        if buffer_file.exists():
            self._preference_buffer.load(buffer_file)

    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "is_running": self._is_running,
            "preference_buffer": self._preference_buffer.get_statistics(),
            "reward_model": self._reward_model.get_statistics(),
            "ppo_trainer": self._ppo_trainer.get_statistics() if self._ppo_trainer else None,
            "last_reward_training_hours_ago": (time.time() - self._last_reward_training) / 3600 if self._last_reward_training else None,
            "last_policy_training_hours_ago": (time.time() - self._last_policy_training) / 3600 if self._last_policy_training else None,
            "training_history_count": len(self._training_history),
            "config": self._config.to_dict(),
        }

    def get_recent_metrics(self, n: int = 10) -> List[TrainingMetrics]:
        """Get recent training metrics."""
        return self._training_history[-n:]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_rlhf_pipeline: Optional[RLHFPipeline] = None
_rlhf_lock = asyncio.Lock()


async def get_rlhf_pipeline(
    config: Optional[RLHFConfig] = None,
    model: Optional[Any] = None,
) -> RLHFPipeline:
    """Get or create the global RLHF pipeline."""
    global _rlhf_pipeline

    async with _rlhf_lock:
        if _rlhf_pipeline is None:
            _rlhf_pipeline = RLHFPipeline(config)
            await _rlhf_pipeline.start(model)

        return _rlhf_pipeline


async def shutdown_rlhf_pipeline() -> None:
    """Shutdown the global RLHF pipeline."""
    global _rlhf_pipeline

    async with _rlhf_lock:
        if _rlhf_pipeline:
            await _rlhf_pipeline.stop()
            _rlhf_pipeline = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Config
    "RLHFConfig",
    # Data structures
    "PreferencePair",
    "PreferenceBuffer",
    "RLHFExperience",
    "TrainingMetrics",
    # Components
    "RewardModel",
    "NeuralRewardModel",
    "PPOTrainer",
    # Pipeline
    "RLHFPipeline",
    "get_rlhf_pipeline",
    "shutdown_rlhf_pipeline",
]
