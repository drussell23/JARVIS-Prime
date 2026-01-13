#!/usr/bin/env python3
"""
Adaptive Strategy Selector - ML-Based Reasoning Strategy Selection
==================================================================

v92.1 - Intelligent, Learning-Based Strategy Selection

This module implements an ML-based adaptive strategy selector that:
1. Learns from historical strategy performance
2. Extracts features from input text (complexity, domain, length, etc.)
3. Uses a lightweight neural network for strategy prediction
4. Supports online learning from feedback
5. Provides confidence-weighted strategy recommendations

ARCHITECTURE:
    Input Text
        ↓
    [Feature Extraction]
    ├─ Text Complexity (vocabulary diversity, sentence length)
    ├─ Domain Classification (code, math, creative, factual)
    ├─ Task Type Detection (question, instruction, analysis)
    ├─ Context Requirements (single-step vs multi-step)
    └─ Urgency/Priority Signals
        ↓
    [Strategy Predictor Neural Network]
    ├─ Input Layer (feature vector)
    ├─ Hidden Layers (with dropout)
    └─ Output Layer (strategy probabilities)
        ↓
    [Confidence-Weighted Selection]
    ├─ Top-k candidates with confidence scores
    ├─ Exploration vs exploitation (epsilon-greedy)
    └─ Historical performance weighting
        ↓
    Selected Strategy + Confidence

STRATEGIES:
    - chain_of_thought: Step-by-step reasoning (default for complex tasks)
    - tree_of_thoughts: Multi-path exploration (for open-ended problems)
    - self_reflection: Iterative refinement (for quality-critical tasks)
    - hypothesis_testing: Scientific method (for analytical tasks)
    - direct: No special reasoning (for simple tasks)

USAGE:
    selector = await get_adaptive_strategy_selector()
    strategy, confidence = await selector.select_strategy(
        input_text="Explain quantum entanglement",
        context={"domain": "physics"},
    )

    # After getting result, provide feedback for learning
    await selector.record_feedback(
        input_text="Explain quantum entanglement",
        strategy="chain_of_thought",
        quality_score=0.92,
        execution_time_ms=1500,
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    SELF_REFLECTION = "self_reflection"
    HYPOTHESIS_TESTING = "hypothesis_testing"
    DIRECT = "direct"


@dataclass
class StrategyProfile:
    """Profile describing a strategy's characteristics."""
    name: ReasoningStrategy
    complexity_affinity: float  # 0-1, how well it handles complex tasks
    latency_factor: float  # Relative time cost (1.0 = baseline)
    quality_baseline: float  # Expected quality for average task
    domains: List[str]  # Preferred domains
    min_tokens: int  # Minimum input size where this shines
    max_tokens: int  # Maximum efficient input size


DEFAULT_STRATEGY_PROFILES: Dict[ReasoningStrategy, StrategyProfile] = {
    ReasoningStrategy.CHAIN_OF_THOUGHT: StrategyProfile(
        name=ReasoningStrategy.CHAIN_OF_THOUGHT,
        complexity_affinity=0.8,
        latency_factor=1.2,
        quality_baseline=0.85,
        domains=["math", "logic", "code", "analysis"],
        min_tokens=20,
        max_tokens=2000,
    ),
    ReasoningStrategy.TREE_OF_THOUGHTS: StrategyProfile(
        name=ReasoningStrategy.TREE_OF_THOUGHTS,
        complexity_affinity=0.95,
        latency_factor=2.5,
        quality_baseline=0.90,
        domains=["creative", "planning", "open_ended"],
        min_tokens=50,
        max_tokens=1000,
    ),
    ReasoningStrategy.SELF_REFLECTION: StrategyProfile(
        name=ReasoningStrategy.SELF_REFLECTION,
        complexity_affinity=0.7,
        latency_factor=1.8,
        quality_baseline=0.88,
        domains=["writing", "review", "quality"],
        min_tokens=30,
        max_tokens=1500,
    ),
    ReasoningStrategy.HYPOTHESIS_TESTING: StrategyProfile(
        name=ReasoningStrategy.HYPOTHESIS_TESTING,
        complexity_affinity=0.85,
        latency_factor=2.0,
        quality_baseline=0.87,
        domains=["science", "debugging", "investigation"],
        min_tokens=40,
        max_tokens=1200,
    ),
    ReasoningStrategy.DIRECT: StrategyProfile(
        name=ReasoningStrategy.DIRECT,
        complexity_affinity=0.3,
        latency_factor=0.5,
        quality_baseline=0.75,
        domains=["simple", "factual", "lookup"],
        min_tokens=1,
        max_tokens=500,
    ),
}


@dataclass
class AdaptiveSelectorConfig:
    """Configuration for adaptive strategy selector."""
    # Learning parameters
    learning_rate: float = 0.01
    exploration_rate: float = 0.1  # Epsilon for epsilon-greedy
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01

    # History management
    max_history_size: int = 10000
    min_samples_for_prediction: int = 50

    # Feature extraction
    complexity_weight: float = 0.3
    domain_weight: float = 0.25
    length_weight: float = 0.15
    context_weight: float = 0.3

    # Neural network (simple)
    hidden_size: int = 64
    dropout_rate: float = 0.2

    # Persistence
    state_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "adaptive_selector"
    )
    save_interval_seconds: float = 300.0  # Auto-save every 5 minutes

    def __post_init__(self):
        if isinstance(self.state_dir, str):
            self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

@dataclass
class InputFeatures:
    """Extracted features from input text."""
    # Text statistics
    token_count: int
    sentence_count: int
    avg_sentence_length: float
    vocabulary_diversity: float  # Unique words / total words

    # Complexity indicators
    complexity_score: float  # 0-1
    has_code: bool
    has_math: bool
    has_questions: bool

    # Domain signals
    detected_domains: List[str]
    primary_domain: str
    domain_confidence: float

    # Task type
    is_question: bool
    is_instruction: bool
    is_analysis: bool
    requires_creativity: bool
    requires_precision: bool

    # Context
    has_context: bool
    context_complexity: float

    def to_vector(self) -> List[float]:
        """Convert features to a numerical vector for the neural network."""
        return [
            min(self.token_count / 500, 2.0),  # Normalized token count
            min(self.sentence_count / 20, 2.0),  # Normalized sentence count
            min(self.avg_sentence_length / 30, 2.0),  # Normalized avg length
            self.vocabulary_diversity,
            self.complexity_score,
            float(self.has_code),
            float(self.has_math),
            float(self.has_questions),
            self.domain_confidence,
            float(self.is_question),
            float(self.is_instruction),
            float(self.is_analysis),
            float(self.requires_creativity),
            float(self.requires_precision),
            float(self.has_context),
            self.context_complexity,
        ]


class FeatureExtractor:
    """Extracts features from input text for strategy selection."""

    # Domain detection patterns
    DOMAIN_PATTERNS = {
        "code": [
            r"\bdef\s+\w+", r"\bclass\s+\w+", r"\bfunction\s+\w+",
            r"\bimport\s+", r"\breturn\s+", r"```\w*",
            r"\bif\s+.*:", r"\bfor\s+\w+\s+in\s+",
        ],
        "math": [
            r"\b\d+\s*[\+\-\*\/\^]\s*\d+", r"\bequation\b", r"\bcalculate\b",
            r"\bderivative\b", r"\bintegral\b", r"\bsum\b.*\bto\b",
            r"\bprove\b.*\bthat\b", r"\bsolve\b.*\bfor\b",
        ],
        "science": [
            r"\bhypothesis\b", r"\bexperiment\b", r"\btheory\b",
            r"\bobserv", r"\bdata\b.*\bshow", r"\bresearch\b",
            r"\bscientific\b", r"\bempirical\b",
        ],
        "creative": [
            r"\bwrite\s+a\s+(story|poem|song)", r"\bimagine\b", r"\bcreate\b",
            r"\bfiction\b", r"\bnarrative\b", r"\bcharacter\b",
        ],
        "analysis": [
            r"\banalyze\b", r"\bcompare\b", r"\bcontrast\b",
            r"\bevaluate\b", r"\bassess\b", r"\bcritique\b",
            r"\bexplain\s+why\b", r"\bwhat\s+are\s+the\s+reasons\b",
        ],
        "factual": [
            r"\bwhat\s+is\b", r"\bwho\s+is\b", r"\bwhen\s+did\b",
            r"\bwhere\s+is\b", r"\bdefine\b", r"\blist\b",
        ],
    }

    # Task type patterns
    QUESTION_PATTERNS = [r"\?$", r"^(what|who|when|where|why|how|which|can|could|would|should)\s"]
    INSTRUCTION_PATTERNS = [r"^(please|kindly)?\s*(write|create|make|build|implement|design|develop)"]
    ANALYSIS_PATTERNS = [r"analyze|examine|evaluate|assess|review|explain\s+why"]

    def __init__(self):
        self._compiled_domain_patterns = {
            domain: [re.compile(p, re.IGNORECASE) for p in patterns]
            for domain, patterns in self.DOMAIN_PATTERNS.items()
        }
        self._question_patterns = [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS]
        self._instruction_patterns = [re.compile(p, re.IGNORECASE) for p in self.INSTRUCTION_PATTERNS]
        self._analysis_patterns = [re.compile(p, re.IGNORECASE) for p in self.ANALYSIS_PATTERNS]

    def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> InputFeatures:
        """Extract features from input text."""
        context = context or {}

        # Basic text statistics
        words = text.split()
        token_count = len(words)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)
        avg_sentence_length = token_count / sentence_count

        # Vocabulary diversity
        unique_words = set(w.lower() for w in words)
        vocabulary_diversity = len(unique_words) / max(token_count, 1)

        # Detect domains
        domain_scores = {}
        for domain, patterns in self._compiled_domain_patterns.items():
            score = sum(1 for p in patterns if p.search(text))
            if score > 0:
                domain_scores[domain] = score

        detected_domains = list(domain_scores.keys())
        if domain_scores:
            primary_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])
            domain_confidence = min(domain_scores[primary_domain] / 5, 1.0)
        else:
            primary_domain = "general"
            domain_confidence = 0.5

        # Override with context if provided
        if "domain" in context:
            primary_domain = context["domain"]
            domain_confidence = 0.9

        # Task type detection
        is_question = any(p.search(text) for p in self._question_patterns)
        is_instruction = any(p.search(text) for p in self._instruction_patterns)
        is_analysis = any(p.search(text) for p in self._analysis_patterns)

        # Content type detection
        has_code = "code" in detected_domains or "```" in text
        has_math = "math" in detected_domains

        # Complexity estimation
        complexity_factors = [
            min(token_count / 200, 1.0),  # Length factor
            vocabulary_diversity,  # Vocabulary richness
            float(len(detected_domains) > 1) * 0.3,  # Multi-domain
            float(has_code) * 0.2,  # Code presence
            float(has_math) * 0.2,  # Math presence
            float(is_analysis) * 0.2,  # Analysis required
        ]
        complexity_score = min(sum(complexity_factors) / len(complexity_factors) * 1.5, 1.0)

        # Creativity and precision requirements
        requires_creativity = "creative" in detected_domains or is_instruction
        requires_precision = has_code or has_math or "science" in detected_domains

        # Context features
        has_context = bool(context)
        context_complexity = 0.0
        if context:
            context_str = str(context)
            context_complexity = min(len(context_str) / 500, 1.0)

        return InputFeatures(
            token_count=token_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            vocabulary_diversity=vocabulary_diversity,
            complexity_score=complexity_score,
            has_code=has_code,
            has_math=has_math,
            has_questions=is_question,
            detected_domains=detected_domains,
            primary_domain=primary_domain,
            domain_confidence=domain_confidence,
            is_question=is_question,
            is_instruction=is_instruction,
            is_analysis=is_analysis,
            requires_creativity=requires_creativity,
            requires_precision=requires_precision,
            has_context=has_context,
            context_complexity=context_complexity,
        )


# =============================================================================
# SIMPLE NEURAL NETWORK (No External Dependencies)
# =============================================================================

class SimpleNeuralNetwork:
    """
    Lightweight neural network for strategy prediction.

    Uses only numpy-style operations with pure Python lists.
    Architecture: Input -> Hidden (ReLU) -> Output (Softmax)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.01,
        dropout_rate: float = 0.2,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        # Initialize weights with Xavier initialization
        self.w1 = self._xavier_init(input_size, hidden_size)
        self.b1 = [[0.0] for _ in range(hidden_size)]
        self.w2 = self._xavier_init(hidden_size, output_size)
        self.b2 = [[0.0] for _ in range(output_size)]

        # Training state
        self._training = False

    def _xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        """Xavier weight initialization."""
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        return [
            [random.gauss(0, scale) for _ in range(fan_out)]
            for _ in range(fan_in)
        ]

    def _relu(self, x: List[float]) -> List[float]:
        """ReLU activation."""
        return [max(0, v) for v in x]

    def _relu_derivative(self, x: List[float]) -> List[float]:
        """ReLU derivative."""
        return [1.0 if v > 0 else 0.0 for v in x]

    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax activation with numerical stability."""
        max_x = max(x)
        exp_x = [math.exp(v - max_x) for v in x]
        sum_exp = sum(exp_x)
        return [v / sum_exp for v in exp_x]

    def _matmul(
        self,
        x: List[float],
        w: List[List[float]],
        b: List[List[float]],
    ) -> List[float]:
        """Matrix multiplication: x @ W + b."""
        result = []
        for j in range(len(w[0])):
            val = sum(x[i] * w[i][j] for i in range(len(x)))
            val += b[j][0]
            result.append(val)
        return result

    def _dropout(self, x: List[float]) -> Tuple[List[float], List[float]]:
        """Apply dropout during training."""
        if not self._training or self.dropout_rate <= 0:
            return x, [1.0] * len(x)

        mask = [
            0.0 if random.random() < self.dropout_rate else 1.0 / (1 - self.dropout_rate)
            for _ in x
        ]
        return [v * m for v, m in zip(x, mask)], mask

    def forward(self, x: List[float]) -> Tuple[List[float], Dict[str, Any]]:
        """Forward pass through the network."""
        # Input validation
        if len(x) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {len(x)}")

        # Layer 1: Input -> Hidden
        z1 = self._matmul(x, self.w1, self.b1)
        a1 = self._relu(z1)
        a1_dropout, dropout_mask = self._dropout(a1)

        # Layer 2: Hidden -> Output
        z2 = self._matmul(a1_dropout, self.w2, self.b2)
        output = self._softmax(z2)

        # Cache for backprop
        cache = {
            "x": x,
            "z1": z1,
            "a1": a1,
            "a1_dropout": a1_dropout,
            "dropout_mask": dropout_mask,
            "z2": z2,
            "output": output,
        }

        return output, cache

    def backward(
        self,
        cache: Dict[str, Any],
        target_idx: int,
        reward: float = 1.0,
    ) -> None:
        """Backward pass with gradient descent."""
        output = cache["output"]
        x = cache["x"]
        a1_dropout = cache["a1_dropout"]
        z1 = cache["z1"]
        dropout_mask = cache["dropout_mask"]

        # Output layer gradient (cross-entropy loss derivative)
        d_output = output.copy()
        d_output[target_idx] -= 1.0
        d_output = [v * reward for v in d_output]

        # Gradient for W2 and b2
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                grad = a1_dropout[i] * d_output[j]
                self.w2[i][j] -= self.learning_rate * grad

        for j in range(self.output_size):
            self.b2[j][0] -= self.learning_rate * d_output[j]

        # Hidden layer gradient
        d_hidden = [
            sum(d_output[j] * self.w2[i][j] for j in range(self.output_size))
            for i in range(self.hidden_size)
        ]

        # Apply dropout mask and ReLU derivative
        relu_deriv = self._relu_derivative(z1)
        d_hidden = [
            d * r * m
            for d, r, m in zip(d_hidden, relu_deriv, dropout_mask)
        ]

        # Gradient for W1 and b1
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                grad = x[i] * d_hidden[j]
                self.w1[i][j] -= self.learning_rate * grad

        for j in range(self.hidden_size):
            self.b1[j][0] -= self.learning_rate * d_hidden[j]

    def predict(self, x: List[float]) -> Tuple[int, float, List[float]]:
        """
        Make a prediction.

        Returns:
            Tuple of (predicted_idx, confidence, all_probabilities)
        """
        self._training = False
        output, _ = self.forward(x)
        pred_idx = output.index(max(output))
        confidence = output[pred_idx]
        return pred_idx, confidence, output

    def train_step(
        self,
        x: List[float],
        target_idx: int,
        reward: float = 1.0,
    ) -> float:
        """
        Single training step.

        Args:
            x: Input features
            target_idx: Target strategy index
            reward: Reward signal (higher = better)

        Returns:
            Loss value
        """
        self._training = True
        output, cache = self.forward(x)

        # Cross-entropy loss
        loss = -math.log(max(output[target_idx], 1e-10))

        # Backprop with reward weighting
        self.backward(cache, target_idx, reward)

        self._training = False
        return loss

    def to_dict(self) -> Dict[str, Any]:
        """Serialize network to dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimpleNeuralNetwork":
        """Deserialize network from dictionary."""
        nn = cls(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            output_size=data["output_size"],
            learning_rate=data["learning_rate"],
            dropout_rate=data["dropout_rate"],
        )
        nn.w1 = data["w1"]
        nn.b1 = data["b1"]
        nn.w2 = data["w2"]
        nn.b2 = data["b2"]
        return nn


# =============================================================================
# ADAPTIVE STRATEGY SELECTOR
# =============================================================================

@dataclass
class StrategyFeedback:
    """Feedback for a strategy execution."""
    input_hash: str
    strategy: ReasoningStrategy
    features: List[float]
    quality_score: float
    execution_time_ms: float
    timestamp: float


class AdaptiveStrategySelector:
    """
    ML-based adaptive strategy selector.

    Learns from historical performance to predict the best strategy
    for each input type.
    """

    def __init__(self, config: Optional[AdaptiveSelectorConfig] = None):
        self.config = config or AdaptiveSelectorConfig()
        self._feature_extractor = FeatureExtractor()

        # Strategy index mapping
        self._strategies = list(ReasoningStrategy)
        self._strategy_to_idx = {s: i for i, s in enumerate(self._strategies)}
        self._idx_to_strategy = {i: s for s, i in self._strategy_to_idx.items()}

        # Neural network
        feature_size = 16  # Number of features from InputFeatures.to_vector()
        self._network = SimpleNeuralNetwork(
            input_size=feature_size,
            hidden_size=self.config.hidden_size,
            output_size=len(self._strategies),
            learning_rate=self.config.learning_rate,
            dropout_rate=self.config.dropout_rate,
        )

        # Performance tracking
        self._strategy_stats: Dict[ReasoningStrategy, Dict[str, Any]] = {
            s: {
                "total_uses": 0,
                "total_quality": 0.0,
                "total_time_ms": 0.0,
                "recent_quality": [],
            }
            for s in self._strategies
        }

        # History for online learning
        self._feedback_history: List[StrategyFeedback] = []
        self._exploration_rate = self.config.exploration_rate

        # Heuristic fallback (before enough data)
        self._use_heuristics = True

        # State management
        self._initialized = False
        self._lock = asyncio.Lock()
        self._last_save = time.time()

    async def initialize(self) -> None:
        """Initialize the selector, loading persisted state if available."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Load persisted state
            state_path = self.config.state_dir / "selector_state.json"
            if state_path.exists():
                try:
                    with open(state_path) as f:
                        state = json.load(f)

                    # Restore network
                    if "network" in state:
                        self._network = SimpleNeuralNetwork.from_dict(state["network"])

                    # Restore stats
                    if "strategy_stats" in state:
                        for s_name, stats in state["strategy_stats"].items():
                            try:
                                strategy = ReasoningStrategy(s_name)
                                self._strategy_stats[strategy] = stats
                            except ValueError:
                                pass

                    # Restore exploration rate
                    self._exploration_rate = state.get(
                        "exploration_rate",
                        self.config.exploration_rate,
                    )

                    # Check if we have enough data for predictions
                    total_uses = sum(
                        s["total_uses"]
                        for s in self._strategy_stats.values()
                    )
                    self._use_heuristics = total_uses < self.config.min_samples_for_prediction

                    logger.info(
                        f"Loaded adaptive selector state: {total_uses} samples, "
                        f"exploration={self._exploration_rate:.3f}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to load selector state: {e}")

            self._initialized = True

    async def select_strategy(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ReasoningStrategy, float]:
        """
        Select the best strategy for the given input.

        Args:
            input_text: The input text to reason about
            context: Optional context dictionary

        Returns:
            Tuple of (selected_strategy, confidence)
        """
        await self.initialize()

        # Extract features
        features = self._feature_extractor.extract(input_text, context)
        feature_vector = features.to_vector()

        # Decide: exploration vs exploitation
        if random.random() < self._exploration_rate:
            # Exploration: random strategy (weighted by heuristic fit)
            strategy = self._heuristic_select(features)
            confidence = 0.5  # Lower confidence for exploration
            logger.debug(f"Exploration: selected {strategy.value}")
        elif self._use_heuristics:
            # Not enough data: use heuristics
            strategy = self._heuristic_select(features)
            confidence = 0.7
            logger.debug(f"Heuristic: selected {strategy.value}")
        else:
            # Exploitation: use neural network
            pred_idx, nn_confidence, probs = self._network.predict(feature_vector)
            strategy = self._idx_to_strategy[pred_idx]

            # Combine NN confidence with historical performance
            stats = self._strategy_stats[strategy]
            if stats["total_uses"] > 0:
                avg_quality = stats["total_quality"] / stats["total_uses"]
                confidence = 0.6 * nn_confidence + 0.4 * avg_quality
            else:
                confidence = nn_confidence * 0.7

            logger.debug(
                f"ML prediction: {strategy.value} (nn_conf={nn_confidence:.2f}, "
                f"combined_conf={confidence:.2f})"
            )

        return strategy, confidence

    def _heuristic_select(self, features: InputFeatures) -> ReasoningStrategy:
        """Select strategy using rule-based heuristics."""
        # Simple tasks -> direct
        if features.complexity_score < 0.3 and features.token_count < 50:
            return ReasoningStrategy.DIRECT

        # Code tasks -> chain of thought
        if features.has_code:
            return ReasoningStrategy.CHAIN_OF_THOUGHT

        # Math tasks -> chain of thought or hypothesis testing
        if features.has_math:
            if features.is_analysis:
                return ReasoningStrategy.HYPOTHESIS_TESTING
            return ReasoningStrategy.CHAIN_OF_THOUGHT

        # Creative tasks -> tree of thoughts
        if features.requires_creativity:
            return ReasoningStrategy.TREE_OF_THOUGHTS

        # Quality-critical tasks -> self reflection
        if features.requires_precision and not features.has_code:
            return ReasoningStrategy.SELF_REFLECTION

        # Analysis tasks -> hypothesis testing
        if features.is_analysis or "science" in features.detected_domains:
            return ReasoningStrategy.HYPOTHESIS_TESTING

        # Complex tasks -> chain of thought
        if features.complexity_score > 0.6:
            return ReasoningStrategy.CHAIN_OF_THOUGHT

        # Default for medium complexity
        return ReasoningStrategy.CHAIN_OF_THOUGHT

    async def record_feedback(
        self,
        input_text: str,
        strategy: Union[ReasoningStrategy, str],
        quality_score: float,
        execution_time_ms: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record feedback for a strategy execution to improve future predictions.

        Args:
            input_text: The input that was processed
            strategy: The strategy that was used
            quality_score: Quality of the result (0-1)
            execution_time_ms: Execution time in milliseconds
            context: Optional context that was used
        """
        await self.initialize()

        if isinstance(strategy, str):
            strategy = ReasoningStrategy(strategy)

        # Extract features
        features = self._feature_extractor.extract(input_text, context)
        feature_vector = features.to_vector()

        # Create feedback record
        input_hash = hashlib.md5(input_text.encode()).hexdigest()[:16]
        feedback = StrategyFeedback(
            input_hash=input_hash,
            strategy=strategy,
            features=feature_vector,
            quality_score=quality_score,
            execution_time_ms=execution_time_ms,
            timestamp=time.time(),
        )

        async with self._lock:
            # Add to history
            self._feedback_history.append(feedback)

            # Trim history if needed
            if len(self._feedback_history) > self.config.max_history_size:
                self._feedback_history = self._feedback_history[-self.config.max_history_size:]

            # Update strategy stats
            stats = self._strategy_stats[strategy]
            stats["total_uses"] += 1
            stats["total_quality"] += quality_score
            stats["total_time_ms"] += execution_time_ms
            stats["recent_quality"].append(quality_score)
            if len(stats["recent_quality"]) > 100:
                stats["recent_quality"] = stats["recent_quality"][-100:]

            # Online learning: train on this sample
            strategy_idx = self._strategy_to_idx[strategy]
            reward = quality_score * 2 - 1  # Map 0-1 to -1 to 1
            self._network.train_step(feature_vector, strategy_idx, reward)

            # Decay exploration rate
            self._exploration_rate = max(
                self.config.min_exploration_rate,
                self._exploration_rate * self.config.exploration_decay,
            )

            # Check if we can switch from heuristics to ML
            total_uses = sum(s["total_uses"] for s in self._strategy_stats.values())
            if self._use_heuristics and total_uses >= self.config.min_samples_for_prediction:
                self._use_heuristics = False
                logger.info("Switching from heuristics to ML-based selection")

            # Auto-save periodically
            if time.time() - self._last_save > self.config.save_interval_seconds:
                await self._save_state()

    async def _save_state(self) -> None:
        """Save current state to disk."""
        state = {
            "network": self._network.to_dict(),
            "strategy_stats": {
                s.value: stats
                for s, stats in self._strategy_stats.items()
            },
            "exploration_rate": self._exploration_rate,
            "saved_at": datetime.now().isoformat(),
        }

        state_path = self.config.state_dir / "selector_state.json"
        try:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            self._last_save = time.time()
            logger.debug("Saved adaptive selector state")
        except Exception as e:
            logger.warning(f"Failed to save selector state: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current selector statistics."""
        total_uses = sum(s["total_uses"] for s in self._strategy_stats.values())

        strategy_performance = {}
        for strategy, stats in self._strategy_stats.items():
            if stats["total_uses"] > 0:
                strategy_performance[strategy.value] = {
                    "uses": stats["total_uses"],
                    "avg_quality": stats["total_quality"] / stats["total_uses"],
                    "avg_time_ms": stats["total_time_ms"] / stats["total_uses"],
                    "recent_avg_quality": (
                        sum(stats["recent_quality"]) / len(stats["recent_quality"])
                        if stats["recent_quality"]
                        else 0.0
                    ),
                }

        return {
            "total_samples": total_uses,
            "exploration_rate": self._exploration_rate,
            "using_ml": not self._use_heuristics,
            "strategy_performance": strategy_performance,
            "history_size": len(self._feedback_history),
        }

    async def shutdown(self) -> None:
        """Shutdown and save state."""
        async with self._lock:
            await self._save_state()
            logger.info("Adaptive strategy selector shutdown complete")


# =============================================================================
# SINGLETON MANAGEMENT
# =============================================================================

_selector_instance: Optional[AdaptiveStrategySelector] = None
_selector_lock = asyncio.Lock()


async def get_adaptive_strategy_selector() -> AdaptiveStrategySelector:
    """Get or create the singleton adaptive strategy selector."""
    global _selector_instance

    if _selector_instance is None:
        async with _selector_lock:
            if _selector_instance is None:
                _selector_instance = AdaptiveStrategySelector()
                await _selector_instance.initialize()
                logger.info("Adaptive strategy selector initialized")

    return _selector_instance


async def shutdown_adaptive_strategy_selector() -> None:
    """Shutdown the singleton selector."""
    global _selector_instance

    if _selector_instance is not None:
        async with _selector_lock:
            if _selector_instance is not None:
                await _selector_instance.shutdown()
                _selector_instance = None


# =============================================================================
# INTEGRATION WITH REASONING ENGINE
# =============================================================================

async def select_best_strategy(
    input_text: str,
    context: Optional[Dict[str, Any]] = None,
    prefer_speed: bool = False,
    prefer_quality: bool = False,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    High-level API for strategy selection.

    Args:
        input_text: The input to process
        context: Optional context
        prefer_speed: If True, bias toward faster strategies
        prefer_quality: If True, bias toward higher quality strategies

    Returns:
        Tuple of (strategy_name, confidence, metadata)
    """
    selector = await get_adaptive_strategy_selector()
    strategy, confidence = await selector.select_strategy(input_text, context)

    # Apply preferences
    if prefer_speed and confidence < 0.9:
        # Consider faster alternatives
        profile = DEFAULT_STRATEGY_PROFILES.get(strategy)
        if profile and profile.latency_factor > 1.5:
            # Check if direct strategy would work
            features = selector._feature_extractor.extract(input_text, context)
            if features.complexity_score < 0.4:
                strategy = ReasoningStrategy.DIRECT
                confidence *= 0.9

    if prefer_quality and confidence < 0.9:
        # Consider higher quality alternatives
        profile = DEFAULT_STRATEGY_PROFILES.get(strategy)
        if profile and profile.quality_baseline < 0.85:
            strategy = ReasoningStrategy.CHAIN_OF_THOUGHT
            confidence *= 0.95

    metadata = {
        "selector_stats": selector.get_stats(),
        "strategy_profile": {
            "name": strategy.value,
            **{
                k: v for k, v in DEFAULT_STRATEGY_PROFILES.get(
                    strategy,
                    StrategyProfile(
                        name=strategy,
                        complexity_affinity=0.5,
                        latency_factor=1.0,
                        quality_baseline=0.8,
                        domains=[],
                        min_tokens=1,
                        max_tokens=10000,
                    ),
                ).__dict__.items()
                if k != "name"
            },
        },
    }

    return strategy.value, confidence, metadata


if __name__ == "__main__":
    # Quick test
    async def test():
        selector = await get_adaptive_strategy_selector()

        # Test various inputs
        test_cases = [
            ("What is 2+2?", {"domain": "math"}),
            ("Write a poem about the ocean", {}),
            ("def fibonacci(n):\n    pass  # implement this", {"domain": "code"}),
            ("Analyze the causes of World War I", {}),
            ("Hello", {}),
        ]

        for text, ctx in test_cases:
            strategy, conf = await selector.select_strategy(text, ctx)
            print(f"Input: {text[:50]}...")
            print(f"  Strategy: {strategy.value}")
            print(f"  Confidence: {conf:.2f}")
            print()

        # Show stats
        print("Stats:", json.dumps(selector.get_stats(), indent=2))

        await shutdown_adaptive_strategy_selector()

    asyncio.run(test())
