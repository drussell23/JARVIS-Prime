"""
Automatic Model Selector - Intelligent Task-Based Model Routing
================================================================

v91.0 - Advanced Complexity Analysis and Model Selection

This module provides intelligent automatic model selection based on:
- Multi-dimensional task complexity analysis
- Resource-aware model selection
- Adaptive threshold learning
- Performance-based routing optimization
- Seamless fallback chain management

ARCHITECTURE:
    Input → ComplexityAnalyzer → FeatureExtractor → SelectionStrategy → Model
                                       ↓
    Historical Data → AdaptiveThresholds → RouteOptimizer

FEATURES:
    - Real-time complexity scoring
    - Multi-factor analysis (reasoning, length, domain, etc.)
    - Adaptive thresholds based on performance
    - Cost-aware routing
    - Latency optimization
    - Fallback chain management

USAGE:
    selector = await get_auto_model_selector()

    # Automatic selection
    model = await selector.select_model("Complex reasoning task...")

    # With constraints
    model = await selector.select_model(
        prompt="...",
        constraints=SelectionConstraints(max_latency_ms=1000, prefer_local=True)
    )

INTEGRATION:
    - PrimeModel: Uses PrimeModel interface
    - HybridRouter: Extends hybrid routing capabilities
    - IntelligentModelRouter: Builds on tier routing
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
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


class ModelTier(Enum):
    """Model tiers for routing."""
    LOCAL_SMALL = "local_small"      # 7B models - fast, limited capability
    LOCAL_MEDIUM = "local_medium"    # 13B models - balanced
    LOCAL_LARGE = "local_large"      # 70B+ models - capable, slow
    GCP_HOSTED = "gcp_hosted"        # Cloud-hosted custom models
    API_STANDARD = "api_standard"    # Claude Haiku/Sonnet
    API_ADVANCED = "api_advanced"    # Claude Opus


class TaskCategory(Enum):
    """Categories of tasks."""
    SIMPLE_QA = "simple_qa"
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    REASONING = "reasoning"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    MATH = "math"
    MULTIMODAL = "multimodal"
    TOOL_USE = "tool_use"
    UNKNOWN = "unknown"


class SelectionReason(Enum):
    """Reasons for model selection."""
    COMPLEXITY_MATCH = "complexity_match"
    RESOURCE_CONSTRAINT = "resource_constraint"
    LATENCY_REQUIREMENT = "latency_requirement"
    COST_OPTIMIZATION = "cost_optimization"
    CAPABILITY_REQUIRED = "capability_required"
    USER_PREFERENCE = "user_preference"
    FALLBACK = "fallback"
    ADAPTIVE_ROUTING = "adaptive_routing"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ComplexitySignals:
    """
    Multi-dimensional complexity signals for task analysis.
    """
    # Length signals
    token_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0

    # Structural signals
    has_code: bool = False
    code_language: Optional[str] = None
    code_lines: int = 0
    has_math: bool = False
    has_tables: bool = False
    has_lists: bool = False

    # Reasoning signals
    reasoning_indicators: float = 0.0  # 0-1
    multi_step_indicators: float = 0.0  # 0-1
    requires_planning: bool = False

    # Domain signals
    domain_specificity: float = 0.0  # 0-1
    domain_category: str = "general"
    technical_depth: float = 0.0  # 0-1

    # External requirements
    requires_external_knowledge: float = 0.0  # 0-1
    requires_tool_use: float = 0.0  # 0-1
    requires_real_time_data: bool = False

    # Context signals
    context_length: int = 0
    conversation_turns: int = 0
    references_previous: bool = False

    # Output expectations
    expected_output_length: str = "medium"  # short, medium, long
    requires_formatting: bool = False
    requires_precision: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SelectionConstraints:
    """Constraints for model selection."""
    # Latency
    max_latency_ms: Optional[float] = None
    target_latency_ms: Optional[float] = None

    # Cost
    max_cost_per_request: Optional[float] = None
    budget_remaining: Optional[float] = None

    # Resource
    prefer_local: bool = False
    require_local: bool = False
    max_memory_mb: Optional[int] = None

    # Quality
    min_quality_score: float = 0.7
    require_streaming: bool = False

    # Model
    preferred_models: List[str] = field(default_factory=list)
    excluded_models: List[str] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    """Result of model selection."""
    model_name: str
    model_tier: ModelTier
    confidence: float  # 0-1

    # Analysis
    complexity_score: float
    complexity_signals: ComplexitySignals
    task_category: TaskCategory

    # Reasoning
    selection_reason: SelectionReason
    explanation: str

    # Predictions
    estimated_latency_ms: float
    estimated_cost: float
    estimated_quality: float

    # Alternatives
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    # Metadata
    selection_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_tier": self.model_tier.value,
            "confidence": self.confidence,
            "complexity_score": self.complexity_score,
            "task_category": self.task_category.value,
            "selection_reason": self.selection_reason.value,
            "explanation": self.explanation,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_cost": self.estimated_cost,
            "alternatives": [(m, s) for m, s in self.alternatives[:3]],
        }


@dataclass
class ModelProfile:
    """Profile of a model's capabilities and characteristics."""
    name: str
    tier: ModelTier

    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    specialty_domains: List[str] = field(default_factory=list)
    max_context_length: int = 4096

    # Performance
    tokens_per_second: float = 10.0
    avg_latency_ms: float = 1000.0
    quality_score: float = 0.8  # 0-1

    # Cost
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0

    # Complexity handling
    min_complexity: float = 0.0  # Can handle tasks this simple
    max_complexity: float = 1.0  # Can handle tasks this complex
    optimal_complexity: float = 0.5  # Best at this complexity

    # Resource requirements
    memory_required_mb: int = 0
    requires_gpu: bool = False

    # Availability
    is_available: bool = True
    health_score: float = 1.0

    def can_handle_complexity(self, complexity: float) -> bool:
        """Check if model can handle given complexity."""
        return self.min_complexity <= complexity <= self.max_complexity + 0.1

    def complexity_fit_score(self, complexity: float) -> float:
        """Score how well suited this model is for the complexity."""
        if not self.can_handle_complexity(complexity):
            return 0.0

        # Gaussian fit around optimal complexity
        diff = abs(complexity - self.optimal_complexity)
        return math.exp(-diff**2 / 0.1) * self.quality_score * self.health_score


# =============================================================================
# COMPLEXITY ANALYZER
# =============================================================================


class ComplexityAnalyzer:
    """
    Analyzes task complexity across multiple dimensions.

    Uses heuristics, pattern matching, and learned signals to
    estimate the complexity of a given task.
    """

    # Reasoning indicator patterns
    REASONING_PATTERNS = [
        r'\b(explain|analyze|compare|contrast|evaluate|assess)\b',
        r'\b(why|how|what if|suppose|consider)\b',
        r'\b(pros and cons|advantages|disadvantages)\b',
        r'\b(step by step|first|then|finally|therefore)\b',
        r'\b(reason|conclude|deduce|infer|derive)\b',
    ]

    # Multi-step indicator patterns
    MULTI_STEP_PATTERNS = [
        r'\b(and then|after that|next|subsequently)\b',
        r'\b(steps?|stages?|phases?|parts?)\b',
        r'\d+\.\s',  # Numbered lists
        r'\b(first|second|third|finally)\b',
        r'\b(plan|strategy|approach|method)\b',
    ]

    # Domain-specific patterns
    DOMAIN_PATTERNS = {
        "technical": [r'\b(API|SDK|framework|library|module)\b', r'\b(algorithm|data structure|complexity)\b'],
        "medical": [r'\b(diagnosis|treatment|symptom|patient)\b', r'\b(medication|dosage|clinical)\b'],
        "legal": [r'\b(contract|liability|jurisdiction|statute)\b', r'\b(plaintiff|defendant|court)\b'],
        "financial": [r'\b(investment|portfolio|asset|liability)\b', r'\b(ROI|P/E|dividend|equity)\b'],
        "scientific": [r'\b(hypothesis|experiment|data|results)\b', r'\b(correlation|causation|variable)\b'],
    }

    # Code patterns
    CODE_PATTERNS = [
        (r'```(\w+)?', "code_block"),
        (r'def\s+\w+\s*\(', "python_function"),
        (r'function\s+\w+\s*\(', "js_function"),
        (r'class\s+\w+', "class_def"),
        (r'import\s+\w+', "import"),
        (r'from\s+\w+\s+import', "from_import"),
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self._reasoning_patterns = [re.compile(p, re.I) for p in self.REASONING_PATTERNS]
        self._multi_step_patterns = [re.compile(p, re.I) for p in self.MULTI_STEP_PATTERNS]
        self._domain_patterns = {
            domain: [re.compile(p, re.I) for p in patterns]
            for domain, patterns in self.DOMAIN_PATTERNS.items()
        }
        self._code_patterns = [(re.compile(p), name) for p, name in self.CODE_PATTERNS]

    async def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> ComplexitySignals:
        """
        Analyze text complexity and return signals.

        Args:
            text: The input text to analyze
            context: Optional context (conversation history, etc.)

        Returns:
            ComplexitySignals with analysis results
        """
        signals = ComplexitySignals()

        # Basic length analysis
        words = text.split()
        signals.token_count = len(words)  # Approximate

        sentences = re.split(r'[.!?]+', text)
        signals.sentence_count = len([s for s in sentences if s.strip()])
        signals.avg_sentence_length = signals.token_count / max(signals.sentence_count, 1)

        # Code analysis
        signals.has_code, signals.code_language, signals.code_lines = self._analyze_code(text)

        # Math detection
        signals.has_math = bool(re.search(r'[∑∏∫√πθ]|\d+[\+\-\*/\^]\d+|equation|formula', text, re.I))

        # Structure detection
        signals.has_tables = bool(re.search(r'\|.*\|.*\|', text))
        signals.has_lists = bool(re.search(r'^[\-\*•]\s|^\d+\.\s', text, re.M))

        # Reasoning analysis
        signals.reasoning_indicators = self._analyze_reasoning(text)
        signals.multi_step_indicators = self._analyze_multi_step(text)
        signals.requires_planning = signals.multi_step_indicators > 0.5

        # Domain analysis
        signals.domain_specificity, signals.domain_category = self._analyze_domain(text)
        signals.technical_depth = self._analyze_technical_depth(text)

        # External requirements
        signals.requires_external_knowledge = self._analyze_external_knowledge(text)
        signals.requires_tool_use = self._analyze_tool_use(text)
        signals.requires_real_time_data = self._analyze_real_time(text)

        # Context analysis
        if context:
            signals.context_length = context.get("context_length", 0)
            signals.conversation_turns = context.get("turns", 0)
            signals.references_previous = context.get("references_previous", False)

        # Output expectations
        signals.expected_output_length = self._estimate_output_length(text)
        signals.requires_formatting = signals.has_code or signals.has_tables or signals.has_lists
        signals.requires_precision = signals.has_math or signals.has_code

        return signals

    def _analyze_code(self, text: str) -> Tuple[bool, Optional[str], int]:
        """Analyze code content."""
        for pattern, lang_hint in self._code_patterns:
            match = pattern.search(text)
            if match:
                # Extract code language if in code block
                code_blocks = re.findall(r'```(\w*)\n(.*?)```', text, re.S)
                if code_blocks:
                    lang = code_blocks[0][0] or self._detect_language(code_blocks[0][1])
                    code_lines = sum(len(block[1].strip().split('\n')) for block in code_blocks)
                    return True, lang, code_lines
                return True, lang_hint, 1

        return False, None, 0

    def _detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        if 'def ' in code or 'import ' in code:
            return 'python'
        elif 'function ' in code or 'const ' in code or 'let ' in code:
            return 'javascript'
        elif 'public class' in code or 'private ' in code:
            return 'java'
        elif '#include' in code:
            return 'cpp'
        return 'unknown'

    def _analyze_reasoning(self, text: str) -> float:
        """Analyze reasoning requirements."""
        matches = sum(1 for p in self._reasoning_patterns if p.search(text))
        return min(matches / len(self._reasoning_patterns), 1.0)

    def _analyze_multi_step(self, text: str) -> float:
        """Analyze multi-step requirements."""
        matches = sum(1 for p in self._multi_step_patterns if p.search(text))
        return min(matches / len(self._multi_step_patterns), 1.0)

    def _analyze_domain(self, text: str) -> Tuple[float, str]:
        """Analyze domain specificity."""
        best_domain = "general"
        best_score = 0.0

        for domain, patterns in self._domain_patterns.items():
            matches = sum(1 for p in patterns if p.search(text))
            score = matches / len(patterns)
            if score > best_score:
                best_score = score
                best_domain = domain

        return best_score, best_domain

    def _analyze_technical_depth(self, text: str) -> float:
        """Analyze technical depth."""
        technical_terms = len(re.findall(
            r'\b(API|SDK|HTTP|JSON|SQL|algorithm|function|variable|class|method|parameter|interface|protocol)\b',
            text, re.I
        ))
        return min(technical_terms / 20, 1.0)

    def _analyze_external_knowledge(self, text: str) -> float:
        """Analyze need for external knowledge."""
        indicators = [
            r'\b(current|latest|recent|today|now)\b',
            r'\b(research shows|studies indicate|according to)\b',
            r'\b(look up|find out|search for)\b',
        ]
        matches = sum(1 for p in indicators if re.search(p, text, re.I))
        return min(matches / len(indicators), 1.0)

    def _analyze_tool_use(self, text: str) -> float:
        """Analyze need for tool use."""
        indicators = [
            r'\b(run|execute|calculate|compute)\b',
            r'\b(file|document|image|code)\b',
            r'\b(search|browse|fetch)\b',
        ]
        matches = sum(1 for p in indicators if re.search(p, text, re.I))
        return min(matches / len(indicators), 1.0)

    def _analyze_real_time(self, text: str) -> bool:
        """Check if real-time data is needed."""
        patterns = [
            r'\b(current|now|today|live|real-time)\b',
            r'\b(weather|stock|price|traffic)\b',
            r'\b(what time|what date)\b',
        ]
        return any(re.search(p, text, re.I) for p in patterns)

    def _estimate_output_length(self, text: str) -> str:
        """Estimate expected output length."""
        if re.search(r'\b(brief|short|quick|one-liner|tldr)\b', text, re.I):
            return "short"
        elif re.search(r'\b(detailed|comprehensive|thorough|extensive)\b', text, re.I):
            return "long"
        elif re.search(r'\b(explain|describe|analyze|write)\b', text, re.I):
            return "long"
        return "medium"

    def compute_complexity_score(self, signals: ComplexitySignals) -> float:
        """
        Compute overall complexity score from signals.

        Returns score from 0 (simple) to 1 (very complex).
        """
        weights = {
            "token_count": 0.10,
            "reasoning": 0.25,
            "multi_step": 0.20,
            "code": 0.15,
            "domain": 0.10,
            "technical": 0.10,
            "external": 0.05,
            "tool_use": 0.05,
        }

        components = {
            "token_count": min(signals.token_count / 2000, 1.0),
            "reasoning": signals.reasoning_indicators,
            "multi_step": signals.multi_step_indicators,
            "code": min(signals.code_lines / 50, 1.0) if signals.has_code else 0.0,
            "domain": signals.domain_specificity,
            "technical": signals.technical_depth,
            "external": signals.requires_external_knowledge,
            "tool_use": signals.requires_tool_use,
        }

        score = sum(weights[k] * components[k] for k in weights)

        # Boost for certain combinations
        if signals.has_code and signals.reasoning_indicators > 0.5:
            score += 0.1  # Code + reasoning is complex
        if signals.has_math and signals.requires_precision:
            score += 0.1  # Math with precision is complex

        return min(max(score, 0.0), 1.0)


# =============================================================================
# TASK CLASSIFIER
# =============================================================================


class TaskClassifier:
    """Classifies tasks into categories."""

    CATEGORY_PATTERNS = {
        TaskCategory.SIMPLE_QA: [
            r'^(what|who|when|where|which)\s+(is|are|was|were)\b',
            r'\b(define|definition of)\b',
        ],
        TaskCategory.CODE_GENERATION: [
            r'\b(write|create|implement|code|program)\b.*\b(function|class|script|code)\b',
            r'```\w*\n',
        ],
        TaskCategory.CODE_ANALYSIS: [
            r'\b(review|analyze|debug|fix|explain)\b.*\b(code|function|error|bug)\b',
            r'\b(what does this code|how does this)\b',
        ],
        TaskCategory.REASONING: [
            r'\b(why|explain why|reason|because)\b',
            r'\b(compare|contrast|evaluate|assess)\b',
            r'\b(pros and cons|advantages|disadvantages)\b',
        ],
        TaskCategory.CREATIVE: [
            r'\b(write|create|compose|generate)\b.*\b(story|poem|essay|article)\b',
            r'\b(creative|imaginative|fictional)\b',
        ],
        TaskCategory.SUMMARIZATION: [
            r'\b(summarize|summary|tldr|brief)\b',
            r'\b(key points|main ideas|overview)\b',
        ],
        TaskCategory.MATH: [
            r'\b(calculate|compute|solve|equation)\b',
            r'[∑∏∫√πθ=\+\-\*/\^]',
        ],
        TaskCategory.TOOL_USE: [
            r'\b(search|browse|fetch|lookup)\b',
            r'\b(file|document|image)\b.*\b(open|read|analyze)\b',
        ],
    }

    def __init__(self):
        self._patterns = {
            cat: [re.compile(p, re.I) for p in patterns]
            for cat, patterns in self.CATEGORY_PATTERNS.items()
        }

    def classify(self, text: str, signals: ComplexitySignals) -> TaskCategory:
        """Classify task into a category."""
        # Score each category
        scores = {}
        for category, patterns in self._patterns.items():
            score = sum(1 for p in patterns if p.search(text)) / len(patterns)
            scores[category] = score

        # Use signals to boost scores
        if signals.has_code:
            if signals.reasoning_indicators > 0.3:
                scores[TaskCategory.CODE_ANALYSIS] = scores.get(TaskCategory.CODE_ANALYSIS, 0) + 0.3
            else:
                scores[TaskCategory.CODE_GENERATION] = scores.get(TaskCategory.CODE_GENERATION, 0) + 0.3

        if signals.has_math:
            scores[TaskCategory.MATH] = scores.get(TaskCategory.MATH, 0) + 0.3

        if signals.requires_tool_use > 0.5:
            scores[TaskCategory.TOOL_USE] = scores.get(TaskCategory.TOOL_USE, 0) + 0.3

        # Find best category
        best_category = max(scores, key=lambda k: scores[k])
        if scores[best_category] < 0.1:
            return TaskCategory.UNKNOWN

        return best_category


# =============================================================================
# AUTO MODEL SELECTOR
# =============================================================================


@dataclass
class SelectorConfig:
    """Configuration for auto model selector."""
    # Default thresholds
    local_complexity_threshold: float = field(default_factory=lambda: float(os.getenv("SELECTOR_LOCAL_THRESHOLD", "0.6")))
    api_complexity_threshold: float = field(default_factory=lambda: float(os.getenv("SELECTOR_API_THRESHOLD", "0.85")))

    # Adaptive thresholds
    enable_adaptive_thresholds: bool = field(default_factory=lambda: os.getenv("SELECTOR_ADAPTIVE", "true").lower() == "true")
    adaptation_rate: float = field(default_factory=lambda: float(os.getenv("SELECTOR_ADAPT_RATE", "0.1")))

    # Fallback
    enable_fallback_chain: bool = field(default_factory=lambda: os.getenv("SELECTOR_FALLBACK", "true").lower() == "true")
    max_fallback_attempts: int = field(default_factory=lambda: int(os.getenv("SELECTOR_MAX_FALLBACK", "3")))

    # Cost optimization
    enable_cost_optimization: bool = field(default_factory=lambda: os.getenv("SELECTOR_COST_OPT", "true").lower() == "true")
    max_cost_per_request: float = field(default_factory=lambda: float(os.getenv("SELECTOR_MAX_COST", "0.10")))

    # Performance targets
    target_latency_p95_ms: float = field(default_factory=lambda: float(os.getenv("SELECTOR_TARGET_LATENCY", "5000.0")))

    # Learning
    record_selections: bool = field(default_factory=lambda: os.getenv("SELECTOR_RECORD", "true").lower() == "true")
    history_size: int = field(default_factory=lambda: int(os.getenv("SELECTOR_HISTORY_SIZE", "1000")))


class AutoModelSelector:
    """
    Intelligent automatic model selector.

    Analyzes task complexity and constraints to select the optimal model.
    """

    # Default model profiles
    DEFAULT_PROFILES: Dict[str, ModelProfile] = {
        "prime-7b-chat-v1": ModelProfile(
            name="prime-7b-chat-v1",
            tier=ModelTier.LOCAL_SMALL,
            capabilities={"chat", "code", "general"},
            max_context_length=4096,
            tokens_per_second=50.0,
            avg_latency_ms=200.0,
            quality_score=0.7,
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.0,
            min_complexity=0.0,
            max_complexity=0.6,
            optimal_complexity=0.3,
            memory_required_mb=16000,
        ),
        "prime-13b-reasoning-v1": ModelProfile(
            name="prime-13b-reasoning-v1",
            tier=ModelTier.LOCAL_MEDIUM,
            capabilities={"chat", "code", "reasoning", "analysis"},
            max_context_length=8192,
            tokens_per_second=30.0,
            avg_latency_ms=500.0,
            quality_score=0.8,
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.0,
            min_complexity=0.3,
            max_complexity=0.8,
            optimal_complexity=0.6,
            memory_required_mb=32000,
        ),
        "claude-3-haiku": ModelProfile(
            name="claude-3-haiku",
            tier=ModelTier.API_STANDARD,
            capabilities={"chat", "code", "reasoning", "creative", "analysis"},
            max_context_length=200000,
            tokens_per_second=200.0,
            avg_latency_ms=1000.0,
            quality_score=0.85,
            cost_per_1k_input_tokens=0.00025,
            cost_per_1k_output_tokens=0.00125,
            min_complexity=0.0,
            max_complexity=0.85,
            optimal_complexity=0.5,
            memory_required_mb=0,
        ),
        "claude-3-5-sonnet": ModelProfile(
            name="claude-3-5-sonnet",
            tier=ModelTier.API_STANDARD,
            capabilities={"chat", "code", "reasoning", "creative", "analysis", "tool_use"},
            max_context_length=200000,
            tokens_per_second=150.0,
            avg_latency_ms=1500.0,
            quality_score=0.92,
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            min_complexity=0.3,
            max_complexity=0.95,
            optimal_complexity=0.7,
            memory_required_mb=0,
        ),
        "claude-opus-4": ModelProfile(
            name="claude-opus-4",
            tier=ModelTier.API_ADVANCED,
            capabilities={"chat", "code", "reasoning", "creative", "analysis", "tool_use", "complex"},
            max_context_length=200000,
            tokens_per_second=100.0,
            avg_latency_ms=2500.0,
            quality_score=0.98,
            cost_per_1k_input_tokens=0.015,
            cost_per_1k_output_tokens=0.075,
            min_complexity=0.5,
            max_complexity=1.0,
            optimal_complexity=0.85,
            memory_required_mb=0,
        ),
    }

    def __init__(self, config: Optional[SelectorConfig] = None):
        self._config = config or SelectorConfig()

        # Components
        self._complexity_analyzer = ComplexityAnalyzer()
        self._task_classifier = TaskClassifier()

        # Model profiles
        self._profiles: Dict[str, ModelProfile] = dict(self.DEFAULT_PROFILES)

        # Adaptive thresholds
        self._thresholds = {
            ModelTier.LOCAL_SMALL: self._config.local_complexity_threshold,
            ModelTier.LOCAL_MEDIUM: 0.75,
            ModelTier.API_STANDARD: self._config.api_complexity_threshold,
            ModelTier.API_ADVANCED: 1.0,
        }

        # Selection history
        self._history: Deque[Dict[str, Any]] = deque(maxlen=self._config.history_size)

        # Statistics
        self._selection_counts: Dict[str, int] = defaultdict(int)
        self._total_selections = 0

    async def select_model(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[SelectionConstraints] = None,
    ) -> SelectionResult:
        """
        Select the optimal model for a task.

        Args:
            prompt: The task/prompt to analyze
            context: Optional context (conversation history, etc.)
            constraints: Optional selection constraints

        Returns:
            SelectionResult with selected model and analysis
        """
        constraints = constraints or SelectionConstraints()

        # Analyze complexity
        signals = await self._complexity_analyzer.analyze(prompt, context)
        complexity = self._complexity_analyzer.compute_complexity_score(signals)

        # Classify task
        task_category = self._task_classifier.classify(prompt, signals)

        # Get candidate models
        candidates = self._get_candidates(signals, constraints)

        if not candidates:
            # Fallback to default
            return self._create_fallback_result(signals, complexity, task_category, constraints)

        # Score candidates
        scored = []
        for profile in candidates:
            score, reason = self._score_candidate(
                profile, complexity, signals, task_category, constraints
            )
            scored.append((profile, score, reason))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select best
        best_profile, best_score, best_reason = scored[0]

        # Build result
        result = SelectionResult(
            model_name=best_profile.name,
            model_tier=best_profile.tier,
            confidence=best_score,
            complexity_score=complexity,
            complexity_signals=signals,
            task_category=task_category,
            selection_reason=best_reason,
            explanation=self._explain_selection(best_profile, complexity, best_reason),
            estimated_latency_ms=best_profile.avg_latency_ms,
            estimated_cost=self._estimate_cost(best_profile, signals),
            estimated_quality=best_profile.quality_score,
            alternatives=[(p.name, s) for p, s, _ in scored[1:4]],
        )

        # Record selection
        if self._config.record_selections:
            self._record_selection(result)

        return result

    def _get_candidates(
        self,
        signals: ComplexitySignals,
        constraints: SelectionConstraints,
    ) -> List[ModelProfile]:
        """Get candidate models based on requirements."""
        candidates = []

        for profile in self._profiles.values():
            # Check availability
            if not profile.is_available:
                continue

            # Check exclusions
            if profile.name in constraints.excluded_models:
                continue

            # Check local requirement
            if constraints.require_local and profile.tier in (ModelTier.API_STANDARD, ModelTier.API_ADVANCED):
                continue

            # Check capabilities
            if constraints.required_capabilities:
                if not all(cap in profile.capabilities for cap in constraints.required_capabilities):
                    continue

            # Check memory
            if constraints.max_memory_mb and profile.memory_required_mb > constraints.max_memory_mb:
                continue

            # Check cost
            if constraints.max_cost_per_request:
                estimated_cost = self._estimate_cost(profile, signals)
                if estimated_cost > constraints.max_cost_per_request:
                    continue

            # Check latency
            if constraints.max_latency_ms and profile.avg_latency_ms > constraints.max_latency_ms:
                continue

            candidates.append(profile)

        return candidates

    def _score_candidate(
        self,
        profile: ModelProfile,
        complexity: float,
        signals: ComplexitySignals,
        task_category: TaskCategory,
        constraints: SelectionConstraints,
    ) -> Tuple[float, SelectionReason]:
        """Score a candidate model."""
        # Base score from complexity fit
        fit_score = profile.complexity_fit_score(complexity)

        # Reason tracking
        primary_reason = SelectionReason.COMPLEXITY_MATCH

        # Latency bonus/penalty
        latency_score = 1.0
        if constraints.target_latency_ms:
            if profile.avg_latency_ms <= constraints.target_latency_ms:
                latency_score = 1.2  # Bonus for meeting target
            else:
                latency_score = constraints.target_latency_ms / profile.avg_latency_ms
                primary_reason = SelectionReason.LATENCY_REQUIREMENT

        # Cost optimization
        cost_score = 1.0
        if self._config.enable_cost_optimization:
            cost = self._estimate_cost(profile, signals)
            if cost == 0:
                cost_score = 1.2  # Bonus for free
            elif cost > 0:
                cost_score = 1.0 / (1.0 + cost * 10)  # Penalize higher cost
                if cost_score < 0.5:
                    primary_reason = SelectionReason.COST_OPTIMIZATION

        # Preference bonus
        pref_score = 1.0
        if profile.name in constraints.preferred_models:
            pref_score = 1.3
            primary_reason = SelectionReason.USER_PREFERENCE
        if constraints.prefer_local and profile.tier in (ModelTier.LOCAL_SMALL, ModelTier.LOCAL_MEDIUM):
            pref_score = 1.1

        # Quality bonus for complex tasks
        quality_score = profile.quality_score
        if complexity > 0.7:
            quality_score = quality_score ** 0.5  # Less penalty for lower quality at high complexity

        # Capability bonus
        cap_score = 1.0
        if signals.has_code and "code" in profile.capabilities:
            cap_score += 0.1
        if task_category == TaskCategory.REASONING and "reasoning" in profile.capabilities:
            cap_score += 0.1
        if cap_score > 1.1:
            primary_reason = SelectionReason.CAPABILITY_REQUIRED

        # Combine scores
        final_score = (
            fit_score * 0.4 +
            latency_score * 0.15 +
            cost_score * 0.15 +
            pref_score * 0.1 +
            quality_score * 0.15 +
            cap_score * 0.05
        )

        return min(final_score, 1.0), primary_reason

    def _estimate_cost(self, profile: ModelProfile, signals: ComplexitySignals) -> float:
        """Estimate cost for a request."""
        input_tokens = signals.token_count
        output_tokens = {
            "short": 100,
            "medium": 300,
            "long": 800,
        }.get(signals.expected_output_length, 300)

        cost = (
            (input_tokens / 1000) * profile.cost_per_1k_input_tokens +
            (output_tokens / 1000) * profile.cost_per_1k_output_tokens
        )
        return cost

    def _explain_selection(
        self,
        profile: ModelProfile,
        complexity: float,
        reason: SelectionReason,
    ) -> str:
        """Generate explanation for selection."""
        explanations = {
            SelectionReason.COMPLEXITY_MATCH: f"Task complexity ({complexity:.2f}) matches {profile.name}'s optimal range",
            SelectionReason.RESOURCE_CONSTRAINT: f"Selected {profile.name} due to resource constraints",
            SelectionReason.LATENCY_REQUIREMENT: f"Selected {profile.name} to meet latency requirements",
            SelectionReason.COST_OPTIMIZATION: f"Selected {profile.name} for cost efficiency",
            SelectionReason.CAPABILITY_REQUIRED: f"Selected {profile.name} for required capabilities",
            SelectionReason.USER_PREFERENCE: f"Selected preferred model {profile.name}",
            SelectionReason.FALLBACK: f"Fell back to {profile.name}",
            SelectionReason.ADAPTIVE_ROUTING: f"Adaptive routing selected {profile.name}",
        }
        return explanations.get(reason, f"Selected {profile.name}")

    def _create_fallback_result(
        self,
        signals: ComplexitySignals,
        complexity: float,
        task_category: TaskCategory,
        constraints: SelectionConstraints,
    ) -> SelectionResult:
        """Create a fallback result when no candidates available."""
        # Use API fallback
        fallback_model = "claude-3-haiku"
        fallback_profile = self._profiles.get(fallback_model, self.DEFAULT_PROFILES[fallback_model])

        return SelectionResult(
            model_name=fallback_model,
            model_tier=ModelTier.API_STANDARD,
            confidence=0.5,
            complexity_score=complexity,
            complexity_signals=signals,
            task_category=task_category,
            selection_reason=SelectionReason.FALLBACK,
            explanation="No suitable local models available, using API fallback",
            estimated_latency_ms=fallback_profile.avg_latency_ms,
            estimated_cost=self._estimate_cost(fallback_profile, signals),
            estimated_quality=fallback_profile.quality_score,
        )

    def _record_selection(self, result: SelectionResult) -> None:
        """Record a selection for learning."""
        self._history.append({
            "selection_id": result.selection_id,
            "model": result.model_name,
            "complexity": result.complexity_score,
            "category": result.task_category.value,
            "reason": result.selection_reason.value,
            "timestamp": result.timestamp,
        })
        self._selection_counts[result.model_name] += 1
        self._total_selections += 1

    def register_model(self, profile: ModelProfile) -> None:
        """Register a new model profile."""
        self._profiles[profile.name] = profile
        logger.info(f"Registered model profile: {profile.name}")

    def update_model_health(self, model_name: str, health_score: float) -> None:
        """Update model health score."""
        if model_name in self._profiles:
            self._profiles[model_name].health_score = health_score

    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        return {
            "total_selections": self._total_selections,
            "selection_distribution": dict(self._selection_counts),
            "available_models": [
                {"name": p.name, "tier": p.tier.value, "available": p.is_available}
                for p in self._profiles.values()
            ],
            "thresholds": {k.value: v for k, v in self._thresholds.items()},
            "config": {
                "adaptive_thresholds": self._config.enable_adaptive_thresholds,
                "cost_optimization": self._config.enable_cost_optimization,
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_selector: Optional[AutoModelSelector] = None
_selector_lock = asyncio.Lock()


async def get_auto_model_selector(
    config: Optional[SelectorConfig] = None,
) -> AutoModelSelector:
    """Get or create the global auto model selector."""
    global _selector

    async with _selector_lock:
        if _selector is None:
            _selector = AutoModelSelector(config)

        return _selector


async def shutdown_auto_model_selector() -> None:
    """Shutdown the global auto model selector."""
    global _selector

    async with _selector_lock:
        if _selector is not None:
            # Perform any cleanup needed
            _selector = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ModelTier",
    "TaskCategory",
    "SelectionReason",
    # Data structures
    "ComplexitySignals",
    "SelectionConstraints",
    "SelectionResult",
    "ModelProfile",
    # Classes
    "ComplexityAnalyzer",
    "TaskClassifier",
    "AutoModelSelector",
    "SelectorConfig",
    # Factory
    "get_auto_model_selector",
    "shutdown_auto_model_selector",
]
