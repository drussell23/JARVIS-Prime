"""
Hybrid Router - Intelligent Tier 0/1 Classification
====================================================

Routes prompts between local JARVIS-Prime (Tier 0) and cloud APIs (Tier 1)
based on complexity, capabilities, and cost optimization.

Classification Strategy:
- Tier 0 (Local): Simple tasks, formatting, basic code, summarization
- Tier 1 (Cloud): Complex reasoning, multi-step planning, specialized knowledge

Features:
- Async complexity scoring with multiple signals
- Dynamic threshold adjustment based on load
- Cost-aware routing decisions
- OpenAI-compatible API interface
- Fallback chain with retry logic
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class TierClassification(Enum):
    """Request tier classification"""
    TIER_0 = "tier_0"           # Local processing (JARVIS-Prime)
    TIER_1 = "tier_1"           # Cloud API (Claude, GPT-4, etc.)
    TIER_0_PREFERRED = "tier_0_preferred"  # Prefer local but fallback to cloud
    TIER_1_REQUIRED = "tier_1_required"    # Must use cloud (complex/specialized)


class TaskType(Enum):
    """Task type classification for routing decisions"""
    CHAT = "chat"               # General conversation
    CODE_SIMPLE = "code_simple" # Simple code tasks
    CODE_COMPLEX = "code_complex"  # Complex code tasks
    SUMMARIZE = "summarize"     # Text summarization
    FORMAT = "format"           # Formatting, templating
    ANALYZE = "analyze"         # Analysis tasks
    REASON = "reason"           # Complex reasoning
    CREATIVE = "creative"       # Creative writing
    SEARCH = "search"           # Knowledge retrieval
    MULTIMODAL = "multimodal"   # Vision, audio, etc.
    UNKNOWN = "unknown"


@dataclass
class ComplexitySignals:
    """
    Signals used to determine prompt complexity.

    All signals normalized to 0.0-1.0 scale.
    """
    # Token-based signals
    token_count: int = 0
    avg_token_length: float = 0.0

    # Structural signals
    nested_depth: int = 0           # Code nesting, JSON depth
    special_syntax_ratio: float = 0.0  # Code syntax, math notation

    # Semantic signals
    reasoning_indicators: float = 0.0  # "analyze", "compare", "explain why"
    domain_specificity: float = 0.0    # Specialized vocabulary
    multi_step_indicators: float = 0.0 # "first", "then", "finally"

    # Context signals
    requires_external_knowledge: float = 0.0
    requires_tool_use: float = 0.0
    requires_vision: float = 0.0

    # Historical signals
    similar_prompt_success_rate: float = 1.0  # How well Tier 0 handled similar

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_count": self.token_count,
            "avg_token_length": self.avg_token_length,
            "nested_depth": self.nested_depth,
            "special_syntax_ratio": self.special_syntax_ratio,
            "reasoning_indicators": self.reasoning_indicators,
            "domain_specificity": self.domain_specificity,
            "multi_step_indicators": self.multi_step_indicators,
            "requires_external_knowledge": self.requires_external_knowledge,
            "requires_tool_use": self.requires_tool_use,
            "requires_vision": self.requires_vision,
            "similar_prompt_success_rate": self.similar_prompt_success_rate,
        }


@dataclass
class RoutingDecision:
    """
    Result of routing classification.

    Includes tier assignment, confidence, and reasoning.
    """
    tier: TierClassification
    task_type: TaskType
    confidence: float               # 0.0-1.0 confidence in classification
    complexity_score: float         # 0.0-1.0 overall complexity
    signals: ComplexitySignals
    reasoning: str                  # Human-readable explanation

    # Cost estimation
    estimated_local_latency_ms: float = 0.0
    estimated_cloud_latency_ms: float = 0.0
    estimated_cloud_cost_usd: float = 0.0

    # Metadata
    prompt_hash: str = ""           # For caching similar prompts
    classified_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "task_type": self.task_type.value,
            "confidence": round(self.confidence, 3),
            "complexity_score": round(self.complexity_score, 3),
            "signals": self.signals.to_dict(),
            "reasoning": self.reasoning,
            "estimated_local_latency_ms": round(self.estimated_local_latency_ms, 1),
            "estimated_cloud_latency_ms": round(self.estimated_cloud_latency_ms, 1),
            "estimated_cloud_cost_usd": round(self.estimated_cloud_cost_usd, 5),
            "prompt_hash": self.prompt_hash,
            "classified_at": self.classified_at.isoformat(),
        }


class ComplexityAnalyzer(ABC):
    """Abstract base for complexity analysis strategies"""

    @abstractmethod
    async def analyze(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> ComplexitySignals:
        """Analyze prompt complexity"""
        ...


class DefaultComplexityAnalyzer(ComplexityAnalyzer):
    """
    Default complexity analyzer using heuristic rules.

    Fast, no external dependencies, good baseline.
    """

    # Patterns indicating high complexity
    REASONING_PATTERNS = [
        r"\banalyze\b", r"\bcompare\b", r"\bexplain\s+why\b", r"\bevaluate\b",
        r"\bcritique\b", r"\bsynthesiz\w*\b", r"\binfer\b", r"\bdeduc\w*\b",
        r"\bhypothesi[sz]\w*\b", r"\breason\s+through\b", r"\bthink\s+step\s+by\s+step\b",
    ]

    MULTI_STEP_PATTERNS = [
        r"\bfirst\b.*\bthen\b", r"\bstep\s+\d+\b", r"\bnext\b.*\bfinally\b",
        r"\bafter\s+that\b", r"\bonce\s+\w+\s+is\b", r"\bsequentially\b",
    ]

    TOOL_USE_PATTERNS = [
        r"\bsearch\s+for\b", r"\blook\s+up\b", r"\bfind\s+information\b",
        r"\bcalculate\b", r"\bconvert\b", r"\btranslate\b",
    ]

    DOMAIN_KEYWORDS = {
        "medical": ["diagnosis", "symptom", "treatment", "medication", "patient"],
        "legal": ["contract", "statute", "liability", "jurisdiction", "clause"],
        "financial": ["derivative", "portfolio", "hedge", "arbitrage", "yield"],
        "scientific": ["hypothesis", "methodology", "empirical", "correlation"],
    }

    CODE_PATTERNS = [
        r"```\w*\n", r"def\s+\w+\(", r"class\s+\w+", r"function\s+\w+",
        r"import\s+\w+", r"from\s+\w+\s+import", r"const\s+\w+\s*=",
    ]

    async def analyze(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> ComplexitySignals:
        """Analyze prompt using heuristic patterns"""
        signals = ComplexitySignals()
        prompt_lower = prompt.lower()

        # Token-based analysis
        tokens = prompt.split()
        signals.token_count = len(tokens)
        signals.avg_token_length = sum(len(t) for t in tokens) / max(len(tokens), 1)

        # Reasoning indicators
        reasoning_matches = sum(
            1 for pattern in self.REASONING_PATTERNS
            if re.search(pattern, prompt_lower)
        )
        signals.reasoning_indicators = min(reasoning_matches / 3.0, 1.0)

        # Multi-step indicators
        multi_step_matches = sum(
            1 for pattern in self.MULTI_STEP_PATTERNS
            if re.search(pattern, prompt_lower)
        )
        signals.multi_step_indicators = min(multi_step_matches / 2.0, 1.0)

        # Tool use indicators
        tool_matches = sum(
            1 for pattern in self.TOOL_USE_PATTERNS
            if re.search(pattern, prompt_lower)
        )
        signals.requires_tool_use = min(tool_matches / 2.0, 1.0)

        # Domain specificity
        domain_score = 0.0
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in prompt_lower)
            if matches >= 2:
                domain_score = max(domain_score, matches / len(keywords))
        signals.domain_specificity = min(domain_score, 1.0)

        # Code detection
        code_matches = sum(1 for p in self.CODE_PATTERNS if re.search(p, prompt))
        signals.special_syntax_ratio = min(code_matches / 3.0, 1.0)

        # Nested depth (JSON, code blocks)
        signals.nested_depth = self._calculate_nesting(prompt)

        # Context signals
        if context:
            signals.requires_vision = 1.0 if context.get("has_images") else 0.0
            signals.requires_external_knowledge = 1.0 if context.get("needs_current_info") else 0.0

        return signals

    def _calculate_nesting(self, text: str) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0
        open_chars = {'{': '}', '[': ']', '(': ')'}
        close_chars = set(open_chars.values())

        for char in text:
            if char in open_chars:
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in close_chars:
                current_depth = max(0, current_depth - 1)

        return max_depth


class HybridRouter:
    """
    Intelligent router for Tier 0/1 classification.

    Classifies prompts and routes to local (JARVIS-Prime) or cloud APIs
    based on complexity, cost, and capability requirements.

    Usage:
        router = HybridRouter()

        # Classify a prompt
        decision = await router.classify(prompt)

        if decision.tier == TierClassification.TIER_0:
            # Route to local JARVIS-Prime
            response = await local_model.generate(prompt)
        else:
            # Route to cloud API
            response = await cloud_api.complete(prompt)

        # Record outcome for learning
        router.record_outcome(decision.prompt_hash, success=True)
    """

    # Default thresholds (can be overridden via config)
    DEFAULT_THRESHOLDS = {
        "complexity_tier_1": 0.65,      # Above this → Tier 1
        "complexity_tier_0_max": 0.40,  # Below this → definitely Tier 0
        "token_count_tier_1": 2000,     # Long prompts → Tier 1
        "code_complexity_tier_1": 0.7,  # Complex code → Tier 1
    }

    # Task type routing preferences
    TASK_TYPE_TIERS = {
        TaskType.CHAT: TierClassification.TIER_0_PREFERRED,
        TaskType.CODE_SIMPLE: TierClassification.TIER_0,
        TaskType.CODE_COMPLEX: TierClassification.TIER_1_REQUIRED,
        TaskType.SUMMARIZE: TierClassification.TIER_0,
        TaskType.FORMAT: TierClassification.TIER_0,
        TaskType.ANALYZE: TierClassification.TIER_0_PREFERRED,
        TaskType.REASON: TierClassification.TIER_1_REQUIRED,
        TaskType.CREATIVE: TierClassification.TIER_0_PREFERRED,
        TaskType.SEARCH: TierClassification.TIER_1,
        TaskType.MULTIMODAL: TierClassification.TIER_1_REQUIRED,
        TaskType.UNKNOWN: TierClassification.TIER_0_PREFERRED,
    }

    def __init__(
        self,
        analyzer: Optional[ComplexityAnalyzer] = None,
        thresholds: Optional[Dict[str, float]] = None,
        history_file: Optional[Path] = None,
        enable_learning: bool = True,
    ):
        self.analyzer = analyzer or DefaultComplexityAnalyzer()
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.history_file = history_file
        self.enable_learning = enable_learning

        # Learning state
        self._prompt_outcomes: Dict[str, Tuple[int, int]] = {}  # hash → (success, total)
        self._load_history()

        # Statistics
        self._total_classifications = 0
        self._tier_0_count = 0
        self._tier_1_count = 0

        logger.info("HybridRouter initialized")

    def _load_history(self) -> None:
        """Load classification history for learning"""
        if self.history_file and self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self._prompt_outcomes = {
                    k: tuple(v) for k, v in data.get("outcomes", {}).items()
                }
                logger.info(f"Loaded {len(self._prompt_outcomes)} historical outcomes")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

    def _save_history(self) -> None:
        """Save classification history"""
        if self.history_file:
            try:
                data = {
                    "outcomes": {k: list(v) for k, v in self._prompt_outcomes.items()},
                    "updated_at": datetime.now().isoformat(),
                }
                self.history_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.history_file, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save history: {e}")

    async def classify(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        force_tier: Optional[TierClassification] = None,
    ) -> RoutingDecision:
        """
        Classify a prompt for tier routing.

        Args:
            prompt: The prompt to classify
            context: Optional context (images, tool requirements, etc.)
            force_tier: Force a specific tier (bypass classification)

        Returns:
            RoutingDecision with tier assignment and metadata
        """
        self._total_classifications += 1

        # Generate prompt hash for caching/learning
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]

        # Get complexity signals
        signals = await self.analyzer.analyze(prompt, context)

        # Check historical success rate
        if prompt_hash in self._prompt_outcomes:
            success, total = self._prompt_outcomes[prompt_hash]
            signals.similar_prompt_success_rate = success / max(total, 1)

        # Detect task type
        task_type = self._detect_task_type(prompt, signals)

        # Calculate overall complexity
        complexity_score = self._calculate_complexity_score(signals)

        # Determine tier
        if force_tier:
            tier = force_tier
            confidence = 1.0
            reasoning = f"Forced to {force_tier.value}"
        else:
            tier, confidence, reasoning = self._determine_tier(
                complexity_score, signals, task_type, context
            )

        # Update statistics
        if tier in (TierClassification.TIER_0, TierClassification.TIER_0_PREFERRED):
            self._tier_0_count += 1
        else:
            self._tier_1_count += 1

        # Estimate costs/latencies
        token_estimate = signals.token_count + 500  # Response estimate

        decision = RoutingDecision(
            tier=tier,
            task_type=task_type,
            confidence=confidence,
            complexity_score=complexity_score,
            signals=signals,
            reasoning=reasoning,
            prompt_hash=prompt_hash,
            estimated_local_latency_ms=token_estimate * 10,  # ~10ms/token local
            estimated_cloud_latency_ms=token_estimate * 5 + 500,  # 5ms/token + network
            estimated_cloud_cost_usd=(token_estimate / 1000) * 0.003,  # $3/1M tokens estimate
        )

        logger.debug(f"Classified prompt: {tier.value} (complexity={complexity_score:.2f})")
        return decision

    def _detect_task_type(self, prompt: str, signals: ComplexitySignals) -> TaskType:
        """Detect the task type from prompt content"""
        prompt_lower = prompt.lower()

        # Multimodal check
        if signals.requires_vision > 0.5:
            return TaskType.MULTIMODAL

        # Code detection
        if signals.special_syntax_ratio > 0.3:
            if signals.reasoning_indicators > 0.5 or signals.multi_step_indicators > 0.5:
                return TaskType.CODE_COMPLEX
            return TaskType.CODE_SIMPLE

        # Summarization
        if any(kw in prompt_lower for kw in ["summarize", "summary", "tldr", "brief"]):
            return TaskType.SUMMARIZE

        # Formatting
        if any(kw in prompt_lower for kw in ["format", "convert", "template", "restructure"]):
            return TaskType.FORMAT

        # Reasoning
        if signals.reasoning_indicators > 0.5:
            return TaskType.REASON

        # Analysis
        if signals.multi_step_indicators > 0.3:
            return TaskType.ANALYZE

        # Creative
        if any(kw in prompt_lower for kw in ["write", "create", "compose", "story", "poem"]):
            return TaskType.CREATIVE

        # Search/knowledge
        if signals.requires_external_knowledge > 0.5 or signals.requires_tool_use > 0.5:
            return TaskType.SEARCH

        # Default to chat
        return TaskType.CHAT

    def _calculate_complexity_score(self, signals: ComplexitySignals) -> float:
        """Calculate overall complexity score from signals"""
        # Weighted combination of signals
        weights = {
            "token_count": 0.1,      # Normalized by 2000 tokens
            "reasoning": 0.25,
            "multi_step": 0.2,
            "domain": 0.15,
            "external_knowledge": 0.1,
            "tool_use": 0.1,
            "vision": 0.1,
        }

        score = (
            weights["token_count"] * min(signals.token_count / 2000, 1.0) +
            weights["reasoning"] * signals.reasoning_indicators +
            weights["multi_step"] * signals.multi_step_indicators +
            weights["domain"] * signals.domain_specificity +
            weights["external_knowledge"] * signals.requires_external_knowledge +
            weights["tool_use"] * signals.requires_tool_use +
            weights["vision"] * signals.requires_vision
        )

        return min(max(score, 0.0), 1.0)

    def _determine_tier(
        self,
        complexity_score: float,
        signals: ComplexitySignals,
        task_type: TaskType,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[TierClassification, float, str]:
        """Determine the tier classification"""
        reasons = []

        # Check task type preferences
        task_tier = self.TASK_TYPE_TIERS.get(task_type, TierClassification.TIER_0_PREFERRED)

        # Force Tier 1 for certain conditions
        if task_type == TaskType.MULTIMODAL:
            return TierClassification.TIER_1_REQUIRED, 0.95, "Multimodal requires cloud API"

        if signals.requires_vision > 0.5:
            return TierClassification.TIER_1_REQUIRED, 0.95, "Vision processing required"

        # Check complexity thresholds
        if complexity_score > self.thresholds["complexity_tier_1"]:
            reasons.append(f"high complexity ({complexity_score:.2f})")
            return TierClassification.TIER_1, 0.8, f"High complexity: {', '.join(reasons)}"

        if complexity_score < self.thresholds["complexity_tier_0_max"]:
            reasons.append(f"low complexity ({complexity_score:.2f})")
            return TierClassification.TIER_0, 0.9, f"Low complexity, ideal for local: {', '.join(reasons)}"

        # Token count check
        if signals.token_count > self.thresholds["token_count_tier_1"]:
            reasons.append(f"long prompt ({signals.token_count} tokens)")
            return TierClassification.TIER_1, 0.75, f"Long prompt: {', '.join(reasons)}"

        # Historical success check
        if signals.similar_prompt_success_rate < 0.6:
            reasons.append(f"poor historical success ({signals.similar_prompt_success_rate:.0%})")
            return TierClassification.TIER_1, 0.7, f"Historical failures: {', '.join(reasons)}"

        # Middle ground - use task type preference
        confidence = 0.6 + (0.3 * abs(complexity_score - 0.5))
        reasoning = f"Moderate complexity ({complexity_score:.2f}), using task type preference"

        return task_tier, confidence, reasoning

    def record_outcome(
        self,
        prompt_hash: str,
        success: bool,
        tier_used: Optional[TierClassification] = None,
    ) -> None:
        """
        Record the outcome of a classification for learning.

        Args:
            prompt_hash: Hash from RoutingDecision
            success: Whether the response was satisfactory
            tier_used: Which tier was actually used
        """
        if not self.enable_learning:
            return

        if prompt_hash not in self._prompt_outcomes:
            self._prompt_outcomes[prompt_hash] = (0, 0)

        successes, total = self._prompt_outcomes[prompt_hash]
        self._prompt_outcomes[prompt_hash] = (
            successes + (1 if success else 0),
            total + 1,
        )

        # Periodic save
        if sum(v[1] for v in self._prompt_outcomes.values()) % 100 == 0:
            self._save_history()

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics"""
        total = self._tier_0_count + self._tier_1_count
        return {
            "total_classifications": self._total_classifications,
            "tier_0_count": self._tier_0_count,
            "tier_1_count": self._tier_1_count,
            "tier_0_ratio": self._tier_0_count / max(total, 1),
            "tier_1_ratio": self._tier_1_count / max(total, 1),
            "tracked_prompts": len(self._prompt_outcomes),
            "thresholds": self.thresholds,
        }

    def adjust_threshold(self, key: str, value: float) -> None:
        """Dynamically adjust a routing threshold"""
        if key in self.thresholds:
            old_value = self.thresholds[key]
            self.thresholds[key] = value
            logger.info(f"Adjusted threshold {key}: {old_value} → {value}")
