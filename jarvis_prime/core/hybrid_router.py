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

# Safety context integration (v10.3)
# Reads shared safety state from JARVIS ActionSafetyManager
SAFETY_CONTEXT_AVAILABLE = True  # Always available via file reading

logger = logging.getLogger(__name__)

# Cross-repo safety state path (v10.3)
SAFETY_CONTEXT_FILE = Path.home() / ".jarvis" / "cross_repo" / "safety" / "context_for_prime.json"


@dataclass
class SafetyContextSnapshot:
    """
    Snapshot of JARVIS safety context read from shared state file.

    This is written by JARVIS ActionSafetyManager and read by Prime
    for safety-aware routing decisions.
    """
    kill_switch_active: bool = False
    current_risk_level: str = "low"
    pending_confirmation: bool = False
    recent_blocks: int = 0
    recent_confirmations: int = 0
    recent_denials: int = 0
    user_trust_level: float = 1.0
    last_update: str = ""
    session_start: str = ""
    total_audits: int = 0
    total_blocks: int = 0
    is_stale: bool = False  # True if file hasn't been updated recently

    def get_risk_multiplier(self) -> float:
        """
        Get a multiplier for complexity scoring based on safety state.

        Returns:
            Float multiplier (lower = more cautious user)
        """
        # Base multiplier from trust level
        multiplier = self.user_trust_level

        # Reduce multiplier if user has been denying actions
        if self.recent_denials > 0:
            multiplier *= max(0.7, 1.0 - (self.recent_denials * 0.1))

        # Reduce if there have been blocks
        if self.recent_blocks > 0:
            multiplier *= max(0.8, 1.0 - (self.recent_blocks * 0.05))

        return max(0.5, multiplier)

    def should_avoid_risky_actions(self) -> bool:
        """Check if we should route away from risky local actions."""
        return (
            self.kill_switch_active
            or self.current_risk_level in ("high", "critical")
            or self.recent_denials >= 2
            or self.user_trust_level < 0.7
        )

    def to_prompt_context(self) -> str:
        """Generate context string for prompt injection."""
        lines = ["[JARVIS SAFETY CONTEXT]"]

        if self.kill_switch_active:
            lines.append("- KILL SWITCH ACTIVE: All actions paused")

        if self.current_risk_level in ("high", "critical"):
            lines.append(f"- Risk Level: {self.current_risk_level.upper()}")

        if self.pending_confirmation:
            lines.append("- Awaiting user confirmation for risky action")

        if self.recent_blocks > 0:
            lines.append(f"- Recently blocked {self.recent_blocks} risky action(s)")

        if self.recent_denials > 0:
            lines.append(f"- User denied {self.recent_denials} action(s) recently")

        if self.user_trust_level < 0.7:
            lines.append("- User exercising caution with risky actions")

        if len(lines) == 1:
            lines.append("- All clear, normal operation")

        lines.append("[/JARVIS SAFETY CONTEXT]")
        return "\n".join(lines)


class SafetyContextReader:
    """
    Reader for JARVIS safety context shared state.

    Reads from the shared file written by ActionSafetyManager
    to enable safety-aware routing in Prime.
    """

    def __init__(self, stale_threshold_seconds: float = 60.0):
        self.stale_threshold_seconds = stale_threshold_seconds
        self._cache: Optional[SafetyContextSnapshot] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 1.0  # Re-read file every second

    def read_context(self) -> SafetyContextSnapshot:
        """Read current safety context from shared file."""
        now = time.time()

        # Return cached if fresh
        if self._cache and (now - self._cache_time) < self._cache_ttl:
            return self._cache

        # Read from file
        try:
            if not SAFETY_CONTEXT_FILE.exists():
                return SafetyContextSnapshot()

            data = json.loads(SAFETY_CONTEXT_FILE.read_text())

            # Check staleness
            last_update = data.get("last_update", "")
            is_stale = False
            if last_update:
                try:
                    update_time = datetime.fromisoformat(last_update)
                    age = (datetime.now() - update_time).total_seconds()
                    is_stale = age > self.stale_threshold_seconds
                except Exception:
                    is_stale = True

            snapshot = SafetyContextSnapshot(
                kill_switch_active=data.get("kill_switch_active", False),
                current_risk_level=data.get("current_risk_level", "low"),
                pending_confirmation=data.get("pending_confirmation", False),
                recent_blocks=data.get("recent_blocks", 0),
                recent_confirmations=data.get("recent_confirmations", 0),
                recent_denials=data.get("recent_denials", 0),
                user_trust_level=data.get("user_trust_level", 1.0),
                last_update=last_update,
                session_start=data.get("session_start", ""),
                total_audits=data.get("total_audits", 0),
                total_blocks=data.get("total_blocks", 0),
                is_stale=is_stale,
            )

            self._cache = snapshot
            self._cache_time = now
            return snapshot

        except Exception as e:
            logger.warning(f"Failed to read safety context: {e}")
            return SafetyContextSnapshot()

    def get_prompt_context(self) -> str:
        """Get safety context formatted for prompt injection."""
        ctx = self.read_context()
        return ctx.to_prompt_context()

    def is_kill_switch_active(self) -> bool:
        """Quick check if kill switch is active."""
        ctx = self.read_context()
        return ctx.kill_switch_active


def get_safety_context_reader() -> SafetyContextReader:
    """Factory function to get a SafetyContextReader."""
    return SafetyContextReader()


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


class AGIEnhancedAnalyzer(ComplexityAnalyzer):
    """
    AGI-Enhanced complexity analyzer using cognitive models.

    v77.0: Uses MetaReasoner and GoalInference for intelligent routing.
    Falls back to heuristics if AGI hub is not available.
    """

    def __init__(self) -> None:
        self._agi_hub = None
        self._fallback = DefaultComplexityAnalyzer()
        self._initialized = False

    async def _ensure_agi_hub(self) -> bool:
        """Lazily initialize AGI hub connection."""
        if self._initialized:
            return self._agi_hub is not None

        try:
            from jarvis_prime.core.agi_integration import get_agi_hub
            self._agi_hub = await get_agi_hub()
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"AGI hub not available for routing: {e}")
            self._initialized = True
            return False

    async def analyze(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> ComplexitySignals:
        """Analyze prompt using AGI models or fallback to heuristics."""
        # Get baseline signals from heuristics
        signals = await self._fallback.analyze(prompt, context)

        # Try to enhance with AGI analysis
        if await self._ensure_agi_hub() and self._agi_hub:
            try:
                # Use AGI request analyzer for deeper understanding
                from jarvis_prime.core.agi_integration import AGIRequest, RequestAnalyzer, AGIHubConfig

                analyzer = RequestAnalyzer(AGIHubConfig())
                request = AGIRequest(content=prompt, context=context or {})
                analyzed = await analyzer.analyze(request)

                # Enhance signals based on AGI analysis
                if analyzed.complexity:
                    complexity_map = {
                        "TRIVIAL": 0.1,
                        "SIMPLE": 0.3,
                        "MODERATE": 0.5,
                        "COMPLEX": 0.75,
                        "EXPERT": 0.95,
                    }
                    agi_complexity = complexity_map.get(analyzed.complexity.name, 0.5)

                    # Blend heuristic and AGI signals
                    signals.reasoning_indicators = max(
                        signals.reasoning_indicators,
                        agi_complexity * 0.8
                    )

                if analyzed.reasoning_requirement:
                    reasoning_map = {
                        "NONE": 0.0,
                        "CHAIN": 0.3,
                        "TREE": 0.7,
                        "CAUSAL": 0.6,
                        "PLANNING": 0.65,
                        "META": 0.8,
                    }
                    agi_reasoning = reasoning_map.get(analyzed.reasoning_requirement.name, 0.3)
                    signals.multi_step_indicators = max(
                        signals.multi_step_indicators,
                        agi_reasoning
                    )

                # Check if AGI models are required
                if analyzed.required_models:
                    # More required models = higher complexity
                    model_complexity = min(len(analyzed.required_models) * 0.2, 1.0)
                    signals.domain_specificity = max(
                        signals.domain_specificity,
                        model_complexity
                    )

            except Exception as e:
                logger.debug(f"AGI analysis failed, using heuristics: {e}")

        return signals


class HybridRouter:
    """
    Intelligent router for Tier 0/1 classification.

    v77.0: Enhanced with AGI Integration Hub for cognitive-aware routing.
    Uses MetaReasoner for strategy selection and GoalInference for intent.

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
        enable_safety_context: bool = True,
    ):
        self.analyzer = analyzer or DefaultComplexityAnalyzer()
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.history_file = history_file
        self.enable_learning = enable_learning
        self.enable_safety_context = enable_safety_context and SAFETY_CONTEXT_AVAILABLE

        # Learning state
        self._prompt_outcomes: Dict[str, Tuple[int, int]] = {}  # hash → (success, total)
        self._load_history()

        # Safety context reader (v10.3)
        self._safety_reader: Optional[SafetyContextReader] = None
        if self.enable_safety_context:
            try:
                self._safety_reader = get_safety_context_reader()
                logger.info("Safety context reader initialized for HybridRouter")
            except Exception as e:
                logger.warning(f"Failed to initialize safety context reader: {e}")
                self._safety_reader = None

        # Statistics
        self._total_classifications = 0
        self._tier_0_count = 0
        self._tier_1_count = 0
        self._safety_influenced_count = 0  # Count of decisions influenced by safety

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

        # Get safety context and apply risk multiplier (v10.3)
        safety_context = None
        safety_influenced = False
        if self._safety_reader:
            try:
                safety_context = self._safety_reader.read_context()

                # Apply risk multiplier to complexity score
                risk_multiplier = safety_context.get_risk_multiplier()
                if risk_multiplier < 1.0:
                    # Lower risk multiplier means user is being cautious
                    # This increases the effective complexity for risky-looking prompts
                    if self._looks_risky(prompt):
                        complexity_score = min(1.0, complexity_score / risk_multiplier)
                        safety_influenced = True
                        logger.debug(f"Safety context adjusted complexity: {complexity_score:.2f}")

            except Exception as e:
                logger.warning(f"Failed to read safety context: {e}")

        # Determine tier
        if force_tier:
            tier = force_tier
            confidence = 1.0
            reasoning = f"Forced to {force_tier.value}"
        else:
            tier, confidence, reasoning = self._determine_tier(
                complexity_score, signals, task_type, context, safety_context
            )

        # Update statistics
        if tier in (TierClassification.TIER_0, TierClassification.TIER_0_PREFERRED):
            self._tier_0_count += 1
        else:
            self._tier_1_count += 1

        if safety_influenced:
            self._safety_influenced_count += 1

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

    def _looks_risky(self, prompt: str) -> bool:
        """Check if prompt looks like it might involve risky actions."""
        risky_patterns = [
            r"\bdelete\b", r"\bremove\b", r"\berase\b", r"\bwipe\b",
            r"\bformat\b", r"\bkill\b", r"\bterminate\b", r"\bshutdown\b",
            r"\bexecute\b", r"\brun\b", r"\binstall\b", r"\buninstall\b",
            r"\bsudo\b", r"\badmin\b", r"\broot\b", r"\bsystem\b",
            r"\bpassword\b", r"\bcredential\b", r"\bsecret\b",
            r"\bfile\b.*\bwrite\b", r"\bwrite\b.*\bfile\b",
        ]
        prompt_lower = prompt.lower()
        return any(re.search(p, prompt_lower) for p in risky_patterns)

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
        safety_context: Optional["SafetyContextSnapshot"] = None,
    ) -> Tuple[TierClassification, float, str]:
        """Determine the tier classification"""
        reasons = []

        # Check task type preferences
        task_tier = self.TASK_TYPE_TIERS.get(task_type, TierClassification.TIER_0_PREFERRED)

        # Safety context checks (v10.3)
        if safety_context:
            # If kill switch is active, route everything to Tier 1 (cloud)
            # so that risky local actions can be reviewed more carefully
            if safety_context.kill_switch_active:
                reasons.append("kill switch active")
                return TierClassification.TIER_1, 0.95, f"Safety: {', '.join(reasons)}"

            # If user has been denying actions, route to Tier 1 for better reasoning
            if safety_context.should_avoid_risky_actions():
                reasons.append("safety caution active")
                # Boost towards Tier 1 for better safety reasoning
                complexity_score = min(1.0, complexity_score * 1.3)

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
        stats = {
            "total_classifications": self._total_classifications,
            "tier_0_count": self._tier_0_count,
            "tier_1_count": self._tier_1_count,
            "tier_0_ratio": self._tier_0_count / max(total, 1),
            "tier_1_ratio": self._tier_1_count / max(total, 1),
            "tracked_prompts": len(self._prompt_outcomes),
            "thresholds": self.thresholds,
            "safety_influenced_count": self._safety_influenced_count,
            "safety_context_enabled": self.enable_safety_context,
        }

        # Add current safety context if available
        if self._safety_reader:
            try:
                ctx = self._safety_reader.read_context()
                stats["current_safety_context"] = {
                    "kill_switch_active": ctx.kill_switch_active,
                    "risk_level": ctx.current_risk_level,
                    "user_trust_level": ctx.user_trust_level,
                    "is_stale": ctx.is_stale,
                }
            except Exception:
                pass

        return stats

    def get_safety_context_for_prompt(self) -> str:
        """
        Get JARVIS safety context for prompt injection.

        This can be prepended to prompts to inform the model about
        current safety state.

        Returns:
            Formatted safety context string or empty string if unavailable
        """
        if not self._safety_reader:
            return ""

        try:
            return self._safety_reader.get_prompt_context()
        except Exception as e:
            logger.warning(f"Failed to get safety context for prompt: {e}")
            return ""

    def is_kill_switch_active(self) -> bool:
        """Check if JARVIS kill switch is currently active."""
        if not self._safety_reader:
            return False

        try:
            return self._safety_reader.is_kill_switch_active()
        except Exception:
            return False

    def adjust_threshold(self, key: str, value: float) -> None:
        """Dynamically adjust a routing threshold"""
        if key in self.thresholds:
            old_value = self.thresholds[key]
            self.thresholds[key] = value
            logger.info(f"Adjusted threshold {key}: {old_value} → {value}")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_agi_enhanced_router(
    thresholds: Optional[Dict[str, float]] = None,
    history_file: Optional[Path] = None,
    enable_learning: bool = True,
    enable_safety_context: bool = True,
) -> HybridRouter:
    """
    Create a HybridRouter with AGI-enhanced analysis.

    v77.0: Uses AGI Integration Hub for cognitive-aware routing.

    Args:
        thresholds: Custom routing thresholds
        history_file: Path for learning history
        enable_learning: Enable outcome learning
        enable_safety_context: Enable safety context integration

    Returns:
        HybridRouter with AGI-enhanced analyzer
    """
    return HybridRouter(
        analyzer=AGIEnhancedAnalyzer(),
        thresholds=thresholds,
        history_file=history_file,
        enable_learning=enable_learning,
        enable_safety_context=enable_safety_context,
    )


def create_standard_router(
    thresholds: Optional[Dict[str, float]] = None,
    history_file: Optional[Path] = None,
    enable_learning: bool = True,
    enable_safety_context: bool = True,
) -> HybridRouter:
    """
    Create a HybridRouter with standard heuristic analysis.

    Faster but less intelligent routing.

    Args:
        thresholds: Custom routing thresholds
        history_file: Path for learning history
        enable_learning: Enable outcome learning
        enable_safety_context: Enable safety context integration

    Returns:
        HybridRouter with default analyzer
    """
    return HybridRouter(
        analyzer=DefaultComplexityAnalyzer(),
        thresholds=thresholds,
        history_file=history_file,
        enable_learning=enable_learning,
        enable_safety_context=enable_safety_context,
    )
