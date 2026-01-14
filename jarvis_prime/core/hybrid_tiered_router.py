"""
Hybrid Tiered Router - v95.0 Elastic Hybrid Cloud Intelligence
================================================================

The Brain's Decision-Making System for Model Selection

v95.0 ENHANCEMENTS:
    - Elastic cloud burst to GCP when local resources are insufficient
    - Integrated with JARVIS CostTracker for unified budget management
    - Memory pressure detection via JARVIS AdvancedRAMMonitor
    - Intelligent GCP optimizer for VM provisioning decisions
    - Cross-repo integration (no code duplication)

This module implements the v95.0 Hybrid Tiered Architecture from unified_config.yaml:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    JARVIS-Prime Hybrid Model Tiers                       │
    │                                                                          │
    │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
    │  │  TIER 0: LOCAL   │  │  TIER 1: CLOUD   │  │  TIER 2: DEEP    │       │
    │  │  (M1 Mac Fast)   │  │  (GCP A100)      │  │  (Opus-Level)    │       │
    │  │                  │  │                  │  │                  │       │
    │  │  • Llama 3 8B    │  │  • Llama 3.3 70B │  │  • Claude Opus   │       │
    │  │  • Qwen 2.5 7B   │  │  • Qwen 2.5 72B  │  │  • DeepSeek V3   │       │
    │  │  Speed: ~20t/s   │  │  Speed: ~45t/s   │  │  Speed: ~80t/s   │       │
    │  │  Cost: FREE      │  │  Cost: ~$1.20/hr │  │  Cost: ~$0.015/1k│       │
    │  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
    └─────────────────────────────────────────────────────────────────────────┘

ROUTING LOGIC:
    1. Analyze query complexity (lexical, semantic, domain)
    2. Check resource availability (RAM, GPU, network, budget)
    3. Route to optimal tier based on complexity + resources
    4. Fallback to next tier if primary unavailable
    5. Log experience for adaptive learning

Features:
    - Dynamic configuration loading from unified_config.yaml
    - Complexity-based intelligent routing
    - Resource-aware tier selection
    - Cost tracking and budget enforcement
    - Fallback chains with circuit breakers
    - GCP Spot VM auto-provisioning
    - Adaptive threshold learning
    - No hardcoding - fully configurable

Usage:
    router = await get_hybrid_tiered_router()

    # Route a query
    result = await router.route(
        "Explain quantum computing in detail",
        context={"max_latency_ms": 5000}
    )

    # Check which tier was used
    print(f"Routed to: {result.tier_name}")  # "tier_1_cloud_intelligent"
    print(f"Model: {result.model_name}")      # "Llama-3.3-70B-Instruct"
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# v95.0: GCP IMPORT BRIDGE - Access JARVIS infrastructure without duplication
# =============================================================================

# Lazy import to avoid circular dependencies
_gcp_bridge_available: Optional[bool] = None
_neural_orchestrator_available: Optional[bool] = None
_neural_orchestrator_instance: Optional[Any] = None


def _get_gcp_bridge():
    """Lazily import GCP bridge module."""
    global _gcp_bridge_available
    if _gcp_bridge_available is False:
        return None
    try:
        from jarvis_prime.core import gcp_import_bridge
        _gcp_bridge_available = True
        return gcp_import_bridge
    except ImportError as e:
        logger.debug(f"GCP import bridge not available: {e}")
        _gcp_bridge_available = False
        return None


async def _get_neural_orchestrator():
    """
    Lazily get the Neural Orchestrator instance.

    v100.0: HybridTieredRouter can delegate to NeuralOrchestratorCore
    for unified routing decisions.
    """
    global _neural_orchestrator_available, _neural_orchestrator_instance

    if _neural_orchestrator_available is False:
        return None

    if _neural_orchestrator_instance is not None:
        return _neural_orchestrator_instance

    try:
        from jarvis_prime.core.neural_orchestrator_core import (
            NeuralOrchestratorCore,
            get_neural_orchestrator,
        )
        _neural_orchestrator_instance = await get_neural_orchestrator()
        _neural_orchestrator_available = True
        logger.info("HybridTieredRouter: Connected to NeuralOrchestratorCore v100.0")
        return _neural_orchestrator_instance
    except ImportError as e:
        logger.debug(f"Neural Orchestrator not available: {e}")
        _neural_orchestrator_available = False
        return None
    except Exception as e:
        logger.debug(f"Neural Orchestrator initialization failed: {e}")
        _neural_orchestrator_available = False
        return None


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

# Default config paths (tried in order)
CONFIG_PATHS = [
    Path.home() / ".jarvis" / "prime" / "unified_config.yaml",
    Path(__file__).parent.parent.parent / "config" / "unified_config.yaml",
    Path.home() / "Documents" / "repos" / "jarvis-prime" / "config" / "unified_config.yaml",
]


def _load_config() -> Dict[str, Any]:
    """Load unified configuration with environment variable expansion."""
    for config_path in CONFIG_PATHS:
        if config_path.exists():
            try:
                content = config_path.read_text()

                # Expand environment variables
                def expand_env(match):
                    var_name = match.group(1)
                    default = match.group(3) if match.group(3) else ""
                    return os.getenv(var_name, default)

                # Pattern: ${VAR:-default} or ${VAR}
                content = re.sub(
                    r'\$\{([A-Za-z_][A-Za-z0-9_]*)(?:(:-)?([^}]*))?\}',
                    expand_env,
                    content
                )

                config = yaml.safe_load(content)
                logger.info(f"Loaded config from {config_path}")
                return config

            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

    logger.warning("No config file found, using defaults")
    return {}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class ModelTier(Enum):
    """Model tier levels."""
    TIER_0_LOCAL_FAST = "tier_0_local_fast"
    TIER_05_LOCAL_CAPABLE = "tier_05_local_capable"
    TIER_1_CLOUD_INTELLIGENT = "tier_1_cloud_intelligent"
    TIER_2_DEEP_REASONING = "tier_2_deep_reasoning"


@dataclass
class TierConfig:
    """Configuration for a model tier."""
    tier_id: str
    name: str
    description: str
    tier_level: int
    priority: int
    enabled: bool
    endpoint: str
    model_type: str  # "llama_cpp", "vllm", "anthropic", etc.

    # Models
    primary_model: Dict[str, Any] = field(default_factory=dict)
    fallback_models: List[Dict[str, Any]] = field(default_factory=list)

    # Capabilities
    capabilities: List[str] = field(default_factory=list)

    # Performance characteristics
    tokens_per_second: int = 20
    max_context_tokens: int = 8192
    max_output_tokens: int = 2048
    average_latency_ms: float = 500.0
    warm_start_latency_ms: float = 100.0
    cold_start_latency_ms: float = 10000.0

    # Resource requirements
    memory_required_mb: int = 5120
    gpu_layers: int = -1
    cpu_threads: int = 4
    requires_gpu: bool = False

    # Cost
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    cost_per_hour: float = 0.0

    # Routing thresholds
    max_complexity: float = 1.0
    min_complexity: float = 0.0
    max_input_tokens: int = 100000
    prefer_for_latency_ms: float = 60000.0

    # GCP config (for cloud tiers)
    gcp_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_config(cls, tier_id: str, config: Dict[str, Any]) -> "TierConfig":
        """Create TierConfig from YAML config dict."""
        models = config.get("models", {})
        perf = config.get("performance", {})
        resources = config.get("resources", {})
        cost = config.get("cost", {})
        routing = config.get("routing", {})

        return cls(
            tier_id=tier_id,
            name=config.get("name", tier_id),
            description=config.get("description", ""),
            tier_level=config.get("tier_level", 0),
            priority=config.get("priority", 1),
            enabled=str(config.get("enabled", "true")).lower() == "true",
            endpoint=config.get("endpoint", "http://localhost:8000"),
            model_type=config.get("model_type", "llama_cpp"),
            primary_model=models.get("primary", {}),
            fallback_models=models.get("fallbacks", []),
            capabilities=config.get("capabilities", []),
            tokens_per_second=perf.get("tokens_per_second", 20),
            max_context_tokens=perf.get("max_context_tokens", 8192),
            max_output_tokens=perf.get("max_output_tokens", 2048),
            average_latency_ms=perf.get("average_latency_ms", 500.0),
            warm_start_latency_ms=perf.get("warm_start_latency_ms", 100.0),
            cold_start_latency_ms=perf.get("cold_start_latency_ms", 10000.0),
            memory_required_mb=resources.get("memory_required_mb", 5120),
            gpu_layers=resources.get("gpu_layers", -1),
            cpu_threads=resources.get("cpu_threads", 4),
            requires_gpu=resources.get("requires_gpu", False),
            cost_per_1k_input=cost.get("per_1k_input_tokens", 0.0),
            cost_per_1k_output=cost.get("per_1k_output_tokens", 0.0),
            cost_per_hour=cost.get("per_hour", 0.0),
            max_complexity=routing.get("max_complexity", 1.0),
            min_complexity=routing.get("min_complexity", 0.0),
            max_input_tokens=routing.get("max_input_tokens", 100000),
            prefer_for_latency_ms=routing.get("prefer_for_latency_ms", 60000.0),
            gcp_config=config.get("gcp_config"),
        )


@dataclass
class ComplexityAnalysis:
    """Result of complexity analysis on a prompt."""
    score: float  # 0.0-1.0
    confidence: float  # 0.0-1.0

    # Signal breakdown
    token_count: int = 0
    reasoning_depth: float = 0.0
    domain_specificity: float = 0.0
    multi_step_indicators: float = 0.0
    code_complexity: float = 0.0
    context_requirements: float = 0.0

    # Task classification
    task_type: str = "chat"
    detected_capabilities: List[str] = field(default_factory=list)

    # Metadata
    analysis_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "token_count": self.token_count,
            "reasoning_depth": round(self.reasoning_depth, 3),
            "domain_specificity": round(self.domain_specificity, 3),
            "multi_step_indicators": round(self.multi_step_indicators, 3),
            "code_complexity": round(self.code_complexity, 3),
            "task_type": self.task_type,
            "detected_capabilities": self.detected_capabilities,
        }


@dataclass
class RoutingResult:
    """Result of tier routing decision."""
    # Selected tier
    tier_id: str
    tier_name: str
    tier_level: int
    model_name: str
    endpoint: str

    # Decision metadata
    complexity_score: float
    confidence: float
    reasoning: str

    # Performance estimates
    estimated_latency_ms: float
    estimated_cost_usd: float
    estimated_tokens_per_second: float

    # Fallback info
    was_fallback: bool = False
    fallback_reason: Optional[str] = None
    fallback_chain: List[str] = field(default_factory=list)

    # Resource state
    available_tiers: List[str] = field(default_factory=list)
    unavailable_tiers: Dict[str, str] = field(default_factory=dict)  # tier -> reason

    # Request tracking
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tier_id": self.tier_id,
            "tier_name": self.tier_name,
            "model_name": self.model_name,
            "complexity_score": round(self.complexity_score, 3),
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "estimated_latency_ms": round(self.estimated_latency_ms, 1),
            "estimated_cost_usd": round(self.estimated_cost_usd, 5),
            "was_fallback": self.was_fallback,
            "fallback_reason": self.fallback_reason,
        }


# =============================================================================
# COMPLEXITY ANALYZER
# =============================================================================


class AdvancedComplexityAnalyzer:
    """
    Advanced complexity analyzer for intelligent tier routing.

    Analyzes prompts using multiple signals:
    - Lexical complexity (word count, vocabulary)
    - Semantic complexity (reasoning indicators, multi-step)
    - Domain specificity (specialized knowledge required)
    - Code complexity (nesting, syntax patterns)
    - Context requirements (external knowledge, tools)
    """

    # Reasoning patterns with weights (more permissive matching)
    REASONING_PATTERNS = {
        r"analyze": 0.2,
        r"analyse": 0.2,
        r"compare": 0.15,
        r"contrast": 0.15,
        r"explain": 0.15,
        r"evaluate": 0.2,
        r"critique": 0.25,
        r"synthesiz": 0.25,
        r"infer": 0.2,
        r"deduc": 0.2,
        r"hypothes": 0.3,
        r"reason": 0.2,
        r"step.by.step": 0.35,
        r"consider": 0.15,
        r"trade.?off": 0.25,
        r"implication": 0.2,
        r"prove": 0.25,
        r"demonstrat": 0.2,
        r"architecture": 0.2,
        r"design": 0.15,
        r"implement": 0.15,
        r"optimize": 0.2,
        r"scale": 0.15,
        r"distributed": 0.2,
        r"microservice": 0.2,
        r"pattern": 0.1,
        r"complex": 0.15,
        r"strategy": 0.15,
    }

    # Multi-step patterns (more permissive)
    MULTI_STEP_PATTERNS = {
        r"first.*then": 0.35,
        r"step\s*\d": 0.3,
        r"next.*finally": 0.3,
        r"after\s+that": 0.2,
        r"once.*is": 0.2,
        r"sequential": 0.25,
        r"in\s+order\s+to": 0.15,
        r"follow.*steps": 0.25,
        r"break.*down": 0.25,
        r"phase\s*\d": 0.25,
        r"\d+\.\s": 0.2,  # Numbered lists like "1. "
        r"first,": 0.25,
        r"second,": 0.2,
        r"third,": 0.2,
        r"finally,": 0.25,
        r"then,": 0.15,
        r"next,": 0.15,
    }

    # Domain-specific keywords
    DOMAIN_KEYWORDS = {
        "medical": {
            "keywords": ["diagnosis", "symptom", "treatment", "medication", "patient",
                        "clinical", "pathology", "prognosis", "etiology", "syndrome"],
            "weight": 0.8,
        },
        "legal": {
            "keywords": ["contract", "statute", "liability", "jurisdiction", "clause",
                        "litigation", "precedent", "tort", "damages", "plaintiff"],
            "weight": 0.8,
        },
        "financial": {
            "keywords": ["derivative", "portfolio", "hedge", "arbitrage", "yield",
                        "liquidity", "volatility", "securitization", "amortization"],
            "weight": 0.75,
        },
        "scientific": {
            "keywords": ["hypothesis", "methodology", "empirical", "correlation",
                        "variable", "experiment", "quantitative", "qualitative"],
            "weight": 0.7,
        },
        "technical": {
            "keywords": ["algorithm", "architecture", "infrastructure", "latency",
                        "scalability", "throughput", "microservice", "kubernetes"],
            "weight": 0.6,
        },
        "mathematical": {
            "keywords": ["theorem", "proof", "derivative", "integral", "eigenvalue",
                        "matrix", "optimization", "convergence", "polynomial"],
            "weight": 0.85,
        },
    }

    # Code complexity patterns
    CODE_PATTERNS = {
        r"```\w*\n": 0.1,
        r"def\s+\w+\s*\(": 0.15,
        r"class\s+\w+\s*[:\(]": 0.2,
        r"async\s+def\b": 0.25,
        r"import\s+\w+": 0.1,
        r"from\s+\w+\s+import": 0.1,
        r"try\s*:": 0.15,
        r"except\s+\w+": 0.15,
        r"lambda\s+\w+": 0.2,
        r"yield\s+": 0.25,
        r"@\w+": 0.15,  # decorators
        r"\basync\s+for\b": 0.3,
        r"\bawait\b": 0.2,
    }

    # Capability keywords mapping
    CAPABILITY_KEYWORDS = {
        "multimodal": ["image", "picture", "photo", "screenshot", "diagram", "chart"],
        "search": ["search", "look up", "find information", "current", "latest", "recent"],
        "code_complex": ["refactor", "architect", "design system", "implement", "debug"],
        "code_simple": ["write code", "function", "script", "snippet"],
        "summarize": ["summarize", "summary", "tldr", "brief", "condense"],
        "creative": ["write story", "poem", "creative", "imagine", "fiction"],
        "analyze": ["analyze", "breakdown", "examine", "investigate"],
        "reason": ["reason", "explain why", "think through", "deduce"],
        "math": ["calculate", "compute", "solve", "equation", "formula"],
        "translate": ["translate", "convert to", "in spanish", "in french"],
    }

    def __init__(self, task_type_routing: Optional[Dict[str, str]] = None):
        self._task_type_routing = task_type_routing or {}
        self._history: List[ComplexityAnalysis] = []

    async def analyze(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComplexityAnalysis:
        """Analyze prompt complexity with multiple signals."""
        start_time = time.time()
        prompt_lower = prompt.lower()
        context = context or {}

        # Token analysis
        tokens = prompt.split()
        token_count = len(tokens)
        avg_token_length = sum(len(t) for t in tokens) / max(len(tokens), 1)

        # Quick check for very short prompts (greetings, simple questions)
        if token_count <= 5:
            # Very short prompts are always low complexity
            return ComplexityAnalysis(
                score=0.1 * (token_count / 5),
                confidence=0.9,
                token_count=token_count,
                task_type=self._detect_task_type(prompt_lower, []),
                analysis_time_ms=(time.time() - start_time) * 1000,
            )

        # Reasoning depth
        reasoning_score = 0.0
        for pattern, weight in self.REASONING_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                reasoning_score += weight
        reasoning_depth = min(reasoning_score, 1.0)

        # Multi-step indicators
        multi_step_score = 0.0
        for pattern, weight in self.MULTI_STEP_PATTERNS.items():
            if re.search(pattern, prompt_lower):
                multi_step_score += weight
        multi_step_indicators = min(multi_step_score, 1.0)

        # Domain specificity
        domain_score = 0.0
        for domain, info in self.DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in info["keywords"] if kw in prompt_lower)
            if matches >= 2:
                domain_score = max(domain_score, (matches / len(info["keywords"])) * info["weight"])
        domain_specificity = min(domain_score, 1.0)

        # Code complexity
        code_score = 0.0
        for pattern, weight in self.CODE_PATTERNS.items():
            matches = len(re.findall(pattern, prompt))
            code_score += min(matches * weight, 0.3)  # Cap per pattern
        code_complexity = min(code_score, 1.0)

        # Nesting depth (code/JSON complexity)
        nesting_depth = self._calculate_nesting(prompt)
        if nesting_depth > 3:
            code_complexity = min(code_complexity + 0.2, 1.0)

        # Context requirements
        context_requirements = 0.0
        if context.get("has_images"):
            context_requirements += 0.5
        if context.get("needs_current_info"):
            context_requirements += 0.3
        if context.get("needs_tools"):
            context_requirements += 0.2
        context_requirements = min(context_requirements, 1.0)

        # Detect capabilities needed
        detected_capabilities = self._detect_capabilities(prompt_lower)

        # Detect task type
        task_type = self._detect_task_type(prompt_lower, detected_capabilities)

        # Calculate overall complexity score
        # Weighted combination of all signals
        weights = {
            "tokens": 0.15,      # Normalized by 500 tokens (more sensitive)
            "reasoning": 0.25,
            "multi_step": 0.15,
            "domain": 0.20,
            "code": 0.15,
            "context": 0.10,
        }

        # Token complexity: longer prompts tend to be more complex
        # Normalize by 500 for more sensitivity
        token_complexity = min(token_count / 500, 1.0)

        # Boost for very long prompts (>100 words suggests complex task)
        if token_count > 100:
            token_complexity = min(token_complexity + 0.2, 1.0)

        complexity_score = (
            weights["tokens"] * token_complexity +
            weights["reasoning"] * reasoning_depth +
            weights["multi_step"] * multi_step_indicators +
            weights["domain"] * domain_specificity +
            weights["code"] * code_complexity +
            weights["context"] * context_requirements
        )

        # Boost complexity if multiple strong signals detected
        strong_signals = sum([
            1 if reasoning_depth > 0.3 else 0,
            1 if multi_step_indicators > 0.3 else 0,
            1 if domain_specificity > 0.3 else 0,
            1 if code_complexity > 0.3 else 0,
        ])
        if strong_signals >= 2:
            complexity_score = min(complexity_score + 0.15, 1.0)
        elif strong_signals >= 3:
            complexity_score = min(complexity_score + 0.25, 1.0)

        # Adjust based on task type
        if task_type in ["reason", "code_architect", "paper_analysis"]:
            complexity_score = min(complexity_score * 1.2, 1.0)
        elif task_type in ["chat", "greeting", "format"]:
            complexity_score = complexity_score * 0.8

        # Calculate confidence based on signal clarity
        signal_strength = max(
            reasoning_depth,
            multi_step_indicators,
            domain_specificity,
            code_complexity,
        )
        confidence = 0.6 + (0.4 * signal_strength)

        analysis_time_ms = (time.time() - start_time) * 1000

        result = ComplexityAnalysis(
            score=complexity_score,
            confidence=confidence,
            token_count=token_count,
            reasoning_depth=reasoning_depth,
            domain_specificity=domain_specificity,
            multi_step_indicators=multi_step_indicators,
            code_complexity=code_complexity,
            context_requirements=context_requirements,
            task_type=task_type,
            detected_capabilities=detected_capabilities,
            analysis_time_ms=analysis_time_ms,
        )

        self._history.append(result)
        return result

    def _calculate_nesting(self, text: str) -> int:
        """Calculate maximum nesting depth."""
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

    def _detect_capabilities(self, prompt_lower: str) -> List[str]:
        """Detect required capabilities from prompt."""
        capabilities = []
        for capability, keywords in self.CAPABILITY_KEYWORDS.items():
            if any(kw in prompt_lower for kw in keywords):
                capabilities.append(capability)
        return capabilities

    def _detect_task_type(
        self,
        prompt_lower: str,
        capabilities: List[str],
    ) -> str:
        """Detect task type from prompt and capabilities."""
        # Check for specific task types
        if "multimodal" in capabilities:
            return "multimodal"
        if "search" in capabilities:
            return "search"
        if "code_complex" in capabilities:
            return "code_complex"
        if "code_simple" in capabilities:
            return "code_simple"
        if "summarize" in capabilities:
            return "summarize"
        if "reason" in capabilities:
            return "reason"
        if "math" in capabilities:
            return "math"
        if "analyze" in capabilities:
            return "analyze"
        if "creative" in capabilities:
            return "creative"
        if "translate" in capabilities:
            return "translate"

        # Check for greetings
        greetings = ["hello", "hi ", "hey ", "good morning", "good evening"]
        if any(g in prompt_lower for g in greetings):
            return "greeting"

        # Check for formatting requests
        if any(kw in prompt_lower for kw in ["format", "convert", "restructure"]):
            return "format"

        # Default to chat
        return "chat"

    def get_statistics(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        if not self._history:
            return {"total_analyses": 0}

        scores = [a.score for a in self._history]
        return {
            "total_analyses": len(self._history),
            "avg_complexity": sum(scores) / len(scores),
            "max_complexity": max(scores),
            "min_complexity": min(scores),
            "task_type_distribution": {
                tt: sum(1 for a in self._history if a.task_type == tt)
                for tt in set(a.task_type for a in self._history)
            },
        }


# =============================================================================
# RESOURCE MONITOR
# =============================================================================


class ResourceMonitor:
    """
    Monitor system resources for tier availability decisions.

    Tracks:
    - Memory usage (RAM available for model loading)
    - GPU availability
    - Network connectivity
    - Daily cost budget
    """

    def __init__(
        self,
        max_ram_percent: float = 85.0,
        max_daily_budget: float = 10.0,
    ):
        self._max_ram_percent = max_ram_percent
        self._max_daily_budget = max_daily_budget
        self._daily_spend: Dict[str, float] = {}  # date -> spend
        self._lock = asyncio.Lock()

        # Tier health tracking
        self._tier_health: Dict[str, bool] = {}
        self._last_health_check: Dict[str, float] = {}
        self._health_check_interval = 30.0  # seconds

        # Model loading state (optimistic - assume we can load until proven otherwise)
        self._loaded_models: Set[str] = set()
        self._failed_models: Set[str] = set()

    async def get_available_ram_mb(self) -> float:
        """Get available RAM in MB."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return mem.available / (1024 * 1024)
        except ImportError:
            # Fallback: assume 16GB available on modern systems
            return 16384.0

    async def get_total_ram_mb(self) -> float:
        """Get total system RAM in MB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 * 1024)
        except ImportError:
            return 16384.0

    async def get_ram_usage_percent(self) -> float:
        """Get current RAM usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 50.0  # Conservative default

    async def can_load_model(self, memory_required_mb: int, model_name: str = "") -> Tuple[bool, str]:
        """
        Check if we can load a model of given size.

        Returns:
            Tuple of (can_load: bool, reason: str)
        """
        # If we've successfully loaded this model before, we know we can
        if model_name and model_name in self._loaded_models:
            return True, "Model previously loaded successfully"

        # If we've failed to load this model before, don't try again
        if model_name and model_name in self._failed_models:
            return False, f"Model {model_name} previously failed to load"

        available = await self.get_available_ram_mb()
        total = await self.get_total_ram_mb()

        # Be more permissive: if total RAM is sufficient, allow loading
        # (macOS will use swap if needed)
        if total >= memory_required_mb:
            return True, f"Total RAM {total:.0f}MB >= required {memory_required_mb}MB"

        # Check available RAM with some headroom
        if available >= memory_required_mb * 0.8:  # Allow loading if 80% of required is available
            return True, f"Available RAM {available:.0f}MB sufficient"

        return False, f"Insufficient RAM: {available:.0f}MB available, {memory_required_mb}MB required"

    def mark_model_loaded(self, model_name: str):
        """Mark a model as successfully loaded."""
        self._loaded_models.add(model_name)
        self._failed_models.discard(model_name)

    def mark_model_failed(self, model_name: str):
        """Mark a model as failed to load."""
        self._failed_models.add(model_name)
        self._loaded_models.discard(model_name)

    async def record_spend(self, amount: float) -> None:
        """Record spending against daily budget."""
        async with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            self._daily_spend[today] = self._daily_spend.get(today, 0.0) + amount

    async def get_remaining_budget(self) -> float:
        """Get remaining daily budget."""
        today = datetime.now().strftime("%Y-%m-%d")
        spent = self._daily_spend.get(today, 0.0)
        return max(0.0, self._max_daily_budget - spent)

    async def can_spend(self, estimated_cost: float) -> bool:
        """Check if we can afford the estimated cost."""
        remaining = await self.get_remaining_budget()
        return estimated_cost <= remaining

    async def check_tier_health(
        self,
        tier_id: str,
        endpoint: str,
    ) -> bool:
        """Check if a tier's endpoint is healthy."""
        now = time.time()
        last_check = self._last_health_check.get(tier_id, 0.0)

        # Use cached result if fresh
        if now - last_check < self._health_check_interval:
            return self._tier_health.get(tier_id, False)

        # Perform health check
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as response:
                    is_healthy = response.status == 200
                    self._tier_health[tier_id] = is_healthy
                    self._last_health_check[tier_id] = now
                    return is_healthy
        except Exception:
            self._tier_health[tier_id] = False
            self._last_health_check[tier_id] = now
            return False

    # =========================================================================
    # v95.0: GCP INTEGRATION - Uses JARVIS components (no duplication)
    # =========================================================================

    async def get_gcp_budget_status(self) -> Dict[str, Any]:
        """
        Get budget status from JARVIS CostTracker.

        Uses the GCP import bridge to access JARVIS's budget tracking.
        """
        bridge = _get_gcp_bridge()
        if bridge is None:
            return {
                "available": False,
                "fallback": True,
                "daily_remaining": self._max_daily_budget - self._daily_spend.get(
                    datetime.now().strftime("%Y-%m-%d"), 0.0
                ),
            }

        return await bridge.get_budget_status()

    async def can_afford_cloud_tier(self, estimated_cost: float) -> bool:
        """
        Check if budget allows a cloud tier request.

        First checks JARVIS CostTracker, falls back to local tracking.
        """
        bridge = _get_gcp_bridge()
        if bridge is not None:
            try:
                return await bridge.can_afford_cloud_tier(estimated_cost)
            except Exception as e:
                logger.warning(f"GCP budget check failed, using local: {e}")

        # Fallback to local budget tracking
        return await self.can_spend(estimated_cost)

    async def get_memory_pressure(self) -> Dict[str, Any]:
        """
        Get memory pressure from JARVIS AdvancedRAMMonitor.

        Falls back to basic psutil if JARVIS components unavailable.
        """
        bridge = _get_gcp_bridge()
        if bridge is not None:
            try:
                return await bridge.get_ram_snapshot()
            except Exception as e:
                logger.warning(f"JARVIS RAM monitor failed: {e}")

        # Fallback to basic monitoring
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "available": True,
                "fallback": True,
                "total_gb": mem.total / (1024**3),
                "used_gb": mem.used / (1024**3),
                "available_gb": mem.available / (1024**3),
                "percent_used": mem.percent,
                "pressure_level": "critical" if mem.percent > 90 else "high" if mem.percent > 80 else "normal",
            }
        except ImportError:
            return {"available": False, "error": "psutil not available"}

    async def should_burst_to_cloud(self) -> bool:
        """
        Determine if system should burst to GCP cloud.

        Uses JARVIS IntelligentGCPOptimizer for multi-factor decision.
        """
        bridge = _get_gcp_bridge()
        if bridge is not None:
            try:
                return await bridge.should_burst_to_cloud()
            except Exception as e:
                logger.warning(f"GCP burst decision failed: {e}")

        # Fallback: simple RAM threshold
        pressure = await self.get_memory_pressure()
        return pressure.get("percent_used", 0) > 80

    async def request_cloud_burst(
        self,
        reason: str,
        required_ram_gb: float = 0.0,
        estimated_duration_minutes: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Request elastic cloud burst to GCP.

        This triggers JARVIS's GCPVMManager to provision a Spot VM.

        Args:
            reason: Why burst is needed
            required_ram_gb: Minimum RAM required
            estimated_duration_minutes: Expected usage duration

        Returns:
            Result dict with success status and VM info if provisioned
        """
        bridge = _get_gcp_bridge()
        if bridge is None:
            return {
                "success": False,
                "error": "gcp_bridge_unavailable",
                "message": "GCP import bridge not available",
            }

        return await bridge.request_cloud_burst(
            reason=reason,
            required_ram_gb=required_ram_gb,
            estimated_duration_minutes=estimated_duration_minutes,
        )

    async def get_gcp_infrastructure_status(self) -> Dict[str, Any]:
        """
        Get unified status of all JARVIS GCP infrastructure.
        """
        bridge = _get_gcp_bridge()
        if bridge is None:
            return {
                "available": False,
                "error": "GCP import bridge not available",
            }

        return await bridge.get_gcp_infrastructure_status()

    def get_statistics(self) -> Dict[str, Any]:
        """Get resource monitor statistics."""
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "daily_spend": self._daily_spend.get(today, 0.0),
            "remaining_budget": self._max_daily_budget - self._daily_spend.get(today, 0.0),
            "tier_health": self._tier_health.copy(),
            "gcp_bridge_available": _gcp_bridge_available is True,
        }


# =============================================================================
# HYBRID TIERED ROUTER
# =============================================================================


class HybridTieredRouter:
    """
    The Brain's Routing Decision System - v94.0 Hybrid Tiered Architecture

    Intelligently routes prompts to the optimal model tier based on:
    - Complexity analysis
    - Resource availability
    - Cost constraints
    - Latency requirements
    - Capability requirements

    Implements fallback chains with circuit breaker protection.

    Usage:
        router = HybridTieredRouter()
        await router.initialize()

        # Route a complex query
        result = await router.route(
            "Design a microservice architecture for handling 1M requests/day",
            context={"max_latency_ms": 10000}
        )

        print(f"Routed to: {result.tier_name}")  # "Cloud 70B Intelligent"
        print(f"Model: {result.model_name}")      # "Llama-3.3-70B-Instruct"

        # Execute on selected tier
        response = await router.execute(result, prompt)
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        enable_learning: bool = True,
        prefer_free: bool = True,
    ):
        self._config: Dict[str, Any] = {}
        self._tiers: Dict[str, TierConfig] = {}
        self._routing_rules: Dict[str, Any] = {}
        self._fallback_chains: Dict[str, List[str]] = {}

        self._analyzer = AdvancedComplexityAnalyzer()
        self._resource_monitor = ResourceMonitor()

        self._enable_learning = enable_learning
        self._prefer_free = prefer_free

        # Circuit breakers per tier
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._total_routes = 0
        self._tier_usage: Dict[str, int] = {}
        self._fallback_count = 0
        self._route_history: List[RoutingResult] = []

        # Initialize
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize router from configuration."""
        async with self._lock:
            if self._initialized:
                return

            # Load configuration
            self._config = _load_config()

            # Parse model routing configuration
            model_routing = self._config.get("model_routing", {})
            tiers_config = model_routing.get("tiers", {})

            # Create TierConfig objects
            for tier_id, tier_config in tiers_config.items():
                self._tiers[tier_id] = TierConfig.from_config(tier_id, tier_config)

                # Initialize circuit breaker
                self._circuit_breakers[tier_id] = {
                    "failures": 0,
                    "threshold": 5,
                    "last_failure": 0.0,
                    "timeout": 60.0,
                    "state": "closed",
                }

            # Load routing rules
            self._routing_rules = model_routing.get("routing_rules", {})

            # Load fallback chains
            self._fallback_chains = self._routing_rules.get("fallback_chains", {})

            # Sort tiers by priority
            self._sorted_tier_ids = sorted(
                self._tiers.keys(),
                key=lambda t: self._tiers[t].priority
            )

            self._initialized = True
            logger.info(f"HybridTieredRouter initialized with {len(self._tiers)} tiers")

    async def route(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingResult:
        """
        Route a prompt to the optimal model tier.

        v100.0: Optionally delegates to NeuralOrchestratorCore for unified
        intelligent routing. Falls back to native implementation if unavailable.

        Args:
            prompt: The prompt to route
            context: Optional context including:
                - max_latency_ms: Maximum acceptable latency
                - max_cost_usd: Maximum cost for this request
                - prefer_local: Prefer local models
                - require_capability: Required capability
                - use_neural_orchestrator: Enable/disable Neural Orchestrator (default: True)

        Returns:
            RoutingResult with selected tier and metadata
        """
        if not self._initialized:
            await self.initialize()

        context = context or {}
        self._total_routes += 1

        # v100.0: Try Neural Orchestrator first (unified intelligent routing)
        use_neural = context.get("use_neural_orchestrator", True)
        if use_neural:
            neural_orchestrator = await _get_neural_orchestrator()
            if neural_orchestrator:
                try:
                    neural_result = await neural_orchestrator.route(prompt, context)
                    if neural_result:
                        # Convert Neural Orchestrator result to HybridTieredRouter result
                        tier_mapping = {
                            "TIER_0_ULTRA_FAST": "tier_0_local_fast",
                            "TIER_0_FAST": "tier_0_local_fast",
                            "TIER_05_CAPABLE": "tier_05_local_capable",
                            "TIER_1_CLOUD": "tier_1_cloud_intelligent",
                            "TIER_2_DEEP": "tier_2_deep_reasoning",
                        }
                        tier_id = tier_mapping.get(
                            neural_result.tier.name, "tier_0_local_fast"
                        )

                        # Get tier config
                        tier_config = self._tiers.get(tier_id)
                        if tier_config:
                            self._tier_usage[tier_id] = self._tier_usage.get(tier_id, 0) + 1

                            result = RoutingResult(
                                tier_id=tier_id,
                                tier_name=tier_config.name,
                                tier_level=tier_config.tier_level,
                                model_name=neural_result.model_id or tier_config.primary_model.get("name", "unknown"),
                                endpoint=neural_result.endpoint or tier_config.endpoint,
                                complexity_score=neural_result.task_classification.complexity if neural_result.task_classification else 0.5,
                                confidence=neural_result.confidence,
                                reasoning=f"Neural Orchestrator v100.0: {neural_result.decision_reason.value}",
                                estimated_latency_ms=tier_config.average_latency_ms,
                                estimated_cost_usd=0.0,
                                estimated_tokens_per_second=tier_config.tokens_per_second,
                                was_fallback=neural_result.decision_reason.value == "fallback",
                                fallback_reason=None,
                                fallback_chain=[],
                                available_tiers=list(self._tiers.keys()),
                                unavailable_tiers={},
                            )

                            self._route_history.append(result)
                            if len(self._route_history) > 1000:
                                self._route_history = self._route_history[-500:]

                            logger.debug(
                                f"Neural Orchestrator routed to {tier_id} "
                                f"(reason={neural_result.decision_reason.value})"
                            )
                            return result
                except Exception as e:
                    logger.warning(f"Neural Orchestrator routing failed, using native: {e}")

        # Fallback to native HybridTieredRouter implementation
        # Analyze prompt complexity
        analysis = await self._analyzer.analyze(prompt, context)

        # Get available tiers
        available_tiers, unavailable = await self._get_available_tiers(context)

        if not available_tiers:
            raise RuntimeError("No tiers available for routing")

        # Find optimal tier
        selected_tier, reasoning = await self._select_optimal_tier(
            analysis,
            available_tiers,
            context,
        )

        # Check if this was a fallback
        was_fallback = False
        fallback_reason = None
        primary_tier = self._get_primary_tier_for_complexity(analysis.score)

        if primary_tier and primary_tier != selected_tier.tier_id:
            was_fallback = True
            fallback_reason = unavailable.get(primary_tier, "Complexity routing")
            self._fallback_count += 1

        # Estimate performance
        estimated_latency_ms = selected_tier.average_latency_ms
        if analysis.token_count > 500:
            # Longer prompts take longer
            estimated_latency_ms += (analysis.token_count / selected_tier.tokens_per_second) * 1000

        estimated_cost = (
            (analysis.token_count / 1000) * selected_tier.cost_per_1k_input +
            (500 / 1000) * selected_tier.cost_per_1k_output  # Estimate 500 output tokens
        )

        # Get fallback chain
        fallback_chain = self._fallback_chains.get(selected_tier.tier_id, [])

        result = RoutingResult(
            tier_id=selected_tier.tier_id,
            tier_name=selected_tier.name,
            tier_level=selected_tier.tier_level,
            model_name=selected_tier.primary_model.get("name", "unknown"),
            endpoint=selected_tier.endpoint,
            complexity_score=analysis.score,
            confidence=analysis.confidence,
            reasoning=reasoning,
            estimated_latency_ms=estimated_latency_ms,
            estimated_cost_usd=estimated_cost,
            estimated_tokens_per_second=selected_tier.tokens_per_second,
            was_fallback=was_fallback,
            fallback_reason=fallback_reason,
            fallback_chain=fallback_chain,
            available_tiers=list(available_tiers.keys()),
            unavailable_tiers=unavailable,
        )

        # Update statistics
        self._tier_usage[selected_tier.tier_id] = self._tier_usage.get(selected_tier.tier_id, 0) + 1
        self._route_history.append(result)
        if len(self._route_history) > 1000:
            self._route_history = self._route_history[-500:]

        logger.info(
            f"Routed to {selected_tier.name} "
            f"(complexity={analysis.score:.2f}, tier_level={selected_tier.tier_level})"
        )

        return result

    async def _get_available_tiers(
        self,
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, TierConfig], Dict[str, str]]:
        """Get available tiers based on resources and constraints."""
        available: Dict[str, TierConfig] = {}
        unavailable: Dict[str, str] = {}

        max_latency_ms = context.get("max_latency_ms", float("inf"))
        max_cost_usd = context.get("max_cost_usd", float("inf"))
        prefer_local = context.get("prefer_local", self._prefer_free)
        require_capability = context.get("require_capability")

        for tier_id, tier in self._tiers.items():
            # Check if enabled
            if not tier.enabled:
                unavailable[tier_id] = "Tier disabled"
                continue

            # Check circuit breaker
            cb = self._circuit_breakers[tier_id]
            if cb["state"] == "open":
                if time.time() - cb["last_failure"] > cb["timeout"]:
                    cb["state"] = "half_open"
                else:
                    unavailable[tier_id] = "Circuit breaker open"
                    continue

            # Check latency constraint
            if tier.average_latency_ms > max_latency_ms:
                unavailable[tier_id] = f"Latency {tier.average_latency_ms}ms > {max_latency_ms}ms"
                continue

            # Check cost constraint
            estimated_cost = tier.cost_per_1k_input * 2  # Rough estimate
            if estimated_cost > max_cost_usd:
                unavailable[tier_id] = f"Cost ${estimated_cost:.4f} > ${max_cost_usd:.4f}"
                continue

            # Check daily budget for paid tiers
            if tier.cost_per_hour > 0 or tier.cost_per_1k_input > 0:
                if not await self._resource_monitor.can_spend(estimated_cost):
                    unavailable[tier_id] = "Daily budget exceeded"
                    continue

            # Check memory requirements for local tiers
            if tier.model_type == "llama_cpp":
                model_name = tier.primary_model.get("name", "")
                can_load, reason = await self._resource_monitor.can_load_model(
                    tier.memory_required_mb,
                    model_name
                )
                if not can_load:
                    unavailable[tier_id] = reason

                    # v95.0: Check if we should burst to cloud
                    if await self._should_trigger_elastic_burst(tier.memory_required_mb):
                        logger.info(f"Elastic burst triggered for tier {tier_id} (reason: {reason})")
                        # Cloud tiers will be preferred in selection
                    continue

            # Check capability requirements
            if require_capability and require_capability not in tier.capabilities:
                unavailable[tier_id] = f"Missing capability: {require_capability}"
                continue

            available[tier_id] = tier

        return available, unavailable

    async def _select_optimal_tier(
        self,
        analysis: ComplexityAnalysis,
        available_tiers: Dict[str, TierConfig],
        context: Dict[str, Any],
    ) -> Tuple[TierConfig, str]:
        """Select the optimal tier based on complexity and constraints."""
        complexity = analysis.score
        task_type = analysis.task_type

        # Get task type routing hint
        task_routing = self._routing_rules.get("task_type_routing", {})
        preferred_tier_id = task_routing.get(task_type)

        # If preferred tier is available and matches complexity, use it
        if preferred_tier_id and preferred_tier_id in available_tiers:
            tier = available_tiers[preferred_tier_id]
            if tier.min_complexity <= complexity <= tier.max_complexity:
                return tier, f"Task type '{task_type}' routed to preferred tier"

        # Get complexity thresholds
        thresholds = self._routing_rules.get("complexity_thresholds", {})
        tier_0_max = thresholds.get("tier_0_fast_max", 0.35)
        tier_05_max = thresholds.get("tier_05_capable_max", 0.55)
        tier_1_max = thresholds.get("tier_1_cloud_max", 0.80)
        tier_2_min = thresholds.get("tier_2_deep_min", 0.75)

        # Determine target tier based on complexity
        target_tier_id = None
        reasoning = ""

        if complexity <= tier_0_max:
            target_tier_id = "tier_0_local_fast"
            reasoning = f"Low complexity ({complexity:.2f}) → Local Fast"
        elif complexity <= tier_05_max:
            target_tier_id = "tier_05_local_capable"
            reasoning = f"Medium complexity ({complexity:.2f}) → Local Capable"
        elif complexity <= tier_1_max:
            target_tier_id = "tier_1_cloud_intelligent"
            reasoning = f"High complexity ({complexity:.2f}) → Cloud Intelligent"
        else:
            target_tier_id = "tier_2_deep_reasoning"
            reasoning = f"Very high complexity ({complexity:.2f}) → Deep Reasoning"

        # If target tier is available, use it
        if target_tier_id in available_tiers:
            return available_tiers[target_tier_id], reasoning

        # Otherwise, find closest available tier
        # Prefer upgrading over downgrading for better quality
        sorted_available = sorted(
            available_tiers.values(),
            key=lambda t: (t.tier_level, t.priority)
        )

        # Find first tier that can handle this complexity
        for tier in sorted_available:
            if complexity <= tier.max_complexity:
                return tier, f"Fallback: {reasoning}, using {tier.name}"

        # Last resort: use highest capability tier available
        highest_tier = sorted_available[-1]
        return highest_tier, f"Fallback: Using highest available tier {highest_tier.name}"

    def _get_primary_tier_for_complexity(self, complexity: float) -> Optional[str]:
        """Get the primary tier that should handle this complexity."""
        thresholds = self._routing_rules.get("complexity_thresholds", {})

        if complexity <= thresholds.get("tier_0_fast_max", 0.35):
            return "tier_0_local_fast"
        elif complexity <= thresholds.get("tier_05_capable_max", 0.55):
            return "tier_05_local_capable"
        elif complexity <= thresholds.get("tier_1_cloud_max", 0.80):
            return "tier_1_cloud_intelligent"
        else:
            return "tier_2_deep_reasoning"

    # =========================================================================
    # v95.0: ELASTIC CLOUD BURST - GCP Integration
    # =========================================================================

    async def _should_trigger_elastic_burst(self, required_ram_mb: int = 0) -> bool:
        """
        Determine if elastic cloud burst should be triggered.

        Checks multiple factors:
        - Memory pressure from JARVIS AdvancedRAMMonitor
        - Budget availability from JARVIS CostTracker
        - Intelligent optimization from IntelligentGCPOptimizer

        Args:
            required_ram_mb: RAM needed for the requested operation

        Returns:
            True if cloud burst is recommended
        """
        # First check budget - no point bursting if we can't afford it
        can_afford = await self._resource_monitor.can_afford_cloud_tier(0.05)  # ~$0.05 estimate
        if not can_afford:
            logger.debug("Elastic burst blocked: insufficient budget")
            return False

        # Check if burst is actually needed via JARVIS intelligent optimizer
        should_burst = await self._resource_monitor.should_burst_to_cloud()
        if should_burst:
            logger.info("Elastic burst recommended by JARVIS optimizer")
            return True

        # Fallback: check if required RAM exceeds available
        if required_ram_mb > 0:
            pressure = await self._resource_monitor.get_memory_pressure()
            available_gb = pressure.get("available_gb", 16.0)
            required_gb = required_ram_mb / 1024.0

            if required_gb > available_gb * 0.8:  # Need 80%+ of available
                logger.info(f"Elastic burst needed: {required_gb:.1f}GB required, {available_gb:.1f}GB available")
                return True

        return False

    async def request_elastic_burst(
        self,
        reason: str = "routing_decision",
        complexity_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Request elastic cloud burst to GCP.

        This is called when local resources are insufficient and we need
        to burst to cloud. Uses JARVIS's GCPVMManager for provisioning.

        Args:
            reason: Why burst is needed
            complexity_score: Complexity of the task requiring burst

        Returns:
            Result dict with success status and burst details
        """
        # Estimate requirements based on complexity
        if complexity_score > 0.8:
            required_ram_gb = 32.0  # High complexity needs more RAM
            estimated_duration = 60.0
        elif complexity_score > 0.5:
            required_ram_gb = 16.0
            estimated_duration = 30.0
        else:
            required_ram_gb = 8.0
            estimated_duration = 15.0

        result = await self._resource_monitor.request_cloud_burst(
            reason=reason,
            required_ram_gb=required_ram_gb,
            estimated_duration_minutes=estimated_duration,
        )

        if result.get("success"):
            logger.info(f"Elastic burst successful: {result.get('vm_info', {})}")
            # Update tier availability after burst
            # Cloud tiers should now be accessible
        else:
            logger.warning(f"Elastic burst failed: {result.get('error', 'unknown')}")

        return result

    async def get_elastic_burst_status(self) -> Dict[str, Any]:
        """
        Get current elastic burst status.

        Returns:
            Status dict with burst availability and GCP infrastructure state
        """
        # Get GCP infrastructure status
        gcp_status = await self._resource_monitor.get_gcp_infrastructure_status()

        # Get memory pressure
        pressure = await self._resource_monitor.get_memory_pressure()

        # Get budget status
        budget = await self._resource_monitor.get_gcp_budget_status()

        # Determine if burst is possible
        burst_possible = (
            gcp_status.get("summary", {}).get("available_components", 0) >= 2 and
            budget.get("daily_remaining", 0) > 0.05
        )

        # Determine if burst is recommended
        burst_recommended = await self._resource_monitor.should_burst_to_cloud()

        return {
            "burst_possible": burst_possible,
            "burst_recommended": burst_recommended,
            "gcp_infrastructure": gcp_status,
            "memory_pressure": pressure,
            "budget_status": budget,
        }

    def record_success(self, tier_id: str) -> None:
        """Record successful request to a tier."""
        if tier_id in self._circuit_breakers:
            cb = self._circuit_breakers[tier_id]
            cb["failures"] = 0
            if cb["state"] == "half_open":
                cb["state"] = "closed"
                logger.info(f"Circuit breaker for {tier_id} closed after recovery")

    def record_failure(self, tier_id: str) -> None:
        """Record failed request to a tier."""
        if tier_id in self._circuit_breakers:
            cb = self._circuit_breakers[tier_id]
            cb["failures"] += 1
            cb["last_failure"] = time.time()

            if cb["failures"] >= cb["threshold"]:
                cb["state"] = "open"
                logger.warning(f"Circuit breaker for {tier_id} opened after {cb['failures']} failures")

    async def record_cost(self, tier_id: str, cost: float) -> None:
        """Record cost incurred by a tier."""
        await self._resource_monitor.record_spend(cost)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive router statistics."""
        return {
            "total_routes": self._total_routes,
            "fallback_count": self._fallback_count,
            "fallback_rate": self._fallback_count / max(self._total_routes, 1),
            "tier_usage": self._tier_usage.copy(),
            "circuit_breakers": {
                tier_id: {
                    "state": cb["state"],
                    "failures": cb["failures"],
                }
                for tier_id, cb in self._circuit_breakers.items()
            },
            "available_tiers": [
                {
                    "id": tier_id,
                    "name": tier.name,
                    "tier_level": tier.tier_level,
                    "enabled": tier.enabled,
                }
                for tier_id, tier in self._tiers.items()
            ],
            "complexity_analyzer": self._analyzer.get_statistics(),
            "resource_monitor": self._resource_monitor.get_statistics(),
        }

    def get_tier_by_id(self, tier_id: str) -> Optional[TierConfig]:
        """Get tier configuration by ID."""
        return self._tiers.get(tier_id)

    def get_all_tiers(self) -> Dict[str, TierConfig]:
        """Get all tier configurations."""
        return self._tiers.copy()


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================


_router: Optional[HybridTieredRouter] = None
_router_lock = asyncio.Lock()


async def get_hybrid_tiered_router() -> HybridTieredRouter:
    """
    Get or create the global HybridTieredRouter singleton.

    Usage:
        router = await get_hybrid_tiered_router()
        result = await router.route("What is AI?")
    """
    global _router

    async with _router_lock:
        if _router is None:
            _router = HybridTieredRouter()
            await _router.initialize()

        return _router


async def shutdown_hybrid_tiered_router() -> None:
    """Shutdown the global router."""
    global _router

    async with _router_lock:
        if _router is not None:
            logger.info(f"Router statistics: {_router.get_statistics()}")
            _router = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def route_prompt(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
) -> RoutingResult:
    """
    Convenience function to route a prompt.

    Usage:
        result = await route_prompt("Design a REST API for user management")
        print(f"Use tier: {result.tier_name}")
    """
    router = await get_hybrid_tiered_router()
    return await router.route(prompt, context)


async def analyze_complexity(
    prompt: str,
    context: Optional[Dict[str, Any]] = None,
) -> ComplexityAnalysis:
    """
    Convenience function to analyze prompt complexity.

    Usage:
        analysis = await analyze_complexity("Explain quantum entanglement")
        print(f"Complexity: {analysis.score:.2f}")
    """
    router = await get_hybrid_tiered_router()
    return await router._analyzer.analyze(prompt, context)


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Data structures
    "ModelTier",
    "TierConfig",
    "ComplexityAnalysis",
    "RoutingResult",
    # Core classes
    "AdvancedComplexityAnalyzer",
    "ResourceMonitor",
    "HybridTieredRouter",
    # Factory functions
    "get_hybrid_tiered_router",
    "shutdown_hybrid_tiered_router",
    # Convenience functions
    "route_prompt",
    "analyze_complexity",
]
