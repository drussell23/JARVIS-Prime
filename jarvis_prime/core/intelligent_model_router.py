"""
Intelligent Model Router v1.0 - The Brain Router
=================================================

Routes inference requests between Local 7B, GCP 13B, and Claude API
based on complexity, resource availability, and cost optimization.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    INTELLIGENT MODEL ROUTER                     │
    │                       "The Brain Router"                        │
    └──────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ Local 7B │        │ GCP 13B  │        │Claude API│
    │ (M1 Mac) │        │ (Spot VM)│        │(Fallback)│
    │ Priority1│        │ Priority2│        │ Priority3│
    └──────────┘        └──────────┘        └──────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Response Router │
                    │ (Aggregation)   │
                    └─────────────────┘

ROUTING LOGIC:
    1. Analyze prompt complexity using HybridRouter
    2. Check resource availability (RAM, GPU, network)
    3. Select optimal tier based on complexity + resources
    4. Execute with circuit breaker protection
    5. Fallback to next tier on failure
    6. Learn from outcomes for adaptive routing

FEATURES:
    - Async/parallel execution
    - Resource-aware routing (RAM, CPU, GPU)
    - Circuit breakers per endpoint
    - Adaptive threshold learning
    - Cost tracking and optimization
    - Latency-based routing
    - Hot-reload configuration
    - Zero hardcoding (config-driven)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
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
    runtime_checkable,
)

import yaml

logger = logging.getLogger(__name__)

# =============================================================================
# TRY IMPORTS
# =============================================================================

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - resource monitoring limited")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not available - HTTP routing disabled")

try:
    from jarvis_prime.core.hybrid_router import HybridRouter, RoutingDecision, TaskType
    HYBRID_ROUTER_AVAILABLE = True
except ImportError:
    HYBRID_ROUTER_AVAILABLE = False
    logger.warning("HybridRouter not available - using basic complexity analysis")

try:
    from jarvis_prime.core.reliability_engines import (
        CircuitBreakerEngine,
        CircuitOpenError,
        OOMProtectionEngine,
        get_circuit_breaker,
        get_oom_protection,
    )
    RELIABILITY_AVAILABLE = True
except ImportError:
    RELIABILITY_AVAILABLE = False
    logger.warning("Reliability engines not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

class ModelTier(Enum):
    """Model tiers in priority order."""
    LOCAL_7B = "local_7b"
    GCP_13B = "gcp_13b"
    CLAUDE_API = "claude_api"


class RoutingStrategy(Enum):
    """Routing strategies."""
    COST_OPTIMIZED = "cost_optimized"      # Prefer cheapest
    LATENCY_OPTIMIZED = "latency_optimized"  # Prefer fastest
    QUALITY_OPTIMIZED = "quality_optimized"  # Prefer best quality
    BALANCED = "balanced"                   # Balance all factors
    FAILOVER = "failover"                   # Follow failover chain


@dataclass
class ModelEndpointConfig:
    """Configuration for a model endpoint."""
    name: str
    tier: ModelTier
    endpoint: str
    model_type: str  # llama_cpp, anthropic, openai
    model_id: Optional[str] = None
    model_path: Optional[str] = None

    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    max_context_tokens: int = 4096
    max_output_tokens: int = 2048

    # Performance
    tokens_per_second: float = 15.0
    typical_latency_ms: float = 1000.0

    # Cost
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0

    # Resources
    memory_required_mb: int = 0
    gpu_required: bool = False

    # Health
    health_endpoint: Optional[str] = None
    priority: int = 1  # Lower = higher priority

    # Enabled
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier.value,
            "endpoint": self.endpoint,
            "model_type": self.model_type,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class RoutingConfig:
    """Configuration for intelligent routing."""
    # Thresholds
    complexity_local_max: float = 0.40
    complexity_gcp_max: float = 0.75
    token_count_local_max: int = 2000
    token_count_gcp_max: int = 8000

    # Resource thresholds
    local_max_ram_percent: float = 85.0
    local_max_ram_mb: int = 14336

    # Timeout thresholds
    local_timeout_ms: int = 30000
    gcp_timeout_ms: int = 60000
    claude_timeout_ms: int = 120000

    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_recovery_seconds: float = 30.0

    # Adaptive learning
    adaptive_enabled: bool = True
    learning_rate: float = 0.05
    min_samples: int = 10

    # Strategy
    strategy: RoutingStrategy = RoutingStrategy.BALANCED

    # Fallback chain
    fallback_chain: List[ModelTier] = field(default_factory=lambda: [
        ModelTier.LOCAL_7B,
        ModelTier.GCP_13B,
        ModelTier.CLAUDE_API,
    ])


@dataclass
class ResourceSnapshot:
    """Current resource state."""
    timestamp: float
    ram_used_percent: float
    ram_used_mb: float
    ram_available_mb: float
    cpu_percent: float
    gpu_available: bool
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    network_available: bool

    @classmethod
    def capture(cls) -> "ResourceSnapshot":
        """Capture current resource state."""
        if not PSUTIL_AVAILABLE:
            return cls(
                timestamp=time.time(),
                ram_used_percent=0.0,
                ram_used_mb=0.0,
                ram_available_mb=float('inf'),
                cpu_percent=0.0,
                gpu_available=True,
                gpu_memory_used_mb=0.0,
                gpu_memory_total_mb=0.0,
                network_available=True,
            )

        mem = psutil.virtual_memory()

        return cls(
            timestamp=time.time(),
            ram_used_percent=mem.percent,
            ram_used_mb=mem.used / (1024 ** 2),
            ram_available_mb=mem.available / (1024 ** 2),
            cpu_percent=psutil.cpu_percent(),
            gpu_available=True,  # TODO: Check GPU availability
            gpu_memory_used_mb=0.0,
            gpu_memory_total_mb=0.0,
            network_available=True,  # TODO: Check network
        )

    def can_use_local(self, config: RoutingConfig) -> Tuple[bool, str]:
        """Check if local inference is viable."""
        if self.ram_used_percent > config.local_max_ram_percent:
            return False, f"RAM usage {self.ram_used_percent:.1f}% > {config.local_max_ram_percent}%"
        if self.ram_used_mb > config.local_max_ram_mb:
            return False, f"RAM used {self.ram_used_mb:.0f}MB > {config.local_max_ram_mb}MB"
        return True, "Resources available"


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    request_id: str
    tier: ModelTier
    endpoint: str
    confidence: float
    reasoning: str

    # Timing
    routing_time_ms: float
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Cost
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0

    # Response
    success: bool = False
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Fallback info
    fallback_used: bool = False
    original_tier: Optional[ModelTier] = None
    fallback_reason: Optional[str] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tier": self.tier.value,
            "success": self.success,
            "routing_time_ms": round(self.routing_time_ms, 2),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "total_time_ms": round(self.total_time_ms, 2),
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "fallback_used": self.fallback_used,
            "reasoning": self.reasoning,
        }


# =============================================================================
# MODEL ENDPOINT INTERFACE
# =============================================================================

@runtime_checkable
class ModelEndpoint(Protocol):
    """Protocol for model endpoints."""

    async def health_check(self) -> bool:
        """Check if endpoint is healthy."""
        ...

    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate response."""
        ...

    async def generate_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate response as stream."""
        ...


class BaseModelEndpoint(ABC):
    """Base class for model endpoints."""

    def __init__(self, config: ModelEndpointConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._is_healthy = False
        self._last_health_check = 0.0
        self._health_check_interval = 30.0

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not available")

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=60,
                connect=10,
            )
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )
        return self._session

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def health_check(self) -> bool:
        """Check endpoint health."""
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return self._is_healthy

        try:
            session = await self._get_session()
            url = self.config.health_endpoint or f"{self.config.endpoint}/health"

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                self._is_healthy = response.status == 200
                self._last_health_check = now
                return self._is_healthy
        except Exception as e:
            logger.debug(f"Health check failed for {self.config.name}: {e}")
            self._is_healthy = False
            self._last_health_check = now
            return False

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate response."""
        ...

    async def generate_stream(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Default stream implementation - wraps non-streaming."""
        result = await self.generate(prompt, context, **kwargs)
        content = result.get("content", result.get("response", ""))
        yield content


class LocalLlamaEndpoint(BaseModelEndpoint):
    """Endpoint for local llama.cpp inference."""

    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate using local llama.cpp server."""
        session = await self._get_session()

        # OpenAI-compatible format
        payload = {
            "model": self.config.model_path or "local",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_output_tokens),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }

        url = f"{self.config.endpoint}/v1/chat/completions"

        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Local inference failed: {error_text}")

            data = await response.json()

            return {
                "content": data["choices"][0]["message"]["content"],
                "model": self.config.name,
                "usage": data.get("usage", {}),
                "tier": ModelTier.LOCAL_7B.value,
            }


class GCPLlamaEndpoint(BaseModelEndpoint):
    """Endpoint for GCP-hosted llama inference."""

    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate using GCP llama server."""
        session = await self._get_session()

        payload = {
            "model": self.config.model_id or "llama-13b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_output_tokens),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }

        url = f"{self.config.endpoint}/v1/chat/completions"

        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"GCP inference failed: {error_text}")

            data = await response.json()

            return {
                "content": data["choices"][0]["message"]["content"],
                "model": self.config.name,
                "usage": data.get("usage", {}),
                "tier": ModelTier.GCP_13B.value,
            }


class ClaudeAPIEndpoint(BaseModelEndpoint):
    """Endpoint for Claude API."""

    def __init__(self, config: ModelEndpointConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")

    async def health_check(self) -> bool:
        """Claude API is always healthy if we have an API key."""
        return bool(self._api_key)

    async def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate using Claude API."""
        if not self._api_key:
            raise RuntimeError("Claude API key not configured")

        session = await self._get_session()

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.config.model_id or "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.config.max_output_tokens),
        }

        url = f"{self.config.endpoint}/messages"

        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Claude API failed: {error_text}")

            data = await response.json()

            content = ""
            if data.get("content"):
                content = data["content"][0].get("text", "")

            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            return {
                "content": content,
                "model": self.config.name,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
                "tier": ModelTier.CLAUDE_API.value,
            }


# =============================================================================
# COMPLEXITY ANALYZER
# =============================================================================

class ComplexityAnalyzer:
    """
    Analyzes prompt complexity for routing decisions.

    Uses HybridRouter if available, falls back to simple heuristics.
    """

    def __init__(self):
        self._hybrid_router: Optional[HybridRouter] = None
        self._initialized = False

        # Heuristic patterns
        self._reasoning_patterns = [
            "analyze", "compare", "explain why", "evaluate",
            "critique", "synthesize", "infer", "deduce",
            "hypothesize", "reason through", "think step by step",
        ]
        self._complex_patterns = [
            "implement", "design", "architect", "refactor",
            "optimize", "debug", "trace", "investigate",
        ]

    async def initialize(self):
        """Initialize the analyzer."""
        if self._initialized:
            return

        if HYBRID_ROUTER_AVAILABLE:
            try:
                from jarvis_prime.core.hybrid_router import create_agi_enhanced_router
                self._hybrid_router = create_agi_enhanced_router()
                logger.info("ComplexityAnalyzer using AGI-enhanced HybridRouter")
            except Exception as e:
                logger.warning(f"Failed to initialize HybridRouter: {e}")

        self._initialized = True

    async def analyze(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Analyze prompt complexity.

        Returns:
            Tuple of (complexity_score, task_type, signals_dict)
        """
        await self.initialize()

        if self._hybrid_router:
            decision = await self._hybrid_router.classify(prompt, context)
            return (
                decision.complexity_score,
                decision.task_type.value,
                decision.signals.to_dict(),
            )

        # Fallback to simple heuristics
        return self._analyze_simple(prompt)

    def _analyze_simple(
        self,
        prompt: str,
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Simple complexity analysis without HybridRouter."""
        prompt_lower = prompt.lower()
        words = prompt.split()

        # Base complexity from length
        length_score = min(len(words) / 500, 0.5)

        # Reasoning indicators
        reasoning_score = sum(
            1 for p in self._reasoning_patterns
            if p in prompt_lower
        ) / len(self._reasoning_patterns)

        # Complex task indicators
        complex_score = sum(
            1 for p in self._complex_patterns
            if p in prompt_lower
        ) / len(self._complex_patterns)

        # Code detection
        code_indicators = ["```", "def ", "class ", "function ", "import "]
        code_score = sum(1 for c in code_indicators if c in prompt) / len(code_indicators)

        # Calculate overall complexity
        complexity = (
            length_score * 0.2 +
            reasoning_score * 0.3 +
            complex_score * 0.3 +
            code_score * 0.2
        )

        # Determine task type
        if code_score > 0.3:
            task_type = "code_complex" if complexity > 0.5 else "code_simple"
        elif reasoning_score > 0.3:
            task_type = "reason"
        else:
            task_type = "chat"

        signals = {
            "length_score": length_score,
            "reasoning_score": reasoning_score,
            "complex_score": complex_score,
            "code_score": code_score,
            "word_count": len(words),
        }

        return complexity, task_type, signals


# =============================================================================
# ADAPTIVE THRESHOLD MANAGER
# =============================================================================

class AdaptiveThresholdManager:
    """
    Manages adaptive thresholds that learn from routing outcomes.

    Uses exponential moving average to adjust thresholds based on:
    - Success rate per tier
    - Latency per tier
    - Cost efficiency
    """

    def __init__(
        self,
        config: RoutingConfig,
        state_file: Optional[Path] = None,
    ):
        self._config = config
        self._state_file = state_file or Path.home() / ".jarvis" / "trinity" / "routing_state.json"

        # Current thresholds
        self._complexity_local_max = config.complexity_local_max
        self._complexity_gcp_max = config.complexity_gcp_max

        # Outcome tracking
        self._outcomes: Dict[ModelTier, List[Tuple[bool, float, float]]] = {
            tier: [] for tier in ModelTier
        }

        # Statistics
        self._adjustments = 0
        self._lock = asyncio.Lock()

        # Load saved state
        self._load_state()

    @property
    def complexity_local_max(self) -> float:
        return self._complexity_local_max

    @property
    def complexity_gcp_max(self) -> float:
        return self._complexity_gcp_max

    def _load_state(self):
        """Load saved state from disk."""
        if self._state_file.exists():
            try:
                data = json.loads(self._state_file.read_text())
                self._complexity_local_max = data.get("complexity_local_max", self._complexity_local_max)
                self._complexity_gcp_max = data.get("complexity_gcp_max", self._complexity_gcp_max)
                self._adjustments = data.get("adjustments", 0)
                logger.info(f"Loaded routing thresholds: local_max={self._complexity_local_max:.3f}, gcp_max={self._complexity_gcp_max:.3f}")
            except Exception as e:
                logger.warning(f"Failed to load routing state: {e}")

    def _save_state(self):
        """Save state to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "complexity_local_max": self._complexity_local_max,
                "complexity_gcp_max": self._complexity_gcp_max,
                "adjustments": self._adjustments,
                "updated_at": datetime.now().isoformat(),
            }
            self._state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Failed to save routing state: {e}")

    async def record_outcome(
        self,
        tier: ModelTier,
        success: bool,
        complexity: float,
        latency_ms: float,
    ):
        """Record an outcome for threshold adjustment."""
        async with self._lock:
            self._outcomes[tier].append((success, complexity, latency_ms))

            # Keep only last 100 samples
            if len(self._outcomes[tier]) > 100:
                self._outcomes[tier] = self._outcomes[tier][-100:]

            # Attempt adjustment
            await self._maybe_adjust()

    async def _maybe_adjust(self):
        """Adjust thresholds if we have enough samples."""
        if not self._config.adaptive_enabled:
            return

        # Need minimum samples for each tier
        for tier in [ModelTier.LOCAL_7B, ModelTier.GCP_13B]:
            if len(self._outcomes[tier]) < self._config.min_samples:
                return

        # Calculate success rates
        local_outcomes = self._outcomes[ModelTier.LOCAL_7B]
        gcp_outcomes = self._outcomes[ModelTier.GCP_13B]

        local_success_rate = sum(1 for s, _, _ in local_outcomes if s) / len(local_outcomes)
        gcp_success_rate = sum(1 for s, _, _ in gcp_outcomes if s) / len(gcp_outcomes)

        # Calculate average complexity at each tier
        local_avg_complexity = sum(c for _, c, _ in local_outcomes) / len(local_outcomes)
        gcp_avg_complexity = sum(c for _, c, _ in gcp_outcomes) / len(gcp_outcomes)

        # Adjust local threshold
        if local_success_rate > 0.85 and local_avg_complexity > self._complexity_local_max * 0.8:
            # Local is succeeding at higher complexities → raise threshold
            adjustment = (local_avg_complexity - self._complexity_local_max) * self._config.learning_rate
            new_threshold = min(0.6, self._complexity_local_max + adjustment)

            if new_threshold != self._complexity_local_max:
                logger.info(f"Raising local complexity threshold: {self._complexity_local_max:.3f} → {new_threshold:.3f}")
                self._complexity_local_max = new_threshold
                self._adjustments += 1

        elif local_success_rate < 0.7:
            # Local is failing too often → lower threshold
            adjustment = self._complexity_local_max * self._config.learning_rate
            new_threshold = max(0.2, self._complexity_local_max - adjustment)

            if new_threshold != self._complexity_local_max:
                logger.info(f"Lowering local complexity threshold: {self._complexity_local_max:.3f} → {new_threshold:.3f}")
                self._complexity_local_max = new_threshold
                self._adjustments += 1

        # Save state periodically
        if self._adjustments % 5 == 0:
            self._save_state()

    def get_statistics(self) -> Dict[str, Any]:
        """Get threshold statistics."""
        stats = {
            "complexity_local_max": self._complexity_local_max,
            "complexity_gcp_max": self._complexity_gcp_max,
            "adjustments": self._adjustments,
        }

        for tier in ModelTier:
            outcomes = self._outcomes[tier]
            if outcomes:
                success_rate = sum(1 for s, _, _ in outcomes if s) / len(outcomes)
                avg_latency = sum(l for _, _, l in outcomes) / len(outcomes)
                stats[f"{tier.value}_samples"] = len(outcomes)
                stats[f"{tier.value}_success_rate"] = round(success_rate, 3)
                stats[f"{tier.value}_avg_latency_ms"] = round(avg_latency, 1)

        return stats


# =============================================================================
# INTELLIGENT MODEL ROUTER
# =============================================================================

class IntelligentModelRouter:
    """
    The Brain Router - Intelligent routing between Local 7B, GCP 13B, and Claude API.

    Features:
        - Resource-aware routing (RAM, CPU, GPU)
        - Complexity-based tier selection
        - Circuit breaker protection
        - Adaptive threshold learning
        - Fallback chain execution
        - Cost tracking and optimization
        - Zero hardcoding (config-driven)
    """

    def __init__(
        self,
        config: Optional[RoutingConfig] = None,
        config_path: Optional[Path] = None,
    ):
        # Load configuration
        self._config = config or RoutingConfig()
        self._config_path = config_path or Path(__file__).parent.parent.parent / "config" / "unified_config.yaml"

        # Load from YAML if exists
        if self._config_path.exists():
            self._load_yaml_config()

        # Components
        self._complexity_analyzer = ComplexityAnalyzer()
        self._threshold_manager = AdaptiveThresholdManager(self._config)

        # Endpoints
        self._endpoints: Dict[ModelTier, BaseModelEndpoint] = {}

        # Circuit breakers
        self._circuit_breakers: Dict[ModelTier, CircuitBreakerEngine] = {}

        # State
        self._initialized = False
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallbacks_used": 0,
            "total_cost_usd": 0.0,
            "total_latency_ms": 0.0,
            "tier_usage": {tier.value: 0 for tier in ModelTier},
        }

        logger.info("IntelligentModelRouter initialized")

    def _load_yaml_config(self):
        """Load configuration from unified YAML."""
        try:
            with open(self._config_path) as f:
                data = yaml.safe_load(f)

            if "model_routing" in data:
                routing = data["model_routing"]

                # Load thresholds
                if "routing_rules" in routing:
                    rules = routing["routing_rules"]
                    thresholds = rules.get("complexity_thresholds", {})

                    self._config.complexity_local_max = thresholds.get("tier_0_max", self._config.complexity_local_max)
                    self._config.complexity_gcp_max = thresholds.get("tier_1_min", self._config.complexity_gcp_max)

                    resources = rules.get("resource_thresholds", {})
                    self._config.local_max_ram_percent = resources.get("local_max_ram_percent", self._config.local_max_ram_percent)
                    self._config.local_max_ram_mb = resources.get("local_max_ram_mb", self._config.local_max_ram_mb)

                logger.info(f"Loaded routing config from {self._config_path}")

        except Exception as e:
            logger.warning(f"Failed to load YAML config: {e}")

    async def initialize(self):
        """Initialize the router."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Initialize complexity analyzer
            await self._complexity_analyzer.initialize()

            # Create endpoints
            await self._create_endpoints()

            # Create circuit breakers
            if RELIABILITY_AVAILABLE:
                for tier in ModelTier:
                    self._circuit_breakers[tier] = CircuitBreakerEngine()

            self._initialized = True
            logger.info("IntelligentModelRouter initialized with endpoints")

    async def _create_endpoints(self):
        """Create model endpoints from configuration."""
        # Local 7B endpoint
        local_config = ModelEndpointConfig(
            name="Local 7B",
            tier=ModelTier.LOCAL_7B,
            endpoint=os.getenv("JARVIS_PRIME_URL", "http://localhost:8000"),
            model_type="llama_cpp",
            model_path=os.getenv("MODEL_PATH"),
            capabilities=["chat", "summarize", "format", "code_simple", "creative"],
            max_context_tokens=4096,
            max_output_tokens=2048,
            tokens_per_second=15,
            cost_per_1k_input_tokens=0.0,
            memory_required_mb=6144,
            health_endpoint="/health",
            priority=1,
        )
        self._endpoints[ModelTier.LOCAL_7B] = LocalLlamaEndpoint(local_config)

        # GCP 13B endpoint
        gcp_url = os.getenv("GCP_PRIME_URL")
        if gcp_url:
            gcp_config = ModelEndpointConfig(
                name="GCP 13B",
                tier=ModelTier.GCP_13B,
                endpoint=gcp_url,
                model_type="llama_cpp",
                model_id="llama-13b",
                capabilities=["chat", "summarize", "format", "code_simple", "code_complex", "creative", "analyze", "reason"],
                max_context_tokens=4096,
                max_output_tokens=2048,
                tokens_per_second=25,
                cost_per_1k_input_tokens=0.002,
                memory_required_mb=26624,
                health_endpoint="/health",
                priority=2,
            )
            self._endpoints[ModelTier.GCP_13B] = GCPLlamaEndpoint(gcp_config)

        # Claude API endpoint
        claude_config = ModelEndpointConfig(
            name="Claude API",
            tier=ModelTier.CLAUDE_API,
            endpoint="https://api.anthropic.com/v1",
            model_type="anthropic",
            model_id=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022"),
            capabilities=["chat", "summarize", "format", "code_simple", "code_complex", "creative", "analyze", "reason", "multimodal"],
            max_context_tokens=200000,
            max_output_tokens=8192,
            tokens_per_second=50,
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            priority=3,
        )
        self._endpoints[ModelTier.CLAUDE_API] = ClaudeAPIEndpoint(claude_config)

    async def route(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        force_tier: Optional[ModelTier] = None,
        **kwargs,
    ) -> RoutingResult:
        """
        Route a request to the optimal model tier.

        Args:
            prompt: The prompt to process
            context: Optional context
            force_tier: Force a specific tier
            **kwargs: Additional generation parameters

        Returns:
            RoutingResult with response and metadata
        """
        await self.initialize()

        request_id = str(uuid.uuid4())[:12]
        start_time = time.time()

        self._stats["total_requests"] += 1

        # Step 1: Analyze complexity
        complexity, task_type, signals = await self._complexity_analyzer.analyze(prompt, context)

        # Step 2: Check resources
        resources = ResourceSnapshot.capture()

        # Step 3: Determine tier
        if force_tier:
            selected_tier = force_tier
            confidence = 1.0
            reasoning = f"Forced to {force_tier.value}"
        else:
            selected_tier, confidence, reasoning = self._select_tier(
                complexity, task_type, resources, prompt, context
            )

        routing_time = (time.time() - start_time) * 1000

        # Step 4: Execute with fallback
        result = await self._execute_with_fallback(
            request_id=request_id,
            prompt=prompt,
            context=context,
            selected_tier=selected_tier,
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            routing_time=routing_time,
            **kwargs,
        )

        # Step 5: Record outcome for learning
        await self._threshold_manager.record_outcome(
            tier=result.tier,
            success=result.success,
            complexity=complexity,
            latency_ms=result.inference_time_ms,
        )

        return result

    def _select_tier(
        self,
        complexity: float,
        task_type: str,
        resources: ResourceSnapshot,
        prompt: str,
        context: Optional[Dict[str, Any]],
    ) -> Tuple[ModelTier, float, str]:
        """Select the optimal tier based on complexity and resources."""
        reasons = []

        # Get adaptive thresholds
        local_max = self._threshold_manager.complexity_local_max
        gcp_max = self._threshold_manager.complexity_gcp_max

        # Check resource constraints
        can_use_local, local_reason = resources.can_use_local(self._config)

        # Token count check
        token_count = len(prompt.split())

        # Multimodal check
        if context and context.get("has_images"):
            reasons.append("multimodal content requires Claude API")
            return ModelTier.CLAUDE_API, 0.95, "; ".join(reasons)

        # Resource constraint → skip local
        if not can_use_local:
            reasons.append(f"resources: {local_reason}")

            if complexity < gcp_max:
                return ModelTier.GCP_13B, 0.85, "; ".join(reasons)
            else:
                return ModelTier.CLAUDE_API, 0.8, "; ".join(reasons)

        # Complexity-based routing
        if complexity < local_max:
            reasons.append(f"low complexity ({complexity:.2f} < {local_max:.2f})")
            return ModelTier.LOCAL_7B, 0.9, "; ".join(reasons)

        elif complexity < gcp_max:
            # Check if GCP is available
            if ModelTier.GCP_13B in self._endpoints:
                reasons.append(f"medium complexity ({local_max:.2f} < {complexity:.2f} < {gcp_max:.2f})")
                return ModelTier.GCP_13B, 0.85, "; ".join(reasons)
            else:
                reasons.append("GCP not available, using local")
                return ModelTier.LOCAL_7B, 0.7, "; ".join(reasons)

        else:
            reasons.append(f"high complexity ({complexity:.2f} >= {gcp_max:.2f})")
            return ModelTier.CLAUDE_API, 0.8, "; ".join(reasons)

    async def _execute_with_fallback(
        self,
        request_id: str,
        prompt: str,
        context: Optional[Dict[str, Any]],
        selected_tier: ModelTier,
        complexity: float,
        confidence: float,
        reasoning: str,
        routing_time: float,
        **kwargs,
    ) -> RoutingResult:
        """Execute request with fallback chain."""
        original_tier = selected_tier
        fallback_used = False
        fallback_reason = None

        # Build fallback chain starting from selected tier
        fallback_chain = self._get_fallback_chain(selected_tier)

        for tier in fallback_chain:
            if tier not in self._endpoints:
                continue

            endpoint = self._endpoints[tier]

            # Check circuit breaker
            if tier in self._circuit_breakers:
                try:
                    state = self._circuit_breakers[tier].get_state(tier.value)
                    if state["state"] == "open":
                        logger.debug(f"Circuit open for {tier.value}, skipping")
                        if not fallback_used:
                            fallback_used = True
                            fallback_reason = f"circuit open for {tier.value}"
                        continue
                except Exception:
                    pass

            # Check endpoint health
            if not await endpoint.health_check():
                logger.debug(f"Endpoint {tier.value} unhealthy, skipping")
                if not fallback_used:
                    fallback_used = True
                    fallback_reason = f"{tier.value} unhealthy"
                continue

            # Attempt inference
            inference_start = time.time()

            try:
                response = await endpoint.generate(prompt, context, **kwargs)

                inference_time = (time.time() - inference_start) * 1000
                total_time = routing_time + inference_time

                # Calculate cost
                usage = response.get("usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

                cost = (
                    (input_tokens / 1000) * endpoint.config.cost_per_1k_input_tokens +
                    (output_tokens / 1000) * endpoint.config.cost_per_1k_output_tokens
                )

                # Update statistics
                self._stats["successful_requests"] += 1
                self._stats["total_cost_usd"] += cost
                self._stats["total_latency_ms"] += total_time
                self._stats["tier_usage"][tier.value] += 1

                if fallback_used:
                    self._stats["fallbacks_used"] += 1

                return RoutingResult(
                    request_id=request_id,
                    tier=tier,
                    endpoint=endpoint.config.endpoint,
                    confidence=confidence,
                    reasoning=reasoning,
                    routing_time_ms=routing_time,
                    inference_time_ms=inference_time,
                    total_time_ms=total_time,
                    estimated_cost_usd=cost,
                    actual_cost_usd=cost,
                    success=True,
                    response=response,
                    fallback_used=fallback_used,
                    original_tier=original_tier if fallback_used else None,
                    fallback_reason=fallback_reason,
                )

            except Exception as e:
                logger.warning(f"Inference failed on {tier.value}: {e}")

                # Record failure in circuit breaker
                if tier in self._circuit_breakers:
                    await self._circuit_breakers[tier]._record_failure(tier.value)

                if not fallback_used:
                    fallback_used = True
                    fallback_reason = str(e)

                continue

        # All tiers failed
        self._stats["failed_requests"] += 1

        return RoutingResult(
            request_id=request_id,
            tier=original_tier,
            endpoint="",
            confidence=0.0,
            reasoning="All tiers failed",
            routing_time_ms=routing_time,
            success=False,
            error="All tiers in fallback chain failed",
            fallback_used=True,
            original_tier=original_tier,
            fallback_reason=fallback_reason,
        )

    def _get_fallback_chain(self, starting_tier: ModelTier) -> List[ModelTier]:
        """Get fallback chain starting from a tier."""
        chain = list(self._config.fallback_chain)

        # Move starting tier to front
        if starting_tier in chain:
            chain.remove(starting_tier)
            chain.insert(0, starting_tier)

        return chain

    async def close(self):
        """Close all endpoints."""
        for endpoint in self._endpoints.values():
            await endpoint.close()

        # Save threshold state
        self._threshold_manager._save_state()

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        total = self._stats["total_requests"]

        return {
            **self._stats,
            "success_rate": self._stats["successful_requests"] / max(total, 1),
            "fallback_rate": self._stats["fallbacks_used"] / max(total, 1),
            "avg_latency_ms": self._stats["total_latency_ms"] / max(total, 1),
            "avg_cost_usd": self._stats["total_cost_usd"] / max(total, 1),
            "thresholds": self._threshold_manager.get_statistics(),
            "endpoints": {
                tier.value: {
                    "name": ep.config.name,
                    "healthy": ep._is_healthy,
                }
                for tier, ep in self._endpoints.items()
            },
        }

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all endpoints."""
        await self.initialize()

        results = {}
        for tier, endpoint in self._endpoints.items():
            results[tier.value] = await endpoint.health_check()

        return results


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_router: Optional[IntelligentModelRouter] = None
_router_lock = asyncio.Lock()


async def get_intelligent_router(
    config: Optional[RoutingConfig] = None,
) -> IntelligentModelRouter:
    """Get or create the global IntelligentModelRouter."""
    global _router

    if _router is not None:
        return _router

    async with _router_lock:
        if _router is not None:
            return _router

        _router = IntelligentModelRouter(config)
        await _router.initialize()

        return _router


async def shutdown_intelligent_router():
    """Shutdown the global router."""
    global _router

    async with _router_lock:
        if _router is not None:
            await _router.close()
            _router = None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ModelTier",
    "RoutingStrategy",
    # Configuration
    "ModelEndpointConfig",
    "RoutingConfig",
    "ResourceSnapshot",
    "RoutingResult",
    # Endpoints
    "BaseModelEndpoint",
    "LocalLlamaEndpoint",
    "GCPLlamaEndpoint",
    "ClaudeAPIEndpoint",
    # Components
    "ComplexityAnalyzer",
    "AdaptiveThresholdManager",
    # Router
    "IntelligentModelRouter",
    # Factory
    "get_intelligent_router",
    "shutdown_intelligent_router",
]
