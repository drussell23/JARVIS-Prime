"""
Intelligent Request Router v1.0 - The Brain's Routing System
=============================================================

This is the SECOND PIECE that closes the Trinity Loop.
The Event Bus provides the signals, the Router tells who to ask.

THE LOOP (Complete):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         THE TRINITY LOOP                                 │
    │                                                                          │
    │   ┌──────────────┐    experience    ┌───────────────┐    training      │
    │   │    JARVIS    │ ───────────────► │    Reactor    │ ───────────────  │
    │   │    (Body)    │                  │    (Nerves)   │              │   │
    │   └───────┬──────┘                  └───────────────┘              │   │
    │           │                                                        │   │
    │           │ ◄─── route_request() ───┐                              │   │
    │           │                         │                              │   │
    │   ┌───────▼──────┐           ┌──────┴────────┐    model_ready     │   │
    │   │   REQUEST    │           │   INTELLIGENT  │ ◄─────────────────┘   │
    │   │    ROUTER    │           │    ROUTER     │                        │
    │   │  (This File) │           │               │                        │
    │   └───────┬──────┘           └───────────────┘                        │
    │           │                                                           │
    │           ▼                                                           │
    │   ┌──────────────┐                                                    │
    │   │ JARVIS-Prime │                                                    │
    │   │    (Mind)    │                                                    │
    │   └──────────────┘                                                    │
    │                                                                        │
    └─────────────────────────────────────────────────────────────────────────┘

FEATURES:
    - Capability-based routing (text, code, vision, etc.)
    - Circuit breaker protected endpoints
    - Health-aware routing (skip unhealthy endpoints)
    - Dynamic model discovery via Event Bus
    - Fallback chains for resilience
    - Latency-aware load balancing
    - Cost-optimized routing
    - Request complexity analysis

USAGE:
    router = await IntelligentRequestRouter.create()

    # Simple routing
    result = await router.route_request("Generate Python code for...")

    # Capability-based routing
    result = await router.route_request(
        prompt="Analyze this image",
        required_capabilities=["vision", "analysis"],
    )

    # Get best endpoint for a task
    endpoint = await router.get_best_endpoint(
        task_type="code_generation",
        complexity="high",
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)


# =============================================================================
# TRY IMPORTS
# =============================================================================

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from jarvis_prime.core.trinity_event_bus import (
        TrinityEventBus,
        EventType,
        EventPriority,
        ComponentID,
        TrinityEvent,
        get_event_bus,
    )
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False

try:
    from jarvis_prime.core.advanced_primitives import (
        AdvancedCircuitBreaker,
        CircuitBreakerConfig,
        CircuitOpenError,
        ExponentialBackoff,
        BackoffConfig,
        TokenBucketRateLimiter,
    )
    ADVANCED_PRIMITIVES_AVAILABLE = True
except ImportError:
    ADVANCED_PRIMITIVES_AVAILABLE = False


# =============================================================================
# ENUMS
# =============================================================================

class Capability(Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CODE_COMPLETION = "code_completion"
    CODE_REVIEW = "code_review"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    REASONING = "reasoning"
    MATH = "math"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CHAT = "chat"
    INSTRUCTION_FOLLOWING = "instruction_following"
    TOOL_USE = "tool_use"


class EndpointType(Enum):
    """Types of model endpoints."""
    LOCAL_MODEL = "local_model"          # MLX/llama.cpp on local machine
    CLOUD_API = "cloud_api"               # OpenAI, Anthropic, etc.
    SELF_HOSTED = "self_hosted"           # Our GCP VMs
    HYBRID = "hybrid"                     # Can be either


class RoutingPriority(Enum):
    """Routing priority levels."""
    COST = "cost"              # Minimize cost
    LATENCY = "latency"        # Minimize latency
    QUALITY = "quality"        # Maximize quality
    BALANCED = "balanced"      # Balance all factors


class EndpointHealth(Enum):
    """Health status of an endpoint."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EndpointConfig:
    """Configuration for a model endpoint."""
    # Identity
    endpoint_id: str
    name: str
    endpoint_type: EndpointType

    # Connection
    url: str
    api_key_env: Optional[str] = None  # Environment variable for API key

    # Capabilities
    capabilities: Set[Capability] = field(default_factory=set)
    model_name: Optional[str] = None

    # Performance characteristics
    avg_latency_ms: float = 1000.0
    max_tokens: int = 4096
    cost_per_1k_tokens: float = 0.0

    # Limits
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000

    # Health
    health: EndpointHealth = EndpointHealth.UNKNOWN
    last_health_check: float = 0.0
    consecutive_failures: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "name": self.name,
            "type": self.endpoint_type.value,
            "url": self.url,
            "capabilities": [c.value for c in self.capabilities],
            "model_name": self.model_name,
            "health": self.health.value,
            "avg_latency_ms": self.avg_latency_ms,
            "enabled": self.enabled,
        }


@dataclass
class RoutingContext:
    """Context for a routing decision."""
    # Request info
    prompt: str
    required_capabilities: Set[Capability] = field(default_factory=set)
    preferred_capabilities: Set[Capability] = field(default_factory=set)

    # Constraints
    max_latency_ms: Optional[float] = None
    max_cost: Optional[float] = None
    max_tokens: Optional[int] = None

    # Priority
    priority: RoutingPriority = RoutingPriority.BALANCED

    # Metadata
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class RoutingResult:
    """Result of a routing decision."""
    success: bool
    endpoint: Optional[EndpointConfig] = None
    reason: str = ""

    # Fallback chain
    fallback_endpoints: List[EndpointConfig] = field(default_factory=list)

    # Decision metadata
    score: float = 0.0
    latency_estimate_ms: float = 0.0
    cost_estimate: float = 0.0

    # Timing
    decision_time_ms: float = 0.0


@dataclass
class RequestResult:
    """Result of a routed request."""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None

    # Endpoint used
    endpoint_id: str = ""
    endpoint_name: str = ""

    # Metrics
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0

    # Fallback info
    fallbacks_attempted: int = 0


# =============================================================================
# CIRCUIT BREAKER WRAPPER
# =============================================================================

class EndpointCircuitBreaker:
    """Circuit breaker wrapper for endpoints."""

    def __init__(self, endpoint_id: str):
        self._endpoint_id = endpoint_id
        self._state = "closed"  # closed, open, half_open
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._open_until = 0.0

        # Configuration
        self._failure_threshold = 5
        self._success_threshold = 3
        self._reset_timeout_seconds = 30.0

        # Advanced circuit breaker if available
        self._advanced_cb: Optional["AdvancedCircuitBreaker"] = None
        if ADVANCED_PRIMITIVES_AVAILABLE:
            self._advanced_cb = AdvancedCircuitBreaker(
                name=f"endpoint_{endpoint_id}",
                config=CircuitBreakerConfig(
                    failure_threshold=self._failure_threshold,
                    success_threshold=self._success_threshold,
                    timeout_seconds=self._reset_timeout_seconds,
                ),
            )

    @property
    def is_open(self) -> bool:
        """Check if circuit is open."""
        if self._advanced_cb:
            return self._advanced_cb.is_open

        if self._state == "open":
            if time.time() > self._open_until:
                self._state = "half_open"
                return False
            return True
        return False

    async def record_success(self):
        """Record a successful call."""
        if self._advanced_cb:
            await self._advanced_cb.record_success()
            return

        self._failure_count = 0
        self._success_count += 1

        if self._state == "half_open" and self._success_count >= self._success_threshold:
            self._state = "closed"
            self._success_count = 0

    async def record_failure(self):
        """Record a failed call."""
        if self._advanced_cb:
            await self._advanced_cb.record_failure(self._endpoint_id)
            return

        self._failure_count += 1
        self._success_count = 0
        self._last_failure_time = time.time()

        if self._failure_count >= self._failure_threshold:
            self._state = "open"
            self._open_until = time.time() + self._reset_timeout_seconds

    def get_state(self) -> str:
        """Get current circuit state."""
        if self._advanced_cb:
            return self._advanced_cb.get_state().value
        return self._state


# =============================================================================
# ENDPOINT MANAGER
# =============================================================================

class EndpointManager:
    """Manages model endpoints and their health."""

    def __init__(self):
        self._endpoints: Dict[str, EndpointConfig] = {}
        self._circuit_breakers: Dict[str, EndpointCircuitBreaker] = {}
        self._rate_limiters: Dict[str, "TokenBucketRateLimiter"] = {}
        self._latency_history: Dict[str, List[float]] = defaultdict(list)

        # Health check config
        self._health_check_interval = 30.0
        self._health_check_timeout = 10.0

        # Event bus integration
        self._event_bus: Optional["TrinityEventBus"] = None
        self._event_subscription: Optional[str] = None

    async def initialize(self):
        """Initialize the endpoint manager."""
        # Set up default endpoints
        await self._setup_default_endpoints()

        # Subscribe to model events
        await self._subscribe_to_events()

    async def _setup_default_endpoints(self):
        """Set up default model endpoints."""
        # Local MLX endpoint (JARVIS-Prime)
        await self.register_endpoint(EndpointConfig(
            endpoint_id="local_prime",
            name="JARVIS-Prime Local",
            endpoint_type=EndpointType.LOCAL_MODEL,
            url="http://localhost:8000/v1/chat/completions",
            capabilities={
                Capability.TEXT_GENERATION,
                Capability.CODE_GENERATION,
                Capability.CHAT,
                Capability.INSTRUCTION_FOLLOWING,
            },
            model_name="prime-local",
            avg_latency_ms=500.0,
            max_tokens=8192,
            cost_per_1k_tokens=0.0,  # Local = free
            rate_limit_rpm=120,
        ))

        # Anthropic Claude endpoint
        await self.register_endpoint(EndpointConfig(
            endpoint_id="anthropic_claude",
            name="Anthropic Claude",
            endpoint_type=EndpointType.CLOUD_API,
            url="https://api.anthropic.com/v1/messages",
            api_key_env="ANTHROPIC_API_KEY",
            capabilities={
                Capability.TEXT_GENERATION,
                Capability.CODE_GENERATION,
                Capability.REASONING,
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.CHAT,
            },
            model_name="claude-sonnet-4-20250514",
            avg_latency_ms=2000.0,
            max_tokens=200000,
            cost_per_1k_tokens=0.003,
            rate_limit_rpm=50,
        ))

        # OpenAI GPT endpoint
        await self.register_endpoint(EndpointConfig(
            endpoint_id="openai_gpt",
            name="OpenAI GPT",
            endpoint_type=EndpointType.CLOUD_API,
            url="https://api.openai.com/v1/chat/completions",
            api_key_env="OPENAI_API_KEY",
            capabilities={
                Capability.TEXT_GENERATION,
                Capability.CODE_GENERATION,
                Capability.REASONING,
                Capability.VISION,
                Capability.TOOL_USE,
                Capability.CHAT,
            },
            model_name="gpt-4o",
            avg_latency_ms=1500.0,
            max_tokens=128000,
            cost_per_1k_tokens=0.005,
            rate_limit_rpm=60,
        ))

        # GCP Self-hosted endpoint
        await self.register_endpoint(EndpointConfig(
            endpoint_id="gcp_hosted",
            name="GCP Self-Hosted",
            endpoint_type=EndpointType.SELF_HOSTED,
            url="http://gcp-prime:8000/v1/chat/completions",  # Will be updated dynamically
            capabilities={
                Capability.TEXT_GENERATION,
                Capability.CODE_GENERATION,
                Capability.CHAT,
            },
            model_name="prime-gcp",
            avg_latency_ms=800.0,
            max_tokens=16384,
            cost_per_1k_tokens=0.001,  # Our cloud = cheap
            rate_limit_rpm=100,
            enabled=False,  # Disabled until we detect it
        ))

    async def _subscribe_to_events(self):
        """Subscribe to model events from the event bus."""
        if not EVENT_BUS_AVAILABLE:
            return

        try:
            self._event_bus = await get_event_bus(ComponentID.JARVIS_BODY)

            # Subscribe to MODEL_READY events
            self._event_subscription = await self._event_bus.subscribe(
                EventType.MODEL_READY,
                self._on_model_ready,
            )

            logger.info("Endpoint manager subscribed to MODEL_READY events")

        except Exception as e:
            logger.warning(f"Failed to subscribe to events: {e}")

    async def _on_model_ready(self, event: "TrinityEvent"):
        """Handle MODEL_READY event - new model available."""
        model_name = event.payload.get("model_name", "")
        model_path = event.payload.get("model_path", "")
        capabilities = event.payload.get("capabilities", [])

        logger.info(f"[ENDPOINT] New model ready: {model_name}")

        # Update or create endpoint
        endpoint_id = f"dynamic_{model_name.replace(' ', '_').lower()}"

        # Check if this is updating an existing endpoint
        if endpoint_id in self._endpoints:
            endpoint = self._endpoints[endpoint_id]
            endpoint.model_name = model_name
            endpoint.enabled = True
            logger.info(f"  Updated existing endpoint: {endpoint_id}")
        else:
            # Create new endpoint for the model
            await self.register_endpoint(EndpointConfig(
                endpoint_id=endpoint_id,
                name=f"Dynamic Model: {model_name}",
                endpoint_type=EndpointType.LOCAL_MODEL,
                url="http://localhost:8000/v1/chat/completions",
                capabilities={Capability(c) for c in capabilities if c in [e.value for e in Capability]},
                model_name=model_name,
                avg_latency_ms=500.0,
                max_tokens=8192,
                cost_per_1k_tokens=0.0,
                metadata={"model_path": model_path, "dynamic": True},
            ))
            logger.info(f"  Created new endpoint: {endpoint_id}")

    async def register_endpoint(self, config: EndpointConfig):
        """Register a new endpoint."""
        self._endpoints[config.endpoint_id] = config
        self._circuit_breakers[config.endpoint_id] = EndpointCircuitBreaker(config.endpoint_id)

        if ADVANCED_PRIMITIVES_AVAILABLE:
            self._rate_limiters[config.endpoint_id] = TokenBucketRateLimiter(
                rate=config.rate_limit_rpm / 60.0,  # Convert RPM to RPS
                burst=config.rate_limit_rpm,  # Burst capacity = RPM
            )

        logger.debug(f"Registered endpoint: {config.name}")

    async def unregister_endpoint(self, endpoint_id: str):
        """Unregister an endpoint."""
        if endpoint_id in self._endpoints:
            del self._endpoints[endpoint_id]
        if endpoint_id in self._circuit_breakers:
            del self._circuit_breakers[endpoint_id]
        if endpoint_id in self._rate_limiters:
            del self._rate_limiters[endpoint_id]

    def get_endpoint(self, endpoint_id: str) -> Optional[EndpointConfig]:
        """Get an endpoint by ID."""
        return self._endpoints.get(endpoint_id)

    def get_all_endpoints(self) -> List[EndpointConfig]:
        """Get all registered endpoints."""
        return list(self._endpoints.values())

    def get_healthy_endpoints(self) -> List[EndpointConfig]:
        """Get all healthy endpoints."""
        return [
            ep for ep in self._endpoints.values()
            if ep.enabled and ep.health != EndpointHealth.UNHEALTHY
            and not self._circuit_breakers[ep.endpoint_id].is_open
        ]

    def get_endpoints_with_capability(self, capability: Capability) -> List[EndpointConfig]:
        """Get endpoints that have a specific capability."""
        return [
            ep for ep in self.get_healthy_endpoints()
            if capability in ep.capabilities
        ]

    def get_endpoints_with_all_capabilities(self, capabilities: Set[Capability]) -> List[EndpointConfig]:
        """Get endpoints that have all specified capabilities."""
        return [
            ep for ep in self.get_healthy_endpoints()
            if capabilities.issubset(ep.capabilities)
        ]

    async def record_latency(self, endpoint_id: str, latency_ms: float):
        """Record a latency measurement."""
        history = self._latency_history[endpoint_id]
        history.append(latency_ms)

        # Keep last 100 measurements
        if len(history) > 100:
            history.pop(0)

        # Update average
        if endpoint_id in self._endpoints:
            self._endpoints[endpoint_id].avg_latency_ms = sum(history) / len(history)

    async def record_success(self, endpoint_id: str):
        """Record a successful call."""
        if endpoint_id in self._circuit_breakers:
            await self._circuit_breakers[endpoint_id].record_success()

        if endpoint_id in self._endpoints:
            self._endpoints[endpoint_id].consecutive_failures = 0
            if self._endpoints[endpoint_id].health != EndpointHealth.HEALTHY:
                self._endpoints[endpoint_id].health = EndpointHealth.HEALTHY

    async def record_failure(self, endpoint_id: str):
        """Record a failed call."""
        if endpoint_id in self._circuit_breakers:
            await self._circuit_breakers[endpoint_id].record_failure()

        if endpoint_id in self._endpoints:
            self._endpoints[endpoint_id].consecutive_failures += 1
            if self._endpoints[endpoint_id].consecutive_failures >= 3:
                self._endpoints[endpoint_id].health = EndpointHealth.UNHEALTHY

    async def can_call(self, endpoint_id: str) -> bool:
        """Check if we can call an endpoint (rate limit + circuit breaker)."""
        # Check circuit breaker
        if endpoint_id in self._circuit_breakers:
            if self._circuit_breakers[endpoint_id].is_open:
                return False

        # Check rate limiter
        if endpoint_id in self._rate_limiters:
            if not await self._rate_limiters[endpoint_id].acquire():
                return False

        return True


# =============================================================================
# INTELLIGENT REQUEST ROUTER
# =============================================================================

class IntelligentRequestRouter:
    """
    Intelligent Request Router - Routes requests to the best endpoint.

    This is the second piece that closes the Trinity Loop:
    - Event Bus provides signals (model_ready, training_complete, etc.)
    - Request Router tells the Body which model to use
    """

    _instance: Optional["IntelligentRequestRouter"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self):
        self._endpoint_manager = EndpointManager()
        self._initialized = False

        # Routing configuration
        self._default_priority = RoutingPriority.BALANCED
        self._enable_fallbacks = True
        self._max_fallback_attempts = 3

        # Session for HTTP calls
        self._session: Optional[aiohttp.ClientSession] = None

        # Metrics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._fallback_count = 0

    @classmethod
    async def create(cls) -> "IntelligentRequestRouter":
        """Create or get the singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    @classmethod
    async def get_instance(cls) -> Optional["IntelligentRequestRouter"]:
        """Get existing instance or None."""
        return cls._instance

    async def initialize(self):
        """Initialize the router."""
        if self._initialized:
            return

        # Initialize endpoint manager
        await self._endpoint_manager.initialize()

        # Create HTTP session
        if AIOHTTP_AVAILABLE:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60.0)
            )

        self._initialized = True
        logger.info("IntelligentRequestRouter initialized")

    async def shutdown(self):
        """Shutdown the router."""
        if self._session:
            await self._session.close()
            self._session = None

        self._initialized = False

        async with self._lock:
            IntelligentRequestRouter._instance = None

        logger.info("IntelligentRequestRouter shutdown")

    # =========================================================================
    # ROUTING API
    # =========================================================================

    async def route_request(
        self,
        prompt: str,
        required_capabilities: Optional[Set[Capability]] = None,
        priority: RoutingPriority = RoutingPriority.BALANCED,
        max_latency_ms: Optional[float] = None,
        max_cost: Optional[float] = None,
    ) -> RequestResult:
        """
        Route a request to the best available endpoint.

        This is THE KEY METHOD for the Body to call:

        USAGE:
            router = await IntelligentRequestRouter.create()
            result = await router.route_request(
                prompt="Generate Python code for sorting a list",
                required_capabilities={Capability.CODE_GENERATION},
            )

            if result.success:
                print(f"Response: {result.response}")
                print(f"Used endpoint: {result.endpoint_name}")

        Args:
            prompt: The prompt to send
            required_capabilities: Required model capabilities
            priority: Routing priority (cost, latency, quality, balanced)
            max_latency_ms: Maximum acceptable latency
            max_cost: Maximum acceptable cost

        Returns:
            RequestResult with response and metadata
        """
        self._total_requests += 1
        start_time = time.time()

        # Build routing context
        context = RoutingContext(
            prompt=prompt,
            required_capabilities=required_capabilities or set(),
            priority=priority,
            max_latency_ms=max_latency_ms,
            max_cost=max_cost,
            request_id=hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:8],
        )

        # Find best endpoint
        routing_result = await self._find_best_endpoint(context)

        if not routing_result.success or not routing_result.endpoint:
            self._failed_requests += 1
            return RequestResult(
                success=False,
                error=f"No suitable endpoint found: {routing_result.reason}",
            )

        # Try primary endpoint and fallbacks
        endpoints_to_try = [routing_result.endpoint] + routing_result.fallback_endpoints[:self._max_fallback_attempts]

        for i, endpoint in enumerate(endpoints_to_try):
            if i > 0:
                self._fallback_count += 1
                logger.info(f"Trying fallback endpoint: {endpoint.name}")

            result = await self._call_endpoint(endpoint, prompt)

            if result.success:
                self._successful_requests += 1
                result.fallbacks_attempted = i
                return result

        # All endpoints failed
        self._failed_requests += 1
        return RequestResult(
            success=False,
            error="All endpoints failed",
            fallbacks_attempted=len(endpoints_to_try) - 1,
        )

    async def get_best_endpoint(
        self,
        task_type: str = "general",
        complexity: str = "medium",
        required_capabilities: Optional[Set[Capability]] = None,
    ) -> Optional[EndpointConfig]:
        """
        Get the best endpoint for a task type.

        Args:
            task_type: Type of task (code_generation, text_generation, etc.)
            complexity: Task complexity (low, medium, high)
            required_capabilities: Required capabilities

        Returns:
            Best endpoint or None
        """
        # Map task type to capabilities
        capability_map = {
            "code_generation": {Capability.CODE_GENERATION},
            "code_completion": {Capability.CODE_COMPLETION},
            "code_review": {Capability.CODE_REVIEW, Capability.REASONING},
            "text_generation": {Capability.TEXT_GENERATION},
            "chat": {Capability.CHAT},
            "reasoning": {Capability.REASONING},
            "vision": {Capability.VISION},
            "general": {Capability.INSTRUCTION_FOLLOWING},
        }

        capabilities = required_capabilities or capability_map.get(task_type, set())

        # Build context
        context = RoutingContext(
            prompt="",  # Not needed for endpoint selection
            required_capabilities=capabilities,
        )

        result = await self._find_best_endpoint(context)
        return result.endpoint if result.success else None

    # =========================================================================
    # INTERNAL ROUTING LOGIC
    # =========================================================================

    async def _find_best_endpoint(self, context: RoutingContext) -> RoutingResult:
        """Find the best endpoint for a request."""
        start_time = time.time()

        # Get candidate endpoints
        if context.required_capabilities:
            candidates = self._endpoint_manager.get_endpoints_with_all_capabilities(
                context.required_capabilities
            )
        else:
            candidates = self._endpoint_manager.get_healthy_endpoints()

        if not candidates:
            return RoutingResult(
                success=False,
                reason="No endpoints available with required capabilities",
                decision_time_ms=(time.time() - start_time) * 1000,
            )

        # Filter by constraints
        filtered_candidates = []
        for ep in candidates:
            # Check latency constraint
            if context.max_latency_ms and ep.avg_latency_ms > context.max_latency_ms:
                continue

            # Check cost constraint (estimate based on average tokens)
            if context.max_cost:
                estimated_cost = (1000 / 1000) * ep.cost_per_1k_tokens  # Assume 1k tokens
                if estimated_cost > context.max_cost:
                    continue

            # Check if we can call (rate limit + circuit breaker)
            if not await self._endpoint_manager.can_call(ep.endpoint_id):
                continue

            filtered_candidates.append(ep)

        if not filtered_candidates:
            return RoutingResult(
                success=False,
                reason="No endpoints meet constraints (latency/cost/rate limit)",
                decision_time_ms=(time.time() - start_time) * 1000,
            )

        # Score candidates
        scored_candidates = []
        for ep in filtered_candidates:
            score = self._score_endpoint(ep, context)
            scored_candidates.append((score, ep))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        best_score, best_endpoint = scored_candidates[0]
        fallbacks = [ep for _, ep in scored_candidates[1:self._max_fallback_attempts + 1]]

        return RoutingResult(
            success=True,
            endpoint=best_endpoint,
            fallback_endpoints=fallbacks,
            score=best_score,
            latency_estimate_ms=best_endpoint.avg_latency_ms,
            cost_estimate=(1000 / 1000) * best_endpoint.cost_per_1k_tokens,
            decision_time_ms=(time.time() - start_time) * 1000,
        )

    def _score_endpoint(self, endpoint: EndpointConfig, context: RoutingContext) -> float:
        """Score an endpoint based on routing priority."""
        score = 0.0

        # Base scores
        latency_score = 1.0 - min(endpoint.avg_latency_ms / 10000, 1.0)  # 0-10s scale
        cost_score = 1.0 - min(endpoint.cost_per_1k_tokens / 0.01, 1.0)  # 0-$0.01 scale
        quality_score = 0.5  # Default quality

        # Adjust quality based on endpoint type
        if endpoint.endpoint_type == EndpointType.CLOUD_API:
            quality_score = 0.9  # Cloud APIs are generally high quality
        elif endpoint.endpoint_type == EndpointType.LOCAL_MODEL:
            quality_score = 0.7  # Local models are decent

        # Adjust based on health
        if endpoint.health == EndpointHealth.DEGRADED:
            quality_score *= 0.8

        # Weight based on priority
        if context.priority == RoutingPriority.COST:
            score = cost_score * 0.6 + latency_score * 0.2 + quality_score * 0.2
        elif context.priority == RoutingPriority.LATENCY:
            score = latency_score * 0.6 + quality_score * 0.2 + cost_score * 0.2
        elif context.priority == RoutingPriority.QUALITY:
            score = quality_score * 0.6 + latency_score * 0.2 + cost_score * 0.2
        else:  # BALANCED
            score = latency_score * 0.33 + cost_score * 0.33 + quality_score * 0.34

        # Bonus for matching more capabilities
        if context.preferred_capabilities:
            matched = len(context.preferred_capabilities & endpoint.capabilities)
            total = len(context.preferred_capabilities)
            if total > 0:
                score *= (1 + 0.2 * matched / total)

        return score

    async def _call_endpoint(
        self,
        endpoint: EndpointConfig,
        prompt: str,
    ) -> RequestResult:
        """Call an endpoint and return the result."""
        start_time = time.time()

        try:
            if endpoint.endpoint_type == EndpointType.LOCAL_MODEL:
                return await self._call_local_endpoint(endpoint, prompt, start_time)
            elif endpoint.endpoint_type == EndpointType.CLOUD_API:
                return await self._call_cloud_endpoint(endpoint, prompt, start_time)
            elif endpoint.endpoint_type == EndpointType.SELF_HOSTED:
                return await self._call_self_hosted_endpoint(endpoint, prompt, start_time)
            else:
                return RequestResult(
                    success=False,
                    error=f"Unknown endpoint type: {endpoint.endpoint_type}",
                    endpoint_id=endpoint.endpoint_id,
                    endpoint_name=endpoint.name,
                )
        except Exception as e:
            await self._endpoint_manager.record_failure(endpoint.endpoint_id)
            return RequestResult(
                success=False,
                error=str(e),
                endpoint_id=endpoint.endpoint_id,
                endpoint_name=endpoint.name,
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _call_local_endpoint(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        start_time: float,
    ) -> RequestResult:
        """Call a local model endpoint (OpenAI-compatible)."""
        if not self._session:
            return RequestResult(
                success=False,
                error="HTTP session not available",
                endpoint_id=endpoint.endpoint_id,
                endpoint_name=endpoint.name,
            )

        try:
            payload = {
                "model": endpoint.model_name or "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": min(4096, endpoint.max_tokens),
            }

            async with self._session.post(endpoint.url, json=payload) as response:
                latency_ms = (time.time() - start_time) * 1000

                if response.status == 200:
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    tokens = data.get("usage", {}).get("total_tokens", 0)

                    await self._endpoint_manager.record_success(endpoint.endpoint_id)
                    await self._endpoint_manager.record_latency(endpoint.endpoint_id, latency_ms)

                    return RequestResult(
                        success=True,
                        response=content,
                        endpoint_id=endpoint.endpoint_id,
                        endpoint_name=endpoint.name,
                        latency_ms=latency_ms,
                        tokens_used=tokens,
                        cost=tokens / 1000 * endpoint.cost_per_1k_tokens,
                    )
                else:
                    await self._endpoint_manager.record_failure(endpoint.endpoint_id)
                    return RequestResult(
                        success=False,
                        error=f"HTTP {response.status}: {await response.text()}",
                        endpoint_id=endpoint.endpoint_id,
                        endpoint_name=endpoint.name,
                        latency_ms=latency_ms,
                    )

        except asyncio.TimeoutError:
            await self._endpoint_manager.record_failure(endpoint.endpoint_id)
            return RequestResult(
                success=False,
                error="Request timeout",
                endpoint_id=endpoint.endpoint_id,
                endpoint_name=endpoint.name,
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _call_cloud_endpoint(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        start_time: float,
    ) -> RequestResult:
        """Call a cloud API endpoint."""
        # For now, delegate to local endpoint logic (most cloud APIs are OpenAI-compatible)
        # TODO: Add specific handling for Anthropic, OpenAI, etc.
        return await self._call_local_endpoint(endpoint, prompt, start_time)

    async def _call_self_hosted_endpoint(
        self,
        endpoint: EndpointConfig,
        prompt: str,
        start_time: float,
    ) -> RequestResult:
        """Call a self-hosted endpoint (GCP VM)."""
        # Same as local endpoint (OpenAI-compatible)
        return await self._call_local_endpoint(endpoint, prompt, start_time)

    # =========================================================================
    # STATUS AND METRICS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "initialized": self._initialized,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "fallback_count": self._fallback_count,
            "success_rate": (
                self._successful_requests / self._total_requests
                if self._total_requests > 0 else 0.0
            ),
            "endpoints": [ep.to_dict() for ep in self._endpoint_manager.get_all_endpoints()],
            "healthy_endpoints": len(self._endpoint_manager.get_healthy_endpoints()),
        }

    def get_endpoints(self) -> List[Dict[str, Any]]:
        """Get all endpoint configurations."""
        return [ep.to_dict() for ep in self._endpoint_manager.get_all_endpoints()]


# =============================================================================
# GLOBAL ACCESS FUNCTIONS
# =============================================================================

_router: Optional[IntelligentRequestRouter] = None


async def get_request_router() -> IntelligentRequestRouter:
    """Get or create the request router."""
    global _router
    if _router is None:
        _router = await IntelligentRequestRouter.create()
    return _router


async def shutdown_request_router():
    """Shutdown the request router."""
    global _router
    if _router is not None:
        await _router.shutdown()
        _router = None


async def route_request(
    prompt: str,
    required_capabilities: Optional[Set[Capability]] = None,
    priority: RoutingPriority = RoutingPriority.BALANCED,
) -> RequestResult:
    """Convenience function to route a request."""
    router = await get_request_router()
    return await router.route_request(
        prompt=prompt,
        required_capabilities=required_capabilities,
        priority=priority,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "Capability",
    "EndpointType",
    "RoutingPriority",
    "EndpointHealth",
    # Data structures
    "EndpointConfig",
    "RoutingContext",
    "RoutingResult",
    "RequestResult",
    # Router
    "IntelligentRequestRouter",
    "EndpointManager",
    "EndpointCircuitBreaker",
    # Functions
    "get_request_router",
    "shutdown_request_router",
    "route_request",
]
