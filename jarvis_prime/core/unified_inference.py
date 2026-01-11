"""
Unified Inference - Seamless Local/API Fallback System
=========================================================

v91.0 - Production-Grade Unified Inference with Seamless Fallback

This module provides a unified interface for model inference with:
- Automatic fallback from local models to Claude API
- Health-aware routing and circuit breakers
- Retry logic with exponential backoff
- Cost tracking and budget management
- Comprehensive metrics and observability

ARCHITECTURE:
    Request → UnifiedClient → HealthChecker → RouteDecision
                                   ↓
                    Local Model (primary) → Claude API (fallback)
                                   ↓
                    Response ← CircuitBreaker ← RetryHandler

FALLBACK CHAIN:
    1. Local 7B Model (fastest, free)
    2. Local 13B Model (more capable)
    3. GCP Hosted Model (if available)
    4. Claude Haiku (fast, cheap)
    5. Claude Sonnet (balanced)
    6. Claude Opus (most capable)

USAGE:
    client = await get_unified_client()

    # Simple generation
    response = await client.generate("What is AI?")

    # Chat with fallback
    response = await client.chat([
        {"role": "user", "content": "Hello"}
    ])

    # Stream with automatic fallback
    async for chunk in client.stream("Tell me a story"):
        print(chunk, end="")

INTEGRATION:
    - PrimeModel: Uses local models
    - AutoModelSelector: Intelligent routing
    - Claude API: Fallback provider
    - Observability: Metrics and tracing
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
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
# CONFIGURATION
# =============================================================================


@dataclass
class UnifiedInferenceConfig:
    """
    Configuration for unified inference.

    All values configurable via environment variables with UNIFIED_ prefix.
    """
    # Primary model
    primary_model: str = field(default_factory=lambda: os.getenv("UNIFIED_PRIMARY_MODEL", "prime-7b-chat-v1"))

    # Fallback chain
    fallback_chain: List[str] = field(default_factory=lambda: os.getenv(
        "UNIFIED_FALLBACK_CHAIN",
        "prime-13b-reasoning-v1,claude-3-haiku,claude-3-5-sonnet,claude-opus-4"
    ).split(","))

    # Timeouts
    request_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_REQUEST_TIMEOUT", "60.0")))
    connect_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_CONNECT_TIMEOUT", "10.0")))

    # Retries
    max_retries: int = field(default_factory=lambda: int(os.getenv("UNIFIED_MAX_RETRIES", "3")))
    retry_delay_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_RETRY_DELAY", "1.0")))
    retry_exponential_base: float = field(default_factory=lambda: float(os.getenv("UNIFIED_RETRY_EXP_BASE", "2.0")))

    # Circuit breaker
    circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv("UNIFIED_CB_THRESHOLD", "5")))
    circuit_breaker_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_CB_TIMEOUT", "60.0")))

    # Health check
    health_check_interval_seconds: float = field(default_factory=lambda: float(os.getenv("UNIFIED_HEALTH_INTERVAL", "30.0")))

    # Budget
    enable_budget_tracking: bool = field(default_factory=lambda: os.getenv("UNIFIED_BUDGET_TRACKING", "true").lower() == "true")
    daily_budget_usd: float = field(default_factory=lambda: float(os.getenv("UNIFIED_DAILY_BUDGET", "10.0")))

    # Performance
    prefer_local: bool = field(default_factory=lambda: os.getenv("UNIFIED_PREFER_LOCAL", "true").lower() == "true")
    min_local_quality: float = field(default_factory=lambda: float(os.getenv("UNIFIED_MIN_LOCAL_QUALITY", "0.7")))

    # Metrics
    enable_metrics: bool = field(default_factory=lambda: os.getenv("UNIFIED_METRICS", "true").lower() == "true")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_model": self.primary_model,
            "fallback_chain": self.fallback_chain,
            "max_retries": self.max_retries,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "daily_budget_usd": self.daily_budget_usd,
        }


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class InferenceProvider(Enum):
    """Inference provider types."""
    LOCAL = "local"
    GCP = "gcp"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


@dataclass
class InferenceRequest:
    """A unified inference request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    # Content
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None

    # Parameters
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None
    stream: bool = False

    # Routing hints
    preferred_provider: Optional[InferenceProvider] = None
    preferred_model: Optional[str] = None
    require_local: bool = False
    max_cost: Optional[float] = None
    max_latency_ms: Optional[float] = None

    # Metadata
    created_at: float = field(default_factory=time.time)
    trace_id: Optional[str] = None

    def to_chat_format(self) -> List[Dict[str, str]]:
        """Convert to chat message format."""
        if self.messages:
            return self.messages
        elif self.prompt:
            return [{"role": "user", "content": self.prompt}]
        return []


@dataclass
class InferenceResponse:
    """A unified inference response."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    request_id: str = ""

    # Content
    text: str = ""
    finish_reason: str = "stop"

    # Provider info
    provider: InferenceProvider = InferenceProvider.LOCAL
    model: str = ""
    was_fallback: bool = False
    fallback_reason: Optional[str] = None

    # Usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Performance
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    # Metadata
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "model": self.model,
            "provider": self.provider.value,
            "was_fallback": self.was_fallback,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "tokens": self.total_tokens,
        }


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, don't try
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Prevents cascading failures by stopping requests to failing services.
    """
    name: str
    threshold: int = 5
    timeout_seconds: float = 60.0

    # State
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

    # Statistics
    total_successes: int = 0
    total_failures: int = 0
    total_rejections: int = 0

    def record_success(self) -> None:
        """Record a successful call."""
        self.failure_count = 0
        self.last_success_time = time.time()
        self.total_successes += 1

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info(f"Circuit {self.name} closed after recovery")

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.total_failures += 1

        if self.failure_count >= self.threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} opened after {self.failure_count} failures")

    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit {self.name} entering half-open state")
                return True
            self.total_rejections += 1
            return False

        # Half-open: allow one request to test
        return True

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
        }


# =============================================================================
# PROVIDER BACKENDS
# =============================================================================


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    @property
    @abstractmethod
    def provider(self) -> InferenceProvider:
        """Get provider type."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        ...

    @abstractmethod
    async def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Generate a response."""
        ...

    @abstractmethod
    async def stream(
        self,
        request: InferenceRequest,
    ) -> AsyncIterator[str]:
        """Stream a response."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check backend health."""
        ...


class LocalModelBackend(InferenceBackend):
    """Backend for local PRIME models."""

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model: Optional[Any] = None
        self._is_available = False
        self._lock = asyncio.Lock()

    @property
    def provider(self) -> InferenceProvider:
        return InferenceProvider.LOCAL

    @property
    def is_available(self) -> bool:
        return self._is_available

    async def initialize(self) -> bool:
        """Initialize the local model."""
        async with self._lock:
            if self._model:
                return True

            try:
                from jarvis_prime.models.prime_model import PrimeModel
                self._model = await PrimeModel.from_pretrained_async(self._model_name)
                self._is_available = True
                logger.info(f"Initialized local model: {self._model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize local model {self._model_name}: {e}")
                self._is_available = False
                return False

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate using local model."""
        if not self._model:
            await self.initialize()

        if not self._model:
            raise RuntimeError(f"Local model {self._model_name} not available")

        start_time = time.time()

        messages = request.to_chat_format()
        result = await self._model.chat_async(
            messages=messages,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        latency_ms = (time.time() - start_time) * 1000

        return InferenceResponse(
            request_id=request.id,
            text=result.text,
            provider=self.provider,
            model=self._model_name,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.tokens_generated,
            total_tokens=result.prompt_tokens + result.tokens_generated,
            latency_ms=latency_ms,
            cost_usd=0.0,  # Local models are free
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream using local model."""
        if not self._model:
            await self.initialize()

        if not self._model:
            raise RuntimeError(f"Local model {self._model_name} not available")

        messages = request.to_chat_format()
        prompt = self._model._format_chat_messages(messages)

        async for chunk in self._model.stream_async(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            yield chunk

    async def health_check(self) -> bool:
        """Check if model is healthy."""
        return self._is_available and self._model is not None


class AnthropicBackend(InferenceBackend):
    """Backend for Claude API."""

    # Model mapping
    MODEL_MAP = {
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-opus-4": "claude-opus-4-20250514",
    }

    # Pricing per 1K tokens
    PRICING = {
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-opus-4": {"input": 0.015, "output": 0.075},
    }

    def __init__(self, model_name: str = "claude-3-haiku"):
        self._model_name = model_name
        self._client: Optional[Any] = None
        self._is_available = False

    @property
    def provider(self) -> InferenceProvider:
        return InferenceProvider.ANTHROPIC

    @property
    def is_available(self) -> bool:
        return self._is_available

    async def initialize(self) -> bool:
        """Initialize the Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic()
            self._is_available = True
            logger.info(f"Initialized Anthropic backend: {self._model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic backend: {e}")
            self._is_available = False
            return False

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate using Claude API."""
        if not self._client:
            await self.initialize()

        if not self._client:
            raise RuntimeError("Anthropic client not available")

        start_time = time.time()

        messages = request.to_chat_format()
        api_model = self.MODEL_MAP.get(self._model_name, self._model_name)

        response = await self._client.messages.create(
            model=api_model,
            max_tokens=request.max_tokens,
            messages=messages,
            temperature=request.temperature,
        )

        latency_ms = (time.time() - start_time) * 1000
        text = response.content[0].text

        # Calculate cost
        pricing = self.PRICING.get(self._model_name, {"input": 0.01, "output": 0.03})
        cost = (
            (response.usage.input_tokens / 1000) * pricing["input"] +
            (response.usage.output_tokens / 1000) * pricing["output"]
        )

        return InferenceResponse(
            request_id=request.id,
            text=text,
            provider=self.provider,
            model=self._model_name,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream using Claude API."""
        if not self._client:
            await self.initialize()

        if not self._client:
            raise RuntimeError("Anthropic client not available")

        messages = request.to_chat_format()
        api_model = self.MODEL_MAP.get(self._model_name, self._model_name)

        async with self._client.messages.stream(
            model=api_model,
            max_tokens=request.max_tokens,
            messages=messages,
            temperature=request.temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self) -> bool:
        """Check if API is accessible."""
        try:
            if not self._client:
                await self.initialize()
            # Could do a lightweight API call here
            return self._is_available
        except Exception:
            return False


# =============================================================================
# UNIFIED CLIENT
# =============================================================================


class BudgetTracker:
    """Tracks API spending against budget."""

    def __init__(self, daily_budget: float):
        self._daily_budget = daily_budget
        self._daily_spend: Dict[str, float] = {}  # date -> spend
        self._lock = asyncio.Lock()

    async def record_spend(self, amount: float) -> None:
        """Record spending."""
        async with self._lock:
            today = datetime.now().strftime("%Y-%m-%d")
            self._daily_spend[today] = self._daily_spend.get(today, 0.0) + amount

    async def get_remaining_budget(self) -> float:
        """Get remaining daily budget."""
        today = datetime.now().strftime("%Y-%m-%d")
        spent = self._daily_spend.get(today, 0.0)
        return max(0, self._daily_budget - spent)

    async def can_spend(self, amount: float) -> bool:
        """Check if spending is within budget."""
        remaining = await self.get_remaining_budget()
        return amount <= remaining

    def get_statistics(self) -> Dict[str, Any]:
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "daily_budget": self._daily_budget,
            "today_spend": self._daily_spend.get(today, 0.0),
            "remaining": max(0, self._daily_budget - self._daily_spend.get(today, 0.0)),
        }


class UnifiedInferenceClient:
    """
    Unified client for seamless local/API inference with fallback.

    Provides a single interface that automatically handles:
    - Primary model selection
    - Fallback to alternative models
    - Retries with exponential backoff
    - Circuit breaker protection
    - Budget tracking
    """

    def __init__(self, config: Optional[UnifiedInferenceConfig] = None):
        self._config = config or UnifiedInferenceConfig()

        # Backends
        self._backends: Dict[str, InferenceBackend] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Budget tracking
        self._budget_tracker: Optional[BudgetTracker] = None
        if self._config.enable_budget_tracking:
            self._budget_tracker = BudgetTracker(self._config.daily_budget_usd)

        # Health monitoring
        self._health_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_requests = 0
        self._total_fallbacks = 0
        self._total_failures = 0
        self._request_history: Deque[Dict[str, Any]] = deque(maxlen=100)

    async def initialize(self) -> None:
        """Initialize the client and backends."""
        # Initialize primary backend
        await self._get_or_create_backend(self._config.primary_model)

        # Initialize fallback backends (lazy - on first use)
        # Start health monitoring
        self._health_task = asyncio.create_task(self._health_monitor())

        logger.info("UnifiedInferenceClient initialized")

    async def shutdown(self) -> None:
        """Shutdown the client."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        logger.info("UnifiedInferenceClient shutdown")

    async def _get_or_create_backend(self, model_name: str) -> InferenceBackend:
        """Get or create a backend for a model."""
        if model_name in self._backends:
            return self._backends[model_name]

        # Determine backend type
        if model_name.startswith("prime-"):
            backend = LocalModelBackend(model_name)
        elif model_name.startswith("claude-"):
            backend = AnthropicBackend(model_name)
        else:
            backend = LocalModelBackend(model_name)

        # Initialize
        await backend.initialize()

        # Create circuit breaker
        self._circuit_breakers[model_name] = CircuitBreaker(
            name=model_name,
            threshold=self._config.circuit_breaker_threshold,
            timeout_seconds=self._config.circuit_breaker_timeout_seconds,
        )

        self._backends[model_name] = backend
        return backend

    async def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs: Any,
    ) -> InferenceResponse:
        """
        Generate a response with automatic fallback.

        Args:
            prompt: Text prompt (alternative to messages)
            messages: Chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream (returns full response)
            **kwargs: Additional parameters

        Returns:
            InferenceResponse with generation result
        """
        request = InferenceRequest(
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            **{k: v for k, v in kwargs.items() if k in InferenceRequest.__dataclass_fields__}
        )

        return await self._execute_with_fallback(request)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> InferenceResponse:
        """Chat interface."""
        return await self.generate(messages=messages, **kwargs)

    async def stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a response with automatic fallback.

        Yields text chunks as they're generated.
        """
        request = InferenceRequest(
            prompt=prompt,
            messages=messages,
            stream=True,
            **{k: v for k, v in kwargs.items() if k in InferenceRequest.__dataclass_fields__}
        )

        # Get model chain
        model_chain = [self._config.primary_model] + self._config.fallback_chain

        for model_name in model_chain:
            # Check circuit breaker
            breaker = self._circuit_breakers.get(model_name)
            if breaker and not breaker.should_allow_request():
                continue

            try:
                backend = await self._get_or_create_backend(model_name)
                if not backend.is_available:
                    continue

                async for chunk in backend.stream(request):
                    yield chunk

                if breaker:
                    breaker.record_success()
                return

            except Exception as e:
                logger.warning(f"Streaming from {model_name} failed: {e}")
                if breaker:
                    breaker.record_failure()
                continue

        raise RuntimeError("All models failed for streaming")

    async def _execute_with_fallback(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Execute request with fallback chain."""
        self._total_requests += 1

        # Build model chain
        model_chain = []

        # Check if specific model requested
        if request.preferred_model:
            model_chain.append(request.preferred_model)
        else:
            model_chain.append(self._config.primary_model)

        # Add fallbacks (unless local required)
        if not request.require_local:
            model_chain.extend(self._config.fallback_chain)

        # Remove duplicates while preserving order
        seen = set()
        model_chain = [m for m in model_chain if not (m in seen or seen.add(m))]

        last_error: Optional[Exception] = None
        was_fallback = False
        fallback_reason = None

        for model_name in model_chain:
            # Check circuit breaker
            breaker = self._circuit_breakers.get(model_name)
            if breaker and not breaker.should_allow_request():
                was_fallback = True
                fallback_reason = f"Circuit breaker open for {model_name}"
                continue

            # Check budget for API models
            if model_name.startswith("claude-") and self._budget_tracker:
                # Estimate cost
                estimated_cost = 0.01  # Simple estimate
                if not await self._budget_tracker.can_spend(estimated_cost):
                    was_fallback = True
                    fallback_reason = "Budget exceeded"
                    continue

            # Try to generate
            try:
                response = await self._execute_single(request, model_name)

                # Record success
                if breaker:
                    breaker.record_success()

                # Record cost
                if self._budget_tracker and response.cost_usd > 0:
                    await self._budget_tracker.record_spend(response.cost_usd)

                response.was_fallback = was_fallback
                response.fallback_reason = fallback_reason

                if was_fallback:
                    self._total_fallbacks += 1

                # Record to history
                self._request_history.append({
                    "request_id": request.id,
                    "model": model_name,
                    "was_fallback": was_fallback,
                    "latency_ms": response.latency_ms,
                    "cost": response.cost_usd,
                    "timestamp": time.time(),
                })

                return response

            except Exception as e:
                logger.warning(f"Generation from {model_name} failed: {e}")
                last_error = e
                was_fallback = True
                fallback_reason = str(e)

                if breaker:
                    breaker.record_failure()

        # All models failed
        self._total_failures += 1
        raise RuntimeError(f"All models in fallback chain failed. Last error: {last_error}")

    async def _execute_single(
        self,
        request: InferenceRequest,
        model_name: str,
    ) -> InferenceResponse:
        """Execute on a single model with retry."""
        backend = await self._get_or_create_backend(model_name)

        if not backend.is_available:
            raise RuntimeError(f"Backend {model_name} not available")

        # Retry loop
        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                return await asyncio.wait_for(
                    backend.generate(request),
                    timeout=self._config.request_timeout_seconds,
                )
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(f"Request to {model_name} timed out")
            except Exception as e:
                last_error = e

            # Exponential backoff
            if attempt < self._config.max_retries - 1:
                delay = self._config.retry_delay_seconds * (self._config.retry_exponential_base ** attempt)
                await asyncio.sleep(delay)

        raise last_error or RuntimeError(f"Failed to execute on {model_name}")

    async def _health_monitor(self) -> None:
        """Background health monitoring."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                for model_name, backend in self._backends.items():
                    try:
                        is_healthy = await backend.health_check()
                        if not is_healthy:
                            logger.warning(f"Backend {model_name} unhealthy")
                    except Exception as e:
                        logger.error(f"Health check failed for {model_name}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self._total_requests,
            "total_fallbacks": self._total_fallbacks,
            "total_failures": self._total_failures,
            "fallback_rate": self._total_fallbacks / max(self._total_requests, 1),
            "failure_rate": self._total_failures / max(self._total_requests, 1),
            "backends": {
                name: {
                    "available": backend.is_available,
                    "provider": backend.provider.value,
                }
                for name, backend in self._backends.items()
            },
            "circuit_breakers": {
                name: breaker.get_statistics()
                for name, breaker in self._circuit_breakers.items()
            },
            "budget": self._budget_tracker.get_statistics() if self._budget_tracker else None,
            "config": self._config.to_dict(),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_unified_client: Optional[UnifiedInferenceClient] = None
_client_lock = asyncio.Lock()


async def get_unified_client(
    config: Optional[UnifiedInferenceConfig] = None,
) -> UnifiedInferenceClient:
    """Get or create the global unified inference client."""
    global _unified_client

    async with _client_lock:
        if _unified_client is None:
            _unified_client = UnifiedInferenceClient(config)
            await _unified_client.initialize()

        return _unified_client


async def shutdown_unified_client() -> None:
    """Shutdown the global unified client."""
    global _unified_client

    async with _client_lock:
        if _unified_client:
            await _unified_client.shutdown()
            _unified_client = None


# Aliases for compatibility
get_unified_inference_client = get_unified_client
shutdown_unified_inference_client = shutdown_unified_client


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Config
    "UnifiedInferenceConfig",
    # Data structures
    "InferenceProvider",
    "InferenceRequest",
    "InferenceResponse",
    # Components
    "CircuitBreaker",
    "CircuitState",
    "BudgetTracker",
    # Backends
    "InferenceBackend",
    "LocalModelBackend",
    "AnthropicBackend",
    # Client
    "UnifiedInferenceClient",
    "get_unified_client",
    "shutdown_unified_client",
    # Aliases
    "get_unified_inference_client",
    "shutdown_unified_inference_client",
]
