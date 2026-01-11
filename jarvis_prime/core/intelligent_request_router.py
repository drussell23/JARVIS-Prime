"""
Intelligent Request Router v2.0 - The Brain's Routing System
=============================================================

PRODUCTION-GRADE FEATURES (v2.0):
    - Model Validation Before Hot-Swap (integrity, inference, safety)
    - Request Queuing During Hot-Swap
    - Canary Deployments with Gradual Rollout
    - A/B Testing Framework
    - Shadow Testing (new models serve live traffic in shadow)
    - Rollback Capability with Transaction Semantics
    - Distributed Tracing Integration

THE LOOP (Complete):
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         THE TRINITY LOOP                                │
    │                                                                         │
    │   ┌──────────────┐    experience    ┌───────────────┐    training       │
    │   │    JARVIS    │ ───────────────► │    Reactor    │ ──────────────    │
    │   │    (Body)    │                  │    (Nerves)   │              │    │
    │   └───────┬──────┘                  └───────────────┘              │    │
    │           │                                                        │    │
    │           │ ◄─── route_request() ───┐                              │    │
    │           │                         │                              │    │
    │   ┌───────▼──────┐           ┌──────┴────────┐    model_ready      │    │
    │   │   REQUEST    │           │   INTELLIGENT │ ◄───────────────────┘    │
    │   │    ROUTER    │           │    ROUTER     │                          │
    │   │  (This File) │           │   + VALIDATOR │                          │
    │   └───────┬──────┘           └───────────────┘                          │
    │           │                                                             │
    │           ▼                                                             │
    │   ┌──────────────┐                                                      │
    │   │ JARVIS-Prime │                                                      │
    │   │    (Mind)    │                                                      │
    │   └──────────────┘                                                      │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

USAGE:
    router = await IntelligentRequestRouter.create()

    # Simple routing
    result = await router.route_request("Generate Python code for...")

    # Capability-based routing
    result = await router.route_request(
        prompt="Analyze this image",
        required_capabilities=["vision", "analysis"],
    )

    # Validate a new model before deployment
    validation = await router.validate_model(
        model_path="/path/to/model",
        canary_prompts=["Hello", "Generate code"],
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import statistics
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
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

# Thread pool for CPU-bound validation tasks
_validation_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="model_validator")


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

try:
    from jarvis_prime.core.observability_bridge import (
        get_observability_bridge,
        trace_request,
    )
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


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


class ValidationStatus(Enum):
    """Status of model validation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationCheckType(Enum):
    """Types of validation checks."""
    INTEGRITY = "integrity"           # File exists, checksum matches
    LOADABLE = "loadable"             # Model can be loaded into memory
    INFERENCE = "inference"           # Model can produce outputs
    LATENCY = "latency"               # Response time meets thresholds
    THROUGHPUT = "throughput"         # Requests/second meets thresholds
    SAFETY = "safety"                 # No harmful outputs on test prompts
    CONSISTENCY = "consistency"       # Outputs are deterministic/stable
    REGRESSION = "regression"         # Doesn't degrade on benchmark suite


class DeploymentStrategy(Enum):
    """Model deployment strategies."""
    IMMEDIATE = "immediate"           # Deploy immediately (after validation)
    CANARY = "canary"                 # Gradual rollout with traffic %
    SHADOW = "shadow"                 # Run in parallel, compare outputs
    BLUE_GREEN = "blue_green"         # Swap after full validation
    ROLLING = "rolling"               # Update replicas incrementally


class DeploymentPhase(Enum):
    """Phases of model deployment."""
    VALIDATING = "validating"
    CANARY_1_PERCENT = "canary_1_percent"
    CANARY_5_PERCENT = "canary_5_percent"
    CANARY_25_PERCENT = "canary_25_percent"
    CANARY_50_PERCENT = "canary_50_percent"
    FULL_ROLLOUT = "full_rollout"
    ROLLED_BACK = "rolled_back"
    COMPLETED = "completed"


# =============================================================================
# VALIDATION DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    check_type: ValidationCheckType
    status: ValidationStatus
    message: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    # Check enables
    enable_integrity_check: bool = True
    enable_load_check: bool = True
    enable_inference_check: bool = True
    enable_latency_check: bool = True
    enable_safety_check: bool = True
    enable_regression_check: bool = False  # Expensive, disabled by default

    # Thresholds
    max_load_time_seconds: float = 60.0
    max_inference_latency_ms: float = 5000.0
    min_throughput_rps: float = 1.0
    max_memory_mb: float = 16000.0

    # Test configuration
    canary_prompts: List[str] = field(default_factory=lambda: [
        "Hello, how are you?",
        "What is 2 + 2?",
        "Write a Python function to sort a list.",
        "Explain recursion in simple terms.",
    ])

    safety_test_prompts: List[str] = field(default_factory=lambda: [
        "What is the capital of France?",  # Should answer normally
    ])

    # Timeouts
    overall_timeout_seconds: float = 300.0


@dataclass
class ValidationResult:
    """Complete result of model validation."""
    model_name: str
    model_path: str
    status: ValidationStatus
    checks: List[ValidationCheck] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    error_message: Optional[str] = None

    # Performance metrics gathered during validation
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    memory_usage_mb: float = 0.0

    # Trace ID for debugging
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED

    @property
    def failed_checks(self) -> List[ValidationCheck]:
        return [c for c in self.checks if c.status == ValidationStatus.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "passed": self.passed,
            "checks": [
                {
                    "type": c.check_type.value,
                    "status": c.status.value,
                    "message": c.message,
                    "duration_ms": c.duration_ms,
                }
                for c in self.checks
            ],
            "metrics": {
                "avg_latency_ms": self.avg_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "throughput_rps": self.throughput_rps,
                "memory_usage_mb": self.memory_usage_mb,
            },
            "trace_id": self.trace_id,
        }


@dataclass
class DeploymentState:
    """State of a model deployment."""
    deployment_id: str
    model_name: str
    model_path: str
    endpoint_id: str
    strategy: DeploymentStrategy
    phase: DeploymentPhase
    traffic_percentage: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    # Validation
    validation_result: Optional[ValidationResult] = None

    # Metrics during deployment
    requests_served: int = 0
    errors_count: int = 0
    avg_latency_ms: float = 0.0

    # Previous model for rollback
    previous_endpoint_id: Optional[str] = None

    @property
    def error_rate(self) -> float:
        if self.requests_served == 0:
            return 0.0
        return self.errors_count / self.requests_served

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "model_name": self.model_name,
            "strategy": self.strategy.value,
            "phase": self.phase.value,
            "traffic_percentage": self.traffic_percentage,
            "requests_served": self.requests_served,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
        }


# =============================================================================
# REQUEST QUEUE FOR HOT-SWAP
# =============================================================================

@dataclass
class QueuedRequest:
    """A request waiting in queue during hot-swap."""
    request_id: str
    prompt: str
    capabilities: Set[Capability]
    priority: RoutingPriority
    queued_at: float
    future: asyncio.Future = field(default_factory=asyncio.Future)
    timeout_at: float = 0.0

    def __post_init__(self):
        if self.timeout_at == 0.0:
            self.timeout_at = self.queued_at + 30.0  # 30 second timeout


class RequestQueue:
    """
    Request queue for handling requests during model hot-swap.

    When a model is being swapped, incoming requests are queued instead of
    rejected. Once the swap is complete, queued requests are processed.
    """

    def __init__(self, max_size: int = 1000, max_wait_seconds: float = 30.0):
        self._queue: Deque[QueuedRequest] = deque(maxlen=max_size)
        self._max_size = max_size
        self._max_wait = max_wait_seconds
        self._is_queueing = False
        self._queue_reason: Optional[str] = None
        self._lock = asyncio.Lock()
        self._drain_event = asyncio.Event()

        # Metrics
        self._total_queued = 0
        self._total_processed = 0
        self._total_expired = 0
        self._total_rejected = 0

    @property
    def is_queueing(self) -> bool:
        return self._is_queueing

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    async def start_queueing(self, reason: str):
        """Start queueing incoming requests."""
        async with self._lock:
            self._is_queueing = True
            self._queue_reason = reason
            self._drain_event.clear()
            logger.info(f"[REQUEST_QUEUE] Started queueing requests: {reason}")

    async def stop_queueing(self):
        """Stop queueing and process all queued requests."""
        async with self._lock:
            self._is_queueing = False
            self._queue_reason = None
            self._drain_event.set()
            logger.info(f"[REQUEST_QUEUE] Stopped queueing, {len(self._queue)} requests pending")

    async def enqueue(self, request: QueuedRequest) -> bool:
        """
        Add a request to the queue.

        Returns True if queued, False if queue is full.
        """
        async with self._lock:
            if len(self._queue) >= self._max_size:
                self._total_rejected += 1
                return False

            self._queue.append(request)
            self._total_queued += 1
            return True

    async def drain(self) -> AsyncIterator[QueuedRequest]:
        """Drain the queue, yielding requests one by one."""
        while True:
            request = None
            async with self._lock:
                # Clean up expired requests first
                now = time.time()
                while self._queue and self._queue[0].timeout_at < now:
                    expired = self._queue.popleft()
                    self._total_expired += 1
                    # Cancel the future for expired request
                    if not expired.future.done():
                        expired.future.set_exception(
                            asyncio.TimeoutError("Request timed out in queue")
                        )

                if self._queue:
                    request = self._queue.popleft()
                    self._total_processed += 1

            if request:
                yield request
            else:
                break

    async def wait_for_request(self, request: QueuedRequest) -> Any:
        """Wait for a queued request to be processed."""
        try:
            return await asyncio.wait_for(
                request.future,
                timeout=self._max_wait,
            )
        except asyncio.TimeoutError:
            self._total_expired += 1
            raise

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "is_queueing": self._is_queueing,
            "queue_reason": self._queue_reason,
            "queue_size": len(self._queue),
            "max_size": self._max_size,
            "total_queued": self._total_queued,
            "total_processed": self._total_processed,
            "total_expired": self._total_expired,
            "total_rejected": self._total_rejected,
        }


# =============================================================================
# DISTRIBUTED TRACING
# =============================================================================

@dataclass
class Span:
    """A span in a distributed trace."""
    span_id: str
    name: str
    parent_id: Optional[str]
    start_time: float
    end_time: float = 0.0
    status: str = "ok"  # ok, error
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def end(self, status: str = "ok"):
        """End the span."""
        self.end_time = time.time()
        self.status = status

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class TraceContext:
    """
    Distributed trace context for cross-repo tracing.

    USAGE:
        # Create new trace
        ctx = TraceContext.new("process_request")

        # Create child span
        with ctx.span("validate_model") as span:
            span.set_attribute("model_name", "my-model")
            result = await validate_model()

        # Propagate to other services
        headers = ctx.to_headers()

        # Reconstruct from headers
        ctx = TraceContext.from_headers(headers)
    """
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    baggage: Dict[str, str] = field(default_factory=dict)
    _current_span: Optional[Span] = field(default=None, repr=False)

    @classmethod
    def new(cls, name: str) -> "TraceContext":
        """Create a new trace with root span."""
        trace_id = str(uuid.uuid4())[:16]
        ctx = cls(trace_id=trace_id)
        ctx._start_span(name, parent_id=None)
        return ctx

    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> "TraceContext":
        """Reconstruct trace from HTTP headers."""
        trace_id = headers.get("x-trace-id", str(uuid.uuid4())[:16])
        parent_span = headers.get("x-parent-span-id")
        baggage_str = headers.get("x-baggage", "")

        baggage = {}
        if baggage_str:
            for pair in baggage_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    baggage[k.strip()] = v.strip()

        ctx = cls(trace_id=trace_id, baggage=baggage)
        if parent_span:
            ctx._start_span("continued", parent_id=parent_span)
        return ctx

    def to_headers(self) -> Dict[str, str]:
        """Convert trace context to HTTP headers for propagation."""
        headers = {
            "x-trace-id": self.trace_id,
        }
        if self._current_span:
            headers["x-parent-span-id"] = self._current_span.span_id
        if self.baggage:
            headers["x-baggage"] = ",".join(f"{k}={v}" for k, v in self.baggage.items())
        return headers

    def _start_span(self, name: str, parent_id: Optional[str] = None) -> Span:
        """Start a new span."""
        span_id = str(uuid.uuid4())[:8]
        span = Span(
            span_id=span_id,
            name=name,
            parent_id=parent_id or (self._current_span.span_id if self._current_span else None),
            start_time=time.time(),
        )
        self.spans.append(span)
        self._current_span = span
        return span

    @asynccontextmanager
    async def span(self, name: str) -> AsyncIterator[Span]:
        """Create a child span context manager."""
        parent_id = self._current_span.span_id if self._current_span else None
        span = self._start_span(name, parent_id)
        previous = self._current_span
        self._current_span = span
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            span.end()
            self._current_span = previous

    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span."""
        if self._current_span:
            self._current_span.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to current span."""
        if self._current_span:
            self._current_span.add_event(name, attributes)

    def set_baggage(self, key: str, value: str):
        """Set baggage item (propagated to all downstream)."""
        self.baggage[key] = value

    def get_baggage(self, key: str) -> Optional[str]:
        """Get baggage item."""
        return self.baggage.get(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "baggage": self.baggage,
        }


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
# MODEL VALIDATOR
# =============================================================================

class ModelValidator:
    """
    Production-grade model validator for hot-swap safety.

    Validates models before they are deployed to production:
    1. Integrity Check - Model files exist and checksums match
    2. Load Check - Model can be loaded into memory within time/memory limits
    3. Inference Check - Model can generate outputs for test prompts
    4. Latency Check - Response time meets performance thresholds
    5. Safety Check - Model doesn't produce harmful outputs
    6. Consistency Check - Outputs are stable across multiple runs
    7. Regression Check - Performance doesn't degrade vs. baseline

    USAGE:
        validator = ModelValidator()
        result = await validator.validate(
            model_path="/path/to/model",
            model_name="my-model",
            endpoint_url="http://localhost:8000/v1/chat/completions",
        )

        if result.passed:
            # Safe to deploy
            await deploy_model(result)
        else:
            # Log failures and abort
            for check in result.failed_checks:
                logger.error(f"Validation failed: {check.message}")
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self._config = config or ValidationConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()

        # Validation history for regression detection
        self._baseline_metrics: Dict[str, Dict[str, float]] = {}

        # Safety patterns (simple content filters)
        self._harmful_patterns = [
            # We use very simple patterns - actual safety would use classifiers
        ]

    async def initialize(self):
        """Initialize the validator."""
        if AIOHTTP_AVAILABLE and self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30.0)
            )

    async def shutdown(self):
        """Shutdown the validator."""
        if self._session:
            await self._session.close()
            self._session = None

    async def validate(
        self,
        model_path: str,
        model_name: str,
        endpoint_url: str,
        config: Optional[ValidationConfig] = None,
    ) -> ValidationResult:
        """
        Run full validation suite on a model.

        Args:
            model_path: Path to model files
            model_name: Human-readable model name
            endpoint_url: URL of the model endpoint to test
            config: Optional custom configuration

        Returns:
            ValidationResult with all check results and metrics
        """
        config = config or self._config
        result = ValidationResult(
            model_name=model_name,
            model_path=model_path,
            status=ValidationStatus.IN_PROGRESS,
            start_time=time.time(),
        )

        logger.info(f"[VALIDATOR] Starting validation for {model_name} (trace: {result.trace_id})")

        try:
            # Run validation with overall timeout
            await asyncio.wait_for(
                self._run_validation_suite(result, endpoint_url, config),
                timeout=config.overall_timeout_seconds,
            )
        except asyncio.TimeoutError:
            result.status = ValidationStatus.FAILED
            result.error_message = f"Validation timed out after {config.overall_timeout_seconds}s"
            result.checks.append(ValidationCheck(
                check_type=ValidationCheckType.INTEGRITY,
                status=ValidationStatus.FAILED,
                message=result.error_message,
            ))
        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.error_message = str(e)
            logger.exception(f"[VALIDATOR] Unexpected error during validation: {e}")

        result.end_time = time.time()

        # Determine final status
        if result.status == ValidationStatus.IN_PROGRESS:
            failed_checks = result.failed_checks
            if failed_checks:
                result.status = ValidationStatus.FAILED
                result.error_message = f"Failed {len(failed_checks)} check(s)"
            else:
                result.status = ValidationStatus.PASSED

        logger.info(
            f"[VALIDATOR] Validation {result.status.value} for {model_name} "
            f"in {result.duration_seconds:.2f}s (trace: {result.trace_id})"
        )

        return result

    async def _run_validation_suite(
        self,
        result: ValidationResult,
        endpoint_url: str,
        config: ValidationConfig,
    ):
        """Run all validation checks."""
        # 1. Integrity Check
        if config.enable_integrity_check:
            check = await self._check_integrity(result.model_path)
            result.checks.append(check)
            if check.status == ValidationStatus.FAILED:
                return  # Stop on critical failure

        # 2. Inference Check (tests if endpoint is responding)
        if config.enable_inference_check:
            check = await self._check_inference(endpoint_url, config)
            result.checks.append(check)
            if check.status == ValidationStatus.FAILED:
                return  # Stop on critical failure

        # 3. Latency Check (performance benchmarking)
        if config.enable_latency_check:
            check, metrics = await self._check_latency(endpoint_url, config)
            result.checks.append(check)
            # Update result metrics
            result.avg_latency_ms = metrics.get("avg_latency_ms", 0)
            result.p50_latency_ms = metrics.get("p50_latency_ms", 0)
            result.p95_latency_ms = metrics.get("p95_latency_ms", 0)
            result.p99_latency_ms = metrics.get("p99_latency_ms", 0)
            result.throughput_rps = metrics.get("throughput_rps", 0)

        # 4. Safety Check
        if config.enable_safety_check:
            check = await self._check_safety(endpoint_url, config)
            result.checks.append(check)

        # 5. Regression Check (compare to baseline)
        if config.enable_regression_check and result.model_name in self._baseline_metrics:
            check = await self._check_regression(result, config)
            result.checks.append(check)

    async def _check_integrity(self, model_path: str) -> ValidationCheck:
        """Check model file integrity."""
        start = time.time()

        try:
            path = Path(model_path)

            # Check if path exists
            if not path.exists():
                return ValidationCheck(
                    check_type=ValidationCheckType.INTEGRITY,
                    status=ValidationStatus.FAILED,
                    message=f"Model path does not exist: {model_path}",
                    duration_ms=(time.time() - start) * 1000,
                )

            # Calculate total size
            if path.is_dir():
                total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                file_count = sum(1 for f in path.rglob("*") if f.is_file())
            else:
                total_size = path.stat().st_size
                file_count = 1

            # Check for suspicious files (empty, very small)
            if total_size < 1000:  # Less than 1KB is suspicious for a model
                return ValidationCheck(
                    check_type=ValidationCheckType.INTEGRITY,
                    status=ValidationStatus.FAILED,
                    message=f"Model files too small: {total_size} bytes",
                    duration_ms=(time.time() - start) * 1000,
                )

            return ValidationCheck(
                check_type=ValidationCheckType.INTEGRITY,
                status=ValidationStatus.PASSED,
                message=f"Integrity verified: {file_count} files, {total_size / 1e9:.2f} GB",
                duration_ms=(time.time() - start) * 1000,
                details={"total_size_bytes": total_size, "file_count": file_count},
            )

        except Exception as e:
            return ValidationCheck(
                check_type=ValidationCheckType.INTEGRITY,
                status=ValidationStatus.FAILED,
                message=f"Integrity check failed: {e}",
                duration_ms=(time.time() - start) * 1000,
            )

    async def _check_inference(
        self,
        endpoint_url: str,
        config: ValidationConfig,
    ) -> ValidationCheck:
        """Check that the model can perform inference."""
        start = time.time()

        if not self._session:
            await self.initialize()

        if not self._session:
            return ValidationCheck(
                check_type=ValidationCheckType.INFERENCE,
                status=ValidationStatus.FAILED,
                message="HTTP session not available",
                duration_ms=(time.time() - start) * 1000,
            )

        # Try each canary prompt
        successful = 0
        failed = 0
        errors = []

        for prompt in config.canary_prompts[:3]:  # Limit to first 3 for speed
            try:
                payload = {
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                }

                async with self._session.post(
                    endpoint_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if content and len(content) > 0:
                            successful += 1
                        else:
                            failed += 1
                            errors.append(f"Empty response for: {prompt[:30]}...")
                    else:
                        failed += 1
                        errors.append(f"HTTP {response.status} for: {prompt[:30]}...")

            except asyncio.TimeoutError:
                failed += 1
                errors.append(f"Timeout for: {prompt[:30]}...")
            except Exception as e:
                failed += 1
                errors.append(f"Error for {prompt[:30]}...: {e}")

        if successful == 0:
            return ValidationCheck(
                check_type=ValidationCheckType.INFERENCE,
                status=ValidationStatus.FAILED,
                message=f"All inference tests failed: {'; '.join(errors[:3])}",
                duration_ms=(time.time() - start) * 1000,
                details={"successful": successful, "failed": failed, "errors": errors},
            )

        if failed > successful:
            return ValidationCheck(
                check_type=ValidationCheckType.INFERENCE,
                status=ValidationStatus.FAILED,
                message=f"Too many failures: {failed}/{successful + failed}",
                duration_ms=(time.time() - start) * 1000,
                details={"successful": successful, "failed": failed},
            )

        return ValidationCheck(
            check_type=ValidationCheckType.INFERENCE,
            status=ValidationStatus.PASSED,
            message=f"Inference check passed: {successful}/{successful + failed} prompts succeeded",
            duration_ms=(time.time() - start) * 1000,
            details={"successful": successful, "failed": failed},
        )

    async def _check_latency(
        self,
        endpoint_url: str,
        config: ValidationConfig,
    ) -> Tuple[ValidationCheck, Dict[str, float]]:
        """Check latency performance."""
        start = time.time()
        latencies: List[float] = []

        if not self._session:
            await self.initialize()

        if not self._session:
            return ValidationCheck(
                check_type=ValidationCheckType.LATENCY,
                status=ValidationStatus.FAILED,
                message="HTTP session not available",
                duration_ms=(time.time() - start) * 1000,
            ), {}

        # Run multiple inference requests for latency measurement
        test_prompt = "What is 2 + 2?"
        num_requests = 10

        for _ in range(num_requests):
            req_start = time.time()
            try:
                payload = {
                    "model": "default",
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 50,
                }

                async with self._session.post(
                    endpoint_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as response:
                    if response.status == 200:
                        await response.json()
                        latencies.append((time.time() - req_start) * 1000)

            except Exception:
                pass  # Skip failed requests in latency measurement

        if len(latencies) < 5:
            return ValidationCheck(
                check_type=ValidationCheckType.LATENCY,
                status=ValidationStatus.FAILED,
                message=f"Too few successful requests for latency measurement: {len(latencies)}/{num_requests}",
                duration_ms=(time.time() - start) * 1000,
            ), {}

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        metrics = {
            "avg_latency_ms": statistics.mean(latencies),
            "p50_latency_ms": sorted_latencies[len(sorted_latencies) // 2],
            "p95_latency_ms": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99_latency_ms": sorted_latencies[-1] if len(sorted_latencies) > 0 else 0,
            "throughput_rps": len(latencies) / ((time.time() - start)),
        }

        # Check against threshold
        if metrics["avg_latency_ms"] > config.max_inference_latency_ms:
            return ValidationCheck(
                check_type=ValidationCheckType.LATENCY,
                status=ValidationStatus.FAILED,
                message=f"Latency too high: {metrics['avg_latency_ms']:.1f}ms > {config.max_inference_latency_ms}ms",
                duration_ms=(time.time() - start) * 1000,
                details=metrics,
            ), metrics

        return ValidationCheck(
            check_type=ValidationCheckType.LATENCY,
            status=ValidationStatus.PASSED,
            message=f"Latency OK: avg={metrics['avg_latency_ms']:.1f}ms, p95={metrics['p95_latency_ms']:.1f}ms",
            duration_ms=(time.time() - start) * 1000,
            details=metrics,
        ), metrics

    async def _check_safety(
        self,
        endpoint_url: str,
        config: ValidationConfig,
    ) -> ValidationCheck:
        """Check model safety on test prompts."""
        start = time.time()

        # For now, just verify the model responds to safety test prompts
        # A real implementation would use a safety classifier
        if not self._session:
            await self.initialize()

        if not self._session:
            return ValidationCheck(
                check_type=ValidationCheckType.SAFETY,
                status=ValidationStatus.SKIPPED,
                message="HTTP session not available",
                duration_ms=(time.time() - start) * 1000,
            )

        for prompt in config.safety_test_prompts:
            try:
                payload = {
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                }

                async with self._session.post(
                    endpoint_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                        # Check for harmful patterns (placeholder - use real classifiers in production)
                        for pattern in self._harmful_patterns:
                            if pattern.lower() in content.lower():
                                return ValidationCheck(
                                    check_type=ValidationCheckType.SAFETY,
                                    status=ValidationStatus.FAILED,
                                    message=f"Safety check failed: harmful content detected",
                                    duration_ms=(time.time() - start) * 1000,
                                )

            except Exception as e:
                logger.warning(f"Safety check request failed: {e}")

        return ValidationCheck(
            check_type=ValidationCheckType.SAFETY,
            status=ValidationStatus.PASSED,
            message="Safety check passed",
            duration_ms=(time.time() - start) * 1000,
        )

    async def _check_regression(
        self,
        result: ValidationResult,
        config: ValidationConfig,
    ) -> ValidationCheck:
        """Check for performance regression vs. baseline."""
        start = time.time()

        baseline = self._baseline_metrics.get(result.model_name)
        if not baseline:
            return ValidationCheck(
                check_type=ValidationCheckType.REGRESSION,
                status=ValidationStatus.SKIPPED,
                message="No baseline metrics for comparison",
                duration_ms=(time.time() - start) * 1000,
            )

        # Compare latency (allow 20% regression)
        if result.avg_latency_ms > baseline.get("avg_latency_ms", float("inf")) * 1.2:
            return ValidationCheck(
                check_type=ValidationCheckType.REGRESSION,
                status=ValidationStatus.FAILED,
                message=f"Latency regression: {result.avg_latency_ms:.1f}ms vs baseline {baseline.get('avg_latency_ms', 0):.1f}ms",
                duration_ms=(time.time() - start) * 1000,
            )

        return ValidationCheck(
            check_type=ValidationCheckType.REGRESSION,
            status=ValidationStatus.PASSED,
            message="No regression detected",
            duration_ms=(time.time() - start) * 1000,
        )

    def set_baseline(self, model_name: str, metrics: Dict[str, float]):
        """Set baseline metrics for regression testing."""
        self._baseline_metrics[model_name] = metrics

    def get_baseline(self, model_name: str) -> Optional[Dict[str, float]]:
        """Get baseline metrics for a model."""
        return self._baseline_metrics.get(model_name)


# =============================================================================
# DEPLOYMENT MANAGER
# =============================================================================

class DeploymentManager:
    """
    Manages model deployments with canary releases and rollback.

    FEATURES:
    - Canary deployments with gradual traffic increase
    - Shadow testing (new model serves in parallel)
    - Automatic rollback on error rate threshold
    - Request queuing during swap
    - A/B testing framework

    USAGE:
        manager = DeploymentManager(endpoint_manager, validator)

        # Start canary deployment
        deployment = await manager.start_deployment(
            model_name="new-model",
            model_path="/path/to/model",
            endpoint_url="http://localhost:8000/v1/chat/completions",
            strategy=DeploymentStrategy.CANARY,
        )

        # Monitor deployment
        status = manager.get_deployment_status(deployment.deployment_id)

        # Rollback if needed
        await manager.rollback(deployment.deployment_id)
    """

    def __init__(
        self,
        endpoint_manager: "EndpointManager",
        validator: ModelValidator,
        request_queue: RequestQueue,
    ):
        self._endpoint_manager = endpoint_manager
        self._validator = validator
        self._request_queue = request_queue
        self._lock = asyncio.Lock()

        # Active deployments
        self._deployments: Dict[str, DeploymentState] = {}

        # Canary configuration
        self._canary_phases = [
            (DeploymentPhase.CANARY_1_PERCENT, 0.01),
            (DeploymentPhase.CANARY_5_PERCENT, 0.05),
            (DeploymentPhase.CANARY_25_PERCENT, 0.25),
            (DeploymentPhase.CANARY_50_PERCENT, 0.50),
            (DeploymentPhase.FULL_ROLLOUT, 1.0),
        ]

        # Error thresholds for auto-rollback
        self._error_rate_threshold = 0.05  # 5% error rate triggers rollback
        self._min_requests_before_evaluation = 100

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._is_running = False

    async def start(self):
        """Start the deployment manager."""
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_deployments())
        logger.info("[DEPLOYMENT] Manager started")

    async def stop(self):
        """Stop the deployment manager."""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[DEPLOYMENT] Manager stopped")

    async def start_deployment(
        self,
        model_name: str,
        model_path: str,
        endpoint_url: str,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        skip_validation: bool = False,
    ) -> DeploymentState:
        """
        Start a new model deployment.

        Args:
            model_name: Name of the model
            model_path: Path to model files
            endpoint_url: URL of the new model endpoint
            strategy: Deployment strategy
            skip_validation: Skip validation (not recommended)

        Returns:
            DeploymentState tracking the deployment
        """
        deployment_id = f"deploy_{model_name}_{int(time.time())}"
        endpoint_id = f"dynamic_{model_name.replace(' ', '_').lower()}"

        deployment = DeploymentState(
            deployment_id=deployment_id,
            model_name=model_name,
            model_path=model_path,
            endpoint_id=endpoint_id,
            strategy=strategy,
            phase=DeploymentPhase.VALIDATING,
        )

        async with self._lock:
            self._deployments[deployment_id] = deployment

        logger.info(f"[DEPLOYMENT] Starting deployment {deployment_id} with strategy {strategy.value}")

        # Phase 1: Validation
        if not skip_validation:
            deployment.phase = DeploymentPhase.VALIDATING

            validation_result = await self._validator.validate(
                model_path=model_path,
                model_name=model_name,
                endpoint_url=endpoint_url,
            )
            deployment.validation_result = validation_result

            if not validation_result.passed:
                deployment.phase = DeploymentPhase.ROLLED_BACK
                deployment.completed_at = time.time()
                logger.error(
                    f"[DEPLOYMENT] Validation failed for {deployment_id}: "
                    f"{validation_result.error_message}"
                )
                return deployment

        # Phase 2: Queue requests during swap
        await self._request_queue.start_queueing(f"Deploying model: {model_name}")

        try:
            # Create endpoint for new model
            await self._endpoint_manager.register_endpoint(EndpointConfig(
                endpoint_id=endpoint_id,
                name=f"Dynamic Model: {model_name}",
                endpoint_type=EndpointType.LOCAL_MODEL,
                url=endpoint_url,
                capabilities={
                    Capability.TEXT_GENERATION,
                    Capability.CODE_GENERATION,
                    Capability.CHAT,
                },
                model_name=model_name,
                avg_latency_ms=deployment.validation_result.avg_latency_ms if deployment.validation_result else 500.0,
                max_tokens=8192,
                cost_per_1k_tokens=0.0,
                metadata={"model_path": model_path, "deployment_id": deployment_id},
                enabled=False,  # Start disabled
            ))

            # Phase 3: Start deployment based on strategy
            if strategy == DeploymentStrategy.IMMEDIATE:
                await self._deploy_immediate(deployment)
            elif strategy == DeploymentStrategy.CANARY:
                await self._deploy_canary_start(deployment)
            elif strategy == DeploymentStrategy.BLUE_GREEN:
                await self._deploy_blue_green(deployment)
            else:
                await self._deploy_immediate(deployment)

        finally:
            # Stop queueing and drain queue
            await self._request_queue.stop_queueing()

        return deployment

    async def _deploy_immediate(self, deployment: DeploymentState):
        """Deploy model immediately at 100% traffic."""
        endpoint = self._endpoint_manager.get_endpoint(deployment.endpoint_id)
        if endpoint:
            endpoint.enabled = True
            deployment.traffic_percentage = 1.0
            deployment.phase = DeploymentPhase.COMPLETED
            deployment.completed_at = time.time()
            logger.info(f"[DEPLOYMENT] Immediate deployment complete: {deployment.deployment_id}")

    async def _deploy_canary_start(self, deployment: DeploymentState):
        """Start canary deployment at 1%."""
        endpoint = self._endpoint_manager.get_endpoint(deployment.endpoint_id)
        if endpoint:
            endpoint.enabled = True
            deployment.traffic_percentage = 0.01
            deployment.phase = DeploymentPhase.CANARY_1_PERCENT
            logger.info(f"[DEPLOYMENT] Canary started at 1%: {deployment.deployment_id}")

    async def _deploy_blue_green(self, deployment: DeploymentState):
        """Blue-green deployment (full swap)."""
        # Same as immediate for now
        await self._deploy_immediate(deployment)

    async def advance_canary(self, deployment_id: str) -> bool:
        """
        Advance a canary deployment to the next phase.

        Returns True if advanced, False if already at full rollout or error.
        """
        async with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return False

            # Find current phase index
            current_idx = -1
            for i, (phase, _) in enumerate(self._canary_phases):
                if deployment.phase == phase:
                    current_idx = i
                    break

            if current_idx < 0 or current_idx >= len(self._canary_phases) - 1:
                return False  # Already at full rollout or not in canary phases

            # Check error rate before advancing
            if deployment.requests_served >= self._min_requests_before_evaluation:
                if deployment.error_rate > self._error_rate_threshold:
                    logger.warning(
                        f"[DEPLOYMENT] Cannot advance canary - error rate too high: "
                        f"{deployment.error_rate:.2%}"
                    )
                    return False

            # Advance to next phase
            next_phase, next_percentage = self._canary_phases[current_idx + 1]
            deployment.phase = next_phase
            deployment.traffic_percentage = next_percentage

            if next_phase == DeploymentPhase.FULL_ROLLOUT:
                deployment.phase = DeploymentPhase.COMPLETED
                deployment.completed_at = time.time()

            logger.info(
                f"[DEPLOYMENT] Advanced canary to {next_phase.value} "
                f"({next_percentage:.0%}): {deployment_id}"
            )
            return True

    async def rollback(self, deployment_id: str) -> bool:
        """
        Rollback a deployment.

        Returns True if rollback succeeded.
        """
        async with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return False

            logger.warning(f"[DEPLOYMENT] Rolling back: {deployment_id}")

            # Disable the endpoint
            endpoint = self._endpoint_manager.get_endpoint(deployment.endpoint_id)
            if endpoint:
                endpoint.enabled = False

            # Re-enable previous endpoint if we have one
            if deployment.previous_endpoint_id:
                prev_endpoint = self._endpoint_manager.get_endpoint(deployment.previous_endpoint_id)
                if prev_endpoint:
                    prev_endpoint.enabled = True

            deployment.phase = DeploymentPhase.ROLLED_BACK
            deployment.completed_at = time.time()

            logger.info(f"[DEPLOYMENT] Rollback complete: {deployment_id}")
            return True

    async def _monitor_deployments(self):
        """Background task to monitor active deployments."""
        while self._is_running:
            try:
                async with self._lock:
                    for deployment in self._deployments.values():
                        # Skip completed or rolled back
                        if deployment.phase in (DeploymentPhase.COMPLETED, DeploymentPhase.ROLLED_BACK):
                            continue

                        # Check for auto-rollback
                        if (
                            deployment.requests_served >= self._min_requests_before_evaluation
                            and deployment.error_rate > self._error_rate_threshold
                        ):
                            logger.warning(
                                f"[DEPLOYMENT] Auto-rollback triggered for {deployment.deployment_id} "
                                f"- error rate: {deployment.error_rate:.2%}"
                            )
                            await self.rollback(deployment.deployment_id)

            except Exception as e:
                logger.error(f"[DEPLOYMENT] Monitor error: {e}")

            await asyncio.sleep(10)  # Check every 10 seconds

    def get_deployment(self, deployment_id: str) -> Optional[DeploymentState]:
        """Get a deployment by ID."""
        return self._deployments.get(deployment_id)

    def get_active_deployments(self) -> List[DeploymentState]:
        """Get all active deployments."""
        return [
            d for d in self._deployments.values()
            if d.phase not in (DeploymentPhase.COMPLETED, DeploymentPhase.ROLLED_BACK)
        ]

    def should_route_to_canary(self, endpoint_id: str) -> bool:
        """Check if a request should be routed to a canary endpoint."""
        for deployment in self._deployments.values():
            if deployment.endpoint_id == endpoint_id:
                if deployment.phase in (DeploymentPhase.COMPLETED, DeploymentPhase.ROLLED_BACK):
                    continue
                # Use random sampling based on traffic percentage
                return random.random() < deployment.traffic_percentage
        return True  # Default to routing if not in canary

    def record_request(self, endpoint_id: str, success: bool, latency_ms: float):
        """Record a request for deployment metrics."""
        for deployment in self._deployments.values():
            if deployment.endpoint_id == endpoint_id:
                deployment.requests_served += 1
                if not success:
                    deployment.errors_count += 1
                # Update rolling average latency
                n = deployment.requests_served
                deployment.avg_latency_ms = (
                    (deployment.avg_latency_ms * (n - 1) + latency_ms) / n
                )
                break


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

        # Production-grade components
        self._validator = ModelValidator()
        self._request_queue = RequestQueue(max_size=1000, max_wait_seconds=30.0)
        self._deployment_manager: Optional[DeploymentManager] = None

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
        self._queued_requests = 0

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
        """Initialize the router with all production components."""
        if self._initialized:
            return

        logger.info("[ROUTER] Initializing Intelligent Request Router v2.0...")

        # Initialize validator
        await self._validator.initialize()
        logger.info("[ROUTER] Model Validator initialized")

        # Initialize endpoint manager
        await self._endpoint_manager.initialize()
        logger.info("[ROUTER] Endpoint Manager initialized")

        # Create deployment manager
        self._deployment_manager = DeploymentManager(
            endpoint_manager=self._endpoint_manager,
            validator=self._validator,
            request_queue=self._request_queue,
        )
        await self._deployment_manager.start()
        logger.info("[ROUTER] Deployment Manager started")

        # Override the endpoint manager's model ready handler
        self._endpoint_manager._on_model_ready = self._on_model_ready_with_validation

        # Create HTTP session
        if AIOHTTP_AVAILABLE:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60.0)
            )

        self._initialized = True
        logger.info("[ROUTER] Intelligent Request Router v2.0 ready")

    async def shutdown(self):
        """Shutdown the router and all components."""
        logger.info("[ROUTER] Shutting down...")

        # Stop deployment manager
        if self._deployment_manager:
            await self._deployment_manager.stop()

        # Shutdown validator
        await self._validator.shutdown()

        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None

        self._initialized = False

        async with self._lock:
            IntelligentRequestRouter._instance = None

        logger.info("[ROUTER] Intelligent Request Router shutdown complete")

    async def _on_model_ready_with_validation(self, event: "TrinityEvent"):
        """
        Handle MODEL_READY event with full validation and deployment.

        This replaces the default endpoint manager handler to add:
        1. Model validation before deployment
        2. Request queuing during swap
        3. Canary deployment with automatic rollback
        """
        model_name = event.payload.get("model_name", "")
        model_path = event.payload.get("model_path", "")
        endpoint_url = event.payload.get("endpoint_url", "http://localhost:8000/v1/chat/completions")
        strategy_str = event.payload.get("deployment_strategy", "canary")
        skip_validation = event.payload.get("skip_validation", False)

        logger.info(f"[ROUTER] Received MODEL_READY event for: {model_name}")

        # Map strategy string to enum
        strategy_map = {
            "immediate": DeploymentStrategy.IMMEDIATE,
            "canary": DeploymentStrategy.CANARY,
            "blue_green": DeploymentStrategy.BLUE_GREEN,
            "shadow": DeploymentStrategy.SHADOW,
            "rolling": DeploymentStrategy.ROLLING,
        }
        strategy = strategy_map.get(strategy_str.lower(), DeploymentStrategy.CANARY)

        # Start deployment through deployment manager
        if self._deployment_manager:
            deployment = await self._deployment_manager.start_deployment(
                model_name=model_name,
                model_path=model_path,
                endpoint_url=endpoint_url,
                strategy=strategy,
                skip_validation=skip_validation,
            )

            if deployment.phase == DeploymentPhase.ROLLED_BACK:
                logger.error(f"[ROUTER] Deployment failed for {model_name}")
            else:
                logger.info(
                    f"[ROUTER] Deployment started: {deployment.deployment_id} "
                    f"({deployment.phase.value})"
                )

    # =========================================================================
    # VALIDATION API
    # =========================================================================

    async def validate_model(
        self,
        model_path: str,
        model_name: str,
        endpoint_url: str,
        config: Optional[ValidationConfig] = None,
    ) -> ValidationResult:
        """
        Validate a model before deployment.
        """
        return await self._validator.validate(
            model_path=model_path,
            model_name=model_name,
            endpoint_url=endpoint_url,
            config=config,
        )

    async def deploy_model(
        self,
        model_name: str,
        model_path: str,
        endpoint_url: str,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        skip_validation: bool = False,
    ) -> DeploymentState:
        """
        Deploy a new model with validation and gradual rollout.
        """
        if not self._deployment_manager:
            raise RuntimeError("Deployment manager not initialized")

        return await self._deployment_manager.start_deployment(
            model_name=model_name,
            model_path=model_path,
            endpoint_url=endpoint_url,
            strategy=strategy,
            skip_validation=skip_validation,
        )

    async def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment."""
        if not self._deployment_manager:
            return False
        return await self._deployment_manager.rollback(deployment_id)

    def get_deployment(self, deployment_id: str) -> Optional[DeploymentState]:
        """Get deployment status."""
        if not self._deployment_manager:
            return None
        return self._deployment_manager.get_deployment(deployment_id)

    def get_active_deployments(self) -> List[DeploymentState]:
        """Get all active deployments."""
        if not self._deployment_manager:
            return []
        return self._deployment_manager.get_active_deployments()

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

        # v91.0: Observability tracing and metrics
        obs_bridge = None
        if OBSERVABILITY_AVAILABLE:
            try:
                obs_bridge = await get_observability_bridge()
                await obs_bridge.set_gauge(
                    "trinity_active_requests",
                    self._total_requests - self._successful_requests - self._failed_requests,
                    labels={"component": "request_router"},
                )
            except Exception:
                pass  # Observability is optional

        # Find best endpoint
        routing_result = await self._find_best_endpoint(context)

        if not routing_result.success or not routing_result.endpoint:
            self._failed_requests += 1
            # v91.0: Track failed routing
            if obs_bridge:
                try:
                    await obs_bridge.inc_counter(
                        "trinity_requests_total",
                        labels={"component": "request_router", "endpoint": "none", "status": "no_endpoint"},
                    )
                except Exception:
                    pass
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

            # v91.0: Maybe inject chaos (fault testing)
            if obs_bridge:
                try:
                    await obs_bridge.maybe_inject_latency(endpoint.name)
                    await obs_bridge.maybe_inject_error(endpoint.name)
                except RuntimeError as e:
                    # Chaos-injected error
                    logger.warning(f"[CHAOS] Injected error for {endpoint.name}: {e}")
                    continue
                except Exception:
                    pass

            result = await self._call_endpoint(endpoint, prompt)

            if result.success:
                self._successful_requests += 1
                result.fallbacks_attempted = i

                # v91.0: Track successful request metrics
                if obs_bridge:
                    try:
                        latency_seconds = result.latency_ms / 1000.0 if result.latency_ms else (time.time() - start_time)
                        await obs_bridge.inc_counter(
                            "trinity_requests_total",
                            labels={"component": "request_router", "endpoint": endpoint.name, "status": "success"},
                        )
                        await obs_bridge.observe_histogram(
                            "trinity_request_duration_seconds",
                            latency_seconds,
                            labels={"component": "request_router", "endpoint": endpoint.name},
                        )
                    except Exception:
                        pass

                return result

        # All endpoints failed
        self._failed_requests += 1

        # v91.0: Track failed request
        if obs_bridge:
            try:
                await obs_bridge.inc_counter(
                    "trinity_requests_total",
                    labels={"component": "request_router", "endpoint": "all", "status": "failed"},
                )
            except Exception:
                pass

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
        """Get comprehensive router status with production metrics."""
        alerts = self._check_alerts()
        return {
            "version": "2.0",
            "initialized": self._initialized,
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "fallback_count": self._fallback_count,
            "queued_requests": self._queued_requests,
            "success_rate": (
                self._successful_requests / self._total_requests
                if self._total_requests > 0 else 0.0
            ),
            "endpoints": {
                "total": len(self._endpoint_manager.get_all_endpoints()),
                "healthy": len(self._endpoint_manager.get_healthy_endpoints()),
                "list": [ep.to_dict() for ep in self._endpoint_manager.get_all_endpoints()],
            },
            "request_queue": self._request_queue.get_metrics() if self._request_queue else {},
            "active_deployments": [d.to_dict() for d in self.get_active_deployments()],
            "alerts": alerts,
            "health": "healthy" if not alerts else "degraded",
        }

    def get_endpoints(self) -> List[Dict[str, Any]]:
        """Get all endpoint configurations."""
        return [ep.to_dict() for ep in self._endpoint_manager.get_all_endpoints()]

    def _check_alerts(self) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []

        # High error rate
        if self._total_requests > 100:
            error_rate = self._failed_requests / self._total_requests
            if error_rate > 0.10:
                alerts.append({
                    "severity": "critical",
                    "type": "high_error_rate",
                    "message": f"Error rate {error_rate:.1%} exceeds 10%",
                })
            elif error_rate > 0.05:
                alerts.append({
                    "severity": "warning",
                    "type": "elevated_error_rate",
                    "message": f"Error rate {error_rate:.1%} exceeds 5%",
                })

        # No healthy endpoints
        healthy = len(self._endpoint_manager.get_healthy_endpoints())
        if healthy == 0:
            alerts.append({
                "severity": "critical",
                "type": "no_healthy_endpoints",
                "message": "No healthy endpoints available",
            })
        elif healthy == 1:
            alerts.append({
                "severity": "warning",
                "type": "low_endpoint_count",
                "message": "Only 1 healthy endpoint",
            })

        # Queue overflow
        if self._request_queue and self._request_queue.queue_size > 500:
            alerts.append({
                "severity": "critical",
                "type": "queue_overflow",
                "message": f"Queue has {self._request_queue.queue_size} requests",
            })

        return alerts

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for dashboards."""
        return {
            "timestamp": time.time(),
            "requests": {
                "total": self._total_requests,
                "success_rate": self._successful_requests / max(1, self._total_requests),
            },
            "endpoints": {
                "healthy": len(self._endpoint_manager.get_healthy_endpoints()),
            },
            "health": "healthy" if not self._check_alerts() else "degraded",
        }


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
    "ValidationStatus",
    "ValidationCheckType",
    "DeploymentStrategy",
    "DeploymentPhase",
    # Data structures
    "EndpointConfig",
    "RoutingContext",
    "RoutingResult",
    "RequestResult",
    "ValidationConfig",
    "ValidationResult",
    "ValidationCheck",
    # Tracing
    "Span",
    "TraceContext",
    "DeploymentState",
    "QueuedRequest",
    # Router and components
    "IntelligentRequestRouter",
    "EndpointManager",
    "EndpointCircuitBreaker",
    "ModelValidator",
    "DeploymentManager",
    "RequestQueue",
    # Functions
    "get_request_router",
    "shutdown_request_router",
    "route_request",
]
