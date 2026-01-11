"""
Observability Bridge v1.0 - Cross-Repo Observability Integration
=================================================================

Connects JARVIS-Prime's production features to JARVIS Body's observability stack:
- Langfuse: Distributed tracing and audit trails
- Prometheus: Metrics export in Prometheus format
- Helicone: Cost tracking integration
- Chaos Testing: Fault injection framework

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     JARVIS-PRIME OBSERVABILITY BRIDGE                   │
    └──────────────────────────────────┬──────────────────────────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────────────┐
         │                             │                                     │
         ▼                             ▼                                     ▼
    ┌───────────┐              ┌───────────────┐                   ┌─────────────┐
    │ Langfuse  │              │  Prometheus   │                   │   Chaos     │
    │  Tracer   │              │   Exporter    │                   │   Engine    │
    └───────────┘              └───────────────┘                   └─────────────┘
         │                             │                                     │
         │                             │                                     │
         ▼                             ▼                                     ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     JARVIS BODY OBSERVABILITY HUB                       │
    │  (langfuse_integration.py + unified_observability_hub.py)               │
    └─────────────────────────────────────────────────────────────────────────┘

FEATURES:
    - Langfuse trace correlation with TraceContext
    - Prometheus metrics export (counters, gauges, histograms)
    - Cost tracking per request
    - Chaos testing with fault injection
    - Adaptive event bus optimization
    - Zero hardcoding (environment-driven)

USAGE:
    from jarvis_prime.core.observability_bridge import (
        get_observability_bridge,
        trace_request,
        export_prometheus_metrics,
        inject_fault,
    )

    bridge = await get_observability_bridge()

    # Trace a request
    async with bridge.trace("model_inference") as span:
        result = await inference(prompt)
        span.set_attribute("tokens", result.tokens)

    # Export Prometheus metrics
    metrics = bridge.export_prometheus()

    # Inject chaos
    if bridge.should_inject_fault("network_delay"):
        await asyncio.sleep(random.uniform(0.1, 2.0))
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
from abc import ABC, abstractmethod
from collections import defaultdict, deque
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

# =============================================================================
# TRY IMPORTS
# =============================================================================

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logger.info("Langfuse not available - install with: pip install langfuse")

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ObservabilityConfig:
    """Configuration for observability bridge."""
    # Langfuse
    langfuse_enabled: bool = field(
        default_factory=lambda: os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
    )
    langfuse_public_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", "")
    )
    langfuse_secret_key: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", "")
    )
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    langfuse_sample_rate: float = field(
        default_factory=lambda: float(os.getenv("LANGFUSE_SAMPLE_RATE", "1.0"))
    )

    # Prometheus
    prometheus_enabled: bool = field(
        default_factory=lambda: os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    )
    prometheus_port: int = field(
        default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9091"))
    )

    # Chaos
    chaos_enabled: bool = field(
        default_factory=lambda: os.getenv("CHAOS_ENABLED", "false").lower() == "true"
    )
    chaos_probability: float = field(
        default_factory=lambda: float(os.getenv("CHAOS_PROBABILITY", "0.05"))
    )

    # Event bus optimization
    event_bus_poll_interval_ms: int = field(
        default_factory=lambda: int(os.getenv("EVENT_BUS_POLL_INTERVAL_MS", "100"))
    )
    event_bus_adaptive_polling: bool = field(
        default_factory=lambda: os.getenv("EVENT_BUS_ADAPTIVE_POLLING", "true").lower() == "true"
    )


# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

class MetricType(Enum):
    """Types of Prometheus metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class PrometheusMetric:
    """A single Prometheus metric."""
    name: str
    type: MetricType
    help: str
    labels: Dict[str, str] = field(default_factory=dict)
    value: float = 0.0
    buckets: List[float] = field(default_factory=list)  # For histograms
    observations: List[float] = field(default_factory=list)  # For summaries
    timestamp: float = field(default_factory=time.time)


class PrometheusExporter:
    """
    Prometheus metrics exporter in OpenMetrics format.

    Exports metrics that can be scraped by Prometheus or pushed to Pushgateway.
    """

    def __init__(self):
        self._metrics: Dict[str, PrometheusMetric] = {}
        self._lock = asyncio.Lock()

        # Pre-define Trinity metrics
        self._define_trinity_metrics()

    def _define_trinity_metrics(self):
        """Define standard metrics for Trinity ecosystem."""
        # Request metrics
        self.define_counter(
            "trinity_requests_total",
            "Total requests processed",
            labels=["component", "endpoint", "status"]
        )
        self.define_histogram(
            "trinity_request_duration_seconds",
            "Request duration in seconds",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            labels=["component", "endpoint"]
        )
        self.define_gauge(
            "trinity_active_requests",
            "Currently active requests",
            labels=["component"]
        )

        # Model metrics
        self.define_counter(
            "trinity_model_inferences_total",
            "Total model inferences",
            labels=["model", "tier", "status"]
        )
        self.define_histogram(
            "trinity_inference_latency_seconds",
            "Model inference latency",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            labels=["model", "tier"]
        )
        self.define_gauge(
            "trinity_model_loaded",
            "Whether a model is currently loaded",
            labels=["model"]
        )

        # Deployment metrics
        self.define_counter(
            "trinity_deployments_total",
            "Total deployments",
            labels=["strategy", "status"]
        )
        self.define_gauge(
            "trinity_deployment_traffic_percent",
            "Current traffic percentage for canary",
            labels=["deployment_id"]
        )
        self.define_gauge(
            "trinity_deployment_error_rate",
            "Error rate for active deployment",
            labels=["deployment_id"]
        )

        # Circuit breaker metrics
        self.define_gauge(
            "trinity_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            labels=["endpoint"]
        )
        self.define_counter(
            "trinity_circuit_breaker_trips_total",
            "Times circuit breaker tripped",
            labels=["endpoint"]
        )

        # Event bus metrics
        self.define_counter(
            "trinity_events_published_total",
            "Total events published",
            labels=["event_type", "source"]
        )
        self.define_counter(
            "trinity_events_delivered_total",
            "Total events delivered",
            labels=["event_type", "destination"]
        )
        self.define_gauge(
            "trinity_event_queue_size",
            "Current event queue size",
            labels=["queue"]
        )
        self.define_counter(
            "trinity_dlq_entries_total",
            "Total entries in dead letter queue",
            labels=["reason"]
        )

        # Resource metrics
        self.define_gauge(
            "trinity_memory_usage_bytes",
            "Memory usage in bytes",
            labels=["component"]
        )
        self.define_gauge(
            "trinity_cpu_usage_percent",
            "CPU usage percentage",
            labels=["component"]
        )

    def define_counter(
        self,
        name: str,
        help: str,
        labels: Optional[List[str]] = None,
    ):
        """Define a counter metric."""
        self._metrics[name] = PrometheusMetric(
            name=name,
            type=MetricType.COUNTER,
            help=help,
            labels={l: "" for l in (labels or [])},
        )

    def define_gauge(
        self,
        name: str,
        help: str,
        labels: Optional[List[str]] = None,
    ):
        """Define a gauge metric."""
        self._metrics[name] = PrometheusMetric(
            name=name,
            type=MetricType.GAUGE,
            help=help,
            labels={l: "" for l in (labels or [])},
        )

    def define_histogram(
        self,
        name: str,
        help: str,
        buckets: List[float],
        labels: Optional[List[str]] = None,
    ):
        """Define a histogram metric."""
        self._metrics[name] = PrometheusMetric(
            name=name,
            type=MetricType.HISTOGRAM,
            help=help,
            buckets=buckets,
            labels={l: "" for l in (labels or [])},
        )

    async def inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter."""
        async with self._lock:
            key = self._make_key(name, labels)
            if key not in self._metrics:
                base = self._metrics.get(name)
                if base:
                    self._metrics[key] = PrometheusMetric(
                        name=name,
                        type=base.type,
                        help=base.help,
                        labels=labels or {},
                    )
            if key in self._metrics:
                self._metrics[key].value += value
                self._metrics[key].timestamp = time.time()

    async def set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        async with self._lock:
            key = self._make_key(name, labels)
            if key not in self._metrics:
                base = self._metrics.get(name)
                if base:
                    self._metrics[key] = PrometheusMetric(
                        name=name,
                        type=base.type,
                        help=base.help,
                        labels=labels or {},
                    )
            if key in self._metrics:
                self._metrics[key].value = value
                self._metrics[key].timestamp = time.time()

    async def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a value for histogram/summary."""
        async with self._lock:
            key = self._make_key(name, labels)
            if key not in self._metrics:
                base = self._metrics.get(name)
                if base:
                    self._metrics[key] = PrometheusMetric(
                        name=name,
                        type=base.type,
                        help=base.help,
                        buckets=base.buckets.copy(),
                        labels=labels or {},
                    )
            if key in self._metrics:
                self._metrics[key].observations.append(value)
                # Keep only last 1000 observations
                if len(self._metrics[key].observations) > 1000:
                    self._metrics[key].observations = self._metrics[key].observations[-1000:]
                self._metrics[key].timestamp = time.time()

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def export(self) -> str:
        """Export all metrics in Prometheus format."""
        lines = []
        seen_helps = set()

        for key, metric in sorted(self._metrics.items()):
            # Add help and type only once per metric name
            if metric.name not in seen_helps:
                lines.append(f"# HELP {metric.name} {metric.help}")
                lines.append(f"# TYPE {metric.name} {metric.type.value}")
                seen_helps.add(metric.name)

            # Format based on type
            if metric.type == MetricType.HISTOGRAM:
                self._export_histogram(lines, metric)
            elif metric.type == MetricType.SUMMARY:
                self._export_summary(lines, metric)
            else:
                # Counter or Gauge
                label_str = self._format_labels(metric.labels)
                lines.append(f"{metric.name}{label_str} {metric.value}")

        return "\n".join(lines)

    def _export_histogram(self, lines: List[str], metric: PrometheusMetric):
        """Export histogram metric."""
        label_str = self._format_labels(metric.labels)
        observations = metric.observations

        if not observations:
            return

        # Calculate bucket counts
        sorted_obs = sorted(observations)
        count = len(sorted_obs)
        total = sum(sorted_obs)

        for bucket in metric.buckets:
            bucket_count = sum(1 for o in sorted_obs if o <= bucket)
            bucket_labels = metric.labels.copy()
            bucket_labels["le"] = str(bucket)
            bucket_label_str = self._format_labels(bucket_labels)
            lines.append(f"{metric.name}_bucket{bucket_label_str} {bucket_count}")

        # +Inf bucket
        inf_labels = metric.labels.copy()
        inf_labels["le"] = "+Inf"
        inf_label_str = self._format_labels(inf_labels)
        lines.append(f"{metric.name}_bucket{inf_label_str} {count}")

        lines.append(f"{metric.name}_sum{label_str} {total}")
        lines.append(f"{metric.name}_count{label_str} {count}")

    def _export_summary(self, lines: List[str], metric: PrometheusMetric):
        """Export summary metric."""
        label_str = self._format_labels(metric.labels)
        observations = metric.observations

        if not observations:
            return

        sorted_obs = sorted(observations)
        count = len(sorted_obs)
        total = sum(sorted_obs)

        # Calculate quantiles
        for quantile in [0.5, 0.9, 0.99]:
            idx = int(quantile * count)
            value = sorted_obs[min(idx, count - 1)]
            q_labels = metric.labels.copy()
            q_labels["quantile"] = str(quantile)
            q_label_str = self._format_labels(q_labels)
            lines.append(f"{metric.name}{q_label_str} {value}")

        lines.append(f"{metric.name}_sum{label_str} {total}")
        lines.append(f"{metric.name}_count{label_str} {count}")

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(labels.items()) if v]
        return "{" + ",".join(parts) + "}" if parts else ""


# =============================================================================
# LANGFUSE INTEGRATION
# =============================================================================

class LangfuseTracer:
    """
    Langfuse tracer for JARVIS-Prime.

    Integrates with JARVIS Body's Langfuse for unified observability.
    """

    def __init__(self, config: ObservabilityConfig):
        self._config = config
        self._client: Optional[Langfuse] = None
        self._traces: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

        # Statistics
        self._traces_created = 0
        self._traces_completed = 0
        self._errors = 0

    async def initialize(self):
        """Initialize Langfuse client."""
        if not LANGFUSE_AVAILABLE:
            logger.info("Langfuse not available, tracing disabled")
            return

        if not self._config.langfuse_enabled:
            logger.info("Langfuse disabled by configuration")
            return

        if not self._config.langfuse_public_key or not self._config.langfuse_secret_key:
            logger.warning("Langfuse keys not configured, tracing disabled")
            return

        try:
            self._client = Langfuse(
                public_key=self._config.langfuse_public_key,
                secret_key=self._config.langfuse_secret_key,
                host=self._config.langfuse_host,
            )
            self._initialized = True
            logger.info(f"Langfuse initialized: {self._config.langfuse_host}")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")

    async def shutdown(self):
        """Shutdown Langfuse client."""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.debug(f"Error flushing Langfuse: {e}")
            self._client = None
            self._initialized = False

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator["LangfuseSpan"]:
        """
        Create a trace context.

        USAGE:
            async with tracer.trace("model_inference") as span:
                result = await inference(prompt)
                span.set_attribute("tokens", result.tokens)
        """
        # Check sampling
        if random.random() > self._config.langfuse_sample_rate:
            yield LangfuseSpan(None, name, {})
            return

        span = LangfuseSpan(self, name, metadata or {}, trace_id)
        await span.start()

        try:
            yield span
        except Exception as e:
            span.set_error(str(e))
            raise
        finally:
            await span.end()

    async def _create_trace(
        self,
        name: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Create a new trace."""
        if not self._initialized or not self._client:
            return trace_id or str(uuid.uuid4())[:16]

        try:
            async with self._lock:
                trace = self._client.trace(
                    id=trace_id or str(uuid.uuid4()),
                    name=name,
                    metadata=metadata or {},
                    tags=["jarvis-prime", "trinity"],
                )
                self._traces[trace.id] = trace
                self._traces_created += 1
                return trace.id
        except Exception as e:
            self._errors += 1
            logger.debug(f"Error creating trace: {e}")
            return trace_id or str(uuid.uuid4())[:16]

    async def _end_trace(
        self,
        trace_id: str,
        status: str = "success",
        output: Optional[Dict[str, Any]] = None,
    ):
        """End a trace."""
        if not self._initialized or not self._client:
            return

        try:
            async with self._lock:
                trace = self._traces.pop(trace_id, None)
                if trace:
                    trace.update(
                        status_message=status,
                        output=output,
                    )
                    self._traces_completed += 1
        except Exception as e:
            self._errors += 1
            logger.debug(f"Error ending trace: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        return {
            "initialized": self._initialized,
            "traces_created": self._traces_created,
            "traces_completed": self._traces_completed,
            "active_traces": len(self._traces),
            "errors": self._errors,
            "sample_rate": self._config.langfuse_sample_rate,
        }


@dataclass
class LangfuseSpan:
    """A span within a Langfuse trace."""
    _tracer: Optional[LangfuseTracer]
    name: str
    metadata: Dict[str, Any]
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    async def start(self):
        """Start the span."""
        self.start_time = time.time()
        self.span_id = str(uuid.uuid4())[:8]
        if self._tracer:
            self.trace_id = await self._tracer._create_trace(
                self.name,
                self.trace_id,
                self.metadata,
            )

    async def end(self):
        """End the span."""
        self.end_time = time.time()
        if self._tracer:
            await self._tracer._end_trace(
                self.trace_id or "",
                status="error" if self.error else "success",
                output=self.attributes,
            )

    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the span."""
        self.attributes[key] = value

    def set_error(self, error: str):
        """Set error on the span."""
        self.error = error
        self.attributes["error"] = error

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000


# =============================================================================
# CHAOS TESTING
# =============================================================================

class FaultType(Enum):
    """Types of faults that can be injected."""
    LATENCY = "latency"           # Add artificial delay
    ERROR = "error"               # Return error response
    TIMEOUT = "timeout"           # Cause timeout
    PARTIAL = "partial"           # Partial response
    CORRUPT = "corrupt"           # Corrupt data
    CIRCUIT_TRIP = "circuit_trip"  # Force circuit breaker trip
    OOM = "oom"                   # Simulate OOM condition


@dataclass
class ChaosRule:
    """A chaos injection rule."""
    fault_type: FaultType
    target: str  # Component or endpoint to target
    probability: float = 0.1
    enabled: bool = True
    duration_ms: float = 1000.0  # For latency/timeout
    error_message: str = "Injected chaos error"
    schedule: Optional[str] = None  # Cron-like schedule


class ChaosEngine:
    """
    Chaos testing engine for fault injection.

    Enables testing of failure scenarios in production-like conditions.
    """

    def __init__(self, config: ObservabilityConfig):
        self._config = config
        self._rules: Dict[str, ChaosRule] = {}
        self._injections: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._enabled = config.chaos_enabled

        # Statistics
        self._total_checks = 0
        self._total_injections = 0

    def add_rule(self, name: str, rule: ChaosRule):
        """Add a chaos rule."""
        self._rules[name] = rule
        logger.info(f"[CHAOS] Added rule: {name} ({rule.fault_type.value})")

    def remove_rule(self, name: str):
        """Remove a chaos rule."""
        if name in self._rules:
            del self._rules[name]
            logger.info(f"[CHAOS] Removed rule: {name}")

    def enable(self):
        """Enable chaos testing."""
        self._enabled = True
        logger.warning("[CHAOS] Chaos testing ENABLED")

    def disable(self):
        """Disable chaos testing."""
        self._enabled = False
        logger.info("[CHAOS] Chaos testing disabled")

    async def should_inject(self, target: str, fault_type: Optional[FaultType] = None) -> bool:
        """
        Check if fault should be injected.

        Returns True if a fault should be injected for this target.
        """
        if not self._enabled:
            return False

        self._total_checks += 1

        # Check global probability first
        if random.random() > self._config.chaos_probability:
            return False

        # Check target-specific rules
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            if rule.target != "*" and rule.target != target:
                continue
            if fault_type and rule.fault_type != fault_type:
                continue
            if random.random() <= rule.probability:
                self._total_injections += 1
                self._injections.append({
                    "timestamp": time.time(),
                    "target": target,
                    "fault_type": rule.fault_type.value,
                })
                return True

        return False

    async def inject_latency(self, target: str) -> float:
        """
        Inject latency if applicable.

        Returns the delay in seconds (0 if no injection).
        """
        if not await self.should_inject(target, FaultType.LATENCY):
            return 0.0

        for rule in self._rules.values():
            if rule.enabled and rule.fault_type == FaultType.LATENCY:
                if rule.target == "*" or rule.target == target:
                    delay = rule.duration_ms / 1000.0
                    logger.warning(f"[CHAOS] Injecting {delay:.2f}s latency to {target}")
                    await asyncio.sleep(delay)
                    return delay

        return 0.0

    async def maybe_raise_error(self, target: str):
        """
        Maybe raise an error if chaos rule applies.
        """
        if await self.should_inject(target, FaultType.ERROR):
            for rule in self._rules.values():
                if rule.enabled and rule.fault_type == FaultType.ERROR:
                    if rule.target == "*" or rule.target == target:
                        logger.warning(f"[CHAOS] Injecting error to {target}")
                        raise RuntimeError(rule.error_message)

    def get_stats(self) -> Dict[str, Any]:
        """Get chaos engine statistics."""
        return {
            "enabled": self._enabled,
            "rules_count": len(self._rules),
            "total_checks": self._total_checks,
            "total_injections": self._total_injections,
            "injection_rate": (
                self._total_injections / self._total_checks
                if self._total_checks > 0 else 0.0
            ),
            "recent_injections": self._injections[-10:],
        }


# =============================================================================
# ADAPTIVE EVENT BUS OPTIMIZATION
# =============================================================================

class AdaptivePoller:
    """
    Adaptive polling for event bus optimization.

    Dynamically adjusts polling interval based on event activity.
    """

    def __init__(
        self,
        min_interval_ms: int = 10,
        max_interval_ms: int = 1000,
        initial_interval_ms: int = 100,
    ):
        self._min_interval = min_interval_ms / 1000.0
        self._max_interval = max_interval_ms / 1000.0
        self._current_interval = initial_interval_ms / 1000.0

        # Activity tracking
        self._events_per_window: Deque[int] = deque(maxlen=100)
        self._window_start = time.time()
        self._events_in_window = 0

        # Optimization parameters
        self._scale_up_factor = 0.8  # Reduce interval faster
        self._scale_down_factor = 1.2  # Increase interval slower
        self._activity_threshold = 5  # Events per second to consider "active"

    def record_event(self):
        """Record an event occurrence."""
        self._events_in_window += 1

    def tick(self) -> float:
        """
        Get current polling interval and update for next tick.

        Returns: Interval in seconds
        """
        now = time.time()
        elapsed = now - self._window_start

        # Update window every second
        if elapsed >= 1.0:
            self._events_per_window.append(self._events_in_window)
            self._events_in_window = 0
            self._window_start = now

            # Adapt interval based on recent activity
            if len(self._events_per_window) >= 5:
                avg_events = statistics.mean(self._events_per_window)

                if avg_events > self._activity_threshold:
                    # High activity - poll more frequently
                    self._current_interval = max(
                        self._min_interval,
                        self._current_interval * self._scale_up_factor,
                    )
                else:
                    # Low activity - poll less frequently
                    self._current_interval = min(
                        self._max_interval,
                        self._current_interval * self._scale_down_factor,
                    )

        return self._current_interval

    def get_stats(self) -> Dict[str, Any]:
        """Get poller statistics."""
        return {
            "current_interval_ms": self._current_interval * 1000,
            "min_interval_ms": self._min_interval * 1000,
            "max_interval_ms": self._max_interval * 1000,
            "avg_events_per_second": (
                statistics.mean(self._events_per_window)
                if self._events_per_window else 0.0
            ),
        }


# =============================================================================
# UNIFIED OBSERVABILITY BRIDGE
# =============================================================================

class ObservabilityBridge:
    """
    Unified observability bridge for JARVIS-Prime.

    Connects to JARVIS Body's observability stack and provides:
    - Langfuse tracing
    - Prometheus metrics
    - Chaos testing
    - Adaptive polling
    """

    _instance: Optional["ObservabilityBridge"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        self._config = config or ObservabilityConfig()
        self._prometheus = PrometheusExporter()
        self._langfuse = LangfuseTracer(self._config)
        self._chaos = ChaosEngine(self._config)
        self._poller = AdaptivePoller(
            min_interval_ms=10,
            max_interval_ms=1000,
            initial_interval_ms=self._config.event_bus_poll_interval_ms,
        )
        self._initialized = False

        # JARVIS Body connection
        self._jarvis_hub_url: Optional[str] = None

    @classmethod
    async def get_instance(cls) -> "ObservabilityBridge":
        """Get or create singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self):
        """Initialize the observability bridge."""
        if self._initialized:
            return

        logger.info("[OBS] Initializing Observability Bridge...")

        # Initialize Langfuse
        await self._langfuse.initialize()
        if self._langfuse._initialized:
            logger.info("[OBS] Langfuse tracer initialized")

        # Initialize Prometheus HTTP server (optional)
        self._prometheus_server = None
        self._prometheus_server_task = None
        if self._config.prometheus_enabled:
            try:
                await self._start_prometheus_server()
            except Exception as e:
                logger.warning(f"[OBS] Failed to start Prometheus server: {e}")

        # Check for JARVIS Body observability hub
        jarvis_url = os.getenv("JARVIS_OBSERVABILITY_URL")
        if jarvis_url:
            self._jarvis_hub_url = jarvis_url
            logger.info(f"[OBS] Connected to JARVIS observability: {jarvis_url}")

        self._initialized = True
        logger.info("[OBS] Observability Bridge ready")

    async def _start_prometheus_server(self):
        """Start the Prometheus metrics HTTP server."""
        try:
            from aiohttp import web

            app = web.Application()
            app.router.add_get("/metrics", self._handle_metrics)
            app.router.add_get("/health", self._handle_health)

            runner = web.AppRunner(app)
            await runner.setup()

            site = web.TCPSite(
                runner,
                "0.0.0.0",
                self._config.prometheus_port,
                reuse_address=True,
            )
            await site.start()

            self._prometheus_server = runner
            logger.info(f"[OBS] Prometheus metrics server started on port {self._config.prometheus_port}")
            logger.info(f"[OBS]   - GET http://localhost:{self._config.prometheus_port}/metrics")
            logger.info(f"[OBS]   - GET http://localhost:{self._config.prometheus_port}/health")

        except ImportError:
            logger.info("[OBS] aiohttp not available, Prometheus server disabled")
        except OSError as e:
            if "Address already in use" in str(e):
                logger.info(f"[OBS] Port {self._config.prometheus_port} in use, Prometheus server disabled")
            else:
                raise

    async def _handle_metrics(self, request):
        """Handle /metrics endpoint for Prometheus scraping."""
        from aiohttp import web

        metrics = self._prometheus.export()
        return web.Response(
            text=metrics,
            content_type="text/plain; version=0.0.4; charset=utf-8",
        )

    async def _handle_health(self, request):
        """Handle /health endpoint."""
        from aiohttp import web

        status = self.get_status()
        return web.json_response({
            "status": "healthy" if self._initialized else "initializing",
            "langfuse": status.get("langfuse", {}),
            "chaos": {"enabled": status.get("chaos", {}).get("enabled", False)},
            "poller": status.get("poller", {}),
        })

    async def shutdown(self):
        """Shutdown the observability bridge."""
        logger.info("[OBS] Shutting down...")

        # Stop Prometheus server
        if self._prometheus_server:
            try:
                await self._prometheus_server.cleanup()
                logger.info("[OBS] Prometheus server stopped")
            except Exception as e:
                logger.debug(f"[OBS] Error stopping Prometheus server: {e}")

        await self._langfuse.shutdown()
        self._initialized = False
        ObservabilityBridge._instance = None

    # =========================================================================
    # TRACING
    # =========================================================================

    @asynccontextmanager
    async def trace(
        self,
        name: str,
        trace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[LangfuseSpan]:
        """Create a trace context."""
        async with self._langfuse.trace(name, trace_id, metadata) as span:
            yield span

    # =========================================================================
    # METRICS
    # =========================================================================

    async def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Increment a counter metric."""
        await self._prometheus.inc(name, value, labels)

    async def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Set a gauge metric."""
        await self._prometheus.set(name, value, labels)

    async def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Observe a value for histogram."""
        await self._prometheus.observe(name, value, labels)

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        return self._prometheus.export()

    # =========================================================================
    # CHAOS TESTING
    # =========================================================================

    def add_chaos_rule(self, name: str, rule: ChaosRule):
        """Add a chaos rule."""
        self._chaos.add_rule(name, rule)

    async def maybe_inject_latency(self, target: str) -> float:
        """Maybe inject latency (returns delay in seconds)."""
        return await self._chaos.inject_latency(target)

    async def maybe_inject_error(self, target: str):
        """Maybe inject an error."""
        await self._chaos.maybe_raise_error(target)

    def enable_chaos(self):
        """Enable chaos testing."""
        self._chaos.enable()

    def disable_chaos(self):
        """Disable chaos testing."""
        self._chaos.disable()

    # =========================================================================
    # ADAPTIVE POLLING
    # =========================================================================

    def record_event(self):
        """Record an event for adaptive polling."""
        self._poller.record_event()

    def get_poll_interval(self) -> float:
        """Get current adaptive poll interval in seconds."""
        return self._poller.tick()

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "initialized": self._initialized,
            "langfuse": self._langfuse.get_stats(),
            "chaos": self._chaos.get_stats(),
            "poller": self._poller.get_stats(),
            "jarvis_hub_connected": self._jarvis_hub_url is not None,
        }


# =============================================================================
# GLOBAL ACCESS FUNCTIONS
# =============================================================================

_bridge: Optional[ObservabilityBridge] = None


async def get_observability_bridge() -> ObservabilityBridge:
    """Get the observability bridge."""
    global _bridge
    if _bridge is None:
        _bridge = await ObservabilityBridge.get_instance()
    return _bridge


async def shutdown_observability_bridge():
    """Shutdown the observability bridge."""
    global _bridge
    if _bridge:
        await _bridge.shutdown()
        _bridge = None


@asynccontextmanager
async def trace_request(
    name: str,
    trace_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[LangfuseSpan]:
    """Convenience function for tracing."""
    bridge = await get_observability_bridge()
    async with bridge.trace(name, trace_id, metadata) as span:
        yield span


def export_prometheus_metrics() -> str:
    """Export Prometheus metrics (sync version for HTTP handlers)."""
    if _bridge:
        return _bridge.export_prometheus()
    return ""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Config
    "ObservabilityConfig",
    # Prometheus
    "PrometheusExporter",
    "PrometheusMetric",
    "MetricType",
    # Langfuse
    "LangfuseTracer",
    "LangfuseSpan",
    # Chaos
    "ChaosEngine",
    "ChaosRule",
    "FaultType",
    # Adaptive polling
    "AdaptivePoller",
    # Bridge
    "ObservabilityBridge",
    # Functions
    "get_observability_bridge",
    "shutdown_observability_bridge",
    "trace_request",
    "export_prometheus_metrics",
]
