"""
AGI Metrics and Monitoring
============================

v78.0 - Comprehensive observability for AGI systems

Provides real-time metrics collection, aggregation, and export:
- Request/response latencies
- Model inference statistics
- Reasoning engine performance
- Learning progress
- Resource utilization
- Error rates

FEATURES:
    - Time-series data collection
    - Histograms and percentiles
    - Prometheus-compatible export
    - Alerting thresholds
    - Dashboard-ready data
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)


# =============================================================================
# METRIC TYPES
# =============================================================================


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = auto()      # Monotonically increasing
    GAUGE = auto()        # Point-in-time value
    HISTOGRAM = auto()    # Distribution of values
    SUMMARY = auto()      # Similar to histogram with percentiles
    TIMER = auto()        # Duration measurements


@dataclass
class MetricLabel:
    """Label for metric dimensionality."""

    name: str
    value: str


@dataclass
class MetricSample:
    """Single metric sample."""

    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# METRIC COLLECTORS
# =============================================================================


class Counter:
    """Counter metric that only increases."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value = 0.0
        self._labels: Dict[str, str] = {}

    def inc(self, value: float = 1.0) -> None:
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter can only be incremented")
        self._value += value

    def get(self) -> float:
        """Get current value."""
        return self._value

    def labels(self, **kwargs: str) -> "Counter":
        """Create labeled counter."""
        labeled = Counter(self.name, self.description)
        labeled._value = self._value
        labeled._labels = kwargs
        return labeled


class Gauge:
    """Gauge metric for point-in-time values."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._value = 0.0

    def set(self, value: float) -> None:
        """Set gauge value."""
        self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        self._value -= value

    def get(self) -> float:
        """Get current value."""
        return self._value


class Histogram:
    """Histogram for value distribution."""

    DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
    )

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts: Dict[float, int] = {b: 0 for b in self._buckets}
        self._bucket_counts[float("inf")] = 0
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float) -> None:
        """Observe a value."""
        self._sum += value
        self._count += 1

        for bucket in sorted(self._bucket_counts.keys()):
            if value <= bucket:
                self._bucket_counts[bucket] += 1

    def get_bucket_counts(self) -> Dict[float, int]:
        """Get bucket counts."""
        return dict(self._bucket_counts)

    def get_sum(self) -> float:
        """Get sum of all observations."""
        return self._sum

    def get_count(self) -> int:
        """Get count of observations."""
        return self._count

    def get_mean(self) -> float:
        """Get mean of observations."""
        return self._sum / max(self._count, 1)


class Timer:
    """Timer for measuring durations."""

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._histogram = Histogram(
            name,
            description,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        )

    def time(self) -> "TimerContext":
        """Context manager for timing."""
        return TimerContext(self)

    def observe(self, duration: float) -> None:
        """Observe a duration."""
        self._histogram.observe(duration)

    def get_mean(self) -> float:
        """Get mean duration."""
        return self._histogram.get_mean()

    def get_count(self) -> int:
        """Get number of observations."""
        return self._histogram.get_count()


class TimerContext:
    """Context manager for timing."""

    def __init__(self, timer: Timer) -> None:
        self._timer = timer
        self._start = 0.0

    def __enter__(self) -> "TimerContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        duration = time.perf_counter() - self._start
        self._timer.observe(duration)

    async def __aenter__(self) -> "TimerContext":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, *args: Any) -> None:
        duration = time.perf_counter() - self._start
        self._timer.observe(duration)


class Summary:
    """Summary with percentile calculations."""

    def __init__(
        self,
        name: str,
        description: str = "",
        max_samples: int = 10000,
    ) -> None:
        self.name = name
        self.description = description
        self._samples: Deque[float] = deque(maxlen=max_samples)
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float) -> None:
        """Observe a value."""
        self._samples.append(value)
        self._sum += value
        self._count += 1

    def get_percentile(self, percentile: float) -> float:
        """Get percentile value."""
        if not self._samples:
            return 0.0

        sorted_samples = sorted(self._samples)
        index = int(len(sorted_samples) * percentile / 100)
        return sorted_samples[min(index, len(sorted_samples) - 1)]

    def get_mean(self) -> float:
        """Get mean of observations."""
        if not self._samples:
            return 0.0
        return statistics.mean(self._samples)

    def get_stddev(self) -> float:
        """Get standard deviation."""
        if len(self._samples) < 2:
            return 0.0
        return statistics.stdev(self._samples)

    def get_quantiles(self) -> Dict[str, float]:
        """Get common quantiles."""
        return {
            "p50": self.get_percentile(50),
            "p90": self.get_percentile(90),
            "p95": self.get_percentile(95),
            "p99": self.get_percentile(99),
        }


# =============================================================================
# METRICS REGISTRY
# =============================================================================


class MetricsRegistry:
    """Registry for all metrics."""

    def __init__(self) -> None:
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._timers: Dict[str, Timer] = {}
        self._summaries: Dict[str, Summary] = {}

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create counter."""
        if name not in self._counters:
            self._counters[name] = Counter(name, description)
        return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create gauge."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description)
        return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[Tuple[float, ...]] = None,
    ) -> Histogram:
        """Get or create histogram."""
        if name not in self._histograms:
            self._histograms[name] = Histogram(name, description, buckets)
        return self._histograms[name]

    def timer(self, name: str, description: str = "") -> Timer:
        """Get or create timer."""
        if name not in self._timers:
            self._timers[name] = Timer(name, description)
        return self._timers[name]

    def summary(
        self,
        name: str,
        description: str = "",
        max_samples: int = 10000,
    ) -> Summary:
        """Get or create summary."""
        if name not in self._summaries:
            self._summaries[name] = Summary(name, description, max_samples)
        return self._summaries[name]

    def get_all(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        result = {
            "counters": {n: c.get() for n, c in self._counters.items()},
            "gauges": {n: g.get() for n, g in self._gauges.items()},
            "histograms": {
                n: {
                    "count": h.get_count(),
                    "sum": h.get_sum(),
                    "mean": h.get_mean(),
                }
                for n, h in self._histograms.items()
            },
            "timers": {
                n: {
                    "count": t.get_count(),
                    "mean": t.get_mean(),
                }
                for n, t in self._timers.items()
            },
            "summaries": {
                n: {
                    "mean": s.get_mean(),
                    "stddev": s.get_stddev(),
                    **s.get_quantiles(),
                }
                for n, s in self._summaries.items()
            },
        }
        return result


# =============================================================================
# AGI METRICS COLLECTOR
# =============================================================================


class AGIMetricsCollector:
    """
    Specialized metrics collector for AGI systems.

    Pre-defines common AGI metrics and provides convenience methods.

    Usage:
        metrics = AGIMetricsCollector()

        # Record inference
        with metrics.inference_timer():
            result = await model.generate(prompt)

        # Record reasoning
        metrics.record_reasoning("chain_of_thought", 1.5, 0.85)

        # Record learning
        metrics.record_learning_update(100, 0.05)

        # Get all metrics
        data = metrics.get_metrics()
    """

    def __init__(self) -> None:
        self._registry = MetricsRegistry()

        # Pre-define AGI metrics
        # Inference metrics
        self._inference_requests = self._registry.counter(
            "agi_inference_requests_total",
            "Total inference requests",
        )
        self._inference_errors = self._registry.counter(
            "agi_inference_errors_total",
            "Total inference errors",
        )
        self._inference_latency = self._registry.timer(
            "agi_inference_latency_seconds",
            "Inference latency distribution",
        )
        self._inference_tokens = self._registry.histogram(
            "agi_inference_tokens",
            "Tokens per inference",
            buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000),
        )

        # Reasoning metrics
        self._reasoning_requests = self._registry.counter(
            "agi_reasoning_requests_total",
            "Total reasoning requests",
        )
        self._reasoning_latency = self._registry.timer(
            "agi_reasoning_latency_seconds",
            "Reasoning latency distribution",
        )
        self._reasoning_confidence = self._registry.summary(
            "agi_reasoning_confidence",
            "Reasoning confidence scores",
        )
        self._reasoning_depth = self._registry.histogram(
            "agi_reasoning_depth",
            "Reasoning depth (steps)",
            buckets=(1, 2, 3, 5, 7, 10, 15, 20),
        )

        # Learning metrics
        self._learning_updates = self._registry.counter(
            "agi_learning_updates_total",
            "Total learning updates",
        )
        self._learning_experiences = self._registry.gauge(
            "agi_learning_experiences_buffered",
            "Experiences in buffer",
        )
        self._learning_loss = self._registry.summary(
            "agi_learning_loss",
            "Training loss values",
        )

        # Model metrics
        self._model_loaded = self._registry.gauge(
            "agi_model_loaded",
            "Whether model is loaded (1=yes, 0=no)",
        )
        self._model_memory_mb = self._registry.gauge(
            "agi_model_memory_mb",
            "Model memory usage in MB",
        )

        # Request metrics
        self._active_requests = self._registry.gauge(
            "agi_active_requests",
            "Currently active requests",
        )
        self._queue_size = self._registry.gauge(
            "agi_queue_size",
            "Request queue size",
        )

        # Error metrics
        self._error_by_category = self._registry.counter(
            "agi_errors_by_category",
            "Errors by category",
        )

    # -------------------------------------------------------------------------
    # INFERENCE METRICS
    # -------------------------------------------------------------------------

    def inference_timer(self) -> TimerContext:
        """Context manager for timing inference."""
        return self._inference_latency.time()

    def record_inference(
        self,
        tokens: int,
        latency_ms: Optional[float] = None,
        success: bool = True,
    ) -> None:
        """Record an inference."""
        self._inference_requests.inc()
        self._inference_tokens.observe(tokens)

        if not success:
            self._inference_errors.inc()

        if latency_ms is not None:
            self._inference_latency.observe(latency_ms / 1000)

    def set_active_requests(self, count: int) -> None:
        """Set number of active requests."""
        self._active_requests.set(count)

    def inc_active_requests(self) -> None:
        """Increment active requests."""
        self._active_requests.inc()

    def dec_active_requests(self) -> None:
        """Decrement active requests."""
        self._active_requests.dec()

    # -------------------------------------------------------------------------
    # REASONING METRICS
    # -------------------------------------------------------------------------

    def reasoning_timer(self) -> TimerContext:
        """Context manager for timing reasoning."""
        return self._reasoning_latency.time()

    def record_reasoning(
        self,
        strategy: str,
        latency_seconds: Optional[float] = None,
        confidence: Optional[float] = None,
        depth: Optional[int] = None,
    ) -> None:
        """Record a reasoning operation."""
        self._reasoning_requests.inc()

        if latency_seconds is not None:
            self._reasoning_latency.observe(latency_seconds)

        if confidence is not None:
            self._reasoning_confidence.observe(confidence)

        if depth is not None:
            self._reasoning_depth.observe(depth)

    # -------------------------------------------------------------------------
    # LEARNING METRICS
    # -------------------------------------------------------------------------

    def record_learning_update(
        self,
        experiences_used: int,
        loss: float,
    ) -> None:
        """Record a learning update."""
        self._learning_updates.inc()
        self._learning_loss.observe(loss)

    def set_buffered_experiences(self, count: int) -> None:
        """Set number of buffered experiences."""
        self._learning_experiences.set(count)

    # -------------------------------------------------------------------------
    # MODEL METRICS
    # -------------------------------------------------------------------------

    def set_model_loaded(self, loaded: bool) -> None:
        """Set model loaded status."""
        self._model_loaded.set(1 if loaded else 0)

    def set_model_memory(self, memory_mb: float) -> None:
        """Set model memory usage."""
        self._model_memory_mb.set(memory_mb)

    # -------------------------------------------------------------------------
    # ERROR METRICS
    # -------------------------------------------------------------------------

    def record_error(self, category: str) -> None:
        """Record an error by category."""
        self._error_by_category.inc()

    # -------------------------------------------------------------------------
    # EXPORT
    # -------------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        return self._registry.get_all()

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, value in self._registry._counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value.get()}")

        for name, value in self._registry._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value.get()}")

        for name, hist in self._registry._histograms.items():
            lines.append(f"# TYPE {name} histogram")
            for bucket, count in sorted(hist.get_bucket_counts().items()):
                bucket_str = f"+Inf" if bucket == float("inf") else str(bucket)
                lines.append(f'{name}_bucket{{le="{bucket_str}"}} {count}')
            lines.append(f"{name}_sum {hist.get_sum()}")
            lines.append(f"{name}_count {hist.get_count()}")

        return "\n".join(lines)


# =============================================================================
# HEALTH CHECKER
# =============================================================================


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Health checker for AGI components.

    Runs periodic health checks and aggregates results.
    """

    def __init__(self) -> None:
        self._checks: Dict[str, Callable[[], asyncio.coroutine]] = {}
        self._results: Dict[str, HealthCheck] = {}
        self._check_task: Optional[asyncio.Task] = None
        self._interval = 30.0  # seconds

    def register(
        self,
        name: str,
        check_fn: Callable[[], asyncio.coroutine],
    ) -> None:
        """Register a health check."""
        self._checks[name] = check_fn

    async def check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self._checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check not found",
            )

        start = time.perf_counter()
        try:
            check_fn = self._checks[name]
            result = await check_fn()

            latency = (time.perf_counter() - start) * 1000

            if isinstance(result, HealthCheck):
                result.latency_ms = latency
                return result
            elif isinstance(result, bool):
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    latency_ms=latency,
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    latency_ms=latency,
                )

        except Exception as e:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.perf_counter() - start) * 1000,
            )

    async def check_all(self) -> Dict[str, HealthCheck]:
        """Run all health checks."""
        results = {}
        for name in self._checks:
            results[name] = await self.check(name)
        self._results = results
        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall health status."""
        if not self._results:
            return HealthStatus.UNHEALTHY

        statuses = [r.status for r in self._results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED

    async def start_periodic_checks(self, interval: float = 30.0) -> None:
        """Start periodic health checks."""
        self._interval = interval
        self._check_task = asyncio.create_task(self._check_loop())

    async def stop_periodic_checks(self) -> None:
        """Stop periodic health checks."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

    async def _check_loop(self) -> None:
        """Periodic check loop."""
        while True:
            try:
                await self.check_all()
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check error: {e}")
                await asyncio.sleep(self._interval)


# =============================================================================
# SINGLETON
# =============================================================================


_metrics_collector: Optional[AGIMetricsCollector] = None
_health_checker: Optional[HealthChecker] = None


def get_metrics() -> AGIMetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = AGIMetricsCollector()
    return _metrics_collector


def get_health_checker() -> HealthChecker:
    """Get global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
