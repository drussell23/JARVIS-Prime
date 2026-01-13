#!/usr/bin/env python3
"""
Cross-Repository Health Monitor
================================

v92.1 - Unified Health Monitoring for Trinity Ecosystem

This module provides comprehensive health monitoring across all JARVIS repositories:
- JARVIS-Prime (Mind): LLM inference, reasoning engine, cognitive processing
- JARVIS (Body): Computer use, action execution, screen monitoring
- Reactor-Core (Nerves): Training pipeline, model registry, deployment

FEATURES:
    - Real-time health status aggregation
    - Automatic degradation detection
    - Service dependency tracking
    - Latency monitoring with percentiles
    - Circuit breaker coordination
    - Alert management with severity levels
    - Self-healing recommendations
    - Prometheus-compatible metrics export

ARCHITECTURE:
    Cross-Repo Health Monitor
            |
    +-------+-------+-------+
    |       |       |       |
    v       v       v       v
   Prime  Body  Reactor  External
   /health /health /health  APIs
            |
            v
    [Health Aggregator]
            |
    +-------+-------+
    |       |       |
    v       v       v
  Alerts  Metrics  Actions

USAGE:
    monitor = await get_cross_repo_health_monitor()

    # Get unified health status
    status = await monitor.get_unified_status()

    # Get specific component health
    prime_health = await monitor.get_component_health("jarvis_prime")

    # Check if system is degraded
    if await monitor.is_system_degraded():
        recommendations = await monitor.get_recovery_recommendations()
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ComponentConfig:
    """Configuration for a monitored component."""
    name: str
    display_name: str
    health_url: str
    port: int
    required: bool = True
    timeout_seconds: float = 5.0
    check_interval_seconds: float = 10.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    dependencies: List[str] = field(default_factory=list)


@dataclass
class HealthMonitorConfig:
    """Configuration for health monitor."""
    components: Dict[str, ComponentConfig] = field(default_factory=dict)
    state_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "health_monitor"
    )
    alert_cooldown_seconds: float = 300.0  # 5 minutes between same alerts
    metrics_retention_seconds: float = 3600.0  # 1 hour of metrics
    enable_self_healing: bool = True
    enable_prometheus_export: bool = True
    prometheus_port: int = 9090

    def __post_init__(self):
        if isinstance(self.state_dir, str):
            self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Default components if not specified
        if not self.components:
            self.components = {
                "jarvis_prime": ComponentConfig(
                    name="jarvis_prime",
                    display_name="JARVIS-Prime (Mind)",
                    health_url="http://localhost:8000/health",
                    port=8000,
                    required=True,
                    dependencies=[],
                ),
                "jarvis_body": ComponentConfig(
                    name="jarvis_body",
                    display_name="JARVIS (Body)",
                    health_url="http://localhost:8080/health",
                    port=8080,
                    required=False,
                    dependencies=["jarvis_prime"],
                ),
                "reactor_core": ComponentConfig(
                    name="reactor_core",
                    display_name="Reactor-Core (Nerves)",
                    health_url="http://localhost:8090/health",
                    port=8090,
                    required=False,
                    dependencies=["jarvis_prime"],
                ),
            }


# =============================================================================
# HEALTH CHECK RESULT
# =============================================================================

@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    component: str
    status: HealthStatus
    timestamp: float
    latency_ms: float
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ComponentHealth:
    """Aggregated health status for a component."""
    name: str
    display_name: str
    status: HealthStatus
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check: Optional[float] = None
    last_healthy: Optional[float] = None
    last_unhealthy: Optional[float] = None
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    uptime_percent: float = 100.0
    recent_checks: List[HealthCheckResult] = field(default_factory=list)
    dependencies_healthy: bool = True
    error_message: Optional[str] = None


@dataclass
class Alert:
    """Health alert."""
    id: str
    component: str
    severity: AlertSeverity
    message: str
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# CROSS-REPO HEALTH MONITOR
# =============================================================================

class CrossRepoHealthMonitor:
    """
    Unified health monitor for the Trinity ecosystem.

    Monitors all components across repositories and provides:
    - Real-time health status
    - Alert management
    - Latency tracking
    - Self-healing recommendations
    """

    def __init__(self, config: Optional[HealthMonitorConfig] = None):
        self.config = config or HealthMonitorConfig()

        # Component health tracking
        self._component_health: Dict[str, ComponentHealth] = {}
        for name, comp_config in self.config.components.items():
            self._component_health[name] = ComponentHealth(
                name=name,
                display_name=comp_config.display_name,
                status=HealthStatus.UNKNOWN,
            )

        # Latency tracking (for percentiles)
        self._latency_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        # Alerts
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._alert_cooldowns: Dict[str, float] = {}

        # Background tasks
        self._monitor_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._lock = asyncio.Lock()

        # HTTP session for health checks
        self._session = None

        # Metrics for Prometheus
        self._metrics: Dict[str, Any] = {}

    async def start(self) -> None:
        """Start the health monitor."""
        if self._running:
            return

        self._running = True

        # Create aiohttp session
        try:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
        except ImportError:
            logger.warning("aiohttp not available, health checks will be limited")

        # Start monitoring tasks for each component
        for name, comp_config in self.config.components.items():
            task = asyncio.create_task(self._monitor_component(name, comp_config))
            self._monitor_tasks[name] = task

        logger.info(f"Cross-repo health monitor started for {len(self.config.components)} components")

    async def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False

        # Cancel all monitoring tasks
        for task in self._monitor_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._monitor_tasks.clear()

        # Close HTTP session
        if self._session:
            await self._session.close()
            self._session = None

        # Persist state
        await self._save_state()

        logger.info("Cross-repo health monitor stopped")

    async def _monitor_component(
        self,
        name: str,
        config: ComponentConfig,
    ) -> None:
        """Continuously monitor a single component."""
        while self._running:
            try:
                result = await self._check_health(name, config)
                await self._process_health_result(name, config, result)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Error monitoring {name}: {e}")

            await asyncio.sleep(config.check_interval_seconds)

    async def _check_health(
        self,
        name: str,
        config: ComponentConfig,
    ) -> HealthCheckResult:
        """Perform a single health check."""
        start = time.perf_counter()

        try:
            if not self._session:
                # Fallback without aiohttp
                return HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNKNOWN,
                    timestamp=time.time(),
                    latency_ms=0,
                    error="aiohttp not available",
                )

            async with self._session.get(
                config.health_url,
                timeout=config.timeout_seconds,
            ) as response:
                latency_ms = (time.perf_counter() - start) * 1000

                if response.status == 200:
                    try:
                        data = await response.json()
                    except Exception:
                        data = {}

                    # Check for degraded status in response
                    response_status = data.get("status", "healthy")
                    if response_status == "degraded":
                        status = HealthStatus.DEGRADED
                    elif response_status in ("healthy", "ok"):
                        status = HealthStatus.HEALTHY
                    else:
                        status = HealthStatus.UNKNOWN

                    return HealthCheckResult(
                        component=name,
                        status=status,
                        timestamp=time.time(),
                        latency_ms=latency_ms,
                        response=data,
                    )
                else:
                    return HealthCheckResult(
                        component=name,
                        status=HealthStatus.UNHEALTHY,
                        timestamp=time.time(),
                        latency_ms=latency_ms,
                        error=f"HTTP {response.status}",
                    )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                component=name,
                status=HealthStatus.UNHEALTHY,
                timestamp=time.time(),
                latency_ms=(time.perf_counter() - start) * 1000,
                error="Timeout",
            )
        except Exception as e:
            return HealthCheckResult(
                component=name,
                status=HealthStatus.UNHEALTHY,
                timestamp=time.time(),
                latency_ms=(time.perf_counter() - start) * 1000,
                error=str(e),
            )

    async def _process_health_result(
        self,
        name: str,
        config: ComponentConfig,
        result: HealthCheckResult,
    ) -> None:
        """Process a health check result and update state."""
        async with self._lock:
            health = self._component_health[name]

            # Update consecutive counts
            if result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED):
                health.consecutive_successes += 1
                health.consecutive_failures = 0
                health.last_healthy = result.timestamp
            else:
                health.consecutive_failures += 1
                health.consecutive_successes = 0
                health.last_unhealthy = result.timestamp
                health.error_message = result.error

            # Update status based on thresholds
            if health.consecutive_failures >= config.unhealthy_threshold:
                health.status = HealthStatus.UNHEALTHY
            elif health.consecutive_successes >= config.healthy_threshold:
                health.status = result.status  # Could be HEALTHY or DEGRADED

            # Track latency
            health.last_check = result.timestamp
            self._latency_history[name].append((result.timestamp, result.latency_ms))

            # Trim old latency data
            cutoff = time.time() - self.config.metrics_retention_seconds
            self._latency_history[name] = [
                (t, l) for t, l in self._latency_history[name]
                if t > cutoff
            ]

            # Calculate latency percentiles
            latencies = [l for _, l in self._latency_history[name]]
            if latencies:
                latencies_sorted = sorted(latencies)
                n = len(latencies_sorted)
                health.avg_latency_ms = sum(latencies) / n
                health.p50_latency_ms = latencies_sorted[int(n * 0.5)]
                health.p95_latency_ms = latencies_sorted[int(n * 0.95)]
                health.p99_latency_ms = latencies_sorted[min(int(n * 0.99), n - 1)]

            # Track recent checks
            health.recent_checks.append(result)
            if len(health.recent_checks) > 10:
                health.recent_checks = health.recent_checks[-10:]

            # Calculate uptime
            healthy_checks = sum(
                1 for r in health.recent_checks
                if r.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            )
            health.uptime_percent = (
                (healthy_checks / len(health.recent_checks) * 100)
                if health.recent_checks
                else 100.0
            )

            # Check dependencies
            health.dependencies_healthy = all(
                self._component_health.get(dep, ComponentHealth(
                    name=dep, display_name=dep, status=HealthStatus.UNKNOWN
                )).status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
                for dep in config.dependencies
            )

            # Generate alerts if needed
            await self._check_and_generate_alerts(name, config, health, result)

    async def _check_and_generate_alerts(
        self,
        name: str,
        config: ComponentConfig,
        health: ComponentHealth,
        result: HealthCheckResult,
    ) -> None:
        """Check conditions and generate alerts."""
        now = time.time()

        # Alert: Component became unhealthy
        if health.status == HealthStatus.UNHEALTHY and health.consecutive_failures == config.unhealthy_threshold:
            alert_key = f"{name}_unhealthy"
            if self._should_alert(alert_key):
                severity = AlertSeverity.CRITICAL if config.required else AlertSeverity.ERROR
                await self._create_alert(
                    component=name,
                    severity=severity,
                    message=f"{config.display_name} is unhealthy: {result.error}",
                    details={"error": result.error, "consecutive_failures": health.consecutive_failures},
                )

        # Alert: High latency
        if health.p95_latency_ms > 5000:  # > 5 seconds
            alert_key = f"{name}_high_latency"
            if self._should_alert(alert_key):
                await self._create_alert(
                    component=name,
                    severity=AlertSeverity.WARNING,
                    message=f"{config.display_name} has high latency (p95: {health.p95_latency_ms:.0f}ms)",
                    details={"p95_latency_ms": health.p95_latency_ms, "avg_latency_ms": health.avg_latency_ms},
                )

        # Alert: Low uptime
        if health.uptime_percent < 90 and len(health.recent_checks) >= 5:
            alert_key = f"{name}_low_uptime"
            if self._should_alert(alert_key):
                await self._create_alert(
                    component=name,
                    severity=AlertSeverity.WARNING,
                    message=f"{config.display_name} has low uptime ({health.uptime_percent:.1f}%)",
                    details={"uptime_percent": health.uptime_percent},
                )

        # Alert: Dependency issues
        if not health.dependencies_healthy and health.status == HealthStatus.HEALTHY:
            alert_key = f"{name}_dep_issues"
            if self._should_alert(alert_key):
                await self._create_alert(
                    component=name,
                    severity=AlertSeverity.INFO,
                    message=f"{config.display_name} dependencies are unhealthy",
                    details={"dependencies": config.dependencies},
                )

        # Resolve alerts if component recovered
        if health.status == HealthStatus.HEALTHY and health.consecutive_successes >= config.healthy_threshold:
            await self._resolve_alerts_for_component(name)

    def _should_alert(self, alert_key: str) -> bool:
        """Check if we should generate an alert (respecting cooldown)."""
        now = time.time()
        last_alert = self._alert_cooldowns.get(alert_key, 0)
        if now - last_alert < self.config.alert_cooldown_seconds:
            return False
        self._alert_cooldowns[alert_key] = now
        return True

    async def _create_alert(
        self,
        component: str,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create a new alert."""
        import uuid
        alert = Alert(
            id=str(uuid.uuid4())[:8],
            component=component,
            severity=severity,
            message=message,
            timestamp=time.time(),
            details=details,
        )

        self._active_alerts[alert.id] = alert
        self._alert_history.append(alert)

        # Trim history
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-500:]

        logger.warning(f"[{severity.value.upper()}] {component}: {message}")
        return alert

    async def _resolve_alerts_for_component(self, component: str) -> None:
        """Resolve all active alerts for a component."""
        now = time.time()
        for alert_id, alert in list(self._active_alerts.items()):
            if alert.component == component and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = now
                logger.info(f"Alert resolved: {alert.message}")

    async def _save_state(self) -> None:
        """Save current state to disk."""
        state = {
            "timestamp": time.time(),
            "components": {
                name: {
                    "status": health.status.value,
                    "consecutive_failures": health.consecutive_failures,
                    "consecutive_successes": health.consecutive_successes,
                    "last_check": health.last_check,
                    "last_healthy": health.last_healthy,
                    "avg_latency_ms": health.avg_latency_ms,
                    "uptime_percent": health.uptime_percent,
                }
                for name, health in self._component_health.items()
            },
            "active_alerts": len(self._active_alerts),
        }

        state_path = self.config.state_dir / "health_state.json"
        try:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save health state: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def get_unified_status(self) -> Dict[str, Any]:
        """Get unified health status for all components."""
        async with self._lock:
            # Calculate overall status
            statuses = [h.status for h in self._component_health.values()]
            required_statuses = [
                h.status
                for name, h in self._component_health.items()
                if self.config.components[name].required
            ]

            if any(s == HealthStatus.UNHEALTHY for s in required_statuses):
                overall = HealthStatus.UNHEALTHY
            elif any(s == HealthStatus.DEGRADED for s in statuses):
                overall = HealthStatus.DEGRADED
            elif all(s == HealthStatus.HEALTHY for s in required_statuses):
                overall = HealthStatus.HEALTHY
            else:
                overall = HealthStatus.UNKNOWN

            return {
                "overall_status": overall.value,
                "timestamp": datetime.now().isoformat(),
                "components": {
                    name: {
                        "display_name": health.display_name,
                        "status": health.status.value,
                        "last_check": health.last_check,
                        "avg_latency_ms": round(health.avg_latency_ms, 2),
                        "p95_latency_ms": round(health.p95_latency_ms, 2),
                        "uptime_percent": round(health.uptime_percent, 2),
                        "dependencies_healthy": health.dependencies_healthy,
                        "error": health.error_message,
                    }
                    for name, health in self._component_health.items()
                },
                "active_alerts": [
                    {
                        "id": a.id,
                        "component": a.component,
                        "severity": a.severity.value,
                        "message": a.message,
                        "timestamp": a.timestamp,
                    }
                    for a in self._active_alerts.values()
                    if not a.resolved
                ],
            }

    async def get_component_health(self, component: str) -> Optional[Dict[str, Any]]:
        """Get detailed health for a specific component."""
        async with self._lock:
            health = self._component_health.get(component)
            if not health:
                return None

            config = self.config.components.get(component)

            return {
                "name": health.name,
                "display_name": health.display_name,
                "status": health.status.value,
                "consecutive_failures": health.consecutive_failures,
                "consecutive_successes": health.consecutive_successes,
                "last_check": health.last_check,
                "last_healthy": health.last_healthy,
                "last_unhealthy": health.last_unhealthy,
                "latency": {
                    "avg_ms": round(health.avg_latency_ms, 2),
                    "p50_ms": round(health.p50_latency_ms, 2),
                    "p95_ms": round(health.p95_latency_ms, 2),
                    "p99_ms": round(health.p99_latency_ms, 2),
                },
                "uptime_percent": round(health.uptime_percent, 2),
                "dependencies": config.dependencies if config else [],
                "dependencies_healthy": health.dependencies_healthy,
                "error": health.error_message,
                "recent_checks": [
                    {
                        "status": r.status.value,
                        "latency_ms": round(r.latency_ms, 2),
                        "timestamp": r.timestamp,
                        "error": r.error,
                    }
                    for r in health.recent_checks[-5:]
                ],
            }

    async def is_system_degraded(self) -> bool:
        """Check if the system is in a degraded state."""
        async with self._lock:
            for name, health in self._component_health.items():
                config = self.config.components.get(name)
                if config and config.required:
                    if health.status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED):
                        return True
            return False

    async def get_recovery_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for recovering unhealthy components."""
        recommendations = []

        async with self._lock:
            for name, health in self._component_health.items():
                if health.status == HealthStatus.UNHEALTHY:
                    config = self.config.components.get(name)

                    rec = {
                        "component": name,
                        "display_name": health.display_name,
                        "status": health.status.value,
                        "error": health.error_message,
                        "suggestions": [],
                    }

                    # Generate suggestions based on error
                    error = health.error_message or ""
                    if "Timeout" in error:
                        rec["suggestions"].extend([
                            f"Check if {config.display_name} is running",
                            f"Verify network connectivity to port {config.port}",
                            "Check system resources (CPU, memory)",
                        ])
                    elif "Connection refused" in error:
                        rec["suggestions"].extend([
                            f"Start {config.display_name} service",
                            f"Check if port {config.port} is in use by another process",
                            "Review service logs for startup errors",
                        ])
                    elif "HTTP 5" in error:
                        rec["suggestions"].extend([
                            f"Review {config.display_name} logs for errors",
                            "Check service dependencies",
                            "Restart the service if stuck",
                        ])
                    else:
                        rec["suggestions"].extend([
                            f"Check {config.display_name} logs",
                            "Verify configuration",
                            "Check for resource exhaustion",
                        ])

                    # Check dependencies
                    if not health.dependencies_healthy:
                        rec["suggestions"].insert(
                            0,
                            f"Fix dependencies first: {config.dependencies}",
                        )

                    recommendations.append(rec)

        return recommendations

    async def get_alerts(
        self,
        include_resolved: bool = False,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        alerts = []

        for alert in self._alert_history:
            if not include_resolved and alert.resolved:
                continue
            if severity and alert.severity != severity:
                continue

            alerts.append({
                "id": alert.id,
                "component": alert.component,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at,
                "details": alert.details,
            })

        return sorted(alerts, key=lambda a: a["timestamp"], reverse=True)

    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        lines.append("# HELP jarvis_component_health Component health status (1=healthy, 0=unhealthy)")
        lines.append("# TYPE jarvis_component_health gauge")

        for name, health in self._component_health.items():
            value = 1 if health.status == HealthStatus.HEALTHY else 0
            lines.append(f'jarvis_component_health{{component="{name}"}} {value}')

        lines.append("")
        lines.append("# HELP jarvis_component_latency_ms Component latency in milliseconds")
        lines.append("# TYPE jarvis_component_latency_ms gauge")

        for name, health in self._component_health.items():
            lines.append(f'jarvis_component_latency_ms{{component="{name}",percentile="avg"}} {health.avg_latency_ms:.2f}')
            lines.append(f'jarvis_component_latency_ms{{component="{name}",percentile="p50"}} {health.p50_latency_ms:.2f}')
            lines.append(f'jarvis_component_latency_ms{{component="{name}",percentile="p95"}} {health.p95_latency_ms:.2f}')
            lines.append(f'jarvis_component_latency_ms{{component="{name}",percentile="p99"}} {health.p99_latency_ms:.2f}')

        lines.append("")
        lines.append("# HELP jarvis_component_uptime_percent Component uptime percentage")
        lines.append("# TYPE jarvis_component_uptime_percent gauge")

        for name, health in self._component_health.items():
            lines.append(f'jarvis_component_uptime_percent{{component="{name}"}} {health.uptime_percent:.2f}')

        lines.append("")
        lines.append("# HELP jarvis_active_alerts Number of active alerts")
        lines.append("# TYPE jarvis_active_alerts gauge")
        active_count = sum(1 for a in self._active_alerts.values() if not a.resolved)
        lines.append(f"jarvis_active_alerts {active_count}")

        return "\n".join(lines)


# =============================================================================
# SINGLETON MANAGEMENT
# =============================================================================

_monitor_instance: Optional[CrossRepoHealthMonitor] = None
_monitor_lock = asyncio.Lock()


async def get_cross_repo_health_monitor() -> CrossRepoHealthMonitor:
    """Get or create the singleton health monitor."""
    global _monitor_instance

    if _monitor_instance is None:
        async with _monitor_lock:
            if _monitor_instance is None:
                _monitor_instance = CrossRepoHealthMonitor()
                await _monitor_instance.start()
                logger.info("Cross-repo health monitor initialized")

    return _monitor_instance


async def shutdown_cross_repo_health_monitor() -> None:
    """Shutdown the singleton health monitor."""
    global _monitor_instance

    if _monitor_instance is not None:
        async with _monitor_lock:
            if _monitor_instance is not None:
                await _monitor_instance.stop()
                _monitor_instance = None


if __name__ == "__main__":
    # Quick test
    async def test():
        monitor = await get_cross_repo_health_monitor()

        print("Waiting for health checks...")
        await asyncio.sleep(15)  # Let some checks run

        status = await monitor.get_unified_status()
        print(json.dumps(status, indent=2))

        if await monitor.is_system_degraded():
            print("\nSystem is degraded!")
            recs = await monitor.get_recovery_recommendations()
            for rec in recs:
                print(f"\n{rec['display_name']}:")
                for suggestion in rec["suggestions"]:
                    print(f"  - {suggestion}")

        print("\nPrometheus metrics:")
        print(monitor.get_prometheus_metrics())

        await shutdown_cross_repo_health_monitor()

    asyncio.run(test())
