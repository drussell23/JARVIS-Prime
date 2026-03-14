"""Tests for vram_pressure_monitor.py — GPU memory watchdog."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jarvis_prime.core.vram_pressure_monitor import (
    VRAMPressureZone,
    VRAMPressureEvent,
    VRAMMonitorConfig,
    VRAMPressureMonitor,
    estimate_effective_free,
)


class TestEstimateEffectiveFree:

    def test_post_unload_high_usability(self):
        free = estimate_effective_free(10_000_000_000, model_loaded=False)
        assert free == int(10_000_000_000 * 0.95)

    def test_with_model_loaded_conservative(self):
        free = estimate_effective_free(5_000_000_000, model_loaded=True)
        assert free == int(5_000_000_000 * 0.80)


class TestVRAMPressureZone:

    def test_zone_from_utilization(self):
        config = VRAMMonitorConfig()
        assert VRAMPressureMonitor._zone_for_utilization(0.50, config) == VRAMPressureZone.GREEN
        assert VRAMPressureMonitor._zone_for_utilization(0.75, config) == VRAMPressureZone.YELLOW
        assert VRAMPressureMonitor._zone_for_utilization(0.88, config) == VRAMPressureZone.RED
        assert VRAMPressureMonitor._zone_for_utilization(0.95, config) == VRAMPressureZone.CRITICAL


class TestVRAMPressureMonitor:

    @pytest.mark.asyncio
    async def test_mock_backend_reports_zones(self):
        """Mock backend should allow manual VRAM setting."""
        config = VRAMMonitorConfig(backend="mock")
        monitor = VRAMPressureMonitor(config=config, node_id="test-node")
        monitor.set_mock_vram(total=23_000_000_000, used=10_000_000_000)

        snapshot = await monitor.sample()
        assert snapshot.zone == VRAMPressureZone.GREEN

    @pytest.mark.asyncio
    async def test_event_emitted_on_zone_change(self):
        """Zone transition should emit a VRAMPressureEvent."""
        config = VRAMMonitorConfig(backend="mock", sustained_threshold_s=0.0)
        monitor = VRAMPressureMonitor(config=config, node_id="test-node")
        events: list = []
        monitor.on_pressure_change(events.append)

        # Start in GREEN
        monitor.set_mock_vram(total=23_000_000_000, used=10_000_000_000)
        await monitor.sample()

        # Jump to RED
        monitor.set_mock_vram(total=23_000_000_000, used=20_000_000_000)
        await monitor.sample()

        assert len(events) >= 1
        assert events[-1].zone == VRAMPressureZone.RED
        assert events[-1].node_id == "test-node"

    @pytest.mark.asyncio
    async def test_no_event_on_same_zone(self):
        """No event when zone stays the same."""
        config = VRAMMonitorConfig(backend="mock", sustained_threshold_s=0.0)
        monitor = VRAMPressureMonitor(config=config, node_id="test-node")
        events: list = []
        monitor.on_pressure_change(events.append)

        monitor.set_mock_vram(total=23_000_000_000, used=10_000_000_000)
        await monitor.sample()
        await monitor.sample()

        # Only initial transition from UNKNOWN → GREEN
        assert len(events) <= 1
