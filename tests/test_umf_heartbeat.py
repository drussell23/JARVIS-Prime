"""Tests for UMF heartbeat publishing in jarvis-prime (Task 16)."""
import asyncio
import pytest
from pathlib import Path

from umf_client import PrimeUmfClient


class TestPrimeUmfHeartbeat:

    @pytest.fixture
    def client(self, tmp_path):
        return PrimeUmfClient(
            session_id="test-session",
            instance_id="test-inst",
            dedup_db_path=tmp_path / "dedup.db",
        )

    @pytest.mark.asyncio
    async def test_send_heartbeat_running(self, client):
        await client.start()
        try:
            result = await client.send_heartbeat(state="running")
            assert result is True
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_send_heartbeat_with_extras(self, client):
        await client.start()
        try:
            result = await client.send_heartbeat(
                state="running",
                liveness=True,
                readiness=True,
                queue_depth=5,
                resource_pressure=0.42,
            )
            assert result is True
        finally:
            await client.stop()
