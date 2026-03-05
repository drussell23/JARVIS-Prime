"""Integration tests for PrimeUmfClient -- heartbeat and event publishing."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import pytest

from umf_types import Kind, Stream, UmfMessage


class TestPrimeUmfClient:

    @pytest.fixture()
    def dedup_db_path(self, tmp_path: Path) -> Path:
        return tmp_path / "dedup.db"

    async def _make_client(self, dedup_db_path: Path):
        from umf_client import PrimeUmfClient

        client = PrimeUmfClient(
            session_id="test-session-001",
            instance_id="test-instance-001",
            dedup_db_path=dedup_db_path,
        )
        await client.start()
        return client

    @pytest.mark.asyncio
    async def test_prime_can_send_heartbeat(self, dedup_db_path: Path) -> None:
        client = await self._make_client(dedup_db_path)
        try:
            received: List[UmfMessage] = []

            async def handler(msg: UmfMessage) -> None:
                received.append(msg)

            await client.subscribe(Stream.lifecycle, handler)

            ok = await client.send_heartbeat(state="ready")
            assert ok is True, "send_heartbeat should return True on first publish"

            assert len(received) == 1
            msg = received[0]
            assert msg.source.repo == "jarvis-prime"
            assert msg.payload["state"] == "ready"
            assert msg.payload["liveness"] is True
            assert msg.payload["readiness"] is True
            assert msg.stream == Stream.lifecycle
            assert msg.kind == Kind.heartbeat
        finally:
            await client.stop()

    @pytest.mark.asyncio
    async def test_prime_can_publish_event(self, dedup_db_path: Path) -> None:
        client = await self._make_client(dedup_db_path)
        try:
            received: List[UmfMessage] = []

            async def handler(msg: UmfMessage) -> None:
                received.append(msg)

            await client.subscribe(Stream.event, handler)

            ok = await client.publish_event(
                target_repo="jarvis",
                target_component="supervisor",
                payload={"action": "model_loaded", "model": "llama-13b"},
            )
            assert ok is True, "publish_event should return True on first publish"

            assert len(received) == 1
            msg = received[0]
            assert msg.kind == Kind.event
            assert msg.stream == Stream.event
            assert msg.source.repo == "jarvis-prime"
            assert msg.target.repo == "jarvis"
            assert msg.target.component == "supervisor"
            assert msg.payload["action"] == "model_loaded"
        finally:
            await client.stop()
