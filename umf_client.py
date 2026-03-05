"""UMF client for jarvis-prime -- self-contained, imports only from umf_types."""
from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from umf_types import (
    Kind,
    MessageSource,
    MessageTarget,
    ReserveResult,
    Stream,
    UmfMessage,
)


class _InlineDedupLedger:
    """Minimal dedup ledger for Prime (same semantics as SqliteDedupLedger)."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS dedup_ledger ("
            "idempotency_key TEXT NOT NULL, "
            "message_id TEXT PRIMARY KEY, "
            "reserved_at_ms INTEGER NOT NULL, "
            "ttl_ms INTEGER NOT NULL, "
            "committed INTEGER NOT NULL DEFAULT 0"
            ")"
        )

    async def stop(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    async def reserve(self, key: str, msg_id: str, ttl_ms: int) -> ReserveResult:
        async with self._lock:
            assert self._conn is not None
            now_ms = int(time.time() * 1000)
            row = self._conn.execute(
                "SELECT message_id, reserved_at_ms, ttl_ms "
                "FROM dedup_ledger WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                _, reserved_at, existing_ttl = row
                if (now_ms - reserved_at) > existing_ttl:
                    self._conn.execute(
                        "DELETE FROM dedup_ledger WHERE idempotency_key = ?",
                        (key,),
                    )
                else:
                    return ReserveResult.duplicate
            try:
                self._conn.execute(
                    "INSERT INTO dedup_ledger "
                    "(idempotency_key, message_id, reserved_at_ms, ttl_ms) "
                    "VALUES (?, ?, ?, ?)",
                    (key, msg_id, now_ms, ttl_ms),
                )
                return ReserveResult.reserved
            except sqlite3.IntegrityError:
                return ReserveResult.conflict


class PrimeUmfClient:
    """UMF client for jarvis-prime."""

    def __init__(
        self,
        session_id: str,
        instance_id: str,
        dedup_db_path: Path,
    ) -> None:
        self._source = MessageSource(
            repo="jarvis-prime",
            component="orchestrator",
            instance_id=instance_id,
            session_id=session_id,
        )
        self._dedup = _InlineDedupLedger(db_path=dedup_db_path)
        self._subscribers: Dict[str, List[Callable]] = {}
        self._running = False

    async def start(self) -> None:
        await self._dedup.start()
        self._running = True

    async def stop(self) -> None:
        self._running = False
        await self._dedup.stop()

    async def subscribe(self, stream: str, handler: Callable) -> str:
        key = stream.value if hasattr(stream, "value") else stream
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(handler)
        return f"sub-{key}-{len(self._subscribers[key])}"

    async def publish(self, msg: UmfMessage) -> bool:
        result = await self._dedup.reserve(
            msg.idempotency_key, msg.message_id, msg.routing_ttl_ms,
        )
        if result != ReserveResult.reserved:
            return False
        stream_key = msg.stream.value if hasattr(msg.stream, "value") else msg.stream
        for handler in self._subscribers.get(stream_key, []):
            try:
                r = handler(msg)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass
        return True

    async def send_heartbeat(
        self,
        state: str,
        liveness: bool = True,
        readiness: bool = True,
        **extra: Any,
    ) -> bool:
        payload: Dict[str, Any] = {
            "liveness": liveness,
            "readiness": readiness,
            "subsystem_role": self._source.component,
            "state": state,
            "last_error_code": extra.pop("last_error_code", ""),
            "queue_depth": extra.pop("queue_depth", 0),
            "resource_pressure": extra.pop("resource_pressure", 0.0),
        }
        payload.update(extra)
        msg = UmfMessage(
            stream=Stream.lifecycle,
            kind=Kind.heartbeat,
            source=self._source,
            target=MessageTarget(repo="jarvis", component="supervisor"),
            payload=payload,
        )
        return await self.publish(msg)

    async def publish_event(
        self,
        target_repo: str,
        target_component: str,
        payload: Dict[str, Any],
    ) -> bool:
        msg = UmfMessage(
            stream=Stream.event,
            kind=Kind.event,
            source=self._source,
            target=MessageTarget(repo=target_repo, component=target_component),
            payload=payload,
        )
        return await self.publish(msg)
