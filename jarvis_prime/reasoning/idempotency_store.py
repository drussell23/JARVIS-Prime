"""
SQLite-backed idempotency store for request_id deduplication.

Survives J-Prime restarts. Auto-prunes old entries. INSERT OR IGNORE
ensures first-write-wins semantics.

Spec: Section 7 of unified-pipeline-step2-design.md
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger("reasoning.idempotency")

_DEFAULT_DB_PATH = os.path.expanduser("~/.jarvis-prime/reasoning/idempotency.db")


class IdempotencyStore:
    def __init__(
        self,
        db_path: Optional[str] = None,
        window_hours: int = 24,
        max_entries: int = 10_000,
    ) -> None:
        self._db_path = db_path or os.getenv("REASON_IDEMPOTENCY_DB", _DEFAULT_DB_PATH)
        self._window_hours = int(os.getenv("REASON_IDEMPOTENCY_WINDOW_H", str(window_hours)))
        self._max_entries = max_entries

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_table()
        self.prune(self._window_hours)

    def _create_table(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS idempotency (
                request_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                response_json TEXT NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_created ON idempotency(created_at)"
        )
        self._conn.commit()

    def get(self, request_id: str) -> Optional[str]:
        """Return the cached response JSON for request_id, or None on miss."""
        row = self._conn.execute(
            "SELECT response_json FROM idempotency WHERE request_id = ?",
            (request_id,),
        ).fetchone()
        return row[0] if row else None

    def store(self, request_id: str, session_id: str, response_json: str) -> None:
        """Persist a response. First-write-wins via INSERT OR IGNORE."""
        self._conn.execute(
            "INSERT OR IGNORE INTO idempotency (request_id, session_id, response_json) VALUES (?, ?, ?)",
            (request_id, session_id, response_json),
        )
        self._conn.commit()
        self._maybe_evict()

    def prune(self, window_hours: Optional[int] = None) -> int:
        """Delete entries older than window_hours. Returns number of rows deleted."""
        hours = window_hours if window_hours is not None else self._window_hours
        cursor = self._conn.execute(
            "DELETE FROM idempotency WHERE created_at < datetime('now', ?)",
            (f"-{hours} hours",),
        )
        self._conn.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.debug("idempotency_store: pruned %d expired entries", deleted)
        return deleted

    def _maybe_evict(self) -> None:
        """LRU eviction: trim oldest rows when count exceeds max_entries."""
        count = self._conn.execute("SELECT COUNT(*) FROM idempotency").fetchone()[0]
        if count > self._max_entries:
            excess = count - self._max_entries
            self._conn.execute(
                "DELETE FROM idempotency WHERE request_id IN "
                "(SELECT request_id FROM idempotency ORDER BY created_at ASC LIMIT ?)",
                (excess,),
            )
            self._conn.commit()
            logger.debug("idempotency_store: evicted %d excess entries", excess)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
