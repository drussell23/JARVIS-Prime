"""
Trinity Lock Bridge for JARVIS-Prime v1.0
==========================================

Provides unified cross-repo lock coordination with JARVIS and Reactor-Core.

This module bridges JARVIS-Prime with the unified Trinity lock system.

Usage:
    from jarvis_prime.core.trinity_lock_bridge import acquire_trinity_lock

    async with acquire_trinity_lock("model_sync") as (acquired, meta):
        if acquired:
            await sync_model()

Integration Options:
1. Standalone (this module) - Uses JARVIS lock manager if available
2. Full integration - Set JARVIS_REPO_PATH environment variable

Author: JARVIS AI System (JARVIS-Prime Integration)
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# Try to import JARVIS lock manager
JARVIS_AVAILABLE = False
JARVIS_REPO_PATH = os.getenv("JARVIS_REPO_PATH", "")

# Try common paths if not explicitly set
JARVIS_PATHS = [
    JARVIS_REPO_PATH,
    str(Path.home() / "Documents" / "repos" / "JARVIS-AI-Agent"),
    str(Path.home() / "repos" / "JARVIS-AI-Agent"),
    "/Users/djrussell23/Documents/repos/JARVIS-AI-Agent",
]

for path in JARVIS_PATHS:
    if path and os.path.isdir(path):
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            from backend.core.cross_repo_lock_bridge import (
                acquire_trinity_lock as jarvis_acquire_trinity_lock,
                TrinityLockManager as JarvisTrinityLockManager,
                TrinityLocks,
                LockMetadata as JarvisLockMetadata,
            )
            JARVIS_AVAILABLE = True
            logger.info(f"[TrinityBridge] JARVIS lock manager loaded from {path}")
            break
        except ImportError as e:
            logger.debug(f"[TrinityBridge] Could not import from {path}: {e}")
            continue


# =============================================================================
# Local LockMetadata (Compatible with JARVIS)
# =============================================================================

@dataclass
class LockMetadata:
    """Lock metadata compatible with JARVIS LockMetadata."""
    acquired_at: float = 0.0
    expires_at: float = 0.0
    owner: str = ""
    token: str = ""
    lock_name: str = ""
    process_start_time: float = 0.0
    process_name: str = ""
    process_cmdline: str = ""
    machine_id: str = ""
    backend: str = "file"
    fencing_token: int = 0
    repo_source: str = "jarvis-prime"
    extensions: int = 0

    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    def time_remaining(self) -> float:
        return self.expires_at - time.time()


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_LOCK_TTL = float(os.getenv("DISTRIBUTED_LOCK_TTL", "10.0"))
DEFAULT_TIMEOUT = float(os.getenv("DISTRIBUTED_LOCK_TIMEOUT", "5.0"))
LOCK_DIR = Path.home() / ".jarvis" / "cross_repo" / "locks"
TRINITY_LOCK_PREFIX = "jarvis:lock:trinity:"


# =============================================================================
# Local File-Based Lock Manager (Fallback)
# =============================================================================

class LocalFileLockManager:
    """
    Local file-based lock manager for JARVIS-Prime.

    Used as fallback when JARVIS lock manager is not available.
    """

    def __init__(self):
        self._fencing_counter = 0
        self._owner_id = f"jarvis-prime-{os.getpid()}-{time.time():.1f}"
        self._machine_id = self._get_machine_id()
        self._lock_dir = LOCK_DIR
        self._initialized = False

    def _get_machine_id(self) -> str:
        try:
            import platform
            import socket
            return f"{platform.system().lower()}-{socket.gethostname()}"
        except Exception:
            return "unknown"

    async def initialize(self) -> None:
        if self._initialized:
            return

        self._lock_dir.mkdir(parents=True, exist_ok=True)
        self._initialized = True

    @asynccontextmanager
    async def lock(
        self,
        name: str,
        timeout: float = DEFAULT_TIMEOUT,
        ttl: float = DEFAULT_LOCK_TTL,
    ) -> AsyncIterator[Tuple[bool, Optional[LockMetadata]]]:
        """Acquire a lock using file-based locking."""
        if not self._initialized:
            await self.initialize()

        token = str(uuid4())
        lock_file = self._lock_dir / f"{name}.dlm.lock"
        acquired = False
        metadata: Optional[LockMetadata] = None

        start_time = time.time()
        attempt = 0

        try:
            while time.time() - start_time < timeout:
                # Try to acquire
                if await self._try_acquire(lock_file, name, token, ttl):
                    acquired = True
                    self._fencing_counter += 1
                    now = time.time()
                    metadata = LockMetadata(
                        acquired_at=now,
                        expires_at=now + ttl,
                        owner=self._owner_id,
                        token=token,
                        lock_name=name,
                        machine_id=self._machine_id,
                        backend="file",
                        fencing_token=self._fencing_counter,
                        repo_source="jarvis-prime",
                    )
                    logger.debug(f"[TrinityBridge] Lock acquired: {name}")
                    break

                attempt += 1
                await asyncio.sleep(0.1)

            yield acquired, metadata

        finally:
            if acquired:
                await self._release(lock_file, token)
                logger.debug(f"[TrinityBridge] Lock released: {name}")

    async def _try_acquire(
        self,
        lock_file: Path,
        name: str,
        token: str,
        ttl: float,
    ) -> bool:
        """Try to acquire a file-based lock."""
        try:
            # Check if lock exists
            if lock_file.exists():
                try:
                    data = json.loads(lock_file.read_text())
                    expires_at = data.get("expires_at", 0)
                    if time.time() < expires_at:
                        # Lock is still valid
                        return False
                    # Lock expired - remove it
                    lock_file.unlink()
                except (json.JSONDecodeError, KeyError):
                    lock_file.unlink()

            # Create new lock
            now = time.time()
            lock_data = {
                "acquired_at": now,
                "expires_at": now + ttl,
                "owner": self._owner_id,
                "token": token,
                "lock_name": name,
                "backend": "file",
                "repo_source": "jarvis-prime",
            }

            temp_file = lock_file.with_suffix(f".tmp.{os.getpid()}.{token[:8]}")
            temp_file.write_text(json.dumps(lock_data, indent=2))
            temp_file.rename(lock_file)

            # Verify we got the lock
            verify_data = json.loads(lock_file.read_text())
            return verify_data.get("token") == token

        except Exception as e:
            logger.debug(f"[TrinityBridge] Lock acquire error: {e}")
            return False

    async def _release(self, lock_file: Path, token: str) -> None:
        """Release a file-based lock."""
        try:
            if not lock_file.exists():
                return

            data = json.loads(lock_file.read_text())
            if data.get("token") == token:
                lock_file.unlink()
        except Exception as e:
            logger.debug(f"[TrinityBridge] Lock release error: {e}")


# =============================================================================
# Global Instance
# =============================================================================

_local_manager: Optional[LocalFileLockManager] = None


async def get_local_manager() -> LocalFileLockManager:
    """Get or create local file lock manager."""
    global _local_manager
    if _local_manager is None:
        _local_manager = LocalFileLockManager()
        await _local_manager.initialize()
    return _local_manager


# =============================================================================
# Main API
# =============================================================================

@asynccontextmanager
async def acquire_trinity_lock(
    name: str,
    timeout: float = DEFAULT_TIMEOUT,
    ttl: float = DEFAULT_LOCK_TTL,
    enable_keepalive: bool = True,
) -> AsyncIterator[Tuple[bool, Optional[LockMetadata]]]:
    """
    Acquire a cross-repo Trinity lock.

    Uses JARVIS lock manager if available, falls back to local file locks.

    Args:
        name: Lock name
        timeout: Max wait time for acquisition
        ttl: Lock time-to-live
        enable_keepalive: Auto-extend TTL (only with JARVIS manager)

    Yields:
        Tuple of (acquired: bool, metadata: Optional[LockMetadata])

    Example:
        async with acquire_trinity_lock("model_sync") as (acquired, meta):
            if acquired:
                print(f"Fencing token: {meta.fencing_token}")
                await sync_model()
    """
    if JARVIS_AVAILABLE:
        # Use JARVIS unified lock manager
        async with jarvis_acquire_trinity_lock(
            name,
            repo="jarvis-prime",
            timeout=timeout,
            ttl=ttl,
            enable_keepalive=enable_keepalive,
        ) as result:
            # Convert to local LockMetadata type if needed
            acquired, jarvis_meta = result
            if jarvis_meta:
                metadata = LockMetadata(
                    acquired_at=jarvis_meta.acquired_at,
                    expires_at=jarvis_meta.expires_at,
                    owner=jarvis_meta.owner,
                    token=jarvis_meta.token,
                    lock_name=jarvis_meta.lock_name,
                    process_start_time=jarvis_meta.process_start_time,
                    process_name=jarvis_meta.process_name,
                    machine_id=jarvis_meta.machine_id,
                    backend=jarvis_meta.backend,
                    fencing_token=jarvis_meta.fencing_token,
                    repo_source=jarvis_meta.repo_source,
                    extensions=jarvis_meta.extensions,
                )
                yield acquired, metadata
            else:
                yield acquired, None
    else:
        # Fall back to local file lock manager
        manager = await get_local_manager()
        async with manager.lock(name, timeout, ttl) as result:
            yield result


# =============================================================================
# Standard Lock Names
# =============================================================================

if not JARVIS_AVAILABLE:
    class TrinityLocks:
        """Standard lock names for cross-repo coordination."""
        MODEL_SYNC = "trinity:model_sync"
        MODEL_UPDATE = "trinity:model_update"
        MODEL_DEPLOY = "trinity:model_deploy"
        TRAINING_JOB = "trinity:training_job"
        TRAINING_DATA_EXPORT = "trinity:training_data_export"
        CHECKPOINT_SAVE = "trinity:checkpoint_save"
        INFERENCE_BATCH = "trinity:inference_batch"
        CACHE_UPDATE = "trinity:cache_update"
        STATE_SYNC = "trinity:state_sync"
        CONFIG_UPDATE = "trinity:config_update"
        HEALTH_CHECK = "trinity:health_check"
        VBIA_EVENTS = "trinity:vbia_events"
        SPEAKER_PROFILE = "trinity:speaker_profile"
        AUTH_STATE = "trinity:auth_state"


__all__ = [
    "acquire_trinity_lock",
    "LockMetadata",
    "TrinityLocks",
    "get_local_manager",
    "LocalFileLockManager",
    "JARVIS_AVAILABLE",
]
