"""Recoverable cross-repo partition degradation — jarvis-prime.

Slice 98 Phase 3 (distributed immune response). When the cryptographic handshake
with JARVIS goes STALE (no freshly-VERIFIED ripple within a staleness window —
i.e. a likely network partition), this sibling SOFTLY + REVERSIBLY degrades its
local posture to OBSERVATION_ONLY. The exact moment a fresh JARVIS ripple
verifies again, the posture recovers to NORMAL.

THE RECOVERY INVARIANT (non-negotiable): the posture is a PURE FUNCTION of
``now`` vs the last-verified timestamp — NOT a latch. There is NO irreversible
sever, NO write-access cut, NO state that needs a manual reset. When a fresh
handshake lands (``record_verified``), ``now - last_verified`` drops below the
window and the posture relaxes automatically. Cold start (never received a
ripple) is NORMAL, not partition — we never false-paranoia before a connection
was ever established.

Master flag ``JARVIS_PRIME_PARTITION_DEGRADATION_ENABLED`` (default FALSE):
inert when off (posture is always NORMAL). Never raises.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("jarvis_prime.partition_degradation")

_ENV_ENABLED = "JARVIS_PRIME_PARTITION_DEGRADATION_ENABLED"
_ENV_STALENESS_S = "JARVIS_PRIME_PARTITION_STALENESS_S"
_ENV_STATE_PATH = "JARVIS_PRIME_LAST_VERIFIED_PATH"
_DEFAULT_STALENESS_S = 300.0
_DEFAULT_STATE_PATH = ".jarvis/cross_repo_last_verified"
_TRUTHY = ("true", "1", "yes", "on")

POSTURE_NORMAL = "normal"
POSTURE_OBSERVATION_ONLY = "observation_only"


def degradation_enabled() -> bool:
    """§33.1 master — default FALSE."""
    raw = os.environ.get(_ENV_ENABLED)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _staleness_s() -> float:
    raw = os.environ.get(_ENV_STALENESS_S, "").strip()
    try:
        v = float(raw)
        return v if v > 0 else _DEFAULT_STALENESS_S
    except (ValueError, TypeError):
        return _DEFAULT_STALENESS_S


def _state_path(state_path: Optional[str] = None) -> Path:
    if state_path:
        return Path(state_path)
    raw = os.environ.get(_ENV_STATE_PATH, "").strip()
    return Path(raw) if raw else Path(_DEFAULT_STATE_PATH)


def record_verified(now_unix: Optional[float] = None, *, state_path: Optional[str] = None) -> None:
    """Stamp the last-verified-handshake time (called on every VERIFIED ripple).
    Best-effort; never raises. This is the ONLY write — it strictly moves the
    freshness forward, which is what makes recovery automatic."""
    if now_unix is None:
        now_unix = time.time()
    try:
        path = _state_path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(repr(float(now_unix)), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — best-effort; never raises
        logger.debug("[PartitionDegradation] record_verified failed: %s", exc)


def read_last_verified(state_path: Optional[str] = None) -> Optional[float]:
    """Read the last-verified-handshake time, or None if never recorded.
    Never raises."""
    try:
        path = _state_path(state_path)
        if not path.is_file():
            return None
        return float(path.read_text(encoding="utf-8").strip())
    except Exception as exc:  # noqa: BLE001
        logger.debug("[PartitionDegradation] read_last_verified failed: %s", exc)
        return None


def partition_posture(
    now_unix: Optional[float] = None,
    *,
    last_verified_unix: Optional[float] = None,
    state_path: Optional[str] = None,
) -> str:
    """Pure-function local posture. NORMAL unless the handshake is STALE:

      * master OFF                       -> NORMAL (inert)
      * never received a ripple (None)   -> NORMAL (cold start, not partition)
      * now - last_verified <= staleness -> NORMAL (fresh)
      * now - last_verified  > staleness -> OBSERVATION_ONLY (partition)

    Recovers AUTOMATICALLY: a fresh ``record_verified`` moves last_verified
    forward so the next call returns NORMAL. No latch, no manual reset, no
    irreversible action. Never raises."""
    if not degradation_enabled():
        return POSTURE_NORMAL
    if now_unix is None:
        now_unix = time.time()
    lv = last_verified_unix if last_verified_unix is not None else read_last_verified(state_path)
    if lv is None:
        return POSTURE_NORMAL
    # Clock-skew safe: a future/equal stamp is "fresh" (age <= 0).
    if (now_unix - lv) > _staleness_s():
        return POSTURE_OBSERVATION_ONLY
    return POSTURE_NORMAL
