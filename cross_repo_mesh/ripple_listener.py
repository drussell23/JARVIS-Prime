"""Inbound cross-repo ripple listener — jarvis-prime (the Mind).

PREDICTIONS, NOT REQUESTS. This listener INDEPENDENTLY verifies an inbound
JARVIS ripple (HMAC-SHA256 + replay nonce + TTL + origin) using the vendored
``ripple_contract``, and on success emits a *local* intent record into this
repo's own ledger for jarvis-prime's pipeline to decide on. It NEVER executes
anything JARVIS sends — ``payload.intent`` is a plain string, logged verbatim,
never invoked. Any verification failure is a SILENT DROP (no raise, no intent).

Master flag ``JARVIS_PRIME_RIPPLE_LISTENER_ENABLED`` (default FALSE): when off,
inbound ripples are ignored. The shared PSK is read from
``JARVIS_CROSS_REPO_EMIT_PSK`` (bytes; never module-level) — the SAME secret
JARVIS signs with. No PSK → listener is inert (an unsigned/unverifiable ripple
is never accepted).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from .ripple_contract import RipplePayload, VerifyVerdict, verify_ripple

logger = logging.getLogger("jarvis_prime.cross_repo_listener")

_ENV_ENABLED = "JARVIS_PRIME_RIPPLE_LISTENER_ENABLED"
_ENV_PSK = "JARVIS_CROSS_REPO_EMIT_PSK"
_ENV_INTENT_LEDGER = "JARVIS_PRIME_RIPPLE_INTENT_LEDGER_PATH"
_DEFAULT_INTENT_LEDGER = ".jarvis/inbound_ripple_intents.jsonl"
_EXPECTED_ORIGINS = ("jarvis",)
_TRUTHY = ("true", "1", "yes", "on")

_SOURCE_REPO = "jarvis-prime"


def listener_enabled() -> bool:
    """§33.1 master — default FALSE."""
    raw = os.environ.get(_ENV_ENABLED)
    if raw is None:
        return False
    return raw.strip().lower() in _TRUTHY


def _psk() -> Optional[bytes]:
    raw = os.environ.get(_ENV_PSK, "")
    if not raw:
        return None
    return raw.encode("utf-8")


def _intent_ledger_path() -> Path:
    raw = os.environ.get(_ENV_INTENT_LEDGER, "").strip()
    return Path(raw) if raw else Path(_DEFAULT_INTENT_LEDGER)


def _emit_local_intent(payload: RipplePayload, *, now_unix: float) -> dict:
    """Build + persist a LOCAL intent record. This is a NOTIFICATION for the
    jarvis-prime pipeline — NOT an instruction to execute. Never raises."""
    intent_record = {
        "schema_version": "jarvis_prime_inbound_ripple_intent.1",
        "received_by": _SOURCE_REPO,
        "ripple_kind": payload.ripple_kind,
        "intent": payload.intent,  # plain string — logged, never invoked
        "payload_sha256": payload.payload_sha256,
        "origin": payload.source_repo,
        "ripple_nonce": payload.nonce,
        "verified_at_unix": now_unix,
    }
    try:
        path = _intent_ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(intent_record, sort_keys=True) + "\n")
    except Exception as exc:  # noqa: BLE001 — best-effort; never raises
        logger.debug("[CrossRepoListener] intent ledger append failed: %s", exc)
    return intent_record


def handle_inbound_ripple(
    token: str,
    *,
    now_unix: Optional[float] = None,
    seen_nonces: Optional[Any] = None,
    psk: Optional[bytes] = None,
) -> Tuple[VerifyVerdict, Optional[dict]]:
    """Independently verify an inbound ripple token; on VERIFIED emit a local
    intent record. Returns (verdict, intent_record|None). NEVER raises, NEVER
    executes the payload. Inert (DISABLED) unless the master flag is on AND a
    PSK is configured."""
    if now_unix is None:
        now_unix = time.time()
    if not listener_enabled():
        return VerifyVerdict.DISABLED, None
    key = psk if psk is not None else _psk()
    if not key:
        # No shared secret → cannot verify → never accept an unsigned ripple.
        return VerifyVerdict.DISABLED, None
    try:
        verdict, payload = verify_ripple(
            token, key,
            now_unix=now_unix,
            seen_nonces=seen_nonces,
            expected_origins=_EXPECTED_ORIGINS,
        )
    except Exception as exc:  # noqa: BLE001 — verify_ripple shouldn't raise; belt-and-suspenders
        logger.debug("[CrossRepoListener] verify raised (dropped): %s", exc)
        return VerifyVerdict.DROPPED_MALFORMED, None
    if verdict is not VerifyVerdict.VERIFIED or payload is None:
        # Silent drop — no intent emitted, nothing executed.
        return verdict, None
    intent_record = _emit_local_intent(payload, now_unix=now_unix)
    return VerifyVerdict.VERIFIED, intent_record
