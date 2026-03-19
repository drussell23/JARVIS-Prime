"""
Mind-Body protocol v1.0.0 — endpoint handler functions.

Two pure async functions that back the FastAPI routes registered in server.py:

  GET /v1/reason/health     → get_reason_health()
  GET /v1/protocol/version  → get_protocol_version()

Design constraints
------------------
- No FastAPI imports here; callers wrap these in @app.get decorators.
- All external dependencies (hybrid_router, yaml, pathlib) are imported
  lazily with graceful try/except so the module loads cleanly in any env.
- brain_policy_hash is a SHA-256 hex digest of brain_selection_policy.yaml
  contents, or "unavailable" when the file cannot be located/read.
- brains_loaded: populated from BrainPolicyReader task-complexity keys;
  falls back to [] when hybrid_router is not accessible.
- reasoning_graph_ready is always False in v295.0 (graph not wired yet).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from jarvis_prime.reasoning.protocol import (
    PROTOCOL_VERSION,
    MIN_SUPPORTED_VERSION,
    MAX_SUPPORTED_VERSION,
    Classification,
    ReasonRequest,
    ReasonResponse,
    RoutingTrace,
)

# ---------------------------------------------------------------------------
# Feature catalogue — every capability this Mind endpoint advertises
# ---------------------------------------------------------------------------
_FEATURES: List[str] = [
    "brain_selection",
    "cost_gates",
    "fallback_policy",
    "multi_step_planning",
    "approval_flow",
    "protocol_negotiation",
]

# ---------------------------------------------------------------------------
# brain_selection_policy.yaml search paths (most-specific first)
# Mirrors the same list in BrainPolicyReader so both find the same file.
# ---------------------------------------------------------------------------
_POLICY_SEARCH_PATHS: List[Path] = [
    # Side-by-side repos layout
    Path(__file__).parent.parent.parent.parent
    / "JARVIS-AI-Agent"
    / "backend" / "core" / "ouroboros" / "governance"
    / "brain_selection_policy.yaml",
    # Explicit env var
    Path(os.getenv("JARVIS_BRAIN_POLICY_PATH", "__missing__")),
    # JARVIS_REPO_PATH env
    Path(os.getenv("JARVIS_REPO_PATH", "."))
    / "backend" / "core" / "ouroboros" / "governance"
    / "brain_selection_policy.yaml",
]


def _find_policy_file() -> Optional[Path]:
    """Return the first readable brain_selection_policy.yaml path, or None."""
    for p in _POLICY_SEARCH_PATHS:
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            pass
    return None


def _compute_policy_hash() -> str:
    """Return SHA-256 hex digest of the policy file, or 'unavailable'."""
    path = _find_policy_file()
    if path is None:
        return "unavailable"
    try:
        raw = path.read_bytes()
        return "sha256:" + hashlib.sha256(raw).hexdigest()
    except Exception:
        return "unavailable"


def _get_brains_loaded() -> List[str]:
    """
    Return a list of known brain identifiers from BrainPolicyReader.

    BrainPolicyReader exposes a task-type → complexity map; we use its keys
    as a proxy for 'known task classes the brain can handle'.  Falls back to
    [] gracefully when hybrid_router is unavailable or the JARVIS repo is not
    mounted.
    """
    try:
        from jarvis_prime.core.hybrid_router import get_brain_policy_reader  # lazy
        reader = get_brain_policy_reader()
        task_map = reader.get_task_complexity_map()
        # Return distinct complexity tiers as the brain capability list.
        # If the map is populated, also expose the raw task keys as brain hints.
        if task_map:
            # Return sorted unique task type names — these are the 'brains' the
            # Mind knows about from the shared policy.
            return sorted(set(task_map.keys()))
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public handler functions
# ---------------------------------------------------------------------------

async def get_protocol_version() -> Dict[str, Any]:
    """
    Handler for GET /v1/protocol/version.

    Returns the current Mind-Body protocol version and feature catalogue.
    Used by JARVIS at boot-time for compatibility negotiation.
    """
    return {
        "current_version": PROTOCOL_VERSION,
        "min_supported_version": MIN_SUPPORTED_VERSION,
        "max_supported_version": MAX_SUPPORTED_VERSION,
        "features": list(_FEATURES),
        "brain_policy_hash": _compute_policy_hash(),
    }


async def get_reason_health() -> Dict[str, Any]:
    """
    Handler for GET /v1/reason/health.

    Returns the current health of the reasoning subsystem:
    - status:               "ready" | "starting" | "degraded"
    - protocol_version:     current Mind-Body protocol semver
    - brains_loaded:        task-type keys from the shared brain policy
    - brain_policy_hash:    SHA-256 of brain_selection_policy.yaml
    - reasoning_graph_ready: False (graph not wired in v295.0)
    """
    brains = _get_brains_loaded()
    policy_hash = _compute_policy_hash()

    # Determine status: if we can at minimum compute the policy hash the Mind
    # is operational; fall back to "starting" if the policy file is absent.
    if policy_hash != "unavailable":
        status = "ready"
    else:
        # Policy file missing is non-fatal but indicates a partial environment.
        status = "starting"

    return {
        "status": status,
        "protocol_version": PROTOCOL_VERSION,
        "brains_loaded": brains,
        "brain_policy_hash": policy_hash,
        "reasoning_graph_ready": False,
    }


# ---------------------------------------------------------------------------
# Brain selector singleton (lazy-initialised, one instance per process)
# ---------------------------------------------------------------------------

_brain_selector = None


def _get_brain_selector():
    """Return the process-level UnifiedBrainSelector, creating it on first call."""
    global _brain_selector
    if _brain_selector is None:
        from jarvis_prime.reasoning.unified_brain_selector import UnifiedBrainSelector  # lazy
        _brain_selector = UnifiedBrainSelector()
    return _brain_selector


# ---------------------------------------------------------------------------
# handle_brain_select — backs POST /v1/reason/select
# ---------------------------------------------------------------------------


async def handle_brain_select(req: ReasonRequest) -> dict:
    """
    Handler for POST /v1/reason/select.

    Runs the UnifiedBrainSelector 4-layer gate against the incoming
    ReasonRequest and returns a ReasonResponse serialised to a plain dict.

    Response fields
    ---------------
    status          "plan_ready"
    served_mode     "LEVEL_0_PRIMARY"
    classification  brain_used, complexity, confidence, graph_depth, intent
    routing_trace   analysis_brain, cost_gate_passed, resource_gate_passed
    """
    selector = _get_brain_selector()

    # Extract task_type from request context; fall back to "classification"
    task_type: str = str(req.context.get("task_type", "classification"))

    selection = selector.select(task_type=task_type, command=req.command)

    # Confidence: 1.0 when cost gate passes, 0.5 when it fails (degraded mode)
    confidence: float = 1.0 if selection.cost_gate_passed else 0.5

    classification = Classification(
        intent=task_type,
        complexity=selection.complexity,
        confidence=confidence,
        brain_used=selection.brain_id,
        graph_depth=selection.graph_depth,
    )

    routing_trace = RoutingTrace(
        analysis_brain=selection.brain_id,
        planning_brain=selection.brain_id,
        cost_gate_passed=selection.cost_gate_passed,
        resource_gate_passed=selection.resource_gate_passed,
    )

    resp = ReasonResponse(
        protocol_version=PROTOCOL_VERSION,
        request_id=req.request_id,
        session_id=req.session_id,
        trace_id=req.trace_id,
        status="plan_ready",
        served_mode="LEVEL_0_PRIMARY",
        classification=classification,
        routing_trace=routing_trace,
    )

    return resp.model_dump(mode="json")
