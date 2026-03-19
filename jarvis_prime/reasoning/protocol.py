"""
Mind-Body Protocol v1.0.0 — Pydantic v2 schemas.

Defines the communication contract between JARVIS (Body) and J-Prime (Mind)
in the Trinity architecture (Mind / Body / Soul).

All models use sensible defaults so callers can construct minimal instances
without supplying every field. Identifiers like ``request_id`` and ``command``
default to empty strings rather than being required, keeping construction
ergonomic while still carrying semantic meaning when populated.

Key design decisions
--------------------
- Pydantic v2 syntax throughout (``model_dump``, ``model_validate``,
  ``model_dump_json``, ``model_config``).
- ``ErrorDetail.error_class`` is exposed via ``Field(alias="class")`` so that
  JSON interchange uses the reserved word ``"class"`` as the key, while Python
  code always uses ``error_class``.  ``populate_by_name=True`` means both
  names are accepted on input.
- No required fields except where semantically mandatory (all fields carry
  defaults), making it easy to build partial objects in tests and consumers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION: str = "1.0.0"
MIN_SUPPORTED_VERSION: str = "1.0.0"
MAX_SUPPORTED_VERSION: str = "1.0.999"


# ---------------------------------------------------------------------------
# Leaf models (no nested protocol types)
# ---------------------------------------------------------------------------


class AuthEnvelope(BaseModel):
    """Authentication metadata attached to every request."""

    token_id: str = ""
    signature: str = ""
    nonce: str = ""
    issued_at: str = ""


class FallbackPolicy(BaseModel):
    """Controls how the Mind degrades when primary execution is unavailable."""

    allow_tier1: bool = True
    allow_tier2: bool = True
    queue_on_fail: bool = False
    require_approval_in_degraded: bool = False


class Constraints(BaseModel):
    """Deadline and budget guardrails for a reasoning request."""

    deadline_ms: Optional[int] = None
    deadline_at_ms: Optional[int] = None
    hard_cost_cap_usd: Optional[float] = None
    soft_cost_target_usd: Optional[float] = None
    session_budget_remaining_usd: Optional[float] = None
    allowed_task_classes: List[str] = Field(default_factory=list)


class Classification(BaseModel):
    """Output of the intent-classification brain."""

    intent: str = ""
    complexity: str = ""
    confidence: float = 0.0
    brain_used: str = ""
    graph_depth: int = 0


class SubGoal(BaseModel):
    """A single atomic step within an execution plan."""

    step_id: str = ""
    action_id: str = ""
    goal: str = ""
    task_type: str = ""
    brain_assigned: str = ""
    tool_required: str = ""
    depends_on: List[str] = Field(default_factory=list)
    estimated_ms: int = 0


class RoutingTrace(BaseModel):
    """Records which brains handled analysis/planning and gate outcomes."""

    analysis_brain: str = ""
    planning_brain: str = ""
    cost_gate_passed: bool = False
    resource_gate_passed: bool = False
    sai_health: str = ""
    cai_confidence: float = 0.0


class ErrorDetail(BaseModel):
    """Structured error payload returned when a request cannot be fulfilled."""

    model_config = {"populate_by_name": True}

    code: str = ""
    # ``error_class`` serialises as ``"class"`` in JSON (reserved Python word).
    error_class: str = Field(default="", alias="class")
    message: str = ""
    retry_after_ms: Optional[int] = None
    recovery_strategy: str = ""


class StepResult(BaseModel):
    """Result of a single executed sub-goal step."""

    step_id: str = ""
    action_id: str = ""
    success: bool = False
    output: str = ""
    latency_ms: int = 0
    tool_used: str = ""
    artifact_refs: List[str] = Field(default_factory=list)
    side_effects_committed: bool = False


# ---------------------------------------------------------------------------
# Composite models
# ---------------------------------------------------------------------------


class Plan(BaseModel):
    """A structured execution plan composed of ordered sub-goals."""

    plan_id: str = ""
    plan_hash: str = ""
    sub_goals: List[SubGoal] = Field(default_factory=list)
    execution_strategy: str = ""
    total_estimated_ms: int = 0
    risk_level: str = ""
    approval_required: bool = False
    approval_reason_codes: List[str] = Field(default_factory=list)
    approval_scope: str = ""


# ---------------------------------------------------------------------------
# Top-level request / response / feedback envelopes
# ---------------------------------------------------------------------------


class ReasonRequest(BaseModel):
    """
    Body → Mind: a request for reasoning, planning, or action execution.

    ``request_id`` and ``command`` are the semantically significant
    identifiers; all other fields enrich context but carry defaults.
    """

    protocol_version: str = PROTOCOL_VERSION
    request_id: str = ""
    idempotency_scope: str = ""
    session_id: str = ""
    trace_id: str = ""
    parent_request_id: str = ""
    auth: AuthEnvelope = Field(default_factory=AuthEnvelope)
    command: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: Constraints = Field(default_factory=Constraints)
    fallback_policy: FallbackPolicy = Field(default_factory=FallbackPolicy)


class ReasonResponse(BaseModel):
    """
    Mind → Body: the result of reasoning, including plan and classification.

    All complex sub-objects are optional so that partial / error responses
    can be constructed without fabricating placeholder data.
    """

    protocol_version: str = PROTOCOL_VERSION
    request_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    status: str = ""
    served_mode: str = ""
    requested_mode: str = ""
    degraded_reason_code: str = ""
    classification: Optional[Classification] = None
    plan: Optional[Plan] = None
    routing_trace: Optional[RoutingTrace] = None
    error: Optional[ErrorDetail] = None


class ReasonFeedback(BaseModel):
    """
    Body → Mind: post-execution telemetry reporting step-level outcomes.

    Enables the Mind to update its model, refine future plans, and support
    replay / idempotency logic.
    """

    request_id: str = ""
    session_id: str = ""
    trace_id: str = ""
    plan_id: str = ""
    plan_hash: str = ""
    step_results: List[StepResult] = Field(default_factory=list)
    final_outcome: str = ""
    replay_token: str = ""
    original_enqueued_at: str = ""
    replay_attempt: int = 0
    max_replay_attempts: int = 0


class ProtocolVersionInfo(BaseModel):
    """
    Describes the protocol version supported by a Mind endpoint.

    Returned by the ``/version`` health endpoint so the Body can perform
    compatibility negotiation before sending requests.
    """

    current_version: str = PROTOCOL_VERSION
    min_supported_version: str = MIN_SUPPORTED_VERSION
    max_supported_version: str = MAX_SUPPORTED_VERSION
    features: List[str] = Field(default_factory=list)
    brain_policy_hash: str = ""
