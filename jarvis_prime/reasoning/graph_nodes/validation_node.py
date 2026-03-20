"""
ValidationNode — fail-closed 3-gate validation for the J-Prime unified reasoning pipeline.

Runs on ALL plan paths including trivial and light complexity. Pure rule-based,
deterministic, no model calls.

Gates (in order)
----------------
1. CostGate     — checks estimated sub-goal cost against the daily budget.
2. ResourceGate — verifies every tool_required is a known Body capability.
3. ApprovalGate — flags high-risk task types and degrades heavy tasks in
                  LEVEL_1_DEGRADED mode.

Fail-closed guarantee
---------------------
Any exception inside ``process()`` (including malformed sub_goals) returns
``approval_required=True`` with ``VALIDATION_UNAVAILABLE`` in the reason codes.
The node NEVER returns plan_ready state when validation cannot complete.

Design rules
------------
- No model calls (pure deterministic logic).
- All policy tables are module-level constants — no hardcoded values in methods.
- Budget and spend are injected via constructor DI for testability; production
  instances read the daily budget from the ``JARVIS_DAILY_BUDGET_USD`` env var.
- ``state`` immutability is preserved via ``model_copy(update=...)`` (Pydantic v2).
- One ``reasoning_trace`` entry is always appended with ``node="validation"``.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

from jarvis_prime.reasoning.protocol import ReasoningGraphState


# ---------------------------------------------------------------------------
# Policy tables — all configurable, no hardcoded values inside methods
# ---------------------------------------------------------------------------

# Body capabilities that J-Prime sub-goals may request.
# Any tool_required value NOT in this set triggers TOOL_UNAVAILABLE.
KNOWN_BODY_CAPABILITIES: frozenset = frozenset(
    {
        "app_control",
        "visual_browser",
        "screen_capture",
        "computer_use",
        "voice_speak",
        "voice_listen",
        "file_ops",
        "workspace_query",
        "vision_observe",
    }
)

# Task types that always require human approval before execution.
HIGH_RISK_TASK_TYPES: frozenset = frozenset(
    {
        "file_delete",
        "file_overwrite",
        "payment",
        "purchase",
        "subscribe",
        "email_compose",
        "message_send",
        "unlock",
        "auth",
        "permission_change",
        "system_shutdown",
        "process_kill",
    }
)

# Mapping from high-risk task_type → reason code reported in approval_reason_codes.
HIGH_RISK_REASON_MAP: Dict[str, str] = {
    "file_delete": "DESTRUCTIVE_FILE_OP",
    "file_overwrite": "DESTRUCTIVE_FILE_OP",
    "payment": "FINANCIAL",
    "purchase": "FINANCIAL",
    "subscribe": "FINANCIAL",
    "email_compose": "COMMUNICATION",
    "message_send": "COMMUNICATION",
    "unlock": "SECURITY",
    "auth": "SECURITY",
    "permission_change": "SECURITY",
    "system_shutdown": "SYSTEM_DISRUPTION",
    "process_kill": "SYSTEM_DISRUPTION",
}

# Estimated per-call cost (USD) for each known brain.
# Unknown brains receive the _DEFAULT_BRAIN_COST fallback.
BRAIN_COST_ESTIMATES: Dict[str, float] = {
    "phi3_lightweight": 0.0001,
    "qwen_coder": 0.0005,
    "qwen_coder_14b": 0.001,
    "qwen_coder_32b": 0.003,
    "deepseek_r1": 0.001,
    "mistral_7b_fallback": 0.0005,
}

# Fallback cost for unknown brains (conservative estimate to avoid undercount).
_DEFAULT_BRAIN_COST: float = 0.001

# Default daily budget when env var is absent or unparseable.
_DEFAULT_DAILY_BUDGET_USD: float = 10.0

# Complexity values that trigger the degraded-mode approval gate.
# "trivial" and "light" are explicitly NOT in this set.
_HEAVY_COMPLEXITIES: frozenset = frozenset({"heavy", "complex"})

# Served-mode value that activates degraded-mode stricter rules.
_DEGRADED_SERVED_MODE: str = "LEVEL_1_DEGRADED"

# Reason codes
_RC_COST_EXCEEDED: str = "COST_EXCEEDED"
_RC_TOOL_UNAVAILABLE: str = "TOOL_UNAVAILABLE"
_RC_DEGRADED_HEAVY: str = "DEGRADED_HEAVY_TASK"
_RC_VALIDATION_UNAVAILABLE: str = "VALIDATION_UNAVAILABLE"


# ---------------------------------------------------------------------------
# ValidationNode
# ---------------------------------------------------------------------------


class ValidationNode:
    """
    Fail-closed 3-gate validation. Runs on ALL plan paths.

    Parameters
    ----------
    daily_spend:
        Accumulated USD already spent today. Injected by tests; production
        callers may leave at 0.0 and let the node read an external counter.
    daily_budget:
        Override the daily budget ceiling. When ``None``, reads
        ``JARVIS_DAILY_BUDGET_USD`` from the environment, falling back to
        ``_DEFAULT_DAILY_BUDGET_USD``.
    """

    def __init__(
        self,
        daily_spend: float = 0.0,
        daily_budget: float | None = None,
    ) -> None:
        self._daily_spend = daily_spend
        if daily_budget is not None:
            self._daily_budget = daily_budget
        else:
            raw = os.environ.get("JARVIS_DAILY_BUDGET_USD", "")
            try:
                self._daily_budget = float(raw) if raw else _DEFAULT_DAILY_BUDGET_USD
            except ValueError:
                self._daily_budget = _DEFAULT_DAILY_BUDGET_USD

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, state: ReasoningGraphState) -> ReasoningGraphState:
        """
        Run all 3 gates against *state* and return an updated state.

        Always sets:
          - phase = "validating"
          - cost_gate_passed (bool)
          - resource_gate_passed (bool)
          - approval_required (bool)
          - approval_reason_codes (list of strings)
          - reasoning_trace += [{"node": "validation", ...}]

        On ANY internal exception:
          - approval_required = True
          - approval_reason_codes = [VALIDATION_UNAVAILABLE]
          - cost_gate_passed = False
          - resource_gate_passed = False
        """
        t_start = time.monotonic()

        # Fail-closed wrapper: any exception triggers the unavailable path.
        try:
            reason_codes, cost_passed, resource_passed = self._run_all_gates(state)
        except Exception:  # noqa: BLE001
            reason_codes = [_RC_VALIDATION_UNAVAILABLE]
            cost_passed = False
            resource_passed = False

        approval_required = len(reason_codes) > 0
        elapsed_ms = int((time.monotonic() - t_start) * 1000)

        trace_entry: Dict[str, Any] = {
            "node": "validation",
            "approval_required": approval_required,
            "reason_codes": reason_codes,
            "cost_gate_passed": cost_passed,
            "resource_gate_passed": resource_passed,
            "elapsed_ms": elapsed_ms,
        }

        return state.model_copy(
            update={
                "phase": "validating",
                "approval_required": approval_required,
                "approval_reason_codes": reason_codes,
                "cost_gate_passed": cost_passed,
                "resource_gate_passed": resource_passed,
                "reasoning_trace": state.reasoning_trace + [trace_entry],
            }
        )

    # ------------------------------------------------------------------
    # Gate orchestration
    # ------------------------------------------------------------------

    def _run_all_gates(
        self, state: ReasoningGraphState
    ) -> Tuple[List[str], bool, bool]:
        """
        Execute cost → resource → approval gates in sequence.

        Returns ``(reason_codes, cost_gate_passed, resource_gate_passed)``.

        Raises on malformed state so the outer fail-closed handler catches it.
        """
        # sub_goals must be a list; raises TypeError/AttributeError otherwise.
        sub_goals: List[Dict[str, Any]] = state.sub_goals  # type: ignore[assignment]
        if not isinstance(sub_goals, list):
            raise TypeError(
                f"sub_goals must be a list, got {type(sub_goals).__name__}"
            )

        reason_codes: List[str] = []

        # Gate 1: Cost
        cost_passed, cost_codes = self._cost_gate(sub_goals)
        reason_codes.extend(cost_codes)

        # Gate 2: Resource
        resource_passed, resource_codes = self._resource_gate(sub_goals)
        reason_codes.extend(resource_codes)

        # Gate 3: Approval (high-risk task types + degraded mode)
        approval_codes = self._approval_gate(
            sub_goals,
            complexity=state.complexity,
            served_mode=state.served_mode,
        )
        reason_codes.extend(approval_codes)

        return reason_codes, cost_passed, resource_passed

    # ------------------------------------------------------------------
    # Individual gates
    # ------------------------------------------------------------------

    def _cost_gate(
        self, sub_goals: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Check whether the total estimated cost of *sub_goals* exceeds the
        remaining daily budget.

        Returns ``(passed, reason_codes)``.
        """
        estimated_total = sum(
            BRAIN_COST_ESTIMATES.get(
                str(sg.get("brain_assigned", "")), _DEFAULT_BRAIN_COST
            )
            for sg in sub_goals
        )

        projected_spend = self._daily_spend + estimated_total
        if projected_spend > self._daily_budget:
            return False, [_RC_COST_EXCEEDED]
        return True, []

    def _resource_gate(
        self, sub_goals: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Verify that every ``tool_required`` in *sub_goals* is a known Body
        capability.

        Returns ``(passed, reason_codes)``.
        """
        for sg in sub_goals:
            tool = str(sg.get("tool_required", ""))
            if tool not in KNOWN_BODY_CAPABILITIES:
                return False, [_RC_TOOL_UNAVAILABLE]
        return True, []

    def _approval_gate(
        self,
        sub_goals: List[Dict[str, Any]],
        complexity: str,
        served_mode: str,
    ) -> List[str]:
        """
        Flag conditions that always require explicit human approval.

        Rules (all are additive — multiple codes can fire):
        1. Any sub-goal with a HIGH_RISK_TASK_TYPE → mapped reason code.
        2. LEVEL_1_DEGRADED + complexity in _HEAVY_COMPLEXITIES → DEGRADED_HEAVY_TASK.

        Returns list of reason codes (empty = no approval needed).
        """
        codes: List[str] = []
        seen_risk_codes: set = set()

        for sg in sub_goals:
            task_type = str(sg.get("task_type", ""))
            if task_type in HIGH_RISK_TASK_TYPES:
                rc = HIGH_RISK_REASON_MAP.get(task_type, "HIGH_RISK_TASK")
                if rc not in seen_risk_codes:
                    codes.append(rc)
                    seen_risk_codes.add(rc)

        # Degraded mode is strictly stricter — never looser
        if served_mode == _DEGRADED_SERVED_MODE and complexity in _HEAVY_COMPLEXITIES:
            if _RC_DEGRADED_HEAVY not in codes:
                codes.append(_RC_DEGRADED_HEAVY)

        return codes
