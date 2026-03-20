"""
TDD tests for ValidationNode — fail-closed 3-gate validation.

Written BEFORE implementation. All tests must fail with ImportError initially,
then pass once validation_node.py is implemented.

Tests cover:
- CostGate: under-budget passes, over-budget requires approval
- ResourceGate: known tool passes, unknown tool requires approval
- ApprovalGate: high-risk task types, degraded mode + heavy complexity
- FailClosed: any exception → VALIDATION_UNAVAILABLE
- LightValidation: trivial complexity still validated, high-risk still caught
- Trace: reasoning_trace entry with node="validation"
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from jarvis_prime.reasoning.protocol import ReasoningGraphState
from jarvis_prime.reasoning.graph_nodes.validation_node import ValidationNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(
    command: str = "test command",
    sub_goals: Any = None,
    complexity: str = "light",
    served_mode: str = "LEVEL_0_PRIMARY",
    task_type: str = "system_command",
    tool_required: str = "app_control",
) -> ReasoningGraphState:
    """
    Build a minimal ReasoningGraphState for validation tests.

    sub_goals defaults to a single safe sub-goal dict unless explicitly
    overridden (including None, which tests the fail-closed path).
    """
    default_sg: List[Dict[str, Any]] = [
        {
            "brain_assigned": "phi3_lightweight",
            "tool_required": tool_required,
            "task_type": task_type,
            "step_id": "s1",
            "action_id": "a1",
            "goal": "test",
        }
    ]
    # Allow callers to pass None explicitly to test fail-closed path
    resolved_sub_goals = default_sg if sub_goals is None else sub_goals
    return ReasoningGraphState(
        request_id="r1",
        session_id="s1",
        trace_id="t1",
        command=command,
        sub_goals=resolved_sub_goals,
        complexity=complexity,
        served_mode=served_mode,
    )


def make_state_no_subgoals() -> ReasoningGraphState:
    """State with an explicitly empty sub_goals list (still valid list)."""
    return ReasoningGraphState(
        request_id="r1",
        session_id="s1",
        trace_id="t1",
        command="test",
        sub_goals=[],
        complexity="light",
        served_mode="LEVEL_0_PRIMARY",
    )


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# TestCostGate
# ---------------------------------------------------------------------------


class TestCostGate:
    """Cost gate: accumulated spend vs. daily budget."""

    def test_under_budget_passes(self):
        """Sub-goals with cheap brains, zero spend → cost_gate_passed=True."""
        sg = [
            {
                "brain_assigned": "phi3_lightweight",
                "tool_required": "app_control",
                "task_type": "system_command",
                "step_id": "s1",
                "action_id": "a1",
                "goal": "do something cheap",
            }
        ]
        state = make_state(sub_goals=sg)
        node = ValidationNode(daily_spend=0.0)
        result = run(node.process(state))
        assert result.cost_gate_passed is True

    def test_over_budget_requires_approval(self):
        """Daily spend near budget + expensive sub_goals → approval_required=True."""
        # Use an expensive brain and pre-load spend close to the default budget
        sg = [
            {
                "brain_assigned": "qwen_coder_32b",
                "tool_required": "app_control",
                "task_type": "system_command",
                "step_id": "s1",
                "action_id": "a1",
                "goal": "run expensive reasoning",
            }
        ]
        state = make_state(sub_goals=sg)
        # Inject spend that puts us at 99.9% of budget so the expensive sub_goal tips it over
        node = ValidationNode(daily_spend=9.999)
        result = run(node.process(state))
        assert result.approval_required is True
        assert "COST_EXCEEDED" in result.approval_reason_codes

    def test_cost_gate_passed_false_when_over_budget(self):
        """cost_gate_passed must be False when budget is exceeded."""
        sg = [
            {
                "brain_assigned": "qwen_coder_32b",
                "tool_required": "app_control",
                "task_type": "system_command",
                "step_id": "s1",
                "action_id": "a1",
                "goal": "expensive",
            }
        ]
        state = make_state(sub_goals=sg)
        node = ValidationNode(daily_spend=9.999)
        result = run(node.process(state))
        assert result.cost_gate_passed is False


# ---------------------------------------------------------------------------
# TestResourceGate
# ---------------------------------------------------------------------------


class TestResourceGate:
    """Resource gate: every tool_required must be a known Body capability."""

    def test_known_tool_passes(self):
        """tool_required='app_control' → resource_gate_passed=True."""
        sg = [
            {
                "brain_assigned": "phi3_lightweight",
                "tool_required": "app_control",
                "task_type": "system_command",
                "step_id": "s1",
                "action_id": "a1",
                "goal": "open app",
            }
        ]
        state = make_state(sub_goals=sg)
        node = ValidationNode()
        result = run(node.process(state))
        assert result.resource_gate_passed is True

    def test_unknown_tool_requires_approval(self):
        """tool_required='nonexistent_tool' → approval_required, TOOL_UNAVAILABLE."""
        sg = [
            {
                "brain_assigned": "phi3_lightweight",
                "tool_required": "nonexistent_tool",
                "task_type": "system_command",
                "step_id": "s1",
                "action_id": "a1",
                "goal": "do something",
            }
        ]
        state = make_state(sub_goals=sg)
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "TOOL_UNAVAILABLE" in result.approval_reason_codes

    def test_resource_gate_passed_false_on_unknown_tool(self):
        """resource_gate_passed must be False when any tool is unavailable."""
        sg = [
            {
                "brain_assigned": "phi3_lightweight",
                "tool_required": "nonexistent_tool",
                "task_type": "system_command",
                "step_id": "s1",
                "action_id": "a1",
                "goal": "do something",
            }
        ]
        state = make_state(sub_goals=sg)
        node = ValidationNode()
        result = run(node.process(state))
        assert result.resource_gate_passed is False

    def test_all_known_tools_pass(self):
        """Multiple sub_goals with all known tools → resource_gate_passed=True."""
        sg = [
            {
                "brain_assigned": "phi3_lightweight",
                "tool_required": "app_control",
                "task_type": "system_command",
                "step_id": "s1",
                "action_id": "a1",
                "goal": "open app",
            },
            {
                "brain_assigned": "phi3_lightweight",
                "tool_required": "screen_capture",
                "task_type": "screen_observation",
                "step_id": "s2",
                "action_id": "a2",
                "goal": "take screenshot",
            },
        ]
        state = make_state(sub_goals=sg)
        node = ValidationNode()
        result = run(node.process(state))
        assert result.resource_gate_passed is True


# ---------------------------------------------------------------------------
# TestApprovalGate
# ---------------------------------------------------------------------------


class TestApprovalGate:
    """Approval gate: high-risk task types and degraded-mode rules."""

    def test_high_risk_email_requires_approval(self):
        """task_type='email_compose' → approval_required, 'COMMUNICATION' in codes."""
        state = make_state(task_type="email_compose", tool_required="workspace_query")
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "COMMUNICATION" in result.approval_reason_codes

    def test_high_risk_file_delete(self):
        """task_type='file_delete' → 'DESTRUCTIVE_FILE_OP' in codes."""
        state = make_state(task_type="file_delete", tool_required="file_ops")
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "DESTRUCTIVE_FILE_OP" in result.approval_reason_codes

    def test_high_risk_payment(self):
        """task_type='payment' → 'FINANCIAL' in codes."""
        state = make_state(task_type="payment", tool_required="workspace_query")
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "FINANCIAL" in result.approval_reason_codes

    def test_high_risk_unlock(self):
        """task_type='unlock' → 'SECURITY' in codes."""
        state = make_state(task_type="unlock", tool_required="app_control")
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "SECURITY" in result.approval_reason_codes

    def test_high_risk_system_shutdown(self):
        """task_type='system_shutdown' → 'SYSTEM_DISRUPTION' in codes."""
        state = make_state(task_type="system_shutdown", tool_required="app_control")
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "SYSTEM_DISRUPTION" in result.approval_reason_codes

    def test_safe_task_no_approval(self):
        """task_type='system_command' (safe) → approval_required=False."""
        state = make_state(task_type="system_command", tool_required="app_control")
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is False

    def test_degraded_mode_heavy_requires_approval(self):
        """LEVEL_1_DEGRADED + complexity='heavy' → 'DEGRADED_HEAVY_TASK' in codes."""
        state = make_state(
            complexity="heavy",
            served_mode="LEVEL_1_DEGRADED",
            task_type="system_command",
            tool_required="app_control",
        )
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "DEGRADED_HEAVY_TASK" in result.approval_reason_codes

    def test_degraded_mode_complex_requires_approval(self):
        """LEVEL_1_DEGRADED + complexity='complex' → approval required."""
        state = make_state(
            complexity="complex",
            served_mode="LEVEL_1_DEGRADED",
            task_type="system_command",
            tool_required="app_control",
        )
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "DEGRADED_HEAVY_TASK" in result.approval_reason_codes

    def test_degraded_mode_trivial_passes(self):
        """LEVEL_1_DEGRADED + complexity='trivial' + safe task → no extra approval."""
        state = make_state(
            complexity="trivial",
            served_mode="LEVEL_1_DEGRADED",
            task_type="system_command",
            tool_required="app_control",
        )
        node = ValidationNode()
        result = run(node.process(state))
        # No approval reasons means no approval required
        assert result.approval_required is False

    def test_degraded_mode_light_passes(self):
        """LEVEL_1_DEGRADED + complexity='light' + safe task → no extra approval."""
        state = make_state(
            complexity="light",
            served_mode="LEVEL_1_DEGRADED",
            task_type="system_command",
            tool_required="app_control",
        )
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is False

    def test_multiple_risk_codes_accumulated(self):
        """High-risk task type in degraded mode → both reason codes present."""
        state = make_state(
            complexity="heavy",
            served_mode="LEVEL_1_DEGRADED",
            task_type="email_compose",
            tool_required="workspace_query",
        )
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "COMMUNICATION" in result.approval_reason_codes
        assert "DEGRADED_HEAVY_TASK" in result.approval_reason_codes


# ---------------------------------------------------------------------------
# TestFailClosed
# ---------------------------------------------------------------------------


class TestFailClosed:
    """Fail-closed invariant: any internal error → needs_approval + VALIDATION_UNAVAILABLE."""

    def test_exception_returns_validation_unavailable(self):
        """Malformed sub_goals (string instead of list) → VALIDATION_UNAVAILABLE."""
        # Construct state normally then inject bad sub_goals via model_copy
        base = ReasoningGraphState(
            request_id="r1",
            session_id="s1",
            trace_id="t1",
            command="test",
        )
        # Force sub_goals to a string (breaks iteration) at the dict level
        bad_state = base.model_copy(update={"sub_goals": "not_a_list"})  # type: ignore[arg-type]
        node = ValidationNode()
        result = run(node.process(bad_state))
        assert result.approval_required is True
        assert "VALIDATION_UNAVAILABLE" in result.approval_reason_codes

    def test_none_sub_goals_fail_closed(self):
        """sub_goals=None (injected) → VALIDATION_UNAVAILABLE."""
        base = ReasoningGraphState(
            request_id="r1",
            session_id="s1",
            trace_id="t1",
            command="test",
        )
        bad_state = base.model_copy(update={"sub_goals": None})  # type: ignore[arg-type]
        node = ValidationNode()
        result = run(node.process(bad_state))
        assert result.approval_required is True
        assert "VALIDATION_UNAVAILABLE" in result.approval_reason_codes

    def test_fail_closed_never_returns_plan_ready_on_error(self):
        """When validation fails internally, approval_required must be True."""
        base = ReasoningGraphState(
            request_id="r1",
            session_id="s1",
            trace_id="t1",
            command="test",
        )
        bad_state = base.model_copy(update={"sub_goals": {"not": "a list"}})  # type: ignore[arg-type]
        node = ValidationNode()
        result = run(node.process(bad_state))
        # The critical invariant: NEVER plan_ready when validation fails
        assert result.approval_required is True

    def test_fail_closed_trace_still_appended_on_error(self):
        """Even on internal error, a trace entry must be appended."""
        base = ReasoningGraphState(
            request_id="r1",
            session_id="s1",
            trace_id="t1",
            command="test",
        )
        bad_state = base.model_copy(update={"sub_goals": "not_a_list"})  # type: ignore[arg-type]
        node = ValidationNode()
        result = run(node.process(bad_state))
        assert len(result.reasoning_trace) >= 1
        assert result.reasoning_trace[-1]["node"] == "validation"


# ---------------------------------------------------------------------------
# TestLightValidation
# ---------------------------------------------------------------------------


class TestLightValidation:
    """Validation runs on ALL plan paths — trivial complexity is NOT skipped."""

    def test_trivial_command_still_validated(self):
        """Trivial safe task → cost_gate_passed=True (validation ran)."""
        state = make_state(
            complexity="trivial",
            task_type="system_command",
            tool_required="app_control",
        )
        node = ValidationNode()
        result = run(node.process(state))
        # cost_gate ran and passed
        assert result.cost_gate_passed is True
        # resource_gate ran and passed
        assert result.resource_gate_passed is True
        # No approval needed for safe trivial task
        assert result.approval_required is False

    def test_trivial_high_risk_caught(self):
        """Trivial complexity + high-risk task_type → approval still required."""
        state = make_state(
            complexity="trivial",
            task_type="file_delete",
            tool_required="file_ops",
        )
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is True
        assert "DESTRUCTIVE_FILE_OP" in result.approval_reason_codes

    def test_light_safe_passes(self):
        """Light complexity + safe task → no approval needed."""
        state = make_state(
            complexity="light",
            task_type="system_command",
            tool_required="app_control",
        )
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is False

    def test_empty_sub_goals_still_validates(self):
        """Empty sub_goals list → gates run (no violations), no approval needed."""
        state = make_state_no_subgoals()
        node = ValidationNode()
        result = run(node.process(state))
        assert result.approval_required is False
        assert result.cost_gate_passed is True
        assert result.resource_gate_passed is True


# ---------------------------------------------------------------------------
# TestTrace
# ---------------------------------------------------------------------------


class TestTrace:
    """reasoning_trace must receive one entry with node='validation'."""

    def test_reasoning_trace_appended(self):
        """On success, one trace entry with node='validation' is appended."""
        state = make_state()
        node = ValidationNode()
        result = run(node.process(state))
        assert len(result.reasoning_trace) == 1
        assert result.reasoning_trace[0]["node"] == "validation"

    def test_trace_accumulated_with_existing_entries(self):
        """When state already has trace entries, new entry is appended (not replaced)."""
        state = make_state()
        state = state.model_copy(
            update={"reasoning_trace": [{"node": "analysis", "elapsed_ms": 10}, {"node": "planning", "elapsed_ms": 20}]}
        )
        node = ValidationNode()
        result = run(node.process(state))
        assert len(result.reasoning_trace) == 3
        assert result.reasoning_trace[-1]["node"] == "validation"

    def test_trace_entry_has_elapsed_ms(self):
        """Trace entry must contain elapsed_ms field."""
        state = make_state()
        node = ValidationNode()
        result = run(node.process(state))
        assert "elapsed_ms" in result.reasoning_trace[0]

    def test_trace_entry_has_approval_required(self):
        """Trace entry must contain approval_required field."""
        state = make_state()
        node = ValidationNode()
        result = run(node.process(state))
        assert "approval_required" in result.reasoning_trace[0]

    def test_trace_entry_has_reason_codes(self):
        """Trace entry must contain reason_codes field."""
        state = make_state()
        node = ValidationNode()
        result = run(node.process(state))
        assert "reason_codes" in result.reasoning_trace[0]

    def test_trace_node_field_is_validation_on_error(self):
        """Even on internal error, trace entry node must be 'validation'."""
        base = ReasoningGraphState(
            request_id="r1",
            session_id="s1",
            trace_id="t1",
            command="test",
        )
        bad_state = base.model_copy(update={"sub_goals": "broken"})  # type: ignore[arg-type]
        node = ValidationNode()
        result = run(node.process(bad_state))
        assert result.reasoning_trace[-1]["node"] == "validation"
