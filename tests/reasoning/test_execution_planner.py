"""
TDD tests for ExecutionPlanner — final node in the J-Prime unified reasoning pipeline.

Written BEFORE implementation. All tests must fail with ImportError initially,
then pass once execution_planner.py is implemented.

Tests cover:
- Plan assembly: plan_id, plan_hash, sub_goals as SubGoal objects
- plan_hash is a full 64-char SHA-256 hex digest
- plan_hash is deterministic (same input → same hash)
- Empty sub_goals still produce a valid Plan
- Brain and tool assignment for each sub_goal
- Auto-generation of a sub_goal when state.sub_goals is empty
- Auto-generated sub_goal has all required fields
- reasoning_trace entry with node="execution_planner"
- phase == "assembling"
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List

import pytest

from jarvis_prime.reasoning.protocol import ReasoningGraphState, SubGoal
from jarvis_prime.reasoning.unified_brain_selector import UnifiedBrainSelector
from jarvis_prime.reasoning.graph_nodes.execution_planner import ExecutionPlanner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_SUB_GOAL_FIELDS = {
    "step_id",
    "action_id",
    "goal",
    "task_type",
    "brain_assigned",
    "tool_required",
    "depends_on",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def make_state(
    command: str = "open Safari",
    intent: str = "system_command",
    complexity: str = "trivial",
    sub_goals: List[Dict[str, Any]] = None,
) -> ReasoningGraphState:
    return ReasoningGraphState(
        request_id="r1",
        session_id="s1",
        trace_id="t1",
        command=command,
        intent=intent,
        complexity=complexity,
        sub_goals=sub_goals if sub_goals is not None else [
            {
                "step_id": "s1",
                "action_id": "a1",
                "goal": "open Safari",
                "task_type": "system_command",
                "brain_assigned": "",
                "tool_required": "",
                "depends_on": [],
            }
        ],
    )


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# TestPlanAssembly
# ---------------------------------------------------------------------------


class TestPlanAssembly:
    """Plan is built from state.sub_goals, carrying plan_id and plan_hash."""

    def test_builds_plan_from_sub_goals(self):
        node = ExecutionPlanner()
        state = run(node.process(make_state()))
        plan = state.context.get("plan")
        assert plan is not None
        assert plan["plan_id"] != ""
        assert plan["plan_hash"] != ""
        sub_goals = plan["sub_goals"]
        assert len(sub_goals) >= 1

    def test_plan_hash_is_full_sha256(self):
        node = ExecutionPlanner()
        state = run(node.process(make_state()))
        plan_hash = state.context["plan"]["plan_hash"]
        assert _SHA256_RE.match(plan_hash), (
            f"plan_hash must be 64 hex chars, got {plan_hash!r}"
        )

    def test_plan_hash_deterministic(self):
        """Same sub_goals content → same hash on two separate runs."""
        sub_goals = [
            {
                "step_id": "s1",
                "action_id": "a1",
                "goal": "open Safari",
                "task_type": "system_command",
                "brain_assigned": "",
                "tool_required": "",
                "depends_on": [],
            }
        ]
        node1 = ExecutionPlanner()
        node2 = ExecutionPlanner()
        state1 = run(node1.process(make_state(sub_goals=sub_goals)))
        state2 = run(node2.process(make_state(sub_goals=sub_goals)))
        # Hashes must match (plan_id will differ — that is expected)
        assert state1.context["plan"]["plan_hash"] == state2.context["plan"]["plan_hash"]

    def test_empty_sub_goals_still_produces_plan(self):
        """Auto-generation path: empty sub_goals → 1 auto sub_goal, valid Plan."""
        node = ExecutionPlanner()
        state = run(node.process(make_state(sub_goals=[])))
        plan = state.context.get("plan")
        assert plan is not None
        assert plan["plan_id"] != ""
        assert plan["plan_hash"] != ""


# ---------------------------------------------------------------------------
# TestBrainAssignment
# ---------------------------------------------------------------------------


class TestBrainAssignment:
    """Each sub_goal gets non-empty brain_assigned and tool_required."""

    def test_each_sub_goal_gets_brain_assigned(self):
        sub_goals = [
            {
                "step_id": "s1",
                "action_id": "a1",
                "goal": "open Safari",
                "task_type": "system_command",
                "brain_assigned": "",
                "tool_required": "",
                "depends_on": [],
            },
            {
                "step_id": "s2",
                "action_id": "a2",
                "goal": "navigate to google.com",
                "task_type": "browser_navigation",
                "brain_assigned": "",
                "tool_required": "",
                "depends_on": ["s1"],
            },
        ]
        node = ExecutionPlanner()
        state = run(node.process(make_state(sub_goals=sub_goals)))
        plan_sub_goals = state.context["plan"]["sub_goals"]
        for sg in plan_sub_goals:
            brain = sg["brain_assigned"] if isinstance(sg, dict) else sg.brain_assigned
            assert brain != "", f"brain_assigned must not be empty: {sg}"

    def test_each_sub_goal_gets_tool_required(self):
        sub_goals = [
            {
                "step_id": "s1",
                "action_id": "a1",
                "goal": "click submit button",
                "task_type": "vision_action",
                "brain_assigned": "",
                "tool_required": "",
                "depends_on": [],
            },
        ]
        node = ExecutionPlanner()
        state = run(node.process(make_state(sub_goals=sub_goals)))
        plan_sub_goals = state.context["plan"]["sub_goals"]
        for sg in plan_sub_goals:
            tool = sg["tool_required"] if isinstance(sg, dict) else sg.tool_required
            assert tool != "", f"tool_required must not be empty: {sg}"


# ---------------------------------------------------------------------------
# TestAutoGeneration
# ---------------------------------------------------------------------------


class TestAutoGeneration:
    """When sub_goals is empty, a single sub_goal is auto-generated from command."""

    def test_trivial_command_auto_generates_sub_goal(self):
        node = ExecutionPlanner()
        state = run(node.process(make_state(
            command="open Safari",
            intent="system_command",
            sub_goals=[],
        )))
        plan_sub_goals = state.context["plan"]["sub_goals"]
        assert len(plan_sub_goals) == 1

    def test_auto_generated_has_required_fields(self):
        node = ExecutionPlanner()
        state = run(node.process(make_state(
            command="open Safari",
            intent="system_command",
            sub_goals=[],
        )))
        plan_sub_goals = state.context["plan"]["sub_goals"]
        sg = plan_sub_goals[0]
        # Support both dict and SubGoal Pydantic object
        if isinstance(sg, dict):
            for field in REQUIRED_SUB_GOAL_FIELDS:
                assert field in sg, f"Auto-generated sub_goal missing field: {field!r}"
            assert sg["step_id"] != ""
            assert sg["action_id"] != ""
            assert sg["goal"] != ""
            assert sg["task_type"] != ""
            assert sg["brain_assigned"] != ""
            assert sg["tool_required"] != ""
            assert isinstance(sg["depends_on"], list)
        else:
            assert sg.step_id != ""
            assert sg.action_id != ""
            assert sg.goal != ""
            assert sg.task_type != ""
            assert sg.brain_assigned != ""
            assert sg.tool_required != ""
            assert isinstance(sg.depends_on, list)


# ---------------------------------------------------------------------------
# TestTrace
# ---------------------------------------------------------------------------


class TestTrace:
    """reasoning_trace receives one entry with node="execution_planner"."""

    def test_reasoning_trace_appended(self):
        node = ExecutionPlanner()
        state = run(node.process(make_state()))
        assert len(state.reasoning_trace) == 1
        assert state.reasoning_trace[0]["node"] == "execution_planner"

    def test_trace_accumulated_with_existing_entries(self):
        """When state already has trace entries, new entry is appended."""
        node = ExecutionPlanner()
        initial = make_state()
        initial = initial.model_copy(
            update={
                "reasoning_trace": [
                    {"node": "analysis", "elapsed_ms": 10},
                    {"node": "planning", "elapsed_ms": 20},
                ]
            }
        )
        state = run(node.process(initial))
        assert len(state.reasoning_trace) == 3
        assert state.reasoning_trace[2]["node"] == "execution_planner"

    def test_trace_has_elapsed_ms(self):
        node = ExecutionPlanner()
        state = run(node.process(make_state()))
        assert "elapsed_ms" in state.reasoning_trace[0]

    def test_phase_set(self):
        node = ExecutionPlanner()
        state = run(node.process(make_state()))
        assert state.phase == "assembling"
