"""
TDD tests for PlanningNode — second node in the J-Prime unified reasoning pipeline.

Written BEFORE implementation. All tests must fail with ImportError initially,
then pass once planning_node.py is implemented.

Tests cover:
- Happy-path model response → state.sub_goals populated, planning_brain_used set
- Model failure → LEVEL_1_DEGRADED, keyword fallback produces sub_goals
- Keyword fallback splits on conjunctions ("and", "then", commas, semicolons)
- Single-goal command → exactly 1 sub-goal
- execution_strategy is one of "sequential", "parallel", "dag"
- reasoning_trace receives exactly one entry with node="planning"
- phase is set to "planning" after process()
- Each sub-goal dict has required fields: step_id, action_id, goal, task_type,
  brain_assigned, tool_required, depends_on
"""
from __future__ import annotations

import asyncio
from typing import List

import pytest

from jarvis_prime.reasoning.model_provider import MockModelProvider
from jarvis_prime.reasoning.protocol import ReasoningGraphState
from jarvis_prime.reasoning.unified_brain_selector import UnifiedBrainSelector
from jarvis_prime.reasoning.graph_nodes.planning_node import PlanningNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_STRATEGIES = {"sequential", "parallel", "dag"}
REQUIRED_SUB_GOAL_FIELDS = {
    "step_id",
    "action_id",
    "goal",
    "task_type",
    "brain_assigned",
    "tool_required",
    "depends_on",
}

_HAPPY_RESPONSE = (
    '{"sub_goals": ['
    '{"goal": "open browser and go to linkedin.com", "task_type": "browser_navigation"}, '
    '{"goal": "search for competitor profiles", "task_type": "vision_action"}'
    '], "execution_strategy": "sequential"}'
)


def make_state(
    command: str = "research competitors and build spreadsheet",
    inferred_goals: List[str] = None,
) -> ReasoningGraphState:
    return ReasoningGraphState(
        request_id="req-plan-001",
        session_id="sess-plan-001",
        trace_id="trace-plan-001",
        command=command,
        inferred_goals=inferred_goals or [],
    )


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# TestSuccessfulDecomposition
# ---------------------------------------------------------------------------


class TestSuccessfulDecomposition:
    """Happy-path: model returns valid JSON with sub_goals list."""

    def test_sub_goals_populated(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert len(state.sub_goals) >= 1

    def test_planning_brain_used_set(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.planning_brain_used != ""

    def test_sub_goals_count_matches_model_output(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert len(state.sub_goals) == 2

    def test_served_mode_stays_primary(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.served_mode == "LEVEL_0_PRIMARY"

    def test_degraded_reason_code_none_on_success(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.degraded_reason_code is None


# ---------------------------------------------------------------------------
# TestModelFailureDegrades
# ---------------------------------------------------------------------------


class TestModelFailureDegrades:
    """Model raises RuntimeError → state must degrade, keyword fallback fires."""

    def test_served_mode_degraded(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("research competitors and build spreadsheet")))
        assert state.served_mode == "LEVEL_1_DEGRADED"

    def test_confidence_capped_at_0_6(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("research competitors and build spreadsheet")))
        assert state.confidence <= 0.6

    def test_sub_goals_still_produced(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("research competitors and build spreadsheet")))
        assert len(state.sub_goals) >= 1

    def test_degraded_reason_code_set(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.degraded_reason_code is not None
        assert len(state.degraded_reason_code) > 0


# ---------------------------------------------------------------------------
# TestKeywordFallbackSplitsOnConjunctions
# ---------------------------------------------------------------------------


class TestKeywordFallbackSplitsOnConjunctions:
    """Keyword fallback splits on 'and', 'then', commas, semicolons."""

    def test_and_conjunction_gives_two_goals(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("research competitors and build spreadsheet")))
        assert len(state.sub_goals) >= 2

    def test_then_conjunction_gives_two_goals(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("open Safari then go to linkedin.com")))
        assert len(state.sub_goals) >= 2

    def test_comma_gives_multiple_goals(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("open browser, navigate to site, take screenshot")))
        assert len(state.sub_goals) >= 2

    def test_semicolon_gives_multiple_goals(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("open terminal; run tests; commit changes")))
        assert len(state.sub_goals) >= 2

    def test_inferred_goals_used_when_multiple(self):
        """If inferred_goals has multiple entries, those are used as sub-goals."""
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state(
            command="do complex task",
            inferred_goals=["step one", "step two", "step three"],
        )))
        assert len(state.sub_goals) >= 3


# ---------------------------------------------------------------------------
# TestSingleGoalCommand
# ---------------------------------------------------------------------------


class TestSingleGoalCommand:
    """A simple command with no conjunctions → exactly 1 sub-goal."""

    def test_single_goal_on_success(self):
        single_response = (
            '{"sub_goals": [{"goal": "launch Safari browser", "task_type": "system_command"}],'
            ' "execution_strategy": "sequential"}'
        )
        provider = MockModelProvider(response=single_response)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert len(state.sub_goals) == 1

    def test_single_goal_fallback_no_conjunctions(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert len(state.sub_goals) >= 1


# ---------------------------------------------------------------------------
# TestExecutionStrategySet
# ---------------------------------------------------------------------------


class TestExecutionStrategySet:
    """execution_strategy must be 'sequential', 'parallel', or 'dag'."""

    def test_execution_strategy_valid_on_success(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.execution_strategy in VALID_STRATEGIES

    def test_execution_strategy_sequential_from_model(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.execution_strategy == "sequential"

    def test_execution_strategy_valid_on_fallback(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.execution_strategy in VALID_STRATEGIES

    def test_execution_strategy_parallel_from_model(self):
        parallel_response = (
            '{"sub_goals": [{"goal": "fetch data A", "task_type": "browser_navigation"},'
            '{"goal": "fetch data B", "task_type": "browser_navigation"}],'
            ' "execution_strategy": "parallel"}'
        )
        provider = MockModelProvider(response=parallel_response)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("fetch data A and data B in parallel")))
        assert state.execution_strategy == "parallel"


# ---------------------------------------------------------------------------
# TestReasoningTraceAppended
# ---------------------------------------------------------------------------


class TestReasoningTraceAppended:
    """reasoning_trace must have exactly one entry with node="planning"."""

    def test_trace_has_one_entry_on_success(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert len(state.reasoning_trace) == 1

    def test_trace_node_field_is_planning_on_success(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.reasoning_trace[0]["node"] == "planning"

    def test_trace_has_one_entry_on_failure(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert len(state.reasoning_trace) == 1

    def test_trace_node_field_is_planning_on_failure(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.reasoning_trace[0]["node"] == "planning"

    def test_trace_entry_has_elapsed_ms(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert "elapsed_ms" in state.reasoning_trace[0]

    def test_trace_accumulated_with_existing_entries(self):
        """When state already has trace entries, the new entry is appended."""
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        initial_state = make_state()
        # Pre-seed a trace entry (simulates AnalysisNode having run before)
        initial_state = initial_state.model_copy(
            update={"reasoning_trace": [{"node": "analysis", "elapsed_ms": 42}]}
        )
        state = run(node.process(initial_state))
        assert len(state.reasoning_trace) == 2
        assert state.reasoning_trace[1]["node"] == "planning"


# ---------------------------------------------------------------------------
# TestPhaseSet
# ---------------------------------------------------------------------------


class TestPhaseSet:
    """phase must be "planning" after process() regardless of outcome."""

    def test_phase_planning_on_success(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.phase == "planning"

    def test_phase_planning_on_model_failure(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.phase == "planning"


# ---------------------------------------------------------------------------
# TestSubGoalsHaveRequiredFields
# ---------------------------------------------------------------------------


class TestSubGoalsHaveRequiredFields:
    """Every sub-goal dict must contain all required fields."""

    def test_all_required_fields_present_on_success(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        for sg in state.sub_goals:
            missing = REQUIRED_SUB_GOAL_FIELDS - sg.keys()
            assert not missing, f"Sub-goal missing fields: {missing}"

    def test_all_required_fields_present_on_fallback(self):
        provider = MockModelProvider(should_fail=True)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("research competitors and build spreadsheet")))
        for sg in state.sub_goals:
            missing = REQUIRED_SUB_GOAL_FIELDS - sg.keys()
            assert not missing, f"Sub-goal missing fields: {missing}"

    def test_step_ids_are_sequential(self):
        """step_id values should be s1, s2, s3, ... (1-based)."""
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        for i, sg in enumerate(state.sub_goals, start=1):
            assert sg["step_id"] == f"s{i}", (
                f"Expected step_id s{i}, got {sg['step_id']!r}"
            )

    def test_action_ids_are_unique_uuids(self):
        """action_id for each sub-goal must be a non-empty unique string."""
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        action_ids = [sg["action_id"] for sg in state.sub_goals]
        assert all(aid != "" for aid in action_ids)
        assert len(set(action_ids)) == len(action_ids), "action_ids must be unique"

    def test_depends_on_is_list(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        for sg in state.sub_goals:
            assert isinstance(sg["depends_on"], list)

    def test_brain_assigned_non_empty(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        for sg in state.sub_goals:
            assert sg["brain_assigned"] != ""

    def test_tool_required_non_empty(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        for sg in state.sub_goals:
            assert sg["tool_required"] != ""


# ---------------------------------------------------------------------------
# TestToolInference
# ---------------------------------------------------------------------------


class TestToolInference:
    """tool_required must be inferred correctly from task_type."""

    def test_browser_navigation_maps_to_visual_browser(self):
        response = (
            '{"sub_goals": [{"goal": "go to site", "task_type": "browser_navigation"}],'
            ' "execution_strategy": "sequential"}'
        )
        provider = MockModelProvider(response=response)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("go to site")))
        assert state.sub_goals[0]["tool_required"] == "visual_browser"

    def test_vision_action_maps_to_computer_use(self):
        response = (
            '{"sub_goals": [{"goal": "click button", "task_type": "vision_action"}],'
            ' "execution_strategy": "sequential"}'
        )
        provider = MockModelProvider(response=response)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("click button")))
        assert state.sub_goals[0]["tool_required"] == "computer_use"

    def test_system_command_maps_to_app_control(self):
        response = (
            '{"sub_goals": [{"goal": "open app", "task_type": "system_command"}],'
            ' "execution_strategy": "sequential"}'
        )
        provider = MockModelProvider(response=response)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("open app")))
        assert state.sub_goals[0]["tool_required"] == "app_control"

    def test_unknown_task_type_defaults_to_app_control(self):
        response = (
            '{"sub_goals": [{"goal": "do something", "task_type": "totally_unknown_type"}],'
            ' "execution_strategy": "sequential"}'
        )
        provider = MockModelProvider(response=response)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state("do something")))
        assert state.sub_goals[0]["tool_required"] == "app_control"


# ---------------------------------------------------------------------------
# TestBrainSelectorIntegration
# ---------------------------------------------------------------------------


class TestBrainSelectorIntegration:
    """PlanningNode should accept an optional UnifiedBrainSelector."""

    def test_accepts_custom_brain_selector(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        selector = UnifiedBrainSelector()
        node = PlanningNode(model_provider=provider, brain_selector=selector)
        state = run(node.process(make_state()))
        assert state.planning_brain_used != ""

    def test_planning_brain_used_set_without_custom_selector(self):
        provider = MockModelProvider(response=_HAPPY_RESPONSE)
        node = PlanningNode(model_provider=provider)
        state = run(node.process(make_state()))
        assert state.planning_brain_used != ""
