"""
ExecutionPlanner — final node in the J-Prime unified reasoning pipeline.

Assembles a complete execution Plan from the sub_goals produced by
PlanningNode, filling in brain_assigned and tool_required via
UnifiedBrainSelector, converting dicts to SubGoal Pydantic objects,
and stamping the Plan with a unique plan_id and a canonical plan_hash.

Design rules
------------
- No ModelProvider dependency: this node is purely deterministic logic.
- If state.sub_goals is empty, auto-generate a single sub_goal derived
  from state.command and state.intent so the pipeline never produces a
  Plan with zero steps.
- Brain assignment uses UnifiedBrainSelector.select(task_type, goal).
- Tool inference uses the same _TOOL_MAP as PlanningNode (no duplication
  of business logic in the tables themselves; the map is local here and
  kept in sync by convention).
- Plan is stored as a dict on state.context["plan"] so _build_response
  in ReasoningGraph can consume it without an extra import cycle.
- state immutability is preserved via model_copy(update=...) (Pydantic v2).
- One reasoning_trace entry is always appended with node="execution_planner".
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from jarvis_prime.reasoning.protocol import Plan, ReasoningGraphState, SubGoal, compute_plan_hash
from jarvis_prime.reasoning.unified_brain_selector import UnifiedBrainSelector


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# task_type → Body tool name  (mirrors PlanningNode._TOOL_MAP; extended here
# to cover vision_verification and proactive_narration).
_TOOL_MAP: Dict[str, str] = {
    "system_command": "app_control",
    "browser_navigation": "visual_browser",
    "vision_action": "computer_use",
    "vision_verification": "computer_use",
    "email_compose": "workspace_query",
    "email_triage": "workspace_query",
    "calendar_query": "workspace_query",
    "screen_observation": "screen_capture",
    "proactive_narration": "screen_capture",
}

# Fallback tool when task_type is absent from the map
_DEFAULT_TOOL = "app_control"

# Fallback task_type for auto-generated sub_goals when intent is unknown
_DEFAULT_TASK_TYPE = "system_command"


# ---------------------------------------------------------------------------
# ExecutionPlanner
# ---------------------------------------------------------------------------


class ExecutionPlanner:
    """
    Final node in the J-Prime reasoning pipeline.

    Parameters
    ----------
    brain_selector:
        Optional UnifiedBrainSelector.  A default instance is created if
        not supplied so that brain_assigned is always populated.
    """

    def __init__(self, brain_selector: Optional[UnifiedBrainSelector] = None) -> None:
        self._selector: UnifiedBrainSelector = brain_selector or UnifiedBrainSelector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, state: ReasoningGraphState) -> ReasoningGraphState:
        """
        Assemble a Plan and return an updated ReasoningGraphState.

        Always sets:
          - phase = "assembling"
          - context["plan"] (dict representation of Plan)
          - reasoning_trace += [{"node": "execution_planner", ...}]

        Steps
        -----
        1. If state.sub_goals is empty → auto-generate one sub_goal from
           state.command + state.intent.
        2. For each sub_goal dict: call selector.select() to fill
           brain_assigned and tool_required when they are empty strings.
        3. Convert sub_goal dicts to SubGoal Pydantic objects.
        4. Build Plan(plan_id, sub_goals, execution_strategy, …).
        5. Compute plan_hash via compute_plan_hash(plan) and store it.
        6. Persist plan as dict in state.context["plan"].
        7. Append trace entry; return via model_copy(update=…).
        """
        t_start = time.monotonic()

        # Step 1: ensure we have at least one sub_goal
        raw_sub_goals: List[Dict[str, Any]] = list(state.sub_goals)
        if not raw_sub_goals:
            raw_sub_goals = self._auto_generate_sub_goal(state)

        # Step 2 + 3: fill missing brain/tool and convert to SubGoal objects
        sub_goal_objects: List[SubGoal] = []
        for sg_dict in raw_sub_goals:
            enriched = self._enrich(sg_dict)
            sub_goal_objects.append(SubGoal(**enriched))

        # Step 4: build Plan
        plan = Plan(
            plan_id=f"plan-{uuid.uuid4().hex[:8]}",
            sub_goals=sub_goal_objects,
            execution_strategy=state.execution_strategy or "sequential",
            approval_required=state.approval_required,
            approval_reason_codes=list(state.approval_reason_codes),
            risk_level=state.risk_level,
        )

        # Step 5: compute and attach canonical hash
        plan_hash = compute_plan_hash(plan)
        plan = plan.model_copy(update={"plan_hash": plan_hash})

        # Step 6: serialize to dict for storage on state.context
        plan_dict = plan.model_dump()

        # Step 7: build trace entry
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        trace_entry: Dict[str, Any] = {
            "node": "execution_planner",
            "plan_id": plan.plan_id,
            "plan_hash": plan_hash,
            "sub_goal_count": len(sub_goal_objects),
            "execution_strategy": plan.execution_strategy,
            "elapsed_ms": elapsed_ms,
        }

        # Build updated context (immutable merge)
        updated_context = {**state.context, "plan": plan_dict}

        updated = state.model_copy(
            update={
                "phase": "assembling",
                "context": updated_context,
                "reasoning_trace": state.reasoning_trace + [trace_entry],
            }
        )
        return updated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_generate_sub_goal(self, state: ReasoningGraphState) -> List[Dict[str, Any]]:
        """
        Produce a single sub_goal dict derived from command + intent.

        Used when no sub_goals were produced by PlanningNode (e.g. trivial
        commands that bypass full planning).
        """
        task_type = state.intent if state.intent else _DEFAULT_TASK_TYPE
        goal_text = state.command.strip() or f"execute {task_type.replace('_', ' ')}"
        selection = self._selector.select(task_type, goal_text)
        return [
            {
                "step_id": "s1",
                "action_id": str(uuid.uuid4()),
                "goal": goal_text,
                "task_type": task_type,
                "brain_assigned": selection.brain_id,
                "tool_required": _TOOL_MAP.get(task_type, _DEFAULT_TOOL),
                "depends_on": [],
            }
        ]

    def _enrich(self, sg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a copy of *sg* with brain_assigned and tool_required filled.

        If either field is already non-empty it is left unchanged so that
        upstream assignments from PlanningNode are preserved.
        """
        task_type = str(sg.get("task_type", _DEFAULT_TASK_TYPE)).strip() or _DEFAULT_TASK_TYPE
        goal_text = str(sg.get("goal", "")).strip()

        brain_assigned = str(sg.get("brain_assigned", "")).strip()
        tool_required = str(sg.get("tool_required", "")).strip()

        if not brain_assigned:
            selection = self._selector.select(task_type, goal_text)
            brain_assigned = selection.brain_id

        if not tool_required:
            tool_required = _TOOL_MAP.get(task_type, _DEFAULT_TOOL)

        return {
            "step_id": str(sg.get("step_id", "")).strip() or "s1",
            "action_id": str(sg.get("action_id", "")).strip() or str(uuid.uuid4()),
            "goal": goal_text,
            "task_type": task_type,
            "brain_assigned": brain_assigned,
            "tool_required": tool_required,
            "depends_on": list(sg.get("depends_on", [])),
            "estimated_ms": int(sg.get("estimated_ms", 0)),
        }
