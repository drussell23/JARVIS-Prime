"""
PlanningNode — second node in the J-Prime unified reasoning pipeline.

Decomposes a command into an ordered list of executable sub-goals by calling
a ModelProvider (GPU model) for structured JSON output. Falls back to a
deterministic keyword-splitting strategy when the model call fails or returns
non-parseable text.

Design rules
------------
- Never import from server.py. Accepts ModelProvider via constructor DI.
- json.loads() for model output; catches JSONDecodeError explicitly.
- Pattern-fallback confidence is capped at 0.6 (spec §3).
- Degraded path sets served_mode=LEVEL_1_DEGRADED and degraded_reason_code.
- Each sub-goal is enriched with: step_id (s1, s2, ...), action_id (uuid4),
  brain_assigned (from selector), tool_required (inferred from task_type),
  depends_on (empty list for sequential/parallel; filled for dag if provided).
- One reasoning_trace entry is always appended with node="planning".
- state immutability is preserved via model_copy(update=...) (Pydantic v2).
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from jarvis_prime.reasoning.model_provider import ModelProvider
from jarvis_prime.reasoning.protocol import ReasoningGraphState
from jarvis_prime.reasoning.unified_brain_selector import UnifiedBrainSelector


# ---------------------------------------------------------------------------
# Constants — no hardcoded business logic outside these tables
# ---------------------------------------------------------------------------

# Valid execution strategy values
_VALID_STRATEGIES = frozenset({"sequential", "parallel", "dag"})

# Default execution strategy when missing or unrecognised
_DEFAULT_STRATEGY = "sequential"

# Cap for confidence in degraded / keyword-fallback mode
_DEGRADED_CONFIDENCE_CAP = 0.6

# task_type → Body tool name
_TOOL_MAP: Dict[str, str] = {
    "system_command": "app_control",
    "browser_navigation": "visual_browser",
    "vision_action": "computer_use",
    "email_compose": "workspace_query",
    "calendar_query": "workspace_query",
    "screen_observation": "screen_capture",
}

# Default tool when task_type is not in the map
_DEFAULT_TOOL = "app_control"

# task_type keywords for inference during keyword fallback
# Ordered from most-specific to most-generic
_TASK_TYPE_KEYWORDS: List[Tuple[str, List[str]]] = [
    ("browser_navigation", ["navigate", "go to", "browse", "visit", "open website",
                             "open url", "open tab", "search", "linkedin", "google",
                             "website", ".com", ".org", ".net"]),
    ("vision_action", ["click", "type", "fill", "submit", "scroll", "screenshot"]),
    ("email_compose", ["email", "send", "reply", "draft", "compose", "write to",
                        "message to"]),
    ("calendar_query", ["calendar", "meeting", "schedule", "appointment", "event",
                         "remind"]),
    ("screen_observation", ["screen", "window", "observe", "watch", "monitor"]),
    ("system_command", ["open", "close", "launch", "quit", "exit", "volume",
                         "brightness", "lock", "unlock", "restart", "shutdown",
                         "run", "execute", "terminal", "app", "application"]),
]

# Default task_type when no keyword matches
_DEFAULT_TASK_TYPE = "general_query"

# Conjunction / delimiter patterns used to split a command into sub-goals
# Ordered from most-specific to least-specific
_SPLIT_PATTERN = re.compile(
    r"\s+and\s+|\s+then\s+|;\s*|,\s*",
    re.IGNORECASE,
)

# System prompt for the GPU model
_SYSTEM_PROMPT = (
    "You are JARVIS's planning engine. Given a command and its inferred goals,\n"
    "decompose into executable sub-goals. Return JSON:\n"
    '- "sub_goals": list of {"goal": "...", "task_type": "..."}\n'
    '- "execution_strategy": "sequential" | "parallel" | "dag"\n'
    "Only return valid JSON."
)


# ---------------------------------------------------------------------------
# PlanningNode
# ---------------------------------------------------------------------------


class PlanningNode:
    """
    Second node in the J-Prime reasoning pipeline.

    Parameters
    ----------
    model_provider:
        Any object satisfying the ModelProvider protocol.
    brain_selector:
        Optional UnifiedBrainSelector. A default instance is created if not
        supplied so that ``planning_brain_used`` is always populated.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        brain_selector: Optional[UnifiedBrainSelector] = None,
    ) -> None:
        self._model = model_provider
        self._selector: UnifiedBrainSelector = brain_selector or UnifiedBrainSelector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, state: ReasoningGraphState) -> ReasoningGraphState:
        """
        Decompose *state.command* into sub-goals and return an updated
        ReasoningGraphState.

        Always sets:
          - phase = "planning"
          - execution_strategy (from model or default)
          - reasoning_trace += [{"node": "planning", ...}]

        On success (model returns valid JSON with sub_goals list):
          - sub_goals enriched with step_id, action_id, brain_assigned,
            tool_required, depends_on
          - planning_brain_used set
          - served_mode stays LEVEL_0_PRIMARY

        On failure (model error OR JSON parse error OR missing sub_goals):
          - Delegates to _keyword_fallback()
          - served_mode = LEVEL_1_DEGRADED
          - confidence <= 0.6
          - degraded_reason_code set
        """
        t_start = time.monotonic()

        # Phase 1: brain selection (never raises — UnifiedBrainSelector is pure)
        selection = self._selector.select("multi_step_planning", state.command)

        # Phase 2: attempt GPU model decomposition
        model_result: Optional[Dict[str, Any]] = None
        degraded = False
        degraded_reason = ""

        # Build user prompt from command + inferred_goals
        goals_text = ""
        if state.inferred_goals:
            goals_text = "\nInferred goals:\n" + "\n".join(
                f"- {g}" for g in state.inferred_goals
            )
        user_prompt = f"Command: {state.command}{goals_text}"

        try:
            raw = await self._model.infer(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=512,
                temperature=0.2,
                timeout_s=15.0,
            )
            content: str = raw.get("content", "")
            parsed = json.loads(content)

            raw_sub_goals = parsed.get("sub_goals")
            if not isinstance(raw_sub_goals, list) or len(raw_sub_goals) == 0:
                raise ValueError(
                    f"Model returned invalid sub_goals: {raw_sub_goals!r}"
                )

            strategy = str(parsed.get("execution_strategy", _DEFAULT_STRATEGY)).strip().lower()
            if strategy not in _VALID_STRATEGIES:
                strategy = _DEFAULT_STRATEGY

            model_result = {
                "raw_sub_goals": raw_sub_goals,
                "execution_strategy": strategy,
            }

        except json.JSONDecodeError as exc:
            degraded = True
            degraded_reason = f"model_json_parse_error:{type(exc).__name__}"
        except (ValueError, KeyError, TypeError) as exc:
            degraded = True
            degraded_reason = f"model_output_invalid:{type(exc).__name__}"
        except Exception as exc:  # noqa: BLE001 — covers RuntimeError + timeout
            degraded = True
            degraded_reason = f"model_inference_error:{type(exc).__name__}"

        # Phase 3: enrich sub-goals or run keyword fallback
        if model_result is not None and not degraded:
            raw_sub_goals = model_result["raw_sub_goals"]
            execution_strategy = model_result["execution_strategy"]
            sub_goals = self._enrich_sub_goals(
                raw_sub_goals, selection.brain_id, execution_strategy
            )
            served_mode = "LEVEL_0_PRIMARY"
            new_degraded_reason: Optional[str] = None
            confidence = state.confidence  # preserve upstream confidence unchanged
        else:
            sub_goals, execution_strategy = self._keyword_fallback(
                state.command, state.inferred_goals, selection.brain_id
            )
            served_mode = "LEVEL_1_DEGRADED"
            new_degraded_reason = degraded_reason or "keyword_fallback"
            # Cap confidence at the degraded maximum
            confidence = min(state.confidence, _DEGRADED_CONFIDENCE_CAP)

        # Phase 4: build trace entry
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        trace_entry: Dict[str, Any] = {
            "node": "planning",
            "sub_goal_count": len(sub_goals),
            "execution_strategy": execution_strategy,
            "served_mode": served_mode,
            "brain_used": selection.brain_id,
            "elapsed_ms": elapsed_ms,
        }
        if new_degraded_reason:
            trace_entry["degraded_reason"] = new_degraded_reason

        # Phase 5: return updated state (Pydantic v2 — model_copy)
        update: Dict[str, Any] = {
            "phase": "planning",
            "sub_goals": sub_goals,
            "execution_strategy": execution_strategy,
            "planning_brain_used": selection.brain_id,
            "served_mode": served_mode,
            "degraded_reason_code": new_degraded_reason,
            "reasoning_trace": state.reasoning_trace + [trace_entry],
        }
        if served_mode == "LEVEL_1_DEGRADED":
            update["confidence"] = confidence

        updated = state.model_copy(update=update)
        return updated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enrich_sub_goals(
        self,
        raw_sub_goals: List[Dict[str, Any]],
        brain_id: str,
        execution_strategy: str,
    ) -> List[Dict[str, Any]]:
        """
        Enrich raw sub-goals from the model with all required fields.

        Each dict gets:
          step_id, action_id, goal, task_type, brain_assigned,
          tool_required, depends_on
        """
        enriched: List[Dict[str, Any]] = []
        for idx, raw in enumerate(raw_sub_goals):
            step_num = idx + 1
            task_type = str(raw.get("task_type", _DEFAULT_TASK_TYPE)).strip()
            goal_text = str(raw.get("goal", "")).strip() or f"step {step_num}"

            # Assign brain per sub-goal based on its task_type
            sub_selection = self._selector.select(task_type, goal_text)

            enriched.append({
                "step_id": f"s{step_num}",
                "action_id": str(uuid.uuid4()),
                "goal": goal_text,
                "task_type": task_type,
                "brain_assigned": sub_selection.brain_id,
                "tool_required": self._infer_tool(task_type),
                "depends_on": self._build_depends_on(
                    step_num, execution_strategy, len(raw_sub_goals)
                ),
            })
        return enriched

    def _keyword_fallback(
        self,
        command: str,
        inferred_goals: List[str],
        brain_id: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Build sub-goals without the GPU model.

        Strategy:
          1. If inferred_goals has multiple entries, use those directly.
          2. Otherwise, split command on conjunctions/delimiters.
          3. Each piece becomes one sub-goal with inferred task_type.

        Returns (sub_goals, execution_strategy).
        Confidence is NOT set here — caller is responsible for capping.
        """
        # Determine raw goal strings
        if inferred_goals and len(inferred_goals) >= 2:
            raw_pieces = inferred_goals
        else:
            # Split command on conjunctions / delimiters
            pieces = [p.strip() for p in _SPLIT_PATTERN.split(command) if p.strip()]
            # If splitting produced nothing meaningful, use the whole command
            raw_pieces = pieces if pieces else [command.strip()]

        # Build enriched sub-goals
        sub_goals: List[Dict[str, Any]] = []
        for idx, piece in enumerate(raw_pieces):
            step_num = idx + 1
            task_type = self._infer_task_type(piece)
            sub_selection = self._selector.select(task_type, piece)
            sub_goals.append({
                "step_id": f"s{step_num}",
                "action_id": str(uuid.uuid4()),
                "goal": piece,
                "task_type": task_type,
                "brain_assigned": sub_selection.brain_id,
                "tool_required": self._infer_tool(task_type),
                "depends_on": [],
            })

        return sub_goals, _DEFAULT_STRATEGY

    @staticmethod
    def _infer_tool(task_type: str) -> str:
        """Map task_type to the corresponding Body tool name."""
        return _TOOL_MAP.get(task_type, _DEFAULT_TOOL)

    @staticmethod
    def _infer_task_type(text: str) -> str:
        """Infer task_type from text via keyword scoring."""
        text_lower = text.lower()
        best_type = _DEFAULT_TASK_TYPE
        best_score = 0
        for task_type, keywords in _TASK_TYPE_KEYWORDS:
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_type = task_type
        return best_type

    @staticmethod
    def _build_depends_on(
        step_num: int,
        execution_strategy: str,
        total_steps: int,
    ) -> List[str]:
        """
        Build the depends_on list for a sub-goal.

        - sequential: each step depends on the previous (except the first)
        - parallel:   all steps are independent (empty list)
        - dag:        same as sequential by default (model may override later)
        """
        if execution_strategy == "parallel":
            return []
        if step_num == 1:
            return []
        return [f"s{step_num - 1}"]
