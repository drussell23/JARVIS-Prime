"""
ReasoningGraph -- the ONE thinking pipeline for J-Prime.

Wires AnalysisNode -> [PlanningNode] -> ValidationNode -> ExecutionPlanner
into a depth-routed graph. Trivial/light commands skip PlanningNode but still
pass through validation and execution planning. All paths produce a complete
wire-format response dict matching the ReasonResponse schema.

Depth routing
-------------
- fast (trivial/light):  Analysis -> Validation -> ExecutionPlanner
- standard (heavy):      Analysis -> Planning -> Validation -> ExecutionPlanner
- full (complex):        Analysis -> Planning -> Validation -> ExecutionPlanner

Design rules
------------
- No LangGraph library dependency -- simple async pipeline with conditional routing.
- Constructor DI for ModelProvider and optional UnifiedBrainSelector.
- All 4 node instances share the same model_provider and brain_selector.
- Response dict matches ReasonResponse schema from protocol.py.
- state immutability preserved via Pydantic v2 model_copy(update=...) in nodes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from jarvis_prime.reasoning.graph_nodes.analysis_node import AnalysisNode
from jarvis_prime.reasoning.graph_nodes.execution_planner import ExecutionPlanner
from jarvis_prime.reasoning.graph_nodes.planning_node import PlanningNode
from jarvis_prime.reasoning.graph_nodes.validation_node import ValidationNode
from jarvis_prime.reasoning.model_provider import ModelProvider
from jarvis_prime.reasoning.protocol import PROTOCOL_VERSION, ReasoningGraphState
from jarvis_prime.reasoning.unified_brain_selector import UnifiedBrainSelector

logger = logging.getLogger(__name__)

# Graph depths that require the PlanningNode to run
_PLANNING_DEPTHS = frozenset({"standard", "full"})


class ReasoningGraph:
    """
    The ONE thinking pipeline that wires all reasoning nodes together.

    Parameters
    ----------
    model_provider:
        Any object satisfying the ModelProvider protocol. Injected into
        AnalysisNode and PlanningNode for GPU inference.
    brain_selector:
        Optional UnifiedBrainSelector shared across all nodes. A default
        instance is created if not supplied.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        brain_selector: Optional[UnifiedBrainSelector] = None,
    ) -> None:
        selector = brain_selector or UnifiedBrainSelector()

        self._analysis = AnalysisNode(
            model_provider=model_provider, brain_selector=selector
        )
        self._planning = PlanningNode(
            model_provider=model_provider, brain_selector=selector
        )
        self._validation = ValidationNode()
        self._execution_planner = ExecutionPlanner(brain_selector=selector)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        request_id: str,
        session_id: str,
        trace_id: str,
        command: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full reasoning pipeline and return a wire-format response.

        Parameters
        ----------
        request_id:
            Unique identifier for this request.
        session_id:
            Session scope identifier.
        trace_id:
            Distributed trace identifier.
        command:
            The user command to reason about.
        context:
            Optional additional context dict.

        Returns
        -------
        dict
            Wire-format response matching the ReasonResponse schema.
        """
        # 1. Create initial state
        state = ReasoningGraphState(
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
            command=command,
            context=context or {},
        )

        # 2. AnalysisNode (always runs)
        state = await self._analysis.process(state)
        logger.debug(
            "Analysis complete: intent=%s complexity=%s confidence=%.3f depth=%s",
            state.intent, state.complexity, state.confidence, state.graph_depth,
        )

        # 3. Depth routing: PlanningNode only for standard/full depths
        if state.graph_depth in _PLANNING_DEPTHS:
            state = await self._planning.process(state)
            logger.debug(
                "Planning complete: %d sub_goals, strategy=%s",
                len(state.sub_goals), state.execution_strategy,
            )

        # 4. ValidationNode (always runs)
        state = await self._validation.process(state)
        logger.debug(
            "Validation complete: approval_required=%s reason_codes=%s",
            state.approval_required, state.approval_reason_codes,
        )

        # 5. ExecutionPlanner (always runs)
        state = await self._execution_planner.process(state)
        logger.debug(
            "ExecutionPlanner complete: plan_id=%s",
            state.context.get("plan", {}).get("plan_id", ""),
        )

        # 6. Build and return wire-format response
        return self._build_response(state)

    # ------------------------------------------------------------------
    # Response building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response(state: ReasoningGraphState) -> Dict[str, Any]:
        """
        Convert final ReasoningGraphState into a wire-format dict matching
        the ReasonResponse schema.

        The plan is read from ``state.context["plan"]`` where ExecutionPlanner
        stores it as a dict (with ``plan_id``, ``plan_hash``, ``sub_goals`` as
        list of dicts, etc.).
        """
        # Extract the plan dict produced by ExecutionPlanner
        plan_dict: Dict[str, Any] = state.context.get("plan", {})

        # Convert SubGoal pydantic dicts to plain dicts (they already are dicts
        # because ExecutionPlanner calls plan.model_dump())
        sub_goals_raw: List[Any] = plan_dict.get("sub_goals", [])
        sub_goals_out: List[Dict[str, Any]] = []
        for sg in sub_goals_raw:
            if isinstance(sg, dict):
                sub_goals_out.append(sg)
            else:
                # Safety: if somehow a Pydantic model leaked through
                sub_goals_out.append(
                    sg.model_dump() if hasattr(sg, "model_dump") else dict(sg)
                )

        # Determine status
        status = "needs_approval" if state.approval_required else "plan_ready"

        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": state.request_id,
            "session_id": state.session_id,
            "trace_id": state.trace_id,
            "status": status,
            "served_mode": state.served_mode,
            "degraded_reason_code": state.degraded_reason_code,
            "classification": {
                "intent": state.intent,
                "complexity": state.complexity,
                "confidence": state.confidence,
                "brain_used": state.analysis_brain_used,
                "graph_depth": state.graph_depth,
            },
            "plan": {
                "plan_id": plan_dict.get("plan_id", ""),
                "plan_hash": plan_dict.get("plan_hash", ""),
                "sub_goals": sub_goals_out,
                "execution_strategy": plan_dict.get(
                    "execution_strategy", state.execution_strategy
                ),
                "total_estimated_ms": plan_dict.get("total_estimated_ms", 0),
                "approval_required": state.approval_required,
                "approval_reason_codes": list(state.approval_reason_codes),
                "approval_scope": plan_dict.get("approval_scope", "plan"),
                "risk_level": plan_dict.get("risk_level", state.risk_level),
            },
            "routing_trace": {
                "analysis_brain": state.analysis_brain_used,
                "planning_brain": state.planning_brain_used,
                "cost_gate_passed": state.cost_gate_passed,
                "resource_gate_passed": state.resource_gate_passed,
            },
        }
