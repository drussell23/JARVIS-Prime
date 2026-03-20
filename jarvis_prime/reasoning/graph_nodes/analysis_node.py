"""
AnalysisNode — first node in the J-Prime unified reasoning pipeline.

Classifies the incoming command's intent, complexity, confidence, and
inferred sub-goals by calling a ModelProvider (GPU model) for JSON output.
Falls back to a deterministic keyword-pattern classifier when the model
call fails or returns non-parseable text.

Design rules
------------
- Never import from server.py. Accepts ModelProvider via constructor DI.
- json.loads() for model output; catches JSONDecodeError explicitly.
- Pattern-fallback confidence is capped at 0.6 (spec §3).
- Degraded path sets served_mode=LEVEL_1_DEGRADED and degraded_reason_code.
- graph_depth is derived from complexity via the same table as
  UnifiedBrainSelector (_GRAPH_DEPTH mapping: trivial/light→fast,
  heavy→standard, complex→full).
- One reasoning_trace entry is always appended with node="analysis".
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from jarvis_prime.reasoning.model_provider import ModelProvider
from jarvis_prime.reasoning.protocol import ReasoningGraphState
from jarvis_prime.reasoning.unified_brain_selector import UnifiedBrainSelector


# ---------------------------------------------------------------------------
# Constants — no hardcoded business logic outside these tables
# ---------------------------------------------------------------------------

# Valid complexity values accepted from the model or produced by fallback
_VALID_COMPLEXITIES = frozenset({"trivial", "light", "heavy", "complex"})

# Complexity → LangGraph pipeline depth
_GRAPH_DEPTH: Dict[str, str] = {
    "trivial": "fast",
    "light": "fast",
    "heavy": "standard",
    "complex": "full",
}

# Default fallback when a complexity string is unknown / missing
_DEFAULT_COMPLEXITY = "light"
_DEFAULT_GRAPH_DEPTH = "fast"

# Pattern-fallback: intent category → keyword list (all lowercase)
# Longer / more-specific lists are checked first for best-match scoring.
_INTENT_PATTERNS: Dict[str, List[str]] = {
    "multi_step_planning": [
        "plan", "workflow", "automate", "orchestrate", "pipeline",
        "sequence", "schedule",
    ],
    "complex_reasoning": [
        "analyze", "analyse", "compare", "evaluate", "investigate",
        "explain why", "root cause", "diagnose", "summarize", "summarise",
    ],
    "email_compose": [
        "email", "send", "reply", "draft", "compose", "write to",
        "message to",
    ],
    "browser_navigation": [
        "navigate", "go to", "browse", "visit", "search", "open website",
        "open url", "open tab",
    ],
    "system_command": [
        "open", "close", "launch", "quit", "exit", "volume", "brightness",
        "lock", "unlock", "screenshot", "restart", "shutdown",
    ],
    "calendar_query": [
        "calendar", "meeting", "schedule", "appointment", "event", "remind",
    ],
    "vision_action": [
        "click", "type", "fill", "submit", "scroll", "screenshot of",
        "screen", "window",
    ],
}

# Fallback intent when no pattern matches at all
_FALLBACK_INTENT = "system_command"

# Cap for confidence in degraded / pattern-fallback mode
_DEGRADED_CONFIDENCE_CAP = 0.6

# System prompt sent to the model for classification
_SYSTEM_PROMPT = (
    "You are a command classifier for JARVIS AI assistant.\n"
    "Given a user command, classify it as JSON with these fields:\n"
    '- "intent": the primary intent category\n'
    '- "complexity": one of "trivial", "light", "heavy", "complex"\n'
    '- "confidence": 0.0 to 1.0\n'
    '- "goals": list of inferred sub-goals (strings)\n'
    "Respond with ONLY valid JSON, no explanation."
)


# ---------------------------------------------------------------------------
# AnalysisNode
# ---------------------------------------------------------------------------


class AnalysisNode:
    """
    First node in the J-Prime reasoning pipeline.

    Parameters
    ----------
    model_provider:
        Any object satisfying the ModelProvider protocol.
    brain_selector:
        Optional UnifiedBrainSelector. A default instance is created if not
        supplied so that ``analysis_brain_used`` is always populated.
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
        Classify *state.command* and return an updated ReasoningGraphState.

        Always sets:
          - phase = "analyzing"
          - graph_depth (derived from resolved complexity)
          - reasoning_trace += [{"node": "analysis", ...}]

        On success (model returns valid JSON with intent + complexity):
          - intent, complexity, confidence, inferred_goals, analysis_brain_used
          - served_mode stays LEVEL_0_PRIMARY

        On failure (model error OR JSON parse error OR missing required fields):
          - Delegates to _pattern_fallback()
          - served_mode = LEVEL_1_DEGRADED
          - confidence <= 0.6
          - degraded_reason_code set
        """
        t_start = time.monotonic()

        # Phase 1: brain selection (never raises — UnifiedBrainSelector is pure)
        selection = self._selector.select("classification", state.command)

        # Phase 2: attempt GPU model classification
        model_result: Optional[Dict[str, Any]] = None
        degraded = False
        degraded_reason = ""

        try:
            raw = await self._model.infer(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": state.command},
                ],
                max_tokens=256,
                temperature=0.2,
                timeout_s=10.0,
            )
            content: str = raw.get("content", "")
            parsed = json.loads(content)

            # Validate required keys are present and have meaningful values
            intent = str(parsed.get("intent", "")).strip()
            complexity = str(parsed.get("complexity", "")).strip().lower()

            if not intent or complexity not in _VALID_COMPLEXITIES:
                raise ValueError(
                    f"Invalid model output: intent={intent!r}, "
                    f"complexity={complexity!r}"
                )

            confidence = float(parsed.get("confidence", 0.0))
            goals: List[str] = [str(g) for g in parsed.get("goals", [])]
            model_result = {
                "intent": intent,
                "complexity": complexity,
                "confidence": confidence,
                "goals": goals,
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

        # Phase 3: apply result or pattern fallback
        if model_result is not None and not degraded:
            intent = model_result["intent"]
            complexity = model_result["complexity"]
            confidence = model_result["confidence"]
            goals = model_result["goals"]
            served_mode = "LEVEL_0_PRIMARY"
            new_degraded_reason: Optional[str] = None
        else:
            intent, complexity, confidence, goals = self._pattern_fallback(
                state.command
            )
            served_mode = "LEVEL_1_DEGRADED"
            new_degraded_reason = degraded_reason or "pattern_fallback"

        # Phase 4: derive graph_depth from complexity
        graph_depth = _GRAPH_DEPTH.get(complexity, _DEFAULT_GRAPH_DEPTH)

        # Phase 5: build trace entry
        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        trace_entry: Dict[str, Any] = {
            "node": "analysis",
            "intent": intent,
            "complexity": complexity,
            "confidence": confidence,
            "graph_depth": graph_depth,
            "served_mode": served_mode,
            "brain_used": selection.brain_id,
            "elapsed_ms": elapsed_ms,
        }
        if new_degraded_reason:
            trace_entry["degraded_reason"] = new_degraded_reason

        # Phase 6: return updated state (Pydantic v2 — model_copy)
        updated = state.model_copy(
            update={
                "phase": "analyzing",
                "intent": intent,
                "complexity": complexity,
                "confidence": confidence,
                "inferred_goals": goals,
                "analysis_brain_used": selection.brain_id,
                "graph_depth": graph_depth,
                "served_mode": served_mode,
                "degraded_reason_code": new_degraded_reason,
                "reasoning_trace": state.reasoning_trace + [trace_entry],
            }
        )
        return updated

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pattern_fallback(
        self, command: str
    ) -> Tuple[str, str, float, List[str]]:
        """
        Classify *command* via keyword scoring when the model is unavailable.

        Returns (intent, complexity, confidence, goals).
        Confidence is always <= _DEGRADED_CONFIDENCE_CAP (0.6).
        """
        cmd_lower = command.lower()

        # Score each intent category by counting keyword matches
        scores: Dict[str, int] = {}
        for intent_name, keywords in _INTENT_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in cmd_lower)
            if score > 0:
                scores[intent_name] = score

        if scores:
            best_intent = max(scores, key=lambda k: scores[k])
            match_count = scores[best_intent]
            # Scale confidence: 1 match→0.40, 2→0.50, 3+→0.55 (all <= 0.6)
            raw_confidence = min(0.35 + match_count * 0.10, _DEGRADED_CONFIDENCE_CAP)
        else:
            best_intent = _FALLBACK_INTENT
            raw_confidence = 0.30  # well below cap — genuinely unknown

        # Derive complexity via brain selector (uses same keyword rules)
        selection = self._selector.select(best_intent, command)
        complexity = selection.complexity if selection.complexity in _VALID_COMPLEXITIES else _DEFAULT_COMPLEXITY

        # Confidence is capped at the degraded maximum
        confidence = min(raw_confidence, _DEGRADED_CONFIDENCE_CAP)

        # Produce minimal goals list from the matched intent
        goals: List[str] = [f"execute {best_intent.replace('_', ' ')}"]

        return best_intent, complexity, confidence, goals
