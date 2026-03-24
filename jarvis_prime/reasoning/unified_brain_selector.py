"""
UnifiedBrainSelector — 4-layer gate brain selector for J-Prime.

Merges three separate selectors from the JARVIS codebase:
  - BrainSelector (Ouroboros governance)
  - InteractiveBrainRouter (task_type → complexity mapping)
  - RouteDecisionService (cost gate + fallback chains)

Called at every LangGraph reasoning node to determine which GPU brain
to use for a given task, based on:

  Layer 1 — Intent:    task_type → base complexity via _TASK_COMPLEXITY
  Layer 2 — Keyword:   command text regex escalation / de-escalation
  Layer 3 — Resource:  placeholder (always passes, future: VRAM probe)
  Layer 4 — Cost:      daily_spend_gcp / daily_spend_claude vs budget

Environment variables
---------------------
JARVIS_DAILY_BUDGET_GCP    float  (default 5.0)  — GCP daily spend ceiling ($)
JARVIS_DAILY_BUDGET_CLAUDE float  (default 2.0)  — Claude daily spend ceiling ($)
JARVIS_VISION_MODEL_NAME   str    (default "llava-v1.5-7b") — vision model name
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Output dataclass (frozen — immutable after construction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnifiedBrainSelection:
    """The complete routing decision returned by UnifiedBrainSelector.select()."""

    brain_id: str
    model_name: str
    complexity: str
    graph_depth: str
    routing_reason: str
    fallback_chain: List[str]
    claude_fallback: str
    cost_gate_passed: bool
    resource_gate_passed: bool
    vision_model: Optional[str] = None
    doubleword_model: Optional[str] = None  # Tier 0 batch model for complex/coding tasks


# ---------------------------------------------------------------------------
# Module-level lookup tables  (no hardcoded business logic in methods)
# ---------------------------------------------------------------------------

# Layer 1: task_type → complexity tier
_TASK_COMPLEXITY: Dict[str, str] = {
    # trivial — sub-100 ms, no reasoning required
    "workspace_fastpath": "trivial",
    "system_command": "trivial",
    "reflex_match": "trivial",
    # light — single-step reasoning / classification
    "classification": "light",
    "step_decomposition": "light",
    "email_triage": "light",
    "calendar_query": "light",
    "goal_chain_step": "light",
    # heavy — multi-modal or multi-step with tool use
    "vision_action": "heavy",
    "vision_verification": "heavy",
    "screen_observation": "heavy",
    "proactive_narration": "heavy",
    "email_compose": "heavy",
    "browser_navigation": "heavy",
    # complex — deep reasoning, long context, planning
    "multi_step_planning": "complex",
    "email_summarization": "complex",
    "complex_reasoning": "complex",
}

# Brain IDs per complexity tier
_DEFAULT_BRAIN: Dict[str, str] = {
    "trivial": "phi3_lightweight",
    "light": "qwen_coder",
    "heavy": "qwen_coder",
    "complex": "qwen_coder_32b",
}

# Model name strings per complexity tier
_DEFAULT_MODEL: Dict[str, str] = {
    "trivial": "llama-3.2-1b",
    "light": "qwen-2.5-coder-7b",
    "heavy": "qwen-2.5-coder-7b",
    "complex": "qwen-2.5-coder-32b",
}

# Claude fallback model per complexity tier
_CLAUDE_FALLBACK: Dict[str, str] = {
    "trivial": "claude-haiku-4-5-20251001",
    "light": "claude-sonnet-4-20250514",
    "heavy": "claude-sonnet-4-20250514",
    "complex": "claude-sonnet-4-20250514",
}

# LangGraph pipeline depth per complexity tier
_GRAPH_DEPTH: Dict[str, str] = {
    "trivial": "fast",
    "light": "fast",
    "heavy": "standard",
    "complex": "full",
}

# Ordered fallback chains per primary brain_id
_FALLBACK_CHAINS: Dict[str, List[str]] = {
    "phi3_lightweight": ["qwen_coder", "mistral_7b_fallback"],
    "qwen_coder": ["qwen_coder_14b", "mistral_7b_fallback"],
    "qwen_coder_14b": ["qwen_coder_32b", "mistral_7b_fallback"],
    "qwen_coder_32b": ["mistral_7b_fallback"],
    "mistral_7b_fallback": [],
}

# Task types that require a vision model
_VISION_TASKS: Set[str] = {
    "vision_action",
    "vision_verification",
    "screen_observation",
    "proactive_narration",
}

# Task types eligible for Doubleword Tier 0 (batch, latency-insensitive)
# These tasks benefit from 397B-class reasoning and don't need real-time response
_DOUBLEWORD_ELIGIBLE: Set[str] = {
    "complex_reasoning",
    "multi_step_planning",
    "email_summarization",
}

# Doubleword model per complexity tier (None = not eligible)
_DOUBLEWORD_MODEL: Dict[str, Optional[str]] = {
    "trivial": None,
    "light": None,
    "heavy": None,
    "complex": os.environ.get("DOUBLEWORD_MODEL", "Qwen/Qwen3.5-397B-A17B-FP8"),
}

# ---------------------------------------------------------------------------
# Keyword regex patterns for Layer 2 escalation / de-escalation
# Compiled once at module load — zero per-call overhead.
# ---------------------------------------------------------------------------

# Patterns that escalate complexity to "complex"
_COMPLEX_RE: re.Pattern[str] = re.compile(
    r"\b("
    r"analyz[e|ing|ed]|analyse|root\s+cause|diagnos[e|ing|ed]|"
    r"explain\s+why|trace\s+the|architect|refactor|rewrite|"
    r"summarize|summarise|synthesize|synthesise|"
    r"compare\s+and\s+contrast|design\s+a\s+system|"
    r"multi.?step|step.by.step|comprehensive|"
    r"deep\s+dive|in.?depth|thoroughly|investigate"
    r")\b",
    re.IGNORECASE,
)

# Patterns that de-escalate complexity to "trivial"
_TRIVIAL_RE: re.Pattern[str] = re.compile(
    r"\b("
    r"what\s+time|current\s+time|what\s+day|today.?s\s+date|"
    r"open\s+app|close\s+app|volume\s+up|volume\s+down|"
    r"lock\s+screen|unlock\s+screen|take\s+screenshot|"
    r"play\s+music|pause\s+music|skip\s+track|"
    r"set\s+timer|set\s+alarm|remind\s+me\s+in|"
    r"quick\s+note|jot\s+down"
    r")\b",
    re.IGNORECASE,
)

# Complexity ordering for escalation guards
_COMPLEXITY_ORDER: Dict[str, int] = {
    "trivial": 0,
    "light": 1,
    "heavy": 2,
    "complex": 3,
}


# ---------------------------------------------------------------------------
# UnifiedBrainSelector
# ---------------------------------------------------------------------------


class UnifiedBrainSelector:
    """
    4-layer gate brain selector.

    Instantiate once per process (or per request if you prefer stateless use).
    Daily spend counters are instance-level — reset automatically at midnight.

    Usage::

        selector = UnifiedBrainSelector()
        selection = selector.select("vision_action", "take a screenshot of the error")
        selector.record_cost("gcp", 0.002)
    """

    def __init__(self) -> None:
        # Daily spend accumulators (public so tests can inspect / mutate them)
        self.daily_spend_gcp: float = 0.0
        self.daily_spend_claude: float = 0.0
        self.daily_spend_doubleword: float = 0.0

        # Budget ceilings from environment (float, with safe fallback)
        self._budget_gcp: float = _parse_float_env("JARVIS_DAILY_BUDGET_GCP", 5.0)
        self._budget_claude: float = _parse_float_env("JARVIS_DAILY_BUDGET_CLAUDE", 2.0)
        self._budget_doubleword: float = _parse_float_env("JARVIS_DAILY_BUDGET_DOUBLEWORD", 1.0)

        # Vision model name from environment
        self._vision_model: str = os.environ.get(
            "JARVIS_VISION_MODEL_NAME", "llava-v1.5-7b"
        )

        # Doubleword availability (requires API key)
        self._doubleword_available: bool = bool(os.environ.get("DOUBLEWORD_API_KEY", ""))

        # Day tracking for automatic midnight reset (POSIX day number)
        self._current_day: int = _today_day_number()

        # Optional BrainPolicyReader (loaded lazily, non-fatal if absent)
        self._policy_reader = _try_load_policy_reader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self, task_type: str, command: str = "") -> UnifiedBrainSelection:
        """
        Run the 4-layer gate and return a routing decision.

        Parameters
        ----------
        task_type:
            Semantic task category, e.g. "vision_action", "classification".
        command:
            Raw command text (used for Layer 2 keyword escalation).

        Returns
        -------
        UnifiedBrainSelection
            Frozen dataclass with brain_id, model_name, graph_depth, etc.
        """
        self._maybe_reset_daily()

        # --- Layer 1: Intent mapping ---
        complexity = _TASK_COMPLEXITY.get(task_type, "light")
        reason_parts: List[str] = [f"task_type={task_type!r} → base_complexity={complexity!r}"]

        # --- Layer 2: Keyword escalation / de-escalation ---
        if command:
            if _TRIVIAL_RE.search(command):
                if _COMPLEXITY_ORDER[complexity] > _COMPLEXITY_ORDER["trivial"]:
                    reason_parts.append(
                        f"keyword de-escalation: {complexity!r} → 'trivial'"
                    )
                    complexity = "trivial"
            elif _COMPLEX_RE.search(command):
                if _COMPLEXITY_ORDER[complexity] < _COMPLEXITY_ORDER["complex"]:
                    reason_parts.append(
                        f"keyword escalation: {complexity!r} → 'complex'"
                    )
                    complexity = "complex"

        # --- Layer 3: Resource gate (placeholder — always passes) ---
        resource_gate_passed = True
        reason_parts.append("resource_gate=pass (placeholder)")

        # --- Layer 4: Cost gate ---
        cost_gate_passed = self._check_cost_gate()
        reason_parts.append(
            f"cost_gate={'pass' if cost_gate_passed else 'FAIL'} "
            f"(gcp_spend=${self.daily_spend_gcp:.4f}/{self._budget_gcp}, "
            f"claude_spend=${self.daily_spend_claude:.4f}/{self._budget_claude})"
        )

        # --- Resolve brain + model ---
        brain_id = _DEFAULT_BRAIN[complexity]
        model_name = _DEFAULT_MODEL[complexity]
        claude_fallback = _CLAUDE_FALLBACK[complexity]
        graph_depth = _GRAPH_DEPTH[complexity]
        fallback_chain = list(_FALLBACK_CHAINS.get(brain_id, []))

        # Vision model (only for vision task types)
        vision_model: Optional[str] = (
            self._vision_model if task_type in _VISION_TASKS else None
        )

        # Doubleword Tier 0 (batch 397B for complex/coding tasks)
        doubleword_model: Optional[str] = None
        if (task_type in _DOUBLEWORD_ELIGIBLE or complexity == "complex") and self._doubleword_available:
            doubleword_model = _DOUBLEWORD_MODEL.get(complexity)
            if doubleword_model:
                reason_parts.append(
                    f"doubleword_tier0={doubleword_model} (batch, 29x cheaper)"
                )

        routing_reason = "; ".join(reason_parts)

        return UnifiedBrainSelection(
            brain_id=brain_id,
            model_name=model_name,
            complexity=complexity,
            graph_depth=graph_depth,
            routing_reason=routing_reason,
            fallback_chain=fallback_chain,
            claude_fallback=claude_fallback,
            cost_gate_passed=cost_gate_passed,
            resource_gate_passed=resource_gate_passed,
            vision_model=vision_model,
            doubleword_model=doubleword_model,
        )

    def record_cost(self, provider: str, usd: float) -> None:
        """
        Accumulate spend for a provider so the cost gate stays current.

        Parameters
        ----------
        provider:
            "gcp" or "claude". Unknown values are silently ignored.
        usd:
            Cost in US dollars (positive float).
        """
        self._maybe_reset_daily()
        if provider == "gcp":
            self.daily_spend_gcp += usd
        elif provider == "claude":
            self.daily_spend_claude += usd
        elif provider == "doubleword":
            self.daily_spend_doubleword += usd
        # Unknown providers: no-op (graceful degradation)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_cost_gate(self) -> bool:
        """Return True if all providers are within budget."""
        return (
            self.daily_spend_gcp < self._budget_gcp
            and self.daily_spend_claude < self._budget_claude
            and self.daily_spend_doubleword < self._budget_doubleword
        )

    def _maybe_reset_daily(self) -> None:
        """Reset daily accumulators when the calendar day has changed."""
        today = _today_day_number()
        if today != self._current_day:
            self.daily_spend_gcp = 0.0
            self.daily_spend_claude = 0.0
            self.daily_spend_doubleword = 0.0
            self._current_day = today


# ---------------------------------------------------------------------------
# Module-level helpers (pure functions, no state)
# ---------------------------------------------------------------------------


def _parse_float_env(name: str, default: float) -> float:
    """Read an env var as float, falling back to *default* on any error."""
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _today_day_number() -> int:
    """Return the current UTC day as an integer (seconds // 86400)."""
    return int(time.time()) // 86_400


def _try_load_policy_reader():
    """
    Attempt to import BrainPolicyReader from the JARVIS codebase.

    Returns the reader instance if available, None otherwise.
    Failure is non-fatal — the selector works without it.
    """
    try:
        from jarvis_prime.core.hybrid_router import get_brain_policy_reader  # type: ignore[import]
        return get_brain_policy_reader()
    except Exception:
        return None
