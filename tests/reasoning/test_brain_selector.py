"""
TDD tests for UnifiedBrainSelector — 4-layer gate brain selector.

Written BEFORE implementation. All tests must fail with ModuleNotFoundError
initially, then pass once unified_brain_selector.py is implemented.
"""
from __future__ import annotations

import pytest

from jarvis_prime.reasoning.unified_brain_selector import (
    UnifiedBrainSelection,
    UnifiedBrainSelector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def selector() -> UnifiedBrainSelector:
    return UnifiedBrainSelector()


# ---------------------------------------------------------------------------
# TestComplexityMapping
# ---------------------------------------------------------------------------


class TestComplexityMapping:
    """Layer 1: task_type → base complexity string."""

    def test_trivial_system_command(self, selector: UnifiedBrainSelector):
        result = selector.select("system_command", "")
        assert result.complexity in ("trivial", "light")

    def test_trivial_workspace_fastpath(self, selector: UnifiedBrainSelector):
        result = selector.select("workspace_fastpath", "")
        assert result.complexity in ("trivial", "light")

    def test_light_classification(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "")
        assert result.complexity == "light"

    def test_heavy_vision_action(self, selector: UnifiedBrainSelector):
        result = selector.select("vision_action", "")
        assert result.complexity == "heavy"

    def test_complex_complex_reasoning(self, selector: UnifiedBrainSelector):
        result = selector.select("complex_reasoning", "")
        assert result.complexity == "complex"

    def test_unknown_task_defaults_to_light(self, selector: UnifiedBrainSelector):
        result = selector.select("totally_unknown_task_xyz", "")
        assert result.complexity == "light"


# ---------------------------------------------------------------------------
# TestBrainSelection
# ---------------------------------------------------------------------------


class TestBrainSelection:
    """Verify output shape is complete and well-typed."""

    def test_returns_nonempty_brain_id(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "classify this")
        assert isinstance(result.brain_id, str)
        assert len(result.brain_id) > 0

    def test_returns_nonempty_model_name(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "classify this")
        assert isinstance(result.model_name, str)
        assert len(result.model_name) > 0

    def test_returns_fallback_chain_as_list(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "classify this")
        assert isinstance(result.fallback_chain, list)

    def test_returns_nonempty_routing_reason(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "classify this")
        assert isinstance(result.routing_reason, str)
        assert len(result.routing_reason) > 0

    def test_returns_claude_fallback_containing_claude(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "classify this")
        assert "claude" in result.claude_fallback.lower()

    def test_selection_is_frozen_dataclass(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "")
        with pytest.raises((AttributeError, TypeError)):
            result.brain_id = "mutated"  # type: ignore[misc]

    def test_cost_gate_passed_is_bool(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "")
        assert isinstance(result.cost_gate_passed, bool)

    def test_resource_gate_passed_is_bool(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "")
        assert isinstance(result.resource_gate_passed, bool)


# ---------------------------------------------------------------------------
# TestKeywordEscalation
# ---------------------------------------------------------------------------


class TestKeywordEscalation:
    """Layer 2: keyword scanning escalates or de-escalates complexity."""

    def test_analyze_root_cause_escalates_to_complex(self, selector: UnifiedBrainSelector):
        # Start with a light task; keyword should escalate to complex.
        result = selector.select("classification", "analyze the root cause of this failure")
        assert result.complexity == "complex"

    def test_trivial_phrase_de_escalates(self, selector: UnifiedBrainSelector):
        # "what time is it" should produce trivial complexity.
        result = selector.select("complex_reasoning", "what time is it")
        assert result.complexity == "trivial"

    def test_no_keyword_preserves_base_complexity(self, selector: UnifiedBrainSelector):
        # Neutral command; base complexity should be preserved.
        result = selector.select("classification", "run the classifier")
        assert result.complexity == "light"

    def test_escalation_reflected_in_routing_reason(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "analyze the root cause")
        assert len(result.routing_reason) > 0


# ---------------------------------------------------------------------------
# TestCostGate
# ---------------------------------------------------------------------------


class TestCostGate:
    """Layer 4: daily spend tracking and budget enforcement."""

    def test_default_cost_gate_passes(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "test")
        assert result.cost_gate_passed is True

    def test_record_cost_updates_daily_spend_gcp(self, selector: UnifiedBrainSelector):
        before = selector.daily_spend_gcp
        selector.record_cost("gcp", 0.05)
        assert selector.daily_spend_gcp == pytest.approx(before + 0.05)

    def test_record_cost_updates_daily_spend_claude(self, selector: UnifiedBrainSelector):
        before = selector.daily_spend_claude
        selector.record_cost("claude", 0.03)
        assert selector.daily_spend_claude == pytest.approx(before + 0.03)

    def test_record_cost_unknown_provider_no_crash(self, selector: UnifiedBrainSelector):
        # Should not raise even for an unknown provider name.
        selector.record_cost("unknown_provider", 0.01)

    def test_cost_gate_fails_when_budget_exceeded(self):
        # Create selector with zero budget so first call fails the gate.
        import os
        original_gcp = os.environ.get("JARVIS_DAILY_BUDGET_GCP")
        original_claude = os.environ.get("JARVIS_DAILY_BUDGET_CLAUDE")
        try:
            os.environ["JARVIS_DAILY_BUDGET_GCP"] = "0.0"
            os.environ["JARVIS_DAILY_BUDGET_CLAUDE"] = "0.0"
            s = UnifiedBrainSelector()
            s.daily_spend_gcp = 999.0  # simulate exhausted budget
            result = s.select("classification", "test")
            assert result.cost_gate_passed is False
        finally:
            if original_gcp is None:
                os.environ.pop("JARVIS_DAILY_BUDGET_GCP", None)
            else:
                os.environ["JARVIS_DAILY_BUDGET_GCP"] = original_gcp
            if original_claude is None:
                os.environ.pop("JARVIS_DAILY_BUDGET_CLAUDE", None)
            else:
                os.environ["JARVIS_DAILY_BUDGET_CLAUDE"] = original_claude


# ---------------------------------------------------------------------------
# TestGraphDepthMapping
# ---------------------------------------------------------------------------


class TestGraphDepthMapping:
    """Complexity → graph_depth string used by LangGraph pipeline."""

    def test_trivial_maps_to_fast(self, selector: UnifiedBrainSelector):
        result = selector.select("system_command", "")
        assert result.graph_depth == "fast"

    def test_light_maps_to_fast(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "")
        assert result.graph_depth == "fast"

    def test_heavy_maps_to_standard(self, selector: UnifiedBrainSelector):
        result = selector.select("vision_action", "")
        assert result.graph_depth == "standard"

    def test_complex_maps_to_full(self, selector: UnifiedBrainSelector):
        result = selector.select("complex_reasoning", "")
        assert result.graph_depth == "full"

    def test_graph_depth_after_keyword_escalation(self, selector: UnifiedBrainSelector):
        # Keyword escalation to complex should yield "full" graph depth.
        result = selector.select("classification", "analyze the root cause of this")
        assert result.graph_depth == "full"


# ---------------------------------------------------------------------------
# TestVisionModel
# ---------------------------------------------------------------------------


class TestVisionModel:
    """Vision tasks should populate vision_model; others should not."""

    def test_vision_action_has_vision_model(self, selector: UnifiedBrainSelector):
        result = selector.select("vision_action", "")
        assert result.vision_model is not None
        assert len(result.vision_model) > 0

    def test_vision_verification_has_vision_model(self, selector: UnifiedBrainSelector):
        result = selector.select("vision_verification", "")
        assert result.vision_model is not None

    def test_non_vision_task_has_no_vision_model(self, selector: UnifiedBrainSelector):
        result = selector.select("classification", "")
        assert result.vision_model is None

    def test_system_command_has_no_vision_model(self, selector: UnifiedBrainSelector):
        result = selector.select("system_command", "")
        assert result.vision_model is None
