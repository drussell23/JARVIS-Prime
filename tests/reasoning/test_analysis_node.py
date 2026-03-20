"""
TDD tests for AnalysisNode — first node in the unified reasoning pipeline.

Written BEFORE implementation. All tests must fail with ImportError initially,
then pass once analysis_node.py is implemented.

Tests cover:
- Happy-path model response → state fields updated, LEVEL_0_PRIMARY
- Model failure → LEVEL_1_DEGRADED, confidence capped at 0.6
- Pattern fallback classifies known keyword commands correctly
- graph_depth is derived from complexity (fast/standard/full)
- reasoning_trace receives exactly one entry with node="analysis"
- phase is set to "analyzing"
- Non-JSON model response degrades gracefully (no uncaught exception)
"""
from __future__ import annotations

import asyncio
import pytest

from jarvis_prime.reasoning.model_provider import MockModelProvider
from jarvis_prime.reasoning.protocol import ReasoningGraphState
from jarvis_prime.reasoning.unified_brain_selector import UnifiedBrainSelector
from jarvis_prime.reasoning.graph_nodes.analysis_node import AnalysisNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_COMPLEXITIES = {"trivial", "light", "heavy", "complex"}
VALID_GRAPH_DEPTHS = {"fast", "standard", "full"}


def make_state(command: str = "open Safari") -> ReasoningGraphState:
    return ReasoningGraphState(
        request_id="req-test-001",
        session_id="sess-test-001",
        trace_id="trace-test-001",
        command=command,
    )


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# TestSuccessfulAnalysis
# ---------------------------------------------------------------------------


class TestSuccessfulAnalysis:
    """Happy-path: model returns valid JSON classification."""

    def test_successful_analysis_intent(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": ["navigate to linkedin"]}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.intent == "browser_navigation"

    def test_successful_analysis_complexity(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": ["navigate to linkedin"]}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.complexity == "heavy"

    def test_successful_analysis_confidence(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": ["navigate to linkedin"]}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert abs(state.confidence - 0.88) < 1e-9

    def test_successful_analysis_goals(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": ["navigate to linkedin"]}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert "navigate to linkedin" in state.inferred_goals

    def test_successful_analysis_served_mode_stays_primary(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": ["navigate to linkedin"]}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.served_mode == "LEVEL_0_PRIMARY"


# ---------------------------------------------------------------------------
# TestModelFailureDegrades
# ---------------------------------------------------------------------------


class TestModelFailureDegrades:
    """Model raises RuntimeError → state must degrade cleanly."""

    def test_model_failure_served_mode(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.served_mode == "LEVEL_1_DEGRADED"

    def test_model_failure_confidence_capped(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.confidence <= 0.6

    def test_model_failure_degraded_reason_code_set(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.degraded_reason_code is not None
        assert len(state.degraded_reason_code) > 0


# ---------------------------------------------------------------------------
# TestPatternFallbackClassifies
# ---------------------------------------------------------------------------


class TestPatternFallbackClassifies:
    """On model failure, keyword patterns must produce a meaningful intent."""

    def test_pattern_fallback_browser_navigation(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("navigate to google.com")))
        assert state.intent != ""
        assert "browser" in state.intent or "navigation" in state.intent

    def test_pattern_fallback_system_command_open(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.intent != ""

    def test_pattern_fallback_email_compose(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("compose an email to John")))
        assert state.intent != ""
        assert "email" in state.intent or "compose" in state.intent

    def test_pattern_fallback_complexity_valid(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.complexity in VALID_COMPLEXITIES

    def test_pattern_fallback_unknown_command_still_sets_intent(self):
        """Even a fully unknown command must produce a non-empty intent."""
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("xyzzy frobnicate quux")))
        assert state.intent != ""


# ---------------------------------------------------------------------------
# TestGraphDepthSet
# ---------------------------------------------------------------------------


class TestGraphDepthSet:
    """graph_depth must be one of fast / standard / full after process()."""

    def test_graph_depth_after_success(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": []}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("go to linkedin")))
        assert state.graph_depth in VALID_GRAPH_DEPTHS

    def test_graph_depth_trivial_or_light_gives_fast(self):
        provider = MockModelProvider(
            response='{"intent": "system_command", "complexity": "trivial",'
                     ' "confidence": 0.95, "goals": []}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.graph_depth == "fast"

    def test_graph_depth_heavy_gives_standard(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": []}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("navigate to linkedin")))
        assert state.graph_depth == "standard"

    def test_graph_depth_complex_gives_full(self):
        provider = MockModelProvider(
            response='{"intent": "complex_reasoning", "complexity": "complex",'
                     ' "confidence": 0.75, "goals": []}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("analyze the root cause")))
        assert state.graph_depth == "full"

    def test_graph_depth_after_fallback(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.graph_depth in VALID_GRAPH_DEPTHS


# ---------------------------------------------------------------------------
# TestReasoningTraceAppended
# ---------------------------------------------------------------------------


class TestReasoningTraceAppended:
    """reasoning_trace must have exactly one entry with node="analysis"."""

    def test_trace_has_one_entry_on_success(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": []}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert len(state.reasoning_trace) == 1

    def test_trace_node_field_is_analysis_on_success(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": []}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.reasoning_trace[0]["node"] == "analysis"

    def test_trace_has_one_entry_on_failure(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert len(state.reasoning_trace) == 1

    def test_trace_node_field_is_analysis_on_failure(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.reasoning_trace[0]["node"] == "analysis"


# ---------------------------------------------------------------------------
# TestPhaseSetToAnalyzing
# ---------------------------------------------------------------------------


class TestPhaseSetToAnalyzing:
    """phase must be "analyzing" after process() regardless of outcome."""

    def test_phase_analyzing_on_success(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": []}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.phase == "analyzing"

    def test_phase_analyzing_on_model_failure(self):
        provider = MockModelProvider(should_fail=True)
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.phase == "analyzing"


# ---------------------------------------------------------------------------
# TestNonJsonModelResponse
# ---------------------------------------------------------------------------


class TestNonJsonModelResponse:
    """Non-JSON model response must degrade gracefully — no uncaught exception."""

    def test_non_json_response_does_not_raise(self):
        provider = MockModelProvider(response="Sure! The intent is browser navigation.")
        node = AnalysisNode(model_provider=provider)
        # Must complete without raising
        state = run(node.process(make_state("open Safari")))
        assert state is not None

    def test_non_json_response_degrades_served_mode(self):
        provider = MockModelProvider(response="Sure! The intent is browser navigation.")
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.served_mode == "LEVEL_1_DEGRADED"

    def test_non_json_response_confidence_capped(self):
        provider = MockModelProvider(response="Sure! The intent is browser navigation.")
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.confidence <= 0.6

    def test_non_json_response_degraded_reason_code_set(self):
        provider = MockModelProvider(response="Sure! The intent is browser navigation.")
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.degraded_reason_code is not None
        assert len(state.degraded_reason_code) > 0

    def test_partial_json_missing_fields_degrades(self):
        """JSON missing required fields (intent/complexity) should degrade."""
        provider = MockModelProvider(response='{"reason": "unknown"}')
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("open Safari")))
        assert state.served_mode == "LEVEL_1_DEGRADED"


# ---------------------------------------------------------------------------
# TestBrainSelectorIntegration
# ---------------------------------------------------------------------------


class TestBrainSelectorIntegration:
    """AnalysisNode should accept an optional UnifiedBrainSelector."""

    def test_accepts_custom_brain_selector(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": []}'
        )
        selector = UnifiedBrainSelector()
        node = AnalysisNode(model_provider=provider, brain_selector=selector)
        state = run(node.process(make_state("navigate to linkedin")))
        assert state.analysis_brain_used != ""

    def test_analysis_brain_used_set_without_custom_selector(self):
        provider = MockModelProvider(
            response='{"intent": "browser_navigation", "complexity": "heavy",'
                     ' "confidence": 0.88, "goals": []}'
        )
        node = AnalysisNode(model_provider=provider)
        state = run(node.process(make_state("navigate to linkedin")))
        assert state.analysis_brain_used != ""
