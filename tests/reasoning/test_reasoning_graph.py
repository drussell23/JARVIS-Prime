"""Tests for the ONE thinking pipeline -- ReasoningGraph."""
import pytest
from jarvis_prime.reasoning.model_provider import MockModelProvider
from jarvis_prime.reasoning.reasoning_graph import ReasoningGraph


@pytest.fixture
def trivial_graph():
    """Graph with model that classifies everything as trivial."""
    provider = MockModelProvider(
        response='{"intent": "system_command", "complexity": "trivial", "confidence": 0.95, "goals": ["open Safari"]}'
    )
    return ReasoningGraph(model_provider=provider)


@pytest.fixture
def complex_graph():
    """Graph with model that classifies as complex and decomposes into sub-goals."""
    # The model returns different things for analysis vs planning prompts.
    # MockModelProvider returns the same response for all calls, so we test
    # that the graph handles the complex path correctly even with this response.
    provider = MockModelProvider(
        response='{"intent": "multi_step_planning", "complexity": "complex", "confidence": 0.85, "goals": ["research competitors", "build spreadsheet"], "sub_goals": [{"goal": "research competitors", "task_type": "browser_navigation"}, {"goal": "build spreadsheet", "task_type": "system_command"}], "execution_strategy": "sequential"}'
    )
    return ReasoningGraph(model_provider=provider)


@pytest.fixture
def failing_graph():
    """Graph where model always fails -- degraded mode."""
    provider = MockModelProvider(should_fail=True)
    return ReasoningGraph(model_provider=provider)


class TestTrivialPath:
    @pytest.mark.asyncio
    async def test_trivial_returns_plan_ready(self, trivial_graph):
        result = await trivial_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="open Safari",
        )
        assert result["status"] in ("plan_ready", "needs_approval")
        assert result["classification"]["complexity"] in ("trivial", "light")

    @pytest.mark.asyncio
    async def test_trivial_has_plan(self, trivial_graph):
        result = await trivial_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="open Safari",
        )
        assert "plan" in result
        assert result["plan"]["plan_id"] != ""
        assert len(result["plan"]["sub_goals"]) >= 1

    @pytest.mark.asyncio
    async def test_trivial_skips_planning_node(self, trivial_graph):
        result = await trivial_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="open Safari",
        )
        trace = result.get("routing_trace", {})
        # planning_brain should be empty for trivial (planning was skipped)
        assert trace.get("planning_brain", "") == ""

    @pytest.mark.asyncio
    async def test_trivial_still_validated(self, trivial_graph):
        result = await trivial_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="open Safari",
        )
        # Should NOT have VALIDATION_UNAVAILABLE -- validation ran
        plan = result.get("plan", {})
        reasons = plan.get("approval_reason_codes", [])
        assert "VALIDATION_UNAVAILABLE" not in reasons


class TestComplexPath:
    @pytest.mark.asyncio
    async def test_complex_runs_full_pipeline(self, complex_graph):
        result = await complex_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="research competitors and build spreadsheet",
        )
        assert result["status"] in ("plan_ready", "needs_approval")
        trace = result.get("routing_trace", {})
        # Planning brain should be set (planning node ran)
        assert trace.get("planning_brain", "") != ""

    @pytest.mark.asyncio
    async def test_complex_has_plan_hash(self, complex_graph):
        result = await complex_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="research competitors and build spreadsheet",
        )
        plan = result.get("plan", {})
        assert plan.get("plan_hash", "") != ""
        assert len(plan.get("plan_hash", "")) == 64  # full SHA-256

    @pytest.mark.asyncio
    async def test_complex_has_multiple_sub_goals(self, complex_graph):
        result = await complex_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="research competitors and build spreadsheet",
        )
        plan = result.get("plan", {})
        assert len(plan.get("sub_goals", [])) >= 2


class TestDegradedMode:
    @pytest.mark.asyncio
    async def test_model_failure_returns_degraded(self, failing_graph):
        result = await failing_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="analyze competitor strategy",
        )
        assert result["served_mode"] == "LEVEL_1_DEGRADED"
        assert result["classification"]["confidence"] <= 0.6
        assert result.get("degraded_reason_code") is not None

    @pytest.mark.asyncio
    async def test_degraded_still_has_plan(self, failing_graph):
        result = await failing_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="open Safari",
        )
        assert "plan" in result
        assert result["plan"]["plan_id"] != ""


class TestResponseFormat:
    @pytest.mark.asyncio
    async def test_has_required_fields(self, trivial_graph):
        result = await trivial_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="test",
        )
        assert "protocol_version" in result
        assert "request_id" in result
        assert "session_id" in result
        assert "status" in result
        assert "served_mode" in result
        assert "classification" in result
        assert "plan" in result
        assert "routing_trace" in result

    @pytest.mark.asyncio
    async def test_classification_fields(self, trivial_graph):
        result = await trivial_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="test",
        )
        cls = result["classification"]
        assert "intent" in cls
        assert "complexity" in cls
        assert "confidence" in cls
        assert "brain_used" in cls
        assert "graph_depth" in cls

    @pytest.mark.asyncio
    async def test_plan_fields(self, trivial_graph):
        result = await trivial_graph.run(
            request_id="r1", session_id="s1", trace_id="t1",
            command="test",
        )
        plan = result["plan"]
        assert "plan_id" in plan
        assert "plan_hash" in plan
        assert "sub_goals" in plan
        assert "execution_strategy" in plan
        assert "approval_required" in plan
