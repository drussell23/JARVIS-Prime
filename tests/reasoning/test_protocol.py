"""
Tests for Mind-Body protocol v1.0.0 Pydantic schemas.
TDD: these tests are written BEFORE the implementation exists.
"""
from __future__ import annotations

import json
import uuid

import pytest

from jarvis_prime.reasoning.protocol import (
    PROTOCOL_VERSION,
    MIN_SUPPORTED_VERSION,
    MAX_SUPPORTED_VERSION,
    AuthEnvelope,
    Classification,
    Constraints,
    ErrorDetail,
    FallbackPolicy,
    Plan,
    ProtocolVersionInfo,
    ReasonFeedback,
    ReasonRequest,
    ReasonResponse,
    RoutingTrace,
    StepResult,
    SubGoal,
)


# ---------------------------------------------------------------------------
# PROTOCOL_VERSION constants
# ---------------------------------------------------------------------------

class TestProtocolVersionConstants:
    def test_current_version_format(self):
        assert PROTOCOL_VERSION == "1.0.0"

    def test_min_supported_version(self):
        assert MIN_SUPPORTED_VERSION == "1.0.0"

    def test_max_supported_version(self):
        assert MAX_SUPPORTED_VERSION == "1.0.999"


# ---------------------------------------------------------------------------
# AuthEnvelope
# ---------------------------------------------------------------------------

class TestAuthEnvelope:
    def test_default_construction(self):
        auth = AuthEnvelope()
        assert isinstance(auth.token_id, str)
        assert isinstance(auth.signature, str)
        assert isinstance(auth.nonce, str)
        assert isinstance(auth.issued_at, str)

    def test_roundtrip(self):
        auth = AuthEnvelope(
            token_id="tok-abc",
            signature="sig-xyz",
            nonce="n123",
            issued_at="2026-03-19T00:00:00Z",
        )
        restored = AuthEnvelope.model_validate(auth.model_dump())
        assert restored.token_id == "tok-abc"
        assert restored.signature == "sig-xyz"
        assert restored.nonce == "n123"
        assert restored.issued_at == "2026-03-19T00:00:00Z"

    def test_json_serializable(self):
        auth = AuthEnvelope(token_id="tok-1")
        raw = auth.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["token_id"] == "tok-1"


# ---------------------------------------------------------------------------
# FallbackPolicy
# ---------------------------------------------------------------------------

class TestFallbackPolicy:
    def test_default_construction(self):
        fp = FallbackPolicy()
        assert isinstance(fp.allow_tier1, bool)
        assert isinstance(fp.allow_tier2, bool)
        assert isinstance(fp.queue_on_fail, bool)
        assert isinstance(fp.require_approval_in_degraded, bool)

    def test_roundtrip(self):
        fp = FallbackPolicy(
            allow_tier1=True,
            allow_tier2=False,
            queue_on_fail=True,
            require_approval_in_degraded=True,
        )
        restored = FallbackPolicy.model_validate(fp.model_dump())
        assert restored.allow_tier1 is True
        assert restored.allow_tier2 is False
        assert restored.queue_on_fail is True
        assert restored.require_approval_in_degraded is True

    def test_json_serializable(self):
        fp = FallbackPolicy(allow_tier1=True)
        raw = fp.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["allow_tier1"] is True


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_default_all_optional(self):
        c = Constraints()
        assert c.deadline_ms is None
        assert c.deadline_at_ms is None
        assert c.hard_cost_cap_usd is None
        assert c.soft_cost_target_usd is None
        assert c.session_budget_remaining_usd is None
        assert isinstance(c.allowed_task_classes, list)

    def test_roundtrip_with_values(self):
        c = Constraints(
            deadline_ms=5000,
            deadline_at_ms=1_700_000_000_000,
            hard_cost_cap_usd=0.10,
            soft_cost_target_usd=0.05,
            session_budget_remaining_usd=1.00,
            allowed_task_classes=["code_gen", "reasoning"],
        )
        restored = Constraints.model_validate(c.model_dump())
        assert restored.deadline_ms == 5000
        assert restored.hard_cost_cap_usd == pytest.approx(0.10)
        assert restored.allowed_task_classes == ["code_gen", "reasoning"]

    def test_json_serializable(self):
        c = Constraints(deadline_ms=1000)
        raw = c.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["deadline_ms"] == 1000


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassification:
    def test_default_construction(self):
        cl = Classification()
        assert isinstance(cl.intent, str)
        assert isinstance(cl.complexity, str)
        assert isinstance(cl.confidence, float)
        assert isinstance(cl.brain_used, str)
        assert isinstance(cl.graph_depth, int)

    def test_roundtrip(self):
        cl = Classification(
            intent="code_generation",
            complexity="medium",
            confidence=0.95,
            brain_used="prime-7b",
            graph_depth=3,
        )
        restored = Classification.model_validate(cl.model_dump())
        assert restored.intent == "code_generation"
        assert restored.confidence == pytest.approx(0.95)
        assert restored.graph_depth == 3

    def test_json_serializable(self):
        cl = Classification(intent="chat")
        raw = cl.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["intent"] == "chat"


# ---------------------------------------------------------------------------
# SubGoal
# ---------------------------------------------------------------------------

class TestSubGoal:
    def test_default_construction(self):
        sg = SubGoal()
        assert isinstance(sg.step_id, str)
        assert isinstance(sg.action_id, str)
        assert isinstance(sg.goal, str)
        assert isinstance(sg.task_type, str)
        assert isinstance(sg.brain_assigned, str)
        assert isinstance(sg.tool_required, str)
        assert isinstance(sg.depends_on, list)
        assert isinstance(sg.estimated_ms, int)

    def test_roundtrip(self):
        sg = SubGoal(
            step_id="step-1",
            action_id="act-42",
            goal="Generate a sorting function",
            task_type="code_gen",
            brain_assigned="prime-7b",
            tool_required="code_writer",
            depends_on=["step-0"],
            estimated_ms=2000,
        )
        restored = SubGoal.model_validate(sg.model_dump())
        assert restored.step_id == "step-1"
        assert restored.depends_on == ["step-0"]
        assert restored.estimated_ms == 2000

    def test_json_serializable(self):
        sg = SubGoal(step_id="s1", goal="do something")
        raw = sg.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["step_id"] == "s1"


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class TestPlan:
    def test_default_construction(self):
        p = Plan()
        assert isinstance(p.plan_id, str)
        assert isinstance(p.plan_hash, str)
        assert isinstance(p.sub_goals, list)
        assert isinstance(p.execution_strategy, str)
        assert isinstance(p.total_estimated_ms, int)
        assert isinstance(p.risk_level, str)
        assert isinstance(p.approval_required, bool)
        assert isinstance(p.approval_reason_codes, list)
        assert isinstance(p.approval_scope, str)

    def test_roundtrip_with_sub_goals(self):
        sg = SubGoal(step_id="s1", goal="Write code")
        p = Plan(
            plan_id="plan-abc",
            plan_hash="sha256:deadbeef",
            sub_goals=[sg],
            execution_strategy="sequential",
            total_estimated_ms=5000,
            risk_level="low",
            approval_required=False,
            approval_reason_codes=[],
            approval_scope="local",
        )
        restored = Plan.model_validate(p.model_dump())
        assert restored.plan_id == "plan-abc"
        assert len(restored.sub_goals) == 1
        assert restored.sub_goals[0].step_id == "s1"

    def test_json_serializable(self):
        p = Plan(plan_id="p1")
        raw = p.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["plan_id"] == "p1"


# ---------------------------------------------------------------------------
# RoutingTrace
# ---------------------------------------------------------------------------

class TestRoutingTrace:
    def test_default_construction(self):
        rt = RoutingTrace()
        assert isinstance(rt.analysis_brain, str)
        assert isinstance(rt.planning_brain, str)
        assert isinstance(rt.cost_gate_passed, bool)
        assert isinstance(rt.resource_gate_passed, bool)
        assert isinstance(rt.sai_health, str)
        assert isinstance(rt.cai_confidence, float)

    def test_roundtrip(self):
        rt = RoutingTrace(
            analysis_brain="classifier-v2",
            planning_brain="prime-7b",
            cost_gate_passed=True,
            resource_gate_passed=True,
            sai_health="healthy",
            cai_confidence=0.88,
        )
        restored = RoutingTrace.model_validate(rt.model_dump())
        assert restored.analysis_brain == "classifier-v2"
        assert restored.cai_confidence == pytest.approx(0.88)

    def test_json_serializable(self):
        rt = RoutingTrace(sai_health="degraded")
        raw = rt.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["sai_health"] == "degraded"


# ---------------------------------------------------------------------------
# ErrorDetail  (has alias: error_class → "class")
# ---------------------------------------------------------------------------

class TestErrorDetail:
    def test_default_construction(self):
        e = ErrorDetail()
        assert isinstance(e.code, str)
        assert isinstance(e.error_class, str)
        assert isinstance(e.message, str)
        assert e.retry_after_ms is None
        assert isinstance(e.recovery_strategy, str)

    def test_alias_class_in_json(self):
        """Serializing with by_alias=True should produce "class" key in JSON."""
        e = ErrorDetail(code="E001", error_class="auth_failure", message="Token expired")
        dumped = e.model_dump(by_alias=True)
        assert "class" in dumped
        assert dumped["class"] == "auth_failure"

    def test_populate_by_name(self):
        """model_validate should accept both 'error_class' and 'class' keys."""
        # via field name
        e1 = ErrorDetail.model_validate({"code": "E002", "error_class": "rate_limit"})
        assert e1.error_class == "rate_limit"
        # via alias
        e2 = ErrorDetail.model_validate({"code": "E002", "class": "rate_limit"})
        assert e2.error_class == "rate_limit"

    def test_roundtrip(self):
        e = ErrorDetail(
            code="E003",
            error_class="timeout",
            message="Deadline exceeded",
            retry_after_ms=3000,
            recovery_strategy="retry",
        )
        restored = ErrorDetail.model_validate(e.model_dump())
        assert restored.code == "E003"
        assert restored.error_class == "timeout"
        assert restored.retry_after_ms == 3000

    def test_json_serializable(self):
        e = ErrorDetail(code="E001", error_class="unknown")
        raw = e.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["code"] == "E001"

    def test_error_codes_variety(self):
        codes = ["AUTH_FAILED", "RATE_LIMITED", "TIMEOUT", "RESOURCE_EXHAUSTED", "UNKNOWN"]
        for code in codes:
            e = ErrorDetail(code=code)
            assert e.code == code


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------

class TestStepResult:
    def test_default_construction(self):
        sr = StepResult()
        assert isinstance(sr.step_id, str)
        assert isinstance(sr.action_id, str)
        assert isinstance(sr.success, bool)
        assert isinstance(sr.output, str)
        assert isinstance(sr.latency_ms, int)
        assert isinstance(sr.tool_used, str)
        assert isinstance(sr.artifact_refs, list)
        assert isinstance(sr.side_effects_committed, bool)

    def test_roundtrip(self):
        sr = StepResult(
            step_id="s1",
            action_id="act-1",
            success=True,
            output="def sort(arr): return sorted(arr)",
            latency_ms=450,
            tool_used="code_writer",
            artifact_refs=["file://main.py"],
            side_effects_committed=True,
        )
        restored = StepResult.model_validate(sr.model_dump())
        assert restored.step_id == "s1"
        assert restored.success is True
        assert restored.latency_ms == 450
        assert restored.artifact_refs == ["file://main.py"]

    def test_json_serializable(self):
        sr = StepResult(step_id="s1", success=True)
        raw = sr.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["step_id"] == "s1"
        assert parsed["success"] is True


# ---------------------------------------------------------------------------
# ReasonRequest
# ---------------------------------------------------------------------------

class TestReasonRequest:
    def test_minimal_construction(self):
        req = ReasonRequest(request_id="req-001", command="unlock my screen")
        assert req.request_id == "req-001"
        assert req.command == "unlock my screen"
        assert req.protocol_version == PROTOCOL_VERSION

    def test_full_roundtrip(self):
        req = ReasonRequest(
            protocol_version="1.0.0",
            request_id="req-abc",
            idempotency_scope="session-xyz",
            session_id="sess-1",
            trace_id="trace-99",
            parent_request_id="req-000",
            auth=AuthEnvelope(token_id="tok-1"),
            command="write a sorting algorithm",
            context={"language": "python", "repo": "jarvis"},
            constraints=Constraints(deadline_ms=10_000),
            fallback_policy=FallbackPolicy(allow_tier1=True),
        )
        restored = ReasonRequest.model_validate(req.model_dump())
        assert restored.request_id == "req-abc"
        assert restored.command == "write a sorting algorithm"
        assert restored.auth.token_id == "tok-1"
        assert restored.constraints.deadline_ms == 10_000
        assert restored.context["language"] == "python"

    def test_json_serializable(self):
        req = ReasonRequest(request_id="r1", command="hello")
        raw = req.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["request_id"] == "r1"
        assert parsed["command"] == "hello"

    def test_protocol_version_default(self):
        req = ReasonRequest(request_id="r2", command="test")
        assert req.protocol_version == PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# ReasonResponse
# ---------------------------------------------------------------------------

class TestReasonResponse:
    def test_minimal_construction(self):
        resp = ReasonResponse(request_id="req-001")
        assert resp.request_id == "req-001"
        assert resp.protocol_version == PROTOCOL_VERSION

    def test_with_plan_and_classification(self):
        cl = Classification(intent="code_gen", complexity="medium", confidence=0.92)
        sg = SubGoal(step_id="s1", goal="Write function")
        plan = Plan(plan_id="plan-1", sub_goals=[sg])
        rt = RoutingTrace(analysis_brain="cls-v1", planning_brain="prime-7b")

        resp = ReasonResponse(
            protocol_version="1.0.0",
            request_id="req-001",
            session_id="sess-1",
            trace_id="trace-1",
            status="success",
            served_mode="prime",
            requested_mode="prime",
            classification=cl,
            plan=plan,
            routing_trace=rt,
        )
        restored = ReasonResponse.model_validate(resp.model_dump())
        assert restored.status == "success"
        assert restored.classification.intent == "code_gen"
        assert len(restored.plan.sub_goals) == 1
        assert restored.routing_trace.planning_brain == "prime-7b"

    def test_with_sub_goals_list(self):
        resp = ReasonResponse(
            request_id="req-002",
            plan=Plan(
                sub_goals=[
                    SubGoal(step_id="s1", goal="Step one"),
                    SubGoal(step_id="s2", goal="Step two", depends_on=["s1"]),
                ]
            ),
        )
        assert len(resp.plan.sub_goals) == 2
        assert resp.plan.sub_goals[1].depends_on == ["s1"]

    def test_degraded_with_error(self):
        err = ErrorDetail(code="TIMEOUT", error_class="timeout", message="Deadline exceeded")
        resp = ReasonResponse(
            request_id="req-003",
            status="degraded",
            degraded_reason_code="timeout",
            error=err,
        )
        assert resp.status == "degraded"
        assert resp.error.code == "TIMEOUT"

    def test_json_serializable(self):
        resp = ReasonResponse(request_id="r1", status="success")
        raw = resp.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["request_id"] == "r1"


# ---------------------------------------------------------------------------
# ReasonFeedback
# ---------------------------------------------------------------------------

class TestReasonFeedback:
    def test_minimal_construction(self):
        fb = ReasonFeedback(request_id="req-001")
        assert fb.request_id == "req-001"
        assert isinstance(fb.step_results, list)

    def test_roundtrip_with_step_results(self):
        sr1 = StepResult(step_id="s1", success=True, latency_ms=300)
        sr2 = StepResult(step_id="s2", success=False, latency_ms=150)
        fb = ReasonFeedback(
            request_id="req-xyz",
            session_id="sess-1",
            trace_id="trace-2",
            plan_id="plan-1",
            plan_hash="sha256:abcdef",
            step_results=[sr1, sr2],
            final_outcome="partial",
            replay_token="rp-tok-1",
            original_enqueued_at="2026-03-19T01:00:00Z",
            replay_attempt=1,
            max_replay_attempts=3,
        )
        restored = ReasonFeedback.model_validate(fb.model_dump())
        assert restored.request_id == "req-xyz"
        assert len(restored.step_results) == 2
        assert restored.step_results[0].step_id == "s1"
        assert restored.step_results[1].success is False
        assert restored.replay_attempt == 1
        assert restored.max_replay_attempts == 3

    def test_json_serializable(self):
        fb = ReasonFeedback(request_id="r1")
        raw = fb.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["request_id"] == "r1"

    def test_empty_step_results_default(self):
        fb = ReasonFeedback(request_id="r2")
        assert fb.step_results == []


# ---------------------------------------------------------------------------
# ProtocolVersionInfo
# ---------------------------------------------------------------------------

class TestProtocolVersionInfo:
    def test_minimal_construction(self):
        pvi = ProtocolVersionInfo()
        assert pvi.current_version == PROTOCOL_VERSION
        assert isinstance(pvi.min_supported_version, str)
        assert isinstance(pvi.max_supported_version, str)
        assert isinstance(pvi.features, list)
        assert isinstance(pvi.brain_policy_hash, str)

    def test_features_list(self):
        pvi = ProtocolVersionInfo(
            features=["streaming", "multi_step_planning", "cost_gates", "approval_flow"]
        )
        assert "streaming" in pvi.features
        assert "multi_step_planning" in pvi.features

    def test_roundtrip(self):
        pvi = ProtocolVersionInfo(
            current_version="1.0.0",
            min_supported_version="1.0.0",
            max_supported_version="1.0.999",
            features=["streaming"],
            brain_policy_hash="sha256:cafebabe",
        )
        restored = ProtocolVersionInfo.model_validate(pvi.model_dump())
        assert restored.current_version == "1.0.0"
        assert restored.features == ["streaming"]
        assert restored.brain_policy_hash == "sha256:cafebabe"

    def test_version_fields_match_constants(self):
        pvi = ProtocolVersionInfo()
        assert pvi.current_version == PROTOCOL_VERSION
        assert pvi.min_supported_version == MIN_SUPPORTED_VERSION
        assert pvi.max_supported_version == MAX_SUPPORTED_VERSION

    def test_json_serializable(self):
        pvi = ProtocolVersionInfo(features=["f1", "f2"])
        raw = pvi.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["features"] == ["f1", "f2"]
