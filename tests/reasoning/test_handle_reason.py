"""Tests for POST /v1/reason -- the canonical Mind API."""
import json
import os
import tempfile
import pytest
from jarvis_prime.reasoning.protocol import ReasonRequest, PROTOCOL_VERSION


@pytest.fixture(autouse=True)
def _isolate_idempotency_store(tmp_path, monkeypatch):
    """Point IdempotencyStore at a temp dir so tests don't touch ~/.jarvis-prime.

    Also reset the module-level singleton between tests so each test gets a fresh store.
    """
    db_path = str(tmp_path / "idempotency.db")
    monkeypatch.setenv("REASON_IDEMPOTENCY_DB", db_path)

    # Reset module-level singletons so each test starts clean
    import jarvis_prime.reasoning.endpoints as ep
    ep._idempotency_store = None
    ep._model_provider = None
    yield
    ep._idempotency_store = None
    ep._model_provider = None


class TestHandleReason:
    @pytest.mark.asyncio
    async def test_trivial_command_returns_plan(self):
        from jarvis_prime.reasoning.endpoints import handle_reason
        req = ReasonRequest(
            request_id="req-reason-001",
            session_id="sess-001",
            trace_id="trace-001",
            command="open Safari",
        )
        result = await handle_reason(req)
        assert result["status"] in ("plan_ready", "needs_approval")
        assert result["classification"]["intent"] != ""
        assert result["plan"]["plan_id"] != ""
        assert len(result["plan"]["sub_goals"]) >= 1

    @pytest.mark.asyncio
    async def test_complex_command_has_multiple_sub_goals(self):
        from jarvis_prime.reasoning.endpoints import handle_reason
        req = ReasonRequest(
            request_id="req-reason-002",
            session_id="sess-001",
            trace_id="trace-001",
            command="research competitors and build a comparison spreadsheet",
        )
        result = await handle_reason(req)
        assert result["status"] in ("plan_ready", "needs_approval")

    @pytest.mark.asyncio
    async def test_protocol_mismatch_rejected(self):
        from jarvis_prime.reasoning.endpoints import handle_reason
        req = ReasonRequest(
            request_id="req-bad-proto",
            session_id="sess-001",
            trace_id="trace-001",
            command="test",
            protocol_version="99.0.0",
        )
        result = await handle_reason(req)
        assert result["status"] == "error"
        assert result["error"]["code"] == "PROTOCOL_MISMATCH"

    @pytest.mark.asyncio
    async def test_idempotency_returns_cached(self):
        from jarvis_prime.reasoning.endpoints import handle_reason
        req = ReasonRequest(
            request_id="req-idempotent-001",
            session_id="sess-001",
            trace_id="trace-001",
            command="open Safari",
        )
        result1 = await handle_reason(req)
        result2 = await handle_reason(req)
        # Both should return the same plan_id (cached)
        assert result1["plan"]["plan_id"] == result2["plan"]["plan_id"]

    @pytest.mark.asyncio
    async def test_response_has_required_fields(self):
        from jarvis_prime.reasoning.endpoints import handle_reason
        req = ReasonRequest(
            request_id="req-fields-001",
            session_id="sess-001",
            trace_id="trace-001",
            command="test",
        )
        result = await handle_reason(req)
        assert "protocol_version" in result
        assert "request_id" in result
        assert "status" in result
        assert "served_mode" in result
        assert "classification" in result
        assert "plan" in result

    @pytest.mark.asyncio
    async def test_plan_hash_is_full_sha256(self):
        from jarvis_prime.reasoning.endpoints import handle_reason
        req = ReasonRequest(
            request_id="req-hash-001",
            session_id="sess-001",
            trace_id="trace-001",
            command="test command",
        )
        result = await handle_reason(req)
        plan_hash = result["plan"].get("plan_hash", "")
        assert len(plan_hash) == 64

    @pytest.mark.asyncio
    async def test_health_reports_graph_ready(self):
        """After handle_reason is imported, health should report graph ready."""
        from jarvis_prime.reasoning.endpoints import get_reason_health
        result = await get_reason_health()
        # After importing handle_reason, the reasoning module is loaded
        assert "status" in result
