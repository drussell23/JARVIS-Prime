"""Tests for POST /v1/vision/analyze endpoint."""
import pytest
from jarvis_prime.reasoning.vision_assist import handle_vision_analyze, VisionAnalyzeRequest


class TestHandleVisionAnalyze:
    @pytest.mark.asyncio
    async def test_find_element_returns_coords(self):
        # Mock model to return element detection
        from jarvis_prime.reasoning.endpoints import set_model_provider
        from jarvis_prime.reasoning.model_provider import MockModelProvider
        set_model_provider(MockModelProvider(
            response='{"elements": [{"description": "Submit button", "coords": [523, 187], "confidence": 0.92, "element_type": "button", "text_content": "Submit", "interactable": true}], "scene_summary": "A form with submit button"}'
        ))
        req = VisionAnalyzeRequest(
            request_id="vis-001",
            session_id="sess-001",
            trace_id="trace-001",
            frame={"artifact_ref": "test.jpg", "width": 1440, "height": 900,
                   "scale_factor": 2.0, "captured_at_ms": 0, "display_id": 0},
            task={"type": "find_element", "target_description": "submit button",
                  "action_intent": "click"},
        )
        result = await handle_vision_analyze(req)
        assert result["status"] == "found"
        assert len(result["elements"]) >= 1
        assert result["elements"][0]["coords"] == [523, 187]
        assert result["elements"][0]["confidence"] >= 0.8

    @pytest.mark.asyncio
    async def test_not_found(self):
        from jarvis_prime.reasoning.endpoints import set_model_provider
        from jarvis_prime.reasoning.model_provider import MockModelProvider
        set_model_provider(MockModelProvider(
            response='{"elements": [], "scene_summary": "Empty desktop"}'
        ))
        req = VisionAnalyzeRequest(
            request_id="vis-002",
            session_id="sess-001",
            trace_id="trace-001",
            frame={"artifact_ref": "test.jpg", "width": 1440, "height": 900,
                   "scale_factor": 2.0, "captured_at_ms": 0, "display_id": 0},
            task={"type": "find_element", "target_description": "nonexistent button"},
        )
        result = await handle_vision_analyze(req)
        assert result["status"] == "not_found"

    @pytest.mark.asyncio
    async def test_model_failure_returns_error(self):
        from jarvis_prime.reasoning.endpoints import set_model_provider
        from jarvis_prime.reasoning.model_provider import MockModelProvider
        set_model_provider(MockModelProvider(should_fail=True))
        req = VisionAnalyzeRequest(
            request_id="vis-003",
            session_id="sess-001",
            trace_id="trace-001",
            frame={"artifact_ref": "test.jpg", "width": 1440, "height": 900,
                   "scale_factor": 2.0, "captured_at_ms": 0, "display_id": 0},
            task={"type": "find_element", "target_description": "button"},
        )
        result = await handle_vision_analyze(req)
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_response_has_coord_space(self):
        from jarvis_prime.reasoning.endpoints import set_model_provider
        from jarvis_prime.reasoning.model_provider import MockModelProvider
        set_model_provider(MockModelProvider(
            response='{"elements": [{"description": "btn", "coords": [100, 100], "confidence": 0.9}], "scene_summary": "test"}'
        ))
        req = VisionAnalyzeRequest(
            request_id="vis-004", session_id="s", trace_id="t",
            frame={"artifact_ref": "test.jpg", "width": 1440, "height": 900,
                   "scale_factor": 2.0, "captured_at_ms": 0, "display_id": 0},
            task={"type": "find_element", "target_description": "button"},
        )
        result = await handle_vision_analyze(req)
        assert result["coord_space"] == "logical_pixels"

    @pytest.mark.asyncio
    async def test_response_has_model_used(self):
        from jarvis_prime.reasoning.endpoints import set_model_provider
        from jarvis_prime.reasoning.model_provider import MockModelProvider
        set_model_provider(MockModelProvider(
            response='{"elements": [{"coords": [100, 100], "confidence": 0.9}], "scene_summary": "t"}'
        ))
        req = VisionAnalyzeRequest(
            request_id="vis-005", session_id="s", trace_id="t",
            frame={"artifact_ref": "t.jpg", "width": 1440, "height": 900,
                   "scale_factor": 2.0, "captured_at_ms": 0, "display_id": 0},
            task={"type": "find_element", "target_description": "btn"},
        )
        result = await handle_vision_analyze(req)
        assert "model_used" in result
