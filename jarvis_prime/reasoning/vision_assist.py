"""
Vision Assist — POST /v1/vision/analyze endpoint handler.

Accepts a frame reference + target description and returns UI element
coordinates with confidence scores.

Design
------
- VisionAnalyzeRequest: Pydantic model for the wire format.
- handle_vision_analyze: pure async handler backed by the injected ModelProvider.
- Uses _get_model_provider() from endpoints.py — same DI seam as the reasoning graph.
- In production the provider is swapped to a vision-capable model (LLaVA) via
  set_model_provider().  With MockModelProvider the frame is described in the
  user message text; the canned JSON response is parsed normally.
- coord_space is always "logical_pixels" — callers must scale by display
  scale_factor themselves when issuing OS-level clicks.
- JSON parse errors and model failures both return status="error" so callers
  can handle them uniformly without inspecting exception types.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a UI element detector. Given a screenshot description and a target "
    "element, identify matching UI elements. Return JSON:\n"
    '{"elements": [{"description": "...", "coords": [x, y], "confidence": 0.0-1.0,\n'
    '  "element_type": "button|input|link|...", "text_content": "...", '
    '"interactable": true|false}],\n'
    ' "scene_summary": "brief description of what\'s on screen"}\n'
    "Only return valid JSON."
)

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class VisionAnalyzeRequest(BaseModel):
    """Wire format for POST /v1/vision/analyze."""

    request_id: str
    session_id: str
    trace_id: str

    # Frame metadata — in production artifact_ref is resolved to raw bytes
    # before being sent to the vision model.  Here it is passed as context.
    frame: Dict[str, Any] = Field(
        ...,
        description=(
            "Frame descriptor: artifact_ref, width, height, scale_factor, "
            "captured_at_ms, display_id"
        ),
    )

    # Task descriptor
    task: Dict[str, Any] = Field(
        ...,
        description=(
            "Task: type (find_element), target_description, "
            "target_context (optional), action_intent (optional)"
        ),
    )

    # Optional constraints with sensible defaults
    constraints: Dict[str, Any] = Field(
        default_factory=lambda: {"timeout_ms": 5000, "max_elements": 10},
        description="timeout_ms and max_elements",
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


async def handle_vision_analyze(req: VisionAnalyzeRequest) -> Dict[str, Any]:
    """
    Handler for POST /v1/vision/analyze.

    Steps
    -----
    1. Obtain the model provider via _get_model_provider() from endpoints.py.
    2. Build a system + user message describing the frame and the target.
    3. Call provider.infer() with low temperature (0.1) for deterministic JSON.
    4. Parse the JSON response and extract the elements list.
    5. Return a response dict with:
         status          "found" | "not_found" | "error"
         elements        list of detected UI elements (empty on not_found/error)
         scene_summary   brief description of the screen
         model_used      provider.loaded_model_name()
         inference_latency_ms  wall-clock ms for the infer() call
         coord_space     always "logical_pixels"
         request_id      echo of the incoming request_id

    Error handling
    --------------
    - RuntimeError from provider.infer() (e.g. model not loaded, mock failure)
      → status="error", error_detail contains the exception message.
    - json.JSONDecodeError / KeyError during response parsing
      → status="error", error_detail contains parse failure context.
    """
    from jarvis_prime.reasoning.endpoints import _get_model_provider  # lazy import avoids circularity

    provider = _get_model_provider()
    model_name = provider.loaded_model_name()

    target_description: str = req.task.get("target_description", "")
    action_intent: str = req.task.get("action_intent", "")
    target_context: str = req.task.get("target_context", "")
    task_type: str = req.task.get("type", "find_element")

    frame_width: int = req.frame.get("width", 0)
    frame_height: int = req.frame.get("height", 0)
    scale_factor: float = req.frame.get("scale_factor", 1.0)
    artifact_ref: str = req.frame.get("artifact_ref", "")
    display_id: int = req.frame.get("display_id", 0)

    max_elements: int = req.constraints.get("max_elements", 10)
    timeout_ms: int = req.constraints.get("timeout_ms", 5000)
    timeout_s: float = timeout_ms / 1000.0

    # Build user message — describes the frame in text so MockModelProvider
    # can handle it; a vision model would receive the image bytes alongside.
    user_message = (
        f"Task: {task_type}\n"
        f"Target: {target_description}\n"
    )
    if action_intent:
        user_message += f"Action intent: {action_intent}\n"
    if target_context:
        user_message += f"Context: {target_context}\n"
    user_message += (
        f"\nFrame: {artifact_ref} ({frame_width}x{frame_height} logical pixels, "
        f"scale={scale_factor}, display={display_id})\n"
        f"Return at most {max_elements} elements."
    )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    t_start = time.monotonic()
    try:
        raw = await provider.infer(
            messages=messages,
            max_tokens=512,
            temperature=0.1,
            timeout_s=timeout_s,
        )
        inference_latency_ms = int((time.monotonic() - t_start) * 1000)
    except Exception as exc:
        logger.warning(
            "vision_analyze infer failed: request_id=%s error=%s",
            req.request_id,
            exc,
        )
        return {
            "request_id": req.request_id,
            "status": "error",
            "elements": [],
            "scene_summary": "",
            "model_used": model_name,
            "inference_latency_ms": int((time.monotonic() - t_start) * 1000),
            "coord_space": "logical_pixels",
            "error_detail": str(exc),
        }

    # Parse the JSON payload from the model's content field
    content: str = raw.get("content", "")
    try:
        parsed: Dict[str, Any] = json.loads(content)
        elements: List[Dict[str, Any]] = parsed.get("elements", [])
        scene_summary: str = parsed.get("scene_summary", "")
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "vision_analyze JSON parse failed: request_id=%s error=%s content=%r",
            req.request_id,
            exc,
            content[:200],
        )
        return {
            "request_id": req.request_id,
            "status": "error",
            "elements": [],
            "scene_summary": "",
            "model_used": model_name,
            "inference_latency_ms": inference_latency_ms,
            "coord_space": "logical_pixels",
            "error_detail": f"JSON parse error: {exc}",
        }

    # Determine status
    status: str = "found" if elements else "not_found"

    logger.info(
        "vision_analyze: request_id=%s status=%s elements=%d latency_ms=%d",
        req.request_id,
        status,
        len(elements),
        inference_latency_ms,
    )

    return {
        "request_id": req.request_id,
        "status": status,
        "elements": elements,
        "scene_summary": scene_summary,
        "model_used": model_name,
        "inference_latency_ms": inference_latency_ms,
        "coord_space": "logical_pixels",
    }
