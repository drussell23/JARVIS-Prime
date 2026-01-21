#!/usr/bin/env python3
"""
JARVIS-Prime Server - Quick Start Script (v93.2 with Immediate HTTP Startup)
=============================================================================

CRITICAL FIX v93.2: HTTP server starts IMMEDIATELY, heavy initialization runs
in background. This solves the 61.9s timeout issue where ML imports blocked
the server from responding to health checks.

Runs JARVIS-Prime with llama-cpp-python backend.
Integrates with main JARVIS infrastructure for unified cost tracking.

Usage:
    # Default (TinyLlama on port 8000)
    python run_server.py

    # Custom model
    python run_server.py --model models/mistral-7b.gguf --port 8080

    # With Metal GPU (M1/M2/M3)
    python run_server.py --gpu-layers -1

    # Connect to JARVIS infrastructure (default: auto-detect)
    python run_server.py --bridge-enabled

Endpoints:
    POST /v1/chat/completions  - OpenAI-compatible chat
    POST /generate             - Simple text generation
    GET  /health               - Health check (IMMEDIATE response)
    GET  /metrics              - Cost tracking & inference metrics
"""

# =============================================================================
# v93.16: COMPREHENSIVE Warning Suppression - BEFORE any imports
# =============================================================================
# Set environment variable FIRST to suppress warnings at the interpreter level
import os
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning')
# Also set TF and other library-specific environment variables
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # Suppress TensorFlow warnings
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')  # Suppress transformers warnings
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')  # Suppress tokenizers warning

import warnings
import sys

# v93.16: Use simplefilter FIRST to catch everything, then add specific filters
warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.simplefilter('ignore', category=FutureWarning)

# v93.16: Specific filters for known warning messages
warnings.filterwarnings('ignore', message='.*urllib3.*OpenSSL.*LibreSSL.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='urllib3')
warnings.filterwarnings('ignore', message='.*Torch version.*has not been tested.*')
warnings.filterwarnings('ignore', message='.*coremltools.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', message='.*scikit-learn version.*is not supported.*')
warnings.filterwarnings('ignore', message='.*Disabling scikit-learn conversion API.*')
warnings.filterwarnings('ignore', message='.*Minimum required version.*')
warnings.filterwarnings('ignore', message='.*Maximum required version.*')

# v93.16: Aggressive suppression for coremltools at module level
warnings.filterwarnings('ignore', module='coremltools.*')
warnings.filterwarnings('ignore', module='sklearn.*')
warnings.filterwarnings('ignore', module='torch.*')

# v95.0: Override warnings.warn to filter at the source
# CRITICAL: Accept all parameters that Python's warnings.warn() can receive
# including 'source' (added in Python 3.6) to prevent TypeError
_original_warn = warnings.warn
def _filtered_warn(message, category=UserWarning, stacklevel=1, source=None):
    """
    Filtered warn that suppresses known non-critical warnings.

    v95.0: Added 'source' parameter to match Python's warnings.warn() signature.
    Without this, any library calling warnings.warn(..., source=something)
    would raise: TypeError: _filtered_warn() got an unexpected keyword argument 'source'
    """
    msg_str = str(message).lower()
    suppress_patterns = [
        'scikit-learn', 'coremltools', 'torch version', 'not supported',
        'has not been tested', 'minimum required', 'maximum required',
        'disabling', 'conversion api', 'urllib3', 'libressl', 'openssl'
    ]
    if any(pattern in msg_str for pattern in suppress_patterns):
        return  # Suppress
    # Pass source parameter to original warn if provided
    if source is not None:
        _original_warn(message, category, stacklevel + 1, source=source)
    else:
        _original_warn(message, category, stacklevel + 1)
warnings.warn = _filtered_warn

# =============================================================================
# MINIMAL IMPORTS ONLY - Heavy imports happen in background_initialization
# =============================================================================
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis-prime")


def parse_args():
    parser = argparse.ArgumentParser(description="JARVIS-Prime Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--model", default="models/current.gguf", help="Model path")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU layers (-1 for all)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    parser.add_argument(
        "--bridge-enabled",
        action="store_true",
        default=True,
        help="Enable cross-repo bridge for JARVIS integration (default: True)"
    )
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="Disable cross-repo bridge"
    )
    return parser.parse_args()


class StartupState:
    """
    v93.2: Track server startup state for immediate health checks.

    This allows the HTTP server to start IMMEDIATELY and respond to health
    checks while heavy initialization (ML imports, model loading) happens
    in the background.
    """
    def __init__(self):
        self.phase = "starting"  # starting -> initializing -> loading_model -> ready | error
        self.start_time = time.time()
        self.error: Optional[str] = None
        self.init_elapsed: Optional[float] = None
        self.model_load_start: Optional[float] = None
        self.model_load_elapsed: Optional[float] = None
        self.model_path: Optional[str] = None
        self.model_loaded: bool = False
        self.details: Dict[str, Any] = {}

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status for health endpoint.

        v93.7: Enhanced with detailed step information and loading progress.
        Reports model_load_elapsed_seconds DURING loading (not just after)
        to enable intelligent progress-based timeout extension.
        """
        elapsed = time.time() - self.start_time
        result = {
            # v93.13: Add "service" field for Trinity cross-repo discovery
            # TrinityIntegrator._discover_running_component() checks this field
            # to verify it's talking to the correct service during startup
            "service": "jarvis_prime",
            "status": "error" if self.error else ("healthy" if self.phase == "ready" else "starting"),
            "phase": self.phase,
            "startup_elapsed_seconds": round(elapsed, 1),
            "pid": os.getpid(),
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
        }
        if self.init_elapsed:
            result["init_elapsed_seconds"] = round(self.init_elapsed, 1)

        # v93.7: Report current initialization step for better debugging
        if self.details:
            result["current_step"] = self.details.get("step", "unknown")
            result["details"] = self.details

        # v93.5: Report model load elapsed DURING loading (not just after)
        # This enables the orchestrator's intelligent timeout extension
        if self.model_load_elapsed:
            result["model_load_elapsed_seconds"] = round(self.model_load_elapsed, 1)
        elif self.model_load_start:
            # Model is currently loading - report elapsed time so far
            current_elapsed = time.time() - self.model_load_start
            result["model_load_elapsed_seconds"] = round(current_elapsed, 1)
            result["model_loading_in_progress"] = True
            # v93.7: Estimate progress based on typical load time
            model_timeout = self.details.get("model_load_timeout", 600.0)
            result["model_load_progress_pct"] = min(95, round((current_elapsed / model_timeout) * 100, 1))

        if self.error:
            result["error"] = self.error
        return result


# =============================================================================
# GLOBAL STATE - Populated during background initialization
# =============================================================================
_startup_state: Optional[StartupState] = None
_bridge = None
_neural_orchestrator = None
_executor = None
_agi_hub = None
_trinity_initialized = False
_trinity_record_inference = None
_neural_routing_enabled = False
_model_path: Optional[Path] = None
_args = None


async def main():
    """
    v93.2: Main entry point with IMMEDIATE HTTP server startup.

    CRITICAL FIX: The HTTP server starts FIRST before any heavy imports
    or model loading. This ensures health checks succeed immediately while
    initialization happens in the background.

    Startup sequence:
    1. Parse args (instant)
    2. Create minimal FastAPI app with health endpoint (instant)
    3. Start uvicorn server (instant - server is now LISTENING)
    4. FastAPI startup event triggers background_initialization()
    5. Heavy imports, model loading, bridges all run in background
    6. Health endpoint reports "starting" -> "ready" as init completes
    """
    global _startup_state, _args

    _args = parse_args()

    if _args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize startup state FIRST
    _startup_state = StartupState()

    logger.info("[v93.2] JARVIS-Prime starting with IMMEDIATE HTTP server...")

    # =========================================================================
    # STEP 1: Import FastAPI (lightweight, instant)
    # =========================================================================
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
        import uvicorn
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install with: pip install fastapi uvicorn pydantic")
        sys.exit(1)

    # =========================================================================
    # STEP 2: Create MINIMAL FastAPI app that responds to health IMMEDIATELY
    # =========================================================================
    app = FastAPI(
        title="JARVIS-Prime",
        description="Tier-0 Muscle Memory Brain - OpenAI-compatible API (v93.2 Immediate Start)",
        version="93.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # =========================================================================
    # IMMEDIATE HEALTH ENDPOINT - Responds BEFORE initialization completes
    # =========================================================================
    @app.get("/health")
    async def health_check():
        """
        v93.2: Immediate health endpoint.

        Returns status IMMEDIATELY - even during heavy initialization.
        This is the key fix for the 61.9s timeout issue.

        Status meanings:
        - "starting": Server is up, initialization in progress
        - "healthy": Fully initialized and ready for inference
        - "error": Initialization failed
        """
        if _startup_state:
            status = _startup_state.get_status()

            # Add runtime component status once initialized
            if _startup_state.phase == "ready":
                status["bridge_enabled"] = _bridge is not None
                status["trinity_enabled"] = _trinity_initialized
                status["agi_enabled"] = _agi_hub is not None
                status["neural_routing_enabled"] = _neural_routing_enabled
                status["ready_for_inference"] = _executor is not None and _executor.is_loaded() if hasattr(_executor, 'is_loaded') else False

            return status

        return {"status": "starting", "phase": "pre-init"}

    # =========================================================================
    # PLACEHOLDER ENDPOINTS - Return 503 until initialization completes
    # =========================================================================

    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: str = "jarvis-prime"
        messages: List[Message]
        max_tokens: int = 512
        temperature: float = 0.7
        stream: bool = False

    class GenerateRequest(BaseModel):
        prompt: str
        max_tokens: int = 512
        temperature: float = 0.7

    def _check_ready():
        """Check if server is ready for inference requests."""
        if _startup_state and _startup_state.phase != "ready":
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Server still initializing",
                    "phase": _startup_state.phase if _startup_state else "unknown",
                    "status": _startup_state.get_status() if _startup_state else {}
                }
            )
        if _executor is None:
            raise HTTPException(
                status_code=503,
                detail={"error": "Executor not initialized"}
            )
        if hasattr(_executor, 'is_loaded') and not _executor.is_loaded():
            raise HTTPException(
                status_code=503,
                detail={"error": "Model not loaded"}
            )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        """OpenAI-compatible chat completions with unified intelligent routing (v100.0)."""
        _check_ready()

        # Format messages
        from jarvis_prime.core.model_manager import ChatMessage
        messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
        prompt = _executor.format_messages(messages)
        prompt_tokens = len(prompt.split())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # v100.0: Neural Orchestrator Routing Decision
        routing_result = None
        routing_metadata = {}
        if _neural_orchestrator and _neural_routing_enabled:
            try:
                user_messages = [m.content for m in request.messages if m.role == "user"]
                classification_prompt = user_messages[-1] if user_messages else prompt

                routing_result = await _neural_orchestrator.route(
                    prompt=classification_prompt,
                    context={
                        "request_id": completion_id,
                        "message_count": len(request.messages),
                        "model_requested": request.model,
                        "stream": request.stream,
                    }
                )

                routing_metadata = {
                    "tier": routing_result.tier.name if routing_result else "unknown",
                    "model_id": routing_result.model_id if routing_result else None,
                    "decision_reason": routing_result.decision_reason.value if routing_result else "none",
                    "confidence": routing_result.confidence if routing_result else 0.0,
                    "routing_latency_ms": routing_result.latency_ms if routing_result else 0.0,
                }

                logger.debug(
                    f"Neural Routing: {completion_id} -> {routing_metadata['tier']} "
                    f"(confidence={routing_metadata['confidence']:.2f}, "
                    f"reason={routing_metadata['decision_reason']})"
                )
            except Exception as e:
                logger.warning(f"Neural routing failed, using default: {e}")

        # v74.0: Streaming Response (SSE format)
        if request.stream:
            async def stream_generator():
                start = time.time()
                token_count = 0

                try:
                    async for token in _executor.generate_stream(
                        prompt=prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    ):
                        token_count += 1
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "jarvis-prime",
                            "choices": [{
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"

                    final_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "jarvis-prime",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }],
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                    latency_ms = (time.time() - start) * 1000
                    _record_inference_metrics(prompt_tokens, token_count, latency_ms, True)

                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    _record_inference_metrics(0, 0, 0, False)
                    error_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "jarvis-prime",
                        "choices": [{
                            "index": 0,
                            "delta": {"content": f"\n[Error: {str(e)}]"},
                            "finish_reason": "error",
                        }],
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming response
        try:
            start = time.time()

            response = await _executor.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency_ms = (time.time() - start) * 1000
            completion_tokens = len(response.split())

            _record_inference_metrics(prompt_tokens, completion_tokens, latency_ms, True)

            if _neural_orchestrator and routing_result:
                try:
                    await _neural_orchestrator.record_circuit_success(routing_result.tier.name)
                except Exception:
                    pass

            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": "jarvis-prime",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "x_latency_ms": latency_ms,
                "x_routing": routing_metadata,
            }
        except Exception as e:
            logger.error(f"Chat error: {e}")
            _record_inference_metrics(0, 0, 0, False)
            if _neural_orchestrator and routing_result:
                try:
                    await _neural_orchestrator.record_circuit_failure(routing_result.tier.name)
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        """Simple text generation with unified intelligent routing (v100.0)."""
        _check_ready()

        generate_id = f"gen-{uuid.uuid4().hex[:8]}"
        routing_result = None
        routing_metadata = {}

        if _neural_orchestrator and _neural_routing_enabled:
            try:
                routing_result = await _neural_orchestrator.route(
                    prompt=request.prompt,
                    context={"request_id": generate_id, "endpoint": "generate"}
                )
                routing_metadata = {
                    "tier": routing_result.tier.name if routing_result else "unknown",
                    "model_id": routing_result.model_id if routing_result else None,
                    "decision_reason": routing_result.decision_reason.value if routing_result else "none",
                    "confidence": routing_result.confidence if routing_result else 0.0,
                }
            except Exception as e:
                logger.warning(f"Neural routing failed for generate: {e}")

        try:
            start = time.time()

            response = await _executor.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency_ms = (time.time() - start) * 1000
            prompt_tokens = len(request.prompt.split())
            completion_tokens = len(response.split())

            _record_inference_metrics(prompt_tokens, completion_tokens, latency_ms, True)

            if _neural_orchestrator and routing_result:
                try:
                    await _neural_orchestrator.record_circuit_success(routing_result.tier.name)
                except Exception:
                    pass

            return {
                "text": response,
                "latency_ms": latency_ms,
                "x_routing": routing_metadata,
            }
        except Exception as e:
            logger.error(f"Generate error: {e}")
            _record_inference_metrics(0, 0, 0, False)
            if _neural_orchestrator and routing_result:
                try:
                    await _neural_orchestrator.record_circuit_failure(routing_result.tier.name)
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics")
    async def metrics():
        """Get inference and cost metrics."""
        if _bridge:
            try:
                from jarvis_prime.core.cross_repo_bridge import get_cost_summary
                cost_summary = get_cost_summary()
                inference_metrics = _bridge.get_metrics()
                return {
                    "status": "ok",
                    "cost_summary": cost_summary,
                    "inference_metrics": inference_metrics,
                    "connected_to_jarvis": _bridge.state.connected_to_jarvis,
                }
            except Exception as e:
                return {"status": "error", "error": str(e)}
        else:
            return {"status": "disabled", "message": "Cross-repo bridge not enabled"}

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        model_status = "loading"
        if _startup_state and _startup_state.phase == "ready":
            model_status = "ready" if (_executor and hasattr(_executor, 'is_loaded') and _executor.is_loaded()) else "not_loaded"

        return {
            "object": "list",
            "data": [{
                "id": "jarvis-prime",
                "object": "model",
                "owned_by": "jarvis",
                "status": model_status,
            }],
        }

    # =========================================================================
    # MODEL HOT-RELOAD ENDPOINT - Reactor-Core Integration
    # =========================================================================
    class ModelReloadRequest(BaseModel):
        model_path: str
        model_version: str = "unknown"
        model_id: str = ""

    @app.post("/api/v1/models/reload")
    async def reload_model(request: ModelReloadRequest):
        """Hot-reload model from Reactor-Core."""
        global _model_path

        _check_ready()
        logger.info(f"Model reload requested: {request.model_path} (v{request.model_version})")

        try:
            new_model_path = Path(request.model_path)

            if not new_model_path.exists():
                raise HTTPException(status_code=404, detail=f"Model file not found: {request.model_path}")

            if _executor.is_loaded():
                logger.info("Unloading current model...")
                await _executor.close()

            logger.info(f"Loading new model: {new_model_path}")
            start = time.time()
            await _executor.load(new_model_path)
            load_time = time.time() - start

            _model_path = new_model_path

            if _bridge:
                try:
                    from jarvis_prime.core.cross_repo_bridge import update_model_status
                    update_model_status(loaded=True, model_path=str(_model_path))
                    await _bridge.notify_jarvis("model_reloaded", {
                        "model_path": str(_model_path),
                        "model_version": request.model_version,
                        "load_time_seconds": load_time,
                    })
                except Exception as e:
                    logger.warning(f"Failed to notify bridge: {e}")

            if _trinity_initialized:
                try:
                    from jarvis_prime.core.trinity_bridge import update_model_status as trinity_update
                    trinity_update(loaded=True, model_path=str(_model_path))
                except Exception as e:
                    logger.warning(f"Failed to notify Trinity: {e}")

            logger.info(f"Model reloaded in {load_time:.2f}s: {_model_path}")

            return {
                "status": "success",
                "model_path": str(_model_path),
                "model_version": request.model_version,
                "load_time_seconds": load_time,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # AGI v77.0 ENDPOINTS
    # =========================================================================
    class AGIReasonRequest(BaseModel):
        query: str
        strategy: str = "chain_of_thought"
        context: dict = {}

    class AGIPlanRequest(BaseModel):
        goal: str
        context: dict = {}
        constraints: List[str] = []

    class AGIFeedbackRequest(BaseModel):
        experience_id: str
        score: float
        comment: Optional[str] = None

    class AGIProcessRequest(BaseModel):
        content: str
        modalities: List[str] = ["text"]
        context: dict = {}
        enable_reasoning: bool = True
        enable_learning: bool = True

    @app.post("/agi/reason")
    async def agi_reason(request: AGIReasonRequest):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            start = time.time()
            result = await _agi_hub.reason(
                query=request.query,
                strategy=request.strategy,
                context=request.context,
            )
            latency_ms = (time.time() - start) * 1000

            return {
                "status": "success",
                "query": request.query,
                "strategy": request.strategy,
                "conclusion": result.get("conclusion"),
                "trace": result.get("trace", []),
                "confidence": result.get("confidence", 0.0),
                "latency_ms": latency_ms,
            }
        except Exception as e:
            logger.error(f"AGI reasoning error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agi/plan")
    async def agi_plan(request: AGIPlanRequest):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            start = time.time()
            result = await _agi_hub.plan(goal=request.goal, context=request.context)
            latency_ms = (time.time() - start) * 1000

            return {
                "status": "success",
                "goal": request.goal,
                "plan": result,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            logger.error(f"AGI planning error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agi/process")
    async def agi_process(request: AGIProcessRequest):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            async def inference_fn(prompt: str, **kwargs):
                if _executor and hasattr(_executor, 'is_loaded') and _executor.is_loaded():
                    return await _executor.generate(prompt=prompt, max_tokens=512, temperature=0.7)
                return prompt

            result = await _agi_hub.process(
                content=request.content,
                modalities=request.modalities,
                context=request.context,
                inference_fn=inference_fn if request.enable_reasoning else None,
            )

            return {
                "status": "success",
                "request_id": result.request_id,
                "content": result.content,
                "reasoning_trace": result.reasoning_trace,
                "confidence": result.confidence,
                "models_used": result.models_used,
                "processing_time_ms": result.processing_time_ms,
                "feedback_recorded": result.feedback_recorded,
            }
        except Exception as e:
            logger.error(f"AGI process error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agi/feedback")
    async def agi_feedback(request: AGIFeedbackRequest):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            success = await _agi_hub.record_feedback(
                experience_id=request.experience_id,
                score=request.score,
                comment=request.comment,
            )
            return {"status": "success" if success else "failed", "experience_id": request.experience_id, "score": request.score}
        except Exception as e:
            logger.error(f"AGI feedback error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agi/learning/trigger")
    async def agi_learning_trigger(force: bool = False):
        if not _agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            result = await _agi_hub.trigger_learning_update(force=force)
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"AGI learning trigger error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agi/status")
    async def agi_status():
        if not _agi_hub:
            return {"status": "not_initialized", "message": "AGI Hub not available"}

        try:
            status = _agi_hub.get_status()
            health = await _agi_hub.health_check()
            return {
                "status": "ok",
                "initialized": status["initialized"],
                "healthy": health["healthy"],
                "subsystems": status["subsystems"],
                "metrics": status["metrics"],
            }
        except Exception as e:
            logger.error(f"AGI status error: {e}")
            return {"status": "error", "error": str(e)}

    @app.get("/agi/learning/stats")
    async def agi_learning_stats():
        if not _agi_hub or not _agi_hub.learning_engine:
            return {"status": "not_available", "message": "Learning engine not initialized"}

        try:
            stats = _agi_hub.learning_engine.get_statistics()
            return {"status": "ok", "statistics": stats}
        except Exception as e:
            logger.error(f"AGI learning stats error: {e}")
            return {"status": "error", "error": str(e)}

    # =========================================================================
    # NEURAL ORCHESTRATOR v100.0 ENDPOINTS
    # =========================================================================
    class NeuralRouteRequest(BaseModel):
        prompt: str
        context: dict = {}

    @app.get("/neural-orchestrator/health")
    async def neural_orchestrator_health():
        if not _neural_orchestrator or not _neural_routing_enabled:
            return {"status": "not_initialized", "message": "Neural Orchestrator not available"}

        try:
            health = _neural_orchestrator.get_health_status()
            return {"status": "ok", **health}
        except Exception as e:
            logger.error(f"Neural Orchestrator health error: {e}")
            return {"status": "error", "error": str(e)}

    @app.get("/neural-orchestrator/stats")
    async def neural_orchestrator_stats():
        if not _neural_orchestrator or not _neural_routing_enabled:
            return {"status": "not_initialized", "message": "Neural Orchestrator not available"}

        try:
            stats = _neural_orchestrator.get_comprehensive_stats()
            return {"status": "ok", **stats}
        except Exception as e:
            logger.error(f"Neural Orchestrator stats error: {e}")
            return {"status": "error", "error": str(e)}

    @app.post("/neural-orchestrator/route")
    async def neural_orchestrator_route(request: NeuralRouteRequest):
        if not _neural_orchestrator or not _neural_routing_enabled:
            raise HTTPException(status_code=503, detail="Neural Orchestrator not available")

        try:
            result = await _neural_orchestrator.route(prompt=request.prompt, context=request.context)
            return {
                "status": "ok",
                "routing": result.to_dict() if result else None,
                "task_classification": {
                    "task_type": result.task_classification.task_type.value if result and result.task_classification else None,
                    "complexity": result.task_classification.complexity if result and result.task_classification else None,
                    "confidence": result.task_classification.confidence if result and result.task_classification else None,
                    "recommended_tier": result.task_classification.recommended_tier.name if result and result.task_classification else None,
                } if result and result.task_classification else None,
            }
        except Exception as e:
            logger.error(f"Neural routing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/neural-orchestrator/memory")
    async def neural_orchestrator_memory():
        if not _neural_orchestrator or not _neural_routing_enabled:
            return {"status": "not_initialized", "message": "Neural Orchestrator not available"}

        try:
            memory = await _neural_orchestrator.get_memory_status()
            should_burst = await _neural_orchestrator.should_burst_to_cloud()
            return {"status": "ok", "memory": memory, "should_burst_to_cloud": should_burst}
        except Exception as e:
            logger.error(f"Neural Orchestrator memory error: {e}")
            return {"status": "error", "error": str(e)}

    @app.post("/neural-orchestrator/classify")
    async def neural_orchestrator_classify(request: NeuralRouteRequest):
        if not _neural_orchestrator or not _neural_routing_enabled:
            raise HTTPException(status_code=503, detail="Neural Orchestrator not available")

        try:
            classification = await _neural_orchestrator.classify_task(prompt=request.prompt, context=request.context)
            return {
                "status": "ok",
                "task_type": classification.task_type.value,
                "complexity": classification.complexity,
                "confidence": classification.confidence,
                "signals": classification.signals,
                "recommended_tier": classification.recommended_tier.name,
                "requires_fast_response": classification.requires_fast_response,
                "is_coding_session": classification.is_coding_session,
            }
        except Exception as e:
            logger.error(f"Neural classification error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # =========================================================================
    # WEBSOCKET EVENT STREAM - For Neural Mesh Integration (v93.15)
    # =========================================================================
    from fastapi import WebSocket, WebSocketDisconnect
    from datetime import datetime

    # v93.15: Global event queue and subscribers for WebSocket streaming
    _ws_event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _ws_subscribers: List[WebSocket] = []

    async def _broadcast_ws_event(event_type: str, data: Dict[str, Any]) -> None:
        """Broadcast an event to all connected WebSocket subscribers."""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }
        for ws in _ws_subscribers[:]:  # Copy to avoid modification during iteration
            try:
                await ws.send_json(event)
            except Exception:
                # Remove dead connections
                try:
                    _ws_subscribers.remove(ws)
                except ValueError:
                    pass

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket):
        """
        v93.15: WebSocket endpoint for real-time event streaming to Neural Mesh.

        This endpoint allows JARVIS Body and other clients to receive real-time
        events from JARVIS-Prime without polling.

        Events include:
        - connected: Initial connection event with status
        - heartbeat: Periodic keep-alive (every 30s)
        - inference_complete: After each inference
        - model_loaded: When model loading completes
        - error: On errors
        """
        await websocket.accept()
        _ws_subscribers.append(websocket)
        logger.info(f"[WebSocket] Client connected ({len(_ws_subscribers)} active)")

        try:
            # Send initial connection event
            await websocket.send_json({
                "event_type": "connected",
                "data": {
                    "status": "starting" if not _startup_state.ready else "ready",
                    "phase": _startup_state.phase,
                    "model_loaded": _startup_state.model_loaded,
                    "instance_id": str(uuid.uuid4())[:8],
                },
                "timestamp": datetime.now().isoformat(),
            })

            # Keep connection alive with heartbeats
            while True:
                try:
                    # Wait for incoming message or timeout for heartbeat
                    try:
                        message = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=30.0  # Heartbeat every 30s
                        )
                        # Handle ping/pong
                        if message == "ping":
                            await websocket.send_text("pong")
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await websocket.send_json({
                            "event_type": "heartbeat",
                            "data": {
                                "status": "ready" if _startup_state.ready else "starting",
                                "model_loaded": _startup_state.model_loaded,
                            },
                            "timestamp": datetime.now().isoformat(),
                        })
                except WebSocketDisconnect:
                    break

        except Exception as e:
            logger.debug(f"[WebSocket] Error: {e}")
        finally:
            try:
                _ws_subscribers.remove(websocket)
            except ValueError:
                pass
            logger.info(f"[WebSocket] Client disconnected ({len(_ws_subscribers)} remaining)")

    # =========================================================================
    # HELPER FUNCTIONS
    # =========================================================================
    def _record_inference_metrics(tokens_in: int, tokens_out: int, latency_ms: float, success: bool):
        """Record inference metrics to all tracking systems."""
        if _bridge:
            try:
                from jarvis_prime.core.cross_repo_bridge import record_inference
                record_inference(tokens_in=tokens_in, tokens_out=tokens_out, latency_ms=latency_ms)
            except Exception:
                pass

        if _trinity_record_inference:
            try:
                _trinity_record_inference(latency_ms=latency_ms, success=success)
            except Exception:
                pass

    # =========================================================================
    # BACKGROUND INITIALIZATION - Runs AFTER server starts listening
    # =========================================================================
    async def background_initialization():
        """
        v93.2: Run ALL heavy initialization in background.

        This function is triggered by FastAPI's startup event, which means
        the HTTP server is ALREADY listening when this runs.

        Initialization order:
        1. Import ML libraries (torch, sklearn - triggers warnings)
        2. Initialize cross-repo bridge
        3. Initialize Trinity bridge
        4. Initialize AGI Hub
        5. Initialize Neural Orchestrator
        6. Download model if needed
        7. Load model
        8. Mark ready
        """
        global _bridge, _executor, _agi_hub, _neural_orchestrator, _model_path
        global _trinity_initialized, _trinity_record_inference, _neural_routing_enabled

        try:
            _startup_state.phase = "initializing"
            init_start = time.time()

            # v93.7: Enhanced step logging with timing
            def log_step(step_name: str, step_num: int, total_steps: int = 9):
                """Log step with progress indicator."""
                _startup_state.details["step"] = step_name
                _startup_state.details["step_num"] = step_num
                _startup_state.details["total_steps"] = total_steps
                elapsed = time.time() - init_start
                logger.info("")
                logger.info(f"{'='*60}")
                logger.info(f"📍 STEP {step_num}/{total_steps}: {step_name.upper()}")
                logger.info(f"   Elapsed: {elapsed:.1f}s")
                logger.info(f"{'='*60}")

            def log_step_complete(step_name: str, duration: float):
                """Log step completion."""
                logger.info(f"✅ {step_name} complete ({duration:.2f}s)")

            logger.info("")
            logger.info("=" * 70)
            logger.info("🚀 JARVIS-PRIME BACKGROUND INITIALIZATION STARTING")
            logger.info("=" * 70)
            logger.info(f"   PID: {os.getpid()}")
            logger.info(f"   Python: {sys.version.split()[0]}")
            logger.info("")

            # -----------------------------------------------------------------
            # STEP 1: Import ML libraries (this triggers the warnings)
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("importing_ml_libraries", 1)

            try:
                from jarvis_prime.core.llama_cpp_executor import LlamaCppExecutor, LlamaCppConfig
                log_step_complete("ML libraries import", time.time() - step_start)
            except ImportError as e:
                logger.error(f"❌ Import error: {e}")
                _startup_state.phase = "error"
                _startup_state.error = f"Missing llama-cpp-python: {e}"
                return

            # -----------------------------------------------------------------
            # STEP 2: Initialize cross-repo bridge
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_bridge", 2)
            bridge_enabled = _args.bridge_enabled and not _args.no_bridge
            if bridge_enabled:
                try:
                    from jarvis_prime.core.cross_repo_bridge import (
                        initialize_bridge,
                        shutdown_bridge,
                        record_inference,
                        update_model_status,
                        get_cost_summary,
                    )
                    _bridge = await initialize_bridge(port=_args.port)
                    log_step_complete("Cross-repo bridge", time.time() - step_start)
                except Exception as e:
                    logger.warning(f"⚠️ Cross-repo bridge failed: {e}")
                    _bridge = None
            else:
                logger.info("   ℹ️ Cross-repo bridge disabled (--no-bridge)")
                _bridge = None

            # -----------------------------------------------------------------
            # STEP 3: Initialize Trinity bridge
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_trinity", 3)
            try:
                from jarvis_prime.core.trinity_bridge import (
                    initialize_trinity,
                    shutdown_trinity,
                    update_model_status as trinity_update_model_status,
                    record_inference as _trinity_rec_inf,
                    TRINITY_ENABLED,
                )
                _trinity_record_inference = _trinity_rec_inf
                if TRINITY_ENABLED:
                    _trinity_initialized = await initialize_trinity(port=_args.port)
                    if _trinity_initialized:
                        log_step_complete("Trinity bridge", time.time() - step_start)
                    else:
                        logger.warning("   ⚠️ Trinity init returned False")
                else:
                    logger.info("   ℹ️ Trinity disabled")
            except ImportError as e:
                logger.warning(f"   ⚠️ Trinity module not available ({e})")
            except Exception as e:
                logger.warning(f"   ⚠️ Trinity init failed ({e})")

            # -----------------------------------------------------------------
            # STEP 4: Initialize AGI Hub
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_agi_hub", 4)
            try:
                from jarvis_prime.core.agi_integration import (
                    AGIIntegrationHub,
                    AGIHubConfig,
                    get_agi_hub,
                    shutdown_agi_hub,
                    AGIEnhancedInference,
                )

                # v93.12: Configure AGI Hub with sensible timeouts
                # These can be overridden via environment variables
                agi_config = AGIHubConfig(
                    enable_orchestrator=True,
                    enable_reasoning=True,
                    enable_learning=True,
                    enable_multimodal=True,
                    enable_hardware_optimization=True,
                    enable_auto_reasoning=True,
                    enable_experience_recording=True,
                    # v93.12: Timeout configuration (prevents hanging)
                    enable_agi_models_v80=os.getenv("ENABLE_AGI_MODELS_V80", "true").lower() == "true",
                    agi_models_v80_timeout=float(os.getenv("AGI_MODELS_V80_TIMEOUT", "30.0")),
                    agi_models_v80_graceful_degradation=True,  # Don't fail startup if v80.0 models fail
                    subsystem_init_timeout=float(os.getenv("SUBSYSTEM_INIT_TIMEOUT", "60.0")),
                    parallel_init_timeout=float(os.getenv("PARALLEL_INIT_TIMEOUT", "120.0")),
                )

                _agi_hub = await get_agi_hub(agi_config)
                log_step_complete("AGI Integration Hub", time.time() - step_start)
            except asyncio.TimeoutError:
                # v93.12: Handle timeout gracefully - don't block startup
                logger.warning(f"   ⏱️ AGI Hub init timed out - continuing without it")
                _agi_hub = None
            except ImportError as e:
                logger.warning(f"   ⚠️ AGI Hub not available: {e}")
            except Exception as e:
                logger.warning(f"   ⚠️ AGI Hub init failed: {e}")
                import traceback
                traceback.print_exc()

            # -----------------------------------------------------------------
            # STEP 5: Initialize Neural Orchestrator
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("initializing_neural_orchestrator", 5)
            try:
                from jarvis_prime.core.neural_orchestrator_core import (
                    NeuralOrchestratorCore,
                    DynamicConfig as NeuralConfig,
                    neural_route,
                    get_neural_orchestrator,
                    shutdown_neural_orchestrator,
                    RoutingTier,
                )

                neural_config = NeuralConfig.from_env_and_yaml()
                _neural_orchestrator = await get_neural_orchestrator(neural_config)
                _neural_routing_enabled = True
                log_step_complete("Neural Orchestrator", time.time() - step_start)
            except ImportError as e:
                logger.warning(f"   ⚠️ Neural Orchestrator not available: {e}")
            except Exception as e:
                logger.warning(f"   ⚠️ Neural Orchestrator init failed: {e}")
                import traceback
                traceback.print_exc()

            # -----------------------------------------------------------------
            # STEP 6: Resolve model path and download if needed
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("resolving_model", 6)
            _startup_state.phase = "loading_model"

            model_path = Path(_args.model)
            if not model_path.exists():
                model_path = Path(__file__).parent / _args.model

            # Try GCS download
            gcs_model_uri = os.getenv("MODEL_GCS_URI")
            auto_download_model = os.getenv("AUTO_DOWNLOAD_MODEL", "true").lower() == "true"

            if not model_path.exists() and gcs_model_uri:
                logger.info(f"[Background] Downloading model from GCS: {gcs_model_uri}")
                try:
                    from google.cloud import storage
                    import re

                    model_path.parent.mkdir(parents=True, exist_ok=True)

                    match = re.match(r"gs://([^/]+)/(.+)", gcs_model_uri)
                    if match:
                        bucket_name, blob_path = match.groups()
                        client = storage.Client()
                        bucket = client.bucket(bucket_name)
                        blob = bucket.blob(blob_path)

                        if blob.exists():
                            blob.reload()
                            size_mb = blob.size / 1024 / 1024 if blob.size else 0
                            logger.info(f"[Background] Downloading {size_mb:.1f}MB model...")
                            blob.download_to_filename(str(model_path))
                            if model_path.exists():
                                logger.info(f"[Background] Model downloaded: {model_path}")
                except ImportError:
                    logger.info("[Background] GCS client not available, trying gsutil...")
                    try:
                        import subprocess
                        subprocess.run(["gsutil", "cp", gcs_model_uri, str(model_path)], check=True, capture_output=True)
                        logger.info(f"[Background] Model downloaded via gsutil")
                    except Exception as e:
                        logger.warning(f"[Background] gsutil download failed: {e}")
                except Exception as e:
                    logger.warning(f"[Background] GCS download failed: {e}")

            # Try HuggingFace download
            if not model_path.exists() and auto_download_model:
                logger.info("[Background] Attempting HuggingFace download...")
                try:
                    from jarvis_prime.docker.model_downloader import download_model, recommend_model, MODEL_CATALOG

                    default_model = os.getenv("DEFAULT_MODEL", "tinyllama-chat")
                    if default_model not in MODEL_CATALOG:
                        recommended = recommend_model(use_case="balanced", max_memory_gb=8.0)
                        default_model = recommended if recommended else "tinyllama-chat"

                    logger.info(f"[Background] Auto-downloading: {default_model}")
                    models_dir = Path(__file__).parent / "models"
                    downloaded_path = await download_model(
                        model_key=default_model,
                        models_dir=str(models_dir),
                        set_active=True,
                    )
                    model_path = models_dir / "current.gguf"
                    if model_path.exists():
                        logger.info(f"[Background] Model auto-downloaded: {downloaded_path}")
                except ImportError as e:
                    logger.warning(f"[Background] HuggingFace downloader not available: {e}")
                except Exception as e:
                    logger.warning(f"[Background] HuggingFace download failed: {e}")
                    import traceback
                    traceback.print_exc()

            _model_path = model_path
            logger.info(f"   Model path: {model_path}")
            logger.info(f"   Exists: {model_path.exists()}")
            if model_path.exists():
                size_gb = model_path.stat().st_size / (1024**3)
                logger.info(f"   Size: {size_gb:.2f} GB")
            log_step_complete("Model resolution", time.time() - step_start)

            # -----------------------------------------------------------------
            # STEP 7: Hardware optimization and executor creation
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("configuring_hardware", 7)
            optimized_gpu_layers = _args.gpu_layers
            optimized_threads = _args.threads
            optimized_ctx_size = _args.ctx_size

            if _agi_hub and _agi_hub.hardware_optimizer:
                try:
                    hw_opt = _agi_hub.hardware_optimizer
                    recommendations = hw_opt.get_recommendations()
                    if recommendations:
                        if recommendations.get("use_mps", False):
                            optimized_gpu_layers = -1
                            logger.info("[Background] Apple Silicon: MPS enabled")
                        if "optimal_threads" in recommendations:
                            optimized_threads = recommendations["optimal_threads"]
                            logger.info(f"[Background] Apple Silicon: {optimized_threads} threads")
                except Exception as e:
                    logger.warning(f"[Background] Apple Silicon optimization failed: {e}")
            else:
                try:
                    from jarvis_prime.core.llama_cpp_executor import HardwareDetector, HardwareBackend
                    hw = HardwareDetector.detect()
                    logger.info(f"[Background] Hardware: {hw.backend.name}")
                    if hw.backend == HardwareBackend.METAL:
                        optimized_gpu_layers = -1
                        optimized_threads = hw.performance_cores or _args.threads
                        logger.info(f"[Background] Metal GPU enabled")
                    elif hw.backend == HardwareBackend.CUDA:
                        optimized_gpu_layers = -1
                        logger.info("[Background] CUDA GPU enabled")
                except Exception as e:
                    logger.warning(f"[Background] Hardware detection failed: {e}")

            config = LlamaCppConfig(
                n_ctx=optimized_ctx_size,
                n_threads=optimized_threads,
                n_gpu_layers=optimized_gpu_layers,
                verbose=_args.debug,
                flash_attn=True,
                cache_prompt=True,
            )
            _executor = LlamaCppExecutor(config)
            logger.info(f"   GPU Layers: {optimized_gpu_layers}")
            logger.info(f"   Threads: {optimized_threads}")
            logger.info(f"   Context: {optimized_ctx_size}")
            log_step_complete("Hardware configuration", time.time() - step_start)

            # -----------------------------------------------------------------
            # STEP 8: Load model (v93.7: with timeout and progress reporting)
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("loading_model", 8)
            _startup_state.model_load_start = time.time()

            # v93.7: Configurable model loading timeout
            model_load_timeout = float(os.environ.get("MODEL_LOAD_TIMEOUT", "600.0"))  # 10 min default

            if model_path.exists():
                model_size_mb = model_path.stat().st_size / (1024 * 1024)
                logger.info(f"[Background] Loading model: {model_path} ({model_size_mb:.1f}MB)")
                logger.info(f"[Background] Model load timeout: {model_load_timeout}s")
                _startup_state.details["model_size_mb"] = round(model_size_mb, 1)
                _startup_state.details["model_load_timeout"] = model_load_timeout

                start = time.time()

                # v93.7: Progress reporting task
                async def report_progress():
                    """Report progress every 30 seconds during model loading."""
                    while True:
                        await asyncio.sleep(30)
                        elapsed = time.time() - start
                        remaining = model_load_timeout - elapsed
                        logger.info(
                            f"[Background] Model loading... {elapsed:.0f}s elapsed, "
                            f"{remaining:.0f}s remaining (timeout: {model_load_timeout}s)"
                        )
                        _startup_state.details["loading_elapsed"] = round(elapsed, 1)

                progress_task = asyncio.create_task(report_progress())

                try:
                    # v93.7: Model loading with timeout protection
                    await asyncio.wait_for(
                        _executor.load(model_path),
                        timeout=model_load_timeout
                    )
                    load_time = time.time() - start
                    _startup_state.model_load_elapsed = load_time
                    _startup_state.model_loaded = True
                    _startup_state.model_path = str(model_path)
                    logger.info(f"[Background] Model loaded in {load_time:.2f}s")

                    # v93.7: Log step completion with timing
                    log_step_complete("Model loading", load_time)

                except asyncio.TimeoutError:
                    load_time = time.time() - start
                    logger.error(
                        f"[Background] MODEL LOAD TIMEOUT after {load_time:.1f}s "
                        f"(limit: {model_load_timeout}s)"
                    )
                    _startup_state.phase = "error"
                    _startup_state.error = f"Model load timeout after {load_time:.1f}s"
                    _startup_state.model_load_elapsed = load_time
                    progress_task.cancel()
                    return

                finally:
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass

                if _bridge:
                    try:
                        from jarvis_prime.core.cross_repo_bridge import update_model_status
                        update_model_status(loaded=True, model_path=str(model_path))
                        await _bridge.notify_jarvis("model_loaded", {
                            "model_path": str(model_path),
                            "load_time_seconds": load_time,
                        })
                    except Exception as e:
                        logger.warning(f"[Background] Failed to notify bridge: {e}")
            else:
                logger.warning(f"[Background] Model not found: {_args.model}")
                logger.warning("[Background] Server will run in health-check-only mode")
                _startup_state.model_path = None

                if _bridge:
                    try:
                        from jarvis_prime.core.cross_repo_bridge import update_model_status
                        update_model_status(loaded=False, model_path="")
                    except Exception:
                        pass

            # -----------------------------------------------------------------
            # STEP 9: Mark ready (v93.7: with enhanced logging)
            # -----------------------------------------------------------------
            step_start = time.time()
            log_step("marking_ready", 9)

            _startup_state.phase = "ready"
            _startup_state.init_elapsed = time.time() - init_start
            _startup_state.details = {}  # Clear step details

            # v93.7: Calculate detailed timing breakdown
            total_time = _startup_state.init_elapsed
            model_load_time = _startup_state.model_load_elapsed or 0
            init_overhead = total_time - model_load_time

            logger.info("")
            logger.info("=" * 70)
            logger.info("🎉 JARVIS-PRIME INITIALIZATION COMPLETE")
            logger.info("=" * 70)
            logger.info("")
            logger.info("📊 TIMING BREAKDOWN:")
            logger.info(f"   Total initialization: {total_time:.2f}s")
            logger.info(f"   ├─ Model loading:     {model_load_time:.2f}s ({model_load_time/total_time*100:.1f}%)" if total_time > 0 else f"   ├─ Model loading:     {model_load_time:.2f}s")
            logger.info(f"   └─ Other init:        {init_overhead:.2f}s ({init_overhead/total_time*100:.1f}%)" if total_time > 0 else f"   └─ Other init:        {init_overhead:.2f}s")
            logger.info("")
            logger.info("🖥️  SERVER CONFIGURATION:")
            logger.info(f"   Model: {model_path.name if model_path.exists() else 'Not loaded'}")
            if model_path.exists():
                model_size_gb = model_path.stat().st_size / (1024**3)
                logger.info(f"   Size:  {model_size_gb:.2f} GB")
            logger.info(f"   Context: {_args.ctx_size} tokens")
            logger.info(f"   GPU layers: {optimized_gpu_layers}")
            logger.info(f"   Threads: {optimized_threads}")
            logger.info(f"   Listening: http://{_args.host}:{_args.port}")
            logger.info("")
            logger.info("🔌 INTEGRATIONS:")

            if _bridge:
                logger.info(f"   ├─ JARVIS Bridge: {'Connected' if _bridge.state.connected_to_jarvis else 'Enabled (standalone)'}")
            else:
                logger.info("   ├─ JARVIS Bridge: Disabled")
            if _trinity_initialized:
                logger.info("   ├─ PROJECT TRINITY: Connected (Mind component)")
            else:
                logger.info("   ├─ PROJECT TRINITY: Not initialized")
            if _agi_hub:
                logger.info("   ├─ AGI Integration Hub: Active")
            else:
                logger.info("   ├─ AGI Integration Hub: Not initialized")
            if _neural_routing_enabled:
                logger.info("   └─ Neural Orchestrator v100.0: Active")
            else:
                logger.info("   └─ Neural Orchestrator: Not initialized")

            logger.info("")
            logger.info("=" * 70)
            logger.info("✅ READY FOR INFERENCE")
            logger.info("=" * 70)
            logger.info("")

            log_step_complete("Marking ready", time.time() - step_start)

        except Exception as e:
            logger.error(f"[Background] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            _startup_state.phase = "error"
            _startup_state.error = str(e)

    # =========================================================================
    # STARTUP EVENT - Triggers background initialization AFTER server starts
    # =========================================================================
    @app.on_event("startup")
    async def on_startup():
        """
        v93.2: FastAPI startup event.

        This runs AFTER uvicorn binds to the port and starts listening.
        This is the key to solving the 61.9s timeout - the health endpoint
        is already responding when this function starts.
        """
        logger.info("[v93.2] Server is now LISTENING - starting background initialization")
        asyncio.create_task(background_initialization())

    # =========================================================================
    # SHUTDOWN EVENT - Clean up all components
    # =========================================================================
    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("Shutting down...")

        if _neural_orchestrator:
            try:
                from jarvis_prime.core.neural_orchestrator_core import shutdown_neural_orchestrator
                stats = _neural_orchestrator.get_comprehensive_stats()
                routing_stats = stats.get("routing", {})
                logger.info("=" * 50)
                logger.info("Neural Orchestrator v100.0 Session Summary")
                logger.info("=" * 50)
                logger.info(f"  Total Requests Routed: {routing_stats.get('total_requests', 0)}")
                logger.info(f"  Successful Routes: {routing_stats.get('successful_routes', 0)}")
                logger.info(f"  Fallback Routes: {routing_stats.get('fallback_routes', 0)}")
                logger.info("=" * 50)
                await shutdown_neural_orchestrator()
                logger.info("Neural Orchestrator shutdown complete")
            except Exception as e:
                logger.warning(f"Neural Orchestrator shutdown error: {e}")

        if _agi_hub:
            try:
                from jarvis_prime.core.agi_integration import shutdown_agi_hub
                await shutdown_agi_hub()
                logger.info("AGI Integration Hub shutdown complete")
            except Exception as e:
                logger.warning(f"AGI Hub shutdown error: {e}")

        if _trinity_initialized:
            try:
                from jarvis_prime.core.trinity_bridge import shutdown_trinity
                await shutdown_trinity()
                logger.info("PROJECT TRINITY: J-Prime disconnected")
            except Exception as e:
                logger.warning(f"Trinity shutdown error: {e}")

        if _bridge:
            try:
                from jarvis_prime.core.cross_repo_bridge import get_cost_summary, shutdown_bridge
                cost_summary = get_cost_summary()

                if cost_summary.get("total_requests", 0) > 0:
                    logger.info("=" * 50)
                    logger.info("JARVIS-Prime Session Cost Summary")
                    logger.info("=" * 50)
                    logger.info(f"  Total Requests: {cost_summary.get('total_requests', 0)}")
                    logger.info(f"  Total Tokens: {cost_summary.get('total_tokens', 0)}")
                    logger.info(f"  Local Cost: ${cost_summary.get('local_cost_usd', 0):.4f}")
                    logger.info(f"  Savings: ${cost_summary.get('savings_usd', 0):.4f}")
                    logger.info("=" * 50)

                await _bridge.notify_jarvis("shutdown", cost_summary)
                await shutdown_bridge()
                logger.info("Cross-repo bridge shutdown complete")
            except Exception as e:
                logger.warning(f"Bridge shutdown error: {e}")

        if _executor:
            try:
                await _executor.close()
            except Exception as e:
                logger.warning(f"Executor close error: {e}")

    # =========================================================================
    # START SERVER - This is IMMEDIATE, background init happens via startup event
    # =========================================================================
    logger.info(f"[v93.2] Starting uvicorn server on {_args.host}:{_args.port}...")

    config = uvicorn.Config(
        app,
        host=_args.host,
        port=_args.port,
        reload=_args.reload,
        log_level="debug" if _args.debug else "info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
