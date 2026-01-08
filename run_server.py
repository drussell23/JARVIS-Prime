#!/usr/bin/env python3
"""
JARVIS-Prime Server - Quick Start Script (v2.0 with Cross-Repo Bridge)
=======================================================================

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
    GET  /health               - Health check
    GET  /metrics              - Cost tracking & inference metrics
"""

import argparse
import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis-prime")

# Cross-repo bridge (lazy import)
_bridge = None


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


async def main():
    global _bridge
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Import after parsing args to show help faster
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
        import uvicorn
        import json
    except ImportError:
        logger.error("Missing dependencies. Install with:")
        logger.error("  pip install fastapi uvicorn pydantic")
        sys.exit(1)

    try:
        from jarvis_prime.core.llama_cpp_executor import LlamaCppExecutor, LlamaCppConfig
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure llama-cpp-python is installed:")
        logger.error("  pip install llama-cpp-python")
        sys.exit(1)

    # Initialize cross-repo bridge for JARVIS integration
    bridge_enabled = args.bridge_enabled and not args.no_bridge
    if bridge_enabled:
        try:
            from jarvis_prime.core.cross_repo_bridge import (
                initialize_bridge,
                shutdown_bridge,
                record_inference,
                update_model_status,
                get_cost_summary,
            )
            _bridge = await initialize_bridge(port=args.port)
            logger.info("Cross-repo bridge initialized - connected to JARVIS infrastructure")
        except Exception as e:
            logger.warning(f"Cross-repo bridge initialization failed: {e}")
            logger.warning("Continuing without cross-repo integration")
            _bridge = None
    else:
        logger.info("Cross-repo bridge disabled")
        _bridge = None

    # v72.0: Initialize PROJECT TRINITY connection for distributed architecture
    # This enables JARVIS Body to detect J-Prime is online via heartbeat files
    # v73.0: Added inference health tracking integration
    trinity_initialized = False
    trinity_record_inference = None  # v73.0: Function reference for inference health tracking
    try:
        from jarvis_prime.core.trinity_bridge import (
            initialize_trinity,
            shutdown_trinity,
            update_model_status as trinity_update_model_status,
            record_inference as _trinity_record_inference,  # v73.0
            TRINITY_ENABLED,
        )
        trinity_record_inference = _trinity_record_inference  # Store for endpoint use
        if TRINITY_ENABLED:
            trinity_initialized = await initialize_trinity(port=args.port)
            if trinity_initialized:
                logger.info("PROJECT TRINITY: J-Prime (Mind) connected to Trinity network")
                logger.info("PROJECT TRINITY: Inference health tracking enabled (v73.0)")
            else:
                logger.warning("PROJECT TRINITY: Initialization returned False")
        else:
            logger.info("PROJECT TRINITY: Disabled (TRINITY_ENABLED=false)")
    except ImportError as e:
        logger.warning(f"PROJECT TRINITY: Module not available ({e})")
    except Exception as e:
        logger.warning(f"PROJECT TRINITY: Initialization failed ({e})")

    # v77.0: Initialize AGI Integration Hub
    # Connects all AGI subsystems: Orchestrator, Reasoning, Learning, MultiModal, Hardware
    agi_hub = None
    agi_inference = None
    try:
        from jarvis_prime.core.agi_integration import (
            AGIIntegrationHub,
            AGIHubConfig,
            get_agi_hub,
            shutdown_agi_hub,
            AGIEnhancedInference,
        )

        agi_config = AGIHubConfig(
            enable_orchestrator=True,
            enable_reasoning=True,
            enable_learning=True,
            enable_multimodal=True,
            enable_hardware_optimization=True,
            enable_auto_reasoning=True,
            enable_experience_recording=True,
        )

        agi_hub = await get_agi_hub(agi_config)
        logger.info("AGI v77.0: Integration Hub initialized")
        logger.info(f"  - Orchestrator: {agi_hub._subsystem_status.get('ORCHESTRATOR', {})}")
        logger.info(f"  - Reasoning Engine: Active")
        logger.info(f"  - Continuous Learning: Active")
        logger.info(f"  - MultiModal Fusion: Active")

    except ImportError as e:
        logger.warning(f"AGI Integration Hub not available: {e}")
    except Exception as e:
        logger.warning(f"AGI Integration Hub initialization failed: {e}")

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to script
        model_path = Path(__file__).parent / args.model

    # v73.0: Auto-ensure model exists - download if missing
    # Priority: 1. GCS (cloud deploy), 2. HuggingFace (local dev), 3. Skip (health-only mode)
    gcs_model_uri = os.getenv("MODEL_GCS_URI")
    auto_download_model = os.getenv("AUTO_DOWNLOAD_MODEL", "true").lower() == "true"

    if not model_path.exists() and gcs_model_uri:
        logger.info(f"Model not found locally, downloading from GCS: {gcs_model_uri}")
        try:
            # Use Python GCS client (more reliable than gsutil in containers)
            from google.cloud import storage
            import re

            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Parse GCS URI: gs://bucket/path/to/file.gguf
            match = re.match(r"gs://([^/]+)/(.+)", gcs_model_uri)
            if match:
                bucket_name, blob_path = match.groups()
                logger.info(f"Downloading from bucket={bucket_name}, path={blob_path}")

                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)

                # Check if blob exists and get metadata
                if not blob.exists():
                    logger.error(f"Model not found in GCS: {gcs_model_uri}")
                else:
                    # Reload to get size metadata
                    blob.reload()
                    size_mb = blob.size / 1024 / 1024 if blob.size else 0
                    logger.info(f"Downloading {size_mb:.1f}MB model from GCS...")

                    # Download with timeout for large models
                    blob.download_to_filename(str(model_path))

                    # Verify download
                    if model_path.exists():
                        actual_size_mb = model_path.stat().st_size / 1024 / 1024
                        logger.info(f"Model downloaded successfully: {actual_size_mb:.1f}MB to {model_path}")
                    else:
                        logger.error(f"Download completed but model file not found at {model_path}")
            else:
                logger.warning(f"Invalid GCS URI format: {gcs_model_uri}")

        except ImportError:
            # Fallback to gsutil if google-cloud-storage not installed
            logger.info("GCS Python client not available, trying gsutil...")
            try:
                import subprocess
                subprocess.run(
                    ["gsutil", "cp", gcs_model_uri, str(model_path)],
                    check=True,
                    capture_output=True,
                )
                logger.info(f"Model downloaded via gsutil to: {model_path}")
            except Exception as e:
                logger.warning(f"gsutil download failed: {e}")
        except Exception as e:
            logger.warning(f"Failed to download model from GCS: {e}")

    # v73.0: Auto-download from HuggingFace if model still missing and auto-download enabled
    if not model_path.exists() and auto_download_model:
        logger.info("Model not found, attempting auto-download from HuggingFace...")
        try:
            from jarvis_prime.docker.model_downloader import (
                download_model,
                recommend_model,
                MODEL_CATALOG,
            )

            # Auto-select best model for available memory (default: TinyLlama for quick start)
            default_model = os.getenv("DEFAULT_MODEL", "tinyllama-chat")

            if default_model not in MODEL_CATALOG:
                # Try to recommend based on memory
                recommended = recommend_model(use_case="balanced", max_memory_gb=8.0)
                if recommended:
                    default_model = recommended
                else:
                    default_model = "tinyllama-chat"  # Fallback

            logger.info(f"Auto-downloading model: {default_model}")
            print(f"  📥 Downloading {default_model} from HuggingFace...")

            # Download model (this may take a while for first run)
            models_dir = Path(__file__).parent / "models"
            downloaded_path = await download_model(
                model_key=default_model,
                models_dir=str(models_dir),
                set_active=True,
            )

            # Update model_path to use downloaded model
            model_path = models_dir / "current.gguf"
            if model_path.exists():
                logger.info(f"Model auto-downloaded: {downloaded_path}")
                print(f"  ✅ Model downloaded successfully: {default_model}")
            else:
                logger.warning(f"Auto-download completed but model not at expected path")

        except ImportError as e:
            logger.warning(f"HuggingFace downloader not available: {e}")
            logger.warning("Install with: pip install huggingface-hub tqdm")
        except Exception as e:
            logger.warning(f"Auto-download from HuggingFace failed: {e}")
            import traceback
            traceback.print_exc()

    # v77.0: Apply Apple Silicon optimizations if available
    optimized_gpu_layers = args.gpu_layers
    optimized_threads = args.threads
    optimized_ctx_size = args.ctx_size

    if agi_hub and agi_hub.hardware_optimizer:
        try:
            hw_opt = agi_hub.hardware_optimizer
            recommendations = hw_opt.get_recommendations()

            if recommendations:
                # Use optimized settings from Apple Silicon analyzer
                if recommendations.get("use_mps", False):
                    optimized_gpu_layers = -1  # All layers on GPU
                    logger.info("Apple Silicon: MPS acceleration enabled")

                if "optimal_threads" in recommendations:
                    optimized_threads = recommendations["optimal_threads"]
                    logger.info(f"Apple Silicon: Using {optimized_threads} threads")

                if "optimal_batch_size" in recommendations:
                    # Adjust context based on memory
                    logger.info(f"Apple Silicon: Batch size optimized for UMA")

                logger.info(f"Apple Silicon: Generation {recommendations.get('generation', 'unknown')}")

        except Exception as e:
            logger.warning(f"Apple Silicon optimization failed: {e}")

    # Create executor (may be unloaded)
    config = LlamaCppConfig(
        n_ctx=optimized_ctx_size,
        n_threads=optimized_threads,
        n_gpu_layers=optimized_gpu_layers,
        verbose=args.debug,
    )
    executor = LlamaCppExecutor(config)

    # Load model if available
    if model_path.exists():
        logger.info(f"Loading model: {model_path}")
        start = time.time()
        await executor.load(model_path)
        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.2f}s")

        # Notify bridge of model status
        if _bridge:
            try:
                from jarvis_prime.core.cross_repo_bridge import update_model_status
                update_model_status(loaded=True, model_path=str(model_path))
                await _bridge.notify_jarvis("model_loaded", {
                    "model_path": str(model_path),
                    "load_time_seconds": load_time,
                })
            except Exception as e:
                logger.warning(f"Failed to notify bridge of model status: {e}")
    else:
        logger.warning(f"Model not found: {args.model}")
        logger.warning("Server will start without a model (health checks only)")
        logger.warning("Set MODEL_GCS_URI env var or mount a model to enable inference")

        # Notify bridge of no model
        if _bridge:
            try:
                from jarvis_prime.core.cross_repo_bridge import update_model_status
                update_model_status(loaded=False, model_path="")
            except Exception as e:
                pass  # Silent fail for missing model notification

    # Create FastAPI app
    app = FastAPI(
        title="JARVIS-Prime",
        description="Tier-0 Muscle Memory Brain - OpenAI-compatible API",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
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

    # Endpoints
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        """OpenAI-compatible chat completions with streaming support."""
        if not executor.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Set MODEL_GCS_URI or mount model to /app/models/current.gguf"
            )

        # Format messages
        from jarvis_prime.core.model_manager import ChatMessage
        messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
        prompt = executor.format_messages(messages)
        prompt_tokens = len(prompt.split())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # v74.0: Streaming Response (SSE format)
        if request.stream:
            async def stream_generator():
                """Generate SSE stream in OpenAI-compatible format."""
                start = time.time()
                token_count = 0

                try:
                    async for token in executor.generate_stream(
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

                    # Final chunk with finish_reason
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

                    # Record metrics after streaming completes
                    latency_ms = (time.time() - start) * 1000

                    if _bridge:
                        try:
                            from jarvis_prime.core.cross_repo_bridge import record_inference
                            record_inference(
                                tokens_in=prompt_tokens,
                                tokens_out=token_count,
                                latency_ms=latency_ms,
                            )
                        except Exception:
                            pass

                    if trinity_record_inference:
                        try:
                            trinity_record_inference(latency_ms=latency_ms, success=True)
                        except Exception:
                            pass

                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    if trinity_record_inference:
                        try:
                            trinity_record_inference(latency_ms=0, success=False)
                        except Exception:
                            pass
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
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                },
            )

        # Non-streaming response (original behavior)
        try:
            start = time.time()

            # Generate
            response = await executor.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency = time.time() - start
            latency_ms = latency * 1000

            # Estimate tokens
            completion_tokens = len(response.split())

            # Record inference metrics for cost tracking
            if _bridge:
                try:
                    from jarvis_prime.core.cross_repo_bridge import record_inference
                    record_inference(
                        tokens_in=prompt_tokens,
                        tokens_out=completion_tokens,
                        latency_ms=latency_ms,
                    )
                except Exception:
                    pass  # Don't fail inference for metrics

            # v73.0: Record inference health for Trinity heartbeat
            # This enables JARVIS Body to detect "Silent Brain Freeze"
            if trinity_record_inference:
                try:
                    trinity_record_inference(latency_ms=latency_ms, success=True)
                except Exception:
                    pass  # Don't fail inference for metrics

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
            }
        except Exception as e:
            logger.error(f"Chat error: {e}")
            # v73.0: Record failed inference
            if trinity_record_inference:
                try:
                    trinity_record_inference(latency_ms=0, success=False)
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        """Simple text generation."""
        if not executor.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Set MODEL_GCS_URI or mount model to /app/models/current.gguf"
            )
        try:
            start = time.time()

            response = await executor.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency_ms = (time.time() - start) * 1000

            # Estimate tokens
            prompt_tokens = len(request.prompt.split())
            completion_tokens = len(response.split())

            # Record inference metrics
            if _bridge:
                try:
                    from jarvis_prime.core.cross_repo_bridge import record_inference
                    record_inference(
                        tokens_in=prompt_tokens,
                        tokens_out=completion_tokens,
                        latency_ms=latency_ms,
                    )
                except Exception:
                    pass  # Don't fail inference for metrics

            # v73.0: Record inference health for Trinity heartbeat
            if trinity_record_inference:
                try:
                    trinity_record_inference(latency_ms=latency_ms, success=True)
                except Exception:
                    pass

            return {
                "text": response,
                "latency_ms": latency_ms,
            }
        except Exception as e:
            logger.error(f"Generate error: {e}")
            # v73.0: Record failed inference
            if trinity_record_inference:
                try:
                    trinity_record_inference(latency_ms=0, success=False)
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check - always returns healthy for Cloud Run."""
        stats = executor.get_statistics() if executor.is_loaded() else {}

        # Include bridge connection status
        bridge_info = {}
        if _bridge:
            bridge_info = {
                "jarvis_connected": _bridge.state.connected_to_jarvis,
                "jarvis_session_id": _bridge.state.jarvis_session_id or None,
            }

        # v73.0: Include inference health from Trinity bridge
        inference_health = {}
        if trinity_initialized:
            try:
                from jarvis_prime.core.trinity_bridge import get_inference_health
                inference_health = get_inference_health()
            except Exception:
                pass

        return {
            "status": "healthy",
            "model_loaded": executor.is_loaded(),
            "model_path": str(model_path) if model_path.exists() else None,
            "ready_for_inference": executor.is_loaded(),
            "bridge_enabled": _bridge is not None,
            "trinity_enabled": trinity_initialized,
            "inference_health": inference_health,  # v73.0
            **bridge_info,
            **stats,
        }

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
                return {
                    "status": "error",
                    "error": str(e),
                }
        else:
            return {
                "status": "disabled",
                "message": "Cross-repo bridge not enabled",
            }

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [{
                "id": "jarvis-prime",
                "object": "model",
                "owned_by": "jarvis",
            }],
        }

    # ==========================================================================
    # MODEL HOT-RELOAD ENDPOINT - Reactor-Core Integration
    # ==========================================================================
    # Called by Reactor-Core after training to hot-swap the model without restart.

    class ModelReloadRequest(BaseModel):
        """Request to reload model from Reactor-Core."""
        model_path: str
        model_version: str = "unknown"
        model_id: str = ""

    @app.post("/api/v1/models/reload")
    async def reload_model(request: ModelReloadRequest):
        """
        Hot-reload model from Reactor-Core.

        This endpoint is called after Reactor-Core completes training and
        deploys a new model. It allows JARVIS-Prime to reload without restart.
        """
        nonlocal model_path

        logger.info(f"Model reload requested: {request.model_path} (v{request.model_version})")

        try:
            new_model_path = Path(request.model_path)

            if not new_model_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file not found: {request.model_path}"
                )

            # Unload current model
            if executor.is_loaded():
                logger.info("Unloading current model...")
                await executor.close()

            # Load new model
            logger.info(f"Loading new model: {new_model_path}")
            start = time.time()
            await executor.load(new_model_path)
            load_time = time.time() - start

            # Update model_path reference
            model_path = new_model_path

            # Notify bridges
            if _bridge:
                try:
                    from jarvis_prime.core.cross_repo_bridge import update_model_status
                    update_model_status(loaded=True, model_path=str(model_path))
                    await _bridge.notify_jarvis("model_reloaded", {
                        "model_path": str(model_path),
                        "model_version": request.model_version,
                        "load_time_seconds": load_time,
                    })
                except Exception as e:
                    logger.warning(f"Failed to notify bridge: {e}")

            if trinity_initialized:
                try:
                    from jarvis_prime.core.trinity_bridge import update_model_status as trinity_update
                    trinity_update(loaded=True, model_path=str(model_path))
                except Exception as e:
                    logger.warning(f"Failed to notify Trinity: {e}")

            logger.info(f"Model reloaded in {load_time:.2f}s: {model_path}")

            return {
                "status": "success",
                "model_path": str(model_path),
                "model_version": request.model_version,
                "load_time_seconds": load_time,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Model reload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ==========================================================================
    # AGI v77.0 ENDPOINTS - Advanced Cognitive Capabilities
    # ==========================================================================

    class AGIReasonRequest(BaseModel):
        """Request for AGI reasoning."""
        query: str
        strategy: str = "chain_of_thought"  # chain_of_thought, tree_of_thoughts, self_reflection
        context: dict = {}

    class AGIPlanRequest(BaseModel):
        """Request for AGI action planning."""
        goal: str
        context: dict = {}
        constraints: List[str] = []

    class AGIFeedbackRequest(BaseModel):
        """Request to record learning feedback."""
        experience_id: str
        score: float  # -1.0 to 1.0
        comment: Optional[str] = None

    class AGIProcessRequest(BaseModel):
        """Request for full AGI processing pipeline."""
        content: str
        modalities: List[str] = ["text"]
        context: dict = {}
        enable_reasoning: bool = True
        enable_learning: bool = True

    @app.post("/agi/reason")
    async def agi_reason(request: AGIReasonRequest):
        """
        Execute advanced reasoning on a query.

        Strategies:
        - chain_of_thought: Sequential step-by-step reasoning
        - tree_of_thoughts: Parallel exploration of multiple paths
        - self_reflection: Meta-cognitive analysis and correction
        - hypothesis_test: Evidence-based hypothesis testing
        """
        if not agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            start = time.time()
            result = await agi_hub.reason(
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
        """
        Generate an action plan for a goal.

        Uses ActionModel and GoalInference from AGI Orchestrator
        to create executable action sequences.
        """
        if not agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            start = time.time()
            result = await agi_hub.plan(
                goal=request.goal,
                context=request.context,
            )
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
        """
        Full AGI processing pipeline.

        1. Analyzes request complexity
        2. Applies appropriate reasoning strategy
        3. Coordinates multiple AGI models
        4. Records experience for learning
        """
        if not agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            # Create inference function wrapper
            async def inference_fn(prompt: str, **kwargs):
                if executor.is_loaded():
                    return await executor.generate(
                        prompt=prompt,
                        max_tokens=512,
                        temperature=0.7,
                    )
                return prompt  # Passthrough if no model

            result = await agi_hub.process(
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
        """
        Record feedback for continuous learning.

        Feedback is used to improve future responses through
        experience replay and online fine-tuning.
        """
        if not agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            success = await agi_hub.record_feedback(
                experience_id=request.experience_id,
                score=request.score,
                comment=request.comment,
            )

            return {
                "status": "success" if success else "failed",
                "experience_id": request.experience_id,
                "score": request.score,
            }
        except Exception as e:
            logger.error(f"AGI feedback error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agi/learning/trigger")
    async def agi_learning_trigger(force: bool = False):
        """
        Trigger a learning update.

        Processes accumulated experiences to update model weights
        using EWC (Elastic Weight Consolidation) to prevent forgetting.
        """
        if not agi_hub:
            raise HTTPException(status_code=503, detail="AGI Hub not initialized")

        try:
            result = await agi_hub.trigger_learning_update(force=force)
            return {
                "status": "success",
                "result": result,
            }
        except Exception as e:
            logger.error(f"AGI learning trigger error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agi/status")
    async def agi_status():
        """Get AGI subsystem status and metrics."""
        if not agi_hub:
            return {
                "status": "not_initialized",
                "message": "AGI Hub not available",
            }

        try:
            status = agi_hub.get_status()
            health = await agi_hub.health_check()

            return {
                "status": "ok",
                "initialized": status["initialized"],
                "healthy": health["healthy"],
                "subsystems": status["subsystems"],
                "metrics": status["metrics"],
            }
        except Exception as e:
            logger.error(f"AGI status error: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    @app.get("/agi/learning/stats")
    async def agi_learning_stats():
        """Get continuous learning statistics."""
        if not agi_hub or not agi_hub.learning_engine:
            return {
                "status": "not_available",
                "message": "Learning engine not initialized",
            }

        try:
            stats = agi_hub.learning_engine.get_statistics()
            return {
                "status": "ok",
                "statistics": stats,
            }
        except Exception as e:
            logger.error(f"AGI learning stats error: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down...")

        # v77.0: Shutdown AGI Integration Hub
        if agi_hub:
            try:
                await shutdown_agi_hub()
                logger.info("AGI Integration Hub shutdown complete")
            except Exception as e:
                logger.warning(f"AGI Hub shutdown error: {e}")

        # v72.0: Shutdown PROJECT TRINITY connection
        if trinity_initialized:
            try:
                from jarvis_prime.core.trinity_bridge import shutdown_trinity
                await shutdown_trinity()
                logger.info("PROJECT TRINITY: J-Prime disconnected from Trinity network")
            except Exception as e:
                logger.warning(f"Trinity shutdown error: {e}")

        # Log final cost summary
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
                    logger.info(f"  Cloud Equivalent: ${cost_summary.get('cloud_equivalent_cost_usd', 0):.4f}")
                    logger.info(f"  Savings: ${cost_summary.get('savings_usd', 0):.4f} ({cost_summary.get('savings_percent', 0):.1f}%)")
                    logger.info("=" * 50)

                # Notify JARVIS of shutdown
                await _bridge.notify_jarvis("shutdown", cost_summary)

                # Shutdown bridge
                await shutdown_bridge()
                logger.info("Cross-repo bridge shutdown complete")
            except Exception as e:
                logger.warning(f"Bridge shutdown error: {e}")

        await executor.close()

    # Print startup info
    logger.info("=" * 60)
    logger.info("JARVIS-Prime Tier-0 Brain Server (v2.0)")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path.name if model_path.exists() else 'Not loaded'}")
    logger.info(f"Context: {args.ctx_size} tokens")
    logger.info(f"GPU layers: {args.gpu_layers}")
    logger.info(f"Listening: http://{args.host}:{args.port}")

    # Bridge status
    if _bridge:
        if _bridge.state.connected_to_jarvis:
            logger.info(f"JARVIS Bridge: Connected (session: {_bridge.state.jarvis_session_id[:8]}...)")
        else:
            logger.info("JARVIS Bridge: Enabled (standalone mode)")
    else:
        logger.info("JARVIS Bridge: Disabled")

    # v72.0: Trinity status
    if trinity_initialized:
        logger.info("PROJECT TRINITY: Connected (Mind component online)")
    else:
        logger.info("PROJECT TRINITY: Not connected")

    # v77.0: AGI status
    if agi_hub:
        logger.info("AGI Integration Hub: Active")
        logger.info("  - Reasoning Engine: Chain-of-Thought, Tree-of-Thoughts, Self-Reflection")
        logger.info("  - Continuous Learning: EWC, Synaptic Intelligence")
        logger.info("  - MultiModal Fusion: Screen, Audio, Gesture")
    else:
        logger.info("AGI Integration Hub: Not connected")

    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  POST http://localhost:{args.port}/v1/chat/completions")
    logger.info(f"  POST http://localhost:{args.port}/generate")
    logger.info(f"  GET  http://localhost:{args.port}/health")
    logger.info(f"  GET  http://localhost:{args.port}/metrics")
    if agi_hub:
        logger.info("")
        logger.info("AGI Endpoints (v77.0):")
        logger.info(f"  POST http://localhost:{args.port}/agi/reason")
        logger.info(f"  POST http://localhost:{args.port}/agi/plan")
        logger.info(f"  POST http://localhost:{args.port}/agi/process")
        logger.info(f"  POST http://localhost:{args.port}/agi/feedback")
        logger.info(f"  POST http://localhost:{args.port}/agi/learning/trigger")
        logger.info(f"  GET  http://localhost:{args.port}/agi/status")
        logger.info(f"  GET  http://localhost:{args.port}/agi/learning/stats")
    logger.info("=" * 60)

    # Run server
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.debug else "info",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
