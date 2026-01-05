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
        from pydantic import BaseModel
        import uvicorn
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

    # Create executor (may be unloaded)
    config = LlamaCppConfig(
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
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
        """OpenAI-compatible chat completions."""
        if not executor.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Set MODEL_GCS_URI or mount model to /app/models/current.gguf"
            )
        try:
            start = time.time()

            # Format messages
            from jarvis_prime.core.model_manager import ChatMessage
            messages = [ChatMessage(role=m.role, content=m.content) for m in request.messages]
            prompt = executor.format_messages(messages)

            # Generate
            response = await executor.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency = time.time() - start
            latency_ms = latency * 1000

            # Estimate tokens
            prompt_tokens = len(prompt.split())
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
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
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

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down...")

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

    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  POST http://localhost:{args.port}/v1/chat/completions")
    logger.info(f"  POST http://localhost:{args.port}/generate")
    logger.info(f"  GET  http://localhost:{args.port}/health")
    logger.info(f"  GET  http://localhost:{args.port}/metrics")
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
