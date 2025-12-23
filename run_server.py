#!/usr/bin/env python3
"""
JARVIS-Prime Server - Quick Start Script
=========================================

Runs JARVIS-Prime with llama-cpp-python backend.

Usage:
    # Default (TinyLlama on port 8000)
    python run_server.py

    # Custom model
    python run_server.py --model models/mistral-7b.gguf --port 8080

    # With Metal GPU (M1/M2/M3)
    python run_server.py --gpu-layers -1

Endpoints:
    POST /v1/chat/completions  - OpenAI-compatible chat
    POST /generate             - Simple text generation
    GET  /health               - Health check
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
    return parser.parse_args()


async def main():
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

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to script
        model_path = Path(__file__).parent / args.model

    if not model_path.exists():
        logger.error(f"Model not found: {args.model}")
        logger.error("Download a model with:")
        logger.error("  python -m jarvis_prime.docker.model_downloader tinyllama-chat")
        sys.exit(1)

    # Create executor
    config = LlamaCppConfig(
        n_ctx=args.ctx_size,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
        verbose=args.debug,
    )
    executor = LlamaCppExecutor(config)

    # Load model
    logger.info(f"Loading model: {model_path}")
    start = time.time()
    await executor.load(model_path)
    logger.info(f"Model loaded in {time.time() - start:.2f}s")

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
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(prompt.split()) + len(response.split()),
                },
                "x_latency_ms": latency * 1000,
            }
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        """Simple text generation."""
        try:
            start = time.time()

            response = await executor.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            return {
                "text": response,
                "latency_ms": (time.time() - start) * 1000,
            }
        except Exception as e:
            logger.error(f"Generate error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check."""
        return {
            "status": "healthy",
            "model_loaded": executor.is_loaded(),
            "model_path": str(model_path),
            **executor.get_statistics(),
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
        await executor.close()

    # Print startup info
    logger.info("=" * 60)
    logger.info("JARVIS-Prime Tier-0 Brain Server")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path.name}")
    logger.info(f"Context: {args.ctx_size} tokens")
    logger.info(f"GPU layers: {args.gpu_layers}")
    logger.info(f"Listening: http://{args.host}:{args.port}")
    logger.info("")
    logger.info("Endpoints:")
    logger.info(f"  POST http://localhost:{args.port}/v1/chat/completions")
    logger.info(f"  POST http://localhost:{args.port}/generate")
    logger.info(f"  GET  http://localhost:{args.port}/health")
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
