#!/usr/bin/env python3
"""
JARVIS-Prime Vision Server — Standalone LLaVA FastAPI server (v236.0)
=====================================================================

Serves LLaVA multimodal inference on a dedicated port (default 8001)
alongside the existing text server on port 8000.

Design:
  - [F2] Import: llama_cpp.llama_chat_format.Llava16ChatHandler (correct path)
  - [F6] Non-blocking startup: model loads via asyncio.create_task()
  - [F7] Model discovery by convention: scans models-dir for llava*.gguf + mmproj*.gguf
  - Semaphore(1) serializes inference (llama-cpp-python not thread-safe for LLaVA)
  - Queue depth cap via VISION_MAX_QUEUE env var (default 3) → HTTP 429

Usage:
    python vision_server.py --port 8001 --models-dir /opt/jarvis-prime/models
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("jarvis-prime-vision")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Pydantic request/response models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ContentBlock(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None  # {"url": "data:image/jpeg;base64,..."}


class VisionMessage(BaseModel):
    role: str
    content: Union[str, List[ContentBlock]]


class VisionChatRequest(BaseModel):
    model: str = "llava-v1.6-mistral-7b"
    messages: List[VisionMessage]
    max_tokens: int = 512
    temperature: float = 0.1
    stream: bool = False


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class _ServerState:
    """Mutable state holder for model + server lifecycle."""
    def __init__(self):
        self.llm = None
        self.model_loaded: bool = False
        self.phase: str = "init"  # init → loading → ready | error
        self.error: Optional[str] = None
        self.model_path: Optional[str] = None
        self.mmproj_path: Optional[str] = None
        self.start_time: float = time.time()
        self.requests_served: int = 0


_state = _ServerState()

# Concurrency control — llama-cpp-python is NOT thread-safe for LLaVA
_inference_sem: asyncio.Semaphore = asyncio.Semaphore(1)
_max_queue: int = int(os.getenv("VISION_MAX_QUEUE", "3"))
_queue_depth: int = 0


# ---------------------------------------------------------------------------
# Model discovery (F7: by convention, not manifest)
# ---------------------------------------------------------------------------

def _discover_models(models_dir: str) -> Tuple[str, str]:
    """Find LLaVA GGUF and mmproj by filename convention."""
    model_path = mmproj_path = None
    for f in Path(models_dir).glob("*.gguf"):
        name = f.name.lower()
        if "mmproj" in name:
            mmproj_path = str(f)
        elif "llava" in name:
            model_path = str(f)
    if not model_path:
        raise FileNotFoundError(f"No llava*.gguf found in {models_dir}")
    if not mmproj_path:
        raise FileNotFoundError(f"No mmproj*.gguf found in {models_dir}")
    return model_path, mmproj_path


# ---------------------------------------------------------------------------
# Synchronous model loading (runs in executor)
# ---------------------------------------------------------------------------

def _load_model_sync(model_path: str, mmproj_path: str):
    """Load LLaVA + mmproj CLIP into llama-cpp-python. Runs in thread executor."""
    # F2: Correct import path — NOT llama_cpp.llava_cpp
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava16ChatHandler

    logger.info(f"Loading mmproj CLIP: {mmproj_path}")
    chat_handler = Llava16ChatHandler(clip_model_path=mmproj_path)

    logger.info(f"Loading LLaVA model: {model_path}")
    _state.llm = Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=int(os.getenv("VISION_N_CTX", "4096")),
        n_gpu_layers=0,  # CPU only on GCP
        n_threads=int(os.getenv("VISION_N_THREADS", "4")),
        use_mmap=True,
        verbose=False,
    )
    logger.info("LLaVA model loaded successfully")


# ---------------------------------------------------------------------------
# Background model loader (F6: non-blocking startup)
# ---------------------------------------------------------------------------

async def _load_model_background(models_dir: str):
    """Load model in background so health endpoint is reachable immediately."""
    loop = asyncio.get_event_loop()
    _state.phase = "loading"
    try:
        model_path, mmproj_path = _discover_models(models_dir)
        _state.model_path = model_path
        _state.mmproj_path = mmproj_path
        logger.info(f"Discovered: model={model_path}, mmproj={mmproj_path}")
        await loop.run_in_executor(None, _load_model_sync, model_path, mmproj_path)
        _state.model_loaded = True
        _state.phase = "ready"
    except Exception as e:
        _state.phase = "error"
        _state.error = str(e)
        logger.error(f"Model loading failed: {e}")


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def _convert_messages(messages: List[VisionMessage]) -> List[dict]:
    """Convert Pydantic VisionMessage list to dicts for create_chat_completion."""
    converted = []
    for msg in messages:
        if isinstance(msg.content, str):
            converted.append({"role": msg.role, "content": msg.content})
        else:
            parts = []
            for block in msg.content:
                if block.type == "text" and block.text:
                    parts.append({"type": "text", "text": block.text})
                elif block.type == "image_url" and block.image_url:
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": block.image_url["url"]},
                    })
            converted.append({"role": msg.role, "content": parts})
    return converted


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(models_dir: str = "/opt/jarvis-prime/models") -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(title="JARVIS-Prime Vision Server", version="236.0")

    @app.on_event("startup")
    async def startup():
        # F6: Non-blocking — health endpoint reachable during model load
        asyncio.create_task(_load_model_background(models_dir))

    @app.get("/health")
    async def health():
        return {
            "status": "healthy" if _state.model_loaded else _state.phase,
            "model_loaded": _state.model_loaded,
            "ready_for_inference": _state.model_loaded and _state.phase == "ready",
            "phase": _state.phase,
            "error": _state.error,
            "model_path": _state.model_path,
            "mmproj_path": _state.mmproj_path,
            "uptime_seconds": round(time.time() - _state.start_time, 1),
            "requests_served": _state.requests_served,
            "server": "jarvis-prime-vision",
            "version": "236.0",
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: VisionChatRequest):
        global _queue_depth

        if not _state.model_loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Model still loading (phase: {_state.phase})",
            )

        if request.stream:
            raise HTTPException(
                status_code=400,
                detail="Streaming not supported for vision inference",
            )

        # Queue depth cap — reject if too many waiting
        if _queue_depth >= _max_queue:
            raise HTTPException(
                status_code=429,
                detail=f"Vision server busy ({_queue_depth} requests queued, max {_max_queue})",
            )

        _queue_depth += 1
        try:
            async with _inference_sem:
                messages = _convert_messages(request.messages)
                loop = asyncio.get_event_loop()
                start = time.time()

                result = await loop.run_in_executor(
                    None,
                    lambda: _state.llm.create_chat_completion(
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    ),
                )

                elapsed = time.time() - start
                _state.requests_served += 1
                logger.info(f"Vision inference completed in {elapsed:.1f}s")

                # Return OpenAI-compatible response with extra timing field
                return {
                    "id": f"vision-{_state.requests_served}",
                    "object": "chat.completion",
                    "model": request.model,
                    "choices": result.get("choices", []),
                    "usage": result.get("usage", {}),
                    "x_inference_time_seconds": round(elapsed, 2),
                }
        finally:
            _queue_depth -= 1

    return app


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="JARVIS-Prime Vision Server (LLaVA)")
    parser.add_argument("--port", type=int, default=int(os.getenv("VISION_PORT", "8001")))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--models-dir", type=str, default=os.getenv("MODEL_CACHE", "/opt/jarvis-prime/models"))
    args = parser.parse_args()

    app = create_app(models_dir=args.models_dir)
    logger.info(f"Starting vision server on {args.host}:{args.port} (models: {args.models_dir})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
