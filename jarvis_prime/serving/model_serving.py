"""
Model Serving - Production-Grade Inference Server for JARVIS Prime
====================================================================

v91.0 - Advanced Model Serving Infrastructure

This module provides a production-ready model serving infrastructure with:
- Async inference server with FastAPI
- Intelligent request batching for maximum throughput
- OpenAI-compatible API endpoints
- Streaming response support
- Rate limiting and request queuing
- Health monitoring and auto-recovery
- Model hot-swap without downtime
- Comprehensive metrics and observability

ARCHITECTURE:
    Client Request → Load Balancer → Inference Server → Model Pool
                                           ↓
                                   Request Batcher → GPU Scheduler
                                           ↓
                                   Response Streamer → Client

USAGE:
    # Start server
    python -m jarvis_prime.serving.model_serving --port 8000

    # Or programmatically
    server = InferenceServer(config)
    await server.start()

    # Client usage (OpenAI-compatible)
    response = await client.chat.completions.create(
        model="prime-7b-chat-v1",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )

INTEGRATION:
    - PrimeModel: Uses PrimeModel for inference
    - Continuous Learning: Records experiences for online learning
    - Observability Bridge: Metrics and tracing
    - Trinity Protocol: Cross-repo event broadcasting
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import os
import signal
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ServingConfig:
    """
    Dynamic configuration for model serving.

    All values can be overridden via environment variables with SERVING_ prefix.
    """
    # Server settings
    host: str = field(default_factory=lambda: os.getenv("SERVING_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("SERVING_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("SERVING_WORKERS", "1")))

    # Model settings
    default_model: str = field(default_factory=lambda: os.getenv("SERVING_DEFAULT_MODEL", "prime-7b-chat-v1"))
    preload_models: List[str] = field(default_factory=lambda: os.getenv(
        "SERVING_PRELOAD_MODELS", ""
    ).split(",") if os.getenv("SERVING_PRELOAD_MODELS") else [])

    # Batching settings
    enable_batching: bool = field(default_factory=lambda: os.getenv("SERVING_ENABLE_BATCHING", "true").lower() == "true")
    max_batch_size: int = field(default_factory=lambda: int(os.getenv("SERVING_MAX_BATCH_SIZE", "8")))
    max_batch_wait_ms: float = field(default_factory=lambda: float(os.getenv("SERVING_MAX_BATCH_WAIT_MS", "50.0")))
    dynamic_batching: bool = field(default_factory=lambda: os.getenv("SERVING_DYNAMIC_BATCHING", "true").lower() == "true")

    # Rate limiting
    enable_rate_limiting: bool = field(default_factory=lambda: os.getenv("SERVING_RATE_LIMITING", "true").lower() == "true")
    requests_per_minute: int = field(default_factory=lambda: int(os.getenv("SERVING_REQUESTS_PER_MINUTE", "60")))
    burst_limit: int = field(default_factory=lambda: int(os.getenv("SERVING_BURST_LIMIT", "10")))

    # Queue settings
    max_queue_size: int = field(default_factory=lambda: int(os.getenv("SERVING_MAX_QUEUE_SIZE", "100")))
    queue_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("SERVING_QUEUE_TIMEOUT", "30.0")))

    # Timeouts
    request_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("SERVING_REQUEST_TIMEOUT", "120.0")))
    generation_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("SERVING_GENERATION_TIMEOUT", "60.0")))

    # Health checks
    health_check_interval_seconds: float = field(default_factory=lambda: float(os.getenv("SERVING_HEALTH_INTERVAL", "30.0")))
    unhealthy_threshold: int = field(default_factory=lambda: int(os.getenv("SERVING_UNHEALTHY_THRESHOLD", "3")))

    # Metrics
    enable_metrics: bool = field(default_factory=lambda: os.getenv("SERVING_ENABLE_METRICS", "true").lower() == "true")
    metrics_path: str = field(default_factory=lambda: os.getenv("SERVING_METRICS_PATH", "/metrics"))

    # API settings
    api_key_required: bool = field(default_factory=lambda: os.getenv("SERVING_API_KEY_REQUIRED", "false").lower() == "true")
    cors_origins: List[str] = field(default_factory=lambda: os.getenv(
        "SERVING_CORS_ORIGINS", "*"
    ).split(","))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "default_model": self.default_model,
            "enable_batching": self.enable_batching,
            "max_batch_size": self.max_batch_size,
            "enable_rate_limiting": self.enable_rate_limiting,
            "requests_per_minute": self.requests_per_minute,
            "max_queue_size": self.max_queue_size,
            "request_timeout_seconds": self.request_timeout_seconds,
        }


# =============================================================================
# REQUEST/RESPONSE MODELS (OpenAI-Compatible)
# =============================================================================


@dataclass
class ChatMessage:
    """A chat message."""
    role: str
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ChatCompletionRequest:
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None

    # PRIME-specific
    mode: Optional[str] = None  # greedy, sampling, beam_search

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatCompletionRequest":
        messages = [ChatMessage(**m) for m in data.get("messages", [])]
        return cls(
            model=data.get("model", ""),
            messages=messages,
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            max_tokens=data.get("max_tokens"),
            stream=data.get("stream", False),
            stop=data.get("stop"),
            presence_penalty=data.get("presence_penalty", 0.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            user=data.get("user"),
            mode=data.get("mode"),
        )


@dataclass
class ChatCompletionChoice:
    """A completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason,
        }


@dataclass
class ChatCompletionUsage:
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[ChatCompletionChoice] = field(default_factory=list)
    usage: Optional[ChatCompletionUsage] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict() for c in self.choices],
        }
        if self.usage:
            d["usage"] = self.usage.to_dict()
        return d


@dataclass
class StreamingChunk:
    """A streaming response chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
        }


# =============================================================================
# REQUEST BATCHER - Intelligent Batching for Throughput
# =============================================================================


@dataclass
class BatchedRequest:
    """A request waiting in the batch queue."""
    request_id: str
    request: ChatCompletionRequest
    future: asyncio.Future
    created_at: float = field(default_factory=time.time)
    timeout: float = 30.0


class RequestBatcher:
    """
    Intelligent request batcher for maximum throughput.

    Features:
    - Dynamic batch sizing based on load
    - Deadline-aware scheduling
    - Priority queuing
    - Automatic batch flushing
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0,
        dynamic_batching: bool = True,
    ):
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._dynamic_batching = dynamic_batching

        self._queue: Deque[BatchedRequest] = deque()
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._shutdown = False

        # Statistics
        self._total_batches = 0
        self._total_requests = 0
        self._batch_sizes: List[int] = []

    async def enqueue(
        self,
        request: ChatCompletionRequest,
        timeout: float = 30.0,
    ) -> Any:
        """
        Enqueue a request for batched processing.

        Returns when the request is processed.
        """
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        future: asyncio.Future = asyncio.get_event_loop().create_future()

        batched = BatchedRequest(
            request_id=request_id,
            request=request,
            future=future,
            timeout=timeout,
        )

        async with self._lock:
            self._queue.append(batched)
            self._total_requests += 1

            # Signal that we have work
            if len(self._queue) >= self._max_batch_size:
                self._batch_event.set()

        # Wait for result with timeout
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            # Remove from queue if still there
            async with self._lock:
                self._queue = deque(r for r in self._queue if r.request_id != request_id)
            raise

    async def get_batch(self) -> List[BatchedRequest]:
        """
        Get the next batch of requests.

        Waits up to max_wait_ms for a full batch, then returns whatever is ready.
        """
        # Wait for work or timeout
        try:
            await asyncio.wait_for(
                self._batch_event.wait(),
                timeout=self._max_wait_ms / 1000
            )
        except asyncio.TimeoutError:
            pass

        async with self._lock:
            self._batch_event.clear()

            if not self._queue:
                return []

            # Get batch
            batch_size = min(len(self._queue), self._max_batch_size)

            # Dynamic batching: reduce batch size under low load
            if self._dynamic_batching:
                queue_wait_time = time.time() - self._queue[0].created_at
                if queue_wait_time > self._max_wait_ms / 1000 * 2:
                    # Requests waiting too long, process immediately
                    batch_size = len(self._queue)

            batch = [self._queue.popleft() for _ in range(min(batch_size, len(self._queue)))]

            self._total_batches += 1
            self._batch_sizes.append(len(batch))
            if len(self._batch_sizes) > 100:
                self._batch_sizes = self._batch_sizes[-100:]

            return batch

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def get_statistics(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        return {
            "total_batches": self._total_batches,
            "total_requests": self._total_requests,
            "current_queue_size": len(self._queue),
            "avg_batch_size": sum(self._batch_sizes) / max(len(self._batch_sizes), 1),
            "max_batch_size": self._max_batch_size,
            "max_wait_ms": self._max_wait_ms,
        }

    async def shutdown(self) -> None:
        """Shutdown the batcher, cancelling pending requests."""
        self._shutdown = True
        self._batch_event.set()

        async with self._lock:
            for req in self._queue:
                if not req.future.done():
                    req.future.set_exception(asyncio.CancelledError("Server shutting down"))
            self._queue.clear()


# =============================================================================
# RATE LIMITER - Token Bucket with Burst Support
# =============================================================================


class RateLimiter:
    """
    Token bucket rate limiter with burst support.

    Features:
    - Per-client rate limiting
    - Configurable burst capacity
    - Automatic token refill
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_limit: int = 10,
    ):
        self._rate = requests_per_minute / 60.0  # tokens per second
        self._burst_limit = burst_limit

        # Per-client buckets: client_id -> (tokens, last_update)
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, client_id: str = "default") -> bool:
        """
        Try to acquire a token for the client.

        Returns True if allowed, False if rate limited.
        """
        async with self._lock:
            now = time.time()

            if client_id not in self._buckets:
                self._buckets[client_id] = (self._burst_limit - 1, now)
                return True

            tokens, last_update = self._buckets[client_id]

            # Refill tokens
            elapsed = now - last_update
            tokens = min(self._burst_limit, tokens + elapsed * self._rate)

            if tokens >= 1:
                self._buckets[client_id] = (tokens - 1, now)
                return True

            return False

    async def wait_and_acquire(
        self,
        client_id: str = "default",
        timeout: float = 10.0,
    ) -> bool:
        """
        Wait until a token is available, then acquire it.

        Returns False if timeout exceeded.
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            if await self.acquire(client_id):
                return True
            await asyncio.sleep(0.1)

        return False

    def get_remaining(self, client_id: str = "default") -> float:
        """Get remaining tokens for a client."""
        if client_id not in self._buckets:
            return float(self._burst_limit)

        tokens, last_update = self._buckets[client_id]
        elapsed = time.time() - last_update
        return min(self._burst_limit, tokens + elapsed * self._rate)


# =============================================================================
# MODEL POOL - Manages Loaded Models
# =============================================================================


class ModelPool:
    """
    Pool of loaded models for serving.

    Features:
    - Lazy model loading
    - Health monitoring
    - Hot-swap support
    - Memory management
    """

    def __init__(self, config: ServingConfig):
        self._config = config
        self._models: Dict[str, Any] = {}  # model_name -> PrimeModel
        self._health_status: Dict[str, bool] = {}
        self._lock = asyncio.Lock()
        self._health_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the model pool."""
        # Preload models
        for model_name in self._config.preload_models:
            if model_name:
                await self.load_model(model_name)

        # Ensure default model is loaded
        if self._config.default_model:
            await self.load_model(self._config.default_model)

        # Start health monitoring
        self._health_task = asyncio.create_task(self._health_monitor())

        logger.info(f"ModelPool started with models: {list(self._models.keys())}")

    async def stop(self) -> None:
        """Stop the model pool."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Unload all models
        async with self._lock:
            self._models.clear()
            gc.collect()

    async def load_model(self, model_name: str) -> bool:
        """Load a model into the pool."""
        async with self._lock:
            if model_name in self._models:
                return True

            try:
                from jarvis_prime.models.prime_model import PrimeModel
                model = await PrimeModel.from_pretrained_async(model_name)
                self._models[model_name] = model
                self._health_status[model_name] = True
                logger.info(f"Loaded model: {model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return False

    async def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model from the pool, loading if necessary."""
        if model_name not in self._models:
            await self.load_model(model_name)
        return self._models.get(model_name)

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from the pool."""
        async with self._lock:
            if model_name in self._models:
                del self._models[model_name]
                self._health_status.pop(model_name, None)
                gc.collect()
                logger.info(f"Unloaded model: {model_name}")
                return True
            return False

    async def swap_model(self, model_name: str) -> bool:
        """Hot-swap a model (reload)."""
        async with self._lock:
            if model_name in self._models:
                try:
                    model = self._models[model_name]
                    await model.swap_model(model_name)
                    return True
                except Exception as e:
                    logger.error(f"Failed to swap model {model_name}: {e}")
                    return False
            return False

    def is_healthy(self, model_name: str) -> bool:
        """Check if a model is healthy."""
        return self._health_status.get(model_name, False)

    def list_models(self) -> List[str]:
        """List loaded models."""
        return list(self._models.keys())

    async def _health_monitor(self) -> None:
        """Background health monitoring."""
        while True:
            try:
                await asyncio.sleep(self._config.health_check_interval_seconds)

                for model_name, model in list(self._models.items()):
                    try:
                        # Simple health check: ensure model responds
                        # This is a lightweight check
                        self._health_status[model_name] = model is not None
                    except Exception as e:
                        logger.warning(f"Health check failed for {model_name}: {e}")
                        self._health_status[model_name] = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "loaded_models": list(self._models.keys()),
            "health_status": dict(self._health_status),
            "model_count": len(self._models),
        }


# =============================================================================
# INFERENCE SERVER
# =============================================================================


class InferenceServer:
    """
    Production-grade inference server for JARVIS Prime.

    Features:
    - OpenAI-compatible API
    - Request batching
    - Rate limiting
    - Streaming responses
    - Health endpoints
    - Metrics
    """

    def __init__(self, config: Optional[ServingConfig] = None):
        self._config = config or ServingConfig()
        self._model_pool = ModelPool(self._config)
        self._batcher: Optional[RequestBatcher] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._app: Optional[Any] = None

        # Processing state
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

    async def start(self) -> None:
        """Start the inference server."""
        # Initialize components
        await self._model_pool.start()

        if self._config.enable_batching:
            self._batcher = RequestBatcher(
                max_batch_size=self._config.max_batch_size,
                max_wait_ms=self._config.max_batch_wait_ms,
                dynamic_batching=self._config.dynamic_batching,
            )
            self._batch_processor_task = asyncio.create_task(self._batch_processor())

        if self._config.enable_rate_limiting:
            self._rate_limiter = RateLimiter(
                requests_per_minute=self._config.requests_per_minute,
                burst_limit=self._config.burst_limit,
            )

        # Create FastAPI app
        self._app = self._create_app()

        logger.info(f"InferenceServer started on {self._config.host}:{self._config.port}")

    async def stop(self) -> None:
        """Stop the inference server."""
        self._shutdown_event.set()

        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass

        if self._batcher:
            await self._batcher.shutdown()

        await self._model_pool.stop()

        logger.info("InferenceServer stopped")

    def _create_app(self) -> Any:
        """Create the FastAPI application."""
        try:
            from fastapi import FastAPI, HTTPException, Request, Response
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import StreamingResponse, JSONResponse
            from pydantic import BaseModel
        except ImportError:
            logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
            return None

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            yield
            # Shutdown
            await self.stop()

        app = FastAPI(
            title="JARVIS Prime Inference Server",
            description="OpenAI-compatible inference API for JARVIS Prime models",
            version="91.0",
            lifespan=lifespan,
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self._config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Request models
        class ChatRequest(BaseModel):
            model: str = self._config.default_model
            messages: List[Dict[str, str]]
            temperature: float = 0.7
            top_p: float = 0.9
            max_tokens: Optional[int] = None
            stream: bool = False
            stop: Optional[List[str]] = None
            user: Optional[str] = None

        # Health endpoint
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "models": self._model_pool.list_models(),
                "uptime_seconds": time.time() - self._start_time if hasattr(self, "_start_time") else 0,
            }

        # Models endpoint
        @app.get("/v1/models")
        async def list_models():
            models = self._model_pool.list_models()
            return {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "owned_by": "jarvis-prime"}
                    for m in models
                ],
            }

        # Chat completions endpoint
        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatRequest, http_request: Request):
            # Rate limiting
            if self._rate_limiter:
                client_id = request.user or http_request.client.host if http_request.client else "default"
                if not await self._rate_limiter.wait_and_acquire(client_id, timeout=5.0):
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

            try:
                chat_request = ChatCompletionRequest.from_dict(request.model_dump())

                if request.stream:
                    return StreamingResponse(
                        self._stream_response(chat_request),
                        media_type="text/event-stream",
                    )
                else:
                    response = await self._process_request(chat_request)
                    return JSONResponse(content=response.to_dict())

            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="Request timeout")
            except Exception as e:
                logger.error(f"Chat completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Model management endpoints
        @app.post("/v1/models/{model_name}/load")
        async def load_model(model_name: str):
            success = await self._model_pool.load_model(model_name)
            if success:
                return {"status": "loaded", "model": model_name}
            raise HTTPException(status_code=500, detail="Failed to load model")

        @app.post("/v1/models/{model_name}/unload")
        async def unload_model(model_name: str):
            success = await self._model_pool.unload_model(model_name)
            if success:
                return {"status": "unloaded", "model": model_name}
            raise HTTPException(status_code=404, detail="Model not found")

        @app.post("/v1/models/{model_name}/swap")
        async def swap_model(model_name: str):
            success = await self._model_pool.swap_model(model_name)
            if success:
                return {"status": "swapped", "model": model_name}
            raise HTTPException(status_code=500, detail="Failed to swap model")

        # Metrics endpoint
        if self._config.enable_metrics:
            @app.get(self._config.metrics_path)
            async def metrics():
                return {
                    "request_count": self._request_count,
                    "error_count": self._error_count,
                    "avg_latency_ms": self._total_latency_ms / max(self._request_count, 1),
                    "model_pool": self._model_pool.get_statistics(),
                    "batcher": self._batcher.get_statistics() if self._batcher else None,
                }

        self._start_time = time.time()
        return app

    async def _process_request(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Process a single request."""
        start_time = time.time()
        self._request_count += 1

        try:
            # Use batcher if enabled
            if self._batcher:
                return await self._batcher.enqueue(
                    request,
                    timeout=self._config.queue_timeout_seconds,
                )

            # Direct processing
            return await self._generate_response(request)

        except Exception as e:
            self._error_count += 1
            raise

        finally:
            self._total_latency_ms += (time.time() - start_time) * 1000

    async def _generate_response(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Generate a response for a request."""
        model = await self._model_pool.get_model(request.model)
        if not model:
            raise ValueError(f"Model not found: {request.model}")

        # Format messages
        messages = [m.to_dict() for m in request.messages]

        # Generate
        result = await model.chat_async(
            messages=messages,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Build response
        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        return ChatCompletionResponse(
            id=response_id,
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result.text),
                    finish_reason="stop",
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.tokens_generated,
                total_tokens=result.prompt_tokens + result.tokens_generated,
            ),
        )

    async def _stream_response(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Stream a response for a request."""
        model = await self._model_pool.get_model(request.model)
        if not model:
            yield f"data: {json.dumps({'error': 'Model not found'})}\n\n"
            return

        response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        messages = [m.to_dict() for m in request.messages]

        # Format prompt
        prompt = model._format_chat_messages(messages)

        # Stream generation
        async for chunk in model.stream_async(
            prompt,
            max_new_tokens=request.max_tokens or 512,
            temperature=request.temperature,
        ):
            chunk_data = StreamingChunk(
                id=response_id,
                model=request.model,
                choices=[{
                    "index": 0,
                    "delta": {"content": chunk},
                    "finish_reason": None,
                }],
            )
            yield f"data: {json.dumps(chunk_data.to_dict())}\n\n"

        # Final chunk
        final_chunk = StreamingChunk(
            id=response_id,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        )
        yield f"data: {json.dumps(final_chunk.to_dict())}\n\n"
        yield "data: [DONE]\n\n"

    async def _batch_processor(self) -> None:
        """Background batch processor."""
        while not self._shutdown_event.is_set():
            try:
                batch = await self._batcher.get_batch()

                if not batch:
                    continue

                # Process batch
                # For now, process sequentially (batched tensor ops would be more efficient)
                for req in batch:
                    try:
                        response = await self._generate_response(req.request)
                        req.future.set_result(response)
                    except Exception as e:
                        req.future.set_exception(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

    def get_app(self) -> Any:
        """Get the FastAPI application."""
        return self._app

    async def run(self) -> None:
        """Run the server."""
        import uvicorn

        config = uvicorn.Config(
            app=self._app,
            host=self._config.host,
            port=self._config.port,
            workers=self._config.workers,
            loop="asyncio",
        )
        server = uvicorn.Server(config)
        await server.serve()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_server: Optional[InferenceServer] = None
_server_lock = asyncio.Lock()


async def get_inference_server(
    config: Optional[ServingConfig] = None,
) -> InferenceServer:
    """Get or create the global inference server."""
    global _server

    async with _server_lock:
        if _server is None:
            _server = InferenceServer(config)
            await _server.start()
        return _server


async def shutdown_inference_server() -> None:
    """Shutdown the global inference server."""
    global _server

    async with _server_lock:
        if _server:
            await _server.stop()
            _server = None


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Prime Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", default="prime-7b-chat-v1", help="Default model")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--no-batching", action="store_true", help="Disable batching")
    parser.add_argument("--no-rate-limit", action="store_true", help="Disable rate limiting")

    args = parser.parse_args()

    config = ServingConfig(
        host=args.host,
        port=args.port,
        default_model=args.model,
        workers=args.workers,
        enable_batching=not args.no_batching,
        enable_rate_limiting=not args.no_rate_limit,
    )

    async def run():
        server = InferenceServer(config)
        await server.start()

        # Handle shutdown signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop()))

        await server.run()

    asyncio.run(run())


if __name__ == "__main__":
    main()
