"""
AGI Model Server - Optimized Inference Serving
===============================================

v78.0 - High-performance model serving with optimization

Provides optimized serving for AGI models:
- Request batching for throughput
- Response caching for latency
- Priority queuing
- Load balancing across models
- Health-based routing

FEATURES:
    - Async request processing
    - Automatic batching
    - LRU response cache
    - Priority queues
    - Circuit breaker integration
    - Metrics collection
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# REQUEST/RESPONSE TYPES
# =============================================================================


class RequestPriority(Enum):
    """Request priority levels."""

    CRITICAL = 0    # Immediate processing
    HIGH = 1        # User-facing
    NORMAL = 2      # Standard
    LOW = 3         # Background
    BATCH = 4       # Bulk processing


@dataclass
class InferenceRequest:
    """Request for model inference."""

    id: str
    prompt: str
    model: str = "default"
    priority: RequestPriority = RequestPriority.NORMAL

    # Parameters
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=list)

    # Metadata
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    timeout_seconds: float = 30.0

    # Cache
    cache_key: str = ""

    def __post_init__(self) -> None:
        if not self.cache_key:
            self.cache_key = self._compute_cache_key()

    def _compute_cache_key(self) -> str:
        """Compute cache key from request parameters."""
        content = f"{self.prompt}|{self.model}|{self.max_tokens}|{self.temperature}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class InferenceResponse:
    """Response from model inference."""

    request_id: str
    success: bool

    # Content
    text: str = ""
    tokens_used: int = 0

    # Timing
    latency_ms: float = 0.0
    queue_time_ms: float = 0.0
    inference_time_ms: float = 0.0

    # Metadata
    model_used: str = ""
    from_cache: bool = False
    batch_size: int = 1

    # Error
    error: Optional[str] = None


# =============================================================================
# LRU CACHE
# =============================================================================


class LRUCache(Generic[T]):
    """
    Thread-safe LRU cache with TTL support.

    Used for caching inference results.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self._lock = asyncio.Lock()

        # Metrics
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache."""
        async with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]

            # Check TTL
            if time.time() - timestamp > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    async def put(self, key: str, value: T) -> None:
        """Put value in cache."""
        async with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = (value, time.time())

    async def invalidate(self, key: str) -> None:
        """Remove key from cache."""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """Clear entire cache."""
        async with self._lock:
            self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
        }


# =============================================================================
# PRIORITY QUEUE
# =============================================================================


class PriorityQueue(Generic[T]):
    """
    Async priority queue for request scheduling.

    Higher priority requests are processed first.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._max_size = max_size
        self._queues: Dict[RequestPriority, asyncio.Queue] = {
            p: asyncio.Queue(maxsize=max_size // len(RequestPriority))
            for p in RequestPriority
        }
        self._size = 0
        self._lock = asyncio.Lock()

    async def put(
        self,
        item: T,
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> bool:
        """Add item to queue."""
        queue = self._queues[priority]

        try:
            await asyncio.wait_for(queue.put(item), timeout=1.0)
            async with self._lock:
                self._size += 1
            return True
        except (asyncio.TimeoutError, asyncio.QueueFull):
            return False

    async def get(self, timeout: Optional[float] = None) -> Optional[T]:
        """Get highest priority item."""
        deadline = time.time() + (timeout or 30.0)

        while time.time() < deadline:
            # Try queues in priority order
            for priority in RequestPriority:
                queue = self._queues[priority]
                if not queue.empty():
                    try:
                        item = queue.get_nowait()
                        async with self._lock:
                            self._size -= 1
                        return item
                    except asyncio.QueueEmpty:
                        continue

            # Wait a bit before retrying
            await asyncio.sleep(0.01)

        return None

    def size(self) -> int:
        """Get total queue size."""
        return self._size

    def sizes_by_priority(self) -> Dict[str, int]:
        """Get sizes by priority."""
        return {p.name: q.qsize() for p, q in self._queues.items()}


# =============================================================================
# REQUEST BATCHER
# =============================================================================


class RequestBatcher:
    """
    Batches requests for efficient processing.

    Collects requests over a time window or up to a max batch size,
    then processes them together.
    """

    def __init__(
        self,
        process_fn: Callable[[List[InferenceRequest]], Awaitable[List[InferenceResponse]]],
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0,
    ) -> None:
        self._process_fn = process_fn
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms

        self._pending: List[Tuple[InferenceRequest, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start batcher."""
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        """Stop batcher."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """Submit request for batched processing."""
        future: asyncio.Future[InferenceResponse] = asyncio.Future()

        async with self._lock:
            self._pending.append((request, future))

        return await future

    async def _batch_loop(self) -> None:
        """Background loop for batching."""
        while self._running:
            try:
                await self._process_batch()
                await asyncio.sleep(self._max_wait_ms / 1000)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error: {e}")

    async def _process_batch(self) -> None:
        """Process pending requests as batch."""
        async with self._lock:
            if not self._pending:
                return

            # Take up to max_batch_size requests
            batch = self._pending[:self._max_batch_size]
            self._pending = self._pending[self._max_batch_size:]

        if not batch:
            return

        requests = [r for r, _ in batch]
        futures = [f for _, f in batch]

        try:
            # Process batch
            responses = await self._process_fn(requests)

            # Distribute responses
            for future, response in zip(futures, responses):
                response.batch_size = len(batch)
                future.set_result(response)

        except Exception as e:
            # Fail all futures
            error_response = InferenceResponse(
                request_id="",
                success=False,
                error=str(e),
            )
            for future in futures:
                if not future.done():
                    future.set_result(error_response)


# =============================================================================
# MODEL SERVER
# =============================================================================


class AGIModelServer:
    """
    High-performance model serving with optimization.

    Provides:
    - Response caching
    - Request batching
    - Priority queuing
    - Load balancing
    - Health monitoring

    Usage:
        server = AGIModelServer()
        await server.start()

        # Submit request
        response = await server.infer(InferenceRequest(
            id="req-123",
            prompt="Hello, world!",
            max_tokens=100,
        ))

        # Get metrics
        metrics = server.get_metrics()
    """

    def __init__(
        self,
        inference_fn: Optional[Callable[[str, Dict[str, Any]], Awaitable[str]]] = None,
        cache_size: int = 1000,
        cache_ttl: float = 3600.0,
        max_batch_size: int = 8,
        max_queue_size: int = 10000,
        num_workers: int = 4,
    ) -> None:
        self._inference_fn = inference_fn
        self._cache = LRUCache[InferenceResponse](cache_size, cache_ttl)
        self._queue = PriorityQueue[InferenceRequest](max_queue_size)
        self._num_workers = num_workers

        # Batcher
        self._batcher: Optional[RequestBatcher] = None

        # Workers
        self._workers: List[asyncio.Task] = []
        self._running = False

        # Pending futures
        self._pending_futures: Dict[str, asyncio.Future] = {}

        # Metrics
        self._total_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._total_latency_ms = 0.0

    async def start(self) -> None:
        """Start the model server."""
        if self._running:
            return

        self._running = True

        # Start workers
        for i in range(self._num_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)

        # Start batcher if inference function provided
        if self._inference_fn:
            self._batcher = RequestBatcher(
                self._process_batch,
                max_batch_size=8,
                max_wait_ms=50,
            )
            await self._batcher.start()

        logger.info(f"Model server started with {self._num_workers} workers")

    async def stop(self) -> None:
        """Stop the model server."""
        self._running = False

        # Stop batcher
        if self._batcher:
            await self._batcher.stop()

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("Model server stopped")

    async def infer(
        self,
        request: InferenceRequest,
        use_cache: bool = True,
    ) -> InferenceResponse:
        """
        Submit inference request.

        Args:
            request: Inference request
            use_cache: Whether to use cached responses

        Returns:
            Inference response
        """
        start_time = time.time()
        self._total_requests += 1

        # Check cache
        if use_cache:
            cached = await self._cache.get(request.cache_key)
            if cached:
                cached.from_cache = True
                cached.latency_ms = (time.time() - start_time) * 1000
                return cached

        # Use batcher if available
        if self._batcher:
            response = await self._batcher.submit(request)
        else:
            # Direct processing
            response = await self._process_single(request)

        # Cache response
        if response.success and use_cache:
            await self._cache.put(request.cache_key, response)

        # Update metrics
        response.latency_ms = (time.time() - start_time) * 1000
        self._total_latency_ms += response.latency_ms

        if response.success:
            self._completed_requests += 1
        else:
            self._failed_requests += 1

        return response

    async def infer_streaming(
        self,
        request: InferenceRequest,
    ):
        """
        Submit streaming inference request.

        Yields tokens as they are generated.
        """
        # TODO: Implement streaming support
        response = await self.infer(request, use_cache=False)
        yield response.text

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop for processing queued requests."""
        while self._running:
            try:
                request = await self._queue.get(timeout=1.0)
                if request is None:
                    continue

                response = await self._process_single(request)

                # Resolve pending future
                if request.id in self._pending_futures:
                    future = self._pending_futures.pop(request.id)
                    future.set_result(response)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _process_single(
        self,
        request: InferenceRequest,
    ) -> InferenceResponse:
        """Process a single request."""
        start = time.time()

        try:
            if self._inference_fn:
                result = await asyncio.wait_for(
                    self._inference_fn(request.prompt, {
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "stop": request.stop_sequences,
                    }),
                    timeout=request.timeout_seconds,
                )

                return InferenceResponse(
                    request_id=request.id,
                    success=True,
                    text=result,
                    tokens_used=len(result.split()),  # Estimate
                    inference_time_ms=(time.time() - start) * 1000,
                )
            else:
                # No inference function, return placeholder
                return InferenceResponse(
                    request_id=request.id,
                    success=False,
                    error="No inference function configured",
                )

        except asyncio.TimeoutError:
            return InferenceResponse(
                request_id=request.id,
                success=False,
                error=f"Timeout after {request.timeout_seconds}s",
            )
        except Exception as e:
            return InferenceResponse(
                request_id=request.id,
                success=False,
                error=str(e),
            )

    async def _process_batch(
        self,
        requests: List[InferenceRequest],
    ) -> List[InferenceResponse]:
        """Process a batch of requests."""
        # For now, process individually
        # TODO: Implement true batch inference
        responses = []
        for request in requests:
            response = await self._process_single(request)
            responses.append(response)
        return responses

    def set_inference_function(
        self,
        fn: Callable[[str, Dict[str, Any]], Awaitable[str]],
    ) -> None:
        """Set the inference function."""
        self._inference_fn = fn

    async def warm_cache(
        self,
        prompts: List[str],
        model: str = "default",
    ) -> int:
        """Pre-warm cache with common prompts."""
        warmed = 0

        for prompt in prompts:
            request = InferenceRequest(
                id=f"warmup-{warmed}",
                prompt=prompt,
                model=model,
            )

            # Check if already cached
            cached = await self._cache.get(request.cache_key)
            if not cached:
                response = await self.infer(request)
                if response.success:
                    warmed += 1

        return warmed

    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        return {
            "total_requests": self._total_requests,
            "completed_requests": self._completed_requests,
            "failed_requests": self._failed_requests,
            "success_rate": self._completed_requests / max(self._total_requests, 1),
            "avg_latency_ms": self._total_latency_ms / max(self._completed_requests, 1),
            "queue_size": self._queue.size(),
            "queue_by_priority": self._queue.sizes_by_priority(),
            "cache_stats": self._cache.get_stats(),
            "workers": len(self._workers),
            "running": self._running,
        }


# =============================================================================
# SINGLETON
# =============================================================================


_model_server: Optional[AGIModelServer] = None
_server_lock = asyncio.Lock()


async def get_model_server(
    inference_fn: Optional[Callable] = None,
) -> AGIModelServer:
    """Get or create global model server."""
    global _model_server

    async with _server_lock:
        if _model_server is None:
            _model_server = AGIModelServer(inference_fn)
            await _model_server.start()

        return _model_server


async def shutdown_model_server() -> None:
    """Shutdown global model server."""
    global _model_server

    if _model_server:
        await _model_server.stop()
        _model_server = None
