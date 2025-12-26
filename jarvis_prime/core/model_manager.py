"""
Prime Model Manager - Unified Tier-0 Brain Interface
=====================================================

The central orchestrator for JARVIS-Prime, coordinating:
- Model Registry & Versioning
- Hot-Swap zero-downtime reloads
- Hybrid Tier 0/1 Routing
- Telemetry with PII anonymization
- OpenAI-compatible API interface

This is the main entry point for all inference requests.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Type Definitions
# ============================================================================

T = TypeVar("T")


@dataclass
class ChatMessage:
    """OpenAI-compatible chat message"""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


@dataclass
class CompletionRequest:
    """OpenAI-compatible completion request"""
    model: str = "jarvis-prime"
    messages: List[ChatMessage] = field(default_factory=list)
    prompt: Optional[str] = None  # For legacy completions
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    stop: Optional[List[str]] = None
    user: Optional[str] = None

    # JARVIS-Prime extensions
    force_tier: Optional[str] = None  # "tier_0" or "tier_1"
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionChoice:
    """OpenAI-compatible completion choice"""
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


@dataclass
class Usage:
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class CompletionResponse:
    """OpenAI-compatible completion response"""
    id: str
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = "jarvis-prime"
    choices: List[CompletionChoice] = field(default_factory=list)
    usage: Optional[Usage] = None

    # JARVIS-Prime extensions
    tier_used: str = "tier_0"
    model_version: str = ""
    complexity_score: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {"role": c.message.role, "content": c.message.content},
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            } if self.usage else None,
            # Extensions (prefixed with x_ for OpenAI compatibility)
            "x_tier_used": self.tier_used,
            "x_model_version": self.model_version,
            "x_complexity_score": self.complexity_score,
            "x_latency_ms": self.latency_ms,
        }


# ============================================================================
# Model Executor Interface
# ============================================================================

class ModelExecutor(ABC):
    """
    Abstract interface for model execution.

    Implement this for different model backends (Llama, transformers, etc.)
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate text from prompt"""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text"""
        ...

    @abstractmethod
    async def load(self, model_path: Path, **kwargs) -> None:
        """Load model from path"""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Unload model from memory"""
        ...

    @abstractmethod
    async def validate(self) -> bool:
        """Validate loaded model"""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        ...


class CloudAPIExecutor(ABC):
    """
    Abstract interface for cloud API execution (Tier 1).

    Implement for Claude, GPT-4, etc.
    """

    @abstractmethod
    async def complete(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Complete using cloud API"""
        ...

    @abstractmethod
    async def complete_stream(
        self,
        messages: List[ChatMessage],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream completion from cloud API"""
        ...


# ============================================================================
# LlamaModelExecutor - Concrete implementation using existing LlamaModel
# ============================================================================

class LlamaModelExecutor(ModelExecutor):
    """
    Executor using the existing LlamaModel from jarvis_prime.models
    """

    def __init__(self):
        self._model = None
        self._config = None

    async def load(self, model_path: Path, **kwargs) -> None:
        """Load LlamaModel"""
        from jarvis_prime.models import LlamaModel
        from jarvis_prime.configs import LlamaModelConfig

        # Create config from kwargs or use defaults
        self._config = LlamaModelConfig(
            model_name=str(model_path),
            device=kwargs.get("device", "auto"),
        )

        self._model = LlamaModel(self._config)
        self._model.load()

        logger.info(f"LlamaModelExecutor loaded: {model_path}")

    async def unload(self) -> None:
        """Unload model"""
        if self._model:
            del self._model
            self._model = None

            import gc
            gc.collect()

            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        logger.info("LlamaModelExecutor unloaded")

    async def validate(self) -> bool:
        """Validate by running a simple generation"""
        if not self._model or not self._model._is_loaded:
            return False

        try:
            result = self._model.generate("Hello", max_length=20)
            return len(result) > 0
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def is_loaded(self) -> bool:
        return self._model is not None and self._model._is_loaded

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Generate using LlamaModel"""
        if not self._model:
            raise RuntimeError("Model not loaded")

        return await self._model.generate_async(
            prompt,
            max_length=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generation (simulated for LlamaModel)"""
        # LlamaModel doesn't natively support streaming,
        # so we generate and yield in chunks
        result = await self.generate(prompt, max_tokens, temperature, **kwargs)

        # Yield in word chunks to simulate streaming
        words = result.split()
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3])
            yield chunk + " "
            await asyncio.sleep(0.01)


# ============================================================================
# Prime Model Manager
# ============================================================================

class PrimeModelManager:
    """
    The central orchestrator for JARVIS-Prime Tier-0 Brain.

    Coordinates all components:
    - ModelRegistry: Version tracking and rollback
    - HotSwapManager: Zero-downtime model updates
    - HybridRouter: Tier 0/1 classification
    - TelemetryHook: Logging with PII anonymization

    Usage:
        # Initialize
        manager = PrimeModelManager(
            models_dir="./models",
            telemetry_dir="./telemetry",
        )
        await manager.start()

        # Process request (OpenAI-compatible)
        response = await manager.complete(CompletionRequest(
            messages=[ChatMessage(role="user", content="Hello!")],
        ))

        # Hot-swap to new model
        await manager.hot_swap("v1.1-weekly-2025-05-12")

        # Shutdown
        await manager.stop()
    """

    def __init__(
        self,
        models_dir: str | Path = "./models",
        telemetry_dir: str | Path = "./telemetry",
        reactor_core_watch_dir: Optional[str | Path] = None,
        cloud_executor: Optional[CloudAPIExecutor] = None,
        executor_class: type = LlamaModelExecutor,
        default_lineage: str = "jarvis-prime",
    ):
        self.models_dir = Path(models_dir)
        self.telemetry_dir = Path(telemetry_dir)
        self.reactor_core_watch_dir = Path(reactor_core_watch_dir) if reactor_core_watch_dir else None
        self.cloud_executor = cloud_executor
        self.executor_class = executor_class
        self.default_lineage = default_lineage

        # Components (lazy initialized)
        self._registry = None
        self._hot_swap = None
        self._router = None
        self._telemetry = None

        # State
        self._running = False
        self._current_version = None

        # Statistics
        self._request_count = 0
        self._tier_0_count = 0
        self._tier_1_count = 0
        self._error_count = 0

        logger.info("PrimeModelManager initialized")

    async def start(self, initial_model_path: Optional[Path] = None) -> None:
        """
        Start the model manager and all components.

        Args:
            initial_model_path: Path to initial model to load
        """
        from jarvis_prime.core.model_registry import ModelRegistry
        from jarvis_prime.core.hot_swap_manager import HotSwapManager, ModelLoader
        from jarvis_prime.core.hybrid_router import HybridRouter
        from jarvis_prime.core.telemetry_hook import TelemetryHook

        # Initialize components
        logger.info("Starting PrimeModelManager components...")

        # Registry
        self._registry = ModelRegistry(
            models_dir=self.models_dir,
            reactor_core_watch_dir=self.reactor_core_watch_dir,
        )
        await self._registry.start()

        # Create executor wrapper for HotSwapManager
        executor = self.executor_class()

        class ExecutorLoader(ModelLoader):
            def __init__(self, exec_instance):
                self._executor = exec_instance

            async def load(self, model_path: Path, **kwargs):
                await self._executor.load(model_path, **kwargs)
                return self._executor

            async def unload(self, model):
                await model.unload()

            async def validate(self, model) -> bool:
                return await model.validate()

        # Hot-swap manager
        self._hot_swap = HotSwapManager(
            loader=ExecutorLoader(executor),
            max_memory_gb=12.0,  # M1 Mac optimization
        )

        # Router
        self._router = HybridRouter(
            history_file=self.telemetry_dir / "routing_history.json",
        )

        # Telemetry
        self._telemetry = TelemetryHook(
            output_dir=self.telemetry_dir,
        )
        await self._telemetry.start()

        # Load initial model if provided
        if initial_model_path:
            await self._hot_swap.load_initial(
                model_path=initial_model_path,
                version_id="v1.0-initial",
            )
            self._current_version = "v1.0-initial"

        # Register callbacks
        self._registry.on_new_version(self._on_new_version)
        self._registry.on_activation(self._on_activation)
        self._hot_swap.on_swap_complete(self._on_swap_complete)

        self._running = True
        logger.info("PrimeModelManager started successfully")

    async def stop(self) -> None:
        """Stop the model manager and all components"""
        logger.info("Stopping PrimeModelManager...")

        if self._telemetry:
            await self._telemetry.stop()

        if self._registry:
            await self._registry.stop()

        self._running = False
        logger.info("PrimeModelManager stopped")

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Process a completion request (OpenAI-compatible).

        Routes between Tier 0 (local) and Tier 1 (cloud) based on complexity.
        """
        start_time = time.time()
        self._request_count += 1

        try:
            # Build prompt from messages
            if request.messages:
                prompt = self._format_messages(request.messages)
            elif request.prompt:
                prompt = request.prompt
            else:
                raise ValueError("No prompt or messages provided")

            # Route the request
            from jarvis_prime.core.hybrid_router import TierClassification

            if request.force_tier == "tier_0":
                forced_tier = TierClassification.TIER_0
            elif request.force_tier == "tier_1":
                forced_tier = TierClassification.TIER_1
            else:
                forced_tier = None

            decision = await self._router.classify(
                prompt=prompt,
                context=request.metadata,
                force_tier=forced_tier,
            )

            # Execute based on tier
            if decision.tier in (TierClassification.TIER_0, TierClassification.TIER_0_PREFERRED):
                completion = await self._execute_tier_0(prompt, request)
                tier_used = "tier_0"
                self._tier_0_count += 1
            else:
                if self.cloud_executor:
                    completion = await self._execute_tier_1(request)
                    tier_used = "tier_1"
                    self._tier_1_count += 1
                else:
                    # Fallback to Tier 0 if no cloud executor
                    completion = await self._execute_tier_0(prompt, request)
                    tier_used = "tier_0_fallback"
                    self._tier_0_count += 1

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Estimate tokens
            prompt_tokens = len(prompt.split()) + len(prompt) // 4
            completion_tokens = len(completion.split()) + len(completion) // 4

            # Log telemetry
            await self._telemetry.log(
                prompt=prompt,
                completion=completion,
                model_version=self._current_version or "unknown",
                tier=tier_used,
                task_type=decision.task_type.value,
                complexity_score=decision.complexity_score,
                latency_ms=latency_ms,
                session_id=request.session_id,
                metadata=request.metadata,
            )

            # Record outcome for router learning
            self._router.record_outcome(
                prompt_hash=decision.prompt_hash,
                success=True,
                tier_used=decision.tier,
            )

            # Build response
            response = CompletionResponse(
                id=f"chatcmpl-{decision.prompt_hash}",
                model=request.model,
                choices=[
                    CompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=completion),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
                tier_used=tier_used,
                model_version=self._current_version or "unknown",
                complexity_score=decision.complexity_score,
                latency_ms=latency_ms,
            )

            return response

        except Exception as e:
            self._error_count += 1
            logger.error(f"Completion error: {e}")

            # Log failure
            if self._telemetry:
                await self._telemetry.log(
                    prompt=prompt if 'prompt' in dir() else "",
                    completion="",
                    model_version=self._current_version or "unknown",
                    tier="error",
                    success=False,
                    error_message=str(e),
                )

            raise

    async def _execute_tier_0(self, prompt: str, request: CompletionRequest) -> str:
        """Execute on local model (Tier 0)"""
        async with self._hot_swap.acquire() as executor:
            return await executor.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

    async def _execute_tier_1(self, request: CompletionRequest) -> str:
        """Execute on cloud API (Tier 1)"""
        if not self.cloud_executor:
            raise RuntimeError("No cloud executor configured")

        return await self.cloud_executor.complete(
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into prompt string"""
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"Assistant: {msg.content}")
        formatted.append("Assistant:")
        return "\n".join(formatted)

    async def hot_swap(
        self,
        version_id: str,
        lineage: Optional[str] = None,
    ) -> bool:
        """
        Hot-swap to a specific model version.

        Args:
            version_id: Version to swap to
            lineage: Lineage name (defaults to default_lineage)

        Returns:
            True if swap successful
        """
        lineage = lineage or self.default_lineage

        # Get version from registry
        version = self._registry.get_active_version(lineage)
        if not version:
            # Try to find and activate the version
            lineage_obj = self._registry.lineages.get(lineage)
            if lineage_obj and version_id in lineage_obj.versions:
                await self._registry.activate_version(lineage, version_id)
                version = lineage_obj.versions[version_id]
            else:
                raise ValueError(f"Version {version_id} not found in lineage {lineage}")

        # Perform hot-swap
        result = await self._hot_swap.hot_swap(
            model_path=version.model_path,
            version_id=version_id,
        )

        if result.success:
            self._current_version = version_id

        return result.success

    async def rollback(self, lineage: Optional[str] = None) -> bool:
        """
        Rollback to the previous model version.

        Returns:
            True if rollback successful
        """
        lineage = lineage or self.default_lineage

        # Rollback in registry
        await self._registry.rollback(lineage)

        # Get new active version
        version = self._registry.get_active_version(lineage)
        if version:
            # Hot-swap to rollback version
            result = await self._hot_swap.hot_swap(
                model_path=version.model_path,
                version_id=version.version_id,
            )
            if result.success:
                self._current_version = version.version_id
            return result.success

        return False

    def _on_new_version(self, version) -> None:
        """Callback when new version is registered"""
        logger.info(f"New model version registered: {version.version_id}")

    def _on_activation(self, version) -> None:
        """Callback when version is activated"""
        logger.info(f"Model version activated: {version.version_id}")

    def _on_swap_complete(self, result) -> None:
        """Callback when hot-swap completes"""
        if result.success:
            logger.info(
                f"Hot-swap complete: {result.old_version} → {result.new_version} "
                f"({result.duration_seconds:.2f}s)"
            )
        else:
            logger.error(f"Hot-swap failed: {result.error_message}")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive manager status"""
        return {
            "running": self._running,
            "current_version": self._current_version,
            "statistics": {
                "total_requests": self._request_count,
                "tier_0_count": self._tier_0_count,
                "tier_1_count": self._tier_1_count,
                "error_count": self._error_count,
                "tier_0_ratio": self._tier_0_count / max(self._request_count, 1),
            },
            "registry": self._registry.get_status() if self._registry else None,
            "hot_swap": self._hot_swap.get_status() if self._hot_swap else None,
            "router": self._router.get_statistics() if self._router else None,
            "telemetry": self._telemetry.get_statistics() if self._telemetry else None,
        }


# ============================================================================
# OpenAI-Compatible API Server
# ============================================================================

def create_api_app(manager: PrimeModelManager):
    """
    Create a FastAPI app with OpenAI-compatible endpoints.

    Usage:
        manager = PrimeModelManager(...)
        app = create_api_app(manager)

        # Run with uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    """
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
        from typing import List, Optional
        import json
    except ImportError:
        raise ImportError("FastAPI required: pip install fastapi uvicorn")

    app = FastAPI(
        title="JARVIS-Prime API",
        description="OpenAI-compatible API for JARVIS-Prime Tier-0 Brain",
        version="1.0.0",
    )

    # WebSocket event subscribers for Neural Mesh integration (v10.3)
    _event_subscribers: List[WebSocket] = []
    _event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    class MessageModel(BaseModel):
        role: str
        content: str
        name: Optional[str] = None

    class CompletionRequestModel(BaseModel):
        model: str = "jarvis-prime"
        messages: List[MessageModel] = []
        prompt: Optional[str] = None
        max_tokens: int = 2048
        temperature: float = 0.7
        top_p: float = 0.9
        stream: bool = False
        stop: Optional[List[str]] = None
        user: Optional[str] = None
        # Extensions
        force_tier: Optional[str] = None
        session_id: Optional[str] = None

    @app.post("/v1/chat/completions")
    async def chat_completions(request: CompletionRequestModel):
        """OpenAI-compatible chat completions endpoint"""
        try:
            completion_request = CompletionRequest(
                model=request.model,
                messages=[
                    ChatMessage(role=m.role, content=m.content, name=m.name)
                    for m in request.messages
                ],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream,
                stop=request.stop,
                user=request.user,
                force_tier=request.force_tier,
                session_id=request.session_id,
            )

            response = await manager.complete(completion_request)
            return response.to_dict()

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/completions")
    async def completions(request: CompletionRequestModel):
        """Legacy completions endpoint"""
        try:
            completion_request = CompletionRequest(
                model=request.model,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                force_tier=request.force_tier,
            )

            response = await manager.complete(completion_request)

            # Convert to legacy format
            return {
                "id": response.id,
                "object": "text_completion",
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "text": response.choices[0].message.content if response.choices else "",
                        "index": 0,
                        "finish_reason": "stop",
                    }
                ],
                "usage": response.to_dict()["usage"],
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/models")
    async def list_models():
        """List available models"""
        return {
            "object": "list",
            "data": [
                {
                    "id": "jarvis-prime",
                    "object": "model",
                    "owned_by": "jarvis",
                    "permission": [],
                }
            ],
        }

    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "healthy", **manager.get_status()}

    # ========================================================================
    # WebSocket Event Stream for Neural Mesh Integration (v10.3)
    # ========================================================================

    @app.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket):
        """
        WebSocket endpoint for real-time event streaming to Neural Mesh.

        Events include:
        - model_loaded: When a model is loaded
        - model_swapped: When hot-swap completes
        - inference_complete: After each inference (with telemetry)
        - error: On errors
        - heartbeat: Periodic keep-alive
        """
        await websocket.accept()
        _event_subscribers.append(websocket)
        logger.info(f"[WebSocket] Client connected ({len(_event_subscribers)} active)")

        try:
            # Send initial connection event
            await websocket.send_json({
                "event_type": "connected",
                "data": {
                    "status": "healthy",
                    **manager.get_status(),
                },
                "timestamp": datetime.now().isoformat(),
            })

            # Keep connection alive and send events
            while True:
                try:
                    # Wait for events from queue with timeout for heartbeat
                    try:
                        event = await asyncio.wait_for(
                            _event_queue.get(),
                            timeout=30.0  # Heartbeat every 30s
                        )
                        await websocket.send_json(event)
                    except asyncio.TimeoutError:
                        # Send heartbeat
                        await websocket.send_json({
                            "event_type": "heartbeat",
                            "data": {"status": "alive"},
                            "timestamp": datetime.now().isoformat(),
                        })

                    # Also check for incoming messages (like ping)
                    try:
                        data = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=0.1
                        )
                        if data == "ping":
                            await websocket.send_text("pong")
                    except asyncio.TimeoutError:
                        pass

                except WebSocketDisconnect:
                    break

        except Exception as e:
            logger.warning(f"[WebSocket] Connection error: {e}")
        finally:
            if websocket in _event_subscribers:
                _event_subscribers.remove(websocket)
            logger.info(f"[WebSocket] Client disconnected ({len(_event_subscribers)} active)")

    async def broadcast_event(event_type: str, data: dict):
        """Broadcast an event to all connected WebSocket clients."""
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

        # Add to queue for connected clients
        if not _event_queue.full():
            await _event_queue.put(event)

        # Direct broadcast to all subscribers
        disconnected = []
        for ws in _event_subscribers:
            try:
                await ws.send_json(event)
            except Exception:
                disconnected.append(ws)

        # Cleanup disconnected
        for ws in disconnected:
            if ws in _event_subscribers:
                _event_subscribers.remove(ws)

    # Attach broadcast function to manager for event emission
    manager._broadcast_event = broadcast_event

    @app.post("/admin/hot-swap")
    async def admin_hot_swap(version_id: str, lineage: Optional[str] = None):
        """Admin endpoint to trigger hot-swap"""
        try:
            success = await manager.hot_swap(version_id, lineage)
            return {"success": success, "version": version_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/admin/rollback")
    async def admin_rollback(lineage: Optional[str] = None):
        """Admin endpoint to trigger rollback"""
        try:
            success = await manager.rollback(lineage)
            return {"success": success}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # Model Swap Endpoint - Phase 2: Hot Swap for JARVIS Prime
    # ========================================================================

    class ModelSwapRequest(BaseModel):
        """Request model for hot-swapping to a new model file."""
        model_path: str  # Path to .gguf model file
        version_id: Optional[str] = None  # Optional version identifier
        force: bool = False  # Force swap even if validation fails
        validate_before_swap: bool = True  # Run validation test before swapping

    class ModelSwapResponse(BaseModel):
        """Response for model swap operation."""
        success: bool
        state: str
        old_version: Optional[str] = None
        new_version: Optional[str] = None
        duration_seconds: float = 0.0
        memory_freed_mb: float = 0.0
        error_message: Optional[str] = None

    @app.post("/model/swap", response_model=ModelSwapResponse)
    async def swap_model(request: ModelSwapRequest):
        """
        Hot-swap to a new model file with zero downtime.

        This endpoint is called by JARVIS after Reactor-Core training completes.

        Steps:
        1. Load new model into standby slot (background)
        2. Validate new model with test generation
        3. Drain in-flight requests from primary
        4. Atomic swap primary ↔ standby
        5. Unload old model and free memory

        Args:
            model_path: Absolute path to the .gguf model file
            version_id: Optional version identifier (auto-generated if not provided)
            force: Force swap even if validation fails
            validate_before_swap: Run validation test before swapping

        Returns:
            ModelSwapResponse with operation details
        """
        import os
        import time
        from pathlib import Path as PathLib
        from datetime import datetime

        model_path = PathLib(request.model_path)

        # Validate path exists
        if not model_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {request.model_path}"
            )

        # Validate file extension
        if model_path.suffix.lower() not in ('.gguf', '.bin', '.pt', '.safetensors'):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model format: {model_path.suffix}. "
                       f"Supported: .gguf, .bin, .pt, .safetensors"
            )

        # Generate version_id if not provided
        version_id = request.version_id
        if not version_id:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            version_id = f"v{timestamp}-{model_path.stem[:20]}"

        try:
            # Check if hot-swap manager is available
            if not manager._hot_swap:
                raise HTTPException(
                    status_code=503,
                    detail="Hot-swap manager not initialized"
                )

            # Perform the hot-swap
            result = await manager._hot_swap.hot_swap(
                model_path=model_path,
                version_id=version_id,
                force=request.force,
            )

            # Update manager's current version on success
            if result.success:
                manager._current_version = version_id

            return ModelSwapResponse(
                success=result.success,
                state=result.state.value,
                old_version=result.old_version,
                new_version=result.new_version,
                duration_seconds=result.duration_seconds,
                memory_freed_mb=result.memory_freed_mb,
                error_message=result.error_message,
            )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Hot-swap failed: {str(e)}"
            )

    @app.get("/model/status")
    async def get_model_status():
        """Get current model and hot-swap status."""
        status = {
            "current_version": manager._current_version,
            "running": manager._running,
        }

        if manager._hot_swap:
            status["hot_swap"] = manager._hot_swap.get_status()

        if manager._registry:
            status["registry"] = manager._registry.get_status()

        return status

    @app.post("/model/preload")
    async def preload_model(model_path: str, version_id: Optional[str] = None):
        """
        Preload a model into standby slot for faster swapping.

        Useful when you know a swap is coming but want to minimize
        downtime during the actual swap.
        """
        from pathlib import Path as PathLib
        from datetime import datetime

        path = PathLib(model_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")

        if not version_id:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            version_id = f"v{timestamp}-{path.stem[:20]}"

        try:
            success = await manager._hot_swap.preload_standby(
                model_path=path,
                version_id=version_id,
            )
            return {"success": success, "version_id": version_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/model/swap-from-standby")
    async def swap_from_standby():
        """
        Swap to the preloaded standby model.

        Faster than /model/swap because model is already loaded.
        Call /model/preload first.
        """
        try:
            result = await manager._hot_swap.swap_from_standby()
            if result.success:
                manager._current_version = result.new_version

            return ModelSwapResponse(
                success=result.success,
                state=result.state.value,
                old_version=result.old_version,
                new_version=result.new_version,
                duration_seconds=result.duration_seconds,
                memory_freed_mb=result.memory_freed_mb,
                error_message=result.error_message,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
