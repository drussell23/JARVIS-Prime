"""
Hot-Swap Manager - Zero-Downtime Model Reload
==============================================

Enables seamless model swapping without dropping requests.

Architecture:
1. Background loading: New model loaded into RAM in separate process/thread
2. Traffic draining: In-flight requests complete on old model
3. Atomic switch: Traffic routes to new model instantly
4. Cleanup: Old model unloaded and memory freed

Features:
- Async background loading
- Request counting for safe drain
- Atomic pointer swap
- Rollback on failure
- Memory-aware loading
- M1 MPS optimization
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Generic, Optional, TypeVar

import psutil

logger = logging.getLogger(__name__)

T = TypeVar("T")  # Model type


class SwapState(Enum):
    """State of a hot-swap operation"""
    IDLE = "idle"                    # No swap in progress
    LOADING_BACKGROUND = "loading"   # Loading new model in background
    VALIDATING = "validating"        # Validating new model
    DRAINING = "draining"            # Waiting for in-flight requests
    SWAPPING = "swapping"            # Atomic pointer swap
    CLEANUP = "cleanup"              # Unloading old model
    COMPLETED = "completed"          # Swap finished
    FAILED = "failed"                # Swap failed, rolled back
    ROLLBACK = "rollback"            # Actively rolling back


@dataclass
class SwapResult:
    """Result of a hot-swap operation"""
    success: bool
    state: SwapState
    old_version: Optional[str] = None
    new_version: Optional[str] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    requests_drained: int = 0
    memory_freed_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "state": self.state.value,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "duration_seconds": round(self.duration_seconds, 3),
            "error_message": self.error_message,
            "requests_drained": self.requests_drained,
            "memory_freed_mb": round(self.memory_freed_mb, 2),
        }


@dataclass
class ModelSlot(Generic[T]):
    """
    A slot holding a loaded model.

    Tracks reference counts for safe unloading.
    """
    model: Optional[T] = None
    version_id: Optional[str] = None
    loaded_at: Optional[datetime] = None
    active_requests: int = 0
    total_requests: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def acquire(self) -> Optional[T]:
        """Acquire the model for a request, incrementing reference count"""
        async with self._lock:
            if self.model is not None:
                self.active_requests += 1
                self.total_requests += 1
                return self.model
            return None

    async def release(self) -> None:
        """Release the model after request completion"""
        async with self._lock:
            if self.active_requests > 0:
                self.active_requests -= 1

    async def wait_for_drain(self, timeout: float = 30.0) -> bool:
        """Wait for all active requests to complete"""
        start = time.time()
        while time.time() - start < timeout:
            async with self._lock:
                if self.active_requests == 0:
                    return True
            await asyncio.sleep(0.1)
        return False


class ModelLoader(ABC, Generic[T]):
    """
    Abstract base for model loading strategies.

    Implement for specific model types (Llama, embedding, etc.)
    """

    @abstractmethod
    async def load(self, model_path: Path, **kwargs) -> T:
        """Load a model from path"""
        ...

    @abstractmethod
    async def unload(self, model: T) -> None:
        """Unload a model, freeing memory"""
        ...

    @abstractmethod
    async def validate(self, model: T) -> bool:
        """Validate a loaded model works correctly"""
        ...

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024


class HotSwapManager(Generic[T]):
    """
    Manages zero-downtime model swapping.

    Uses a double-buffer pattern:
    - Primary slot: Currently serving requests
    - Standby slot: Background loading and ready for swap

    Usage:
        loader = MyModelLoader()
        manager = HotSwapManager(loader)

        # Initial load
        await manager.load_initial(Path("./models/v1.0"))

        # Get model for inference (with reference counting)
        async with manager.acquire() as model:
            result = model.generate(prompt)

        # Hot-swap to new version
        result = await manager.hot_swap(Path("./models/v1.1"), "v1.1")
    """

    def __init__(
        self,
        loader: ModelLoader[T],
        drain_timeout: float = 30.0,
        max_memory_gb: float = 12.0,  # Max memory before forcing unload
        enable_memory_optimization: bool = True,
    ):
        self.loader = loader
        self.drain_timeout = drain_timeout
        self.max_memory_gb = max_memory_gb
        self.enable_memory_optimization = enable_memory_optimization

        # Double buffer
        self._primary: ModelSlot[T] = ModelSlot()
        self._standby: ModelSlot[T] = ModelSlot()

        # State
        self._swap_state = SwapState.IDLE
        self._swap_lock = asyncio.Lock()
        self._current_swap_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_swap_complete_callbacks: list[Callable[[SwapResult], None]] = []

        logger.info("HotSwapManager initialized")

    @property
    def swap_state(self) -> SwapState:
        return self._swap_state

    @property
    def current_version(self) -> Optional[str]:
        return self._primary.version_id

    async def load_initial(
        self,
        model_path: Path,
        version_id: str = "v1.0-initial",
        **load_kwargs,
    ) -> bool:
        """
        Load the initial model.

        Should be called once at startup.
        """
        async with self._swap_lock:
            if self._primary.model is not None:
                logger.warning("Primary slot already has a model")
                return False

            try:
                self._swap_state = SwapState.LOADING_BACKGROUND

                # Load model
                model = await self.loader.load(model_path, **load_kwargs)

                # Validate
                self._swap_state = SwapState.VALIDATING
                if not await self.loader.validate(model):
                    await self.loader.unload(model)
                    self._swap_state = SwapState.FAILED
                    return False

                # Set as primary
                self._primary.model = model
                self._primary.version_id = version_id
                self._primary.loaded_at = datetime.now()
                self._swap_state = SwapState.IDLE

                logger.info(f"Initial model loaded: {version_id}")
                return True

            except Exception as e:
                logger.error(f"Initial load failed: {e}")
                self._swap_state = SwapState.FAILED
                return False

    async def hot_swap(
        self,
        model_path: Path,
        version_id: str,
        force: bool = False,
        **load_kwargs,
    ) -> SwapResult:
        """
        Hot-swap to a new model version.

        Steps:
        1. Load new model into standby slot (background)
        2. Validate new model
        3. Drain in-flight requests from primary
        4. Atomic swap primary ↔ standby
        5. Unload old model

        Args:
            model_path: Path to new model
            version_id: Version identifier
            force: Force swap even if validation fails
            **load_kwargs: Additional loader arguments

        Returns:
            SwapResult with operation details
        """
        start_time = time.time()

        async with self._swap_lock:
            if self._swap_state not in (SwapState.IDLE, SwapState.COMPLETED, SwapState.FAILED):
                return SwapResult(
                    success=False,
                    state=self._swap_state,
                    error_message=f"Swap already in progress: {self._swap_state.value}",
                )

            old_version = self._primary.version_id
            result = SwapResult(
                success=False,
                state=SwapState.IDLE,
                old_version=old_version,
                new_version=version_id,
            )

            try:
                # Step 1: Background load into standby
                self._swap_state = SwapState.LOADING_BACKGROUND
                logger.info(f"Loading {version_id} into standby slot...")

                # Check memory before loading
                if self.enable_memory_optimization:
                    await self._ensure_memory_available()

                new_model = await self.loader.load(model_path, **load_kwargs)
                self._standby.model = new_model
                self._standby.version_id = version_id
                self._standby.loaded_at = datetime.now()

                # Step 2: Validate
                self._swap_state = SwapState.VALIDATING
                logger.info(f"Validating {version_id}...")

                if not await self.loader.validate(new_model):
                    if not force:
                        await self.loader.unload(new_model)
                        self._standby.model = None
                        self._swap_state = SwapState.FAILED
                        result.state = SwapState.FAILED
                        result.error_message = "Validation failed"
                        return result
                    logger.warning("Validation failed but force=True, continuing...")

                # Step 3: Drain primary
                self._swap_state = SwapState.DRAINING
                logger.info(f"Draining {self._primary.active_requests} active requests...")

                drained = await self._primary.wait_for_drain(self.drain_timeout)
                result.requests_drained = self._primary.total_requests

                if not drained:
                    logger.warning(f"Drain timeout, {self._primary.active_requests} requests still active")
                    # Continue anyway - requests will complete on old model

                # Step 4: Atomic swap
                self._swap_state = SwapState.SWAPPING
                logger.info("Performing atomic swap...")

                # Swap pointers
                self._primary, self._standby = self._standby, self._primary

                # Step 5: Cleanup old model
                self._swap_state = SwapState.CLEANUP
                memory_before = self.loader.get_memory_usage_mb()

                if self._standby.model is not None:
                    logger.info(f"Unloading old model {old_version}...")
                    await self.loader.unload(self._standby.model)
                    self._standby.model = None
                    self._standby.version_id = None

                    # Force garbage collection
                    gc.collect()

                memory_after = self.loader.get_memory_usage_mb()
                result.memory_freed_mb = max(0, memory_before - memory_after)

                # Done
                self._swap_state = SwapState.COMPLETED
                result.success = True
                result.state = SwapState.COMPLETED
                result.duration_seconds = time.time() - start_time

                logger.info(
                    f"Hot-swap complete: {old_version} → {version_id} "
                    f"({result.duration_seconds:.2f}s, freed {result.memory_freed_mb:.0f}MB)"
                )

                # Notify callbacks
                for callback in self._on_swap_complete_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Swap callback error: {e}")

                return result

            except Exception as e:
                logger.error(f"Hot-swap failed: {e}")

                # Rollback: ensure primary is still valid
                self._swap_state = SwapState.ROLLBACK

                if self._standby.model is not None:
                    try:
                        await self.loader.unload(self._standby.model)
                    except Exception:
                        pass
                    self._standby.model = None

                self._swap_state = SwapState.FAILED
                result.state = SwapState.FAILED
                result.error_message = str(e)
                result.duration_seconds = time.time() - start_time

                return result

    async def _ensure_memory_available(self) -> None:
        """Ensure there's enough memory for loading a new model"""
        process = psutil.Process(os.getpid())
        current_gb = process.memory_info().rss / 1024 / 1024 / 1024

        if current_gb > self.max_memory_gb * 0.8:
            logger.warning(f"Memory pressure: {current_gb:.1f}GB, running GC...")
            gc.collect()

            # If on macOS with MPS, try to clear MPS cache
            try:
                import torch
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except ImportError:
                pass

    class _AcquireContext:
        """Context manager for acquiring model with reference counting"""

        def __init__(self, manager: "HotSwapManager[T]"):
            self._manager = manager
            self._model: Optional[T] = None

        async def __aenter__(self) -> T:
            self._model = await self._manager._primary.acquire()
            if self._model is None:
                raise RuntimeError("No model loaded")
            return self._model

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._manager._primary.release()

    def acquire(self) -> _AcquireContext:
        """
        Acquire the current model for inference.

        Use as async context manager to ensure proper reference counting:

            async with manager.acquire() as model:
                result = model.generate(prompt)

        Returns:
            Context manager yielding the current model
        """
        return self._AcquireContext(self)

    async def get_model(self) -> Optional[T]:
        """
        Get the current model directly (without reference counting).

        WARNING: Prefer using acquire() context manager for proper
        reference counting during hot-swaps.
        """
        return self._primary.model

    async def preload_standby(
        self,
        model_path: Path,
        version_id: str,
        **load_kwargs,
    ) -> bool:
        """
        Preload a model into standby slot for faster swapping.

        Useful when you know a swap is coming but want to minimize
        downtime during the actual swap.
        """
        if self._standby.model is not None:
            logger.warning("Standby slot already has a model")
            return False

        try:
            self._swap_state = SwapState.LOADING_BACKGROUND

            model = await self.loader.load(model_path, **load_kwargs)
            self._standby.model = model
            self._standby.version_id = version_id
            self._standby.loaded_at = datetime.now()

            self._swap_state = SwapState.IDLE
            logger.info(f"Preloaded {version_id} into standby")
            return True

        except Exception as e:
            logger.error(f"Preload failed: {e}")
            self._swap_state = SwapState.IDLE
            return False

    async def swap_from_standby(self) -> SwapResult:
        """
        Swap to the preloaded standby model.

        Faster than hot_swap() because model is already loaded.
        """
        if self._standby.model is None:
            return SwapResult(
                success=False,
                state=SwapState.FAILED,
                error_message="No model in standby slot",
            )

        start_time = time.time()

        async with self._swap_lock:
            old_version = self._primary.version_id
            new_version = self._standby.version_id

            # Drain
            self._swap_state = SwapState.DRAINING
            await self._primary.wait_for_drain(self.drain_timeout)

            # Swap
            self._swap_state = SwapState.SWAPPING
            self._primary, self._standby = self._standby, self._primary

            # Cleanup
            self._swap_state = SwapState.CLEANUP
            memory_before = self.loader.get_memory_usage_mb()

            if self._standby.model is not None:
                await self.loader.unload(self._standby.model)
                self._standby.model = None
                gc.collect()

            memory_after = self.loader.get_memory_usage_mb()

            self._swap_state = SwapState.COMPLETED

            return SwapResult(
                success=True,
                state=SwapState.COMPLETED,
                old_version=old_version,
                new_version=new_version,
                duration_seconds=time.time() - start_time,
                memory_freed_mb=max(0, memory_before - memory_after),
            )

    def get_status(self) -> Dict[str, Any]:
        """Get current hot-swap manager status"""
        return {
            "state": self._swap_state.value,
            "primary": {
                "version": self._primary.version_id,
                "loaded_at": self._primary.loaded_at.isoformat() if self._primary.loaded_at else None,
                "active_requests": self._primary.active_requests,
                "total_requests": self._primary.total_requests,
            },
            "standby": {
                "version": self._standby.version_id,
                "loaded": self._standby.model is not None,
            },
            "memory_usage_mb": self.loader.get_memory_usage_mb(),
            "max_memory_gb": self.max_memory_gb,
        }

    def on_swap_complete(self, callback: Callable[[SwapResult], None]) -> None:
        """Register callback for swap completion events"""
        self._on_swap_complete_callbacks.append(callback)


# ============================================================================
# Docker-Aware Model Loader
# ============================================================================

class DockerAwareModelLoader(ModelLoader[T]):
    """
    Model loader with Docker-specific optimizations.

    Handles:
    - Container memory limits awareness
    - Graceful degradation when near memory limits
    - Health check integration
    - Volume-mounted model path handling
    """

    def __init__(
        self,
        inner_loader: ModelLoader[T],
        container_memory_limit_gb: float = 10.0,
        memory_safety_margin: float = 0.8,
    ):
        self.inner_loader = inner_loader
        self.container_memory_limit_gb = container_memory_limit_gb
        self.memory_safety_margin = memory_safety_margin

    async def load(self, model_path: Path, **kwargs) -> T:
        """Load model with Docker memory awareness."""
        # Check memory before loading
        available_mb = self._get_available_memory_mb()
        model_size_mb = self._estimate_model_size_mb(model_path)

        logger.info(
            f"Loading model: estimated {model_size_mb:.0f}MB, "
            f"available {available_mb:.0f}MB"
        )

        if model_size_mb > available_mb * self.memory_safety_margin:
            logger.warning(
                f"Model may exceed memory: {model_size_mb:.0f}MB > "
                f"{available_mb * self.memory_safety_margin:.0f}MB safe limit"
            )
            # Force garbage collection before loading
            gc.collect()

        return await self.inner_loader.load(model_path, **kwargs)

    async def unload(self, model: T) -> None:
        """Unload model with aggressive cleanup."""
        await self.inner_loader.unload(model)

        # Aggressive cleanup for Docker
        gc.collect()

        # Try to clear GPU caches
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    async def validate(self, model: T) -> bool:
        """Validate model."""
        return await self.inner_loader.validate(model)

    def _get_available_memory_mb(self) -> float:
        """Get available memory in MB, considering container limits."""
        # Check cgroup memory limit (Docker)
        cgroup_limit_path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        container_limit_mb = self.container_memory_limit_gb * 1024

        if cgroup_limit_path.exists():
            try:
                limit_bytes = int(cgroup_limit_path.read_text().strip())
                container_limit_mb = min(
                    container_limit_mb,
                    limit_bytes / (1024 * 1024),
                )
            except Exception:
                pass

        # Get current usage
        process = psutil.Process(os.getpid())
        used_mb = process.memory_info().rss / (1024 * 1024)

        return container_limit_mb - used_mb

    def _estimate_model_size_mb(self, model_path: Path) -> float:
        """Estimate memory needed for model (typically ~1.5x file size)."""
        if model_path.exists():
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            # GGUF models are memory-mapped, so typically need ~1.2x file size
            return file_size_mb * 1.2
        return 0.0


# ============================================================================
# Reactor-Core Integration Helper
# ============================================================================

class ReactorCoreHotSwapBridge:
    """
    Bridge between ReactorCoreWatcher and HotSwapManager.

    Automatically deploys models from reactor-core to JARVIS-Prime
    with hot-swap capability.

    Usage:
        from jarvis_prime.docker.reactor_core_watcher import ReactorCoreWatcher

        bridge = ReactorCoreHotSwapBridge(
            hot_swap_manager=hot_swap_manager,
            models_dir=Path("./models"),
        )

        watcher = ReactorCoreWatcher(
            watch_dir=Path("./reactor-core-output"),
            models_dir=Path("./models"),
            hot_swap_callback=bridge.deploy_model,
            on_deploy_success=bridge.on_success,
            on_deploy_failure=bridge.on_failure,
        )
    """

    def __init__(
        self,
        hot_swap_manager: HotSwapManager,
        models_dir: Path,
        auto_rollback_on_failure: bool = True,
        validation_test_prompt: str = "Hello, this is a test.",
    ):
        self.hot_swap_manager = hot_swap_manager
        self.models_dir = models_dir
        self.auto_rollback_on_failure = auto_rollback_on_failure
        self.validation_test_prompt = validation_test_prompt

        # Track deployments
        self._deployment_history: list[Dict[str, Any]] = []
        self._last_good_version: Optional[str] = None

    async def deploy_model(self, model_path: Path, version_id: str) -> None:
        """
        Deploy a new model via hot-swap.

        Called by ReactorCoreWatcher when a new model is detected.
        """
        logger.info(f"Deploying reactor-core model: {version_id}")

        # Remember last known good version for rollback
        self._last_good_version = self.hot_swap_manager.current_version

        # Perform hot-swap
        result = await self.hot_swap_manager.hot_swap(
            model_path=model_path,
            version_id=version_id,
        )

        # Record deployment
        self._deployment_history.append({
            "version_id": version_id,
            "model_path": str(model_path),
            "result": result.to_dict(),
            "timestamp": datetime.now().isoformat(),
        })

        if not result.success and self.auto_rollback_on_failure:
            logger.error(f"Deployment failed, rolling back to {self._last_good_version}")
            await self._rollback()

    async def _rollback(self) -> bool:
        """Rollback to last known good version."""
        if not self._last_good_version:
            logger.error("No previous version to rollback to")
            return False

        # Find the model file for previous version
        for entry in reversed(self._deployment_history[:-1]):
            if entry["version_id"] == self._last_good_version:
                model_path = Path(entry["model_path"])
                if model_path.exists():
                    result = await self.hot_swap_manager.hot_swap(
                        model_path=model_path,
                        version_id=self._last_good_version,
                        force=True,
                    )
                    return result.success

        logger.error(f"Could not find model file for {self._last_good_version}")
        return False

    def on_success(self, result) -> None:
        """Called when deployment succeeds."""
        self._last_good_version = result.version
        logger.info(f"Reactor-core deployment success: {result.model_id} v{result.version}")

    def on_failure(self, result) -> None:
        """Called when deployment fails."""
        logger.error(
            f"Reactor-core deployment failed: {result.model_id} - {result.error_message}"
        )

    def get_deployment_history(self) -> list[Dict[str, Any]]:
        """Get deployment history."""
        return self._deployment_history.copy()
