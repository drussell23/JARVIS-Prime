"""
PrimeModel - Advanced Unified Interface for JARVIS PRIME Models
================================================================

v91.0 - Production-Grade Model Management System

This module provides the core model interface for JARVIS Prime with:
- Async model loading with dynamic hot-swap capability
- Intelligent model caching with LRU eviction
- Automatic complexity-based model selection
- Seamless fallback to Claude API
- Integration with Continuous Learning pipeline
- Batched async inference for throughput
- Comprehensive health monitoring and metrics

ARCHITECTURE:
    PrimeModel → PrimeModelRegistry → PrimeModelCache → Backend (HF/Local/API)

    ModelSelectionStrategy → ComplexityAnalyzer → Route to optimal model

USAGE:
    # Basic usage
    model = await PrimeModel.from_pretrained_async("prime-7b-chat-v1")
    response = await model.generate_async("What is AI?")

    # With automatic model selection
    orchestrator = await get_model_orchestrator()
    response = await orchestrator.generate("Complex reasoning task...", auto_select=True)

    # Hot-swap models
    await model.swap_model("prime-13b-reasoning-v1", zero_downtime=True)

INTEGRATION:
    - Continuous Learning: Records experiences for online learning
    - Hybrid Router: Uses complexity signals for model selection
    - Reactor Core: Connects for fine-tuning pipelines
    - Trinity Protocol: Cross-repo event broadcasting
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import os
import sys
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Coroutine,
    Deque,
    Dict,
    Generic,
    Iterator,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
    TYPE_CHECKING,
)

from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# v144.0: HOLLOW CLIENT - STRICT LAZY IMPORTS
# =============================================================================
# CRITICAL: torch, transformers, and other heavy ML libraries are NOT imported
# at module level. They are lazily loaded ONLY when actually needed.
#
# This allows jarvis-prime to start in "Hollow Client" mode using only ~300MB
# RAM, with heavy inference routed to GCP Cloud.
#
# The lazy import pattern:
#   1. Check JARVIS_ENABLE_SLIM_MODE environment variable
#   2. If SLIM mode: raise ImportError when heavy libs are requested
#   3. If FULL mode: import the library on first use
#
# This is the NUCLEAR OPTION to break the OOM crash loop on systems < 32GB RAM.
# =============================================================================

# Lazy-loaded module cache
_torch_module: Optional[Any] = None
_torch_cuda_available: Optional[bool] = None


def _get_torch():
    """
    v144.0: Lazy import torch ONLY when actually needed.

    In Slim Mode, this will raise an ImportError to force callers
    to use GCP instead of local inference.
    """
    global _torch_module

    if _torch_module is not None:
        return _torch_module

    # Check if we're in Slim Mode (heavy imports forbidden)
    slim_mode = os.environ.get("JARVIS_ENABLE_SLIM_MODE", "").lower() in ("true", "1", "yes", "on")
    gcp_offload_active = os.environ.get("JARVIS_GCP_OFFLOAD_ACTIVE", "").lower() in ("true", "1", "yes", "on")

    if slim_mode and gcp_offload_active:
        raise ImportError(
            "[v144.0] HOLLOW CLIENT MODE: torch import blocked. "
            "Heavy inference should be routed to GCP. "
            "Set JARVIS_GCP_OFFLOAD_ACTIVE=false to enable local inference."
        )

    # Lazy import torch
    import torch as _torch
    _torch_module = _torch

    logger.info(
        f"[v144.0] Lazy-loaded torch (version: {_torch.__version__}, "
        f"CUDA: {_torch.cuda.is_available()})"
    )

    return _torch_module


def _is_cuda_available() -> bool:
    """Check CUDA availability without importing torch if possible."""
    global _torch_cuda_available

    if _torch_cuda_available is not None:
        return _torch_cuda_available

    # In Slim Mode, assume no CUDA (will use GCP anyway)
    slim_mode = os.environ.get("JARVIS_ENABLE_SLIM_MODE", "").lower() in ("true", "1", "yes", "on")
    if slim_mode:
        _torch_cuda_available = False
        return False

    try:
        torch = _get_torch()
        _torch_cuda_available = torch.cuda.is_available()
        return _torch_cuda_available
    except ImportError:
        _torch_cuda_available = False
        return False


# Type hints for torch types (only for type checking, not runtime import)
if TYPE_CHECKING:
    import torch
    TorchDtype = torch.dtype
else:
    TorchDtype = Any

# Type variables
T = TypeVar("T")
ModelT = TypeVar("ModelT", bound="PrimeModel")


# =============================================================================
# CONFIGURATION - No Hardcoding, All Environment/Config Based
# =============================================================================


@dataclass
class PrimeModelConfig:
    """
    Dynamic configuration for PRIME models.

    All values can be overridden via environment variables with PRIME_ prefix.
    """
    # Model identification
    model_name: str = ""
    model_version: str = "v1"

    # Quantization
    quantization: Optional[str] = field(default_factory=lambda: os.getenv("PRIME_QUANTIZATION"))
    quantization_bits: int = field(default_factory=lambda: int(os.getenv("PRIME_QUANTIZATION_BITS", "8")))
    use_double_quant: bool = field(default_factory=lambda: os.getenv("PRIME_DOUBLE_QUANT", "true").lower() == "true")

    # Device configuration
    device: str = field(default_factory=lambda: os.getenv("PRIME_DEVICE", "auto"))
    device_map: Optional[str] = field(default_factory=lambda: os.getenv("PRIME_DEVICE_MAP"))
    max_memory: Optional[Dict[str, str]] = None

    # Generation defaults
    max_length: int = field(default_factory=lambda: int(os.getenv("PRIME_MAX_LENGTH", "2048")))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("PRIME_MAX_NEW_TOKENS", "512")))
    temperature: float = field(default_factory=lambda: float(os.getenv("PRIME_TEMPERATURE", "0.7")))
    top_p: float = field(default_factory=lambda: float(os.getenv("PRIME_TOP_P", "0.9")))
    top_k: int = field(default_factory=lambda: int(os.getenv("PRIME_TOP_K", "50")))
    repetition_penalty: float = field(default_factory=lambda: float(os.getenv("PRIME_REPETITION_PENALTY", "1.1")))

    # Performance
    use_cache: bool = field(default_factory=lambda: os.getenv("PRIME_USE_CACHE", "true").lower() == "true")
    use_flash_attention: bool = field(default_factory=lambda: os.getenv("PRIME_FLASH_ATTENTION", "true").lower() == "true")
    torch_dtype: str = field(default_factory=lambda: os.getenv("PRIME_TORCH_DTYPE", "float16"))
    low_cpu_mem_usage: bool = field(default_factory=lambda: os.getenv("PRIME_LOW_CPU_MEM", "true").lower() == "true")

    # Batching
    batch_size: int = field(default_factory=lambda: int(os.getenv("PRIME_BATCH_SIZE", "1")))
    max_batch_wait_ms: float = field(default_factory=lambda: float(os.getenv("PRIME_BATCH_WAIT_MS", "50.0")))

    # Caching
    cache_dir: Optional[Path] = field(default_factory=lambda: Path(os.getenv(
        "PRIME_CACHE_DIR", str(Path.home() / ".jarvis" / "models" / "cache")
    )))

    # Learning integration
    enable_learning_hooks: bool = field(default_factory=lambda: os.getenv("PRIME_LEARNING_HOOKS", "true").lower() == "true")
    record_experiences: bool = field(default_factory=lambda: os.getenv("PRIME_RECORD_EXPERIENCES", "true").lower() == "true")

    # Fallback
    enable_api_fallback: bool = field(default_factory=lambda: os.getenv("PRIME_API_FALLBACK", "true").lower() == "true")
    fallback_api_provider: str = field(default_factory=lambda: os.getenv("PRIME_FALLBACK_PROVIDER", "anthropic"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "quantization": self.quantization,
            "device": self.device,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "use_cache": self.use_cache,
            "use_flash_attention": self.use_flash_attention,
            "enable_learning_hooks": self.enable_learning_hooks,
            "enable_api_fallback": self.enable_api_fallback,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrimeModelConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# MODEL REGISTRY - Dynamic Model Discovery
# =============================================================================


class ModelCapability(Enum):
    """Capabilities that models can have."""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    CODE = "code"
    VISION = "vision"
    REASONING = "reasoning"
    MULTILINGUAL = "multilingual"
    TOOL_USE = "tool_use"
    EMBEDDING = "embedding"


@dataclass
class ModelSpec:
    """Specification for a registered model."""
    name: str
    base_model: str
    description: str
    capabilities: List[ModelCapability]

    # Resource requirements
    size_gb: float
    recommended_ram_gb: int
    recommended_vram_gb: Optional[int] = None
    supports_quantization: List[str] = field(default_factory=lambda: ["4bit", "8bit"])

    # Performance characteristics
    tokens_per_second_cpu: float = 5.0
    tokens_per_second_gpu: float = 50.0
    complexity_handling: float = 0.5  # 0-1, higher = handles more complex tasks

    # Paths
    local_path: Optional[Path] = None
    hub_path: Optional[str] = None

    # Metadata
    version: str = "v1"
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    def can_handle_complexity(self, complexity_score: float) -> bool:
        """Check if model can handle given complexity."""
        return complexity_score <= self.complexity_handling + 0.1


class PrimeModelRegistry:
    """
    Dynamic registry for PRIME models.

    Supports:
    - Auto-discovery from HuggingFace Hub
    - Local model registration
    - Custom model specifications
    - Runtime model updates
    """

    # Default models (can be extended via config)
    _DEFAULT_MODELS: Dict[str, ModelSpec] = {}

    def __init__(self, config_path: Optional[Path] = None):
        self._models: Dict[str, ModelSpec] = {}
        self._config_path = config_path or Path.home() / ".jarvis" / "models" / "registry.json"
        self._lock = asyncio.Lock()

        # Load defaults and config
        self._load_default_models()
        self._load_config()

    def _load_default_models(self) -> None:
        """Load default model specifications."""
        defaults = {
            "prime-7b-chat-v1": ModelSpec(
                name="prime-7b-chat-v1",
                base_model=os.getenv("PRIME_7B_BASE", "meta-llama/Llama-2-7b-chat-hf"),
                description="Chat and Q&A optimized for JARVIS",
                capabilities=[ModelCapability.CHAT, ModelCapability.TEXT_GENERATION, ModelCapability.CODE],
                size_gb=13.5,
                recommended_ram_gb=16,
                recommended_vram_gb=8,
                tokens_per_second_cpu=8.0,
                tokens_per_second_gpu=60.0,
                complexity_handling=0.5,
                tags=["chat", "general", "fast"],
            ),
            "prime-7b-vision-v1": ModelSpec(
                name="prime-7b-vision-v1",
                base_model=os.getenv("PRIME_VISION_BASE", "llava-hf/llava-1.5-7b-hf"),
                description="Multimodal vision + text",
                capabilities=[ModelCapability.VISION, ModelCapability.CHAT, ModelCapability.TEXT_GENERATION],
                size_gb=14.0,
                recommended_ram_gb=16,
                recommended_vram_gb=10,
                tokens_per_second_cpu=5.0,
                tokens_per_second_gpu=40.0,
                complexity_handling=0.6,
                tags=["vision", "multimodal"],
            ),
            "prime-13b-reasoning-v1": ModelSpec(
                name="prime-13b-reasoning-v1",
                base_model=os.getenv("PRIME_13B_BASE", "meta-llama/Llama-2-13b-hf"),
                description="Advanced reasoning and analysis",
                capabilities=[ModelCapability.REASONING, ModelCapability.CODE, ModelCapability.TEXT_GENERATION],
                size_gb=26.0,
                recommended_ram_gb=32,
                recommended_vram_gb=16,
                tokens_per_second_cpu=4.0,
                tokens_per_second_gpu=35.0,
                complexity_handling=0.8,
                tags=["reasoning", "complex", "analysis"],
            ),
            "prime-coder-v1": ModelSpec(
                name="prime-coder-v1",
                base_model=os.getenv("PRIME_CODER_BASE", "codellama/CodeLlama-7b-Instruct-hf"),
                description="Specialized for code generation and analysis",
                capabilities=[ModelCapability.CODE, ModelCapability.TEXT_GENERATION],
                size_gb=13.5,
                recommended_ram_gb=16,
                recommended_vram_gb=8,
                tokens_per_second_cpu=8.0,
                tokens_per_second_gpu=55.0,
                complexity_handling=0.7,
                tags=["code", "programming", "fast"],
            ),
        }
        self._models.update(defaults)

    def _load_config(self) -> None:
        """Load additional models from config file."""
        if not self._config_path.exists():
            return

        try:
            with open(self._config_path) as f:
                data = json.load(f)

            for name, spec_data in data.get("models", {}).items():
                spec_data["capabilities"] = [
                    ModelCapability(c) for c in spec_data.get("capabilities", [])
                ]
                self._models[name] = ModelSpec(**spec_data)

            logger.info(f"Loaded {len(data.get('models', {}))} models from config")
        except Exception as e:
            logger.warning(f"Failed to load model config: {e}")

    async def save_config(self) -> None:
        """Save current registry to config file."""
        async with self._lock:
            data = {
                "models": {
                    name: {
                        **spec.__dict__,
                        "capabilities": [c.value for c in spec.capabilities],
                        "created_at": spec.created_at.isoformat(),
                        "local_path": str(spec.local_path) if spec.local_path else None,
                    }
                    for name, spec in self._models.items()
                },
                "updated_at": datetime.now().isoformat(),
            }

            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    def get(self, name: str) -> Optional[ModelSpec]:
        """Get model specification by name."""
        return self._models.get(name)

    def list_models(self) -> Dict[str, ModelSpec]:
        """List all registered models."""
        return dict(self._models)

    def list_by_capability(self, capability: ModelCapability) -> List[ModelSpec]:
        """List models with a specific capability."""
        return [m for m in self._models.values() if capability in m.capabilities]

    def find_best_for_complexity(self, complexity: float) -> Optional[ModelSpec]:
        """Find the best model for a given complexity score."""
        candidates = [m for m in self._models.values() if m.can_handle_complexity(complexity)]
        if not candidates:
            # Return the most capable model
            return max(self._models.values(), key=lambda m: m.complexity_handling)

        # Return the smallest model that can handle it (efficiency)
        return min(candidates, key=lambda m: m.size_gb)

    async def register(self, spec: ModelSpec) -> None:
        """Register a new model."""
        async with self._lock:
            self._models[spec.name] = spec
            await self.save_config()
            logger.info(f"Registered model: {spec.name}")

    async def unregister(self, name: str) -> bool:
        """Unregister a model."""
        async with self._lock:
            if name in self._models:
                del self._models[name]
                await self.save_config()
                return True
            return False


# =============================================================================
# MODEL CACHE - LRU Cache with Memory Management
# =============================================================================


@dataclass
class CachedModel:
    """A cached model with metadata."""
    model: Any
    tokenizer: Any
    spec: ModelSpec
    config: PrimeModelConfig
    loaded_at: datetime
    last_used: datetime
    use_count: int = 0
    memory_bytes: int = 0

    def update_usage(self) -> None:
        """Update usage statistics."""
        self.last_used = datetime.now()
        self.use_count += 1


class PrimeModelCache:
    """
    LRU cache for loaded models with memory pressure management.

    Features:
    - Automatic eviction based on memory pressure
    - LRU eviction policy
    - Async model loading/unloading
    - Memory usage tracking
    """

    def __init__(
        self,
        max_models: int = 3,
        max_memory_gb: float = 32.0,
        eviction_threshold: float = 0.85,
    ):
        self._max_models = int(os.getenv("PRIME_CACHE_MAX_MODELS", str(max_models)))
        self._max_memory_bytes = int(float(os.getenv("PRIME_CACHE_MAX_MEMORY_GB", str(max_memory_gb))) * 1024**3)
        self._eviction_threshold = float(os.getenv("PRIME_CACHE_EVICTION_THRESHOLD", str(eviction_threshold)))

        self._cache: OrderedDict[str, CachedModel] = OrderedDict()
        self._lock = asyncio.Lock()
        self._loading_locks: Dict[str, asyncio.Lock] = {}

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._total_memory_bytes = 0

    @property
    def current_memory_bytes(self) -> int:
        """Get current memory usage."""
        return sum(m.memory_bytes for m in self._cache.values())

    @property
    def memory_pressure(self) -> float:
        """Get current memory pressure (0-1)."""
        return self.current_memory_bytes / self._max_memory_bytes

    async def get(self, model_name: str) -> Optional[CachedModel]:
        """Get a model from cache."""
        async with self._lock:
            if model_name in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(model_name)
                cached = self._cache[model_name]
                cached.update_usage()
                return cached

            self._misses += 1
            return None

    async def put(
        self,
        model_name: str,
        model: Any,
        tokenizer: Any,
        spec: ModelSpec,
        config: PrimeModelConfig,
    ) -> CachedModel:
        """Put a model in cache."""
        async with self._lock:
            # Estimate memory usage
            memory_bytes = self._estimate_memory(model, spec)

            # Evict if necessary
            await self._maybe_evict(memory_bytes)

            cached = CachedModel(
                model=model,
                tokenizer=tokenizer,
                spec=spec,
                config=config,
                loaded_at=datetime.now(),
                last_used=datetime.now(),
                memory_bytes=memory_bytes,
            )

            self._cache[model_name] = cached
            self._total_memory_bytes += memory_bytes

            logger.info(f"Cached model {model_name} ({memory_bytes / 1024**3:.2f} GB)")
            return cached

    def _estimate_memory(self, model: Any, spec: ModelSpec) -> int:
        """Estimate model memory usage in bytes."""
        # Use spec size as base estimate
        estimated_gb = spec.size_gb

        # Try to get actual parameter count
        try:
            if hasattr(model, "num_parameters"):
                params = model.num_parameters()
                # Estimate 2 bytes per param for fp16
                estimated_gb = (params * 2) / 1024**3
        except Exception:
            pass

        return int(estimated_gb * 1024**3)

    async def _maybe_evict(self, needed_bytes: int) -> None:
        """Evict models if necessary to make room."""
        # Check if we need to evict
        if len(self._cache) >= self._max_models:
            await self._evict_lru()

        # Check memory pressure
        while self.memory_pressure > self._eviction_threshold or \
              (self.current_memory_bytes + needed_bytes > self._max_memory_bytes):
            if not self._cache:
                break
            await self._evict_lru()

    async def _evict_lru(self) -> None:
        """Evict least recently used model."""
        if not self._cache:
            return

        # Get LRU (first item)
        model_name = next(iter(self._cache))
        cached = self._cache.pop(model_name)

        # Clear from memory
        del cached.model
        del cached.tokenizer
        gc.collect()

        # v144.0: Use lazy torch import
        if _is_cuda_available():
            _get_torch().cuda.empty_cache()

        self._evictions += 1
        self._total_memory_bytes -= cached.memory_bytes

        logger.info(f"Evicted model {model_name} from cache")

    async def remove(self, model_name: str) -> bool:
        """Remove a specific model from cache."""
        async with self._lock:
            if model_name in self._cache:
                cached = self._cache.pop(model_name)
                del cached.model
                del cached.tokenizer
                gc.collect()
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached models."""
        async with self._lock:
            for cached in self._cache.values():
                del cached.model
                del cached.tokenizer
            self._cache.clear()
            gc.collect()

            # v144.0: Use lazy torch import
            if _is_cuda_available():
                _get_torch().cuda.empty_cache()

    def get_loading_lock(self, model_name: str) -> asyncio.Lock:
        """Get or create a loading lock for a model."""
        if model_name not in self._loading_locks:
            self._loading_locks[model_name] = asyncio.Lock()
        return self._loading_locks[model_name]

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_models": list(self._cache.keys()),
            "num_cached": len(self._cache),
            "max_models": self._max_models,
            "current_memory_gb": self.current_memory_bytes / 1024**3,
            "max_memory_gb": self._max_memory_bytes / 1024**3,
            "memory_pressure": self.memory_pressure,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
            "evictions": self._evictions,
        }


# =============================================================================
# MODEL LOADER - Async Model Loading with Progress
# =============================================================================


class ModelLoadingProgress:
    """Progress tracker for model loading."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.stage = "initializing"
        self.progress = 0.0
        self.message = ""
        self.started_at = time.time()
        self._callbacks: List[Callable[[str, float, str], None]] = []

    def update(self, stage: str, progress: float, message: str = "") -> None:
        """Update progress."""
        self.stage = stage
        self.progress = progress
        self.message = message

        for callback in self._callbacks:
            try:
                callback(stage, progress, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def add_callback(self, callback: Callable[[str, float, str], None]) -> None:
        """Add a progress callback."""
        self._callbacks.append(callback)

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time."""
        return time.time() - self.started_at


class PrimeModelLoader:
    """
    Async model loader with comprehensive features.

    Features:
    - Async loading with progress tracking
    - Automatic device detection and optimization
    - Quantization support (4-bit, 8-bit)
    - Flash attention when available
    - Error recovery and fallback
    """

    def __init__(self, registry: PrimeModelRegistry, cache: PrimeModelCache):
        self._registry = registry
        self._cache = cache

    def _detect_optimal_device(self) -> str:
        """Detect the optimal device for model loading."""
        # v144.0: Use lazy torch import
        torch = _get_torch()
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _get_torch_dtype(self, config: PrimeModelConfig) -> TorchDtype:
        """Get torch dtype from config."""
        # v144.0: Use lazy torch import
        torch = _get_torch()
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(config.torch_dtype, torch.float16)

    async def load(
        self,
        model_name: str,
        config: Optional[PrimeModelConfig] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> CachedModel:
        """
        Load a model asynchronously.

        Args:
            model_name: Name of the model to load
            config: Optional configuration override
            progress_callback: Optional progress callback

        Returns:
            CachedModel instance
        """
        # Check cache first
        cached = await self._cache.get(model_name)
        if cached:
            logger.info(f"Model {model_name} loaded from cache")
            return cached

        # Get loading lock to prevent duplicate loads
        loading_lock = self._cache.get_loading_lock(model_name)

        async with loading_lock:
            # Double-check cache after acquiring lock
            cached = await self._cache.get(model_name)
            if cached:
                return cached

            # Get model spec
            spec = self._registry.get(model_name)
            if not spec:
                raise ValueError(f"Unknown model: {model_name}. Available: {list(self._registry.list_models().keys())}")

            # Prepare config
            config = config or PrimeModelConfig()
            config.model_name = model_name

            # Setup progress tracking
            progress = ModelLoadingProgress(model_name)
            if progress_callback:
                progress.add_callback(progress_callback)

            # Load in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model, tokenizer = await loop.run_in_executor(
                None,
                lambda: self._load_model_sync(spec, config, progress)
            )

            # Cache and return
            return await self._cache.put(model_name, model, tokenizer, spec, config)

    def _load_model_sync(
        self,
        spec: ModelSpec,
        config: PrimeModelConfig,
        progress: ModelLoadingProgress,
    ) -> Tuple[Any, Any]:
        """Synchronous model loading (runs in executor)."""
        # v144.0: Lazy import transformers ONLY when actually loading a model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch = _get_torch()  # v144.0: Lazy torch

        progress.update("initializing", 0.0, f"Loading {spec.name}...")

        # Determine device
        device = config.device if config.device != "auto" else self._detect_optimal_device()

        # Prepare load kwargs
        load_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": config.low_cpu_mem_usage,
            "trust_remote_code": True,
        }

        # Set torch dtype
        if device != "cpu":
            load_kwargs["torch_dtype"] = self._get_torch_dtype(config)

        # Flash attention
        if config.use_flash_attention and device == "cuda":
            try:
                load_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                logger.warning("Flash attention not available")

        # Device map
        if config.device_map:
            load_kwargs["device_map"] = config.device_map
        elif device == "cuda":
            load_kwargs["device_map"] = "auto"

        # Quantization
        if config.quantization:
            progress.update("quantization", 0.1, f"Setting up {config.quantization} quantization...")

            if config.quantization in ("4bit", "4"):
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self._get_torch_dtype(config),
                    bnb_4bit_use_double_quant=config.use_double_quant,
                    bnb_4bit_quant_type="nf4",
                )
            elif config.quantization in ("8bit", "8"):
                load_kwargs["load_in_8bit"] = True

        # Load tokenizer
        progress.update("tokenizer", 0.2, "Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            spec.base_model,
            trust_remote_code=True,
            cache_dir=str(config.cache_dir) if config.cache_dir else None,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        progress.update("model", 0.4, "Loading model weights...")

        model = AutoModelForCausalLM.from_pretrained(
            spec.base_model,
            cache_dir=str(config.cache_dir) if config.cache_dir else None,
            **load_kwargs,
        )

        # Move to device if needed (not using device_map)
        if not config.quantization and not config.device_map and device != "cpu":
            progress.update("device", 0.9, f"Moving to {device}...")
            model = model.to(device)

        model.eval()

        progress.update("complete", 1.0, "Model loaded successfully")
        logger.info(f"Loaded model {spec.name} on {device}")

        return model, tokenizer


# =============================================================================
# PRIME MODEL - Main Interface
# =============================================================================


class GenerationMode(Enum):
    """Generation modes for different use cases."""
    GREEDY = "greedy"
    SAMPLING = "sampling"
    BEAM_SEARCH = "beam_search"
    CONTRASTIVE = "contrastive"


@dataclass
class GenerationResult:
    """Result from model generation."""
    text: str
    model_name: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float

    # Optional metadata
    prompt_tokens: int = 0
    finish_reason: str = "stop"
    logprobs: Optional[List[float]] = None

    # Tracking
    experience_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "model_name": self.model_name,
            "tokens_generated": self.tokens_generated,
            "generation_time_ms": self.generation_time_ms,
            "tokens_per_second": self.tokens_per_second,
            "prompt_tokens": self.prompt_tokens,
            "finish_reason": self.finish_reason,
            "experience_id": self.experience_id,
        }


class PrimeModel:
    """
    Advanced unified interface for JARVIS PRIME models.

    Features:
    - Async generation with streaming
    - Automatic batching for throughput
    - Hot-swap model capability
    - Integration with continuous learning
    - Seamless API fallback
    - Comprehensive metrics

    Usage:
        # Async loading
        model = await PrimeModel.from_pretrained_async("prime-7b-chat-v1")

        # Async generation
        result = await model.generate_async("What is AI?")

        # Streaming generation
        async for chunk in model.stream_async("Tell me a story"):
            print(chunk, end="")

        # Hot-swap
        await model.swap_model("prime-13b-reasoning-v1")
    """

    # Class-level resources (shared across instances)
    _registry: Optional[PrimeModelRegistry] = None
    _cache: Optional[PrimeModelCache] = None
    _loader: Optional[PrimeModelLoader] = None
    _lock = asyncio.Lock()

    def __init__(
        self,
        cached_model: CachedModel,
        config: Optional[PrimeModelConfig] = None,
    ):
        """Initialize with a cached model. Use from_pretrained_async() for creation."""
        self._cached = cached_model
        self._config = config or cached_model.config

        # Device detection
        self._device = self._detect_current_device()

        # Batch queue for throughput optimization
        self._batch_queue: Deque[Tuple[str, asyncio.Future]] = deque()
        self._batch_task: Optional[asyncio.Task] = None

        # Statistics
        self._total_generations = 0
        self._total_tokens = 0
        self._total_time_ms = 0.0

        # Learning hooks
        self._learning_engine: Optional[Any] = None
        if config and config.enable_learning_hooks:
            self._init_learning_hooks()

    @classmethod
    async def _init_class_resources(cls) -> None:
        """Initialize shared class resources."""
        async with cls._lock:
            if cls._registry is None:
                cls._registry = PrimeModelRegistry()
            if cls._cache is None:
                cls._cache = PrimeModelCache()
            if cls._loader is None:
                cls._loader = PrimeModelLoader(cls._registry, cls._cache)

    @classmethod
    async def from_pretrained_async(
        cls,
        model_name: str,
        config: Optional[PrimeModelConfig] = None,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> "PrimeModel":
        """
        Asynchronously load a pretrained model.

        Args:
            model_name: Name of the model to load
            config: Optional configuration
            progress_callback: Optional progress callback

        Returns:
            PrimeModel instance
        """
        await cls._init_class_resources()

        cached = await cls._loader.load(model_name, config, progress_callback)
        return cls(cached, config)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        quantization: Optional[str] = None,
        device: str = "auto",
        **kwargs: Any,
    ) -> "PrimeModel":
        """
        Synchronous loading (for backwards compatibility).

        Prefer from_pretrained_async() for production use.
        """
        config = PrimeModelConfig(
            model_name=model_name,
            quantization=quantization,
            device=device,
            **{k: v for k, v in kwargs.items() if k in PrimeModelConfig.__dataclass_fields__}
        )

        # Run async in new event loop
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(cls.from_pretrained_async(model_name, config))
        finally:
            loop.close()

    def _detect_current_device(self) -> str:
        """Detect the device the model is on."""
        try:
            first_param = next(self._cached.model.parameters())
            return str(first_param.device)
        except Exception:
            return "cpu"

    def _init_learning_hooks(self) -> None:
        """Initialize continuous learning integration."""
        try:
            # Lazy import to avoid circular dependencies
            from jarvis_prime.core.continuous_learning import get_continuous_learning_engine

            async def init():
                self._learning_engine = await get_continuous_learning_engine()

            asyncio.create_task(init())
        except ImportError:
            logger.debug("Continuous learning module not available")

    @property
    def model(self) -> Any:
        """Get the underlying model."""
        return self._cached.model

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        return self._cached.tokenizer

    @property
    def config(self) -> PrimeModelConfig:
        """Get the configuration."""
        return self._config

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._cached.spec.name

    @property
    def device(self) -> str:
        """Get the device."""
        return self._device

    async def generate_async(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        mode: GenerationMode = GenerationMode.SAMPLING,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate text asynchronously.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            mode: Generation mode
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with generated text and metadata
        """
        start_time = time.time()

        # Prepare generation config
        gen_kwargs = self._prepare_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            mode=mode,
            **kwargs,
        )

        # Run generation in executor
        loop = asyncio.get_event_loop()
        generated_text, num_tokens = await loop.run_in_executor(
            None,
            lambda: self._generate_sync(prompt, gen_kwargs)
        )

        # Calculate metrics
        elapsed_ms = (time.time() - start_time) * 1000
        tokens_per_second = (num_tokens / elapsed_ms) * 1000 if elapsed_ms > 0 else 0

        # Update statistics
        self._total_generations += 1
        self._total_tokens += num_tokens
        self._total_time_ms += elapsed_ms
        self._cached.update_usage()

        # Create result
        result = GenerationResult(
            text=generated_text,
            model_name=self.model_name,
            tokens_generated=num_tokens,
            generation_time_ms=elapsed_ms,
            tokens_per_second=tokens_per_second,
            prompt_tokens=len(self.tokenizer.encode(prompt)),
        )

        # Record experience for learning
        if self._config.record_experiences and self._learning_engine:
            try:
                exp_id = await self._learning_engine.record_experience(
                    input_text=prompt,
                    output_text=generated_text,
                    model_version=self.model_name,
                    latency_ms=elapsed_ms,
                    tokens_used=num_tokens,
                )
                result.experience_id = exp_id
            except Exception as e:
                logger.debug(f"Failed to record experience: {e}")

        return result

    def _prepare_generation_kwargs(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        mode: GenerationMode = GenerationMode.SAMPLING,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare generation kwargs from config and overrides."""
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self._config.max_new_tokens,
            "temperature": temperature or self._config.temperature,
            "top_p": top_p or self._config.top_p,
            "top_k": top_k or self._config.top_k,
            "repetition_penalty": self._config.repetition_penalty,
            "use_cache": self._config.use_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Mode-specific settings
        if mode == GenerationMode.GREEDY:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("top_k", None)
        elif mode == GenerationMode.BEAM_SEARCH:
            gen_kwargs["do_sample"] = False
            gen_kwargs["num_beams"] = kwargs.get("num_beams", 4)
            gen_kwargs["early_stopping"] = True
        elif mode == GenerationMode.CONTRASTIVE:
            gen_kwargs["penalty_alpha"] = kwargs.get("penalty_alpha", 0.6)
            gen_kwargs["top_k"] = kwargs.get("top_k", 4)
        else:
            gen_kwargs["do_sample"] = True

        # Add any extra kwargs
        gen_kwargs.update(kwargs)

        return gen_kwargs

    def _generate_sync(self, prompt: str, gen_kwargs: Dict[str, Any]) -> Tuple[str, int]:
        """Synchronous generation (runs in executor)."""
        # v144.0: Lazy torch import for inference
        torch = _get_torch()

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
        )

        # Move to device
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self._cached.model.generate(
                **inputs,
                **gen_kwargs,
            )

        # Decode
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text.strip(), len(generated_ids)

    async def stream_async(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream generation asynchronously.

        Yields text chunks as they're generated.
        """
        from transformers import TextIteratorStreamer
        from threading import Thread

        gen_kwargs = self._prepare_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )

        # Setup streamer
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Generate in thread
        gen_kwargs["streamer"] = streamer

        thread = Thread(
            target=lambda: self._cached.model.generate(**inputs, **gen_kwargs)
        )
        thread.start()

        # Yield chunks
        for text in streamer:
            yield text
            await asyncio.sleep(0)  # Allow other tasks to run

        thread.join()

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous generation (backwards compatible).

        Prefer generate_async() for production use.
        """
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.generate_async(
                    prompt,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    **kwargs,
                )
            )
            return result.text
        finally:
            loop.close()

    async def chat_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Chat interface with conversation history.

        Args:
            messages: List of message dicts with "role" and "content"
            **kwargs: Generation parameters

        Returns:
            GenerationResult with assistant response
        """
        # Format messages
        prompt = self._format_chat_messages(messages)
        return await self.generate_async(prompt, **kwargs)

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """Synchronous chat (backwards compatible)."""
        prompt = self._format_chat_messages(messages)
        return self.generate(prompt, **kwargs)

    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages into a prompt string."""
        # Try to use tokenizer's chat template if available
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass

        # Fallback to simple formatting
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        formatted.append("Assistant:")
        return "\n".join(formatted)

    async def swap_model(
        self,
        new_model_name: str,
        zero_downtime: bool = True,
        config: Optional[PrimeModelConfig] = None,
    ) -> None:
        """
        Hot-swap to a different model.

        Args:
            new_model_name: Name of the new model
            zero_downtime: If True, keeps old model until new one is ready
            config: Optional configuration for the new model
        """
        logger.info(f"Hot-swapping from {self.model_name} to {new_model_name}")

        # Load new model
        new_cached = await self._loader.load(new_model_name, config)

        # Swap atomically
        old_cached = self._cached
        self._cached = new_cached
        self._device = self._detect_current_device()

        # Clean up old model if not zero_downtime or if models are different
        if not zero_downtime or old_cached.spec.name != new_cached.spec.name:
            await self._cache.remove(old_cached.spec.name)

        logger.info(f"Successfully swapped to {new_model_name}")

    async def record_feedback(
        self,
        experience_id: str,
        score: float,
        feedback_text: Optional[str] = None,
    ) -> bool:
        """
        Record feedback for a generation.

        Args:
            experience_id: ID from GenerationResult
            score: Feedback score (0-1)
            feedback_text: Optional feedback text

        Returns:
            True if feedback was recorded
        """
        if not self._learning_engine:
            return False

        try:
            return await self._learning_engine.record_feedback(
                experience_id=experience_id,
                score=score,
                feedback_text=feedback_text,
            )
        except Exception as e:
            logger.warning(f"Failed to record feedback: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        return {
            "model_name": self.model_name,
            "device": self._device,
            "config": self._config.to_dict(),
            "total_generations": self._total_generations,
            "total_tokens": self._total_tokens,
            "total_time_ms": self._total_time_ms,
            "avg_tokens_per_second": (
                (self._total_tokens / self._total_time_ms) * 1000
                if self._total_time_ms > 0 else 0
            ),
            "cache_stats": self._cache.get_statistics() if self._cache else None,
            "loaded_at": self._cached.loaded_at.isoformat(),
            "use_count": self._cached.use_count,
        }

    @classmethod
    def list_models(cls) -> Dict[str, ModelSpec]:
        """List all available PRIME models."""
        if cls._registry is None:
            cls._registry = PrimeModelRegistry()
        return cls._registry.list_models()


# =============================================================================
# MODEL ORCHESTRATOR - Automatic Model Selection
# =============================================================================


class PrimeModelOrchestrator:
    """
    Orchestrates automatic model selection based on task complexity.

    Features:
    - Complexity-based model routing
    - Seamless fallback to API
    - Load balancing across models
    - Health monitoring
    """

    def __init__(
        self,
        registry: PrimeModelRegistry,
        cache: PrimeModelCache,
        loader: PrimeModelLoader,
    ):
        self._registry = registry
        self._cache = cache
        self._loader = loader

        # Active models
        self._models: Dict[str, PrimeModel] = {}
        self._lock = asyncio.Lock()

        # Complexity analyzer
        self._complexity_analyzer: Optional[Any] = None

        # API fallback client
        self._api_client: Optional[Any] = None
        self._api_fallback_enabled = os.getenv("PRIME_API_FALLBACK", "true").lower() == "true"

        # Statistics
        self._routing_stats: Dict[str, int] = {}

    async def _init_complexity_analyzer(self) -> None:
        """Initialize the complexity analyzer."""
        if self._complexity_analyzer:
            return

        try:
            from jarvis_prime.core.hybrid_router import DefaultComplexityAnalyzer
            self._complexity_analyzer = DefaultComplexityAnalyzer()
        except ImportError:
            logger.warning("Complexity analyzer not available, using simple heuristics")

    async def _init_api_client(self) -> None:
        """Initialize the API fallback client."""
        if self._api_client or not self._api_fallback_enabled:
            return

        try:
            import anthropic
            self._api_client = anthropic.AsyncAnthropic()
        except ImportError:
            logger.warning("Anthropic client not available for API fallback")

    async def generate(
        self,
        prompt: str,
        auto_select: bool = True,
        preferred_model: Optional[str] = None,
        max_complexity_for_local: float = 0.8,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generate with automatic model selection.

        Args:
            prompt: Input prompt
            auto_select: Enable automatic model selection
            preferred_model: Preferred model (used if suitable)
            max_complexity_for_local: Max complexity for local models
            **kwargs: Generation parameters

        Returns:
            GenerationResult
        """
        await self._init_complexity_analyzer()

        # Analyze complexity
        complexity_score = 0.5
        if self._complexity_analyzer and auto_select:
            signals = await self._complexity_analyzer.analyze(prompt)
            complexity_score = self._calculate_complexity(signals)

        # Select model
        model_name = await self._select_model(
            complexity_score,
            preferred_model,
            max_complexity_for_local,
        )

        if model_name:
            # Use local model
            model = await self._get_or_load_model(model_name)
            result = await model.generate_async(prompt, **kwargs)
            self._routing_stats[model_name] = self._routing_stats.get(model_name, 0) + 1
            return result

        # Fallback to API
        if self._api_fallback_enabled:
            return await self._api_fallback_generate(prompt, **kwargs)

        raise RuntimeError("No suitable model available and API fallback disabled")

    def _calculate_complexity(self, signals: Any) -> float:
        """Calculate complexity score from signals."""
        try:
            weights = {
                "token_count": 0.1,
                "reasoning": 0.3,
                "multi_step": 0.25,
                "domain": 0.15,
                "external_knowledge": 0.1,
                "tool_use": 0.1,
            }

            score = (
                weights["token_count"] * min(getattr(signals, "token_count", 0) / 2000, 1.0) +
                weights["reasoning"] * getattr(signals, "reasoning_indicators", 0) +
                weights["multi_step"] * getattr(signals, "multi_step_indicators", 0) +
                weights["domain"] * getattr(signals, "domain_specificity", 0) +
                weights["external_knowledge"] * getattr(signals, "requires_external_knowledge", 0) +
                weights["tool_use"] * getattr(signals, "requires_tool_use", 0)
            )

            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.5

    async def _select_model(
        self,
        complexity: float,
        preferred: Optional[str],
        max_local_complexity: float,
    ) -> Optional[str]:
        """Select the best model for the complexity."""
        # Check if complexity is too high for local
        if complexity > max_local_complexity:
            return None

        # Check preferred model
        if preferred:
            spec = self._registry.get(preferred)
            if spec and spec.can_handle_complexity(complexity):
                return preferred

        # Find best model
        spec = self._registry.find_best_for_complexity(complexity)
        return spec.name if spec else None

    async def _get_or_load_model(self, model_name: str) -> PrimeModel:
        """Get or load a model."""
        async with self._lock:
            if model_name in self._models:
                return self._models[model_name]

            model = await PrimeModel.from_pretrained_async(model_name)
            self._models[model_name] = model
            return model

    async def _api_fallback_generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate using API fallback."""
        await self._init_api_client()

        if not self._api_client:
            raise RuntimeError("API client not available")

        start_time = time.time()

        response = await self._api_client.messages.create(
            model=os.getenv("PRIME_FALLBACK_MODEL", "claude-3-haiku-20240307"),
            max_tokens=kwargs.get("max_new_tokens", 512),
            messages=[{"role": "user", "content": prompt}],
        )

        elapsed_ms = (time.time() - start_time) * 1000
        text = response.content[0].text

        self._routing_stats["api_fallback"] = self._routing_stats.get("api_fallback", 0) + 1

        return GenerationResult(
            text=text,
            model_name="api_fallback",
            tokens_generated=response.usage.output_tokens,
            generation_time_ms=elapsed_ms,
            tokens_per_second=(response.usage.output_tokens / elapsed_ms) * 1000 if elapsed_ms > 0 else 0,
            prompt_tokens=response.usage.input_tokens,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "loaded_models": list(self._models.keys()),
            "routing_stats": self._routing_stats,
            "api_fallback_enabled": self._api_fallback_enabled,
            "cache_stats": self._cache.get_statistics(),
            "registry_models": list(self._registry.list_models().keys()),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_orchestrator: Optional[PrimeModelOrchestrator] = None
_orchestrator_lock = asyncio.Lock()


async def get_model_orchestrator() -> PrimeModelOrchestrator:
    """Get or create the global model orchestrator."""
    global _orchestrator

    async with _orchestrator_lock:
        if _orchestrator is None:
            registry = PrimeModelRegistry()
            cache = PrimeModelCache()
            loader = PrimeModelLoader(registry, cache)
            _orchestrator = PrimeModelOrchestrator(registry, cache, loader)

        return _orchestrator


async def load_model(model_name: str, **kwargs: Any) -> PrimeModel:
    """Convenience function to load a model."""
    config = PrimeModelConfig(**kwargs) if kwargs else None
    return await PrimeModel.from_pretrained_async(model_name, config)


# =============================================================================
# CLI / MAIN
# =============================================================================


if __name__ == "__main__":
    async def main():
        print("Available PRIME models:")
        for name, spec in PrimeModel.list_models().items():
            print(f"\n  {name}:")
            print(f"    Description: {spec.description}")
            print(f"    Base: {spec.base_model}")
            print(f"    Size: {spec.size_gb} GB")
            print(f"    Capabilities: {[c.value for c in spec.capabilities]}")
            print(f"    Complexity handling: {spec.complexity_handling}")

    asyncio.run(main())
