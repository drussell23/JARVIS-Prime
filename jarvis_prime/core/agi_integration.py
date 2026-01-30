"""
JARVIS-Prime AGI Integration Layer
===================================

v77.0 - Unified Integration of All AGI Subsystems

This module serves as the central nervous system connecting all AGI components:
- AGIOrchestrator: Multi-model cognitive coordination
- ReasoningEngine: Advanced reasoning strategies
- AppleSiliconOptimizer: Hardware acceleration
- ContinuousLearningEngine: Online learning from interactions
- MultiModalFusionEngine: Cross-modal understanding

ARCHITECTURE:
    ┌──────────────────────────────────────────────────────────────────┐
    │                     AGI Integration Hub                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
    │  │   Request   │──│  Reasoning  │──│  AGI Orchestrator       │  │
    │  │   Router    │  │   Engine    │  │  (Multi-Model Coord)    │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
    │        │                │                      │                │
    │        │                │                      │                │
    │  ┌─────▼─────┐  ┌───────▼──────┐  ┌──────────▼─────────────┐   │
    │  │ Hardware  │  │  Learning    │  │  Multi-Modal Fusion    │   │
    │  │ Optimizer │  │  Engine      │  │  Engine                │   │
    │  └───────────┘  └──────────────┘  └────────────────────────┘   │
    └──────────────────────────────────────────────────────────────────┘

This integration layer provides:
1. Unified initialization of all AGI subsystems
2. Shared cognitive state across components
3. Reasoning-augmented inference pipeline
4. Automatic experience recording for learning
5. Hardware-optimized model loading
"""

from __future__ import annotations

# =============================================================================
# v93.15: CRITICAL - Suppress ML library warnings BEFORE heavy imports
# These warnings are emitted during torch/sklearn import
# =============================================================================
import warnings
warnings.filterwarnings('ignore', message='.*scikit-learn version.*is not supported.*')
warnings.filterwarnings('ignore', message='.*Disabling scikit-learn conversion API.*')
warnings.filterwarnings('ignore', message='.*Torch version.*has not been tested.*')
warnings.filterwarnings('ignore', message='.*coremltools.*')
warnings.filterwarnings('ignore', category=UserWarning, module='coremltools')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', category=DeprecationWarning)
# =============================================================================

import asyncio
import concurrent.futures
import gc
import logging
import os  # v93.14: Added for environment variable access
import threading
import time
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# =============================================================================
# v138.0: MEMORY-AWARE STAGED INITIALIZATION
# Fixes OOM (Exit Code -9) by replacing unbounded parallel initialization
# with a staged pipeline that includes memory gates between phases
# =============================================================================

logger = logging.getLogger(__name__)

# Type variable for generic lazy proxy
T = TypeVar('T')

# psutil availability check (non-blocking)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available - memory monitoring disabled")


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================


class AGISubsystem(Enum):
    """All AGI subsystems managed by the integration hub."""

    ORCHESTRATOR = auto()
    REASONING = auto()
    LEARNING = auto()
    MULTIMODAL = auto()
    HARDWARE = auto()
    # v80.0 AGI Models
    AGI_MODELS = auto()
    CONTINUAL_LEARNING = auto()
    SELF_IMPROVEMENT = auto()


class RequestComplexity(Enum):
    """Request complexity classification for routing."""

    TRIVIAL = auto()      # Simple factual queries
    SIMPLE = auto()       # Single-step operations
    MODERATE = auto()     # Multi-step, single-domain
    COMPLEX = auto()      # Multi-step, multi-domain
    EXPERT = auto()       # Requires deep reasoning


class ReasoningRequirement(Enum):
    """Type of reasoning required for a request."""

    NONE = auto()         # Direct response
    CHAIN = auto()        # Chain-of-thought
    TREE = auto()         # Tree-of-thoughts exploration
    CAUSAL = auto()       # Causal understanding
    PLANNING = auto()     # Action planning
    META = auto()         # Meta-cognitive reasoning


# =============================================================================
# v138.0: MEMORY-AWARE STAGED INITIALIZATION ENUMS AND STRUCTURES
# =============================================================================


class InitStage(Enum):
    """
    v138.0: Initialization stages for memory-aware staged loading.

    Subsystems are grouped into stages based on:
    1. Memory footprint (lighter subsystems first)
    2. Dependencies (foundational subsystems before consumers)
    3. Criticality (core functionality before optional features)
    """
    PRE_FLIGHT = auto()       # Pre-flight checks and OOM protection setup
    STAGE_1_FOUNDATION = auto()  # Hardware optimizer + Orchestrator (critical, light)
    STAGE_2_REASONING = auto()   # Reasoning + Learning (moderate memory)
    STAGE_3_HEAVY = auto()       # Multimodal + AGI v80 models (heavy, optional)
    COMPLETED = auto()        # All stages complete


class MemoryPressure(Enum):
    """
    v138.0: Memory pressure levels for adaptive initialization.

    Used to dynamically adjust initialization strategy based on
    available system memory.
    """
    MINIMAL = auto()      # <50% used - full initialization
    LOW = auto()          # 50-65% used - normal initialization
    MODERATE = auto()     # 65-80% used - conservative initialization
    HIGH = auto()         # 80-90% used - slim mode, defer heavy subsystems
    CRITICAL = auto()     # >90% used - emergency mode, minimal subsystems only


class SubsystemPriority(Enum):
    """
    v138.0: Subsystem priority for adaptive loading.

    In high memory pressure, only CRITICAL subsystems are loaded immediately.
    IMPORTANT subsystems are loaded if memory permits.
    OPTIONAL subsystems are lazy-loaded on first use.
    """
    CRITICAL = 1      # Must load for basic functionality
    IMPORTANT = 2     # Should load, but can be deferred
    OPTIONAL = 3      # Can be lazy-loaded on demand


@dataclass
class MemorySnapshot:
    """
    v138.0: Point-in-time memory state for gate decisions.
    """
    timestamp: float
    total_mb: float
    available_mb: float
    used_percent: float
    process_rss_mb: float
    swap_used_mb: float = 0.0

    @property
    def headroom_mb(self) -> float:
        """Available memory headroom in MB."""
        return self.available_mb

    @property
    def headroom_percent(self) -> float:
        """Available memory as percentage of total."""
        return 100.0 - self.used_percent


@dataclass
class StagedInitConfig:
    """
    v138.0: Configuration for memory-aware staged initialization.
    """
    # Memory gate thresholds (percentage of total memory)
    min_headroom_percent: float = 20.0       # Minimum free memory to proceed
    warning_headroom_percent: float = 30.0   # Trigger GC below this
    slim_mode_threshold: float = 15.0        # Enter slim mode below this

    # Stage timing (max time per stage in seconds)
    stage_timeout: float = 45.0
    pre_flight_timeout: float = 10.0
    gc_timeout: float = 5.0

    # Parallelism within stages
    max_parallel_per_stage: int = 2  # Max subsystems to init in parallel per stage

    # Memory gate behavior
    gc_between_stages: bool = True           # Run GC between stages
    gc_generations: int = 2                  # GC generations to collect (0-2)
    memory_gate_retry_count: int = 3         # Retries if memory gate fails
    memory_gate_retry_delay: float = 2.0     # Delay between retries

    # Adaptive behavior
    enable_slim_mode: bool = True            # Auto-enable slim mode in low memory
    defer_heavy_on_pressure: bool = True     # Defer heavy subsystems on memory pressure
    lazy_load_optional: bool = True          # Lazy-load optional subsystems

    # Environment variable overrides
    @classmethod
    def from_env(cls) -> "StagedInitConfig":
        """
        Create config from environment variables.
        
        v149.0: Hardware-aware configuration.
        On SLIM/CLOUD_ONLY profiles, reduces memory requirements since
        heavy workloads are offloaded to GCP.
        """
        # v149.0: Detect hardware profile from supervisor
        hardware_profile = os.getenv("JARVIS_HARDWARE_PROFILE", "").upper()
        is_slim_hardware = hardware_profile in ("SLIM", "CLOUD_ONLY")
        
        # v149.0: Use relaxed thresholds for SLIM hardware
        # Since GCP handles heavy work, we don't need as much local headroom
        if is_slim_hardware:
            default_min_headroom = "10.0"  # Reduced from 20.0
            default_slim_threshold = "8.0"  # Reduced from 15.0
            logger.info(
                f"[v149.0] SLIM hardware detected ({hardware_profile}) - "
                f"using relaxed memory thresholds"
            )
        else:
            default_min_headroom = "20.0"
            default_slim_threshold = "15.0"
        
        return cls(
            min_headroom_percent=float(os.getenv("AGI_MIN_HEADROOM_PERCENT", default_min_headroom)),
            warning_headroom_percent=float(os.getenv("AGI_WARNING_HEADROOM_PERCENT", "30.0")),
            slim_mode_threshold=float(os.getenv("AGI_SLIM_MODE_THRESHOLD", default_slim_threshold)),
            stage_timeout=float(os.getenv("AGI_STAGE_TIMEOUT", "45.0")),
            enable_slim_mode=os.getenv("AGI_ENABLE_SLIM_MODE", "true").lower() == "true",
            defer_heavy_on_pressure=os.getenv("AGI_DEFER_HEAVY", "true").lower() == "true",
            lazy_load_optional=os.getenv("AGI_LAZY_LOAD", "true").lower() == "true",
        )


@dataclass
class StageResult:
    """
    v138.0: Result of a single initialization stage.
    """
    stage: InitStage
    success: bool
    subsystems_initialized: List[str]
    subsystems_failed: List[str]
    subsystems_deferred: List[str]
    elapsed_seconds: float
    memory_before: Optional[MemorySnapshot] = None
    memory_after: Optional[MemorySnapshot] = None
    error: Optional[str] = None

    @property
    def memory_delta_mb(self) -> float:
        """Memory change during this stage (positive = increased usage)."""
        if self.memory_before and self.memory_after:
            return self.memory_after.process_rss_mb - self.memory_before.process_rss_mb
        return 0.0


# =============================================================================
# v138.0: LAZY LOADING PROXY
# =============================================================================


class LazySubsystemProxy(Generic[T]):
    """
    v138.0: Lazy loading proxy for deferred subsystem initialization.

    Wraps a subsystem factory and only initializes the actual subsystem
    when first accessed. This prevents memory spikes from loading all
    subsystems at startup.

    Features:
    - Thread-safe initialization via asyncio.Lock
    - Transparent attribute forwarding
    - Memory-aware initialization with pressure checks
    - Timeout protection
    - Initialization metrics tracking

    Usage:
        proxy = LazySubsystemProxy(
            factory=lambda: MyHeavySubsystem(),
            async_init=lambda s: s.initialize(),
            name="heavy_subsystem"
        )

        # Later, when actually needed:
        result = await proxy.some_method()  # Initializes on first access
    """

    def __init__(
        self,
        factory: Callable[[], T],
        async_init: Optional[Callable[[T], Awaitable[bool]]] = None,
        name: str = "subsystem",
        timeout: float = 30.0,
        priority: SubsystemPriority = SubsystemPriority.OPTIONAL,
        min_headroom_percent: float = 15.0,
    ):
        self._factory = factory
        self._async_init = async_init
        self._name = name
        self._timeout = timeout
        self._priority = priority
        self._min_headroom_percent = min_headroom_percent

        self._instance: Optional[T] = None
        self._initialized = False
        self._init_lock: Optional[asyncio.Lock] = None
        self._init_error: Optional[Exception] = None
        self._init_time: float = 0.0

    def _get_lock(self) -> asyncio.Lock:
        """Lazy-initialize the lock (must be in event loop context)."""
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        return self._init_lock

    async def _check_memory_ok(self) -> bool:
        """Check if there's enough memory headroom to initialize."""
        if not PSUTIL_AVAILABLE:
            return True  # Can't check, assume OK

        try:
            mem = psutil.virtual_memory()
            headroom_percent = 100.0 - mem.percent
            return headroom_percent >= self._min_headroom_percent
        except Exception:
            return True  # On error, proceed anyway

    async def _ensure_initialized(self) -> T:
        """
        Ensure the subsystem is initialized, creating it if necessary.

        Thread-safe: Uses asyncio.Lock for synchronization.
        """
        # Fast path: Already initialized
        if self._initialized and self._instance is not None:
            return self._instance

        # Check for previous failure
        if self._init_error is not None:
            raise RuntimeError(
                f"LazySubsystemProxy[{self._name}] previously failed: {self._init_error}"
            )

        lock = self._get_lock()
        async with lock:
            # Double-check under lock
            if self._initialized and self._instance is not None:
                return self._instance

            # Check memory headroom
            if not await self._check_memory_ok():
                # Trigger GC and retry once
                gc.collect()
                await asyncio.sleep(0.1)  # Let GC settle

                if not await self._check_memory_ok():
                    raise MemoryError(
                        f"Insufficient memory to initialize {self._name} "
                        f"(requires {self._min_headroom_percent}% headroom)"
                    )

            logger.info(f"[v138.0] Lazy-loading subsystem: {self._name}")
            start_time = time.perf_counter()

            try:
                # Create instance
                self._instance = self._factory()

                # Run async initialization if provided
                if self._async_init is not None:
                    await asyncio.wait_for(
                        self._async_init(self._instance),
                        timeout=self._timeout
                    )

                self._initialized = True
                self._init_time = time.perf_counter() - start_time

                logger.info(
                    f"[v138.0] Lazy-loaded {self._name} in {self._init_time:.2f}s"
                )
                return self._instance

            except asyncio.TimeoutError:
                self._init_error = TimeoutError(
                    f"Lazy initialization of {self._name} timed out after {self._timeout}s"
                )
                raise self._init_error
            except Exception as e:
                self._init_error = e
                logger.error(f"[v138.0] Failed to lazy-load {self._name}: {e}")
                raise

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the wrapped instance.

        Note: This is synchronous, so it blocks if the instance needs
        initialization. For truly non-blocking behavior, call
        _ensure_initialized() explicitly first.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # If already initialized, forward directly
        if self._instance is not None:
            return getattr(self._instance, name)

        # Return a coroutine that will initialize and forward
        async def lazy_forward(*args, **kwargs):
            instance = await self._ensure_initialized()
            attr = getattr(instance, name)
            if callable(attr):
                result = attr(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            return attr

        return lazy_forward

    @property
    def is_initialized(self) -> bool:
        """Check if the subsystem has been initialized."""
        return self._initialized

    @property
    def instance(self) -> Optional[T]:
        """Get the raw instance (may be None if not initialized)."""
        return self._instance


# =============================================================================
# v138.0: MEMORY GATE
# =============================================================================


class MemoryGate:
    """
    v138.0: Memory gate for staged initialization.

    Checks memory headroom and optionally triggers GC before allowing
    the next initialization stage to proceed.

    Features:
    - Non-blocking memory checks via thread pool
    - Configurable retry with exponential backoff
    - GC triggering with generation control
    - Memory trend analysis
    - Detailed logging for debugging
    """

    def __init__(self, config: StagedInitConfig):
        self._config = config
        self._thread_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._memory_history: List[MemorySnapshot] = []

    def _get_thread_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """Lazy-initialize thread pool for non-blocking I/O."""
        if self._thread_pool is None:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="MemoryGate"
            )
        return self._thread_pool

    def _sync_get_memory_snapshot(self) -> Optional[MemorySnapshot]:
        """Synchronously get memory snapshot (runs in thread pool)."""
        if not PSUTIL_AVAILABLE:
            return None

        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            process = psutil.Process()
            mem_info = process.memory_info()

            return MemorySnapshot(
                timestamp=time.time(),
                total_mb=mem.total / (1024 ** 2),
                available_mb=mem.available / (1024 ** 2),
                used_percent=mem.percent,
                process_rss_mb=mem_info.rss / (1024 ** 2),
                swap_used_mb=swap.used / (1024 ** 2),
            )
        except Exception as e:
            logger.debug(f"[MemoryGate] Failed to get snapshot: {e}")
            return None

    async def get_memory_snapshot(self) -> Optional[MemorySnapshot]:
        """Non-blocking memory snapshot acquisition."""
        if not PSUTIL_AVAILABLE:
            return None

        loop = asyncio.get_running_loop()
        pool = self._get_thread_pool()

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(pool, self._sync_get_memory_snapshot),
                timeout=2.0
            )
        except asyncio.TimeoutError:
            logger.warning("[MemoryGate] Memory snapshot timed out")
            return None

    def classify_memory_pressure(
        self,
        snapshot: Optional[MemorySnapshot]
    ) -> MemoryPressure:
        """Classify current memory pressure level."""
        if snapshot is None:
            return MemoryPressure.LOW  # Default if we can't measure

        used = snapshot.used_percent

        if used < 50.0:
            return MemoryPressure.MINIMAL
        elif used < 65.0:
            return MemoryPressure.LOW
        elif used < 80.0:
            return MemoryPressure.MODERATE
        elif used < 90.0:
            return MemoryPressure.HIGH
        else:
            return MemoryPressure.CRITICAL

    async def run_gc(self, generations: Optional[int] = None) -> float:
        """
        Run garbage collection in thread pool to avoid blocking.

        Returns: Time taken for GC in seconds.
        """
        gens = generations if generations is not None else self._config.gc_generations

        def _gc_in_thread():
            start = time.perf_counter()
            # Collect all generations up to the specified one
            for gen in range(gens + 1):
                gc.collect(gen)
            return time.perf_counter() - start

        loop = asyncio.get_running_loop()
        pool = self._get_thread_pool()

        try:
            gc_time = await asyncio.wait_for(
                loop.run_in_executor(pool, _gc_in_thread),
                timeout=self._config.gc_timeout
            )
            logger.debug(f"[MemoryGate] GC completed in {gc_time:.3f}s")
            return gc_time
        except asyncio.TimeoutError:
            logger.warning(f"[MemoryGate] GC timed out after {self._config.gc_timeout}s")
            return self._config.gc_timeout

    async def check_headroom(
        self,
        stage_name: str,
        required_percent: Optional[float] = None
    ) -> Tuple[bool, MemorySnapshot, MemoryPressure]:
        """
        Check if there's sufficient memory headroom to proceed.

        Args:
            stage_name: Name of the stage for logging
            required_percent: Override default minimum headroom

        Returns:
            Tuple of (can_proceed, memory_snapshot, pressure_level)
        """
        min_headroom = required_percent or self._config.min_headroom_percent

        snapshot = await self.get_memory_snapshot()
        if snapshot is None:
            # Can't measure, assume OK but log warning
            logger.warning(f"[MemoryGate] Cannot measure memory for {stage_name}")
            return True, MemorySnapshot(
                timestamp=time.time(),
                total_mb=0, available_mb=0, used_percent=0, process_rss_mb=0
            ), MemoryPressure.LOW

        pressure = self.classify_memory_pressure(snapshot)
        headroom = snapshot.headroom_percent

        # Store in history for trend analysis
        self._memory_history.append(snapshot)
        if len(self._memory_history) > 20:
            self._memory_history.pop(0)

        can_proceed = headroom >= min_headroom

        logger.info(
            f"[MemoryGate] {stage_name}: "
            f"headroom={headroom:.1f}% (min={min_headroom}%), "
            f"pressure={pressure.name}, "
            f"RSS={snapshot.process_rss_mb:.0f}MB"
        )

        return can_proceed, snapshot, pressure

    async def wait_for_headroom(
        self,
        stage_name: str,
        required_percent: Optional[float] = None
    ) -> Tuple[bool, MemorySnapshot, MemoryPressure]:
        """
        Wait for sufficient memory headroom, with GC and retries.

        Returns:
            Tuple of (success, final_snapshot, final_pressure)
        """
        retry_count = self._config.memory_gate_retry_count
        retry_delay = self._config.memory_gate_retry_delay

        for attempt in range(retry_count + 1):
            can_proceed, snapshot, pressure = await self.check_headroom(
                stage_name, required_percent
            )

            if can_proceed:
                return True, snapshot, pressure

            if attempt < retry_count:
                logger.warning(
                    f"[MemoryGate] {stage_name}: Insufficient headroom "
                    f"({snapshot.headroom_percent:.1f}%), "
                    f"running GC and retry {attempt + 1}/{retry_count}"
                )

                # Run GC to try to free memory
                await self.run_gc()

                # Wait with exponential backoff
                await asyncio.sleep(retry_delay * (1.5 ** attempt))

        logger.error(
            f"[MemoryGate] {stage_name}: Failed to achieve headroom after {retry_count} retries"
        )
        return False, snapshot, pressure

    def cleanup(self):
        """Clean up thread pool."""
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=False)
            self._thread_pool = None


@dataclass
class AGIRequest:
    """Unified request structure for AGI processing."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    modalities: List[str] = field(default_factory=list)  # text, image, audio, etc.
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Classification (filled during analysis)
    complexity: Optional[RequestComplexity] = None
    reasoning_requirement: Optional[ReasoningRequirement] = None
    required_models: List[str] = field(default_factory=list)

    # Timing
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.modalities and self.content:
            self.modalities = ["text"]


@dataclass
class AGIResponse:
    """Unified response structure from AGI processing."""

    request_id: str
    content: str
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    models_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Timing
    processing_time_ms: float = 0.0

    # Learning feedback
    feedback_recorded: bool = False


@dataclass
class SubsystemStatus:
    """Status of an AGI subsystem."""

    name: str
    initialized: bool = False
    healthy: bool = False
    last_check: float = 0.0
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AGIHubConfig:
    """Configuration for the AGI Integration Hub."""

    # Subsystem enablement
    enable_orchestrator: bool = True
    enable_reasoning: bool = True
    enable_learning: bool = True
    enable_multimodal: bool = True
    enable_hardware_optimization: bool = True

    # v93.12: Advanced AGI Models Configuration
    # These are OPTIONAL and non-critical - failures should NOT block startup
    enable_agi_models_v80: bool = True  # Enable v80.0 AGI models (model manager, continual learning, self-modification)
    agi_models_v80_timeout: float = 30.0  # Max time (seconds) to wait for v80.0 models initialization
    agi_models_v80_graceful_degradation: bool = True  # Continue even if v80.0 models fail

    # Reasoning settings
    default_reasoning_strategy: str = "chain_of_thought"
    enable_auto_reasoning: bool = True  # Auto-select strategy based on request
    max_reasoning_depth: int = 10
    reasoning_timeout_seconds: float = 30.0

    # Learning settings
    enable_experience_recording: bool = True
    learning_batch_size: int = 32
    min_feedback_for_update: int = 100

    # Hardware settings
    prefer_neural_engine: bool = True
    prefer_metal_gpu: bool = True
    enable_memory_mapping: bool = True

    # v93.12: Per-subsystem timeout configuration
    subsystem_init_timeout: float = 60.0  # Default timeout for any subsystem initialization
    parallel_init_timeout: float = 120.0  # Overall timeout for parallel initialization

    # ==========================================================================
    # v138.0: MEMORY-AWARE STAGED INITIALIZATION CONFIG
    # ==========================================================================

    # Enable staged initialization (replaces Big Bang parallel init)
    enable_staged_init: bool = True

    # Memory thresholds
    min_memory_headroom_percent: float = 20.0  # Minimum free memory to proceed
    slim_mode_threshold_percent: float = 15.0  # Enter slim mode below this
    critical_memory_threshold_percent: float = 10.0  # Emergency mode below this

    # Stage timing
    stage_timeout: float = 45.0  # Max time per stage
    pre_flight_timeout: float = 10.0  # Pre-flight checks timeout

    # GC settings
    gc_between_stages: bool = True  # Run GC between stages
    gc_generations: int = 2  # GC generations to collect (0-2)

    # Adaptive behavior
    enable_slim_mode: bool = True  # Auto-enable slim mode in low memory
    enable_lazy_loading: bool = True  # Lazy-load optional subsystems
    defer_heavy_subsystems: bool = True  # Defer Stage 3 on memory pressure

    # OOM Protection
    enable_oom_protection: bool = True  # Initialize OOM engine before heavy loads
    oom_warning_threshold: float = 85.0  # OOM warning at this % memory usage
    oom_critical_threshold: float = 95.0  # OOM critical at this % memory usage

    # Subsystem priority assignment (for adaptive loading)
    subsystem_priorities: Dict[str, SubsystemPriority] = field(default_factory=lambda: {
        "hardware": SubsystemPriority.CRITICAL,
        "orchestrator": SubsystemPriority.CRITICAL,
        "reasoning": SubsystemPriority.IMPORTANT,
        "learning": SubsystemPriority.IMPORTANT,
        "multimodal": SubsystemPriority.OPTIONAL,
        "agi_models_v80": SubsystemPriority.OPTIONAL,
    })

    # ==========================================================================
    # END v138.0 CONFIG
    # ==========================================================================

    # Analysis settings
    complexity_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "trivial": ["what is", "who is", "when did", "define"],
        "simple": ["how do i", "can you", "please"],
        "moderate": ["explain", "compare", "analyze"],
        "complex": ["why does", "what would happen if", "design", "plan"],
        "expert": ["optimize", "architect", "reason about", "prove"],
    })

    # Routing settings
    model_capabilities: Dict[str, Set[str]] = field(default_factory=lambda: {
        "action": {"planning", "execution", "steps"},
        "meta-reasoner": {"strategy", "approach", "meta"},
        "causal": {"why", "because", "cause", "effect"},
        "world-model": {"physics", "common sense", "reality"},
        "memory": {"remember", "recall", "history"},
        "goal-inference": {"intent", "goal", "objective"},
        "self-model": {"capability", "limitation", "can i"},
    })


# =============================================================================
# REQUEST ANALYZER
# =============================================================================


class RequestAnalyzer:
    """
    Analyzes incoming requests to determine complexity and routing.

    Uses heuristics, keyword matching, and optionally ML classification
    to determine the best processing strategy.
    """

    def __init__(self, config: AGIHubConfig) -> None:
        self._config = config
        self._complexity_cache: Dict[str, RequestComplexity] = {}

    async def analyze(self, request: AGIRequest) -> AGIRequest:
        """Analyze and classify a request."""
        # Classify complexity
        request.complexity = await self._classify_complexity(request.content)

        # Determine reasoning requirement
        request.reasoning_requirement = await self._determine_reasoning(
            request.content, request.complexity
        )

        # Identify required models
        request.required_models = await self._identify_models(request.content)

        return request

    async def _classify_complexity(self, content: str) -> RequestComplexity:
        """Classify request complexity based on content analysis."""
        content_lower = content.lower()

        # Check keyword patterns
        for complexity, keywords in self._config.complexity_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return RequestComplexity[complexity.upper()]

        # Heuristics based on structure
        sentence_count = content.count('.') + content.count('?') + content.count('!')
        word_count = len(content.split())

        if word_count < 10:
            return RequestComplexity.SIMPLE
        elif word_count < 30 and sentence_count < 3:
            return RequestComplexity.MODERATE
        elif word_count < 100:
            return RequestComplexity.COMPLEX
        else:
            return RequestComplexity.EXPERT

    async def _determine_reasoning(
        self,
        content: str,
        complexity: RequestComplexity
    ) -> ReasoningRequirement:
        """Determine what type of reasoning is needed."""
        content_lower = content.lower()

        # Causal reasoning indicators
        if any(w in content_lower for w in ["why", "because", "cause", "effect", "reason"]):
            return ReasoningRequirement.CAUSAL

        # Planning indicators
        if any(w in content_lower for w in ["plan", "steps", "how to", "design", "create"]):
            return ReasoningRequirement.PLANNING

        # Meta-cognitive indicators
        if any(w in content_lower for w in ["think about", "approach", "strategy", "best way"]):
            return ReasoningRequirement.META

        # Complexity-based defaults
        if complexity == RequestComplexity.TRIVIAL:
            return ReasoningRequirement.NONE
        elif complexity == RequestComplexity.SIMPLE:
            return ReasoningRequirement.CHAIN
        elif complexity in (RequestComplexity.COMPLEX, RequestComplexity.EXPERT):
            return ReasoningRequirement.TREE
        else:
            return ReasoningRequirement.CHAIN

    async def _identify_models(self, content: str) -> List[str]:
        """Identify which AGI models should be involved."""
        content_lower = content.lower()
        required = []

        for model, keywords in self._config.model_capabilities.items():
            if any(kw in content_lower for kw in keywords):
                required.append(model)

        # Always include meta-reasoner for complex requests
        if len(required) > 2 and "meta-reasoner" not in required:
            required.append("meta-reasoner")

        return required


# =============================================================================
# AGI INTEGRATION HUB
# =============================================================================


class AGIIntegrationHub:
    """
    Central integration hub for all AGI subsystems.

    This is the main entry point for AGI-enhanced inference, coordinating:
    - Request analysis and routing
    - Reasoning strategy selection and execution
    - Multi-model orchestration
    - Experience recording for learning
    - Hardware optimization
    """

    def __init__(self, config: Optional[AGIHubConfig] = None) -> None:
        self._config = config or AGIHubConfig()
        self._analyzer = RequestAnalyzer(self._config)

        # Subsystem instances (lazy-loaded)
        self._orchestrator: Optional[Any] = None
        self._reasoning_engine: Optional[Any] = None
        self._learning_engine: Optional[Any] = None
        self._multimodal_engine: Optional[Any] = None
        self._hardware_optimizer: Optional[Any] = None
        # v80.0 AGI Models
        self._agi_model_manager: Optional[Any] = None
        self._continual_learner: Optional[Any] = None
        self._self_modifier: Optional[Any] = None
        self._knowledge_distiller: Optional[Any] = None
        self._active_learner: Optional[Any] = None
        self._nas_engine: Optional[Any] = None

        # State
        self._initialized = False
        self._subsystem_status: Dict[AGISubsystem, SubsystemStatus] = {}

        # Cognitive state shared across subsystems
        self._cognitive_state: Optional[Any] = None

        # Metrics
        self._request_count = 0
        self._total_processing_time = 0.0
        self._reasoning_usage: Dict[str, int] = {}
        self._model_usage: Dict[str, int] = {}

        # Lock for thread-safe initialization
        self._init_lock = asyncio.Lock()

        # =======================================================================
        # v138.0: MEMORY-AWARE STAGED INITIALIZATION STATE
        # =======================================================================

        # Memory gate for staged initialization
        self._staged_init_config = StagedInitConfig.from_env()
        self._memory_gate: Optional[MemoryGate] = None

        # OOM Protection Engine (initialized early in pre-flight)
        self._oom_engine: Optional[Any] = None

        # Initialization state tracking
        self._current_stage: InitStage = InitStage.PRE_FLIGHT
        self._stage_results: List[StageResult] = []
        self._memory_pressure: MemoryPressure = MemoryPressure.LOW
        self._slim_mode: bool = False
        self._deferred_subsystems: Set[str] = set()

        # Lazy loading proxies for optional subsystems
        self._lazy_proxies: Dict[str, LazySubsystemProxy] = {}

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    async def initialize(self) -> bool:
        """
        Initialize all AGI subsystems with memory-aware staged loading.

        v138.0: Complete rewrite with Memory-Aware Staged Initialization.

        Fixes OOM (Exit Code -9) by replacing unbounded parallel initialization
        with a staged pipeline that includes:

        1. PRE-FLIGHT: Memory assessment + OOM Protection setup
        2. STAGE 1 (Foundation): Hardware + Orchestrator (critical, light)
        3. STAGE 2 (Reasoning): Reasoning + Learning (moderate memory)
        4. STAGE 3 (Heavy): Multimodal + AGI v80 (heavy, optional)

        Features:
        - Memory gates between stages with GC
        - Adaptive slim mode for low-memory environments
        - Lazy loading proxies for optional subsystems
        - OOM Protection initialized BEFORE heavy loads
        - Dynamic subsystem deferral based on memory pressure
        """
        async with self._init_lock:
            if self._initialized:
                return True

            # Use staged init if enabled, otherwise fall back to legacy
            if self._config.enable_staged_init:
                return await self._initialize_staged()
            else:
                return await self._initialize_legacy()

    async def _initialize_staged(self) -> bool:
        """
        v138.0: Memory-aware staged initialization.

        Implements rolling start pattern to prevent memory spikes.
        """
        init_start = time.time()
        total_initialized = 0
        total_failed = 0
        total_deferred = 0

        logger.info("=" * 70)
        logger.info("v138.0: AGI Integration Hub - Memory-Aware Staged Initialization")
        logger.info("=" * 70)

        try:
            # =================================================================
            # PRE-FLIGHT PHASE
            # =================================================================
            self._current_stage = InitStage.PRE_FLIGHT

            pre_flight_result = await self._run_pre_flight_stage()
            self._stage_results.append(pre_flight_result)

            if not pre_flight_result.success:
                logger.error("[v138.0] Pre-flight checks failed - aborting initialization")
                return False

            # Check if we should use slim mode
            if self._slim_mode:
                logger.warning(
                    "[v138.0] SLIM MODE ACTIVE - deferring heavy subsystems"
                )

            # =================================================================
            # STAGE 1: FOUNDATION (Critical subsystems)
            # =================================================================
            self._current_stage = InitStage.STAGE_1_FOUNDATION

            stage1_result = await self._run_stage_1_foundation()
            self._stage_results.append(stage1_result)
            total_initialized += len(stage1_result.subsystems_initialized)
            total_failed += len(stage1_result.subsystems_failed)

            # Stage 1 is critical - if it fails completely, abort
            if not stage1_result.success and not stage1_result.subsystems_initialized:
                logger.error("[v138.0] Stage 1 (Foundation) failed - aborting")
                return False

            # Memory gate before Stage 2
            if self._config.gc_between_stages and self._memory_gate:
                await self._memory_gate.run_gc()

            if self._memory_gate:
                can_proceed, snapshot, pressure = await self._memory_gate.wait_for_headroom(
                    "Stage 2 Gate"
                )
            else:
                can_proceed, snapshot, pressure = True, None, MemoryPressure.LOW
            self._memory_pressure = pressure

            if not can_proceed:
                logger.warning("[v138.0] Memory gate blocked Stage 2 - minimal mode")
                self._slim_mode = True

            # =================================================================
            # STAGE 2: REASONING (Important but deferrable)
            # =================================================================
            self._current_stage = InitStage.STAGE_2_REASONING

            if self._memory_pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL):
                logger.warning(
                    f"[v138.0] High memory pressure ({pressure.name}) - "
                    "deferring Stage 2 subsystems for lazy loading"
                )
                self._deferred_subsystems.update(["reasoning", "learning"])
                self._setup_lazy_proxies_stage_2()
                stage2_result = StageResult(
                    stage=InitStage.STAGE_2_REASONING,
                    success=True,
                    subsystems_initialized=[],
                    subsystems_failed=[],
                    subsystems_deferred=["reasoning", "learning"],
                    elapsed_seconds=0.0,
                )
            else:
                stage2_result = await self._run_stage_2_reasoning()

            self._stage_results.append(stage2_result)
            total_initialized += len(stage2_result.subsystems_initialized)
            total_failed += len(stage2_result.subsystems_failed)
            total_deferred += len(stage2_result.subsystems_deferred)

            # Memory gate before Stage 3
            if self._config.gc_between_stages and self._memory_gate:
                await self._memory_gate.run_gc()

            if self._memory_gate:
                can_proceed, snapshot, pressure = await self._memory_gate.wait_for_headroom(
                    "Stage 3 Gate"
                )
            else:
                can_proceed, snapshot, pressure = True, None, MemoryPressure.LOW
            self._memory_pressure = pressure

            # =================================================================
            # STAGE 3: HEAVY (Optional, always deferrable)
            # =================================================================
            self._current_stage = InitStage.STAGE_3_HEAVY

            # In slim mode or high pressure, defer Stage 3 for lazy loading
            defer_stage_3 = (
                self._slim_mode
                or self._memory_pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL)
                or (self._config.defer_heavy_subsystems and not can_proceed)
            )

            if defer_stage_3:
                logger.info(
                    "[v138.0] Deferring Stage 3 (Heavy) subsystems for lazy loading"
                )
                self._deferred_subsystems.update(["multimodal", "agi_models_v80"])
                self._setup_lazy_proxies_stage_3()
                stage3_result = StageResult(
                    stage=InitStage.STAGE_3_HEAVY,
                    success=True,
                    subsystems_initialized=[],
                    subsystems_failed=[],
                    subsystems_deferred=["multimodal", "agi_models_v80"],
                    elapsed_seconds=0.0,
                )
            else:
                stage3_result = await self._run_stage_3_heavy()

            self._stage_results.append(stage3_result)
            total_initialized += len(stage3_result.subsystems_initialized)
            total_failed += len(stage3_result.subsystems_failed)
            total_deferred += len(stage3_result.subsystems_deferred)

            # =================================================================
            # COMPLETION
            # =================================================================
            self._current_stage = InitStage.COMPLETED
            elapsed = time.time() - init_start

            # Mark as initialized if we have at least one working subsystem
            self._initialized = total_initialized > 0

            # Calculate memory stats
            total_memory_delta = sum(
                r.memory_delta_mb for r in self._stage_results if r.memory_delta_mb
            )

            logger.info("=" * 70)
            logger.info(
                f"[v138.0] AGI Hub initialized in {elapsed:.1f}s: "
                f"{total_initialized} active, {total_deferred} deferred, {total_failed} failed"
            )
            if total_memory_delta > 0:
                logger.info(f"[v138.0] Total memory growth: +{total_memory_delta:.0f}MB")
            if self._slim_mode:
                logger.info("[v138.0] Running in SLIM MODE - some features lazy-loaded")
            logger.info("=" * 70)

            return self._initialized

        except Exception as e:
            logger.error(f"[v138.0] Staged initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _run_pre_flight_stage(self) -> StageResult:
        """
        v138.0: Pre-flight checks and OOM Protection setup.

        This stage runs BEFORE any heavy subsystem loading to:
        1. Assess available memory
        2. Initialize OOM Protection Engine
        3. Determine slim mode requirements
        4. Set up memory gate infrastructure
        """
        stage_start = time.time()
        initialized = []
        failed = []

        logger.info("[v138.0] === PRE-FLIGHT PHASE ===")

        try:
            # Initialize memory gate
            self._memory_gate = MemoryGate(self._staged_init_config)

            # Get initial memory snapshot
            memory_before = await self._safe_get_memory_snapshot()
            self._memory_pressure = self._memory_gate.classify_memory_pressure(memory_before)

            if memory_before:
                logger.info(
                    f"[v138.0] Initial memory: "
                    f"{memory_before.used_percent:.1f}% used, "
                    f"{memory_before.headroom_mb:.0f}MB available, "
                    f"RSS={memory_before.process_rss_mb:.0f}MB"
                )

            # Determine if we need slim mode
            if memory_before and memory_before.headroom_percent < self._config.slim_mode_threshold_percent:
                self._slim_mode = True
                logger.warning(
                    f"[v138.0] Low memory detected ({memory_before.headroom_percent:.1f}% headroom) "
                    f"- enabling SLIM MODE"
                )

            # Critical memory - we might not be able to proceed at all
            if self._memory_pressure == MemoryPressure.CRITICAL:
                logger.error(
                    "[v138.0] CRITICAL memory pressure detected - running emergency GC"
                )
                await self._memory_gate.run_gc(generations=2)
                # Recheck
                memory_before = await self._safe_get_memory_snapshot()
                self._memory_pressure = self._memory_gate.classify_memory_pressure(memory_before)

            # Initialize OOM Protection Engine FIRST (before heavy loads)
            if self._config.enable_oom_protection:
                try:
                    from jarvis_prime.core.reliability_engines import (
                        OOMProtectionEngine,
                        OOMConfig,
                    )

                    oom_config = OOMConfig(
                        memory_threshold_percent=self._config.oom_critical_threshold,
                        warning_threshold_percent=self._config.oom_warning_threshold,
                        check_interval=5.0,
                        enable_aggressive_gc=True,
                    )

                    self._oom_engine = OOMProtectionEngine(
                        config=oom_config,
                        on_warning=self._on_oom_warning,
                        on_critical=self._on_oom_critical,
                        on_emergency=self._on_oom_emergency,
                    )

                    # Start monitoring BEFORE heavy loads
                    await self._oom_engine.start_monitoring()
                    initialized.append("oom_protection")
                    logger.info("[v138.0] OOM Protection Engine active")

                except ImportError:
                    logger.warning("[v138.0] OOM Protection Engine not available")
                except Exception as e:
                    logger.warning(f"[v138.0] OOM Protection init failed: {e}")
                    failed.append("oom_protection")

            memory_after = await self._safe_get_memory_snapshot()

            return StageResult(
                stage=InitStage.PRE_FLIGHT,
                success=True,  # Pre-flight always succeeds (it's just assessment)
                subsystems_initialized=initialized,
                subsystems_failed=failed,
                subsystems_deferred=[],
                elapsed_seconds=time.time() - stage_start,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        except Exception as e:
            logger.error(f"[v138.0] Pre-flight failed: {e}")
            return StageResult(
                stage=InitStage.PRE_FLIGHT,
                success=False,
                subsystems_initialized=initialized,
                subsystems_failed=failed,
                subsystems_deferred=[],
                elapsed_seconds=time.time() - stage_start,
                error=str(e),
            )

    async def _run_stage_1_foundation(self) -> StageResult:
        """
        v138.0: Stage 1 - Foundation subsystems (Hardware + Orchestrator).

        These are CRITICAL and relatively lightweight.
        We initialize them with limited parallelism (max 2 concurrent).
        """
        stage_start = time.time()
        initialized = []
        failed = []
        deferred = []

        logger.info("[v138.0] === STAGE 1: FOUNDATION ===")

        memory_before = await self._safe_get_memory_snapshot()

        try:
            subsystem_timeout = self._config.subsystem_init_timeout
            tasks = {}

            # Hardware Optimizer (critical)
            if self._config.enable_hardware_optimization:
                tasks["hardware"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_hardware(),
                        subsystem_timeout,
                        "hardware"
                    )
                )

            # AGI Orchestrator (critical)
            if self._config.enable_orchestrator:
                tasks["orchestrator"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_orchestrator(),
                        subsystem_timeout,
                        "orchestrator"
                    )
                )

            # Wait for Stage 1 tasks (limited parallelism - only 2 tasks)
            if tasks:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks.values(), return_exceptions=True),
                        timeout=self._config.stage_timeout
                    )

                    for (name, task), result in zip(tasks.items(), results):
                        if result is True:
                            initialized.append(name)
                            logger.info(f"[v138.0] Stage 1: {name} ✓")
                        elif isinstance(result, Exception):
                            failed.append(name)
                            logger.warning(f"[v138.0] Stage 1: {name} ✗ ({result})")
                        else:
                            failed.append(name)
                            logger.warning(f"[v138.0] Stage 1: {name} returned {result}")

                except asyncio.TimeoutError:
                    logger.error(f"[v138.0] Stage 1 timed out after {self._config.stage_timeout}s")
                    for name, task in tasks.items():
                        if not task.done():
                            task.cancel()
                            failed.append(name)

            memory_after = await self._safe_get_memory_snapshot()

            return StageResult(
                stage=InitStage.STAGE_1_FOUNDATION,
                success=len(initialized) > 0,
                subsystems_initialized=initialized,
                subsystems_failed=failed,
                subsystems_deferred=deferred,
                elapsed_seconds=time.time() - stage_start,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        except Exception as e:
            logger.error(f"[v138.0] Stage 1 failed: {e}")
            return StageResult(
                stage=InitStage.STAGE_1_FOUNDATION,
                success=False,
                subsystems_initialized=initialized,
                subsystems_failed=failed,
                subsystems_deferred=deferred,
                elapsed_seconds=time.time() - stage_start,
                error=str(e),
            )

    async def _run_stage_2_reasoning(self) -> StageResult:
        """
        v138.0: Stage 2 - Reasoning subsystems (Reasoning + Learning).

        These are IMPORTANT but can be deferred if memory is tight.
        """
        stage_start = time.time()
        initialized = []
        failed = []
        deferred = []

        logger.info("[v138.0] === STAGE 2: REASONING ===")

        memory_before = await self._safe_get_memory_snapshot()

        try:
            subsystem_timeout = self._config.subsystem_init_timeout
            tasks = {}

            # Reasoning Engine
            if self._config.enable_reasoning:
                tasks["reasoning"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_reasoning(),
                        subsystem_timeout,
                        "reasoning"
                    )
                )

            # Learning Engine
            if self._config.enable_learning:
                tasks["learning"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_learning(),
                        subsystem_timeout,
                        "learning"
                    )
                )

            if tasks:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*tasks.values(), return_exceptions=True),
                        timeout=self._config.stage_timeout
                    )

                    for (name, task), result in zip(tasks.items(), results):
                        if result is True:
                            initialized.append(name)
                            logger.info(f"[v138.0] Stage 2: {name} ✓")
                        elif isinstance(result, Exception):
                            failed.append(name)
                            logger.warning(f"[v138.0] Stage 2: {name} ✗ ({result})")
                        else:
                            failed.append(name)
                            logger.warning(f"[v138.0] Stage 2: {name} returned {result}")

                except asyncio.TimeoutError:
                    logger.error(f"[v138.0] Stage 2 timed out after {self._config.stage_timeout}s")
                    for name, task in tasks.items():
                        if not task.done():
                            task.cancel()
                            failed.append(name)

            memory_after = await self._safe_get_memory_snapshot()

            return StageResult(
                stage=InitStage.STAGE_2_REASONING,
                success=True,  # Stage 2 is not critical
                subsystems_initialized=initialized,
                subsystems_failed=failed,
                subsystems_deferred=deferred,
                elapsed_seconds=time.time() - stage_start,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        except Exception as e:
            logger.error(f"[v138.0] Stage 2 failed: {e}")
            return StageResult(
                stage=InitStage.STAGE_2_REASONING,
                success=True,  # Stage 2 failures are non-fatal
                subsystems_initialized=initialized,
                subsystems_failed=failed,
                subsystems_deferred=deferred,
                elapsed_seconds=time.time() - stage_start,
                error=str(e),
            )

    async def _run_stage_3_heavy(self) -> StageResult:
        """
        v138.0: Stage 3 - Heavy subsystems (Multimodal + AGI v80).

        These are OPTIONAL and can always be lazy-loaded.
        Only initialize if memory permits.
        """
        stage_start = time.time()
        initialized = []
        failed = []
        deferred = []

        logger.info("[v138.0] === STAGE 3: HEAVY ===")

        memory_before = await self._safe_get_memory_snapshot()

        try:
            subsystem_timeout = self._config.subsystem_init_timeout

            # Initialize one at a time to prevent memory spike
            # (NO parallel initialization in Stage 3)

            # Multimodal Engine
            if self._config.enable_multimodal:
                logger.info("[v138.0] Stage 3: Initializing multimodal...")
                try:
                    result = await asyncio.wait_for(
                        self._init_multimodal(),
                        timeout=subsystem_timeout
                    )
                    if result:
                        initialized.append("multimodal")
                        logger.info("[v138.0] Stage 3: multimodal ✓")
                    else:
                        failed.append("multimodal")
                except Exception as e:
                    failed.append("multimodal")
                    logger.warning(f"[v138.0] Stage 3: multimodal ✗ ({e})")

                # Memory check before next subsystem
                can_continue, _, pressure = await self._safe_check_headroom(
                    "Post-Multimodal Check"
                )
                if not can_continue:
                    logger.warning("[v138.0] Memory gate closed - deferring remaining Stage 3")
                    if self._config.enable_agi_models_v80:
                        deferred.append("agi_models_v80")
                        self._deferred_subsystems.add("agi_models_v80")
                        self._setup_lazy_proxy_agi_v80()

                    memory_after = await self._safe_get_memory_snapshot()
                    return StageResult(
                        stage=InitStage.STAGE_3_HEAVY,
                        success=True,
                        subsystems_initialized=initialized,
                        subsystems_failed=failed,
                        subsystems_deferred=deferred,
                        elapsed_seconds=time.time() - stage_start,
                        memory_before=memory_before,
                        memory_after=memory_after,
                    )

            # AGI v80 Models (heaviest)
            if self._config.enable_agi_models_v80:
                logger.info("[v138.0] Stage 3: Initializing AGI v80 models...")
                try:
                    result = await asyncio.wait_for(
                        self._init_agi_models_v80(),
                        timeout=self._config.agi_models_v80_timeout
                    )
                    if result:
                        initialized.append("agi_models_v80")
                        logger.info("[v138.0] Stage 3: agi_models_v80 ✓")
                    else:
                        failed.append("agi_models_v80")
                except Exception as e:
                    failed.append("agi_models_v80")
                    logger.warning(f"[v138.0] Stage 3: agi_models_v80 ✗ ({e})")

            memory_after = await self._safe_get_memory_snapshot()

            return StageResult(
                stage=InitStage.STAGE_3_HEAVY,
                success=True,  # Stage 3 is optional
                subsystems_initialized=initialized,
                subsystems_failed=failed,
                subsystems_deferred=deferred,
                elapsed_seconds=time.time() - stage_start,
                memory_before=memory_before,
                memory_after=memory_after,
            )

        except Exception as e:
            logger.error(f"[v138.0] Stage 3 failed: {e}")
            return StageResult(
                stage=InitStage.STAGE_3_HEAVY,
                success=True,  # Stage 3 failures are non-fatal
                subsystems_initialized=initialized,
                subsystems_failed=failed,
                subsystems_deferred=deferred,
                elapsed_seconds=time.time() - stage_start,
                error=str(e),
            )

    # -------------------------------------------------------------------------
    # LAZY LOADING PROXIES
    # -------------------------------------------------------------------------

    def _setup_lazy_proxies_stage_2(self) -> None:
        """Set up lazy loading proxies for Stage 2 subsystems."""
        logger.info("[v138.0] Setting up lazy proxies for Stage 2 subsystems")

        if self._config.enable_reasoning and "reasoning" not in self._lazy_proxies:
            self._lazy_proxies["reasoning"] = LazySubsystemProxy(
                factory=self._create_reasoning_engine,
                async_init=lambda e: e.initialize(),
                name="reasoning_engine",
                timeout=self._config.subsystem_init_timeout,
                priority=SubsystemPriority.IMPORTANT,
            )

        if self._config.enable_learning and "learning" not in self._lazy_proxies:
            self._lazy_proxies["learning"] = LazySubsystemProxy(
                factory=self._create_learning_engine,
                async_init=lambda e: e.initialize(),
                name="learning_engine",
                timeout=self._config.subsystem_init_timeout,
                priority=SubsystemPriority.IMPORTANT,
            )

    def _setup_lazy_proxies_stage_3(self) -> None:
        """Set up lazy loading proxies for Stage 3 subsystems."""
        logger.info("[v138.0] Setting up lazy proxies for Stage 3 subsystems")

        if self._config.enable_multimodal and "multimodal" not in self._lazy_proxies:
            self._lazy_proxies["multimodal"] = LazySubsystemProxy(
                factory=self._create_multimodal_engine,
                async_init=lambda e: e.initialize(),
                name="multimodal_engine",
                timeout=self._config.subsystem_init_timeout,
                priority=SubsystemPriority.OPTIONAL,
            )

        if self._config.enable_agi_models_v80:
            self._setup_lazy_proxy_agi_v80()

    def _setup_lazy_proxy_agi_v80(self) -> None:
        """Set up lazy loading proxy for AGI v80 models."""
        if "agi_models_v80" not in self._lazy_proxies:
            # AGI v80 is a composite - we'll lazy-load its components
            logger.debug("[v138.0] AGI v80 will be lazy-loaded on first access")
            self._lazy_proxies["agi_models_v80"] = LazySubsystemProxy(
                factory=lambda: None,  # Placeholder - v80 has multiple components
                async_init=self._lazy_init_agi_v80,
                name="agi_models_v80",
                timeout=self._config.agi_models_v80_timeout,
                priority=SubsystemPriority.OPTIONAL,
            )

    async def _lazy_init_agi_v80(self, _: Any) -> bool:
        """Lazy initialization of AGI v80 models."""
        return await self._init_agi_models_v80()

    def _create_reasoning_engine(self) -> Any:
        """Factory for reasoning engine."""
        from jarvis_prime.core.reasoning_engine import ReasoningEngine
        return ReasoningEngine()

    def _create_learning_engine(self) -> Any:
        """Factory for learning engine."""
        from jarvis_prime.core.continuous_learning import ContinuousLearningEngine
        return ContinuousLearningEngine()

    def _create_multimodal_engine(self) -> Any:
        """Factory for multimodal engine."""
        from jarvis_prime.core.multimodal_fusion import MultiModalFusionEngine
        return MultiModalFusionEngine()

    # -------------------------------------------------------------------------
    # MEMORY HELPERS
    # -------------------------------------------------------------------------

    async def _safe_get_memory_snapshot(self) -> Optional[MemorySnapshot]:
        """Safely get memory snapshot, returning None if unavailable."""
        if self._memory_gate:
            return await self._memory_gate.get_memory_snapshot()
        return None

    async def _safe_check_headroom(
        self, stage_name: str
    ) -> Tuple[bool, Optional[MemorySnapshot], MemoryPressure]:
        """Safely check memory headroom, returning safe defaults if unavailable."""
        if self._memory_gate:
            return await self._memory_gate.check_headroom(stage_name)
        return True, None, MemoryPressure.LOW

    # -------------------------------------------------------------------------
    # OOM PROTECTION CALLBACKS
    # -------------------------------------------------------------------------

    async def _on_oom_warning(self) -> None:
        """Called when memory usage hits warning threshold."""
        logger.warning("[v138.0] OOM Warning - triggering subsystem GC")
        # Could pause background tasks, reduce cache sizes, etc.

    async def _on_oom_critical(self) -> None:
        """Called when memory usage hits critical threshold."""
        logger.error("[v138.0] OOM Critical - aggressive memory reduction")
        # Could stop non-essential subsystems

    async def _on_oom_emergency(self) -> None:
        """Called when memory usage hits emergency threshold."""
        logger.critical("[v138.0] OOM Emergency - survival mode")
        # Last resort - could restart subsystems, clear caches entirely

    # -------------------------------------------------------------------------
    # LEGACY INITIALIZATION (fallback)
    # -------------------------------------------------------------------------

    async def _initialize_legacy(self) -> bool:
        """
        Legacy parallel initialization (v93.12 behavior).

        Used when enable_staged_init=False for backwards compatibility.
        WARNING: This can cause OOM on memory-constrained systems.
        """
        logger.warning(
            "[v138.0] Using legacy parallel initialization - "
            "enable_staged_init=False. This may cause OOM."
        )

        init_start = time.time()

        try:
            init_tasks: Dict[str, asyncio.Task] = {}
            subsystem_timeout = self._config.subsystem_init_timeout

            if self._config.enable_hardware_optimization:
                init_tasks["hardware"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_hardware(),
                        subsystem_timeout,
                        "hardware_optimizer"
                    )
                )

            if self._config.enable_orchestrator:
                init_tasks["orchestrator"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_orchestrator(),
                        subsystem_timeout,
                        "orchestrator"
                    )
                )

            if self._config.enable_reasoning:
                init_tasks["reasoning"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_reasoning(),
                        subsystem_timeout,
                        "reasoning"
                    )
                )

            if self._config.enable_learning:
                init_tasks["learning"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_learning(),
                        subsystem_timeout,
                        "learning"
                    )
                )

            if self._config.enable_multimodal:
                init_tasks["multimodal"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_multimodal(),
                        subsystem_timeout,
                        "multimodal"
                    )
                )

            if self._config.enable_agi_models_v80:
                init_tasks["agi_models_v80"] = asyncio.create_task(
                    self._init_with_timeout(
                        self._init_agi_models_v80(),
                        self._config.agi_models_v80_timeout,
                        "agi_models_v80"
                    )
                )

            if init_tasks:
                results = await asyncio.wait_for(
                    asyncio.gather(*init_tasks.values(), return_exceptions=True),
                    timeout=self._config.parallel_init_timeout
                )

                success_count = sum(1 for r in results if r is True)
                elapsed = time.time() - init_start
                self._initialized = success_count > 0

                logger.info(
                    f"AGI Hub (legacy) initialized in {elapsed:.1f}s: "
                    f"{success_count}/{len(init_tasks)} subsystems"
                )
                return self._initialized

            return False

        except Exception as e:
            logger.error(f"Legacy initialization failed: {e}")
            return False

    async def _init_with_timeout(
        self,
        coro,
        timeout: float,
        name: str
    ) -> bool:
        """
        Wrap a coroutine with timeout protection.

        v93.12: Ensures no subsystem can hang the entire initialization.
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Subsystem {name} timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Subsystem {name} failed: {e}")
            raise

    async def _init_hardware(self) -> bool:
        """Initialize hardware optimization subsystem."""
        try:
            from jarvis_prime.core.apple_silicon_optimizer import AppleSiliconOptimizer

            self._hardware_optimizer = AppleSiliconOptimizer()
            success = await self._hardware_optimizer.initialize()

            self._subsystem_status[AGISubsystem.HARDWARE] = SubsystemStatus(
                name="hardware_optimizer",
                initialized=success,
                healthy=success,
                last_check=time.time(),
            )

            if success:
                logger.info("Hardware optimizer initialized successfully")

            return success

        except ImportError:
            logger.warning("AppleSiliconOptimizer not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize hardware optimizer: {e}")
            self._subsystem_status[AGISubsystem.HARDWARE] = SubsystemStatus(
                name="hardware_optimizer",
                initialized=False,
                healthy=False,
                error=str(e),
            )
            return False

    async def _init_orchestrator(self) -> bool:
        """Initialize AGI orchestrator subsystem."""
        try:
            from jarvis_prime.core.agi_models import AGIOrchestrator, CognitiveState

            self._cognitive_state = CognitiveState()
            self._orchestrator = AGIOrchestrator()
            success = await self._orchestrator.initialize()

            self._subsystem_status[AGISubsystem.ORCHESTRATOR] = SubsystemStatus(
                name="agi_orchestrator",
                initialized=success,
                healthy=success,
                last_check=time.time(),
            )

            if success:
                logger.info("AGI Orchestrator initialized successfully")

            return success

        except ImportError:
            logger.warning("AGIOrchestrator not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize AGI orchestrator: {e}")
            self._subsystem_status[AGISubsystem.ORCHESTRATOR] = SubsystemStatus(
                name="agi_orchestrator",
                initialized=False,
                healthy=False,
                error=str(e),
            )
            return False

    async def _init_reasoning(self) -> bool:
        """Initialize reasoning engine subsystem."""
        try:
            from jarvis_prime.core.reasoning_engine import ReasoningEngine

            self._reasoning_engine = ReasoningEngine()
            await self._reasoning_engine.initialize()

            self._subsystem_status[AGISubsystem.REASONING] = SubsystemStatus(
                name="reasoning_engine",
                initialized=True,
                healthy=True,
                last_check=time.time(),
            )

            logger.info("Reasoning Engine initialized successfully")
            return True

        except ImportError:
            logger.warning("ReasoningEngine not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize reasoning engine: {e}")
            self._subsystem_status[AGISubsystem.REASONING] = SubsystemStatus(
                name="reasoning_engine",
                initialized=False,
                healthy=False,
                error=str(e),
            )
            return False

    async def _init_learning(self) -> bool:
        """Initialize continuous learning subsystem."""
        try:
            from jarvis_prime.core.continuous_learning import ContinuousLearningEngine

            self._learning_engine = ContinuousLearningEngine()
            await self._learning_engine.initialize()

            self._subsystem_status[AGISubsystem.LEARNING] = SubsystemStatus(
                name="continuous_learning",
                initialized=True,
                healthy=True,
                last_check=time.time(),
            )

            logger.info("Continuous Learning Engine initialized successfully")
            return True

        except ImportError:
            logger.warning("ContinuousLearningEngine not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize learning engine: {e}")
            self._subsystem_status[AGISubsystem.LEARNING] = SubsystemStatus(
                name="continuous_learning",
                initialized=False,
                healthy=False,
                error=str(e),
            )
            return False

    async def _init_multimodal(self) -> bool:
        """Initialize multimodal fusion subsystem."""
        try:
            from jarvis_prime.core.multimodal_fusion import MultiModalFusionEngine

            self._multimodal_engine = MultiModalFusionEngine()
            await self._multimodal_engine.initialize()

            self._subsystem_status[AGISubsystem.MULTIMODAL] = SubsystemStatus(
                name="multimodal_fusion",
                initialized=True,
                healthy=True,
                last_check=time.time(),
            )

            logger.info("MultiModal Fusion Engine initialized successfully")
            return True

        except ImportError:
            logger.warning("MultiModalFusionEngine not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize multimodal engine: {e}")
            self._subsystem_status[AGISubsystem.MULTIMODAL] = SubsystemStatus(
                name="multimodal_fusion",
                initialized=False,
                healthy=False,
                error=str(e),
            )
            return False

    async def _init_agi_models_v80(self) -> bool:
        """
        Initialize v80.0 AGI Models subsystems with timeout protection.

        v93.12: Enhanced with:
        - Per-component timeout protection (prevents hanging)
        - Graceful degradation (individual failures don't block others)
        - Detailed progress logging for debugging
        - Parallel initialization where safe

        Includes:
            - AGI Model Manager (MoE, specialized models)
            - Continual Learning Engine (experience replay, RAG)
            - Self-Modification Engine (meta-learning, NAS)
            - Knowledge Distillation Engine
            - Active Learning Engine
        """
        # v93.15: Enhanced per-component timeout with environment variable overrides
        # Default: divide overall timeout among 3 main async components, capped at 30s
        # Increased from 15s to 30s for heavy ML model loading operations
        default_component_timeout = min(
            self._config.agi_models_v80_timeout / 3,
            30.0  # was 15.0 - too short for ML operations
        )

        # v93.14: Per-component timeout overrides via environment variables
        # Allows fine-grained control over each component's initialization timeout
        component_timeouts = {
            "model_manager": float(os.getenv("AGI_MODEL_MANAGER_TIMEOUT", str(default_component_timeout))),
            "continual_learner": float(os.getenv("AGI_CONTINUAL_LEARNING_TIMEOUT", str(default_component_timeout))),
            "self_modifier": float(os.getenv("AGI_SELF_MODIFIER_TIMEOUT", str(default_component_timeout))),
        }
        component_timeout = default_component_timeout  # Fallback for compatibility

        init_results = {
            "model_manager": False,
            "continual_learner": False,
            "self_modifier": False,
            "knowledge_distiller": False,
            "active_learner": False,
            "nas_engine": False,
        }

        try:
            # Import v80.0 models (synchronous import with timeout protection)
            logger.info(f"v80.0: Importing AGI models module...")
            try:
                from jarvis_prime.models import (
                    get_model_manager,
                    get_continual_learner,
                    get_self_modifier,
                    KnowledgeDistillationEngine,
                    ActiveLearningEngine,
                    NeuralArchitectureSearch,
                )
            except ImportError as e:
                logger.warning(f"v80.0 AGI Models module not available: {e}")
                if self._config.agi_models_v80_graceful_degradation:
                    return False  # Graceful degradation
                raise

            # ----------------------------------------------------------------
            # ASYNC COMPONENTS (with individual timeouts)
            # These are the ones that can hang - wrap each with timeout
            # v93.14: Uses per-component timeouts for fine-grained control
            # ----------------------------------------------------------------

            # 1. AGI Model Manager
            mm_timeout = component_timeouts["model_manager"]
            logger.info(f"v80.0: Initializing AGI Model Manager (timeout: {mm_timeout}s)...")
            try:
                self._agi_model_manager = await asyncio.wait_for(
                    get_model_manager(),
                    timeout=mm_timeout
                )
                logger.info("v80.0 AGI Model Manager initialized")
                init_results["model_manager"] = True
            except asyncio.TimeoutError:
                logger.warning(f"AGI Model Manager timed out after {mm_timeout}s - skipping")
            except Exception as e:
                logger.warning(f"AGI Model Manager init failed: {e}")

            # 2. Continual Learning Engine
            # v93.14: Now uses parallel initialization with background loading
            # Should complete much faster as engine is marked ready immediately
            cl_timeout = component_timeouts["continual_learner"]
            logger.info(f"v80.0: Initializing Continual Learning (timeout: {cl_timeout}s)...")
            try:
                self._continual_learner = await asyncio.wait_for(
                    get_continual_learner(),
                    timeout=cl_timeout
                )
                logger.info("v80.0 Continual Learning Engine initialized")
                init_results["continual_learner"] = True
            except asyncio.TimeoutError:
                logger.warning(f"Continual Learning timed out after {cl_timeout}s - skipping")
            except Exception as e:
                logger.warning(f"Continual Learning init failed: {e}")

            # 3. Self-Modification Engine
            sm_timeout = component_timeouts["self_modifier"]
            logger.info(f"v80.0: Initializing Self-Modification Engine (timeout: {sm_timeout}s)...")
            try:
                self._self_modifier = await asyncio.wait_for(
                    get_self_modifier(),
                    timeout=sm_timeout
                )
                logger.info("v80.0 Self-Modification Engine initialized")
                init_results["self_modifier"] = True
            except asyncio.TimeoutError:
                logger.warning(f"Self-Modification timed out after {sm_timeout}s - skipping")
            except Exception as e:
                logger.warning(f"Self-Modification init failed: {e}")

            # ----------------------------------------------------------------
            # SYNC COMPONENTS (fast, shouldn't hang)
            # ----------------------------------------------------------------

            # 4. Knowledge Distillation
            try:
                self._knowledge_distiller = KnowledgeDistillationEngine()
                logger.info("v80.0 Knowledge Distillation Engine initialized")
                init_results["knowledge_distiller"] = True
            except Exception as e:
                logger.warning(f"Knowledge Distillation init failed: {e}")

            # 5. Active Learning
            try:
                self._active_learner = ActiveLearningEngine()
                logger.info("v80.0 Active Learning Engine initialized")
                init_results["active_learner"] = True
            except Exception as e:
                logger.warning(f"Active Learning init failed: {e}")

            # 6. NAS Engine
            try:
                self._nas_engine = NeuralArchitectureSearch()
                logger.info("v80.0 Neural Architecture Search initialized")
                init_results["nas_engine"] = True
            except Exception as e:
                logger.warning(f"NAS init failed: {e}")

            # Update status
            self._subsystem_status[AGISubsystem.AGI_MODELS] = SubsystemStatus(
                name="agi_models_v80",
                initialized=init_results["model_manager"],
                healthy=init_results["model_manager"],
                last_check=time.time(),
            )

            self._subsystem_status[AGISubsystem.CONTINUAL_LEARNING] = SubsystemStatus(
                name="continual_learning_v80",
                initialized=init_results["continual_learner"],
                healthy=init_results["continual_learner"],
                last_check=time.time(),
            )

            self._subsystem_status[AGISubsystem.SELF_IMPROVEMENT] = SubsystemStatus(
                name="self_improvement_v80",
                initialized=init_results["self_modifier"],
                healthy=init_results["self_modifier"],
                last_check=time.time(),
            )

            # Report summary
            success_count = sum(1 for v in init_results.values() if v)
            total_count = len(init_results)
            logger.info(
                f"v80.0 AGI Models: {success_count}/{total_count} components initialized"
            )

            # v93.12: Return True if ANY component initialized (graceful degradation)
            return success_count > 0

        except ImportError as e:
            logger.warning(f"v80.0 AGI Models not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize v80.0 AGI Models: {e}")
            self._subsystem_status[AGISubsystem.AGI_MODELS] = SubsystemStatus(
                name="agi_models_v80",
                initialized=False,
                healthy=False,
                error=str(e),
            )
            if self._config.agi_models_v80_graceful_degradation:
                return False  # Continue with degraded functionality
            raise

    # -------------------------------------------------------------------------
    # V80.0 AGI MODEL ACCESSORS
    # -------------------------------------------------------------------------

    @property
    def agi_model_manager(self) -> Optional[Any]:
        """Get AGI Model Manager (v80.0)."""
        return self._agi_model_manager

    @property
    def continual_learner(self) -> Optional[Any]:
        """Get Continual Learning Engine (v80.0)."""
        return self._continual_learner

    @property
    def self_modifier(self) -> Optional[Any]:
        """Get Self-Modification Engine (v80.0)."""
        return self._self_modifier

    @property
    def knowledge_distiller(self) -> Optional[Any]:
        """Get Knowledge Distillation Engine (v80.0)."""
        return self._knowledge_distiller

    @property
    def active_learner(self) -> Optional[Any]:
        """Get Active Learning Engine (v80.0)."""
        return self._active_learner

    @property
    def nas_engine(self) -> Optional[Any]:
        """Get Neural Architecture Search Engine (v80.0)."""
        return self._nas_engine

    async def shutdown(self) -> None:
        """
        Gracefully shutdown all subsystems with enterprise-grade robustness.

        v95.0: Complete rewrite with defensive patterns:
        - Safe shutdown helper prevents orphaned coroutines
        - hasattr() checks prevent AttributeError cascades
        - Individual timeout protection per subsystem
        - Detailed logging for debugging
        - Graceful degradation on partial failures

        CRITICAL FIX: Previous implementation could leave coroutines orphaned
        if any subsystem's shutdown() attribute didn't exist. The new pattern
        wraps each shutdown in a safe helper that handles ALL edge cases.
        """
        logger.info("Shutting down AGI Integration Hub...")
        shutdown_start = time.time()

        # v95.0: Safe shutdown helper that handles all edge cases
        async def _safe_shutdown(
            subsystem: Any,
            name: str,
            timeout: float = 30.0
        ) -> Tuple[str, bool, Optional[str]]:
            """
            Safely shutdown a subsystem with comprehensive error handling.

            Returns:
                Tuple of (name, success, error_message)
            """
            if subsystem is None:
                return (name, True, None)  # Not initialized, nothing to do

            try:
                # Check if shutdown method exists BEFORE calling
                if not hasattr(subsystem, 'shutdown'):
                    logger.debug(f"  {name}: No shutdown method (skipped)")
                    return (name, True, None)

                # Check if shutdown is callable
                shutdown_method = getattr(subsystem, 'shutdown')
                if not callable(shutdown_method):
                    logger.warning(f"  {name}: shutdown is not callable")
                    return (name, False, "shutdown not callable")

                # Call shutdown and check if it returns a coroutine
                result = shutdown_method()

                # Handle both sync and async shutdown methods
                if asyncio.iscoroutine(result):
                    # Async shutdown - await with timeout protection
                    try:
                        await asyncio.wait_for(result, timeout=timeout)
                        logger.debug(f"  {name}: shutdown complete")
                        return (name, True, None)
                    except asyncio.TimeoutError:
                        logger.warning(f"  {name}: shutdown timed out after {timeout}s")
                        return (name, False, f"timeout after {timeout}s")
                    except asyncio.CancelledError:
                        logger.warning(f"  {name}: shutdown was cancelled")
                        return (name, False, "cancelled")
                else:
                    # Sync shutdown (returned None or a value)
                    logger.debug(f"  {name}: sync shutdown complete")
                    return (name, True, None)

            except AttributeError as e:
                # Shouldn't happen due to hasattr check, but be defensive
                logger.warning(f"  {name}: AttributeError during shutdown: {e}")
                return (name, False, str(e))
            except Exception as e:
                logger.error(f"  {name}: Exception during shutdown: {e}")
                return (name, False, str(e))

        # v95.0: Define subsystems to shutdown with their names
        subsystems = [
            (self._learning_engine, "ContinuousLearningEngine"),
            (self._orchestrator, "AGIOrchestrator"),
            (self._reasoning_engine, "ReasoningEngine"),
            (self._multimodal_engine, "MultiModalFusionEngine"),
            # v80.0 AGI Models
            (self._continual_learner, "ContinualLearner_v80"),
            (self._self_modifier, "SelfModifier_v80"),
        ]

        # v95.0: Create shutdown tasks safely - no coroutines created until we're
        # inside the safe helper, preventing orphaned coroutine issues
        shutdown_tasks = [
            _safe_shutdown(subsystem, name)
            for subsystem, name in subsystems
            if subsystem is not None
        ]

        # Execute all shutdowns in parallel with exception isolation
        if shutdown_tasks:
            results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)

            # Log summary
            success_count = sum(
                1 for r in results
                if isinstance(r, tuple) and r[1] is True
            )
            total_count = len(results)

            # Log any failures
            for r in results:
                if isinstance(r, tuple) and r[2] is not None:
                    logger.warning(f"  Shutdown issue: {r[0]} - {r[2]}")
                elif isinstance(r, Exception):
                    logger.error(f"  Unexpected shutdown error: {r}")

            elapsed = time.time() - shutdown_start
            logger.info(
                f"AGI Integration Hub shutdown complete: "
                f"{success_count}/{total_count} subsystems in {elapsed:.2f}s"
            )
        else:
            logger.info("AGI Integration Hub shutdown complete (no subsystems to shutdown)")

        # Clear references to help garbage collection
        self._learning_engine = None
        self._orchestrator = None
        self._reasoning_engine = None
        self._multimodal_engine = None
        self._continual_learner = None
        self._self_modifier = None
        self._agi_model_manager = None
        self._knowledge_distiller = None
        self._active_learner = None
        self._nas_engine = None

        # v138.0: Clean up staged initialization resources
        if self._oom_engine:
            try:
                await self._oom_engine.stop_monitoring()
                logger.debug("  OOM Protection Engine stopped")
            except Exception as e:
                logger.warning(f"  OOM Engine shutdown error: {e}")
            self._oom_engine = None

        if self._memory_gate:
            try:
                self._memory_gate.cleanup()
                logger.debug("  Memory Gate cleaned up")
            except Exception as e:
                logger.warning(f"  Memory Gate cleanup error: {e}")
            self._memory_gate = None

        # Clean up lazy proxies
        self._lazy_proxies.clear()
        self._deferred_subsystems.clear()
        self._stage_results.clear()

        self._initialized = False

    # -------------------------------------------------------------------------
    # MAIN PROCESSING PIPELINE
    # -------------------------------------------------------------------------

    async def process(
        self,
        content: str,
        modalities: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        inference_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> AGIResponse:
        """
        Process a request through the AGI pipeline.

        This is the main entry point for AGI-enhanced inference:
        1. Analyze and classify the request
        2. Apply appropriate reasoning strategy
        3. Coordinate multiple AGI models if needed
        4. Execute inference with reasoning context
        5. Record experience for learning

        Args:
            content: The main text content of the request
            modalities: List of input modalities (text, image, etc.)
            context: Additional context for processing
            inference_fn: Optional custom inference function
            **kwargs: Additional parameters

        Returns:
            AGIResponse with processed result and reasoning trace
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Create request object
        request = AGIRequest(
            content=content,
            modalities=modalities or ["text"],
            context=context or {},
            metadata=kwargs,
        )

        try:
            # Step 1: Analyze request
            request = await self._analyzer.analyze(request)

            logger.debug(
                f"Request {request.id}: complexity={request.complexity.name}, "
                f"reasoning={request.reasoning_requirement.name}, "
                f"models={request.required_models}"
            )

            # Step 2: Apply reasoning if needed
            reasoning_trace = []
            reasoning_output = None

            if (
                self._config.enable_auto_reasoning
                and request.reasoning_requirement != ReasoningRequirement.NONE
                and self._reasoning_engine
            ):
                reasoning_output, reasoning_trace = await self._apply_reasoning(
                    request
                )

            # Step 3: Multi-modal fusion if needed
            fused_context = context or {}
            if len(request.modalities) > 1 and self._multimodal_engine:
                fused_context = await self._apply_multimodal_fusion(request)

            # Step 4: AGI orchestration for complex requests
            orchestration_output = None
            if (
                request.complexity in (RequestComplexity.COMPLEX, RequestComplexity.EXPERT)
                and self._orchestrator
            ):
                orchestration_output = await self._apply_orchestration(
                    request, reasoning_output
                )

            # Step 5: Execute inference
            if inference_fn:
                # Use provided inference function with enriched context
                enriched_content = self._enrich_prompt(
                    request.content,
                    reasoning_output,
                    orchestration_output,
                )
                result = await inference_fn(enriched_content, **fused_context)
            else:
                # Return reasoning output as result (no inference function)
                result = reasoning_output or request.content

            # Step 6: Build response
            processing_time = (time.time() - start_time) * 1000

            response = AGIResponse(
                request_id=request.id,
                content=result if isinstance(result, str) else str(result),
                reasoning_trace=reasoning_trace,
                confidence=self._calculate_confidence(reasoning_trace),
                models_used=request.required_models,
                processing_time_ms=processing_time,
            )

            # Step 7: Record experience for learning
            if self._config.enable_experience_recording and self._learning_engine:
                await self._record_experience(request, response)
                response.feedback_recorded = True

            # Update metrics
            self._request_count += 1
            self._total_processing_time += processing_time

            return response

        except Exception as e:
            logger.error(f"Error processing AGI request: {e}", exc_info=True)
            return AGIResponse(
                request_id=request.id,
                content=f"Error: {str(e)}",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )

    async def _apply_reasoning(
        self,
        request: AGIRequest
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Apply reasoning strategy to the request."""
        from jarvis_prime.core.reasoning_engine import ReasoningStrategy

        # Map reasoning requirement to strategy
        strategy_map = {
            ReasoningRequirement.CHAIN: ReasoningStrategy.CHAIN_OF_THOUGHT,
            ReasoningRequirement.TREE: ReasoningStrategy.TREE_OF_THOUGHTS,
            ReasoningRequirement.CAUSAL: ReasoningStrategy.CHAIN_OF_THOUGHT,  # TODO: Add causal
            ReasoningRequirement.PLANNING: ReasoningStrategy.TREE_OF_THOUGHTS,
            ReasoningRequirement.META: ReasoningStrategy.SELF_REFLECTION,
        }

        strategy = strategy_map.get(
            request.reasoning_requirement,
            ReasoningStrategy.CHAIN_OF_THOUGHT
        )

        try:
            result = await asyncio.wait_for(
                self._reasoning_engine.reason(
                    query=request.content,
                    strategy=strategy,
                    context=request.context,
                ),
                timeout=self._config.reasoning_timeout_seconds,
            )

            # Track usage
            strategy_name = strategy.name
            self._reasoning_usage[strategy_name] = (
                self._reasoning_usage.get(strategy_name, 0) + 1
            )

            return result.conclusion, result.to_trace()

        except asyncio.TimeoutError:
            logger.warning(f"Reasoning timed out for request {request.id}")
            return None, [{"error": "reasoning_timeout"}]
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return None, [{"error": str(e)}]

    async def _apply_multimodal_fusion(
        self,
        request: AGIRequest
    ) -> Dict[str, Any]:
        """Apply multimodal fusion for multi-modal requests."""
        try:
            from jarvis_prime.core.multimodal_fusion import ModalityInput, Modality

            inputs = []
            for modality in request.modalities:
                mod_enum = getattr(Modality, modality.upper(), Modality.TEXT)
                mod_data = request.context.get(f"{modality}_data")
                if mod_data:
                    inputs.append(ModalityInput(modality=mod_enum, data=mod_data))

            if inputs:
                fused = await self._multimodal_engine.fuse(inputs)
                return {"fused_representation": fused}

            return {}

        except Exception as e:
            logger.error(f"Multimodal fusion failed: {e}")
            return {}

    async def _apply_orchestration(
        self,
        request: AGIRequest,
        reasoning_output: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Apply AGI orchestration for complex requests."""
        try:
            from jarvis_prime.core.agi_models import AGIModelType

            # Map required models to AGI model types
            model_map = {
                "action": AGIModelType.ACTION,
                "meta-reasoner": AGIModelType.META_REASONER,
                "causal": AGIModelType.CAUSAL,
                "world-model": AGIModelType.WORLD_MODEL,
                "memory": AGIModelType.MEMORY,
                "goal-inference": AGIModelType.GOAL_INFERENCE,
                "self-model": AGIModelType.SELF_MODEL,
            }

            required_types = [
                model_map[m] for m in request.required_models
                if m in model_map
            ]

            if not required_types:
                return None

            # Process through orchestrator
            result = await self._orchestrator.process(
                input_text=request.content,
                reasoning_context=reasoning_output,
                required_models=required_types,
                cognitive_state=self._cognitive_state,
            )

            # Track model usage
            for model in request.required_models:
                self._model_usage[model] = self._model_usage.get(model, 0) + 1

            return result

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return None

    def _enrich_prompt(
        self,
        original: str,
        reasoning: Optional[str],
        orchestration: Optional[Dict[str, Any]],
    ) -> str:
        """Enrich the original prompt with reasoning and orchestration context."""
        enriched = original

        if reasoning:
            enriched = f"[Reasoning Context]\n{reasoning}\n\n[Query]\n{original}"

        if orchestration:
            if "plan" in orchestration:
                enriched = f"{enriched}\n\n[Execution Plan]\n{orchestration['plan']}"
            if "context" in orchestration:
                enriched = f"{enriched}\n\n[Additional Context]\n{orchestration['context']}"

        return enriched

    def _calculate_confidence(self, trace: List[Dict[str, Any]]) -> float:
        """Calculate confidence score from reasoning trace."""
        if not trace:
            return 0.5  # Default confidence

        # Average confidence from trace entries
        confidences = [
            t.get("confidence", 0.5) for t in trace
            if isinstance(t, dict) and "confidence" in t
        ]

        if confidences:
            return sum(confidences) / len(confidences)

        return 0.5

    async def _record_experience(
        self,
        request: AGIRequest,
        response: AGIResponse
    ) -> None:
        """Record the interaction for continuous learning."""
        try:
            self._learning_engine.record_experience(
                input_text=request.content,
                output_text=response.content,
                metadata={
                    "complexity": request.complexity.name if request.complexity else None,
                    "reasoning": request.reasoning_requirement.name if request.reasoning_requirement else None,
                    "models_used": response.models_used,
                    "processing_time_ms": response.processing_time_ms,
                    "confidence": response.confidence,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to record experience: {e}")

    # -------------------------------------------------------------------------
    # DIRECT SUBSYSTEM ACCESS
    # -------------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        strategy: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Direct access to reasoning engine."""
        if not self._reasoning_engine:
            raise RuntimeError("Reasoning engine not initialized")

        from jarvis_prime.core.reasoning_engine import ReasoningStrategy

        strat = ReasoningStrategy[strategy.upper()] if strategy else (
            ReasoningStrategy.CHAIN_OF_THOUGHT
        )

        result = await self._reasoning_engine.reason(
            query=query,
            strategy=strat,
            context=context or {},
        )

        return {
            "conclusion": result.conclusion,
            "trace": result.to_trace(),
            "confidence": result.confidence,
        }

    async def plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Direct access to action planning."""
        if not self._orchestrator:
            raise RuntimeError("AGI Orchestrator not initialized")

        from jarvis_prime.core.agi_models import AGIModelType

        result = await self._orchestrator.process(
            input_text=goal,
            required_models=[AGIModelType.ACTION, AGIModelType.GOAL_INFERENCE],
            cognitive_state=self._cognitive_state,
        )

        return result

    async def understand_screen(
        self,
        screen_data: bytes,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Direct access to screen understanding."""
        if not self._multimodal_engine:
            raise RuntimeError("MultiModal engine not initialized")

        from jarvis_prime.core.multimodal_fusion import ModalityInput, Modality

        result = await self._multimodal_engine.fuse([
            ModalityInput(modality=Modality.SCREEN, data=screen_data)
        ])

        return {"understanding": result}

    async def record_feedback(
        self,
        experience_id: str,
        score: float,
        comment: Optional[str] = None,
    ) -> bool:
        """Record feedback for continuous learning."""
        if not self._learning_engine:
            return False

        try:
            await self._learning_engine.record_feedback(
                experience_id=experience_id,
                score=score,
                comment=comment,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False

    async def trigger_learning_update(self, force: bool = False) -> Dict[str, Any]:
        """Trigger a learning update."""
        if not self._learning_engine:
            return {"success": False, "error": "Learning engine not initialized"}

        result = await self._learning_engine.trigger_update(force=force)
        return {"success": True, "result": result}

    # -------------------------------------------------------------------------
    # STATUS AND METRICS
    # -------------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all subsystems."""
        status_dict = {
            "initialized": self._initialized,
            "subsystems": {
                name.name.lower(): {
                    "initialized": status.initialized,
                    "healthy": status.healthy,
                    "error": status.error,
                }
                for name, status in self._subsystem_status.items()
            },
            "metrics": {
                "request_count": self._request_count,
                "avg_processing_time_ms": (
                    self._total_processing_time / self._request_count
                    if self._request_count > 0 else 0
                ),
                "reasoning_usage": self._reasoning_usage,
                "model_usage": self._model_usage,
            },
        }

        # v138.0: Add staged initialization info
        status_dict["staged_init"] = {
            "current_stage": self._current_stage.name if self._current_stage else "UNKNOWN",
            "slim_mode": self._slim_mode,
            "memory_pressure": self._memory_pressure.name if self._memory_pressure else "UNKNOWN",
            "deferred_subsystems": list(self._deferred_subsystems),
            "lazy_proxies": {
                name: proxy.is_initialized
                for name, proxy in self._lazy_proxies.items()
            },
            "stage_results": [
                {
                    "stage": r.stage.name,
                    "success": r.success,
                    "initialized": r.subsystems_initialized,
                    "failed": r.subsystems_failed,
                    "deferred": r.subsystems_deferred,
                    "elapsed_seconds": r.elapsed_seconds,
                    "memory_delta_mb": r.memory_delta_mb,
                }
                for r in self._stage_results
            ] if self._stage_results else [],
        }

        # OOM Protection status
        if self._oom_engine:
            status_dict["oom_protection"] = self._oom_engine.get_current_status()

        return status_dict

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all subsystems."""
        health = {
            "healthy": True,
            "subsystems": {},
        }

        for subsystem, status in self._subsystem_status.items():
            is_healthy = status.initialized and status.healthy
            health["subsystems"][subsystem.name.lower()] = is_healthy
            if not is_healthy:
                health["healthy"] = False

        return health

    @property
    def orchestrator(self) -> Optional[Any]:
        """Access to AGI orchestrator."""
        return self._orchestrator

    @property
    def reasoning_engine(self) -> Optional[Any]:
        """Access to reasoning engine."""
        return self._reasoning_engine

    @property
    def learning_engine(self) -> Optional[Any]:
        """Access to continuous learning engine."""
        return self._learning_engine

    @property
    def multimodal_engine(self) -> Optional[Any]:
        """Access to multimodal fusion engine."""
        return self._multimodal_engine

    @property
    def hardware_optimizer(self) -> Optional[Any]:
        """Access to hardware optimizer."""
        return self._hardware_optimizer


# =============================================================================
# =============================================================================
# SINGLETON INSTANCE - v79.1: Condition-Based Wait/Notify Pattern
# =============================================================================
#
# RACE CONDITION FIX v79.1: Previous v79.0 implementation used recursive call
# with sleep, which could cause thundering herd when multiple coroutines wake
# up simultaneously.
#
# The new pattern uses asyncio.Condition for proper wait/notify:
# 1. Fast path without lock (99% of calls)
# 2. Condition-based waiting for initialization
# 3. notify_all() to wake waiters when ready
# 4. No recursive calls, no thundering herd
# =============================================================================


_global_hub: Optional[AGIIntegrationHub] = None
_hub_condition: Optional[asyncio.Condition] = None
_hub_initializing = False


def _get_hub_condition() -> asyncio.Condition:
    """Lazy-initialize the condition (must be created in event loop)."""
    global _hub_condition
    if _hub_condition is None:
        _hub_condition = asyncio.Condition()
    return _hub_condition


async def get_agi_hub(config: Optional[AGIHubConfig] = None) -> AGIIntegrationHub:
    """
    Get or create the global AGI Integration Hub singleton.

    v79.1: Uses asyncio.Condition for proper wait/notify pattern.

    Thread Safety:
        - Fast path: Returns existing hub without lock acquisition
        - Slow path: Waiters use Condition.wait() instead of sleep+retry
        - notify_all() wakes all waiters atomically when ready
        - No thundering herd, no recursive calls
        - Handles initialization failures gracefully
    """
    global _global_hub, _hub_initializing

    # Fast path: Already initialized (no lock needed)
    if _global_hub is not None and _global_hub._initialized:
        return _global_hub

    # Slow path: Use condition for proper synchronization
    condition = _get_hub_condition()

    async with condition:
        # Check again under lock
        if _global_hub is not None and _global_hub._initialized:
            return _global_hub

        # If someone else is initializing, wait for them
        if _hub_initializing:
            logger.debug("[AGI Hub] Waiting for ongoing initialization...")
            # Wait on condition - will be notified when init completes
            while _hub_initializing:
                await condition.wait()

            # After waking, check if initialization succeeded
            if _global_hub is not None and _global_hub._initialized:
                return _global_hub
            else:
                # Previous initialization failed, we'll try again
                pass

        # We're the initializer
        _hub_initializing = True

    # Initialize OUTSIDE the lock to avoid blocking waiters during slow init
    try:
        new_hub = AGIIntegrationHub(config)
        await new_hub.initialize()

        # Update global state under lock
        async with condition:
            _global_hub = new_hub
            _hub_initializing = False
            condition.notify_all()  # Wake all waiters

        logger.info("[AGI Hub] Singleton initialized successfully")
        return _global_hub

    except Exception as e:
        # Cleanup on failure and notify waiters
        async with condition:
            _global_hub = None
            _hub_initializing = False
            condition.notify_all()  # Wake waiters so they can retry

        logger.error(f"[AGI Hub] Initialization failed: {e}")
        raise


async def shutdown_agi_hub() -> None:
    """
    Shutdown the global AGI Integration Hub.

    v95.0: Enterprise-grade shutdown with comprehensive error handling.

    Features:
    - Thread-safe shutdown with condition notification
    - Timeout protection to prevent hanging
    - Detailed error logging
    - Graceful degradation on failures
    - Proper cleanup even if shutdown partially fails
    """
    global _global_hub, _hub_initializing

    condition = _get_hub_condition()

    async with condition:
        if _global_hub is not None:
            hub_to_shutdown = _global_hub
            logger.info("[AGI Hub] Initiating global shutdown...")

            try:
                # v95.0: Timeout protection - don't let shutdown hang forever
                shutdown_timeout = 60.0  # 60 seconds max for full shutdown

                await asyncio.wait_for(
                    hub_to_shutdown.shutdown(),
                    timeout=shutdown_timeout
                )
                logger.info("[AGI Hub] Global shutdown completed successfully")

            except asyncio.TimeoutError:
                logger.error(
                    f"[AGI Hub] Shutdown timed out after {shutdown_timeout}s - "
                    "forcing cleanup"
                )
            except asyncio.CancelledError:
                logger.warning("[AGI Hub] Shutdown was cancelled - forcing cleanup")
            except Exception as e:
                logger.error(f"[AGI Hub] Shutdown error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # v95.0: ALWAYS clean up global state, even on error
                # This prevents the system from getting stuck in a bad state
                _global_hub = None
                _hub_initializing = False
                condition.notify_all()  # Wake any waiters

                logger.debug("[AGI Hub] Global state cleaned up")
        else:
            logger.debug("[AGI Hub] No active hub to shutdown")


# =============================================================================
# INFERENCE WRAPPER
# =============================================================================


class AGIEnhancedInference:
    """
    Wrapper that adds AGI capabilities to any inference function.

    Usage:
        original_inference = my_llm.generate
        enhanced = AGIEnhancedInference(original_inference)
        result = await enhanced("How do I solve this complex problem?")
    """

    def __init__(
        self,
        inference_fn: Callable,
        hub: Optional[AGIIntegrationHub] = None,
        config: Optional[AGIHubConfig] = None,
    ) -> None:
        self._inference_fn = inference_fn
        self._hub = hub
        self._config = config
        self._initialized = False

    async def _ensure_hub(self) -> AGIIntegrationHub:
        """Ensure the AGI hub is initialized."""
        if self._hub is None:
            self._hub = await get_agi_hub(self._config)
        return self._hub

    async def __call__(
        self,
        prompt: str,
        modalities: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AGIResponse:
        """Enhanced inference with AGI capabilities."""
        hub = await self._ensure_hub()

        return await hub.process(
            content=prompt,
            modalities=modalities,
            context=context,
            inference_fn=self._inference_fn,
            **kwargs,
        )

    async def reason(self, query: str, strategy: str = "chain_of_thought") -> Dict[str, Any]:
        """Direct reasoning access."""
        hub = await self._ensure_hub()
        return await hub.reason(query, strategy)

    async def plan(self, goal: str) -> Dict[str, Any]:
        """Direct planning access."""
        hub = await self._ensure_hub()
        return await hub.plan(goal)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def enhance_inference(
    inference_fn: Callable,
    config: Optional[AGIHubConfig] = None,
) -> AGIEnhancedInference:
    """Enhance an inference function with AGI capabilities."""
    return AGIEnhancedInference(inference_fn, config=config)


async def agi_process(
    content: str,
    inference_fn: Optional[Callable] = None,
    **kwargs: Any,
) -> AGIResponse:
    """Process content through the AGI pipeline."""
    hub = await get_agi_hub()
    return await hub.process(content, inference_fn=inference_fn, **kwargs)
