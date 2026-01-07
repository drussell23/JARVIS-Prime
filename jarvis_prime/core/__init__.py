"""
JARVIS-Prime Core - AGI Operating System v78.0
==============================================

Core components for the JARVIS-Prime AGI inference server:
- Model Registry & Versioning
- Hot-Swap zero-downtime reload
- Hybrid Routing (Tier 0/1 classification)
- Telemetry with PII anonymization
- PROJECT TRINITY: Cross-repo integration
- AGI v76.0: 7 Cognitive Models + Reasoning Engine
- AGI v77.0: Integration Hub + Enhanced Routing
- AGI v78.0: Advanced Integration Layer
  - Trinity Protocol: Cross-repo IPC (WebSocket, File IPC)
  - JARVIS Prime Bridge: Body-Mind connection
  - AGI Persistence: Versioned checkpoints
  - AGI Error Handler: Circuit breaker + Fallback chains
  - AGI Configuration: Hot-reloading config system
  - AGI Metrics: Comprehensive monitoring
  - Data Collector: Continuous learning pipeline
  - Model Server: Batched inference with caching
"""

from jarvis_prime.core.model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelLineage,
    VersionState,
)
from jarvis_prime.core.hot_swap_manager import (
    HotSwapManager,
    SwapState,
    SwapResult,
)
from jarvis_prime.core.hybrid_router import (
    HybridRouter,
    RoutingDecision,
    TierClassification,
)
from jarvis_prime.core.telemetry_hook import (
    TelemetryHook,
    TelemetryRecord,
    PIIAnonymizer,
)
from jarvis_prime.core.model_manager import (
    PrimeModelManager,
    ModelExecutor,
)

# PROJECT TRINITY: J-Prime Bridge
from jarvis_prime.core.trinity_bridge import (
    initialize_trinity,
    shutdown_trinity,
    update_model_status,
    is_trinity_initialized,
    get_trinity_status,
    send_to_jarvis,
    send_plan_to_jarvis,
    start_surveillance,
    stop_surveillance,
    bring_back_windows,
    exile_window,
    freeze_app,
    thaw_app,
    get_jarvis_state,
    is_jarvis_online,
)

# AGI v76.0: Advanced Cognitive Systems
from jarvis_prime.core.agi_models import (
    AGIModelType,
    AGIOrchestrator,
    CognitiveState,
    ActionModel,
    MetaReasoner,
    CausalEngine,
    WorldModel,
    MemoryConsolidator,
    GoalInference,
    SelfModel,
)
from jarvis_prime.core.reasoning_engine import (
    ReasoningEngine,
    ReasoningStrategy,
    Thought,
    ThoughtTree,
    ReasoningResult,
)
from jarvis_prime.core.apple_silicon_optimizer import (
    AppleSiliconOptimizer,
    AppleSiliconGeneration,
    MPSOptimizer,
    CoreMLOptimizer,
    UMAOptimizer,
)
from jarvis_prime.core.continuous_learning import (
    ContinuousLearningEngine,
    ExperienceBuffer,
    ElasticWeightConsolidation,
    SynapticIntelligence,
    ABTestManager,
)
from jarvis_prime.core.multimodal_fusion import (
    MultiModalFusionEngine,
    Modality,
    FusionStrategy,
    SpatialReasoner,
    TemporalReasoner,
)
from jarvis_prime.core.agi_integration import (
    AGIIntegrationHub,
    AGIHubConfig,
    AGIRequest,
    AGIResponse,
    AGIEnhancedInference,
    get_agi_hub,
    shutdown_agi_hub,
    agi_process,
    enhance_inference,
)
from jarvis_prime.core.hybrid_router import (
    create_agi_enhanced_router,
    create_standard_router,
    AGIEnhancedAnalyzer,
)

# AGI v78.0: Advanced Integration Layer
from jarvis_prime.core.trinity_protocol import (
    TrinityMessage,
    MessageType,
    TrinityTransport,
    FileIPCTransport,
    WebSocketTransport,
    TrinityClient,
    send_to_jarvis as trinity_send_jarvis,
    send_to_reactor,
    get_trinity_client,
)
from jarvis_prime.core.jarvis_bridge import (
    JARVISPrimeBridge,
    JARVISCommand,
    PrimeResponse,
    ProcessingMode,
    SafetyContext,
    get_jarvis_bridge,
    process_jarvis_command,
)
from jarvis_prime.core.agi_persistence import (
    AGIPersistenceManager,
    CheckpointManager,
    ModelComponent,
    SemanticVersion,
    ModelMetadata,
    CheckpointType,
    get_persistence_manager,
    save_model,
    load_model,
    create_checkpoint,
)
from jarvis_prime.core.agi_error_handler import (
    AGIErrorHandler,
    AGIError,
    ErrorCategory,
    ErrorSeverity,
    CircuitBreaker,
    CircuitState,
    FallbackChain,
    FallbackStrategy,
    ErrorClassifier,
    get_error_handler,
    handle_error,
    with_error_handling,
)
from jarvis_prime.core.agi_config import (
    AGIConfig,
    ConfigManager,
    ModelConfig,
    ReasoningConfig,
    LearningConfig,
    HardwareConfig,
    MultiModalConfig,
    TrinityConfig,
    SafetyConfig,
    PersistenceConfig,
    MetricsConfig,
    ServerConfig,
    get_config,
    get_config_manager,
)
from jarvis_prime.core.agi_metrics import (
    AGIMetricsCollector,
    MetricsRegistry,
    HealthChecker,
    MetricType,
    Counter,
    Gauge,
    Histogram,
    Timer,
    Summary,
    HealthStatus,
    ComponentHealth,
    get_metrics_collector,
    get_health_checker,
    record_metric,
    check_health,
)
from jarvis_prime.core.data_collector import (
    JARVISDataCollector,
    TrinityDataCollector,
    DataSample,
    DataQuality,
    PIIRedactor,
    AutoLabeler,
    get_data_collector,
    collect_sample,
    get_training_batch,
)
from jarvis_prime.core.model_server import (
    AGIModelServer,
    InferenceRequest,
    InferenceResponse,
    RequestPriority,
    LRUCache,
    PriorityQueue,
    RequestBatcher,
    get_model_server,
    infer,
    batch_infer,
)

__all__ = [
    # Registry
    "ModelRegistry",
    "ModelVersion",
    "ModelLineage",
    "VersionState",
    # Hot-Swap
    "HotSwapManager",
    "SwapState",
    "SwapResult",
    # Router
    "HybridRouter",
    "RoutingDecision",
    "TierClassification",
    # Telemetry
    "TelemetryHook",
    "TelemetryRecord",
    "PIIAnonymizer",
    # Manager
    "PrimeModelManager",
    "ModelExecutor",
    # PROJECT TRINITY
    "initialize_trinity",
    "shutdown_trinity",
    "update_model_status",
    "is_trinity_initialized",
    "get_trinity_status",
    "send_to_jarvis",
    "send_plan_to_jarvis",
    "start_surveillance",
    "stop_surveillance",
    "bring_back_windows",
    "exile_window",
    "freeze_app",
    "thaw_app",
    "get_jarvis_state",
    "is_jarvis_online",
    # AGI Models
    "AGIModelType",
    "AGIOrchestrator",
    "CognitiveState",
    "ActionModel",
    "MetaReasoner",
    "CausalEngine",
    "WorldModel",
    "MemoryConsolidator",
    "GoalInference",
    "SelfModel",
    # Reasoning Engine
    "ReasoningEngine",
    "ReasoningStrategy",
    "Thought",
    "ThoughtTree",
    "ReasoningResult",
    # Apple Silicon Optimization
    "AppleSiliconOptimizer",
    "AppleSiliconGeneration",
    "MPSOptimizer",
    "CoreMLOptimizer",
    "UMAOptimizer",
    # Continuous Learning
    "ContinuousLearningEngine",
    "ExperienceBuffer",
    "ElasticWeightConsolidation",
    "SynapticIntelligence",
    "ABTestManager",
    # Multi-Modal Fusion
    "MultiModalFusionEngine",
    "Modality",
    "FusionStrategy",
    "SpatialReasoner",
    "TemporalReasoner",
    # AGI Integration Hub
    "AGIIntegrationHub",
    "AGIHubConfig",
    "AGIRequest",
    "AGIResponse",
    "AGIEnhancedInference",
    "get_agi_hub",
    "shutdown_agi_hub",
    "agi_process",
    "enhance_inference",
    # AGI-Enhanced Routing
    "create_agi_enhanced_router",
    "create_standard_router",
    "AGIEnhancedAnalyzer",
    # Trinity Protocol (v78.0)
    "TrinityMessage",
    "MessageType",
    "TrinityTransport",
    "FileIPCTransport",
    "WebSocketTransport",
    "TrinityClient",
    "trinity_send_jarvis",
    "send_to_reactor",
    "get_trinity_client",
    # JARVIS Prime Bridge (v78.0)
    "JARVISPrimeBridge",
    "JARVISCommand",
    "PrimeResponse",
    "ProcessingMode",
    "SafetyContext",
    "get_jarvis_bridge",
    "process_jarvis_command",
    # AGI Persistence (v78.0)
    "AGIPersistenceManager",
    "CheckpointManager",
    "ModelComponent",
    "SemanticVersion",
    "ModelMetadata",
    "CheckpointType",
    "get_persistence_manager",
    "save_model",
    "load_model",
    "create_checkpoint",
    # AGI Error Handler (v78.0)
    "AGIErrorHandler",
    "AGIError",
    "ErrorCategory",
    "ErrorSeverity",
    "CircuitBreaker",
    "CircuitState",
    "FallbackChain",
    "FallbackStrategy",
    "ErrorClassifier",
    "get_error_handler",
    "handle_error",
    "with_error_handling",
    # AGI Configuration (v78.0)
    "AGIConfig",
    "ConfigManager",
    "ModelConfig",
    "ReasoningConfig",
    "LearningConfig",
    "HardwareConfig",
    "MultiModalConfig",
    "TrinityConfig",
    "SafetyConfig",
    "PersistenceConfig",
    "MetricsConfig",
    "ServerConfig",
    "get_config",
    "get_config_manager",
    # AGI Metrics (v78.0)
    "AGIMetricsCollector",
    "MetricsRegistry",
    "HealthChecker",
    "MetricType",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "Summary",
    "HealthStatus",
    "ComponentHealth",
    "get_metrics_collector",
    "get_health_checker",
    "record_metric",
    "check_health",
    # Data Collector (v78.0)
    "JARVISDataCollector",
    "TrinityDataCollector",
    "DataSample",
    "DataQuality",
    "PIIRedactor",
    "AutoLabeler",
    "get_data_collector",
    "collect_sample",
    "get_training_batch",
    # Model Server (v78.0)
    "AGIModelServer",
    "InferenceRequest",
    "InferenceResponse",
    "RequestPriority",
    "LRUCache",
    "PriorityQueue",
    "RequestBatcher",
    "get_model_server",
    "infer",
    "batch_infer",
]
