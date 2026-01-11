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

# AGI v79.1: Advanced Integration Layer
from jarvis_prime.core.trinity_protocol import (
    TrinityMessage,
    MessageType,
    Transport as TrinityTransport,  # Aliased for backward compatibility
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
    get_persistence_manager,
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
    get_metrics as get_metrics_collector,
    get_health_checker,
)
from jarvis_prime.core.data_collector import (
    JARVISDataCollector,
    TrinityDataCollector,
    DataSample,
    DataQuality,
    PIIRedactor,
    AutoLabeler,
    get_data_collector,
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
)

# Advanced Async Primitives (v80.0)
from jarvis_prime.core.advanced_async_primitives import (
    AdaptiveTimeoutManager,
    AsyncRLock,
    PriorityAsyncQueue,
    AsyncObjectPool,
    StructuredTaskGroup,
    DeadlockDetector,
    adaptive_timeout,
    get_adaptive_timeout_manager,
    set_request_context,
    get_request_context,
)

# Zero-Copy IPC (v80.0)
from jarvis_prime.core.zero_copy_ipc import (
    ZeroCopyIPCTransport,
    SharedRingBuffer,
    SharedMemoryConfig,
    MessageSize,
    get_zero_copy_transport,
)

# Distributed Tracing (v80.0)
from jarvis_prime.core.distributed_tracing import (
    DistributedTracer,
    TracingConfig,
    tracer,
    with_span,
    with_span_sync,
    add_span_attributes,
    add_span_event,
    record_exception,
)

# Predictive Cache (v80.0)
from jarvis_prime.core.predictive_cache import (
    PredictiveCache,
    EvictionPolicy,
    get_predictive_cache,
)

# Adaptive Rate Limiter (v80.0)
from jarvis_prime.core.adaptive_rate_limiter import (
    AdaptiveRateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    TokenBucket,
    SlidingWindowCounter,
    get_rate_limiter,
)

# Cross-Repo Orchestrator (v80.0)
from jarvis_prime.core.cross_repo_orchestrator import (
    CrossRepoOrchestrator,
    RepoConfig,
    RepoType,
    OrchestratorConfig,
    SignalHandler,
    FileWatcher,
    DependencyResolver,
    get_orchestrator,
)

# Graph Routing (v80.0)
from jarvis_prime.core.graph_routing import (
    GraphRouter,
    Route,
    Edge,
    EdgeType,
    RoutingObjective,
    get_graph_router,
)

# Intelligent Model Router v1.0 - The Brain Router
from jarvis_prime.core.intelligent_model_router import (
    IntelligentModelRouter,
    ModelTier,
    RoutingStrategy,
    ModelEndpointConfig,
    RoutingConfig,
    RoutingResult,
    ResourceSnapshot,
    ComplexityAnalyzer,
    AdaptiveThresholdManager,
    get_intelligent_router,
    shutdown_intelligent_router,
)

# GCP VM Manager v1.0 - Cloud Infrastructure
from jarvis_prime.core.gcp_vm_manager import (
    GCPVMManager,
    VMConfig,
    VMInstance,
    VMState,
    AutoScaleConfig,
    CostConfig,
    PreemptionConfig,
    GCPManagerConfig,
    PreemptionHandler,
    CostTracker,
    get_gcp_manager,
    shutdown_gcp_manager,
)

# Service Mesh v1.0 - Dynamic Service Discovery
from jarvis_prime.core.service_mesh import (
    ServiceMesh,
    ServiceEndpoint,
    ServiceStatus,
    ServiceMeshConfig,
    LoadBalancingStrategy,
    CircuitState,
    ServiceMeshClient,
    get_service_mesh,
    shutdown_service_mesh,
)

# Advanced Primitives v88.0 - Production-Grade Building Blocks
try:
    from jarvis_prime.core.advanced_primitives import (
        AtomicFileWriter,
        WriteAheadLog,
        AdvancedCircuitBreaker,
        CircuitBreakerConfig,
        CircuitOpenError,
        BackoffConfig,
        ExponentialBackoff,
        with_retry,
        OperationTimeoutError,
        operation_timeout,
        with_timeout,
        GPUInfo,
        NetworkInfo,
        SystemResources,
        ResourceMonitor,
        ManagedConnectionPool,
        TraceContext,
        trace_operation,
        TokenBucketRateLimiter,
        MeteredSemaphore,
    )
    ADVANCED_PRIMITIVES_AVAILABLE = True
except ImportError:
    ADVANCED_PRIMITIVES_AVAILABLE = False

# Trinity Orchestrator v88.0 - Unified Cross-Repo Orchestration
try:
    from jarvis_prime.core.trinity_orchestrator import (
        TrinityOrchestrator,
        ComponentState,
        ComponentType as TrinityComponentType,
        ComponentConfig as TrinityComponentConfig,
        OrchestratorConfig as TrinityOrchestratorConfig,
        RepoDiscovery,
        ProcessManager,
        HealthMonitor,
        TrinityProtocol,
    )
    TRINITY_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    TRINITY_ORCHESTRATOR_AVAILABLE = False

# Verification Suite v88.0 - System Verification Tests
try:
    from jarvis_prime.core.verification_suite import (
        VerificationRunner,
        TestStatus,
        TestResult,
        TestSuiteResult,
        BrainRouterVerification,
        PreemptionDrillVerification,
        OOMProtectionVerification,
        ServiceMeshVerification,
        CrossRepoVerification,
    )
    VERIFICATION_SUITE_AVAILABLE = True
except ImportError:
    VERIFICATION_SUITE_AVAILABLE = False

# Trinity Event Bus v89.0 - The Neural Impulses (Closes The Loop)
try:
    from jarvis_prime.core.trinity_event_bus import (
        TrinityEventBus,
        TrinityEvent,
        EventType,
        EventPriority,
        ComponentID,
        Subscription,
        EventTransport,
        FileEventTransport,
        MemoryEventTransport,
        get_event_bus,
        shutdown_event_bus,
    )
    TRINITY_EVENT_BUS_AVAILABLE = True
except ImportError:
    TRINITY_EVENT_BUS_AVAILABLE = False

# Intelligent Request Router v89.0 - Routes Requests to Best Endpoint
try:
    from jarvis_prime.core.intelligent_request_router import (
        IntelligentRequestRouter,
        EndpointManager,
        EndpointCircuitBreaker,
        Capability,
        EndpointType,
        RoutingPriority,
        EndpointHealth,
        EndpointConfig,
        RoutingContext,
        RoutingResult,
        RequestResult,
        get_request_router,
        shutdown_request_router,
        route_request,
    )
    INTELLIGENT_REQUEST_ROUTER_AVAILABLE = True
except ImportError:
    INTELLIGENT_REQUEST_ROUTER_AVAILABLE = False

# Trinity Bridge Adapter v89.0 - Unified Cross-Repo Event Bridge
try:
    from jarvis_prime.core.trinity_bridge_adapter import (
        TrinityBridgeAdapter,
        UnifiedEventType,
        UnifiedEvent,
        get_bridge_adapter,
        start_bridge,
        stop_bridge,
    )
    TRINITY_BRIDGE_ADAPTER_AVAILABLE = True
except ImportError:
    TRINITY_BRIDGE_ADAPTER_AVAILABLE = False

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
    "get_persistence_manager",
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
    "get_metrics_collector",
    "get_health_checker",
    # Data Collector (v78.0)
    "JARVISDataCollector",
    "TrinityDataCollector",
    "DataSample",
    "DataQuality",
    "PIIRedactor",
    "AutoLabeler",
    "get_data_collector",
    # Model Server (v78.0)
    "AGIModelServer",
    "InferenceRequest",
    "InferenceResponse",
    "RequestPriority",
    "LRUCache",
    "PriorityQueue",
    "RequestBatcher",
    "get_model_server",
    # Advanced Async Primitives (v80.0)
    "AdaptiveTimeoutManager",
    "AsyncRLock",
    "PriorityAsyncQueue",
    "AsyncObjectPool",
    "StructuredTaskGroup",
    "DeadlockDetector",
    "adaptive_timeout",
    "get_adaptive_timeout_manager",
    "set_request_context",
    "get_request_context",
    # Zero-Copy IPC (v80.0)
    "ZeroCopyIPCTransport",
    "SharedRingBuffer",
    "SharedMemoryConfig",
    "MessageSize",
    "get_zero_copy_transport",
    # Distributed Tracing (v80.0)
    "DistributedTracer",
    "TracingConfig",
    "tracer",
    "with_span",
    "with_span_sync",
    "add_span_attributes",
    "add_span_event",
    "record_exception",
    # Predictive Cache (v80.0)
    "PredictiveCache",
    "EvictionPolicy",
    "get_predictive_cache",
    # Adaptive Rate Limiter (v80.0)
    "AdaptiveRateLimiter",
    "RateLimitConfig",
    "RateLimitStrategy",
    "TokenBucket",
    "SlidingWindowCounter",
    "get_rate_limiter",
    # Cross-Repo Orchestrator (v80.0)
    "CrossRepoOrchestrator",
    "RepoConfig",
    "RepoType",
    "OrchestratorConfig",
    "SignalHandler",
    "FileWatcher",
    "DependencyResolver",
    "get_orchestrator",
    # Graph Routing (v80.0)
    "GraphRouter",
    "Route",
    "Edge",
    "EdgeType",
    "RoutingObjective",
    "get_graph_router",
    # Intelligent Model Router v1.0
    "IntelligentModelRouter",
    "ModelTier",
    "RoutingStrategy",
    "ModelEndpointConfig",
    "RoutingConfig",
    "RoutingResult",
    "ResourceSnapshot",
    "ComplexityAnalyzer",
    "AdaptiveThresholdManager",
    "get_intelligent_router",
    "shutdown_intelligent_router",
    # GCP VM Manager v1.0
    "GCPVMManager",
    "VMConfig",
    "VMInstance",
    "VMState",
    "AutoScaleConfig",
    "CostConfig",
    "PreemptionConfig",
    "GCPManagerConfig",
    "PreemptionHandler",
    "CostTracker",
    "get_gcp_manager",
    "shutdown_gcp_manager",
    # Service Mesh v1.0
    "ServiceMesh",
    "ServiceEndpoint",
    "ServiceStatus",
    "ServiceMeshConfig",
    "LoadBalancingStrategy",
    "CircuitState",
    "ServiceMeshClient",
    "get_service_mesh",
    "shutdown_service_mesh",
    # Advanced Primitives v88.0
    "AtomicFileWriter",
    "WriteAheadLog",
    "AdvancedCircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "BackoffConfig",
    "ExponentialBackoff",
    "with_retry",
    "OperationTimeoutError",
    "operation_timeout",
    "with_timeout",
    "GPUInfo",
    "NetworkInfo",
    "SystemResources",
    "ResourceMonitor",
    "ManagedConnectionPool",
    "TraceContext",
    "trace_operation",
    "TokenBucketRateLimiter",
    "MeteredSemaphore",
    "ADVANCED_PRIMITIVES_AVAILABLE",
    # Trinity Orchestrator v88.0
    "TrinityOrchestrator",
    "ComponentState",
    "TrinityComponentType",
    "TrinityComponentConfig",
    "TrinityOrchestratorConfig",
    "RepoDiscovery",
    "ProcessManager",
    "HealthMonitor",
    "TrinityProtocol",
    "TRINITY_ORCHESTRATOR_AVAILABLE",
    # Verification Suite v88.0
    "VerificationRunner",
    "TestStatus",
    "TestResult",
    "TestSuiteResult",
    "BrainRouterVerification",
    "PreemptionDrillVerification",
    "OOMProtectionVerification",
    "ServiceMeshVerification",
    "CrossRepoVerification",
    "VERIFICATION_SUITE_AVAILABLE",
    # Trinity Event Bus v89.0 - The Loop
    "TrinityEventBus",
    "TrinityEvent",
    "EventType",
    "EventPriority",
    "ComponentID",
    "Subscription",
    "EventTransport",
    "FileEventTransport",
    "MemoryEventTransport",
    "get_event_bus",
    "shutdown_event_bus",
    "TRINITY_EVENT_BUS_AVAILABLE",
    # Intelligent Request Router v89.0
    "IntelligentRequestRouter",
    "EndpointManager",
    "EndpointCircuitBreaker",
    "Capability",
    "EndpointType",
    "RoutingPriority",
    "EndpointHealth",
    "EndpointConfig",
    "RoutingContext",
    "RoutingResult",
    "RequestResult",
    "get_request_router",
    "shutdown_request_router",
    "route_request",
    "INTELLIGENT_REQUEST_ROUTER_AVAILABLE",
    # Trinity Bridge Adapter v89.0
    "TrinityBridgeAdapter",
    "UnifiedEventType",
    "UnifiedEvent",
    "get_bridge_adapter",
    "start_bridge",
    "stop_bridge",
    "TRINITY_BRIDGE_ADAPTER_AVAILABLE",
]
