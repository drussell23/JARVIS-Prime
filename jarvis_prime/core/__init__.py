"""
JARVIS-Prime Core - Tier-0 Muscle Memory Brain
===============================================

Core components for the JARVIS-Prime inference server:
- Model Registry & Versioning
- Hot-Swap zero-downtime reload
- Hybrid Routing (Tier 0/1 classification)
- Telemetry with PII anonymization
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
]
