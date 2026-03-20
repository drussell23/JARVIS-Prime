"""
Mind-Body protocol package — J-Prime reasoning schemas and brain selector.

Exports all public protocol models, version constants, and the
UnifiedBrainSelector / UnifiedBrainSelection types.
"""
from jarvis_prime.reasoning.protocol import (
    PROTOCOL_VERSION,
    MIN_SUPPORTED_VERSION,
    MAX_SUPPORTED_VERSION,
    AuthEnvelope,
    Classification,
    Constraints,
    ErrorDetail,
    FallbackPolicy,
    Plan,
    ProtocolVersionInfo,
    ReasonFeedback,
    ReasonRequest,
    ReasonResponse,
    RoutingTrace,
    StepResult,
    SubGoal,
    ReasoningGraphState,
    compute_plan_hash,
)
from jarvis_prime.reasoning.unified_brain_selector import (
    UnifiedBrainSelection,
    UnifiedBrainSelector,
)
from jarvis_prime.reasoning.model_provider import (
    LlamaCppModelProvider,
    MockModelProvider,
    ModelProvider,
)

__all__ = [
    # Protocol version constants
    "PROTOCOL_VERSION",
    "MIN_SUPPORTED_VERSION",
    "MAX_SUPPORTED_VERSION",
    # Protocol models
    "AuthEnvelope",
    "Classification",
    "Constraints",
    "ErrorDetail",
    "FallbackPolicy",
    "Plan",
    "ProtocolVersionInfo",
    "ReasonFeedback",
    "ReasonRequest",
    "ReasonResponse",
    "RoutingTrace",
    "StepResult",
    "SubGoal",
    # Brain selector
    "UnifiedBrainSelection",
    "UnifiedBrainSelector",
    # Model provider (DI interface for GPU inference)
    "ModelProvider",
    "MockModelProvider",
    "LlamaCppModelProvider",
    # Internal graph state + plan hashing
    "ReasoningGraphState",
    "compute_plan_hash",
]
