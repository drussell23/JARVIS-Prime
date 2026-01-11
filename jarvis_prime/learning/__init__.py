"""
JARVIS Prime Learning Module
============================

Advanced learning capabilities including:
- Continuous Learning with forgetting prevention
- RLHF (Reinforcement Learning from Human Feedback)
- Experience replay and prioritized sampling
- A/B testing for model updates

Usage:
    from jarvis_prime.learning import (
        get_rlhf_pipeline,
        ContinuousLearningEngine,
    )

    # RLHF
    rlhf = await get_rlhf_pipeline()
    await rlhf.record_preference(...)

    # Continuous Learning
    engine = ContinuousLearningEngine()
    engine.record_experience(...)
"""

from jarvis_prime.learning.rlhf_integration import (
    RLHFConfig,
    RLHFPipeline,
    PreferencePair,
    PreferenceBuffer,
    RLHFExperience,
    TrainingMetrics,
    RewardModel,
    NeuralRewardModel,
    PPOTrainer,
    get_rlhf_pipeline,
    shutdown_rlhf_pipeline,
)

# Re-export from continuous_learning for convenience
try:
    from jarvis_prime.core.continuous_learning import (
        ContinuousLearningEngine,
        LearningConfig,
        LearningMode,
        Experience,
        ExperienceBuffer,
        ExperienceType,
        UpdateStatus,
        ForgettingPrevention,
        ElasticWeightConsolidation,
        SynapticIntelligence,
        ABTestManager,
        create_continuous_learning_engine,
        get_continuous_learning_engine,
        shutdown_continuous_learning_engine,
    )
except ImportError:
    pass

__all__ = [
    # RLHF
    "RLHFConfig",
    "RLHFPipeline",
    "PreferencePair",
    "PreferenceBuffer",
    "RLHFExperience",
    "TrainingMetrics",
    "RewardModel",
    "NeuralRewardModel",
    "PPOTrainer",
    "get_rlhf_pipeline",
    "shutdown_rlhf_pipeline",
    # Continuous Learning
    "ContinuousLearningEngine",
    "LearningConfig",
    "LearningMode",
    "Experience",
    "ExperienceBuffer",
    "ExperienceType",
    "UpdateStatus",
    "ForgettingPrevention",
    "ElasticWeightConsolidation",
    "SynapticIntelligence",
    "ABTestManager",
    "create_continuous_learning_engine",
    "get_continuous_learning_engine",
    "shutdown_continuous_learning_engine",
]
