"""
JARVIS-Prime Models - Advanced AGI Model Architecture
=====================================================

v80.0 - Specialized AGI Models with Continual Learning

This package contains all specialized AGI models for JARVIS-Prime:
- Causal Reasoning Model: Understands cause and effect
- World Model: Physics understanding and prediction
- Theory of Mind: Models other agents' mental states
- Meta-Learning Model: Learns how to learn
- Self-Modification Model: Improves itself safely

FEATURES:
    - Dynamic model loading and unloading
    - Continual learning from interactions
    - Mixture of Experts routing
    - Retrieval-augmented generation
    - Comprehensive evaluation framework
    - Self-improvement with safety constraints
"""

# Base models (existing)
from jarvis_prime.models.prime_model import PrimeModel, PrimeConfig
from jarvis_prime.models.llama_model import (
    LlamaModel,
    load_llama_13b_gcp,
    load_llama_13b_m1,
    load_from_config,
)

# Advanced AGI Models (v80.0)
# AGI Model Orchestrator - Model Management, MoE, and Evaluation
from jarvis_prime.models.agi_model_orchestrator import (
    AGIModelManager,
    AGIModelType,
    ModelCapability,
    get_model_manager,
    ModelEvaluator,
    EvaluationMetric,
    BenchmarkSuite,
    get_model_evaluator,
    MixtureOfExpertsRouter,
    ExpertOutput,
    GatingNetwork,
    get_moe_router,
)

# Continual Learning System - Experience Replay, RAG, Knowledge Distillation, Active Learning
from jarvis_prime.models.continual_learning_system import (
    ContinualLearningEngine,
    ExperienceReplayBuffer,
    LearningStrategy,
    get_continual_learner,
    RAGEngine,
    VectorStore,
    RetrievalStrategy,
    get_rag_engine,
    # Knowledge Distillation (v80.0)
    KnowledgeDistillationEngine,
    DistillationMode,
    DistillationConfig,
    # Active Learning (v80.0)
    ActiveLearningEngine,
    ActiveLearningStrategy,
    ActiveLearningConfig,
    # Infrastructure Integration (v80.0)
    InfrastructureIntegration,
    get_infrastructure,
)

# Self-Improving Agent - Safe Self-Modification, NAS, Optimization
from jarvis_prime.models.self_improving_agent import (
    SelfModificationEngine,
    SafetyConstraints,
    ModificationStrategy,
    get_self_modifier,
    # Neural Architecture Search (v80.0)
    NeuralArchitectureSearch,
    NASSearchStrategy,
    NASSearchSpace,
    Architecture,
    # Bayesian Optimization (v80.0)
    BayesianOptimizer,
    AcquisitionFunction,
    HyperparameterSpace,
    # Evolutionary Optimization (v80.0)
    EvolutionaryOptimizer,
    EvolutionaryConfig,
    SelectionMethod,
    # Infrastructure Integration (v80.0)
    SelfImprovementInfrastructure,
    get_improvement_infrastructure,
)

__all__ = [
    # Base Models
    "PrimeModel",
    "PrimeConfig",
    "LlamaModel",
    "load_llama_13b_gcp",
    "load_llama_13b_m1",
    "load_from_config",
    # AGI Model Manager (v80.0)
    "AGIModelManager",
    "AGIModelType",
    "ModelCapability",
    "get_model_manager",
    # Continual Learning (v80.0)
    "ContinualLearningEngine",
    "ExperienceReplayBuffer",
    "LearningStrategy",
    "get_continual_learner",
    # Model Evaluation (v80.0)
    "ModelEvaluator",
    "EvaluationMetric",
    "BenchmarkSuite",
    "get_model_evaluator",
    # RAG Engine (v80.0)
    "RAGEngine",
    "VectorStore",
    "RetrievalStrategy",
    "get_rag_engine",
    # Mixture of Experts (v80.0)
    "MixtureOfExpertsRouter",
    "ExpertOutput",
    "GatingNetwork",
    "get_moe_router",
    # Self-Modification (v80.0)
    "SelfModificationEngine",
    "SafetyConstraints",
    "ModificationStrategy",
    "get_self_modifier",
    # Knowledge Distillation (v80.0)
    "KnowledgeDistillationEngine",
    "DistillationMode",
    "DistillationConfig",
    # Active Learning (v80.0)
    "ActiveLearningEngine",
    "ActiveLearningStrategy",
    "ActiveLearningConfig",
    # Neural Architecture Search (v80.0)
    "NeuralArchitectureSearch",
    "NASSearchStrategy",
    "NASSearchSpace",
    "Architecture",
    # Bayesian Optimization (v80.0)
    "BayesianOptimizer",
    "AcquisitionFunction",
    "HyperparameterSpace",
    # Evolutionary Optimization (v80.0)
    "EvolutionaryOptimizer",
    "EvolutionaryConfig",
    "SelectionMethod",
    # Infrastructure Integration (v80.0)
    "InfrastructureIntegration",
    "get_infrastructure",
    "SelfImprovementInfrastructure",
    "get_improvement_infrastructure",
]
