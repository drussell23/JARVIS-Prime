"""
JARVIS-Prime Models - Advanced AGI Model Architecture
=====================================================

v141.0 - "HOLLOW CLIENT" - STRICT LAZY IMPORTS

This package now uses STRICT LAZY IMPORTS to prevent OOM crashes on startup.
In Slim Mode, importing this package consumes ~0MB instead of ~4-6GB.

CRITICAL FIX (v141.0):
    The previous implementation eagerly imported torch, transformers, and
    other heavy ML libraries at module load time. This caused 4-6GB RAM
    consumption BEFORE any Slim Mode logic could run, crashing 16GB systems.

    Now all heavy imports are DEFERRED until actually needed. The package
    exports lazy accessor functions that only import when called.

FEATURES:
    - Dynamic model loading and unloading
    - Continual learning from interactions
    - Mixture of Experts routing
    - Retrieval-augmented generation
    - Comprehensive evaluation framework
    - Self-improvement with safety constraints
    - **v141.0: ZERO startup RAM for Slim Mode**

USAGE:
    # OLD (BROKEN - eager import, 4GB RAM):
    # from jarvis_prime.models import PrimeModel

    # NEW (v141.0 - lazy import, 0MB until called):
    from jarvis_prime.models import get_prime_model_class
    PrimeModel = get_prime_model_class()  # Only loads when needed
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

# =============================================================================
# v141.0: HOLLOW CLIENT - STRICT LAZY IMPORTS
# =============================================================================
# This module implements the "Hollow Client" pattern for Slim Mode.
#
# PROBLEM: Python imports ALL top-level code when a module is imported.
#          Heavy ML libraries (torch, transformers) consume 4-6GB immediately.
#
# SOLUTION: Use lazy imports - only import heavy modules INSIDE functions,
#           and only when those functions are actually called.
#
# RESULT: In Slim Mode, jarvis-prime starts with ~100MB RAM instead of ~6GB.
# =============================================================================

# Check if we're in Slim Mode (don't even import heavy modules)
SLIM_MODE = os.environ.get("JARVIS_ENABLE_SLIM_MODE", "false").lower() == "true"

if SLIM_MODE:
    logger.info(
        "[v141.0] 🪶 HOLLOW CLIENT MODE: Heavy ML imports DISABLED. "
        "All model requests will be routed to GCP."
    )


# =============================================================================
# TYPE CHECKING ONLY - These imports are ONLY for IDE type hints, NOT runtime
# =============================================================================
if TYPE_CHECKING:
    # These are only imported by type checkers (mypy, pyright), NOT at runtime
    from jarvis_prime.models.prime_model import PrimeModel, PrimeModelConfig
    from jarvis_prime.models.llama_model import LlamaModel
    from jarvis_prime.models.agi_model_orchestrator import (
        AGIModelManager, AGIModelType, ModelCapability,
        ModelEvaluator, EvaluationMetric, BenchmarkSuite,
        MixtureOfExpertsRouter, ExpertOutput, GatingNetwork,
        SparseMoERouter, SparseMoEConfig, SparseGatingMechanism,
        SparseTransformerBlock, SparseTransformerConfig, SparseAttentionPattern,
        ReactorCoreTrainingPipeline, TrainingJobConfig, TrainingJob, TrainingJobStatus,
        HuggingFaceModelPublisher, ModelCardConfig,
        ComprehensiveBenchmarkSuite, ComprehensiveBenchmarkResult, BenchmarkCategory,
    )
    from jarvis_prime.models.continual_learning_system import (
        ContinualLearningEngine, ExperienceReplayBuffer, LearningStrategy,
        RAGEngine, VectorStore, RetrievalStrategy,
        KnowledgeDistillationEngine, DistillationMode, DistillationConfig,
        ActiveLearningEngine, ActiveLearningStrategy, ActiveLearningConfig,
        InfrastructureIntegration,
    )
    from jarvis_prime.models.self_improving_agent import (
        SelfModificationEngine, SafetyConstraints, ModificationStrategy,
        NeuralArchitectureSearch, NASSearchStrategy, NASSearchSpace, Architecture,
        BayesianOptimizer, AcquisitionFunction, HyperparameterSpace,
        EvolutionaryOptimizer, EvolutionaryConfig, SelectionMethod,
        SelfImprovementInfrastructure,
    )


# =============================================================================
# v141.0: LAZY IMPORT CACHE - Prevents re-importing on every call
# =============================================================================
_lazy_cache: Dict[str, Any] = {}


def _lazy_import(module_path: str, class_name: str) -> Any:
    """
    v141.0: Lazily import a class from a module.

    This function ONLY imports the module when called, not at package load time.
    Results are cached to prevent re-importing on every call.

    Args:
        module_path: Full module path (e.g., "jarvis_prime.models.prime_model")
        class_name: Name of the class to import

    Returns:
        The imported class

    Raises:
        ImportError: If in Slim Mode and trying to import heavy modules
    """
    cache_key = f"{module_path}.{class_name}"

    if cache_key in _lazy_cache:
        return _lazy_cache[cache_key]

    # v141.0: Block heavy imports in Slim Mode
    if SLIM_MODE:
        heavy_modules = [
            "prime_model", "llama_model", "unified_inference",
            "agi_model_orchestrator", "continual_learning_system",
            "self_improving_agent",
        ]
        if any(m in module_path for m in heavy_modules):
            raise ImportError(
                f"[v141.0] HOLLOW CLIENT: Cannot import {class_name} from {module_path} "
                f"in Slim Mode. This module requires PyTorch/Transformers which are "
                f"disabled to prevent OOM. Use GCP for heavy inference."
            )

    # Actually import the module
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    _lazy_cache[cache_key] = cls
    logger.debug(f"[v141.0] Lazy imported {class_name} from {module_path}")

    return cls


# =============================================================================
# v141.0: LAZY ACCESSOR FUNCTIONS - The public API
# =============================================================================
# These functions return the actual classes, but only import them when called.
# In Slim Mode, calling these will raise an ImportError with a helpful message.
# =============================================================================

def get_prime_model_class() -> Type["PrimeModel"]:
    """Get PrimeModel class (lazy import)."""
    return _lazy_import("jarvis_prime.models.prime_model", "PrimeModel")

def get_prime_model_config_class() -> Type["PrimeModelConfig"]:
    """Get PrimeModelConfig class (lazy import)."""
    return _lazy_import("jarvis_prime.models.prime_model", "PrimeModelConfig")

def get_llama_model_class() -> Type["LlamaModel"]:
    """Get LlamaModel class (lazy import)."""
    return _lazy_import("jarvis_prime.models.llama_model", "LlamaModel")

def get_agi_model_manager_class() -> Type["AGIModelManager"]:
    """Get AGIModelManager class (lazy import)."""
    return _lazy_import("jarvis_prime.models.agi_model_orchestrator", "AGIModelManager")

def get_continual_learning_engine_class() -> Type["ContinualLearningEngine"]:
    """Get ContinualLearningEngine class (lazy import)."""
    return _lazy_import("jarvis_prime.models.continual_learning_system", "ContinualLearningEngine")

def get_rag_engine_class() -> Type["RAGEngine"]:
    """Get RAGEngine class (lazy import)."""
    return _lazy_import("jarvis_prime.models.continual_learning_system", "RAGEngine")

def get_self_modification_engine_class() -> Type["SelfModificationEngine"]:
    """Get SelfModificationEngine class (lazy import)."""
    return _lazy_import("jarvis_prime.models.self_improving_agent", "SelfModificationEngine")


# =============================================================================
# v141.0: BACKWARDS COMPATIBILITY - __getattr__ for direct attribute access
# =============================================================================
# This allows old code like `from jarvis_prime.models import PrimeModel` to
# still work, but the import is deferred until the attribute is accessed.
# =============================================================================

def __getattr__(name: str) -> Any:
    """
    v141.0: Lazy attribute access for backwards compatibility.

    This is called when an attribute is accessed that doesn't exist in the
    module's namespace. We use it to provide lazy imports for old-style imports.

    Example:
        from jarvis_prime.models import PrimeModel
        # ^ This calls __getattr__("PrimeModel") which does a lazy import
    """
    # Map attribute names to their module paths and class names
    _lazy_map = {
        # Base models
        "PrimeModel": ("jarvis_prime.models.prime_model", "PrimeModel"),
        "PrimeModelConfig": ("jarvis_prime.models.prime_model", "PrimeModelConfig"),
        "PrimeConfig": ("jarvis_prime.models.prime_model", "PrimeModelConfig"),
        "LlamaModel": ("jarvis_prime.models.llama_model", "LlamaModel"),
        "load_llama_13b_gcp": ("jarvis_prime.models.llama_model", "load_llama_13b_gcp"),
        "load_llama_13b_m1": ("jarvis_prime.models.llama_model", "load_llama_13b_m1"),
        "load_from_config": ("jarvis_prime.models.llama_model", "load_from_config"),

        # AGI Model Orchestrator
        "AGIModelManager": ("jarvis_prime.models.agi_model_orchestrator", "AGIModelManager"),
        "AGIModelType": ("jarvis_prime.models.agi_model_orchestrator", "AGIModelType"),
        "ModelCapability": ("jarvis_prime.models.agi_model_orchestrator", "ModelCapability"),
        "get_model_manager": ("jarvis_prime.models.agi_model_orchestrator", "get_model_manager"),
        "ModelEvaluator": ("jarvis_prime.models.agi_model_orchestrator", "ModelEvaluator"),
        "EvaluationMetric": ("jarvis_prime.models.agi_model_orchestrator", "EvaluationMetric"),
        "BenchmarkSuite": ("jarvis_prime.models.agi_model_orchestrator", "BenchmarkSuite"),
        "get_model_evaluator": ("jarvis_prime.models.agi_model_orchestrator", "get_model_evaluator"),
        "MixtureOfExpertsRouter": ("jarvis_prime.models.agi_model_orchestrator", "MixtureOfExpertsRouter"),
        "ExpertOutput": ("jarvis_prime.models.agi_model_orchestrator", "ExpertOutput"),
        "GatingNetwork": ("jarvis_prime.models.agi_model_orchestrator", "GatingNetwork"),
        "get_moe_router": ("jarvis_prime.models.agi_model_orchestrator", "get_moe_router"),
        "SparseMoERouter": ("jarvis_prime.models.agi_model_orchestrator", "SparseMoERouter"),
        "SparseMoEConfig": ("jarvis_prime.models.agi_model_orchestrator", "SparseMoEConfig"),
        "SparseGatingMechanism": ("jarvis_prime.models.agi_model_orchestrator", "SparseGatingMechanism"),
        "SparseTransformerBlock": ("jarvis_prime.models.agi_model_orchestrator", "SparseTransformerBlock"),
        "SparseTransformerConfig": ("jarvis_prime.models.agi_model_orchestrator", "SparseTransformerConfig"),
        "SparseAttentionPattern": ("jarvis_prime.models.agi_model_orchestrator", "SparseAttentionPattern"),
        "ReactorCoreTrainingPipeline": ("jarvis_prime.models.agi_model_orchestrator", "ReactorCoreTrainingPipeline"),
        "TrainingJobConfig": ("jarvis_prime.models.agi_model_orchestrator", "TrainingJobConfig"),
        "TrainingJob": ("jarvis_prime.models.agi_model_orchestrator", "TrainingJob"),
        "TrainingJobStatus": ("jarvis_prime.models.agi_model_orchestrator", "TrainingJobStatus"),
        "HuggingFaceModelPublisher": ("jarvis_prime.models.agi_model_orchestrator", "HuggingFaceModelPublisher"),
        "ModelCardConfig": ("jarvis_prime.models.agi_model_orchestrator", "ModelCardConfig"),
        "ComprehensiveBenchmarkSuite": ("jarvis_prime.models.agi_model_orchestrator", "ComprehensiveBenchmarkSuite"),
        "ComprehensiveBenchmarkResult": ("jarvis_prime.models.agi_model_orchestrator", "ComprehensiveBenchmarkResult"),
        "BenchmarkCategory": ("jarvis_prime.models.agi_model_orchestrator", "BenchmarkCategory"),

        # Continual Learning System
        "ContinualLearningEngine": ("jarvis_prime.models.continual_learning_system", "ContinualLearningEngine"),
        "ExperienceReplayBuffer": ("jarvis_prime.models.continual_learning_system", "ExperienceReplayBuffer"),
        "LearningStrategy": ("jarvis_prime.models.continual_learning_system", "LearningStrategy"),
        "get_continual_learner": ("jarvis_prime.models.continual_learning_system", "get_continual_learner"),
        "RAGEngine": ("jarvis_prime.models.continual_learning_system", "RAGEngine"),
        "VectorStore": ("jarvis_prime.models.continual_learning_system", "VectorStore"),
        "RetrievalStrategy": ("jarvis_prime.models.continual_learning_system", "RetrievalStrategy"),
        "get_rag_engine": ("jarvis_prime.models.continual_learning_system", "get_rag_engine"),
        "KnowledgeDistillationEngine": ("jarvis_prime.models.continual_learning_system", "KnowledgeDistillationEngine"),
        "DistillationMode": ("jarvis_prime.models.continual_learning_system", "DistillationMode"),
        "DistillationConfig": ("jarvis_prime.models.continual_learning_system", "DistillationConfig"),
        "ActiveLearningEngine": ("jarvis_prime.models.continual_learning_system", "ActiveLearningEngine"),
        "ActiveLearningStrategy": ("jarvis_prime.models.continual_learning_system", "ActiveLearningStrategy"),
        "ActiveLearningConfig": ("jarvis_prime.models.continual_learning_system", "ActiveLearningConfig"),
        "InfrastructureIntegration": ("jarvis_prime.models.continual_learning_system", "InfrastructureIntegration"),
        "get_infrastructure": ("jarvis_prime.models.continual_learning_system", "get_infrastructure"),

        # Self-Improving Agent
        "SelfModificationEngine": ("jarvis_prime.models.self_improving_agent", "SelfModificationEngine"),
        "SafetyConstraints": ("jarvis_prime.models.self_improving_agent", "SafetyConstraints"),
        "ModificationStrategy": ("jarvis_prime.models.self_improving_agent", "ModificationStrategy"),
        "get_self_modifier": ("jarvis_prime.models.self_improving_agent", "get_self_modifier"),
        "NeuralArchitectureSearch": ("jarvis_prime.models.self_improving_agent", "NeuralArchitectureSearch"),
        "NASSearchStrategy": ("jarvis_prime.models.self_improving_agent", "NASSearchStrategy"),
        "NASSearchSpace": ("jarvis_prime.models.self_improving_agent", "NASSearchSpace"),
        "Architecture": ("jarvis_prime.models.self_improving_agent", "Architecture"),
        "BayesianOptimizer": ("jarvis_prime.models.self_improving_agent", "BayesianOptimizer"),
        "AcquisitionFunction": ("jarvis_prime.models.self_improving_agent", "AcquisitionFunction"),
        "HyperparameterSpace": ("jarvis_prime.models.self_improving_agent", "HyperparameterSpace"),
        "EvolutionaryOptimizer": ("jarvis_prime.models.self_improving_agent", "EvolutionaryOptimizer"),
        "EvolutionaryConfig": ("jarvis_prime.models.self_improving_agent", "EvolutionaryConfig"),
        "SelectionMethod": ("jarvis_prime.models.self_improving_agent", "SelectionMethod"),
        "SelfImprovementInfrastructure": ("jarvis_prime.models.self_improving_agent", "SelfImprovementInfrastructure"),
        "get_improvement_infrastructure": ("jarvis_prime.models.self_improving_agent", "get_improvement_infrastructure"),
    }

    if name in _lazy_map:
        module_path, class_name = _lazy_map[name]
        return _lazy_import(module_path, class_name)

    raise AttributeError(f"module 'jarvis_prime.models' has no attribute '{name}'")

# =============================================================================
# v141.0: EXPORTS - All names are lazily loaded via __getattr__
# =============================================================================
__all__ = [
    # v141.0: Lazy accessor functions (recommended for new code)
    "get_prime_model_class",
    "get_prime_model_config_class",
    "get_llama_model_class",
    "get_agi_model_manager_class",
    "get_continual_learning_engine_class",
    "get_rag_engine_class",
    "get_self_modification_engine_class",

    # v141.0: Slim Mode flag
    "SLIM_MODE",

    # Backwards-compatible names (lazily loaded via __getattr__)
    # Base Models
    "PrimeModel",
    "PrimeConfig",
    "PrimeModelConfig",
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
    # Sparse MoE (v80.0 Advanced)
    "SparseMoERouter",
    "SparseMoEConfig",
    "SparseGatingMechanism",
    # Sparse Transformers (v80.0 Advanced)
    "SparseTransformerBlock",
    "SparseTransformerConfig",
    "SparseAttentionPattern",
    # Reactor Core Training Pipeline (v80.0 Advanced)
    "ReactorCoreTrainingPipeline",
    "TrainingJobConfig",
    "TrainingJob",
    "TrainingJobStatus",
    # HuggingFace Model Publisher (v80.0 Advanced)
    "HuggingFaceModelPublisher",
    "ModelCardConfig",
    # Comprehensive Benchmark Suite (v80.0 Advanced)
    "ComprehensiveBenchmarkSuite",
    "ComprehensiveBenchmarkResult",
    "BenchmarkCategory",
]
