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

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


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

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    async def initialize(self) -> bool:
        """Initialize all AGI subsystems."""
        async with self._init_lock:
            if self._initialized:
                return True

            logger.info("Initializing AGI Integration Hub...")

            try:
                # Initialize subsystems in parallel where possible
                init_tasks = []

                if self._config.enable_hardware_optimization:
                    init_tasks.append(self._init_hardware())

                if self._config.enable_orchestrator:
                    init_tasks.append(self._init_orchestrator())

                if self._config.enable_reasoning:
                    init_tasks.append(self._init_reasoning())

                if self._config.enable_learning:
                    init_tasks.append(self._init_learning())

                if self._config.enable_multimodal:
                    init_tasks.append(self._init_multimodal())

                results = await asyncio.gather(*init_tasks, return_exceptions=True)

                # Check results
                success_count = sum(1 for r in results if r is True)
                error_count = sum(1 for r in results if isinstance(r, Exception))

                if error_count > 0:
                    logger.warning(
                        f"AGI Hub initialized with {error_count} subsystem failures"
                    )

                self._initialized = success_count > 0

                logger.info(
                    f"AGI Integration Hub initialized: {success_count}/{len(init_tasks)} "
                    f"subsystems active"
                )

                return self._initialized

            except Exception as e:
                logger.error(f"Failed to initialize AGI Hub: {e}")
                return False

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

    async def shutdown(self) -> None:
        """Gracefully shutdown all subsystems."""
        logger.info("Shutting down AGI Integration Hub...")

        shutdown_tasks = []

        if self._learning_engine:
            shutdown_tasks.append(self._learning_engine.shutdown())

        if self._orchestrator:
            shutdown_tasks.append(self._orchestrator.shutdown())

        if self._reasoning_engine:
            shutdown_tasks.append(self._reasoning_engine.shutdown())

        if self._multimodal_engine:
            shutdown_tasks.append(self._multimodal_engine.shutdown())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self._initialized = False
        logger.info("AGI Integration Hub shutdown complete")

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
        return {
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
# SINGLETON INSTANCE
# =============================================================================


_global_hub: Optional[AGIIntegrationHub] = None
_hub_lock = asyncio.Lock()


async def get_agi_hub(config: Optional[AGIHubConfig] = None) -> AGIIntegrationHub:
    """Get or create the global AGI Integration Hub singleton."""
    global _global_hub

    async with _hub_lock:
        if _global_hub is None:
            _global_hub = AGIIntegrationHub(config)
            await _global_hub.initialize()

        return _global_hub


async def shutdown_agi_hub() -> None:
    """Shutdown the global AGI Integration Hub."""
    global _global_hub

    if _global_hub is not None:
        await _global_hub.shutdown()
        _global_hub = None


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
