#!/usr/bin/env python3
"""
Dynamic Model Registry v99.0 - Neural Switchboard + Multi-Directory Discovery
===============================================================================

ENTERPRISE NEURAL SWITCHBOARD:
    - Intelligent task classification with multi-signal analysis
    - macOS native memory_pressure integration
    - Cross-repo JARVIS AdvancedRAMMonitor coordination
    - Request buffering during hot swap (zero request loss)
    - Sticky routing for session continuity
    - Unified burst decision making (single source of truth)
    - Predictive model preloading based on task patterns

v99.0 ENHANCEMENTS:
    - Multi-directory model scanning with configurable search paths
    - File system watching with Watchdog for real-time model discovery
    - Model validation pipeline (SHA256 verification, format detection)
    - Fuzzy matching for unknown models
    - Reactor Core integration for cross-repo model sync

ADVANCED PATTERNS IMPLEMENTED:
    - Protocol classes for type-safe model interfaces
    - Async semaphores for thundering herd prevention
    - Exponential backoff with decorrelated jitter
    - Model warm-up and predictive preloading
    - Memory pressure monitoring with automatic unloading
    - HuggingFace Hub integration for reliable downloads
    - SHA256 checksum verification with retry
    - Elastic burst controller for cloud scaling
    - Circuit breakers per model endpoint
    - Observer pattern for model state changes
    - LRU eviction with priority-based retention
    - Real-time file system watching (polling + watchdog)
    - Parallel async directory scanning

SUPPORTED MODELS (v97.0 Arsenal):
┌────────────────────────────────────────────────────────────────────────────────┐
│ Tier      │ Model                │ Params  │ VRAM    │ Speed    │ Best For     │
├───────────┼──────────────────────┼─────────┼─────────┼──────────┼──────────────┤
│ 0-Ultra   │ Phi-3.5-mini         │ 3.8B    │ 2.5GB   │ 55 t/s   │ Voice/Quick  │
│ 0-Fast    │ DeepSeek-R1-8B       │ 8B      │ 5GB     │ 25 t/s   │ Reasoning    │
│ 0-Fast    │ Qwen-2.5-7B          │ 7B      │ 4.5GB   │ 28 t/s   │ General      │
│ 0.5-Local │ Qwen-2.5-32B         │ 32B     │ 20GB    │ 8 t/s    │ Coding King  │
│ 0.5-Local │ Qwen-2.5-Coder-32B   │ 32B     │ 20GB    │ 8 t/s    │ Code Expert  │
│ 1-Cloud   │ Llama-3.3-70B        │ 70B     │ 45GB    │ 45 t/s*  │ Workhorse    │
│ 1-Cloud   │ DeepSeek-Coder-V2    │ 236B    │ 142GB   │ 40 t/s*  │ Code Master  │
│ 2-Deep    │ DeepSeek-V2.5        │ 236B    │ 142GB   │ 35 t/s*  │ GPT-5 Level  │
└────────────────────────────────────────────────────────────────────────────────┘
* Cloud inference on A100 80GB

Usage:
    registry = await get_dynamic_model_registry()

    # Neural selection based on task
    model, path = await registry.neural_select(
        prompt="Implement distributed cache",
        complexity=0.75,
        requirements=["code", "architecture"]
    )

    # Elastic burst when RAM pressure high
    if await registry.should_burst_to_cloud():
        model = await registry.get_cloud_model("tier_1_cloud_intelligent")
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    Final,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# PROTOCOLS (Type-Safe Interfaces)
# =============================================================================


@runtime_checkable
class ModelBackendProtocol(Protocol):
    """Protocol for model inference backends (llama.cpp, vllm, etc.)."""

    async def load(self, model_path: Path) -> bool: ...
    async def unload(self) -> bool: ...
    async def generate(self, prompt: str, **kwargs: Any) -> str: ...
    def is_loaded(self) -> bool: ...
    def get_memory_usage(self) -> int: ...


@runtime_checkable
class DownloadProgressCallback(Protocol):
    """Protocol for download progress callbacks."""

    async def on_progress(
        self,
        model_id: str,
        downloaded: int,
        total: int,
        speed_bps: float,
    ) -> None: ...


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class ModelCapability(Enum):
    """Model capabilities for routing decisions."""

    GENERAL_CHAT = "general_chat"
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    CODE_EXPERT = "code_expert"  # Advanced coding
    REASONING = "reasoning"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    MATH = "math"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    INSTRUCTION_FOLLOWING = "instruction_following"
    MULTI_TURN = "multi_turn"
    LONG_CONTEXT = "long_context"
    FAST_INFERENCE = "fast_inference"
    VOICE_OPTIMIZED = "voice_optimized"
    ARCHITECTURE_DESIGN = "architecture_design"
    VISION = "vision"


class ModelTierLevel(Enum):
    """Model tier classification."""

    TIER_0_ULTRA_FAST = 0  # < 4B params, ultra-fast voice
    TIER_0_FAST = 1  # 4-10B params, fast local
    TIER_05_CAPABLE = 2  # 10-40B params, capable local
    TIER_1_CLOUD = 3  # 40-80B params, cloud intelligent
    TIER_2_DEEP = 4  # 80B+ params, deep reasoning (GPT-5 level)


class ModelLocation(Enum):
    """Where the model runs."""

    LOCAL_ONLY = "local_only"  # Fits in 16-32GB RAM
    CLOUD_ONLY = "cloud_only"  # Requires A100 80GB+
    HYBRID = "hybrid"  # Can run locally or cloud


class ModelState(Enum):
    """Model lifecycle state."""

    UNKNOWN = "unknown"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    LOADED = "loaded"
    WARMING_UP = "warming_up"
    READY = "ready"
    UNLOADING = "unloading"
    ERROR = "error"


class QuantizationType(Enum):
    """GGUF quantization types."""

    Q4_K_M = "Q4_K_M"  # Best balance (recommended)
    Q4_K_S = "Q4_K_S"  # Smaller, faster
    Q5_K_M = "Q5_K_M"  # Higher quality
    Q5_K_S = "Q5_K_S"
    Q6_K = "Q6_K"  # Near FP16 quality
    Q8_0 = "Q8_0"  # Highest quality GGUF
    IQ2_XXS = "IQ2_XXS"  # Ultra compressed
    IQ3_XS = "IQ3_XS"
    FP16 = "FP16"  # Full precision


class TaskType(Enum):
    """Unified task classification for Neural Switchboard."""

    # Simple tasks (Tier 0)
    GREETING = "greeting"
    SIMPLE_CHAT = "simple_chat"
    QUICK_QUESTION = "quick_question"

    # Standard tasks (Tier 0 Fast)
    GENERAL_CHAT = "general_chat"
    SUMMARIZE = "summarize"
    TRANSLATE = "translate"

    # Code tasks (Tier 0.5)
    CODE_SIMPLE = "code_simple"  # Small edits, bug fixes
    CODE_REVIEW = "code_review"
    CODE_EXPLAIN = "code_explain"

    # Complex code tasks (Tier 0.5 / 1)
    CODE_COMPLEX = "code_complex"  # New features, refactoring
    CODE_ARCHITECTURE = "code_architecture"  # Design decisions
    CODE_DEBUG = "code_debug"  # Complex debugging

    # Reasoning tasks (Tier 1 / 2)
    REASON_SIMPLE = "reason_simple"
    REASON_COMPLEX = "reason_complex"
    MATH_SIMPLE = "math_simple"
    MATH_COMPLEX = "math_complex"
    ANALYZE = "analyze"

    # Creative tasks
    CREATIVE_WRITE = "creative_write"
    CREATIVE_BRAINSTORM = "creative_brainstorm"

    # Special tasks
    VOICE_COMMAND = "voice_command"  # Voice interface (requires fast)
    MULTIMODAL = "multimodal"
    SEARCH = "search"
    UNKNOWN = "unknown"


class MemoryPressureLevel(Enum):
    """macOS memory pressure levels from memory_pressure command."""

    NORMAL = "normal"  # Green - plenty of free RAM
    WARN = "warn"  # Yellow - approaching limits
    CRITICAL = "critical"  # Red - system is swapping


# Task to tier mapping (which tier handles which tasks best)
TASK_TIER_MAPPING: Dict[TaskType, List[str]] = {
    # Ultra-fast tier (voice, greetings)
    TaskType.GREETING: ["tier_0_local_fast"],
    TaskType.VOICE_COMMAND: ["tier_0_local_fast"],
    TaskType.QUICK_QUESTION: ["tier_0_local_fast"],
    TaskType.SIMPLE_CHAT: ["tier_0_local_fast"],

    # Fast tier (general tasks)
    TaskType.GENERAL_CHAT: ["tier_0_local_fast", "tier_05_local_capable"],
    TaskType.SUMMARIZE: ["tier_0_local_fast", "tier_05_local_capable"],
    TaskType.TRANSLATE: ["tier_0_local_fast"],

    # Capable tier (coding)
    TaskType.CODE_SIMPLE: ["tier_05_local_capable", "tier_0_local_fast"],
    TaskType.CODE_REVIEW: ["tier_05_local_capable"],
    TaskType.CODE_EXPLAIN: ["tier_05_local_capable", "tier_0_local_fast"],
    TaskType.CODE_COMPLEX: ["tier_05_local_capable", "tier_1_cloud_intelligent"],
    TaskType.CODE_ARCHITECTURE: ["tier_1_cloud_intelligent", "tier_05_local_capable"],
    TaskType.CODE_DEBUG: ["tier_05_local_capable", "tier_1_cloud_intelligent"],

    # Reasoning tier (complex tasks)
    TaskType.REASON_SIMPLE: ["tier_0_local_fast", "tier_05_local_capable"],
    TaskType.REASON_COMPLEX: ["tier_1_cloud_intelligent", "tier_2_deep_reasoning"],
    TaskType.MATH_SIMPLE: ["tier_0_local_fast"],
    TaskType.MATH_COMPLEX: ["tier_1_cloud_intelligent", "tier_2_deep_reasoning"],
    TaskType.ANALYZE: ["tier_05_local_capable", "tier_1_cloud_intelligent"],

    # Creative tier
    TaskType.CREATIVE_WRITE: ["tier_05_local_capable", "tier_1_cloud_intelligent"],
    TaskType.CREATIVE_BRAINSTORM: ["tier_1_cloud_intelligent"],

    # Special
    TaskType.MULTIMODAL: ["tier_1_cloud_intelligent"],
    TaskType.SEARCH: ["tier_0_local_fast"],
    TaskType.UNKNOWN: ["tier_0_local_fast", "tier_05_local_capable"],
}


# Task type to required capabilities mapping
TASK_CAPABILITY_MAPPING: Dict[TaskType, List[ModelCapability]] = {
    TaskType.VOICE_COMMAND: [ModelCapability.VOICE_OPTIMIZED, ModelCapability.FAST_INFERENCE],
    TaskType.CODE_SIMPLE: [ModelCapability.CODE_GENERATION],
    TaskType.CODE_COMPLEX: [ModelCapability.CODE_GENERATION, ModelCapability.CODE_ANALYSIS],
    TaskType.CODE_ARCHITECTURE: [ModelCapability.CODE_EXPERT, ModelCapability.ARCHITECTURE_DESIGN],
    TaskType.CODE_DEBUG: [ModelCapability.CODE_ANALYSIS, ModelCapability.REASONING],
    TaskType.REASON_COMPLEX: [ModelCapability.REASONING, ModelCapability.CHAIN_OF_THOUGHT],
    TaskType.MATH_COMPLEX: [ModelCapability.MATH, ModelCapability.CHAIN_OF_THOUGHT],
    TaskType.CREATIVE_WRITE: [ModelCapability.CREATIVE],
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class QuantizationSpec:
    """Quantization specification."""

    type: QuantizationType
    bits: float
    size_multiplier: float  # vs FP16 (1.0)
    quality_retention: float  # 0-1
    recommended_for: Tuple[str, ...]


QUANTIZATION_SPECS: Dict[QuantizationType, QuantizationSpec] = {
    QuantizationType.Q4_K_M: QuantizationSpec(
        type=QuantizationType.Q4_K_M,
        bits=4.5,
        size_multiplier=0.28,
        quality_retention=0.95,
        recommended_for=("balanced", "speed", "default"),
    ),
    QuantizationType.Q5_K_M: QuantizationSpec(
        type=QuantizationType.Q5_K_M,
        bits=5.5,
        size_multiplier=0.34,
        quality_retention=0.97,
        recommended_for=("quality", "balanced"),
    ),
    QuantizationType.Q6_K: QuantizationSpec(
        type=QuantizationType.Q6_K,
        bits=6.5,
        size_multiplier=0.41,
        quality_retention=0.98,
        recommended_for=("quality",),
    ),
    QuantizationType.Q8_0: QuantizationSpec(
        type=QuantizationType.Q8_0,
        bits=8.0,
        size_multiplier=0.50,
        quality_retention=0.99,
        recommended_for=("accuracy", "quality"),
    ),
    QuantizationType.IQ2_XXS: QuantizationSpec(
        type=QuantizationType.IQ2_XXS,
        bits=2.0,
        size_multiplier=0.13,
        quality_retention=0.85,
        recommended_for=("compression", "cloud"),
    ),
}


@dataclass
class ModelSpec:
    """Complete specification for a model."""

    # Identity (required fields first)
    model_id: str
    name: str
    description: str

    # HuggingFace source
    hf_repo: str
    hf_filename_template: str  # Use {quant} placeholder

    # Model characteristics
    parameter_count: str  # Human readable "32B"
    parameter_count_billions: float
    context_length: int
    architecture: str  # llama, qwen2, phi3, deepseek2

    # Classification
    tier_level: ModelTierLevel
    location: ModelLocation
    capabilities: List[ModelCapability]
    primary_use_cases: List[str]

    # Performance (M1 Mac / A100)
    tokens_per_second_m1: float
    tokens_per_second_a100: float
    memory_required_gb: float

    # Optional fields with defaults below
    provider: str = ""  # Model provider (Microsoft, Qwen, Meta, DeepSeek, etc.)
    optimal_batch_size: int = 1

    # Cost (cloud models)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    cost_per_hour_a100: float = 0.0

    # Quantization variants
    available_quantizations: List[QuantizationType] = field(
        default_factory=lambda: [QuantizationType.Q4_K_M]
    )
    default_quantization: QuantizationType = QuantizationType.Q4_K_M

    # Checksums (quant_type -> sha256)
    checksums: Dict[str, str] = field(default_factory=dict)

    # Metadata
    release_date: str = ""
    license: str = "unknown"
    is_moe: bool = False  # Mixture of Experts
    moe_active_params: Optional[float] = None

    def get_filename(self, quant: QuantizationType = None) -> str:
        """Get filename for quantization."""
        quant = quant or self.default_quantization
        return self.hf_filename_template.format(quant=quant.value)

    def get_download_url(self, quant: QuantizationType = None) -> str:
        """Get full HuggingFace download URL."""
        filename = self.get_filename(quant)
        return f"https://huggingface.co/{self.hf_repo}/resolve/main/{filename}"

    def estimated_size_gb(self, quant: QuantizationType = None) -> float:
        """Estimate file size for quantization."""
        quant = quant or self.default_quantization
        spec = QUANTIZATION_SPECS.get(quant)
        if spec:
            # Rough estimate: 2 bytes per param for FP16
            fp16_size = self.parameter_count_billions * 2
            return fp16_size * spec.size_multiplier
        return self.memory_required_gb * 0.8


# =============================================================================
# KNOWN MODELS CATALOG v97.0
# =============================================================================


KNOWN_MODELS: Dict[str, ModelSpec] = {
    # =========================================================================
    # TIER 0: ULTRA-FAST (Voice Interface)
    # =========================================================================
    "phi-3.5-mini": ModelSpec(
        model_id="phi-3.5-mini",
        name="Phi-3.5 Mini Instruct",
        description="Microsoft's ultra-efficient 3.8B - perfect for voice with 128K context",
        hf_repo="bartowski/Phi-3.5-mini-instruct-GGUF",
        hf_filename_template="Phi-3.5-mini-instruct-{quant}.gguf",
        parameter_count="3.8B",
        parameter_count_billions=3.8,
        context_length=128000,
        architecture="phi3",
        tier_level=ModelTierLevel.TIER_0_ULTRA_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=55.0,
        tokens_per_second_a100=150.0,
        memory_required_gb=2.5,
        capabilities=[
            ModelCapability.GENERAL_CHAT,
            ModelCapability.INSTRUCTION_FOLLOWING,
            ModelCapability.FAST_INFERENCE,
            ModelCapability.VOICE_OPTIMIZED,
            ModelCapability.LONG_CONTEXT,
        ],
        primary_use_cases=["voice_interface", "quick_chat", "simple_tasks"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q8_0,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="MIT",
        release_date="2024-08",
    ),
    # =========================================================================
    # TIER 0: FAST (Local Reasoning)
    # =========================================================================
    "deepseek-r1-8b": ModelSpec(
        model_id="deepseek-r1-8b",
        name="DeepSeek R1 Distill Llama 8B",
        description="DeepSeek's reasoning model distilled - Chain of Thought built-in",
        hf_repo="unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF",
        hf_filename_template="DeepSeek-R1-Distill-Llama-8B-{quant}.gguf",
        parameter_count="8B",
        parameter_count_billions=8.0,
        context_length=32768,
        architecture="llama",
        tier_level=ModelTierLevel.TIER_0_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=25.0,
        tokens_per_second_a100=80.0,
        memory_required_gb=5.0,
        capabilities=[
            ModelCapability.REASONING,
            ModelCapability.CHAIN_OF_THOUGHT,
            ModelCapability.MATH,
            ModelCapability.CODE_ANALYSIS,
            ModelCapability.INSTRUCTION_FOLLOWING,
        ],
        primary_use_cases=["reasoning", "math", "planning", "analysis"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q8_0,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="MIT",
        release_date="2025-01",
    ),
    "qwen-2.5-7b": ModelSpec(
        model_id="qwen-2.5-7b",
        name="Qwen 2.5 7B Instruct",
        description="Alibaba's balanced 7B - great all-rounder with 128K context",
        hf_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        hf_filename_template="Qwen2.5-7B-Instruct-{quant}.gguf",
        parameter_count="7B",
        parameter_count_billions=7.0,
        context_length=131072,
        architecture="qwen2",
        tier_level=ModelTierLevel.TIER_0_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=28.0,
        tokens_per_second_a100=90.0,
        memory_required_gb=4.5,
        capabilities=[
            ModelCapability.GENERAL_CHAT,
            ModelCapability.CODE_GENERATION,
            ModelCapability.INSTRUCTION_FOLLOWING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.MULTI_TURN,
        ],
        primary_use_cases=["general", "coding", "chat"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q8_0,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Apache-2.0",
        release_date="2024-09",
    ),
    "llama-3.2-3b": ModelSpec(
        model_id="llama-3.2-3b",
        name="Llama 3.2 3B Instruct",
        description="Meta's compact 3B - fast and efficient for simple tasks",
        hf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF",
        hf_filename_template="Llama-3.2-3B-Instruct-{quant}.gguf",
        parameter_count="3B",
        parameter_count_billions=3.0,
        context_length=131072,
        architecture="llama",
        tier_level=ModelTierLevel.TIER_0_ULTRA_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=50.0,
        tokens_per_second_a100=140.0,
        memory_required_gb=2.0,
        capabilities=[
            ModelCapability.GENERAL_CHAT,
            ModelCapability.FAST_INFERENCE,
            ModelCapability.INSTRUCTION_FOLLOWING,
        ],
        primary_use_cases=["quick_chat", "simple_tasks"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Llama-3.2",
        release_date="2024-09",
    ),
    # =========================================================================
    # TIER 0.5: LOCAL POWERHOUSE (The "Coding King")
    # =========================================================================
    "qwen-2.5-32b": ModelSpec(
        model_id="qwen-2.5-32b",
        name="Qwen 2.5 32B Instruct",
        description="LOCAL CODING KING - Beats GPT-4 on coding, fits in 20GB",
        hf_repo="bartowski/Qwen2.5-32B-Instruct-GGUF",
        hf_filename_template="Qwen2.5-32B-Instruct-{quant}.gguf",
        parameter_count="32B",
        parameter_count_billions=32.0,
        context_length=131072,
        architecture="qwen2",
        tier_level=ModelTierLevel.TIER_05_CAPABLE,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=8.0,
        tokens_per_second_a100=45.0,
        memory_required_gb=20.0,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.CODE_ANALYSIS,
            ModelCapability.CODE_EXPERT,
            ModelCapability.REASONING,
            ModelCapability.ARCHITECTURE_DESIGN,
            ModelCapability.MATH,
            ModelCapability.CREATIVE,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.MULTI_TURN,
        ],
        primary_use_cases=[
            "complex_coding",
            "architecture_design",
            "code_review",
            "refactoring",
        ],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q6_K,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Apache-2.0",
        release_date="2024-09",
    ),
    "qwen-2.5-coder-32b": ModelSpec(
        model_id="qwen-2.5-coder-32b",
        name="Qwen 2.5 Coder 32B Instruct",
        description="Specialized coding model - SOTA on code benchmarks",
        hf_repo="bartowski/Qwen2.5-Coder-32B-Instruct-GGUF",
        hf_filename_template="Qwen2.5-Coder-32B-Instruct-{quant}.gguf",
        parameter_count="32B",
        parameter_count_billions=32.0,
        context_length=131072,
        architecture="qwen2",
        tier_level=ModelTierLevel.TIER_05_CAPABLE,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=8.0,
        tokens_per_second_a100=45.0,
        memory_required_gb=20.0,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.CODE_ANALYSIS,
            ModelCapability.CODE_EXPERT,
            ModelCapability.REASONING,
            ModelCapability.LONG_CONTEXT,
        ],
        primary_use_cases=[
            "expert_coding",
            "code_generation",
            "debugging",
            "unit_tests",
        ],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Apache-2.0",
        release_date="2024-11",
    ),
    "qwen-2.5-14b": ModelSpec(
        model_id="qwen-2.5-14b",
        name="Qwen 2.5 14B Instruct",
        description="Balanced 14B - good quality without 32B RAM needs",
        hf_repo="bartowski/Qwen2.5-14B-Instruct-GGUF",
        hf_filename_template="Qwen2.5-14B-Instruct-{quant}.gguf",
        parameter_count="14B",
        parameter_count_billions=14.0,
        context_length=131072,
        architecture="qwen2",
        tier_level=ModelTierLevel.TIER_05_CAPABLE,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=15.0,
        tokens_per_second_a100=60.0,
        memory_required_gb=10.0,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.GENERAL_CHAT,
            ModelCapability.LONG_CONTEXT,
        ],
        primary_use_cases=["coding", "reasoning", "general"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Apache-2.0",
        release_date="2024-09",
    ),
    # =========================================================================
    # TIER 1: CLOUD INTELLIGENT (A100 Required)
    # =========================================================================
    "llama-3.3-70b": ModelSpec(
        model_id="llama-3.3-70b",
        name="Llama 3.3 70B Instruct",
        description="CLOUD WORKHORSE - Meta's flagship, rock-solid performance",
        hf_repo="bartowski/Llama-3.3-70B-Instruct-GGUF",
        hf_filename_template="Llama-3.3-70B-Instruct-{quant}.gguf",
        parameter_count="70B",
        parameter_count_billions=70.0,
        context_length=128000,
        architecture="llama",
        tier_level=ModelTierLevel.TIER_1_CLOUD,
        location=ModelLocation.CLOUD_ONLY,
        tokens_per_second_m1=2.0,  # Very slow locally
        tokens_per_second_a100=45.0,
        memory_required_gb=45.0,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.CODE_ANALYSIS,
            ModelCapability.CODE_EXPERT,
            ModelCapability.REASONING,
            ModelCapability.CREATIVE,
            ModelCapability.SUMMARIZATION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.MULTI_TURN,
            ModelCapability.ARCHITECTURE_DESIGN,
        ],
        primary_use_cases=[
            "complex_tasks",
            "expert_coding",
            "documentation",
            "research",
        ],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q6_K,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0002,
        cost_per_hour_a100=1.20,
        license="Llama-3.3",
        release_date="2024-12",
    ),
    "deepseek-coder-v2": ModelSpec(
        model_id="deepseek-coder-v2",
        name="DeepSeek Coder V2 Instruct (236B MoE)",
        description="CODE MASTER - 236B MoE, beats GPT-4 Turbo on coding",
        hf_repo="bartowski/DeepSeek-Coder-V2-Instruct-GGUF",
        hf_filename_template="DeepSeek-Coder-V2-Instruct-{quant}.gguf",
        parameter_count="236B (MoE)",
        parameter_count_billions=236.0,
        context_length=128000,
        architecture="deepseek2",
        tier_level=ModelTierLevel.TIER_1_CLOUD,
        location=ModelLocation.CLOUD_ONLY,
        tokens_per_second_m1=1.0,  # Impractical locally
        tokens_per_second_a100=40.0,
        memory_required_gb=142.0,  # Q4_K_M
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.CODE_ANALYSIS,
            ModelCapability.CODE_EXPERT,
            ModelCapability.REASONING,
            ModelCapability.ARCHITECTURE_DESIGN,
            ModelCapability.LONG_CONTEXT,
        ],
        primary_use_cases=[
            "expert_coding",
            "architecture_design",
            "complex_refactoring",
            "code_review",
        ],
        available_quantizations=[
            QuantizationType.IQ2_XXS,
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        cost_per_hour_a100=1.50,
        license="DeepSeek",
        release_date="2024-06",
        is_moe=True,
        moe_active_params=21.0,  # Only 21B params active per token
    ),
    # =========================================================================
    # TIER 2: DEEP REASONING (GPT-5 Level)
    # =========================================================================
    "deepseek-v2.5": ModelSpec(
        model_id="deepseek-v2.5",
        name="DeepSeek V2.5 (236B MoE)",
        description="GPT-5 CONTENDER - SOTA open-source, beats GPT-4 Turbo overall",
        hf_repo="bartowski/DeepSeek-V2.5-GGUF",
        hf_filename_template="DeepSeek-V2.5-{quant}.gguf",
        parameter_count="236B (MoE)",
        parameter_count_billions=236.0,
        context_length=128000,
        architecture="deepseek2",
        tier_level=ModelTierLevel.TIER_2_DEEP,
        location=ModelLocation.CLOUD_ONLY,
        tokens_per_second_m1=1.0,
        tokens_per_second_a100=35.0,
        memory_required_gb=142.0,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.CODE_EXPERT,
            ModelCapability.REASONING,
            ModelCapability.CHAIN_OF_THOUGHT,
            ModelCapability.MATH,
            ModelCapability.CREATIVE,
            ModelCapability.ARCHITECTURE_DESIGN,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.MULTI_TURN,
        ],
        primary_use_cases=[
            "deep_reasoning",
            "complex_architecture",
            "research",
            "expert_analysis",
        ],
        available_quantizations=[
            QuantizationType.IQ2_XXS,  # 52.7 GB
            QuantizationType.Q4_K_M,  # 142 GB
            QuantizationType.Q5_K_M,  # 167 GB
        ],
        default_quantization=QuantizationType.Q4_K_M,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        cost_per_hour_a100=1.50,
        license="DeepSeek",
        release_date="2024-09",
        is_moe=True,
        moe_active_params=21.0,
    ),
    "deepseek-v2-lite": ModelSpec(
        model_id="deepseek-v2-lite",
        name="DeepSeek V2 Lite Chat (16B MoE)",
        description="Efficient MoE - great reasoning in smaller package",
        hf_repo="bartowski/DeepSeek-V2-Lite-Chat-GGUF",
        hf_filename_template="DeepSeek-V2-Lite-Chat-{quant}.gguf",
        parameter_count="16B (MoE)",
        parameter_count_billions=16.0,
        context_length=32768,
        architecture="deepseek2",
        tier_level=ModelTierLevel.TIER_1_CLOUD,
        location=ModelLocation.HYBRID,  # Can run locally with swap
        tokens_per_second_m1=5.0,
        tokens_per_second_a100=50.0,
        memory_required_gb=12.0,
        capabilities=[
            ModelCapability.REASONING,
            ModelCapability.MATH,
            ModelCapability.CODE_ANALYSIS,
            ModelCapability.MULTI_TURN,
        ],
        primary_use_cases=["reasoning", "math", "analysis"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        cost_per_1k_input=0.00007,
        cost_per_1k_output=0.00014,
        license="DeepSeek",
        release_date="2024-06",
        is_moe=True,
        moe_active_params=2.4,
    ),
    # =========================================================================
    # v241.0: Additional models for GCP multi-model golden image
    # =========================================================================
    "qwen-2.5-coder-7b": ModelSpec(
        model_id="qwen-2.5-coder-7b",
        name="Qwen 2.5 Coder 7B Instruct",
        description="Alibaba's code-specialized 7B — best-in-class for code generation & debugging",
        hf_repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        hf_filename_template="qwen2.5-coder-7b-instruct-{quant}.gguf",
        parameter_count="7B",
        parameter_count_billions=7.0,
        context_length=131072,
        architecture="qwen2",
        tier_level=ModelTierLevel.TIER_0_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=28.0,
        tokens_per_second_a100=90.0,
        memory_required_gb=4.5,
        capabilities=[
            ModelCapability.CODE_GENERATION,
            ModelCapability.CODE_ANALYSIS,
            ModelCapability.CODE_EXPERT,
            ModelCapability.INSTRUCTION_FOLLOWING,
            ModelCapability.LONG_CONTEXT,
        ],
        primary_use_cases=["code_generation", "code_review", "debugging", "refactoring"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q8_0,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Apache-2.0",
        release_date="2024-09",
    ),
    "llama-3.1-8b": ModelSpec(
        model_id="llama-3.1-8b",
        name="Meta Llama 3.1 8B Instruct",
        description="Meta's 8B with native 128K context — great for long documents & creative tasks",
        hf_repo="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        hf_filename_template="Meta-Llama-3.1-8B-Instruct-{quant}.gguf",
        parameter_count="8B",
        parameter_count_billions=8.0,
        context_length=131072,
        architecture="llama",
        tier_level=ModelTierLevel.TIER_0_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=26.0,
        tokens_per_second_a100=85.0,
        memory_required_gb=5.0,
        capabilities=[
            ModelCapability.GENERAL_CHAT,
            ModelCapability.CREATIVE,
            ModelCapability.SUMMARIZATION,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.MULTI_TURN,
            ModelCapability.INSTRUCTION_FOLLOWING,
        ],
        primary_use_cases=["creative_writing", "summarization", "long_context", "brainstorming"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q8_0,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Llama-3.1-Community",
        release_date="2024-07",
    ),
    "tinyllama-1.1b": ModelSpec(
        model_id="tinyllama-1.1b",
        name="TinyLlama 1.1B Chat v1.0",
        description="Ultra-small 1.1B — speculative decoding draft model, not for standalone use",
        hf_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        hf_filename_template="tinyllama-1.1b-chat-v1.0.{quant}.gguf",
        parameter_count="1.1B",
        parameter_count_billions=1.1,
        context_length=2048,
        architecture="llama",
        tier_level=ModelTierLevel.TIER_0_ULTRA_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=120.0,
        tokens_per_second_a100=300.0,
        memory_required_gb=0.7,
        capabilities=[
            ModelCapability.FAST_INFERENCE,
        ],
        primary_use_cases=["speculative_decoding_draft"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Apache-2.0",
        release_date="2024-01",
    ),
    # v241.1: Advanced reasoning & specialist models for GCP golden image
    "deepseek-r1-qwen-7b": ModelSpec(
        model_id="deepseek-r1-qwen-7b",
        name="DeepSeek R1 Distill Qwen 7B",
        description="Distilled from DeepSeek-R1 (671B MoE) — explicit chain-of-thought reasoning, "
                    "excels at multi-step logic, planning, and complex analysis. AIME 2024: 55.5% vs Qwen2.5-7B ~30%",
        hf_repo="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        hf_filename_template="DeepSeek-R1-Distill-Qwen-7B-{quant}.gguf",
        parameter_count="7B",
        parameter_count_billions=7.0,
        context_length=131072,
        architecture="qwen2",
        tier_level=ModelTierLevel.TIER_0_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=28.0,
        tokens_per_second_a100=90.0,
        memory_required_gb=4.5,
        capabilities=[
            ModelCapability.CHAIN_OF_THOUGHT,
            ModelCapability.REASONING,
            ModelCapability.MATH,
            ModelCapability.INSTRUCTION_FOLLOWING,
        ],
        primary_use_cases=["complex_reasoning", "logic_puzzles", "planning", "scientific_analysis"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q8_0,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="MIT",
        release_date="2025-01",
    ),
    "gemma-2-9b": ModelSpec(
        model_id="gemma-2-9b",
        name="Google Gemma 2 9B Instruct",
        description="Google's best sub-10B generalist — tops MMLU (72.3), HellaSwag (81.9), ARC-C (68.4) "
                    "at this scale. Strongest general intelligence available in the 7-9B range",
        hf_repo="bartowski/gemma-2-9b-it-GGUF",
        hf_filename_template="gemma-2-9b-it-{quant}.gguf",
        parameter_count="9B",
        parameter_count_billions=9.0,
        context_length=8192,
        architecture="gemma2",
        tier_level=ModelTierLevel.TIER_0_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=24.0,
        tokens_per_second_a100=80.0,
        memory_required_gb=5.5,
        capabilities=[
            ModelCapability.GENERAL_CHAT,
            ModelCapability.REASONING,
            ModelCapability.INSTRUCTION_FOLLOWING,
            ModelCapability.MULTI_TURN,
        ],
        primary_use_cases=["general_knowledge", "nuanced_analysis", "instruction_following", "factual_qa"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q8_0,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Gemma-Terms-of-Use",
        release_date="2024-06",
    ),
    "qwen-2.5-math-7b": ModelSpec(
        model_id="qwen-2.5-math-7b",
        name="Qwen 2.5 Math 7B Instruct",
        description="Alibaba's dedicated math specialist — 83.6% on MATH benchmark (vs GPT-4 ~76%). "
                    "Supports chain-of-thought and tool-integrated reasoning for proofs and calculus",
        hf_repo="Qwen/Qwen2.5-Math-7B-Instruct-GGUF",
        hf_filename_template="qwen2.5-math-7b-instruct-{quant}.gguf",
        parameter_count="7B",
        parameter_count_billions=7.0,
        context_length=4096,
        architecture="qwen2",
        tier_level=ModelTierLevel.TIER_0_FAST,
        location=ModelLocation.LOCAL_ONLY,
        tokens_per_second_m1=28.0,
        tokens_per_second_a100=90.0,
        memory_required_gb=4.5,
        capabilities=[
            ModelCapability.MATH,
            ModelCapability.CHAIN_OF_THOUGHT,
            ModelCapability.REASONING,
            ModelCapability.INSTRUCTION_FOLLOWING,
        ],
        primary_use_cases=["math_proofs", "calculus", "competition_math", "word_problems"],
        available_quantizations=[
            QuantizationType.Q4_K_M,
            QuantizationType.Q5_K_M,
            QuantizationType.Q8_0,
        ],
        default_quantization=QuantizationType.Q4_K_M,
        license="Apache-2.0",
        release_date="2024-09",
    ),
}


# =============================================================================
# v241.0: GCP MULTI-MODEL ROUTING
# =============================================================================

# Task→Model mapping for GCP CPU-only inference (e2-highmem-4, 32 GB RAM)
# NOTE: LLaVA, BGE, TinyLlama excluded — not routable (see manifest.routable flag)
# Architecture note: JARVIS Body's _infer_task_type() in query_handler.py
# produces these task_type strings. If you add new types here, update that
# function too. See gcp_model_swap_coordinator.py for the sister note.
GCP_TASK_MODEL_MAPPING: Dict[str, str] = {
    # Fast tasks → Phi-3.5-mini (2.2 GB, ~3s latency)
    "greeting": "phi-3.5-mini",
    "simple_chat": "phi-3.5-mini",
    "quick_question": "phi-3.5-mini",
    "voice_command": "phi-3.5-mini",

    # Math — split by complexity (v241.1)
    "math_simple": "qwen-2.5-7b",              # Algebra, arithmetic → Qwen2.5-7B
    "math_complex": "qwen-2.5-math-7b",        # Proofs, calculus, competition → Qwen2.5-Math-7B

    # Reasoning — split by complexity (v241.1)
    "reason_simple": "qwen-2.5-7b",            # Basic reasoning → Qwen2.5-7B
    "reason_complex": "deepseek-r1-qwen-7b",   # Multi-step CoT → DeepSeek-R1 distill
    "analyze": "deepseek-r1-qwen-7b",          # Complex analysis → DeepSeek-R1 distill

    # Code → Qwen2.5-Coder-7B (code specialist)
    "code_simple": "qwen-2.5-coder-7b",
    "code_complex": "qwen-2.5-coder-7b",
    "code_review": "qwen-2.5-coder-7b",
    "code_explain": "qwen-2.5-coder-7b",
    "code_architecture": "qwen-2.5-coder-7b",
    "code_debug": "qwen-2.5-coder-7b",

    # Long context / creative → Llama-3.1-8B (128K context)
    "creative_write": "llama-3.1-8b",
    "creative_brainstorm": "llama-3.1-8b",
    "summarize": "llama-3.1-8b",

    # General/default → Gemma-2-9B (v241.1: upgraded from Mistral-7B)
    "general_chat": "gemma-2-9b",
    "translate": "mistral-7b",                  # Mistral still strong for translation
    "unknown": "gemma-2-9b",
}

# Per-model executor config overrides for GCP CPU-only inference.
# Passed as **kwargs to LlamaCppExecutor.load() which merges with base config.
GCP_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "mistral-7b": {
        "n_ctx": 8192,
        "chat_template": "mistral",
        "n_gpu_layers": 0,       # CPU-only on GCP
        "flash_attn": False,     # No GPU flash attention on CPU
    },
    "qwen-2.5-7b": {
        "n_ctx": 32768,
        "chat_template": "chatml",
        "n_gpu_layers": 0,
        "flash_attn": False,
    },
    "qwen-2.5-coder-7b": {
        "n_ctx": 32768,
        "chat_template": "chatml",
        "n_gpu_layers": 0,
        "flash_attn": False,
    },
    "phi-3.5-mini": {
        "n_ctx": 4096,
        "chat_template": "phi3",
        "n_gpu_layers": 0,
        "flash_attn": False,
    },
    "llama-3.1-8b": {
        "n_ctx": 8192,           # Don't use full 128K on 16 GB RAM
        "chat_template": "llama3",
        "n_gpu_layers": 0,
        "flash_attn": False,
    },
    "tinyllama-1.1b": {
        "n_ctx": 2048,
        "chat_template": "chatml",
        "n_gpu_layers": 0,
        "flash_attn": False,
    },
    # v241.1: Advanced reasoning & specialist models
    "deepseek-r1-qwen-7b": {
        "n_ctx": 32768,         # Qwen2 arch supports 128K but 32K is practical on CPU
        "chat_template": "chatml",
        "n_gpu_layers": 0,
        "flash_attn": False,
    },
    "gemma-2-9b": {
        "n_ctx": 8192,
        "chat_template": "gemma",
        "n_gpu_layers": 0,
        "flash_attn": False,
    },
    "qwen-2.5-math-7b": {
        "n_ctx": 4096,          # Math queries are typically short
        "chat_template": "chatml",
        "n_gpu_layers": 0,
        "flash_attn": False,
    },
}


# =============================================================================
# TIER TO MODEL MAPPING v97.0
# =============================================================================


TIER_MODEL_MAPPING: Dict[str, List[str]] = {
    "tier_0_local_fast": [
        "phi-3.5-mini",  # Primary: Ultra-fast voice (55 t/s)
        "llama-3.2-3b",  # Fallback: Fast general
        "deepseek-r1-8b",  # Fallback: Fast reasoning with CoT
        "qwen-2.5-7b",  # Fallback: General capable
    ],
    "tier_05_local_capable": [
        "qwen-2.5-32b",  # Primary: LOCAL CODING KING
        "qwen-2.5-coder-32b",  # Primary for code: Specialized coder
        "qwen-2.5-14b",  # Fallback: Lower RAM
        "qwen-2.5-7b",  # Fallback: Even lower RAM
    ],
    "tier_1_cloud_intelligent": [
        "llama-3.3-70b",  # Primary: Cloud workhorse
        "deepseek-coder-v2",  # For coding: CODE MASTER
        "deepseek-v2-lite",  # Fallback: MoE efficiency
    ],
    "tier_2_deep_reasoning": [
        "deepseek-v2.5",  # Primary: GPT-5 LEVEL
        "deepseek-coder-v2",  # For code: 236B MoE
        "llama-3.3-70b",  # Fallback: Stable 70B
    ],
}


# Capability to preferred models mapping
CAPABILITY_MODEL_MAPPING: Dict[ModelCapability, List[str]] = {
    ModelCapability.VOICE_OPTIMIZED: ["phi-3.5-mini", "llama-3.2-3b"],
    ModelCapability.CHAIN_OF_THOUGHT: ["deepseek-r1-8b", "deepseek-v2.5"],
    ModelCapability.CODE_EXPERT: [
        "qwen-2.5-coder-32b",
        "deepseek-coder-v2",
        "qwen-2.5-32b",
    ],
    ModelCapability.ARCHITECTURE_DESIGN: [
        "deepseek-v2.5",
        "deepseek-coder-v2",
        "llama-3.3-70b",
    ],
    ModelCapability.FAST_INFERENCE: ["phi-3.5-mini", "llama-3.2-3b", "qwen-2.5-7b"],
}


# =============================================================================
# DOWNLOAD MANAGER (Advanced with HF Hub Integration)
# =============================================================================


@dataclass
class DownloadProgress:
    """Track download progress."""

    model_id: str
    filename: str
    total_bytes: int
    downloaded_bytes: int
    speed_bps: float
    eta_seconds: float
    started_at: datetime
    status: str  # pending, downloading, verifying, complete, failed, cancelled
    error: Optional[str] = None
    retries: int = 0

    @property
    def percent_complete(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "filename": self.filename,
            "total_mb": round(self.total_bytes / (1024 * 1024), 1),
            "downloaded_mb": round(self.downloaded_bytes / (1024 * 1024), 1),
            "percent": round(self.percent_complete, 1),
            "speed_mbps": round(self.speed_bps / (1024 * 1024), 2),
            "eta_minutes": round(self.eta_seconds / 60, 1) if self.eta_seconds else 0,
            "status": self.status,
            "retries": self.retries,
            "error": self.error,
        }


class AdvancedModelDownloader:
    """
    Enterprise-grade model downloader with:
    - HuggingFace Hub integration
    - Resumable downloads
    - SHA256 verification
    - Semaphore-based concurrency control
    - Exponential backoff with jitter
    - Progress callbacks
    - Automatic cleanup on failure
    """

    def __init__(
        self,
        models_dir: Path,
        max_concurrent: int = 2,
        max_retries: int = 3,
        chunk_size: int = 8 * 1024 * 1024,  # 8MB chunks
    ):
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)

        self._max_concurrent = max_concurrent
        self._max_retries = max_retries
        self._chunk_size = chunk_size

        # Semaphore prevents thundering herd
        self._download_semaphore = asyncio.Semaphore(max_concurrent)
        self._verification_semaphore = asyncio.Semaphore(4)

        # Active downloads tracking
        self._active_downloads: Dict[str, DownloadProgress] = {}
        self._callbacks: List[Callable[[DownloadProgress], Awaitable[None]]] = []
        self._lock = asyncio.Lock()

        # Download cancellation
        self._cancel_events: Dict[str, asyncio.Event] = {}

    def on_progress(
        self, callback: Callable[[DownloadProgress], Awaitable[None]]
    ) -> None:
        """Register progress callback."""
        self._callbacks.append(callback)

    async def _notify_progress(self, progress: DownloadProgress) -> None:
        """Notify callbacks with error handling."""
        for callback in self._callbacks:
            try:
                await callback(progress)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

    async def download(
        self,
        spec: ModelSpec,
        quant: QuantizationType = None,
        force: bool = False,
    ) -> Path:
        """
        Download a model with full progress tracking.

        Uses HuggingFace Hub if available, falls back to direct download.
        """
        quant = quant or spec.default_quantization
        filename = spec.get_filename(quant)
        output_path = self._models_dir / filename

        # Check if already exists and valid
        if output_path.exists() and not force:
            if await self._verify_file(output_path, spec, quant):
                logger.info(f"Model {spec.model_id} already downloaded and verified")
                return output_path
            else:
                logger.warning(f"Existing file corrupt, re-downloading: {filename}")
                output_path.unlink()

        # Initialize progress tracking
        progress = DownloadProgress(
            model_id=spec.model_id,
            filename=filename,
            total_bytes=0,
            downloaded_bytes=0,
            speed_bps=0.0,
            eta_seconds=0.0,
            started_at=datetime.now(),
            status="pending",
        )

        # Cancel event for this download
        cancel_event = asyncio.Event()
        self._cancel_events[spec.model_id] = cancel_event

        async with self._download_semaphore:
            self._active_downloads[spec.model_id] = progress

            try:
                # Try HuggingFace Hub first
                if await self._try_huggingface_hub(spec, quant, output_path, progress):
                    return output_path

                # Fallback to direct download with retries
                for attempt in range(self._max_retries):
                    if cancel_event.is_set():
                        raise asyncio.CancelledError("Download cancelled")

                    try:
                        progress.retries = attempt
                        await self._direct_download(
                            spec, quant, output_path, progress, cancel_event
                        )
                        break
                    except Exception as e:
                        if attempt == self._max_retries - 1:
                            raise
                        # Exponential backoff with jitter
                        delay = (2**attempt) + (asyncio.get_event_loop().time() % 1)
                        logger.warning(
                            f"Download attempt {attempt + 1} failed, "
                            f"retrying in {delay:.1f}s: {e}"
                        )
                        await asyncio.sleep(delay)

                # Verify download
                progress.status = "verifying"
                await self._notify_progress(progress)

                if not await self._verify_file(output_path, spec, quant):
                    raise ValueError("Checksum verification failed")

                progress.status = "complete"
                await self._notify_progress(progress)
                logger.info(f"Successfully downloaded {spec.model_id}: {output_path}")
                return output_path

            except asyncio.CancelledError:
                progress.status = "cancelled"
                await self._notify_progress(progress)
                if output_path.exists():
                    output_path.unlink()
                raise

            except Exception as e:
                progress.status = "failed"
                progress.error = str(e)
                await self._notify_progress(progress)
                if output_path.exists():
                    output_path.unlink()
                raise

            finally:
                self._active_downloads.pop(spec.model_id, None)
                self._cancel_events.pop(spec.model_id, None)

    async def _try_huggingface_hub(
        self,
        spec: ModelSpec,
        quant: QuantizationType,
        output_path: Path,
        progress: DownloadProgress,
    ) -> bool:
        """Try downloading via HuggingFace Hub library."""
        try:
            from huggingface_hub import hf_hub_download

            progress.status = "downloading"
            await self._notify_progress(progress)

            filename = spec.get_filename(quant)

            # Run in executor to not block
            loop = asyncio.get_event_loop()
            downloaded_path = await loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=spec.hf_repo,
                    filename=filename,
                    local_dir=self._models_dir,
                    local_dir_use_symlinks=False,
                ),
            )

            # Move to expected location if different
            if Path(downloaded_path) != output_path:
                shutil.move(downloaded_path, output_path)

            return True

        except ImportError:
            logger.debug("huggingface_hub not available, using direct download")
            return False
        except Exception as e:
            logger.warning(f"HuggingFace Hub download failed: {e}")
            return False

    async def _direct_download(
        self,
        spec: ModelSpec,
        quant: QuantizationType,
        output_path: Path,
        progress: DownloadProgress,
        cancel_event: asyncio.Event,
    ) -> None:
        """Direct download with progress tracking."""
        url = spec.get_download_url(quant)
        progress.status = "downloading"

        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=60)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise ValueError(f"HTTP {response.status}")

                    progress.total_bytes = int(
                        response.headers.get("content-length", 0)
                    )
                    progress.downloaded_bytes = 0
                    start_time = time.time()
                    last_update = start_time

                    await self._notify_progress(progress)

                    with open(output_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(
                            self._chunk_size
                        ):
                            if cancel_event.is_set():
                                raise asyncio.CancelledError()

                            f.write(chunk)
                            progress.downloaded_bytes += len(chunk)

                            # Update stats every second
                            now = time.time()
                            if now - last_update >= 1.0:
                                elapsed = now - start_time
                                progress.speed_bps = progress.downloaded_bytes / elapsed
                                remaining = (
                                    progress.total_bytes - progress.downloaded_bytes
                                )
                                progress.eta_seconds = (
                                    remaining / progress.speed_bps
                                    if progress.speed_bps > 0
                                    else 0
                                )
                                last_update = now
                                await self._notify_progress(progress)

        except ImportError:
            # Fallback to urllib
            import urllib.request

            progress.status = "downloading"
            await self._notify_progress(progress)

            await asyncio.get_event_loop().run_in_executor(
                None, lambda: urllib.request.urlretrieve(url, output_path)
            )

    async def _verify_file(
        self,
        path: Path,
        spec: ModelSpec,
        quant: QuantizationType,
    ) -> bool:
        """Verify file integrity with checksum."""
        if not path.exists():
            return False

        # Check size is reasonable
        size_gb = path.stat().st_size / (1024**3)
        expected_size = spec.estimated_size_gb(quant)
        if size_gb < expected_size * 0.5:  # Less than 50% expected
            return False

        # Check SHA256 if available
        expected_hash = spec.checksums.get(quant.value)
        if expected_hash:
            async with self._verification_semaphore:
                actual_hash = await self._compute_sha256(path)
                return actual_hash == expected_hash

        return True  # No checksum, trust size check

    async def _compute_sha256(self, path: Path) -> str:
        """Compute SHA256 hash of file."""

        def _hash():
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()

        return await asyncio.get_event_loop().run_in_executor(None, _hash)

    def cancel_download(self, model_id: str) -> bool:
        """Cancel an active download."""
        if model_id in self._cancel_events:
            self._cancel_events[model_id].set()
            return True
        return False

    def get_active_downloads(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active downloads."""
        return {
            model_id: progress.to_dict()
            for model_id, progress in self._active_downloads.items()
        }


# =============================================================================
# NEURAL MODEL SELECTOR (Advanced Routing)
# =============================================================================


class NeuralModelSelector:
    """
    Intelligent model selection using multi-factor analysis:
    - Task complexity scoring
    - Capability matching
    - Resource constraints
    - Cost awareness
    - Historical performance
    - Adaptive learning
    """

    def __init__(self):
        self._selection_history: List[Dict[str, Any]] = []
        self._performance_cache: Dict[str, Dict[str, float]] = {}

        # Adaptive weights (can be tuned)
        self._weights = {
            "tier_match": 0.25,
            "capability_match": 0.30,
            "speed": 0.15,
            "cost": 0.15,
            "quality": 0.15,
        }

    async def select(
        self,
        prompt: str,
        complexity_score: float,
        available_models: List[ModelSpec],
        requirements: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelSpec:
        """
        Select optimal model for the task.

        Args:
            prompt: The prompt to process
            complexity_score: Pre-computed complexity (0-1)
            available_models: Models available for selection
            requirements: Required capabilities (e.g., ["code", "reasoning"])
            context: Additional context (max_latency, budget, prefer_quality)

        Returns:
            Selected ModelSpec
        """
        if not available_models:
            raise ValueError("No models available for selection")

        context = context or {}
        requirements = requirements or []

        # Convert string requirements to capabilities
        required_capabilities = self._parse_requirements(requirements)

        # Score each model
        scored_models: List[Tuple[ModelSpec, float, str]] = []

        for model in available_models:
            score, reasoning = await self._score_model(
                model,
                complexity_score,
                required_capabilities,
                context,
            )
            scored_models.append((model, score, reasoning))

        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)

        best_model, best_score, reasoning = scored_models[0]

        # Record selection
        self._selection_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "complexity": complexity_score,
                "requirements": requirements,
                "selected": best_model.model_id,
                "score": best_score,
                "reasoning": reasoning,
                "candidates": [(m.model_id, s) for m, s, _ in scored_models[:3]],
            }
        )

        # Keep history bounded
        if len(self._selection_history) > 1000:
            self._selection_history = self._selection_history[-500:]

        logger.info(
            f"Neural selection: {best_model.model_id} "
            f"(score={best_score:.3f}, complexity={complexity_score:.2f}) - {reasoning}"
        )

        return best_model

    def _parse_requirements(self, requirements: List[str]) -> Set[ModelCapability]:
        """Parse string requirements into capabilities."""
        mapping = {
            "code": {ModelCapability.CODE_GENERATION, ModelCapability.CODE_ANALYSIS},
            "coding": {ModelCapability.CODE_GENERATION, ModelCapability.CODE_ANALYSIS},
            "expert_code": {ModelCapability.CODE_EXPERT},
            "reasoning": {ModelCapability.REASONING},
            "cot": {ModelCapability.CHAIN_OF_THOUGHT},
            "chain_of_thought": {ModelCapability.CHAIN_OF_THOUGHT},
            "math": {ModelCapability.MATH},
            "creative": {ModelCapability.CREATIVE},
            "fast": {ModelCapability.FAST_INFERENCE},
            "voice": {ModelCapability.VOICE_OPTIMIZED},
            "architecture": {ModelCapability.ARCHITECTURE_DESIGN},
            "long_context": {ModelCapability.LONG_CONTEXT},
        }

        result: Set[ModelCapability] = set()
        for req in requirements:
            req_lower = req.lower().replace("-", "_").replace(" ", "_")
            if req_lower in mapping:
                result.update(mapping[req_lower])
        return result

    async def _score_model(
        self,
        model: ModelSpec,
        complexity: float,
        required_capabilities: Set[ModelCapability],
        context: Dict[str, Any],
    ) -> Tuple[float, str]:
        """Score a model for the task."""
        scores = {}
        reasoning_parts = []

        # 1. Tier match score
        tier_score = self._score_tier_match(model.tier_level, complexity)
        scores["tier_match"] = tier_score
        if tier_score > 0.8:
            reasoning_parts.append("tier_optimal")

        # 2. Capability match score
        cap_score = self._score_capability_match(
            set(model.capabilities), required_capabilities
        )
        scores["capability_match"] = cap_score
        if cap_score >= 1.0:
            reasoning_parts.append("caps_perfect")

        # 3. Speed score
        if context.get("prefer_speed") or context.get("max_latency_ms"):
            max_latency = context.get("max_latency_ms", 10000)
            # Estimate time for 100 tokens
            estimated_ms = (100 / model.tokens_per_second_m1) * 1000
            speed_score = min(1.0, max_latency / max(estimated_ms, 1))
            scores["speed"] = speed_score
            if speed_score > 0.9:
                reasoning_parts.append("fast")
        else:
            scores["speed"] = 0.5  # Neutral

        # 4. Cost score
        if context.get("prefer_free", True):
            if model.location == ModelLocation.LOCAL_ONLY:
                scores["cost"] = 1.0
                reasoning_parts.append("free")
            elif model.cost_per_1k_input == 0:
                scores["cost"] = 1.0
            else:
                scores["cost"] = 0.5
        else:
            scores["cost"] = 0.7  # Neutral when cost doesn't matter

        # 5. Quality score (based on parameter count and tier)
        quality_score = min(model.parameter_count_billions / 70.0, 1.0)
        if model.is_moe:
            quality_score *= 1.2  # MoE bonus
        scores["quality"] = min(quality_score, 1.0)
        if quality_score > 0.8:
            reasoning_parts.append("high_quality")

        # Calculate weighted total
        total_score = sum(
            scores[key] * self._weights[key] for key in self._weights.keys()
        )

        reasoning = ", ".join(reasoning_parts) if reasoning_parts else "balanced"

        return total_score, reasoning

    def _score_tier_match(
        self, tier_level: ModelTierLevel, complexity: float
    ) -> float:
        """Score how well tier matches complexity."""
        # Map complexity to ideal tier
        if complexity < 0.20:
            ideal = ModelTierLevel.TIER_0_ULTRA_FAST
        elif complexity < 0.35:
            ideal = ModelTierLevel.TIER_0_FAST
        elif complexity < 0.55:
            ideal = ModelTierLevel.TIER_05_CAPABLE
        elif complexity < 0.80:
            ideal = ModelTierLevel.TIER_1_CLOUD
        else:
            ideal = ModelTierLevel.TIER_2_DEEP

        # Score based on distance
        distance = abs(tier_level.value - ideal.value)
        return max(0, 1.0 - (distance * 0.2))

    def _score_capability_match(
        self,
        model_caps: Set[ModelCapability],
        required_caps: Set[ModelCapability],
    ) -> float:
        """Score capability match."""
        if not required_caps:
            return 0.8  # Neutral

        matched = model_caps & required_caps
        return len(matched) / len(required_caps)

    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics."""
        if not self._selection_history:
            return {"total_selections": 0}

        model_counts: Dict[str, int] = {}
        for entry in self._selection_history:
            model_id = entry["selected"]
            model_counts[model_id] = model_counts.get(model_id, 0) + 1

        return {
            "total_selections": len(self._selection_history),
            "model_distribution": model_counts,
            "avg_complexity": sum(
                e["complexity"] for e in self._selection_history
            )
            / len(self._selection_history),
            "recent_selections": [
                {
                    "model": e["selected"],
                    "complexity": e["complexity"],
                    "reasoning": e["reasoning"],
                }
                for e in self._selection_history[-5:]
            ],
        }


# =============================================================================
# ELASTIC BURST CONTROLLER
# =============================================================================


class ElasticBurstController:
    """
    Elastic burst controller for automatic cloud scaling.

    Monitors:
    - RAM pressure (via psutil or JARVIS AdvancedRAMMonitor)
    - Task complexity requirements
    - Budget constraints

    Triggers:
    - Cloud burst when local resources insufficient
    - Automatic fallback when budget exhausted
    """

    def __init__(
        self,
        ram_threshold_percent: float = 85.0,
        complexity_threshold: float = 0.75,
        min_budget_for_burst: float = 0.50,
    ):
        self._ram_threshold = ram_threshold_percent
        self._complexity_threshold = complexity_threshold
        self._min_budget = min_budget_for_burst

        self._burst_active = False
        self._burst_start_time: Optional[datetime] = None
        self._burst_count = 0
        self._lock = asyncio.Lock()

    async def should_burst(
        self,
        complexity: float,
        required_ram_gb: float = 0.0,
        available_budget: float = 10.0,
    ) -> Tuple[bool, str]:
        """
        Determine if cloud burst should be triggered.

        Returns:
            Tuple of (should_burst, reason)
        """
        reasons = []

        # Check budget first
        if available_budget < self._min_budget:
            return False, "insufficient_budget"

        # Check RAM pressure
        ram_pressure = await self._get_ram_pressure()
        if ram_pressure > self._ram_threshold:
            reasons.append(f"ram_pressure_{ram_pressure:.0f}%")

        # Check if task requires more RAM than available
        if required_ram_gb > 0:
            available_ram = await self._get_available_ram_gb()
            if required_ram_gb > available_ram * 0.9:
                reasons.append(f"need_{required_ram_gb:.0f}GB")

        # Check complexity threshold
        if complexity > self._complexity_threshold:
            reasons.append(f"complexity_{complexity:.2f}")

        if reasons:
            return True, "+".join(reasons)

        return False, "local_sufficient"

    async def _get_ram_pressure(self) -> float:
        """Get current RAM usage percentage."""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            return 50.0  # Conservative default

    async def _get_available_ram_gb(self) -> float:
        """Get available RAM in GB."""
        try:
            import psutil

            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 8.0  # Conservative default

    async def start_burst(self, reason: str) -> None:
        """Record burst start."""
        async with self._lock:
            self._burst_active = True
            self._burst_start_time = datetime.now()
            self._burst_count += 1
            logger.warning(f"🚀 ELASTIC BURST ACTIVATED: {reason}")

    async def end_burst(self) -> None:
        """Record burst end."""
        async with self._lock:
            if self._burst_start_time:
                duration = datetime.now() - self._burst_start_time
                logger.info(f"🏁 Elastic burst ended (duration: {duration})")
            self._burst_active = False
            self._burst_start_time = None

    def get_status(self) -> Dict[str, Any]:
        """Get burst controller status."""
        return {
            "burst_active": self._burst_active,
            "burst_count": self._burst_count,
            "burst_start": (
                self._burst_start_time.isoformat() if self._burst_start_time else None
            ),
            "ram_threshold": self._ram_threshold,
            "complexity_threshold": self._complexity_threshold,
        }


# =============================================================================
# TASK CLASSIFIER - Neural Switchboard Brain
# =============================================================================


@dataclass
class TaskClassification:
    """Result of task classification."""

    task_type: TaskType
    confidence: float  # 0-1
    complexity: float  # 0-1
    detected_signals: List[str]
    recommended_tiers: List[str]
    required_capabilities: List[ModelCapability]
    requires_fast_response: bool = False
    is_coding_session: bool = False
    context_tokens_estimate: int = 0


class TaskClassifier:
    """
    Intelligent task classification using multi-signal analysis.

    Analyzes:
    - Keyword patterns (code, debug, explain, etc.)
    - Structural signals (code blocks, lists, etc.)
    - Complexity indicators (length, nesting, etc.)
    - Session context (ongoing coding session?)
    - Voice mode detection

    Thread-safe with async lock.
    """

    # Keyword patterns for task detection
    _CODE_KEYWORDS = frozenset([
        "code", "implement", "write", "function", "class", "method", "api",
        "python", "javascript", "typescript", "rust", "golang", "java",
        "fix", "bug", "error", "exception", "debug", "refactor", "optimize",
        "test", "unittest", "pytest", "jest", "async", "await", "import",
        "def ", "class ", "return ", "if ", "for ", "while ",
    ])

    _ARCHITECTURE_KEYWORDS = frozenset([
        "architect", "design", "pattern", "structure", "system", "scale",
        "microservice", "monolith", "database", "schema", "api design",
        "infrastructure", "deployment", "kubernetes", "docker", "ci/cd",
    ])

    _REASONING_KEYWORDS = frozenset([
        "explain", "why", "how", "reason", "think", "analyze", "compare",
        "evaluate", "pros", "cons", "tradeoff", "decision", "strategy",
        "plan", "approach", "consider", "implications",
    ])

    _MATH_KEYWORDS = frozenset([
        "calculate", "compute", "math", "equation", "formula", "derive",
        "solve", "proof", "theorem", "algorithm", "complexity", "o(n)",
        "integral", "derivative", "matrix", "statistics", "probability",
    ])

    _CREATIVE_KEYWORDS = frozenset([
        "write", "story", "poem", "creative", "imagine", "brainstorm",
        "ideas", "generate", "suggest", "alternative", "creative writing",
    ])

    _GREETING_PATTERNS = frozenset([
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "what's up", "how are you", "thanks", "thank you", "bye", "goodbye",
    ])

    _VOICE_PATTERNS = frozenset([
        "jarvis", "hey jarvis", "ok jarvis", "listen", "what time",
        "set timer", "remind me", "play", "stop", "pause", "quick",
    ])

    def __init__(self):
        self._lock = asyncio.Lock()
        self._session_context: Dict[str, Any] = {}
        self._classification_history: List[TaskClassification] = []
        self._coding_session_active = False
        self._coding_session_start: Optional[datetime] = None

        # Adaptive weights based on history
        self._weights = {
            "keyword": 0.35,
            "structure": 0.20,
            "complexity": 0.25,
            "session": 0.20,
        }

    async def classify(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskClassification:
        """
        Classify a prompt into task type with full analysis.

        Returns comprehensive TaskClassification with confidence scores.
        """
        async with self._lock:
            context = context or {}
            prompt_lower = prompt.lower().strip()
            signals: List[str] = []

            # Quick checks for simple cases
            # Voice commands take priority (e.g., "hey jarvis" in voice mode)
            if self._is_voice_command(prompt_lower, context):
                return self._create_classification(
                    TaskType.VOICE_COMMAND, 0.90, 0.10, ["voice_pattern"],
                    context, requires_fast=True
                )

            if self._is_greeting(prompt_lower):
                return self._create_classification(
                    TaskType.GREETING, 0.95, 0.05, ["greeting_pattern"], context
                )

            # Analyze prompt structure
            has_code_block = "```" in prompt or "    " in prompt
            has_code_keywords = self._count_keyword_matches(
                prompt_lower, self._CODE_KEYWORDS
            )
            has_arch_keywords = self._count_keyword_matches(
                prompt_lower, self._ARCHITECTURE_KEYWORDS
            )
            has_reason_keywords = self._count_keyword_matches(
                prompt_lower, self._REASONING_KEYWORDS
            )
            has_math_keywords = self._count_keyword_matches(
                prompt_lower, self._MATH_KEYWORDS
            )
            has_creative_keywords = self._count_keyword_matches(
                prompt_lower, self._CREATIVE_KEYWORDS
            )

            # Calculate complexity
            complexity = self._calculate_complexity(prompt)

            # Determine task type
            task_type, confidence = await self._determine_task_type(
                prompt_lower,
                has_code_block,
                has_code_keywords,
                has_arch_keywords,
                has_reason_keywords,
                has_math_keywords,
                has_creative_keywords,
                complexity,
                signals,
            )

            # Update coding session state
            is_coding = task_type in {
                TaskType.CODE_SIMPLE, TaskType.CODE_COMPLEX,
                TaskType.CODE_REVIEW, TaskType.CODE_ARCHITECTURE,
                TaskType.CODE_DEBUG, TaskType.CODE_EXPLAIN,
            }

            if is_coding:
                if not self._coding_session_active:
                    self._coding_session_active = True
                    self._coding_session_start = datetime.now()
                    signals.append("coding_session_started")
            elif self._coding_session_active:
                # Check if session should end (non-coding for 10+ minutes)
                if self._coding_session_start:
                    elapsed = (datetime.now() - self._coding_session_start).seconds
                    if elapsed > 600:  # 10 minutes
                        self._coding_session_active = False
                        signals.append("coding_session_ended")

            classification = self._create_classification(
                task_type, confidence, complexity, signals, context,
                is_coding_session=self._coding_session_active
            )

            # Store in history (bounded)
            self._classification_history.append(classification)
            if len(self._classification_history) > 100:
                self._classification_history = self._classification_history[-50:]

            return classification

    def _is_greeting(self, prompt: str) -> bool:
        """Check if prompt is a simple greeting."""
        # Very short prompts that match greeting patterns
        if len(prompt) < 50:
            for pattern in self._GREETING_PATTERNS:
                if prompt.startswith(pattern) or prompt == pattern:
                    return True
        return False

    def _is_voice_command(self, prompt: str, context: Dict[str, Any]) -> bool:
        """Check if this is a voice interface command."""
        if context.get("voice_mode") or context.get("from_voice"):
            return True
        for pattern in self._VOICE_PATTERNS:
            if prompt.startswith(pattern):
                return True
        return False

    def _count_keyword_matches(self, text: str, keywords: frozenset) -> int:
        """Count keyword matches in text."""
        count = 0
        for keyword in keywords:
            if keyword in text:
                count += 1
        return count

    def _calculate_complexity(self, prompt: str) -> float:
        """Calculate prompt complexity (0-1)."""
        factors = []

        # Length factor
        length = len(prompt)
        if length < 50:
            factors.append(0.1)
        elif length < 200:
            factors.append(0.3)
        elif length < 500:
            factors.append(0.5)
        elif length < 1000:
            factors.append(0.7)
        else:
            factors.append(0.9)

        # Code block factor
        code_blocks = prompt.count("```")
        if code_blocks > 0:
            factors.append(min(0.3 + code_blocks * 0.1, 0.8))

        # Question depth (nested requirements)
        if "and" in prompt.lower() and "also" in prompt.lower():
            factors.append(0.6)
        elif "and" in prompt.lower() or "also" in prompt.lower():
            factors.append(0.4)

        # Multi-step indicators
        multi_step_words = ["first", "then", "after", "finally", "step"]
        multi_step_count = sum(1 for w in multi_step_words if w in prompt.lower())
        if multi_step_count >= 2:
            factors.append(0.7)

        return sum(factors) / max(len(factors), 1)

    async def _determine_task_type(
        self,
        prompt: str,
        has_code_block: bool,
        code_keywords: int,
        arch_keywords: int,
        reason_keywords: int,
        math_keywords: int,
        creative_keywords: int,
        complexity: float,
        signals: List[str],
    ) -> Tuple[TaskType, float]:
        """Determine task type from signals."""

        # Architecture tasks (highest priority for code + architecture signals)
        if arch_keywords >= 2 or (arch_keywords >= 1 and code_keywords >= 2):
            signals.append(f"arch_kw={arch_keywords}")
            return TaskType.CODE_ARCHITECTURE, 0.85

        # Complex coding
        if code_keywords >= 3 or (has_code_block and code_keywords >= 2):
            signals.append(f"code_kw={code_keywords}")
            if complexity > 0.6:
                return TaskType.CODE_COMPLEX, 0.80
            else:
                return TaskType.CODE_SIMPLE, 0.75

        # Code review / explain
        if code_keywords >= 1 and has_code_block:
            if "review" in prompt or "check" in prompt:
                signals.append("code_review")
                return TaskType.CODE_REVIEW, 0.80
            if "explain" in prompt or "what does" in prompt:
                signals.append("code_explain")
                return TaskType.CODE_EXPLAIN, 0.80

        # Math tasks
        if math_keywords >= 2:
            signals.append(f"math_kw={math_keywords}")
            if complexity > 0.5:
                return TaskType.MATH_COMPLEX, 0.80
            return TaskType.MATH_SIMPLE, 0.75

        # Reasoning tasks
        if reason_keywords >= 2:
            signals.append(f"reason_kw={reason_keywords}")
            if complexity > 0.6:
                return TaskType.REASON_COMPLEX, 0.75
            return TaskType.REASON_SIMPLE, 0.70

        # Creative tasks
        if creative_keywords >= 2:
            signals.append(f"creative_kw={creative_keywords}")
            if "story" in prompt or "poem" in prompt:
                return TaskType.CREATIVE_WRITE, 0.80
            return TaskType.CREATIVE_BRAINSTORM, 0.70

        # Summary / analysis
        if "summarize" in prompt or "summary" in prompt:
            signals.append("summarize")
            return TaskType.SUMMARIZE, 0.85

        if "analyze" in prompt or "analysis" in prompt:
            signals.append("analyze")
            return TaskType.ANALYZE, 0.80

        # Default based on complexity
        if complexity < 0.3:
            return TaskType.SIMPLE_CHAT, 0.60
        elif complexity < 0.6:
            return TaskType.GENERAL_CHAT, 0.55
        else:
            return TaskType.UNKNOWN, 0.40

    def _create_classification(
        self,
        task_type: TaskType,
        confidence: float,
        complexity: float,
        signals: List[str],
        context: Dict[str, Any],
        requires_fast: bool = False,
        is_coding_session: bool = False,
    ) -> TaskClassification:
        """Create TaskClassification with all metadata."""
        recommended_tiers = TASK_TIER_MAPPING.get(
            task_type, ["tier_0_local_fast"]
        )
        required_capabilities = TASK_CAPABILITY_MAPPING.get(task_type, [])

        return TaskClassification(
            task_type=task_type,
            confidence=confidence,
            complexity=complexity,
            detected_signals=signals,
            recommended_tiers=recommended_tiers,
            required_capabilities=required_capabilities,
            requires_fast_response=requires_fast or task_type == TaskType.VOICE_COMMAND,
            is_coding_session=is_coding_session or self._coding_session_active,
            context_tokens_estimate=context.get("estimated_tokens", 0),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        if not self._classification_history:
            return {"total": 0}

        type_counts: Dict[str, int] = {}
        for c in self._classification_history:
            t = c.task_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total": len(self._classification_history),
            "type_distribution": type_counts,
            "avg_complexity": sum(c.complexity for c in self._classification_history)
            / len(self._classification_history),
            "coding_session_active": self._coding_session_active,
        }


# =============================================================================
# macOS MEMORY PRESSURE MONITOR
# =============================================================================


class MacOSMemoryPressureMonitor:
    """
    Native macOS memory pressure monitoring using memory_pressure command.

    Unlike psutil which only shows usage %, this uses Apple's native
    memory_pressure command which accounts for:
    - File system cache (reclaimable)
    - Compressed memory
    - Swap activity
    - System pressure signals

    Falls back gracefully on non-macOS systems.
    """

    def __init__(self, cache_ttl: float = 2.0):
        self._cache_ttl = cache_ttl
        self._cached_level: Optional[MemoryPressureLevel] = None
        self._cache_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
        self._is_macos = sys.platform == "darwin"

        # History for trend detection
        self._pressure_history: List[Tuple[datetime, MemoryPressureLevel]] = []

    async def get_pressure_level(self) -> MemoryPressureLevel:
        """Get current memory pressure level (cached with TTL)."""
        async with self._lock:
            now = datetime.now()

            # Return cached if fresh
            if (
                self._cached_level is not None
                and self._cache_time is not None
                and (now - self._cache_time).total_seconds() < self._cache_ttl
            ):
                return self._cached_level

            # Get fresh reading
            level = await self._read_memory_pressure()
            self._cached_level = level
            self._cache_time = now

            # Record history
            self._pressure_history.append((now, level))
            if len(self._pressure_history) > 60:
                self._pressure_history = self._pressure_history[-30:]

            return level

    async def _read_memory_pressure(self) -> MemoryPressureLevel:
        """Read memory pressure from system."""
        if not self._is_macos:
            return await self._fallback_pressure()

        try:
            # Run memory_pressure command
            proc = await asyncio.create_subprocess_exec(
                "memory_pressure",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            output = stdout.decode().lower()

            # Parse output
            if "critical" in output or "the system is under pressure" in output:
                return MemoryPressureLevel.CRITICAL
            elif "warn" in output or "approaching" in output:
                return MemoryPressureLevel.WARN
            else:
                return MemoryPressureLevel.NORMAL

        except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
            logger.debug(f"memory_pressure command failed: {e}")
            return await self._fallback_pressure()

    async def _fallback_pressure(self) -> MemoryPressureLevel:
        """Fallback to psutil-based pressure detection."""
        try:
            import psutil
            mem = psutil.virtual_memory()

            # Conservative thresholds
            if mem.percent > 90:
                return MemoryPressureLevel.CRITICAL
            elif mem.percent > 80:
                return MemoryPressureLevel.WARN
            else:
                return MemoryPressureLevel.NORMAL
        except ImportError:
            return MemoryPressureLevel.NORMAL

    async def get_detailed_snapshot(self) -> Dict[str, Any]:
        """Get detailed memory snapshot."""
        level = await self.get_pressure_level()

        result = {
            "pressure_level": level.value,
            "is_critical": level == MemoryPressureLevel.CRITICAL,
            "should_burst": level in {MemoryPressureLevel.CRITICAL, MemoryPressureLevel.WARN},
            "trend": self._detect_trend(),
        }

        # Add psutil data if available
        try:
            import psutil
            mem = psutil.virtual_memory()
            result.update({
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
                "percent_used": mem.percent,
                "swap_percent": psutil.swap_memory().percent,
            })
        except ImportError:
            pass

        return result

    def _detect_trend(self) -> str:
        """Detect memory pressure trend."""
        if len(self._pressure_history) < 5:
            return "stable"

        recent = self._pressure_history[-5:]
        levels = [h[1] for h in recent]

        # Count transitions
        increasing = 0
        for i in range(1, len(levels)):
            if levels[i].value > levels[i - 1].value:
                increasing += 1

        if increasing >= 3:
            return "increasing"
        elif levels.count(MemoryPressureLevel.CRITICAL) >= 3:
            return "critical_sustained"
        else:
            return "stable"


# =============================================================================
# CROSS-REPO MEMORY INTEGRATION
# =============================================================================


class CrossRepoMemoryIntegration:
    """
    Integrates with JARVIS AdvancedRAMMonitor for unified memory management.

    Combines:
    - Local macOS memory_pressure
    - JARVIS AdvancedRAMMonitor (via GCP bridge)
    - psutil metrics
    - Process-specific memory tracking

    Thread-safe singleton pattern.
    """

    _instance: ClassVar[Optional["CrossRepoMemoryIntegration"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(self):
        self._macos_monitor = MacOSMemoryPressureMonitor()
        self._gcp_bridge: Optional[Any] = None
        self._jarvis_monitor: Optional[Any] = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls) -> "CrossRepoMemoryIntegration":
        """Get singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self) -> None:
        """Initialize cross-repo connections."""
        async with self._init_lock:
            if self._initialized:
                return

            # Try to get GCP bridge for JARVIS integration
            try:
                from jarvis_prime.core.gcp_import_bridge import GCPImportBridge
                self._gcp_bridge = await GCPImportBridge.get_instance()
                logger.info("CrossRepoMemoryIntegration: GCP bridge connected")
            except Exception as e:
                logger.debug(f"GCP bridge not available: {e}")

            self._initialized = True

    async def get_unified_memory_status(self) -> Dict[str, Any]:
        """
        Get unified memory status from all sources.

        Combines local + cross-repo data for comprehensive view.
        """
        result = {
            "source": "unified",
            "timestamp": datetime.now().isoformat(),
        }

        # Local macOS pressure
        local_snapshot = await self._macos_monitor.get_detailed_snapshot()
        result["local"] = local_snapshot

        # Cross-repo JARVIS data
        if self._gcp_bridge:
            try:
                jarvis_ram = await self._gcp_bridge.get_ram_snapshot()
                result["jarvis"] = jarvis_ram
            except Exception as e:
                logger.debug(f"JARVIS RAM snapshot failed: {e}")
                result["jarvis"] = {"error": str(e)}

        # Decision
        local_should_burst = local_snapshot.get("should_burst", False)
        jarvis_should_burst = result.get("jarvis", {}).get("shift_to_gcp_recommended", False)

        result["should_burst_to_cloud"] = local_should_burst or jarvis_should_burst
        result["burst_reason"] = self._determine_burst_reason(result)

        return result

    def _determine_burst_reason(self, status: Dict[str, Any]) -> Optional[str]:
        """Determine why burst is recommended."""
        reasons = []

        local = status.get("local", {})
        if local.get("pressure_level") == "critical":
            reasons.append("local_critical")
        elif local.get("pressure_level") == "warn":
            reasons.append("local_warn")

        if local.get("trend") == "increasing":
            reasons.append("pressure_increasing")

        jarvis = status.get("jarvis", {})
        if jarvis.get("shift_to_gcp_recommended"):
            reasons.append("jarvis_recommends")

        return "+".join(reasons) if reasons else None

    async def can_load_model(self, required_gb: float) -> Tuple[bool, str]:
        """
        Check if we have enough memory to load a model.

        Returns:
            Tuple of (can_load, reason)
        """
        status = await self.get_unified_memory_status()

        # Check local available
        local = status.get("local", {})
        available_gb = local.get("available_gb", 0)

        # Need buffer for system
        buffer_gb = 2.0
        needed = required_gb + buffer_gb

        if available_gb < needed:
            return False, f"insufficient_ram_{available_gb:.1f}GB<{needed:.1f}GB"

        if local.get("pressure_level") == "critical":
            return False, "memory_pressure_critical"

        if local.get("trend") == "critical_sustained":
            return False, "sustained_critical_pressure"

        return True, "sufficient"


# =============================================================================
# REQUEST BUFFER FOR HOT SWAP
# =============================================================================


@dataclass
class BufferedRequest:
    """A request buffered during model swap."""

    request_id: str
    prompt: str
    context: Dict[str, Any]
    created_at: datetime
    timeout_at: datetime
    future: asyncio.Future


class HotSwapRequestBuffer:
    """
    Buffers requests during model hot swap to prevent request loss.

    Features:
    - Bounded queue with backpressure
    - Timeout handling
    - Priority support (voice requests first)
    - Metrics tracking
    """

    def __init__(
        self,
        max_size: int = 100,
        default_timeout: float = 30.0,
    ):
        self._max_size = max_size
        self._default_timeout = default_timeout
        self._queue: asyncio.Queue[BufferedRequest] = asyncio.Queue(maxsize=max_size)
        self._active = False
        self._lock = asyncio.Lock()

        # Metrics
        self._buffered_count = 0
        self._flushed_count = 0
        self._timeout_count = 0

    async def start_buffering(self) -> None:
        """Start buffering requests (call before hot swap)."""
        async with self._lock:
            self._active = True
            logger.info("🔄 Request buffering started for hot swap")

    async def stop_buffering(self) -> None:
        """Stop buffering (call after hot swap complete)."""
        async with self._lock:
            self._active = False
            logger.info("🔄 Request buffering stopped")

    def is_buffering(self) -> bool:
        """Check if currently buffering."""
        return self._active

    async def enqueue(
        self,
        request_id: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> asyncio.Future:
        """
        Enqueue a request for later processing.

        Returns a Future that will be resolved when the request is processed.
        """
        if not self._active:
            raise RuntimeError("Buffer not active")

        context = context or {}
        timeout = timeout or self._default_timeout

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        request = BufferedRequest(
            request_id=request_id,
            prompt=prompt,
            context=context,
            created_at=datetime.now(),
            timeout_at=datetime.now() + timedelta(seconds=timeout),
            future=future,
        )

        try:
            self._queue.put_nowait(request)
            self._buffered_count += 1
            logger.debug(f"Buffered request {request_id}")
        except asyncio.QueueFull:
            future.set_exception(RuntimeError("Request buffer full"))

        return future

    async def flush_to(
        self,
        processor: Callable[[str, Dict[str, Any]], Awaitable[str]],
    ) -> int:
        """
        Flush all buffered requests to processor.

        Returns count of successfully processed requests.
        """
        processed = 0
        now = datetime.now()

        while not self._queue.empty():
            try:
                request = self._queue.get_nowait()

                # Check timeout
                if now > request.timeout_at:
                    request.future.set_exception(
                        TimeoutError(f"Request {request.request_id} timed out in buffer")
                    )
                    self._timeout_count += 1
                    continue

                # Process request
                try:
                    result = await processor(request.prompt, request.context)
                    request.future.set_result(result)
                    processed += 1
                except Exception as e:
                    request.future.set_exception(e)

            except asyncio.QueueEmpty:
                break

        self._flushed_count += processed
        logger.info(f"Flushed {processed} buffered requests")
        return processed

    def get_queue_length(self) -> int:
        """Get current queue length."""
        return self._queue.qsize()

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "active": self._active,
            "queue_length": self._queue.qsize(),
            "max_size": self._max_size,
            "buffered_total": self._buffered_count,
            "flushed_total": self._flushed_count,
            "timeout_total": self._timeout_count,
        }


# =============================================================================
# STICKY ROUTING MANAGER
# =============================================================================


class StickyRoutingManager:
    """
    Manages sticky routing for session continuity.

    When a user is deep in a coding session, we want to keep
    the coding model loaded instead of swapping on every request.

    Features:
    - Session detection based on task patterns
    - Automatic sticky expiration
    - Priority-based override
    """

    def __init__(
        self,
        session_timeout_minutes: float = 30.0,
        sticky_strength_threshold: float = 0.7,
    ):
        self._session_timeout = session_timeout_minutes * 60
        self._strength_threshold = sticky_strength_threshold
        self._lock = asyncio.Lock()

        # Current sticky state
        self._sticky_model: Optional[str] = None
        self._sticky_tier: Optional[str] = None
        self._sticky_started: Optional[datetime] = None
        self._sticky_strength: float = 0.0

        # Session history
        self._session_tasks: List[TaskType] = []
        self._session_model_uses: Dict[str, int] = {}

    async def get_sticky_model(
        self,
        task_classification: TaskClassification,
    ) -> Optional[str]:
        """
        Get sticky model if session is active and applicable.

        Returns model_id if sticky routing applies, None otherwise.
        """
        async with self._lock:
            # Check if sticky session is expired
            if self._sticky_started:
                elapsed = (datetime.now() - self._sticky_started).total_seconds()
                if elapsed > self._session_timeout:
                    await self._end_session()
                    return None

            # Check if task matches sticky context
            if self._sticky_model and self._sticky_strength >= self._strength_threshold:
                if self._task_matches_session(task_classification):
                    # Extend session
                    self._sticky_started = datetime.now()
                    return self._sticky_model

            return None

    async def update_sticky(
        self,
        model_id: str,
        tier_id: str,
        task_classification: TaskClassification,
    ) -> None:
        """Update sticky state after model selection."""
        async with self._lock:
            # Track task
            self._session_tasks.append(task_classification.task_type)
            if len(self._session_tasks) > 20:
                self._session_tasks = self._session_tasks[-10:]

            # Track model use
            self._session_model_uses[model_id] = (
                self._session_model_uses.get(model_id, 0) + 1
            )

            # Calculate new strength
            new_strength = self._calculate_strength(model_id, task_classification)

            # Update sticky if stronger
            if new_strength >= self._strength_threshold:
                if self._sticky_model != model_id or new_strength > self._sticky_strength:
                    self._sticky_model = model_id
                    self._sticky_tier = tier_id
                    self._sticky_strength = new_strength
                    if self._sticky_started is None:
                        self._sticky_started = datetime.now()
                        logger.info(
                            f"📌 Sticky session started: {model_id} "
                            f"(strength={new_strength:.2f})"
                        )

    def _task_matches_session(self, task: TaskClassification) -> bool:
        """Check if task matches current session type."""
        if not self._session_tasks:
            return True

        # If session is coding-focused
        coding_tasks = {TaskType.CODE_SIMPLE, TaskType.CODE_COMPLEX,
                       TaskType.CODE_REVIEW, TaskType.CODE_EXPLAIN,
                       TaskType.CODE_DEBUG, TaskType.CODE_ARCHITECTURE}

        session_coding_ratio = sum(
            1 for t in self._session_tasks if t in coding_tasks
        ) / len(self._session_tasks)

        if session_coding_ratio > 0.6 and task.task_type in coding_tasks:
            return True

        return False

    def _calculate_strength(
        self,
        model_id: str,
        task: TaskClassification,
    ) -> float:
        """Calculate sticky strength for model."""
        factors = []

        # Model usage factor
        total_uses = sum(self._session_model_uses.values())
        if total_uses > 0:
            model_ratio = self._session_model_uses.get(model_id, 0) / total_uses
            factors.append(model_ratio)

        # Coding session factor
        if task.is_coding_session:
            factors.append(0.8)

        # Task consistency factor
        if len(self._session_tasks) >= 3:
            recent = self._session_tasks[-3:]
            if len(set(recent)) == 1:  # Same task type
                factors.append(0.9)
            elif len(set(recent)) <= 2:  # Similar tasks
                factors.append(0.6)

        return sum(factors) / max(len(factors), 1)

    async def _end_session(self) -> None:
        """End current sticky session."""
        if self._sticky_model:
            logger.info(f"📌 Sticky session ended: {self._sticky_model}")
        self._sticky_model = None
        self._sticky_tier = None
        self._sticky_started = None
        self._sticky_strength = 0.0
        self._session_tasks.clear()
        self._session_model_uses.clear()

    def get_status(self) -> Dict[str, Any]:
        """Get sticky routing status."""
        return {
            "sticky_model": self._sticky_model,
            "sticky_tier": self._sticky_tier,
            "sticky_strength": round(self._sticky_strength, 2),
            "session_duration_seconds": (
                (datetime.now() - self._sticky_started).total_seconds()
                if self._sticky_started else 0
            ),
            "session_task_count": len(self._session_tasks),
        }


# =============================================================================
# MULTI-DIRECTORY MODEL SCANNER (v99.0)
# =============================================================================


@dataclass
class ModelDirectoryConfig:
    """Configuration for a model directory."""

    path: Path
    priority: int = 0  # Higher = more preferred
    recursive: bool = True
    watch: bool = True
    source: str = "local"  # local, reactor, huggingface, custom

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()


@dataclass
class DiscoveredModel:
    """A model discovered during scanning."""

    path: Path
    model_id: Optional[str]
    spec: Optional["ModelSpec"]
    size_bytes: int
    modified_time: datetime
    source_directory: ModelDirectoryConfig
    validation_status: str = "pending"  # pending, valid, invalid, unknown
    sha256_hash: Optional[str] = None
    fuzzy_match_score: float = 0.0
    fuzzy_match_candidates: List[str] = field(default_factory=list)


class MultiDirectoryModelScanner:
    """
    Advanced multi-directory model scanner with:
    - Configurable search paths with priorities
    - Recursive directory scanning
    - Parallel async scanning
    - Duplicate detection across directories
    - Model metadata extraction
    - Directory-specific configuration
    """

    # Default search directories (in priority order)
    DEFAULT_SEARCH_DIRS: ClassVar[List[Tuple[str, int]]] = [
        # Primary JARVIS-Prime models (highest priority)
        ("~/Documents/repos/jarvis-prime/models", 100),
        # User's custom models directory
        ("~/.jarvis/prime/models", 90),
        # Generic AI models directory
        ("~/Documents/ai-models", 80),
        ("~/ai-models", 75),
        # LM Studio models (if user has it)
        ("~/.cache/lm-studio/models", 70),
        # Ollama models
        ("~/.ollama/models", 65),
        # Reactor Core output (trained models)
        ("~/Documents/repos/reactor-core/output/models", 60),
        ("~/Documents/repos/reactor-core/models", 55),
        # JARVIS-AI-Agent models
        ("~/Documents/repos/JARVIS-AI-Agent/models", 50),
    ]

    # File patterns to scan
    MODEL_PATTERNS: ClassVar[List[str]] = [
        "*.gguf",
        "*.ggml",
        "*.bin",
        "*.safetensors",
        "*.pt",
        "*.pth",
    ]

    def __init__(
        self,
        additional_dirs: Optional[List[Path]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_workers: int = 4,
    ):
        self._directories: List[ModelDirectoryConfig] = []
        self._exclude_patterns = exclude_patterns or ["*.tmp", "*.partial", "*.download"]
        self._max_workers = max_workers
        self._scan_lock = asyncio.Lock()
        self._discovered: Dict[str, DiscoveredModel] = {}
        self._last_scan_time: Optional[datetime] = None
        self._scan_semaphore = asyncio.Semaphore(max_workers)

        # Initialize default directories
        self._init_default_directories()

        # Add additional directories
        if additional_dirs:
            for path in additional_dirs:
                self.add_directory(path)

    def _init_default_directories(self) -> None:
        """Initialize default search directories."""
        for path_str, priority in self.DEFAULT_SEARCH_DIRS:
            path = Path(path_str).expanduser().resolve()
            if path.exists() and path.is_dir():
                self._directories.append(
                    ModelDirectoryConfig(
                        path=path,
                        priority=priority,
                        recursive=True,
                        watch=True,
                        source="local" if "repos" in str(path) else "external",
                    )
                )
                logger.debug(f"Model scanner: Added directory {path} (priority={priority})")

    def add_directory(
        self,
        path: Path,
        priority: int = 50,
        recursive: bool = True,
        watch: bool = True,
        source: str = "custom",
    ) -> None:
        """Add a directory to the scanner."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created model directory: {path}")
            except Exception as e:
                logger.warning(f"Cannot create directory {path}: {e}")
                return

        # Check for duplicates
        for existing in self._directories:
            if existing.path == path:
                logger.debug(f"Directory already registered: {path}")
                return

        config = ModelDirectoryConfig(
            path=path,
            priority=priority,
            recursive=recursive,
            watch=watch,
            source=source,
        )
        self._directories.append(config)
        self._directories.sort(key=lambda d: d.priority, reverse=True)
        logger.info(f"Added model directory: {path} (priority={priority}, source={source})")

    async def scan_all(
        self,
        force_rescan: bool = False,
        known_models: Optional[Dict[str, "ModelSpec"]] = None,
    ) -> Dict[str, DiscoveredModel]:
        """
        Scan all directories for models.

        Args:
            force_rescan: If True, rescan even if recently scanned
            known_models: Known model specs for matching

        Returns:
            Dictionary of discovered models by path string
        """
        async with self._scan_lock:
            # Check if we need to rescan
            if not force_rescan and self._last_scan_time:
                time_since_scan = (datetime.now() - self._last_scan_time).total_seconds()
                if time_since_scan < 60:  # Cache for 1 minute
                    return self._discovered

            self._discovered.clear()
            known_models = known_models or KNOWN_MODELS

            # Scan directories in parallel
            tasks = [
                self._scan_directory(config, known_models)
                for config in self._directories
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge results, preferring higher priority directories
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Scan error: {result}")
                    continue
                for path_str, discovered in result.items():
                    if path_str not in self._discovered:
                        self._discovered[path_str] = discovered
                    else:
                        # Keep higher priority version
                        existing = self._discovered[path_str]
                        if discovered.source_directory.priority > existing.source_directory.priority:
                            self._discovered[path_str] = discovered

            self._last_scan_time = datetime.now()
            logger.info(
                f"Multi-directory scan complete: {len(self._discovered)} models "
                f"across {len(self._directories)} directories"
            )
            return self._discovered

    async def _scan_directory(
        self,
        config: ModelDirectoryConfig,
        known_models: Dict[str, "ModelSpec"],
    ) -> Dict[str, DiscoveredModel]:
        """Scan a single directory for models."""
        discovered: Dict[str, DiscoveredModel] = {}

        if not config.path.exists():
            return discovered

        try:
            async with self._scan_semaphore:
                # Use asyncio for non-blocking file system operations
                loop = asyncio.get_event_loop()

                if config.recursive:
                    all_files = await loop.run_in_executor(
                        None,
                        lambda: list(config.path.rglob("*")),
                    )
                else:
                    all_files = await loop.run_in_executor(
                        None,
                        lambda: list(config.path.glob("*")),
                    )

                for file_path in all_files:
                    if not file_path.is_file():
                        continue

                    # Check if matches model patterns
                    if not any(
                        file_path.match(pattern)
                        for pattern in self.MODEL_PATTERNS
                    ):
                        continue

                    # Check exclusions
                    if any(
                        file_path.match(pattern)
                        for pattern in self._exclude_patterns
                    ):
                        continue

                    # Get file info
                    try:
                        stat = await loop.run_in_executor(None, file_path.stat)
                        size_bytes = stat.st_size
                        modified_time = datetime.fromtimestamp(stat.st_mtime)
                    except OSError:
                        continue

                    # Match against known models
                    model_id, spec, match_score = self._match_model(
                        file_path, known_models
                    )

                    path_str = str(file_path)
                    discovered[path_str] = DiscoveredModel(
                        path=file_path,
                        model_id=model_id,
                        spec=spec,
                        size_bytes=size_bytes,
                        modified_time=modified_time,
                        source_directory=config,
                        validation_status="pending",
                        fuzzy_match_score=match_score,
                    )

        except Exception as e:
            logger.error(f"Error scanning {config.path}: {e}")

        return discovered

    def _match_model(
        self,
        file_path: Path,
        known_models: Dict[str, "ModelSpec"],
    ) -> Tuple[Optional[str], Optional["ModelSpec"], float]:
        """
        Match a file against known models.

        Returns:
            (model_id, spec, match_score) tuple
        """
        filename = file_path.name.lower()
        best_match: Optional[str] = None
        best_spec: Optional["ModelSpec"] = None
        best_score: float = 0.0

        for model_id, spec in known_models.items():
            # Exact match check
            for quant in spec.available_quantizations:
                expected = spec.get_filename(quant).lower()
                if filename == expected:
                    return model_id, spec, 1.0

            # Fuzzy matching
            score = self._fuzzy_match_score(filename, model_id, spec)
            if score > best_score:
                best_score = score
                best_match = model_id
                best_spec = spec

        # Only return match if score is above threshold
        if best_score >= 0.6:
            return best_match, best_spec, best_score

        return None, None, best_score

    def _fuzzy_match_score(
        self,
        filename: str,
        model_id: str,
        spec: "ModelSpec",
    ) -> float:
        """Calculate fuzzy match score between filename and model spec."""
        score = 0.0
        filename_lower = filename.lower()
        model_id_lower = model_id.lower().replace("-", "").replace("_", "")

        # Check if model name appears in filename
        name_parts = model_id_lower.split("-")
        for part in name_parts:
            if part in filename_lower.replace("-", "").replace("_", ""):
                score += 0.2

        # Check parameter count
        param_count = spec.parameter_count.lower()
        if param_count.replace("b", "") in filename_lower:
            score += 0.15

        # Check quantization type
        for quant in spec.available_quantizations:
            if quant.value.lower() in filename_lower:
                score += 0.1
                break

        # Check for provider name in filename (if provider is set)
        provider = getattr(spec, 'provider', '') or ''
        if provider and provider.lower() in filename_lower:
            score += 0.15

        # Bonus for GGUF extension matching
        if filename_lower.endswith(".gguf"):
            score += 0.1

        return min(score, 1.0)

    def get_directories(self) -> List[ModelDirectoryConfig]:
        """Get list of configured directories."""
        return self._directories.copy()

    def get_discovered_models(self) -> Dict[str, DiscoveredModel]:
        """Get discovered models from last scan."""
        return self._discovered.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get scanner statistics."""
        matched = sum(1 for m in self._discovered.values() if m.model_id)
        unmatched = sum(1 for m in self._discovered.values() if not m.model_id)
        total_size = sum(m.size_bytes for m in self._discovered.values())

        return {
            "directories_configured": len(self._directories),
            "directories_active": sum(1 for d in self._directories if d.path.exists()),
            "models_discovered": len(self._discovered),
            "models_matched": matched,
            "models_unmatched": unmatched,
            "total_size_gb": round(total_size / (1024**3), 2),
            "last_scan": self._last_scan_time.isoformat() if self._last_scan_time else None,
        }


# =============================================================================
# MODEL FILE WATCHER (v99.0)
# =============================================================================


class ModelFileEvent(Enum):
    """File system event types."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class ModelFileChange:
    """A file system change event."""

    event_type: ModelFileEvent
    path: Path
    old_path: Optional[Path] = None  # For move events
    timestamp: datetime = field(default_factory=datetime.now)


class ModelFileWatcher:
    """
    File system watcher for real-time model discovery.

    Uses polling as a cross-platform fallback (watchdog optional).
    Supports debouncing and batching of events.
    """

    def __init__(
        self,
        directories: List[ModelDirectoryConfig],
        debounce_seconds: float = 2.0,
        poll_interval_seconds: float = 30.0,
    ):
        self._directories = directories
        self._debounce_seconds = debounce_seconds
        self._poll_interval = poll_interval_seconds
        self._running = False
        self._watch_task: Optional[asyncio.Task] = None
        self._event_queue: asyncio.Queue[ModelFileChange] = asyncio.Queue()
        self._callbacks: List[Callable[[ModelFileChange], Awaitable[None]]] = []
        self._file_hashes: Dict[str, Tuple[int, float]] = {}  # path -> (size, mtime)
        self._watchdog_available = False
        self._observers: List[Any] = []

        # Try to import watchdog
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            self._watchdog_available = True
            logger.info("ModelFileWatcher: Using watchdog for efficient file watching")
        except ImportError:
            logger.info("ModelFileWatcher: watchdog not available, using polling")

    def subscribe(
        self,
        callback: Callable[[ModelFileChange], Awaitable[None]],
    ) -> None:
        """Subscribe to file change events."""
        self._callbacks.append(callback)

    def unsubscribe(
        self,
        callback: Callable[[ModelFileChange], Awaitable[None]],
    ) -> None:
        """Unsubscribe from file change events."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def start(self) -> None:
        """Start watching directories."""
        if self._running:
            return

        self._running = True

        # Initialize file hashes
        await self._initialize_file_hashes()

        if self._watchdog_available:
            await self._start_watchdog()
        else:
            self._watch_task = asyncio.create_task(self._poll_loop())

        # Start event processor
        asyncio.create_task(self._process_events())
        logger.info("ModelFileWatcher started")

    async def stop(self) -> None:
        """Stop watching directories."""
        self._running = False

        # Stop watchdog observers
        for observer in self._observers:
            try:
                observer.stop()
                observer.join(timeout=2)
            except Exception:
                pass
        self._observers.clear()

        # Cancel polling task
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

        logger.info("ModelFileWatcher stopped")

    async def _initialize_file_hashes(self) -> None:
        """Initialize file hashes for change detection."""
        loop = asyncio.get_event_loop()

        for config in self._directories:
            if not config.watch or not config.path.exists():
                continue

            try:
                files = await loop.run_in_executor(
                    None,
                    lambda p=config.path: list(p.rglob("*.gguf") if config.recursive else p.glob("*.gguf")),
                )

                for file_path in files:
                    try:
                        stat = file_path.stat()
                        self._file_hashes[str(file_path)] = (stat.st_size, stat.st_mtime)
                    except OSError:
                        continue

            except Exception as e:
                logger.warning(f"Error initializing file hashes for {config.path}: {e}")

    async def _start_watchdog(self) -> None:
        """Start watchdog observers for each directory."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileSystemEvent

            class ModelEventHandler(FileSystemEventHandler):
                def __init__(handler_self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
                    handler_self.queue = queue
                    handler_self.loop = loop

                def on_created(handler_self, event: FileSystemEvent):
                    if not event.is_directory and event.src_path.endswith(".gguf"):
                        handler_self.loop.call_soon_threadsafe(
                            handler_self.queue.put_nowait,
                            ModelFileChange(
                                event_type=ModelFileEvent.CREATED,
                                path=Path(event.src_path),
                            ),
                        )

                def on_modified(handler_self, event: FileSystemEvent):
                    if not event.is_directory and event.src_path.endswith(".gguf"):
                        handler_self.loop.call_soon_threadsafe(
                            handler_self.queue.put_nowait,
                            ModelFileChange(
                                event_type=ModelFileEvent.MODIFIED,
                                path=Path(event.src_path),
                            ),
                        )

                def on_deleted(handler_self, event: FileSystemEvent):
                    if not event.is_directory and event.src_path.endswith(".gguf"):
                        handler_self.loop.call_soon_threadsafe(
                            handler_self.queue.put_nowait,
                            ModelFileChange(
                                event_type=ModelFileEvent.DELETED,
                                path=Path(event.src_path),
                            ),
                        )

                def on_moved(handler_self, event: FileSystemEvent):
                    if not event.is_directory and (
                        event.src_path.endswith(".gguf") or event.dest_path.endswith(".gguf")
                    ):
                        handler_self.loop.call_soon_threadsafe(
                            handler_self.queue.put_nowait,
                            ModelFileChange(
                                event_type=ModelFileEvent.MOVED,
                                path=Path(event.dest_path),
                                old_path=Path(event.src_path),
                            ),
                        )

            loop = asyncio.get_event_loop()
            handler = ModelEventHandler(self._event_queue, loop)

            for config in self._directories:
                if not config.watch or not config.path.exists():
                    continue

                observer = Observer()
                observer.schedule(
                    handler,
                    str(config.path),
                    recursive=config.recursive,
                )
                observer.start()
                self._observers.append(observer)
                logger.debug(f"Started watchdog for {config.path}")

        except Exception as e:
            logger.error(f"Failed to start watchdog: {e}")
            # Fall back to polling
            self._watch_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Polling loop for file changes (fallback when watchdog unavailable)."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                await self._check_for_changes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll loop error: {e}")

    async def _check_for_changes(self) -> None:
        """Check for file changes using polling."""
        loop = asyncio.get_event_loop()
        current_files: Dict[str, Tuple[int, float]] = {}

        for config in self._directories:
            if not config.watch or not config.path.exists():
                continue

            try:
                files = await loop.run_in_executor(
                    None,
                    lambda p=config.path: list(p.rglob("*.gguf") if config.recursive else p.glob("*.gguf")),
                )

                for file_path in files:
                    try:
                        stat = file_path.stat()
                        path_str = str(file_path)
                        current_files[path_str] = (stat.st_size, stat.st_mtime)

                        # Check for new or modified files
                        if path_str not in self._file_hashes:
                            await self._event_queue.put(
                                ModelFileChange(
                                    event_type=ModelFileEvent.CREATED,
                                    path=file_path,
                                )
                            )
                        elif self._file_hashes[path_str] != current_files[path_str]:
                            await self._event_queue.put(
                                ModelFileChange(
                                    event_type=ModelFileEvent.MODIFIED,
                                    path=file_path,
                                )
                            )

                    except OSError:
                        continue

            except Exception as e:
                logger.warning(f"Error polling {config.path}: {e}")

        # Check for deleted files
        for path_str in list(self._file_hashes.keys()):
            if path_str not in current_files:
                await self._event_queue.put(
                    ModelFileChange(
                        event_type=ModelFileEvent.DELETED,
                        path=Path(path_str),
                    )
                )

        self._file_hashes = current_files

    async def _process_events(self) -> None:
        """Process file change events with debouncing."""
        pending_events: Dict[str, ModelFileChange] = {}
        last_flush_time = time.time()

        while self._running:
            try:
                # Get events with timeout
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(),
                        timeout=self._debounce_seconds,
                    )
                    path_str = str(event.path)

                    # Debounce: keep only the latest event per path
                    pending_events[path_str] = event

                except asyncio.TimeoutError:
                    pass

                # Flush pending events if debounce time has passed
                current_time = time.time()
                if (
                    pending_events
                    and current_time - last_flush_time >= self._debounce_seconds
                ):
                    # Notify callbacks
                    for event in pending_events.values():
                        for callback in self._callbacks:
                            try:
                                await callback(event)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")

                    pending_events.clear()
                    last_flush_time = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get watcher statistics."""
        return {
            "running": self._running,
            "watchdog_available": self._watchdog_available,
            "directories_watched": sum(
                1 for d in self._directories if d.watch and d.path.exists()
            ),
            "files_tracked": len(self._file_hashes),
            "observers_active": len(self._observers),
            "callbacks_registered": len(self._callbacks),
        }


# =============================================================================
# MODEL VALIDATION PIPELINE (v99.0)
# =============================================================================


class ValidationResult(Enum):
    """Validation result status."""

    VALID = "valid"
    INVALID = "invalid"
    CORRUPTED = "corrupted"
    INCOMPLETE = "incomplete"
    UNKNOWN_FORMAT = "unknown_format"
    SKIPPED = "skipped"


@dataclass
class ModelValidation:
    """Result of model validation."""

    path: Path
    result: ValidationResult
    sha256_hash: Optional[str] = None
    expected_hash: Optional[str] = None
    file_size_bytes: int = 0
    expected_size_bytes: Optional[int] = None
    error_message: Optional[str] = None
    validation_time_ms: float = 0
    loadable: Optional[bool] = None
    format_detected: Optional[str] = None


class ModelValidationPipeline:
    """
    Comprehensive model validation pipeline:
    - SHA256 checksum verification
    - File integrity checks
    - Format detection
    - Loadability testing
    - Size verification
    """

    # Known file signatures (magic bytes)
    FORMAT_SIGNATURES: ClassVar[Dict[bytes, str]] = {
        b"GGUF": "gguf",
        b"GGML": "ggml",
        b"PK": "safetensors_zip",
        b"\x80\x02": "pytorch_pickle",
    }

    def __init__(
        self,
        enable_sha256: bool = True,
        enable_loadability_test: bool = False,
        chunk_size: int = 8192,
        max_validation_time_seconds: float = 60.0,
    ):
        self._enable_sha256 = enable_sha256
        self._enable_loadability = enable_loadability_test
        self._chunk_size = chunk_size
        self._max_validation_time = max_validation_time_seconds
        self._validation_cache: Dict[str, ModelValidation] = {}
        self._validation_lock = asyncio.Lock()

    async def validate(
        self,
        path: Path,
        expected_hash: Optional[str] = None,
        expected_size: Optional[int] = None,
        force_revalidate: bool = False,
    ) -> ModelValidation:
        """
        Validate a model file.

        Args:
            path: Path to model file
            expected_hash: Expected SHA256 hash (optional)
            expected_size: Expected file size in bytes (optional)
            force_revalidate: Force revalidation even if cached

        Returns:
            ModelValidation result
        """
        path_str = str(path)
        start_time = time.time()

        # Check cache
        if not force_revalidate and path_str in self._validation_cache:
            cached = self._validation_cache[path_str]
            # Verify file hasn't changed
            try:
                stat = path.stat()
                if stat.st_size == cached.file_size_bytes:
                    return cached
            except OSError:
                pass

        async with self._validation_lock:
            try:
                loop = asyncio.get_event_loop()

                # Basic file checks
                if not path.exists():
                    return self._create_result(
                        path, ValidationResult.INVALID,
                        error_message="File does not exist",
                        validation_time_ms=(time.time() - start_time) * 1000,
                    )

                stat = await loop.run_in_executor(None, path.stat)
                file_size = stat.st_size

                # Size check
                if expected_size is not None and file_size != expected_size:
                    return self._create_result(
                        path, ValidationResult.INCOMPLETE,
                        file_size_bytes=file_size,
                        expected_size_bytes=expected_size,
                        error_message=f"Size mismatch: {file_size} != {expected_size}",
                        validation_time_ms=(time.time() - start_time) * 1000,
                    )

                # Format detection
                format_detected = await self._detect_format(path)

                # SHA256 calculation
                sha256_hash = None
                if self._enable_sha256:
                    sha256_hash = await self._calculate_sha256(path)

                    # Verify hash if expected
                    if expected_hash and sha256_hash != expected_hash.lower():
                        return self._create_result(
                            path, ValidationResult.CORRUPTED,
                            sha256_hash=sha256_hash,
                            expected_hash=expected_hash,
                            file_size_bytes=file_size,
                            format_detected=format_detected,
                            error_message="SHA256 hash mismatch",
                            validation_time_ms=(time.time() - start_time) * 1000,
                        )

                # Loadability test (optional, expensive)
                loadable = None
                if self._enable_loadability:
                    loadable = await self._test_loadability(path, format_detected)

                result = self._create_result(
                    path, ValidationResult.VALID,
                    sha256_hash=sha256_hash,
                    expected_hash=expected_hash,
                    file_size_bytes=file_size,
                    expected_size_bytes=expected_size,
                    format_detected=format_detected,
                    loadable=loadable,
                    validation_time_ms=(time.time() - start_time) * 1000,
                )

                # Cache result
                self._validation_cache[path_str] = result
                return result

            except asyncio.TimeoutError:
                return self._create_result(
                    path, ValidationResult.SKIPPED,
                    error_message="Validation timed out",
                    validation_time_ms=(time.time() - start_time) * 1000,
                )
            except Exception as e:
                return self._create_result(
                    path, ValidationResult.INVALID,
                    error_message=str(e),
                    validation_time_ms=(time.time() - start_time) * 1000,
                )

    def _create_result(
        self,
        path: Path,
        result: ValidationResult,
        **kwargs,
    ) -> ModelValidation:
        """Create a validation result."""
        return ModelValidation(path=path, result=result, **kwargs)

    async def _detect_format(self, path: Path) -> Optional[str]:
        """Detect model format from file header."""
        try:
            loop = asyncio.get_event_loop()

            def read_header():
                with open(path, "rb") as f:
                    return f.read(8)

            header = await loop.run_in_executor(None, read_header)

            for signature, format_name in self.FORMAT_SIGNATURES.items():
                if header.startswith(signature):
                    return format_name

            # Check by extension
            suffix = path.suffix.lower()
            if suffix == ".gguf":
                return "gguf"
            elif suffix == ".ggml":
                return "ggml"
            elif suffix == ".safetensors":
                return "safetensors"
            elif suffix in (".pt", ".pth"):
                return "pytorch"
            elif suffix == ".bin":
                return "binary"

            return "unknown"

        except Exception:
            return None

    async def _calculate_sha256(self, path: Path) -> str:
        """Calculate SHA256 hash of file."""
        loop = asyncio.get_event_loop()

        def compute_hash():
            sha256 = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(self._chunk_size):
                    sha256.update(chunk)
            return sha256.hexdigest()

        return await asyncio.wait_for(
            loop.run_in_executor(None, compute_hash),
            timeout=self._max_validation_time,
        )

    async def _test_loadability(
        self,
        path: Path,
        format_detected: Optional[str],
    ) -> Optional[bool]:
        """Test if model can be loaded (basic check)."""
        if format_detected != "gguf":
            return None

        try:
            # Try to read GGUF metadata without loading entire model
            loop = asyncio.get_event_loop()

            def check_gguf():
                with open(path, "rb") as f:
                    magic = f.read(4)
                    if magic != b"GGUF":
                        return False
                    # Read version
                    version = int.from_bytes(f.read(4), "little")
                    return version in (2, 3)

            return await loop.run_in_executor(None, check_gguf)

        except Exception:
            return False

    async def validate_batch(
        self,
        paths: List[Path],
        max_concurrent: int = 4,
    ) -> List[ModelValidation]:
        """Validate multiple models concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def validate_with_semaphore(path: Path) -> ModelValidation:
            async with semaphore:
                return await self.validate(path)

        tasks = [validate_with_semaphore(path) for path in paths]
        return await asyncio.gather(*tasks)

    def clear_cache(self) -> None:
        """Clear validation cache."""
        self._validation_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        valid_count = sum(
            1 for v in self._validation_cache.values()
            if v.result == ValidationResult.VALID
        )
        invalid_count = sum(
            1 for v in self._validation_cache.values()
            if v.result in (ValidationResult.INVALID, ValidationResult.CORRUPTED)
        )

        return {
            "cached_validations": len(self._validation_cache),
            "valid_models": valid_count,
            "invalid_models": invalid_count,
            "sha256_enabled": self._enable_sha256,
            "loadability_test_enabled": self._enable_loadability,
        }


# =============================================================================
# REACTOR CORE MODEL SYNC (v99.0)
# =============================================================================


class ReactorCoreModelSync:
    """
    Synchronizes models with Reactor Core (cross-repo integration).

    Monitors Reactor Core output directory for:
    - Newly trained models
    - Fine-tuned variants
    - Merged models
    - Quantized exports

    Automatically registers them in the DynamicModelRegistry.
    """

    # Reactor Core directory patterns
    REACTOR_PATTERNS: ClassVar[List[str]] = [
        "output/models/*.gguf",
        "output/checkpoints/*/model.gguf",
        "output/merged/*.gguf",
        "models/*.gguf",
        "exports/*.gguf",
    ]

    def __init__(
        self,
        reactor_repo_path: Optional[Path] = None,
        auto_register: bool = True,
        poll_interval_seconds: float = 60.0,
    ):
        # Find Reactor Core repo
        if reactor_repo_path:
            self._reactor_path = Path(reactor_repo_path).expanduser().resolve()
        else:
            self._reactor_path = self._find_reactor_repo()

        self._auto_register = auto_register
        self._poll_interval = poll_interval_seconds
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._known_models: Dict[str, Path] = {}
        self._registry_callback: Optional[Callable[[str, Path, Dict], Awaitable[None]]] = None
        self._last_sync: Optional[datetime] = None
        self._sync_lock = asyncio.Lock()

    def _find_reactor_repo(self) -> Optional[Path]:
        """Auto-discover Reactor Core repository."""
        candidates = [
            Path.home() / "Documents" / "repos" / "reactor-core",
            Path.home() / "repos" / "reactor-core",
            Path.home() / "code" / "reactor-core",
            Path.cwd().parent / "reactor-core",
        ]

        for path in candidates:
            if path.exists() and (path / "run_supervisor.py").exists():
                logger.info(f"ReactorCoreModelSync: Found Reactor Core at {path}")
                return path

        logger.warning("ReactorCoreModelSync: Reactor Core repository not found")
        return None

    def set_registry_callback(
        self,
        callback: Callable[[str, Path, Dict], Awaitable[None]],
    ) -> None:
        """Set callback for registering new models."""
        self._registry_callback = callback

    async def start(self) -> None:
        """Start Reactor Core sync."""
        if self._running or not self._reactor_path:
            return

        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        logger.info(f"ReactorCoreModelSync started: monitoring {self._reactor_path}")

    async def stop(self) -> None:
        """Stop Reactor Core sync."""
        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

    async def _sync_loop(self) -> None:
        """Main sync loop."""
        while self._running:
            try:
                await self.sync_models()
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reactor sync error: {e}")
                await asyncio.sleep(self._poll_interval)

    async def sync_models(self) -> Dict[str, Path]:
        """Sync models from Reactor Core."""
        if not self._reactor_path or not self._reactor_path.exists():
            return {}

        async with self._sync_lock:
            new_models: Dict[str, Path] = {}
            loop = asyncio.get_event_loop()

            for pattern in self.REACTOR_PATTERNS:
                try:
                    matches = await loop.run_in_executor(
                        None,
                        lambda p=pattern: list(self._reactor_path.glob(p)),
                    )

                    for path in matches:
                        if not path.is_file():
                            continue

                        path_str = str(path)
                        if path_str in self._known_models:
                            continue

                        # Generate model ID from path
                        model_id = self._generate_model_id(path)

                        # Register with callback if available
                        if self._auto_register and self._registry_callback:
                            metadata = await self._extract_metadata(path)
                            await self._registry_callback(model_id, path, metadata)

                        self._known_models[path_str] = path
                        new_models[model_id] = path
                        logger.info(f"ReactorCoreModelSync: Discovered {model_id} at {path}")

                except Exception as e:
                    logger.warning(f"Error scanning pattern {pattern}: {e}")

            self._last_sync = datetime.now()
            return new_models

    def _generate_model_id(self, path: Path) -> str:
        """Generate a model ID from path."""
        stem = path.stem.lower()

        # Remove common suffixes
        for suffix in ["-gguf", "-q4_k_m", "-q5_k_m", "-f16", "-f32"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]

        # Clean up
        stem = stem.replace("_", "-").strip("-")

        # Add reactor prefix to distinguish from official models
        return f"reactor-{stem}"

    async def _extract_metadata(self, path: Path) -> Dict[str, Any]:
        """Extract metadata from Reactor Core model."""
        metadata: Dict[str, Any] = {
            "source": "reactor-core",
            "discovered_at": datetime.now().isoformat(),
            "file_size_bytes": 0,
        }

        try:
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(None, path.stat)
            metadata["file_size_bytes"] = stat.st_size
            metadata["modified_time"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

            # Check for metadata file
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                meta_content = await loop.run_in_executor(
                    None,
                    lambda: json.loads(meta_path.read_text()),
                )
                metadata.update(meta_content)

        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")

        return metadata

    def get_known_models(self) -> Dict[str, Path]:
        """Get known Reactor Core models."""
        return self._known_models.copy()

    def get_statistics(self) -> Dict[str, Any]:
        """Get sync statistics."""
        return {
            "reactor_repo_found": self._reactor_path is not None,
            "reactor_repo_path": str(self._reactor_path) if self._reactor_path else None,
            "running": self._running,
            "models_synced": len(self._known_models),
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "auto_register": self._auto_register,
        }


# =============================================================================
# MODEL MEMORY MANAGER
# =============================================================================


class ModelMemoryManager:
    """
    Manages model loading/unloading with LRU eviction.

    Features:
    - Priority-based retention (voice models stay loaded)
    - Automatic unloading when RAM pressure high
    - Warm-up scheduling for frequently used models
    """

    def __init__(
        self,
        max_loaded_models: int = 3,
        max_memory_gb: float = 12.0,
    ):
        self._max_loaded = max_loaded_models
        self._max_memory_gb = max_memory_gb

        # LRU cache of loaded models
        self._loaded_models: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._model_priorities: Dict[str, int] = {}  # Higher = keep longer

        # Memory tracking
        self._current_memory_gb = 0.0
        self._lock = asyncio.Lock()

        # Default priorities (voice models stay loaded)
        self._default_priorities = {
            "phi-3.5-mini": 100,  # Voice - always keep
            "llama-3.2-3b": 80,  # Fast backup
            "deepseek-r1-8b": 50,  # Reasoning
            "qwen-2.5-7b": 40,  # General
        }

    async def register_loaded(
        self,
        model_id: str,
        memory_gb: float,
        backend: Optional[Any] = None,
    ) -> None:
        """Register a newly loaded model."""
        async with self._lock:
            # Check if we need to evict
            while (
                len(self._loaded_models) >= self._max_loaded
                or self._current_memory_gb + memory_gb > self._max_memory_gb
            ):
                if not await self._evict_one():
                    break

            # Register the model
            self._loaded_models[model_id] = {
                "memory_gb": memory_gb,
                "loaded_at": datetime.now(),
                "last_used": datetime.now(),
                "use_count": 0,
                "backend": backend,
            }
            self._loaded_models.move_to_end(model_id)
            self._current_memory_gb += memory_gb

            priority = self._default_priorities.get(model_id, 0)
            self._model_priorities[model_id] = priority

    async def mark_used(self, model_id: str) -> None:
        """Mark a model as recently used."""
        async with self._lock:
            if model_id in self._loaded_models:
                self._loaded_models[model_id]["last_used"] = datetime.now()
                self._loaded_models[model_id]["use_count"] += 1
                self._loaded_models.move_to_end(model_id)

    async def _evict_one(self) -> bool:
        """Evict the least recently used low-priority model."""
        if not self._loaded_models:
            return False

        # Find lowest priority model
        candidates = [
            (model_id, self._model_priorities.get(model_id, 0))
            for model_id in self._loaded_models.keys()
        ]
        candidates.sort(key=lambda x: x[1])

        for model_id, priority in candidates:
            if priority < 90:  # Don't evict high priority models
                info = self._loaded_models.pop(model_id)
                self._current_memory_gb -= info["memory_gb"]

                # Unload backend if available
                if info.get("backend") and hasattr(info["backend"], "unload"):
                    try:
                        await info["backend"].unload()
                    except Exception as e:
                        logger.warning(f"Error unloading {model_id}: {e}")

                logger.info(f"Evicted model {model_id} (priority={priority})")
                return True

        return False

    async def unload_model(self, model_id: str) -> bool:
        """Explicitly unload a model."""
        async with self._lock:
            if model_id not in self._loaded_models:
                return False

            info = self._loaded_models.pop(model_id)
            self._current_memory_gb -= info["memory_gb"]

            if info.get("backend") and hasattr(info["backend"], "unload"):
                await info["backend"].unload()

            logger.info(f"Unloaded model {model_id}")
            return True

    def is_loaded(self, model_id: str) -> bool:
        """Check if model is loaded."""
        return model_id in self._loaded_models

    def get_status(self) -> Dict[str, Any]:
        """Get memory manager status."""
        return {
            "loaded_models": list(self._loaded_models.keys()),
            "total_memory_gb": round(self._current_memory_gb, 2),
            "max_memory_gb": self._max_memory_gb,
            "model_details": {
                model_id: {
                    "memory_gb": info["memory_gb"],
                    "use_count": info["use_count"],
                    "priority": self._model_priorities.get(model_id, 0),
                }
                for model_id, info in self._loaded_models.items()
            },
        }


# =============================================================================
# DYNAMIC MODEL REGISTRY v98.0 - Neural Switchboard
# =============================================================================


class DynamicModelRegistry:
    """
    Enterprise-grade model registry with Neural Switchboard v99.0:
    - Auto-discovery of local models across multiple directories
    - On-demand download from HuggingFace
    - Neural model selection with task classification
    - Elastic burst control with unified decision making
    - Memory management with LRU eviction
    - Model warm-up and preloading
    - Cross-repo GCP integration
    - Sticky routing for session continuity
    - Request buffering during hot swap
    - macOS native memory pressure monitoring
    - Real-time file system watching for model discovery
    - Reactor Core integration for cross-repo model sync
    - Model validation pipeline with SHA256 verification
    """

    _instance: ClassVar[Optional["DynamicModelRegistry"]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        enable_auto_download: bool = True,
        enable_prefetch: bool = True,
        enable_file_watching: bool = True,
        enable_reactor_sync: bool = True,
        max_loaded_models: int = 3,
    ):
        # Models directory
        if models_dir is None:
            models_dir = (
                Path.home() / "Documents" / "repos" / "jarvis-prime" / "models"
            )
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)

        # Feature flags
        self._enable_auto_download = enable_auto_download
        self._enable_prefetch = enable_prefetch
        self._enable_file_watching = enable_file_watching
        self._enable_reactor_sync = enable_reactor_sync

        # Core Components
        self._downloader = AdvancedModelDownloader(models_dir)
        self._selector = NeuralModelSelector()
        self._burst_controller = ElasticBurstController()
        self._memory_manager = ModelMemoryManager(max_loaded_models=max_loaded_models)

        # Neural Switchboard Components (v98.0)
        self._task_classifier = TaskClassifier()
        self._memory_pressure_monitor = MacOSMemoryPressureMonitor()
        self._sticky_routing = StickyRoutingManager()
        self._request_buffer = HotSwapRequestBuffer()

        # Multi-Directory Discovery Components (v99.0)
        self._model_scanner = MultiDirectoryModelScanner()
        self._model_validator = ModelValidationPipeline(
            enable_sha256=True,
            enable_loadability_test=False,  # Only enable when needed
        )
        self._reactor_sync = ReactorCoreModelSync()
        self._file_watcher: Optional[ModelFileWatcher] = None

        # State
        self._discovered_models: Dict[str, Path] = {}
        self._model_states: Dict[str, ModelState] = {}
        self._model_specs = KNOWN_MODELS.copy()
        self._tier_mapping = TIER_MODEL_MAPPING.copy()

        # Initialization
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Background tasks
        self._prefetch_task: Optional[asyncio.Task] = None
        self._warmup_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "downloads": 0,
            "cache_hits": 0,
            "selections": 0,
            "bursts": 0,
            "sticky_hits": 0,
            "task_classifications": 0,
            "file_events_processed": 0,
            "reactor_models_synced": 0,
        }

        # Observers
        self._state_observers: List[Callable[[str, ModelState], Awaitable[None]]] = []

    @classmethod
    async def get_instance(cls) -> "DynamicModelRegistry":
        """Get singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self) -> None:
        """Initialize registry with v99.0 multi-directory discovery."""
        async with self._init_lock:
            if self._initialized:
                return

            # Discover existing models using multi-directory scanner
            await self._discover_local_models()

            # Start file system watcher
            if self._enable_file_watching:
                await self._start_file_watcher()

            # Start Reactor Core sync
            if self._enable_reactor_sync:
                await self._start_reactor_sync()

            # Start background tasks
            if self._enable_prefetch:
                self._prefetch_task = asyncio.create_task(self._prefetch_essential())

            self._initialized = True
            scanner_stats = self._model_scanner.get_statistics()
            logger.info(
                f"DynamicModelRegistry v99.0 initialized: "
                f"{len(self._discovered_models)} models discovered across "
                f"{scanner_stats['directories_active']} directories, "
                f"{len(self._model_specs)} specs available"
            )

    async def _discover_local_models(self) -> None:
        """Scan for locally available models using multi-directory scanner."""
        # Use multi-directory scanner for comprehensive discovery
        discovered = await self._model_scanner.scan_all(
            force_rescan=True,
            known_models=self._model_specs,
        )

        # Register discovered models
        for path_str, discovered_model in discovered.items():
            if discovered_model.model_id:
                self._discovered_models[discovered_model.model_id] = discovered_model.path
                self._model_states[discovered_model.model_id] = ModelState.DOWNLOADED
                logger.debug(
                    f"Discovered: {discovered_model.model_id} at {discovered_model.path} "
                    f"(score={discovered_model.fuzzy_match_score:.2f})"
                )
            else:
                # Log unmatched models with high fuzzy scores for potential manual review
                if discovered_model.fuzzy_match_score >= 0.4:
                    logger.debug(
                        f"Unmatched model: {discovered_model.path.name} "
                        f"(best_score={discovered_model.fuzzy_match_score:.2f})"
                    )

        # Fallback: Also scan primary models directory directly
        if self._models_dir.exists():
            for path in self._models_dir.glob("*.gguf"):
                filename = path.name

                # Match against known models
                for model_id, spec in self._model_specs.items():
                    if model_id in self._discovered_models:
                        continue
                    for quant in spec.available_quantizations:
                        expected = spec.get_filename(quant)
                        if filename.lower() == expected.lower():
                            self._discovered_models[model_id] = path
                            self._model_states[model_id] = ModelState.DOWNLOADED
                            logger.debug(f"Discovered (fallback): {model_id} at {path}")
                            break

    async def _start_file_watcher(self) -> None:
        """Start file system watcher for real-time model discovery."""
        try:
            directories = self._model_scanner.get_directories()
            if not directories:
                return

            self._file_watcher = ModelFileWatcher(
                directories=directories,
                debounce_seconds=2.0,
                poll_interval_seconds=30.0,
            )

            # Subscribe to file change events
            self._file_watcher.subscribe(self._on_model_file_change)
            await self._file_watcher.start()
            logger.info("File system watcher started for model directories")

        except Exception as e:
            logger.warning(f"Failed to start file watcher: {e}")

    async def _on_model_file_change(self, change: ModelFileChange) -> None:
        """Handle file system change events."""
        self._stats["file_events_processed"] += 1

        if change.event_type == ModelFileEvent.CREATED:
            logger.info(f"New model file detected: {change.path}")
            # Re-scan to pick up new model
            await self._discover_local_models()

        elif change.event_type == ModelFileEvent.DELETED:
            logger.info(f"Model file deleted: {change.path}")
            # Remove from discovered models
            path_str = str(change.path)
            for model_id, model_path in list(self._discovered_models.items()):
                if str(model_path) == path_str:
                    del self._discovered_models[model_id]
                    del self._model_states[model_id]
                    logger.info(f"Removed model from registry: {model_id}")
                    break

        elif change.event_type == ModelFileEvent.MODIFIED:
            logger.debug(f"Model file modified: {change.path}")
            # Invalidate validation cache
            self._model_validator.clear_cache()

    async def _start_reactor_sync(self) -> None:
        """Start Reactor Core model synchronization."""
        try:
            # Set up callback for registering new Reactor models
            self._reactor_sync.set_registry_callback(self._on_reactor_model_discovered)
            await self._reactor_sync.start()

        except Exception as e:
            logger.warning(f"Failed to start Reactor Core sync: {e}")

    async def _on_reactor_model_discovered(
        self,
        model_id: str,
        path: Path,
        metadata: Dict[str, Any],
    ) -> None:
        """Handle newly discovered Reactor Core models."""
        self._stats["reactor_models_synced"] += 1

        # Add to discovered models
        self._discovered_models[model_id] = path
        self._model_states[model_id] = ModelState.DOWNLOADED

        # Create a dynamic ModelSpec for the Reactor model
        # This allows it to be used in routing decisions
        file_size_gb = metadata.get("file_size_bytes", 0) / (1024**3)

        logger.info(
            f"Registered Reactor Core model: {model_id} "
            f"({file_size_gb:.1f}GB) at {path}"
        )

    async def _prefetch_essential(self) -> None:
        """Background prefetch of essential models."""
        essential = ["phi-3.5-mini", "qwen-2.5-7b", "deepseek-r1-8b"]

        await asyncio.sleep(5)  # Don't start immediately

        for model_id in essential:
            if model_id in self._discovered_models:
                continue

            spec = self._model_specs.get(model_id)
            if not spec or not self._enable_auto_download:
                continue

            try:
                logger.info(f"Prefetching essential model: {model_id}")
                await self._downloader.download(spec)
                await self._discover_local_models()
            except Exception as e:
                logger.warning(f"Failed to prefetch {model_id}: {e}")

    async def get_or_download(
        self,
        model_id: str,
        quant: Optional[QuantizationType] = None,
    ) -> Path:
        """Get model path, downloading if necessary."""
        if not self._initialized:
            await self.initialize()

        # Check cache
        if model_id in self._discovered_models:
            self._stats["cache_hits"] += 1
            return self._discovered_models[model_id]

        # Get spec
        spec = self._model_specs.get(model_id)
        if not spec:
            raise ValueError(f"Unknown model: {model_id}")

        # Download if enabled
        if not self._enable_auto_download:
            raise FileNotFoundError(f"Model {model_id} not found, auto-download disabled")

        quant = quant or spec.default_quantization

        logger.info(f"Downloading model: {model_id} ({quant.value})")
        self._stats["downloads"] += 1
        self._model_states[model_id] = ModelState.DOWNLOADING

        try:
            path = await self._downloader.download(spec, quant)
            self._discovered_models[model_id] = path
            self._model_states[model_id] = ModelState.DOWNLOADED
            return path
        except Exception as e:
            self._model_states[model_id] = ModelState.ERROR
            raise

    async def get_model_for_tier(
        self,
        tier_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ModelSpec, Optional[Path]]:
        """Get the best available model for a tier."""
        if not self._initialized:
            await self.initialize()

        context = context or {}
        model_ids = self._tier_mapping.get(tier_id, [])

        if not model_ids:
            raise ValueError(f"No models for tier: {tier_id}")

        # Try each model in priority order
        for model_id in model_ids:
            spec = self._model_specs.get(model_id)
            if not spec:
                continue

            # Check if local model is available
            if model_id in self._discovered_models:
                return spec, self._discovered_models[model_id]

            # For local models, try download
            if spec.location == ModelLocation.LOCAL_ONLY and self._enable_auto_download:
                try:
                    path = await self.get_or_download(model_id)
                    return spec, path
                except Exception as e:
                    logger.warning(f"Failed to get {model_id}: {e}")
                    continue

            # For cloud models, return spec without path
            if spec.location == ModelLocation.CLOUD_ONLY:
                return spec, None

        raise RuntimeError(f"No available models for tier: {tier_id}")

    async def neural_select(
        self,
        prompt: str,
        complexity: float,
        requirements: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ModelSpec, Optional[Path]]:
        """Neural selection of optimal model."""
        if not self._initialized:
            await self.initialize()

        self._stats["selections"] += 1
        context = context or {}

        # Check if burst needed
        should_burst, burst_reason = await self._burst_controller.should_burst(
            complexity=complexity,
            required_ram_gb=context.get("required_ram_gb", 0),
            available_budget=context.get("budget", 10.0),
        )

        if should_burst:
            self._stats["bursts"] += 1
            await self._burst_controller.start_burst(burst_reason)
            logger.warning(f"⚡ Elastic burst triggered: {burst_reason}")

        # Get available models
        available_specs = self._get_available_specs(
            local_only=not should_burst,
            context=context,
        )

        # Neural selection
        selected = await self._selector.select(
            prompt=prompt,
            complexity_score=complexity,
            available_models=available_specs,
            requirements=requirements,
            context=context,
        )

        # Get path if local
        path = self._discovered_models.get(selected.model_id)
        if not path and selected.location == ModelLocation.LOCAL_ONLY:
            if self._enable_auto_download:
                path = await self.get_or_download(selected.model_id)

        return selected, path

    def _get_available_specs(
        self,
        local_only: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ModelSpec]:
        """Get available model specs."""
        result = []

        for model_id, spec in self._model_specs.items():
            # Filter by location
            if local_only and spec.location == ModelLocation.CLOUD_ONLY:
                continue

            # Check if downloadable or already available
            if spec.location == ModelLocation.LOCAL_ONLY:
                if model_id in self._discovered_models or self._enable_auto_download:
                    result.append(spec)
            else:
                result.append(spec)

        return result

    async def should_burst_to_cloud(self) -> bool:
        """Check if cloud burst is recommended."""
        should_burst, _ = await self._burst_controller.should_burst(
            complexity=0.8,  # Default high complexity
            available_budget=10.0,
        )
        return should_burst

    # =========================================================================
    # NEURAL SWITCHBOARD v98.0 - Unified Intelligent Routing
    # =========================================================================

    async def neural_switchboard_route(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ModelSpec, Optional[Path], TaskClassification]:
        """
        NEURAL SWITCHBOARD: Unified intelligent routing.

        This is the main entry point for v98.0 routing that combines:
        1. Task classification
        2. Memory pressure check
        3. Sticky routing for sessions
        4. Model selection
        5. Burst decision making

        Returns:
            Tuple of (ModelSpec, Optional[Path], TaskClassification)
        """
        if not self._initialized:
            await self.initialize()

        context = context or {}

        # 1. Classify the task
        task_classification = await self._task_classifier.classify(prompt, context)
        self._stats["task_classifications"] += 1

        # 2. Check memory pressure
        memory_snapshot = await self._memory_pressure_monitor.get_detailed_snapshot()
        memory_critical = memory_snapshot.get("is_critical", False)

        # 3. Check for sticky routing (session continuity)
        sticky_model = await self._sticky_routing.get_sticky_model(task_classification)
        if sticky_model and not memory_critical:
            spec = self._model_specs.get(sticky_model)
            path = self._discovered_models.get(sticky_model)
            if spec and path:
                self._stats["sticky_hits"] += 1
                logger.debug(f"📌 Sticky routing: {sticky_model}")
                return spec, path, task_classification

        # 4. Determine if we need to burst to cloud
        should_burst = memory_critical or memory_snapshot.get("should_burst", False)

        if not should_burst:
            # Check if model fits in memory
            recommended_tier = task_classification.recommended_tiers[0] if task_classification.recommended_tiers else "tier_0_local_fast"
            model_ids = self._tier_mapping.get(recommended_tier, [])
            if model_ids:
                first_spec = self._model_specs.get(model_ids[0])
                if first_spec:
                    can_load, reason = await self._can_load_locally(first_spec)
                    if not can_load:
                        should_burst = True
                        logger.info(f"🔄 Local load blocked: {reason}")

        # 5. Select model based on task classification
        requirements = [cap.value for cap in task_classification.required_capabilities]

        spec, path = await self.neural_select(
            prompt=prompt,
            complexity=task_classification.complexity,
            requirements=requirements,
            context={
                **context,
                "task_type": task_classification.task_type.value,
                "force_cloud": should_burst and memory_critical,
                "prefer_speed": task_classification.requires_fast_response,
            },
        )

        # 6. Update sticky routing
        if spec:
            tier_id = f"tier_{spec.tier_level.value}_" + spec.tier_level.name.lower().split("_")[-1]
            await self._sticky_routing.update_sticky(
                spec.model_id, tier_id, task_classification
            )

        return spec, path, task_classification

    async def _can_load_locally(self, spec: ModelSpec) -> Tuple[bool, str]:
        """Check if model can be loaded locally."""
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)

            # Check memory pressure
            pressure = await self._memory_pressure_monitor.get_pressure_level()
            if pressure == MemoryPressureLevel.CRITICAL:
                return False, "memory_pressure_critical"

            # Check if enough RAM
            buffer_gb = 2.0
            if available_gb < spec.memory_required_gb + buffer_gb:
                return False, f"insufficient_ram_{available_gb:.1f}GB"

            return True, "sufficient"

        except ImportError:
            return True, "psutil_unavailable_assuming_ok"

    async def classify_task(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskClassification:
        """Classify a prompt into task type."""
        return await self._task_classifier.classify(prompt, context)

    async def get_memory_pressure(self) -> Dict[str, Any]:
        """Get current memory pressure status."""
        return await self._memory_pressure_monitor.get_detailed_snapshot()

    def get_sticky_status(self) -> Dict[str, Any]:
        """Get sticky routing status."""
        return self._sticky_routing.get_status()

    def get_request_buffer_status(self) -> Dict[str, Any]:
        """Get request buffer status."""
        return self._request_buffer.get_statistics()

    def get_model_spec(self, model_id: str) -> Optional[ModelSpec]:
        """Get model specification."""
        return self._model_specs.get(model_id)

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of downloaded models."""
        result = []
        for model_id, path in self._discovered_models.items():
            spec = self._model_specs.get(model_id)
            if spec:
                result.append(
                    {
                        "model_id": model_id,
                        "name": spec.name,
                        "tier": spec.tier_level.name,
                        "params": spec.parameter_count,
                        "path": str(path),
                        "capabilities": [c.value for c in spec.capabilities],
                        "state": self._model_states.get(
                            model_id, ModelState.UNKNOWN
                        ).value,
                    }
                )
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics including Neural Switchboard v99.0."""
        return {
            "version": "99.0",
            "total_known_models": len(self._model_specs),
            "downloaded_models": len(self._discovered_models),
            "downloads": self._stats["downloads"],
            "cache_hits": self._stats["cache_hits"],
            "selections": self._stats["selections"],
            "bursts": self._stats["bursts"],
            "sticky_hits": self._stats.get("sticky_hits", 0),
            "task_classifications": self._stats.get("task_classifications", 0),
            "file_events_processed": self._stats.get("file_events_processed", 0),
            "reactor_models_synced": self._stats.get("reactor_models_synced", 0),
            "selector_stats": self._selector.get_statistics(),
            "burst_status": self._burst_controller.get_status(),
            "memory_status": self._memory_manager.get_status(),
            "active_downloads": self._downloader.get_active_downloads(),
            # Neural Switchboard v98.0 stats
            "task_classifier_stats": self._task_classifier.get_statistics(),
            "sticky_routing_status": self._sticky_routing.get_status(),
            "request_buffer_stats": self._request_buffer.get_statistics(),
            # Multi-Directory Discovery v99.0 stats
            "scanner_stats": self._model_scanner.get_statistics(),
            "file_watcher_stats": (
                self._file_watcher.get_statistics() if self._file_watcher else None
            ),
            "reactor_sync_stats": self._reactor_sync.get_statistics(),
            "validation_stats": self._model_validator.get_statistics(),
        }

    def get_scanner_statistics(self) -> Dict[str, Any]:
        """Get multi-directory scanner statistics."""
        return self._model_scanner.get_statistics()

    def get_file_watcher_statistics(self) -> Dict[str, Any]:
        """Get file watcher statistics."""
        if self._file_watcher:
            return self._file_watcher.get_statistics()
        return {"running": False, "enabled": self._enable_file_watching}

    def get_reactor_sync_statistics(self) -> Dict[str, Any]:
        """Get Reactor Core sync statistics."""
        return self._reactor_sync.get_statistics()

    async def validate_model(
        self,
        model_id: str,
        expected_hash: Optional[str] = None,
    ) -> Optional[ModelValidation]:
        """Validate a specific model file."""
        if model_id not in self._discovered_models:
            return None

        path = self._discovered_models[model_id]
        return await self._model_validator.validate(
            path=path,
            expected_hash=expected_hash,
        )

    async def rescan_directories(self) -> Dict[str, Path]:
        """Force rescan of all model directories."""
        await self._discover_local_models()
        return self._discovered_models.copy()

    async def shutdown(self) -> None:
        """Shutdown registry and all v99.0 components."""
        # Stop file watcher
        if self._file_watcher:
            await self._file_watcher.stop()

        # Stop Reactor Core sync
        await self._reactor_sync.stop()

        # Cancel background tasks
        if self._prefetch_task:
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except asyncio.CancelledError:
                pass

        if self._warmup_task:
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except asyncio.CancelledError:
                pass

        await self._burst_controller.end_burst()
        logger.info(f"DynamicModelRegistry v99.0 shutdown: {self.get_statistics()}")


# =============================================================================
# SINGLETON & FACTORY
# =============================================================================


_registry: Optional[DynamicModelRegistry] = None
_registry_lock = asyncio.Lock()


async def get_dynamic_model_registry(
    models_dir: Optional[Path] = None,
) -> DynamicModelRegistry:
    """Get or create the registry singleton."""
    global _registry

    async with _registry_lock:
        if _registry is None:
            _registry = DynamicModelRegistry(models_dir=models_dir)
            await _registry.initialize()
        return _registry


async def shutdown_dynamic_model_registry() -> None:
    """Shutdown the registry."""
    global _registry

    async with _registry_lock:
        if _registry is not None:
            await _registry.shutdown()
            _registry = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ModelCapability",
    "ModelTierLevel",
    "ModelLocation",
    "ModelState",
    "QuantizationType",
    "TaskType",
    "MemoryPressureLevel",
    "ModelFileEvent",
    "ValidationResult",
    # Data classes
    "QuantizationSpec",
    "ModelSpec",
    "DownloadProgress",
    "TaskClassification",
    "BufferedRequest",
    "ModelDirectoryConfig",
    "DiscoveredModel",
    "ModelFileChange",
    "ModelValidation",
    # Constants
    "QUANTIZATION_SPECS",
    "KNOWN_MODELS",
    "TIER_MODEL_MAPPING",
    "CAPABILITY_MODEL_MAPPING",
    "TASK_TIER_MAPPING",
    "TASK_CAPABILITY_MAPPING",
    # Neural Switchboard Classes (v98.0)
    "TaskClassifier",
    "MacOSMemoryPressureMonitor",
    "CrossRepoMemoryIntegration",
    "HotSwapRequestBuffer",
    "StickyRoutingManager",
    # Multi-Directory Discovery Classes (v99.0)
    "MultiDirectoryModelScanner",
    "ModelFileWatcher",
    "ModelValidationPipeline",
    "ReactorCoreModelSync",
    # Core Classes
    "AdvancedModelDownloader",
    "NeuralModelSelector",
    "ElasticBurstController",
    "ModelMemoryManager",
    "DynamicModelRegistry",
    # Factory functions
    "get_dynamic_model_registry",
    "shutdown_dynamic_model_registry",
]
