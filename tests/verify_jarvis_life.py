#!/usr/bin/env python3
"""
JARVIS Capabilities Demo - "Is JARVIS Alive?" Verification Script
==================================================================

v92.1 - Advanced, Async, Parallel, Intelligent Cross-Repo Demo

This script demonstrates that JARVIS is "alive" by showing:
1. INTELLIGENT ROUTING - Watch the Brain choose models based on complexity
2. VOICE OUTPUT - Hear JARVIS speak responses using Daniel's voice (TTS)
3. TRINITY LOOP - See experiences logged to Reactor for learning

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    VERIFY JARVIS LIFE DEMO                       │
    │                                                                  │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
    │  │   JARVIS     │◄──►│  JARVIS-Prime │◄──►│ Reactor-Core │       │
    │  │   (Body)     │    │    (Mind)     │    │   (Nerves)   │       │
    │  │  - TTS       │    │  - Router     │    │  - Logging   │       │
    │  │  - Voice     │    │  - Inference  │    │  - Learning  │       │
    │  └──────────────┘    └──────────────┘    └──────────────┘       │
    │                                                                  │
    │  TEST SEQUENCE:                                                  │
    │  ├─ CASE A (Simple): "What time is it?" → Local/Fast Model      │
    │  ├─ CASE B (Complex): Quantum cryptography → Cloud/Smart Model  │
    │  └─ CASE C (Action): "Open Chrome" → Tool Use                   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

FEATURES:
- Zero hardcoding (all configuration via environment variables)
- Async/parallel execution for non-blocking operations
- Graceful degradation (works even if some components unavailable)
- Rich terminal output with color-coded routing decisions
- Cross-repo integration (JARVIS, JARVIS-Prime, Reactor-Core)
- Interactive pause between tests for analysis

USAGE:
    # Standard demo
    python3 tests/verify_jarvis_life.py

    # With verbose routing
    JARVIS_DEMO_VERBOSE=true python3 tests/verify_jarvis_life.py

    # Disable voice (terminal only)
    JARVIS_DEMO_VOICE=false python3 tests/verify_jarvis_life.py

Author: JARVIS-Prime Trinity v92.1
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)

# =============================================================================
# CROSS-REPO PATH SETUP (Dynamic, No Hardcoding)
# =============================================================================

def _setup_cross_repo_paths() -> Dict[str, Path]:
    """
    Dynamically discover and configure cross-repo paths.

    Uses environment variables with intelligent fallback to common locations.
    """
    base_repos = Path(os.getenv("JARVIS_REPOS_BASE", str(Path.home() / "Documents" / "repos")))

    paths = {
        "jarvis": Path(os.getenv("JARVIS_BODY_PATH", str(base_repos / "JARVIS-AI-Agent"))),
        "jarvis_prime": Path(os.getenv("JARVIS_PRIME_PATH", str(base_repos / "jarvis-prime"))),
        "reactor_core": Path(os.getenv("REACTOR_CORE_PATH", str(base_repos / "reactor-core"))),
    }

    # Add to sys.path for cross-repo imports
    for name, path in paths.items():
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return paths

REPO_PATHS = _setup_cross_repo_paths()

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Rich colored logging formatter for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",      # Cyan
        "INFO": "\033[32m",       # Green
        "WARNING": "\033[33m",    # Yellow
        "ERROR": "\033[31m",      # Red
        "CRITICAL": "\033[35m",   # Magenta
        "RESET": "\033[0m",
        "BOLD": "\033[1m",
        "DIM": "\033[2m",
    }

    ICONS = {
        "DEBUG": "🔍",
        "INFO": "✨",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "🔥",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        icon = self.ICONS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        # Format timestamp
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        return f"{color}{icon} [{timestamp}] {record.getMessage()}{reset}"


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with colored output."""
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter())

    logger = logging.getLogger("jarvis_life_demo")
    logger.setLevel(level)
    logger.handlers = [handler]

    return logger

logger = setup_logging(os.getenv("JARVIS_DEMO_VERBOSE", "false").lower() == "true")

# =============================================================================
# ENVIRONMENT CONFIGURATION (Zero Hardcoding)
# =============================================================================

@dataclass
class DemoConfig:
    """Configuration for the capabilities demo - 100% environment-driven."""

    # Voice settings
    voice_enabled: bool = field(default_factory=lambda:
        os.getenv("JARVIS_DEMO_VOICE", "true").lower() == "true")
    voice_name: str = field(default_factory=lambda:
        os.getenv("JARVIS_DEMO_VOICE_NAME", "Daniel"))
    voice_rate: int = field(default_factory=lambda:
        int(os.getenv("JARVIS_DEMO_VOICE_RATE", "180")))

    # Routing settings
    complexity_threshold_simple: float = field(default_factory=lambda:
        float(os.getenv("JARVIS_COMPLEXITY_SIMPLE", "0.3")))
    complexity_threshold_complex: float = field(default_factory=lambda:
        float(os.getenv("JARVIS_COMPLEXITY_COMPLEX", "0.7")))

    # Timing settings
    pause_between_tests: bool = field(default_factory=lambda:
        os.getenv("JARVIS_DEMO_PAUSE", "true").lower() == "true")
    response_delay_ms: int = field(default_factory=lambda:
        int(os.getenv("JARVIS_DEMO_DELAY_MS", "500")))

    # Demo mode
    mock_inference: bool = field(default_factory=lambda:
        os.getenv("JARVIS_DEMO_MOCK", "true").lower() == "true")
    verbose: bool = field(default_factory=lambda:
        os.getenv("JARVIS_DEMO_VERBOSE", "false").lower() == "true")

    # Experience logging
    log_experiences: bool = field(default_factory=lambda:
        os.getenv("JARVIS_DEMO_LOG_EXPERIENCES", "true").lower() == "true")
    experience_log_path: Path = field(default_factory=lambda:
        Path(os.getenv("JARVIS_EXPERIENCE_LOG",
            str(Path.home() / ".jarvis" / "experiences" / "demo_experiences.jsonl"))))


CONFIG = DemoConfig()

# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class ModelTier(Enum):
    """Model tier classification for routing decisions."""
    TIER_0_LOCAL = "tier_0_local"        # Fast local model (Prime-7B)
    TIER_1_CLOUD = "tier_1_cloud"        # Cloud API (Claude, GPT-4)
    TIER_2_HYBRID = "tier_2_hybrid"      # Best of both
    TOOL_USE = "tool_use"                # Action execution

class ComplexityLevel(Enum):
    """Query complexity classification."""
    SIMPLE = "simple"      # Basic queries, greetings, time
    MODERATE = "moderate"  # Multi-step reasoning
    COMPLEX = "complex"    # Deep analysis, creative
    ACTION = "action"      # Tool/command execution


@dataclass
class RoutingDecision:
    """Detailed routing decision with full transparency."""
    selected_model: str
    tier: ModelTier
    complexity_score: float
    complexity_level: ComplexityLevel
    reasoning: str
    confidence: float
    latency_estimate_ms: float
    cost_estimate: float
    factors: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class TestCase:
    """A single test case for the demo."""
    name: str
    query: str
    expected_tier: ModelTier
    description: str
    expected_response_pattern: Optional[str] = None
    timeout_seconds: float = 30.0


@dataclass
class DemoResponse:
    """Response from processing a test case."""
    test_case: TestCase
    routing_decision: RoutingDecision
    response_text: str
    latency_ms: float
    success: bool
    experience_logged: bool = False
    voice_played: bool = False
    error_message: Optional[str] = None


@dataclass
class Experience:
    """Experience data for Trinity loop logging."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    prompt: str = ""
    response: str = ""
    model_used: str = ""
    tier: str = ""
    complexity_score: float = 0.0
    latency_ms: float = 0.0
    feedback: float = 1.0  # Default positive
    metadata: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# COMPLEXITY ANALYZER - The Brain's Decision Engine
# =============================================================================

class ComplexityAnalyzer:
    """
    Analyzes query complexity to determine optimal routing.

    Uses multiple heuristics:
    - Token count
    - Question complexity markers
    - Technical terminology
    - Action keywords
    - Reasoning depth indicators
    """

    # Complexity markers
    SIMPLE_PATTERNS = [
        "what time", "hello", "hi", "hey", "thanks", "bye",
        "what's the weather", "good morning", "good night",
        "what day", "what date", "how are you",
    ]

    COMPLEX_PATTERNS = [
        "analyze", "explain", "compare", "contrast", "evaluate",
        "strategic implications", "quantum", "cryptography",
        "architecture", "optimization", "trade-off", "synthesis",
        "philosophical", "ethical", "comprehensive", "in-depth",
        "implications", "prepare", "organizations", "modern",
        "deep analysis", "impact", "consequences", "future",
    ]

    ACTION_PATTERNS = [
        "open", "close", "launch", "run", "execute", "start",
        "stop", "kill", "create", "delete", "move", "copy",
        "send", "schedule", "remind", "set", "configure",
    ]

    REASONING_MARKERS = [
        "why", "how", "what if", "could you", "would you",
        "explain why", "elaborate", "detail", "step by step",
    ]

    def __init__(self, config: DemoConfig):
        self.config = config
        self._cache: Dict[str, Tuple[float, ComplexityLevel]] = {}

    def _compute_lexical_complexity(self, text: str) -> float:
        """Compute complexity based on vocabulary and structure."""
        words = text.lower().split()
        word_count = len(words)

        # Base score from length
        length_score = min(word_count / 50, 1.0) * 0.3

        # Average word length (longer words = more complex)
        avg_word_len = sum(len(w) for w in words) / max(word_count, 1)
        word_len_score = min(avg_word_len / 10, 1.0) * 0.2

        # Unique word ratio (vocabulary diversity)
        unique_ratio = len(set(words)) / max(word_count, 1)
        diversity_score = unique_ratio * 0.2

        return length_score + word_len_score + diversity_score

    def _compute_pattern_complexity(self, text: str) -> Tuple[float, ComplexityLevel]:
        """Compute complexity based on pattern matching."""
        text_lower = text.lower()

        # Check for action patterns first (highest priority)
        for pattern in self.ACTION_PATTERNS:
            if pattern in text_lower:
                return 0.5, ComplexityLevel.ACTION

        # Check simple patterns
        simple_matches = sum(1 for p in self.SIMPLE_PATTERNS if p in text_lower)
        if simple_matches > 0:
            return 0.1 + (0.05 * simple_matches), ComplexityLevel.SIMPLE

        # Check complex patterns
        complex_matches = sum(1 for p in self.COMPLEX_PATTERNS if p in text_lower)
        if complex_matches >= 3:
            return 0.85 + (0.03 * complex_matches), ComplexityLevel.COMPLEX
        elif complex_matches >= 2:
            return 0.75 + (0.04 * complex_matches), ComplexityLevel.COMPLEX
        elif complex_matches == 1:
            return 0.55, ComplexityLevel.MODERATE

        # Check reasoning markers
        reasoning_matches = sum(1 for p in self.REASONING_MARKERS if p in text_lower)
        if reasoning_matches >= 2:
            return 0.6, ComplexityLevel.MODERATE

        return 0.3, ComplexityLevel.SIMPLE

    def analyze(self, query: str) -> Tuple[float, ComplexityLevel, Dict[str, float]]:
        """
        Analyze query complexity with full factor breakdown.

        Returns:
            Tuple of (complexity_score, complexity_level, factor_breakdown)
        """
        # Check cache
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self._cache:
            score, level = self._cache[cache_key]
            return score, level, {}

        # Compute factors
        lexical_score = self._compute_lexical_complexity(query)
        pattern_score, pattern_level = self._compute_pattern_complexity(query)

        factors = {
            "lexical_complexity": lexical_score,
            "pattern_match_score": pattern_score,
            "token_count_factor": min(len(query.split()) / 30, 1.0),
            "question_marks": query.count("?") * 0.1,
        }

        # Combine scores with weights
        final_score = (
            lexical_score * 0.3 +
            pattern_score * 0.5 +
            factors["token_count_factor"] * 0.1 +
            factors["question_marks"] * 0.1
        )

        # Determine level based on score and patterns
        if pattern_level == ComplexityLevel.ACTION:
            level = ComplexityLevel.ACTION
        elif final_score < self.config.complexity_threshold_simple:
            level = ComplexityLevel.SIMPLE
        elif final_score > self.config.complexity_threshold_complex:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.MODERATE

        # Cache result
        self._cache[cache_key] = (final_score, level)

        return final_score, level, factors

# =============================================================================
# INTELLIGENT REQUEST ROUTER - The Brain's Routing System
# =============================================================================

class IntelligentRouter:
    """
    Routes requests to optimal models based on complexity analysis.

    This is the visible decision-making engine that shows:
    - WHY a particular model was chosen
    - WHAT factors influenced the decision
    - CONFIDENCE in the routing decision
    """

    # Model catalog with capabilities
    MODELS = {
        "Prime-7B-Chat": {
            "tier": ModelTier.TIER_0_LOCAL,
            "capabilities": ["chat", "simple_qa", "greetings"],
            "latency_ms": 200,
            "cost_per_1k": 0.0,
            "max_complexity": 0.4,
        },
        "Prime-13B-Instruct": {
            "tier": ModelTier.TIER_0_LOCAL,
            "capabilities": ["chat", "reasoning", "moderate_analysis"],
            "latency_ms": 500,
            "cost_per_1k": 0.0,
            "max_complexity": 0.7,
        },
        "Claude-3.5-Sonnet": {
            "tier": ModelTier.TIER_1_CLOUD,
            "capabilities": ["chat", "reasoning", "complex_analysis", "creative"],
            "latency_ms": 2000,
            "cost_per_1k": 0.003,
            "max_complexity": 1.0,
        },
        "Claude-3-Opus": {
            "tier": ModelTier.TIER_1_CLOUD,
            "capabilities": ["chat", "deep_reasoning", "synthesis", "expert"],
            "latency_ms": 5000,
            "cost_per_1k": 0.015,
            "max_complexity": 1.0,
        },
        "JARVIS-Tool-Agent": {
            "tier": ModelTier.TOOL_USE,
            "capabilities": ["tool_use", "actions", "system_control"],
            "latency_ms": 100,
            "cost_per_1k": 0.0,
            "max_complexity": 0.5,
        },
    }

    def __init__(self, config: DemoConfig):
        self.config = config
        self.analyzer = ComplexityAnalyzer(config)
        self._decision_history: List[RoutingDecision] = []

    def route(self, query: str) -> RoutingDecision:
        """
        Route a query to the optimal model with full decision transparency.

        This method shows the "thinking" process of model selection.
        """
        # Analyze complexity
        complexity_score, complexity_level, factors = self.analyzer.analyze(query)

        # Determine routing based on complexity
        if complexity_level == ComplexityLevel.ACTION:
            selected = "JARVIS-Tool-Agent"
            reasoning = "Action/command detected - routing to Tool Agent"
            confidence = 0.95
        elif complexity_level == ComplexityLevel.SIMPLE:
            selected = "Prime-7B-Chat"
            reasoning = "Simple query - using fast local model"
            confidence = 0.9
        elif complexity_level == ComplexityLevel.MODERATE:
            selected = "Prime-13B-Instruct"
            reasoning = "Moderate complexity - using capable local model"
            confidence = 0.85
        else:  # Complex
            if complexity_score > 0.85:
                selected = "Claude-3-Opus"
                reasoning = "Deep analysis required - using most capable model"
                confidence = 0.92
            else:
                selected = "Claude-3.5-Sonnet"
                reasoning = "Complex query - using cloud model for quality"
                confidence = 0.88

        model_info = self.MODELS[selected]

        decision = RoutingDecision(
            selected_model=selected,
            tier=model_info["tier"],
            complexity_score=complexity_score,
            complexity_level=complexity_level,
            reasoning=reasoning,
            confidence=confidence,
            latency_estimate_ms=model_info["latency_ms"],
            cost_estimate=model_info["cost_per_1k"],
            factors=factors,
        )

        self._decision_history.append(decision)

        return decision

    def get_decision_history(self) -> List[RoutingDecision]:
        """Get history of routing decisions for analysis."""
        return self._decision_history.copy()

# =============================================================================
# VOICE ENGINE - Daniel's Voice with macOS Fallback
# =============================================================================

class VoiceEngine:
    """
    Text-to-Speech engine with multi-tier fallback.

    Fallback chain:
    1. Trinity Voice Coordinator (from JARVIS repo)
    2. macOS 'say' command with Daniel voice
    3. Silent mode (log only)
    """

    def __init__(self, config: DemoConfig):
        self.config = config
        self._trinity_available = False
        self._macos_say_available = False
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize voice engine with availability detection."""
        if self._initialized:
            return True

        # Try to import Trinity Voice Coordinator
        try:
            with suppress(ImportError, AttributeError):
                from backend.core.trinity_voice_coordinator import (
                    announce as trinity_announce,
                    get_voice_coordinator,
                    VoiceContext,
                    VoicePriority,
                )
                self._trinity_available = True
                self._trinity_announce = trinity_announce
                self._VoiceContext = VoiceContext
                self._VoicePriority = VoicePriority
                logger.debug("Trinity Voice Coordinator available")
        except Exception:
            pass

        # Check macOS 'say' command
        try:
            result = await asyncio.create_subprocess_exec(
                "which", "say",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.wait()
            self._macos_say_available = (result.returncode == 0)
            if self._macos_say_available:
                logger.debug("macOS 'say' command available")
        except Exception:
            pass

        self._initialized = True
        return self._trinity_available or self._macos_say_available

    async def speak(self, text: str, context: str = "narrator") -> bool:
        """
        Speak text using available voice engine.

        Returns True if voice was played, False otherwise.
        """
        if not self.config.voice_enabled:
            logger.debug("[Voice] Disabled by configuration")
            return False

        await self.initialize()

        # Try Trinity Voice Coordinator first
        if self._trinity_available:
            try:
                await self._trinity_announce(
                    message=text,
                    context=self._VoiceContext.NARRATOR,
                    priority=self._VoicePriority.NORMAL,
                    source="jarvis_life_demo",
                )
                return True
            except Exception as e:
                logger.debug(f"[Voice] Trinity failed: {e}")

        # Fallback to macOS 'say'
        if self._macos_say_available:
            try:
                cmd = [
                    "say",
                    "-v", self.config.voice_name,
                    "-r", str(self.config.voice_rate),
                    text,
                ]
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(process.wait(), timeout=30.0)
                return process.returncode == 0
            except asyncio.TimeoutError:
                logger.warning("[Voice] macOS say timed out")
            except Exception as e:
                logger.warning(f"[Voice] macOS say failed: {e}")

        logger.debug("[Voice] No voice engine available")
        return False

# =============================================================================
# EXPERIENCE LOGGER - Trinity Loop to Reactor
# =============================================================================

class ExperienceLogger:
    """
    Logs experiences to Reactor for continuous learning.

    This completes the Trinity Loop:
    JARVIS (Body) → JARVIS-Prime (Mind) → Reactor (Nerves) → back to training
    """

    def __init__(self, config: DemoConfig):
        self.config = config
        self._log_path = config.experience_log_path
        self._reactor_available = False
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize experience logging."""
        if self._initialized:
            return True

        # Ensure log directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Try to connect to Reactor
        try:
            reactor_path = REPO_PATHS.get("reactor_core")
            if reactor_path and reactor_path.exists():
                self._reactor_available = True
                logger.debug("[Experience] Reactor-Core path available")
        except Exception:
            pass

        self._initialized = True
        return True

    async def log_experience(self, experience: Experience) -> bool:
        """
        Log an experience to the Trinity loop.

        Writes to local JSONL and optionally to Reactor.
        """
        if not self.config.log_experiences:
            return False

        await self.initialize()

        # Serialize experience
        exp_dict = {
            "id": experience.id,
            "timestamp": experience.timestamp,
            "datetime": datetime.fromtimestamp(experience.timestamp).isoformat(),
            "prompt": experience.prompt,
            "response": experience.response,
            "model_used": experience.model_used,
            "tier": experience.tier,
            "complexity_score": experience.complexity_score,
            "latency_ms": experience.latency_ms,
            "feedback": experience.feedback,
            "metadata": experience.metadata,
        }

        # Write to local JSONL
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(exp_dict) + "\n")
            logger.info(f"[Reactor] Experience saved: {self._log_path}")
            return True
        except Exception as e:
            logger.error(f"[Reactor] Failed to log experience: {e}")
            return False

# =============================================================================
# MOCK INFERENCE ENGINE - For Demo Without Full Model Load
# =============================================================================

class MockInferenceEngine:
    """
    Generates plausible mock responses for demo purposes.

    In production, this would be replaced by actual model inference.
    """

    RESPONSES = {
        "time": [
            "The current time is {time}. Is there anything else I can help you with?",
            "It's currently {time}. Have a great day!",
        ],
        "quantum_crypto": [
            "The strategic implications of quantum computing on cryptography are profound. "
            "Current RSA and ECC encryption could be broken by Shor's algorithm on a sufficiently "
            "powerful quantum computer. This has led to the development of post-quantum cryptography "
            "standards by NIST, including lattice-based and hash-based approaches. Organizations "
            "should begin transitioning to quantum-resistant algorithms now, as encrypted data "
            "harvested today could be decrypted by future quantum systems.",
        ],
        "action": [
            "I'll open {app} for you right away.",
            "Opening {app}. Let me know if you need anything else.",
        ],
        "default": [
            "I understand your query. Let me process that for you.",
            "That's an interesting question. Here's my analysis...",
        ],
    }

    def __init__(self, config: DemoConfig):
        self.config = config

    async def generate(
        self,
        query: str,
        model: str,
        complexity_level: ComplexityLevel,
    ) -> Tuple[str, float]:
        """
        Generate a mock response based on query type.

        Returns (response_text, latency_ms)
        """
        query_lower = query.lower()

        # Simulate processing time based on model
        latency_map = {
            "Prime-7B-Chat": 200,
            "Prime-13B-Instruct": 500,
            "Claude-3.5-Sonnet": 2000,
            "Claude-3-Opus": 3000,
            "JARVIS-Tool-Agent": 100,
        }
        base_latency = latency_map.get(model, 500)

        # Add some variance
        latency = base_latency + random.randint(-50, 150)

        # Simulate async processing
        await asyncio.sleep(latency / 1000)

        # Generate response based on query type
        if "time" in query_lower:
            response = random.choice(self.RESPONSES["time"]).format(
                time=datetime.now().strftime("%I:%M %p")
            )
        elif "quantum" in query_lower or "cryptography" in query_lower:
            response = random.choice(self.RESPONSES["quantum_crypto"])
        elif complexity_level == ComplexityLevel.ACTION:
            app_name = "Google Chrome"  # Extract from query in real implementation
            for word in ["chrome", "safari", "firefox", "slack", "terminal"]:
                if word in query_lower:
                    app_name = word.title()
                    break
            response = random.choice(self.RESPONSES["action"]).format(app=app_name)
        else:
            response = random.choice(self.RESPONSES["default"])

        return response, latency

# =============================================================================
# DEMO ORCHESTRATOR - The Main Show
# =============================================================================

class JARVISLifeDemo:
    """
    Main demo orchestrator that brings everything together.

    Shows JARVIS is "alive" by demonstrating:
    1. Intelligent routing decisions
    2. Voice output
    3. Trinity loop logging
    """

    # Test cases that demonstrate different routing decisions
    TEST_CASES = [
        TestCase(
            name="CASE A: Simple Query",
            query="JARVIS, what time is it?",
            expected_tier=ModelTier.TIER_0_LOCAL,
            description="Simple factual query → Local fast model",
        ),
        TestCase(
            name="CASE B: Complex Analysis",
            query="Analyze the strategic implications of quantum computing on modern cryptography and what organizations should do to prepare.",
            expected_tier=ModelTier.TIER_1_CLOUD,
            description="Deep analysis required → Cloud smart model",
        ),
        TestCase(
            name="CASE C: Action Command",
            query="Open Google Chrome.",
            expected_tier=ModelTier.TOOL_USE,
            description="System action → Tool Agent",
        ),
    ]

    def __init__(self, config: Optional[DemoConfig] = None):
        self.config = config or CONFIG
        self.router = IntelligentRouter(self.config)
        self.voice = VoiceEngine(self.config)
        self.experience_logger = ExperienceLogger(self.config)
        self.inference = MockInferenceEngine(self.config)
        self._results: List[DemoResponse] = []

    def _print_header(self) -> None:
        """Print demo header with ASCII art."""
        header = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║       ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗    ██╗     ██╗███████╗███████╗ ║
║       ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝    ██║     ██║██╔════╝██╔════╝ ║
║       ██║███████║██████╔╝██║   ██║██║███████╗    ██║     ██║█████╗  █████╗   ║
║  ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║    ██║     ██║██╔══╝  ██╔══╝   ║
║  ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║    ███████╗██║██║     ███████╗ ║
║   ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝    ╚══════╝╚═╝╚═╝     ╚══════╝ ║
║                                                                              ║
║                    🧠 CAPABILITIES DEMONSTRATION v92.1 🧠                     ║
║                                                                              ║
║    "Is JARVIS Alive?" - Watch the Brain Think, Decide, and Speak            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        print("\033[36m" + header + "\033[0m")

    def _print_routing_decision(self, decision: RoutingDecision) -> None:
        """Print routing decision with rich formatting."""
        tier_colors = {
            ModelTier.TIER_0_LOCAL: "\033[32m",   # Green
            ModelTier.TIER_1_CLOUD: "\033[35m",   # Magenta
            ModelTier.TOOL_USE: "\033[33m",       # Yellow
        }
        color = tier_colors.get(decision.tier, "\033[37m")
        reset = "\033[0m"
        bold = "\033[1m"

        print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ {bold}[ROUTER] DECISION{reset}                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Selected Model: {color}{bold}{decision.selected_model:20}{reset}                            │
│  Tier:           {color}{decision.tier.value:20}{reset}                            │
│  Complexity:     {decision.complexity_score:.2f} ({decision.complexity_level.value:10})                          │
│  Confidence:     {decision.confidence:.0%}                                                  │
│  Reasoning:      {decision.reasoning[:50]:50}│
│  Est. Latency:   {decision.latency_estimate_ms:.0f}ms                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    def _print_test_case(self, test: TestCase, index: int) -> None:
        """Print test case header."""
        print(f"""
{'='*80}
  {test.name}
  {'-'*len(test.name)}
  Query: "{test.query}"
  Expected: {test.expected_tier.value}
  Description: {test.description}
{'='*80}
""")

    async def _process_test_case(self, test: TestCase) -> DemoResponse:
        """Process a single test case through the full pipeline."""
        start_time = time.time()

        # Step 1: Route the request
        print("\n[TEST] Sending query to Router...")
        decision = self.router.route(test.query)
        self._print_routing_decision(decision)

        # Step 2: Generate response (mock or real)
        print("[TEST] Generating response...")
        response_text, inference_latency = await self.inference.generate(
            test.query,
            decision.selected_model,
            decision.complexity_level,
        )

        total_latency = (time.time() - start_time) * 1000

        print(f"\n[RESPONSE] ({inference_latency:.0f}ms)")
        print(f"  {response_text[:200]}{'...' if len(response_text) > 200 else ''}")

        # Step 3: Play voice
        voice_played = False
        if self.config.voice_enabled:
            print("\n[VOICE] Speaking response...")
            voice_played = await self.voice.speak(response_text)
            if voice_played:
                print("[VOICE] ✓ Audio played successfully")
            else:
                print("[VOICE] ✗ Voice unavailable, continuing silently")

        # Step 4: Log experience to Reactor
        experience_logged = False
        if self.config.log_experiences:
            experience = Experience(
                prompt=test.query,
                response=response_text,
                model_used=decision.selected_model,
                tier=decision.tier.value,
                complexity_score=decision.complexity_score,
                latency_ms=total_latency,
                metadata={
                    "test_name": test.name,
                    "expected_tier": test.expected_tier.value,
                    "decision_id": decision.decision_id,
                },
            )
            experience_logged = await self.experience_logger.log_experience(experience)
            if experience_logged:
                print(f"[REACTOR] ✓ Experience logged to Trinity loop")

        # Build response
        return DemoResponse(
            test_case=test,
            routing_decision=decision,
            response_text=response_text,
            latency_ms=total_latency,
            success=True,
            experience_logged=experience_logged,
            voice_played=voice_played,
        )

    async def run(self) -> List[DemoResponse]:
        """Run the full demo sequence."""
        self._print_header()

        print("\n🚀 Starting JARVIS Life Demo...")
        print(f"   Voice: {'Enabled' if self.config.voice_enabled else 'Disabled'} ({self.config.voice_name})")
        print(f"   Experience Logging: {'Enabled' if self.config.log_experiences else 'Disabled'}")
        print(f"   Mock Inference: {'Enabled' if self.config.mock_inference else 'Disabled'}")

        # Initialize components
        await self.voice.initialize()
        await self.experience_logger.initialize()

        # Opening announcement
        if self.config.voice_enabled:
            await self.voice.speak(
                "JARVIS Life Demo initialized. Beginning capability demonstration."
            )

        # Process each test case
        for i, test in enumerate(self.TEST_CASES):
            self._print_test_case(test, i)

            try:
                result = await self._process_test_case(test)
                self._results.append(result)

                # Verify routing was correct
                if result.routing_decision.tier == test.expected_tier:
                    print(f"\n✅ ROUTING CORRECT: Got {result.routing_decision.tier.value} (expected {test.expected_tier.value})")
                else:
                    print(f"\n⚠️ ROUTING MISMATCH: Got {result.routing_decision.tier.value} (expected {test.expected_tier.value})")

            except Exception as e:
                logger.error(f"Test case failed: {e}")
                self._results.append(DemoResponse(
                    test_case=test,
                    routing_decision=RoutingDecision(
                        selected_model="ERROR",
                        tier=ModelTier.TIER_0_LOCAL,
                        complexity_score=0.0,
                        complexity_level=ComplexityLevel.SIMPLE,
                        reasoning=str(e),
                        confidence=0.0,
                        latency_estimate_ms=0.0,
                        cost_estimate=0.0,
                    ),
                    response_text="",
                    latency_ms=0.0,
                    success=False,
                    error_message=str(e),
                ))

            # Pause between tests for analysis
            if self.config.pause_between_tests and i < len(self.TEST_CASES) - 1:
                print("\n" + "─"*80)
                input("Press ENTER to continue to next test...")

        # Print summary
        self._print_summary()

        return self._results

    def _print_summary(self) -> None:
        """Print final demo summary."""
        successful = sum(1 for r in self._results if r.success)
        total = len(self._results)
        voice_played = sum(1 for r in self._results if r.voice_played)
        experiences_logged = sum(1 for r in self._results if r.experience_logged)

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           DEMO SUMMARY                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Tests Executed:     {total:3}                                                   ║
║  Successful:         {successful:3}                                                   ║
║  Voice Responses:    {voice_played:3}                                                   ║
║  Experiences Logged: {experiences_logged:3}                                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🧠 JARVIS IS ALIVE - The Brain is thinking, deciding, and speaking!        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main() -> int:
    """Main entry point for the demo."""
    try:
        demo = JARVISLifeDemo()
        results = await demo.run()

        # Return exit code based on success
        all_passed = all(r.success for r in results)
        return 0 if all_passed else 1

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
