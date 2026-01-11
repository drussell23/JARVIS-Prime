"""
JARVIS-Prime Integration Bridge - Unified LLM Interface
==========================================================

v92.0 - Seamless Local-to-API Inference

This module provides a unified interface that:
- Makes PrimeModel the PRIMARY inference source
- Seamlessly falls back to Claude API when needed
- Captures all interactions for continuous learning
- Provides drop-in replacement for existing Claude API calls
- Manages model selection based on task complexity

DESIGN PHILOSOPHY:
    "Local First, API Fallback" - Maximize local model usage while
    ensuring reliability through intelligent fallback.

ARCHITECTURE:
    JarvisPrimeBridge
        |
        +-- PrimeInferenceEngine (local model inference)
        +-- ClaudeAPIEngine (fallback API inference)
        +-- TaskRouter (complexity-based routing)
        +-- ConversationCapture (training data collection)
        +-- ResponseEnhancer (post-processing)

FALLBACK CHAIN:
    1. Try PrimeModel (local 7B)
    2. If too complex → Try larger model (13B via GCP)
    3. If unavailable → Fall back to Claude API
    4. Always capture for training

USAGE:
    # Drop-in replacement for Claude API
    bridge = await get_jarvis_prime_bridge()

    # Simple generation (like Claude)
    response = await bridge.generate(
        prompt="What is the weather?",
        system="You are JARVIS..."
    )

    # Chat completion (OpenAI-compatible)
    response = await bridge.chat(
        messages=[
            {"role": "system", "content": "You are JARVIS"},
            {"role": "user", "content": "Open Safari"}
        ]
    )

    # Force local only (no API fallback)
    response = await bridge.generate(
        prompt="Simple task",
        force_local=True
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class BridgeConfig:
    """Configuration for the JARVIS-Prime bridge."""

    # Model preferences
    prefer_local: bool = field(default_factory=lambda:
        os.getenv("BRIDGE_PREFER_LOCAL", "true").lower() == "true"
    )
    default_local_model: str = field(default_factory=lambda:
        os.getenv("BRIDGE_DEFAULT_MODEL", "prime-7b-chat-v1")
    )
    complex_task_model: str = field(default_factory=lambda:
        os.getenv("BRIDGE_COMPLEX_MODEL", "prime-13b-reasoning-v1")
    )

    # Fallback settings
    enable_api_fallback: bool = field(default_factory=lambda:
        os.getenv("BRIDGE_API_FALLBACK", "true").lower() == "true"
    )
    fallback_timeout_ms: int = field(default_factory=lambda: int(
        os.getenv("BRIDGE_FALLBACK_TIMEOUT_MS", "5000")
    ))
    max_local_retries: int = field(default_factory=lambda: int(
        os.getenv("BRIDGE_LOCAL_RETRIES", "2")
    ))

    # Complexity thresholds
    simple_task_threshold: float = field(default_factory=lambda: float(
        os.getenv("BRIDGE_SIMPLE_THRESHOLD", "0.3")
    ))
    complex_task_threshold: float = field(default_factory=lambda: float(
        os.getenv("BRIDGE_COMPLEX_THRESHOLD", "0.7")
    ))

    # Capture settings
    capture_interactions: bool = field(default_factory=lambda:
        os.getenv("BRIDGE_CAPTURE", "true").lower() == "true"
    )
    capture_api_responses: bool = field(default_factory=lambda:
        os.getenv("BRIDGE_CAPTURE_API", "true").lower() == "true"
    )

    # Generation defaults
    default_max_tokens: int = field(default_factory=lambda: int(
        os.getenv("BRIDGE_MAX_TOKENS", "1024")
    ))
    default_temperature: float = field(default_factory=lambda: float(
        os.getenv("BRIDGE_TEMPERATURE", "0.7")
    ))

    # Claude API settings
    claude_model: str = field(default_factory=lambda:
        os.getenv("BRIDGE_CLAUDE_MODEL", "claude-sonnet-4-20250514")
    )
    anthropic_api_key: Optional[str] = field(default_factory=lambda:
        os.getenv("ANTHROPIC_API_KEY")
    )


# =============================================================================
# ENUMS
# =============================================================================


class InferenceSource(Enum):
    """Source of inference."""
    LOCAL_PRIME = "local_prime"
    GCP_PRIME = "gcp_prime"
    CLAUDE_API = "claude_api"
    HYBRID = "hybrid"


class TaskComplexity(Enum):
    """Complexity level of a task."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class ResponseQuality(Enum):
    """Quality assessment of response."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class BridgeRequest:
    """A request to the bridge."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Input
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    system: Optional[str] = None

    # Generation settings
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: List[str] = field(default_factory=list)

    # Routing hints
    force_local: bool = False
    force_api: bool = False
    preferred_model: Optional[str] = None

    # Context
    task_type: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BridgeResponse:
    """Response from the bridge."""
    request_id: str
    text: str
    source: InferenceSource

    # Metadata
    model_used: str = ""
    tokens_generated: int = 0
    latency_ms: float = 0.0
    complexity_score: float = 0.0

    # Quality
    quality: ResponseQuality = ResponseQuality.GOOD
    fallback_used: bool = False
    fallback_reason: Optional[str] = None

    # For streaming
    is_complete: bool = True

    # Raw response
    raw_response: Optional[Dict[str, Any]] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class InferenceStats:
    """Statistics for inference."""
    total_requests: int = 0
    local_requests: int = 0
    api_requests: int = 0
    fallback_count: int = 0

    total_tokens: int = 0
    total_latency_ms: float = 0.0

    # Success rates
    local_success: int = 0
    local_failure: int = 0
    api_success: int = 0
    api_failure: int = 0

    # By complexity
    simple_tasks: int = 0
    moderate_tasks: int = 0
    complex_tasks: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def local_success_rate(self) -> float:
        total = self.local_success + self.local_failure
        if total == 0:
            return 1.0
        return self.local_success / total

    @property
    def fallback_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.fallback_count / self.total_requests


# =============================================================================
# TASK ROUTER
# =============================================================================


class TaskRouter:
    """Routes tasks based on complexity analysis."""

    def __init__(self, config: BridgeConfig):
        self._config = config

    async def analyze_complexity(
        self,
        request: BridgeRequest,
    ) -> Tuple[TaskComplexity, float]:
        """Analyze task complexity."""
        # Get text to analyze
        text = request.prompt or ""
        if request.messages:
            text = " ".join(m.get("content", "") for m in request.messages)

        # Use auto model selector if available
        try:
            from jarvis_prime.core.auto_model_selector import get_auto_model_selector

            selector = await get_auto_model_selector()
            result = await selector.select_model(text, request.context)

            score = result.complexity_score

        except Exception:
            # Fallback to simple heuristics
            score = self._simple_complexity_analysis(text)

        # Map score to complexity level
        if score < self._config.simple_task_threshold:
            return TaskComplexity.SIMPLE, score
        elif score < self._config.complex_task_threshold:
            return TaskComplexity.MODERATE, score
        elif score < 0.9:
            return TaskComplexity.COMPLEX, score
        else:
            return TaskComplexity.EXPERT, score

    def _simple_complexity_analysis(self, text: str) -> float:
        """Simple heuristic complexity analysis."""
        score = 0.0

        # Length
        word_count = len(text.split())
        if word_count > 500:
            score += 0.2
        elif word_count > 200:
            score += 0.1

        # Technical indicators
        technical_keywords = [
            "analyze", "explain", "reason", "debug", "optimize",
            "architecture", "implement", "algorithm", "complex",
            "multi-step", "comprehensive", "detailed"
        ]
        for keyword in technical_keywords:
            if keyword in text.lower():
                score += 0.05

        # Code indicators
        if "```" in text or "def " in text or "class " in text:
            score += 0.15

        # Question complexity
        if text.count("?") > 2:
            score += 0.1

        return min(1.0, score)

    async def select_model(
        self,
        request: BridgeRequest,
        complexity: TaskComplexity,
    ) -> str:
        """Select the best model for the task."""
        if request.preferred_model:
            return request.preferred_model

        if complexity == TaskComplexity.SIMPLE:
            return self._config.default_local_model
        elif complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT):
            return self._config.complex_task_model
        else:
            return self._config.default_local_model


# =============================================================================
# PRIME INFERENCE ENGINE
# =============================================================================


class PrimeInferenceEngine:
    """Handles local Prime model inference."""

    def __init__(self, config: BridgeConfig):
        self._config = config
        self._model_cache: Dict[str, Any] = {}

    async def generate(
        self,
        request: BridgeRequest,
        model_name: str,
    ) -> Optional[BridgeResponse]:
        """Generate using local Prime model."""
        start_time = time.perf_counter()

        try:
            # Try unified inference first
            try:
                from jarvis_prime.core.unified_inference import get_unified_client

                client = await get_unified_client()

                if request.messages:
                    response = await client.chat(
                        messages=request.messages,
                        model=model_name,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    )
                else:
                    response = await client.generate(
                        prompt=request.prompt or "",
                        model=model_name,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    )

                latency_ms = (time.perf_counter() - start_time) * 1000

                return BridgeResponse(
                    request_id=request.id,
                    text=response.text,
                    source=InferenceSource.LOCAL_PRIME,
                    model_used=model_name,
                    tokens_generated=response.tokens_used,
                    latency_ms=latency_ms,
                )

            except ImportError:
                pass

            # Fallback to direct PrimeModel
            from jarvis_prime.models.prime_model import PrimeModel

            # Format prompt
            if request.messages:
                prompt = self._format_messages(request.messages, request.system)
            else:
                prompt = request.prompt or ""
                if request.system:
                    prompt = f"System: {request.system}\n\nUser: {prompt}\n\nAssistant:"

            # Get or load model
            model = await self._get_model(model_name)
            if not model:
                return None

            # Generate
            result = await model.generate_async(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            return BridgeResponse(
                request_id=request.id,
                text=result.text,
                source=InferenceSource.LOCAL_PRIME,
                model_used=model_name,
                tokens_generated=result.tokens_generated,
                latency_ms=latency_ms,
            )

        except Exception as e:
            logger.warning(f"Prime inference failed: {e}")
            return None

    async def _get_model(self, model_name: str) -> Optional[Any]:
        """Get or load a model."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        try:
            from jarvis_prime.models.prime_model import PrimeModel

            model = await PrimeModel.from_pretrained_async(model_name)
            self._model_cache[model_name] = model
            return model

        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            return None

    def _format_messages(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
    ) -> str:
        """Format messages into a prompt."""
        parts = []

        if system:
            parts.append(f"System: {system}")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system" and not system:
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        parts.append("Assistant:")

        return "\n\n".join(parts)

    async def stream(
        self,
        request: BridgeRequest,
        model_name: str,
    ) -> AsyncIterator[str]:
        """Stream generation from local Prime model."""
        try:
            from jarvis_prime.models.prime_model import PrimeModel

            # Format prompt
            if request.messages:
                prompt = self._format_messages(request.messages, request.system)
            else:
                prompt = request.prompt or ""
                if request.system:
                    prompt = f"System: {request.system}\n\nUser: {prompt}\n\nAssistant:"

            model = await self._get_model(model_name)
            if not model:
                return

            async for token in model.stream_async(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield token

        except Exception as e:
            logger.warning(f"Prime streaming failed: {e}")


# =============================================================================
# CLAUDE API ENGINE
# =============================================================================


class ClaudeAPIEngine:
    """Handles Claude API inference."""

    def __init__(self, config: BridgeConfig):
        self._config = config
        self._client = None

    async def _get_client(self):
        """Get or create Anthropic client."""
        if self._client:
            return self._client

        if not self._config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(
                api_key=self._config.anthropic_api_key
            )
            return self._client
        except ImportError:
            raise ImportError("anthropic package not installed")

    async def generate(
        self,
        request: BridgeRequest,
    ) -> Optional[BridgeResponse]:
        """Generate using Claude API."""
        start_time = time.perf_counter()

        try:
            client = await self._get_client()

            # Format messages
            messages = request.messages or []
            if request.prompt and not messages:
                messages = [{"role": "user", "content": request.prompt}]

            # Extract system from messages if needed
            system = request.system or ""
            filtered_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system = msg.get("content", "")
                else:
                    filtered_messages.append(msg)

            # Make API call
            response = await client.messages.create(
                model=self._config.claude_model,
                max_tokens=request.max_tokens,
                system=system,
                messages=filtered_messages,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract text
            text = ""
            if response.content:
                text = response.content[0].text

            return BridgeResponse(
                request_id=request.id,
                text=text,
                source=InferenceSource.CLAUDE_API,
                model_used=self._config.claude_model,
                tokens_generated=response.usage.output_tokens,
                latency_ms=latency_ms,
                raw_response={
                    "id": response.id,
                    "model": response.model,
                    "usage": {
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                },
            )

        except Exception as e:
            logger.warning(f"Claude API failed: {e}")
            return None

    async def stream(
        self,
        request: BridgeRequest,
    ) -> AsyncIterator[str]:
        """Stream from Claude API."""
        try:
            client = await self._get_client()

            messages = request.messages or []
            if request.prompt and not messages:
                messages = [{"role": "user", "content": request.prompt}]

            system = request.system or ""
            filtered_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system = msg.get("content", "")
                else:
                    filtered_messages.append(msg)

            async with client.messages.stream(
                model=self._config.claude_model,
                max_tokens=request.max_tokens,
                system=system,
                messages=filtered_messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.warning(f"Claude streaming failed: {e}")


# =============================================================================
# CONVERSATION CAPTURE
# =============================================================================


class ConversationCapture:
    """Captures conversations for training."""

    def __init__(self, config: BridgeConfig):
        self._config = config

    async def capture(
        self,
        request: BridgeRequest,
        response: BridgeResponse,
    ) -> None:
        """Capture a conversation for training."""
        if not self._config.capture_interactions:
            return

        # Don't capture API responses if disabled
        if response.source == InferenceSource.CLAUDE_API:
            if not self._config.capture_api_responses:
                return

        try:
            from jarvis_prime.core.training_data_pipeline import get_training_data_pipeline

            pipeline = await get_training_data_pipeline()

            # Format input
            if request.messages:
                user_input = next(
                    (m["content"] for m in reversed(request.messages) if m["role"] == "user"),
                    ""
                )
            else:
                user_input = request.prompt or ""

            # Capture
            await pipeline.capture_conversation(
                user_input=user_input,
                assistant_response=response.text,
                system_prompt=request.system,
                context={
                    "source": response.source.value,
                    "model": response.model_used,
                    "complexity": response.complexity_score,
                    "quality": response.quality.value,
                    "task_type": request.task_type,
                },
            )

        except Exception as e:
            logger.debug(f"Failed to capture conversation: {e}")


# =============================================================================
# RESPONSE ENHANCER
# =============================================================================


class ResponseEnhancer:
    """Post-processes and enhances responses."""

    def __init__(self, config: BridgeConfig):
        self._config = config

    async def enhance(
        self,
        response: BridgeResponse,
        request: BridgeRequest,
    ) -> BridgeResponse:
        """Enhance a response."""
        # Quality assessment
        response.quality = self._assess_quality(response.text, request)

        # Clean up response
        response.text = self._cleanup_text(response.text)

        return response

    def _assess_quality(
        self,
        text: str,
        request: BridgeRequest,
    ) -> ResponseQuality:
        """Assess response quality."""
        if not text or len(text.strip()) < 10:
            return ResponseQuality.FAILED

        # Check for common issues
        issues = 0

        # Too short
        if len(text) < 50 and not any(
            kw in text.lower() for kw in ["yes", "no", "done", "ok", "understood"]
        ):
            issues += 1

        # Repetition
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                issues += 2

        # Incomplete
        if text.endswith("...") or text.endswith(".."):
            issues += 1

        # Map to quality
        if issues == 0:
            return ResponseQuality.EXCELLENT
        elif issues == 1:
            return ResponseQuality.GOOD
        elif issues == 2:
            return ResponseQuality.ACCEPTABLE
        else:
            return ResponseQuality.POOR

    def _cleanup_text(self, text: str) -> str:
        """Clean up response text."""
        # Remove leading/trailing whitespace
        text = text.strip()

        # Remove common prefixes from local models
        prefixes_to_remove = [
            "Assistant:",
            "Response:",
            "Answer:",
            "JARVIS:",
        ]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text


# =============================================================================
# JARVIS-PRIME BRIDGE
# =============================================================================


class JarvisPrimeBridge:
    """Unified bridge for JARVIS to use Prime models with API fallback."""

    def __init__(self, config: Optional[BridgeConfig] = None):
        self._config = config or BridgeConfig()

        self._router = TaskRouter(self._config)
        self._prime_engine = PrimeInferenceEngine(self._config)
        self._claude_engine = ClaudeAPIEngine(self._config)
        self._capture = ConversationCapture(self._config)
        self._enhancer = ResponseEnhancer(self._config)

        self._stats = InferenceStats()
        self._lock = asyncio.Lock()

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        force_local: bool = False,
        force_api: bool = False,
        **kwargs,
    ) -> BridgeResponse:
        """Generate a response (simple interface)."""
        request = BridgeRequest(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            force_local=force_local,
            force_api=force_api,
            context=kwargs,
        )

        return await self._process_request(request)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        force_local: bool = False,
        force_api: bool = False,
        **kwargs,
    ) -> BridgeResponse:
        """Chat completion interface (OpenAI-compatible)."""
        request = BridgeRequest(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            force_local=force_local,
            force_api=force_api,
            context=kwargs,
        )

        return await self._process_request(request)

    async def stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        force_local: bool = False,
        force_api: bool = False,
    ) -> AsyncIterator[str]:
        """Stream a response."""
        request = BridgeRequest(
            prompt=prompt,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            force_local=force_local,
            force_api=force_api,
        )

        # Analyze complexity
        complexity, score = await self._router.analyze_complexity(request)
        model_name = await self._router.select_model(request, complexity)

        # Try local first
        if not force_api and self._config.prefer_local:
            try:
                async for token in self._prime_engine.stream(request, model_name):
                    yield token
                return
            except Exception:
                pass

        # Fall back to API
        if not force_local and self._config.enable_api_fallback:
            async for token in self._claude_engine.stream(request):
                yield token

    async def _process_request(
        self,
        request: BridgeRequest,
    ) -> BridgeResponse:
        """Process a request through the bridge."""
        async with self._lock:
            self._stats.total_requests += 1

        # Analyze complexity
        complexity, score = await self._router.analyze_complexity(request)

        # Update stats
        if complexity == TaskComplexity.SIMPLE:
            self._stats.simple_tasks += 1
        elif complexity == TaskComplexity.MODERATE:
            self._stats.moderate_tasks += 1
        else:
            self._stats.complex_tasks += 1

        # Select model
        model_name = await self._router.select_model(request, complexity)

        response: Optional[BridgeResponse] = None
        fallback_reason: Optional[str] = None

        # Try local first (unless force_api)
        if not request.force_api and self._config.prefer_local:
            self._stats.local_requests += 1

            for attempt in range(self._config.max_local_retries):
                response = await self._prime_engine.generate(request, model_name)

                if response:
                    self._stats.local_success += 1
                    break

            if not response:
                self._stats.local_failure += 1
                fallback_reason = "Local model failed"

        # Fall back to API if needed
        if not response and not request.force_local:
            if self._config.enable_api_fallback:
                self._stats.api_requests += 1
                self._stats.fallback_count += 1

                response = await self._claude_engine.generate(request)

                if response:
                    self._stats.api_success += 1
                    response.fallback_used = True
                    response.fallback_reason = fallback_reason
                else:
                    self._stats.api_failure += 1

        # Handle complete failure
        if not response:
            response = BridgeResponse(
                request_id=request.id,
                text="I apologize, but I'm unable to process your request at this time.",
                source=InferenceSource.LOCAL_PRIME,
                quality=ResponseQuality.FAILED,
            )

        # Add complexity info
        response.complexity_score = score

        # Enhance response
        response = await self._enhancer.enhance(response, request)

        # Update stats
        self._stats.total_tokens += response.tokens_generated
        self._stats.total_latency_ms += response.latency_ms

        # Capture for training (async, non-blocking)
        asyncio.create_task(self._capture.capture(request, response))

        return response

    def get_stats(self) -> InferenceStats:
        """Get inference statistics."""
        return self._stats

    def get_status(self) -> Dict[str, Any]:
        """Get bridge status."""
        return {
            "config": {
                "prefer_local": self._config.prefer_local,
                "default_model": self._config.default_local_model,
                "api_fallback_enabled": self._config.enable_api_fallback,
                "capture_enabled": self._config.capture_interactions,
            },
            "stats": {
                "total_requests": self._stats.total_requests,
                "local_requests": self._stats.local_requests,
                "api_requests": self._stats.api_requests,
                "fallback_rate": f"{self._stats.fallback_rate:.2%}",
                "local_success_rate": f"{self._stats.local_success_rate:.2%}",
                "avg_latency_ms": f"{self._stats.avg_latency_ms:.1f}",
            },
            "complexity_distribution": {
                "simple": self._stats.simple_tasks,
                "moderate": self._stats.moderate_tasks,
                "complex": self._stats.complex_tasks,
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_bridge: Optional[JarvisPrimeBridge] = None
_bridge_lock = asyncio.Lock()


async def get_jarvis_prime_bridge(
    config: Optional[BridgeConfig] = None,
) -> JarvisPrimeBridge:
    """Get or create the global JARVIS-Prime bridge."""
    global _bridge

    async with _bridge_lock:
        if _bridge is None:
            _bridge = JarvisPrimeBridge(config)

        return _bridge


async def shutdown_jarvis_prime_bridge() -> None:
    """Shutdown the global bridge."""
    global _bridge

    async with _bridge_lock:
        if _bridge is not None:
            _bridge = None


# =============================================================================
# CONVENIENCE FUNCTIONS (Drop-in replacements)
# =============================================================================


async def generate(
    prompt: str,
    system: Optional[str] = None,
    **kwargs,
) -> str:
    """Generate a response (simple interface)."""
    bridge = await get_jarvis_prime_bridge()
    response = await bridge.generate(prompt, system, **kwargs)
    return response.text


async def chat(
    messages: List[Dict[str, str]],
    **kwargs,
) -> str:
    """Chat completion interface."""
    bridge = await get_jarvis_prime_bridge()
    response = await bridge.chat(messages, **kwargs)
    return response.text


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Config
    "BridgeConfig",
    # Enums
    "InferenceSource",
    "TaskComplexity",
    "ResponseQuality",
    # Data structures
    "BridgeRequest",
    "BridgeResponse",
    "InferenceStats",
    # Components
    "TaskRouter",
    "PrimeInferenceEngine",
    "ClaudeAPIEngine",
    "ConversationCapture",
    "ResponseEnhancer",
    # Bridge
    "JarvisPrimeBridge",
    # Factory
    "get_jarvis_prime_bridge",
    "shutdown_jarvis_prime_bridge",
    # Convenience
    "generate",
    "chat",
]
