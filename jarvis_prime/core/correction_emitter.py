"""
Correction Emitter v1.0
=======================

Emits user corrections from JARVIS-Prime to Reactor-Core for training.

When a user corrects Prime's response, this data is valuable for fine-tuning.
This module collects corrections and sends them to Reactor-Core's training pipeline.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    CORRECTION FLOW                              │
    │                                                                 │
    │  User says: "No, that's wrong. The answer is X"                 │
    │       ↓                                                         │
    │  JARVIS-Prime detects correction pattern                        │
    │       ↓                                                         │
    │  CorrectionEmitter (this module)                                │
    │       ↓                                                         │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │  Correction Package                                     │    │
    │  │  - Original prompt                                      │    │
    │  │  - Prime's response                                     │    │
    │  │  - User's correction                                    │    │
    │  │  - Context (conversation history)                       │    │
    │  └─────────────────────────────────────────────────────────┘    │
    │       ↓                                                         │
    │  Reactor-Core /api/v1/corrections/stream                        │
    │       ↓                                                         │
    │  Fine-Tuning Dataset                                            │
    │       ↓                                                         │
    │  Improved JARVIS-Prime Model                                    │
    └─────────────────────────────────────────────────────────────────┘

FEATURES:
    - Pattern-based correction detection
    - Batched emission for efficiency
    - Disk-backed queue for persistence
    - Circuit breaker for Reactor-Core failures
    - Quality scoring for corrections

USAGE:
    from jarvis_prime.core.correction_emitter import get_correction_emitter

    emitter = await get_correction_emitter()

    # Record a correction
    await emitter.record_correction(
        original_prompt="What is the capital of Australia?",
        original_response="The capital of Australia is Sydney.",
        corrected_response="The capital of Australia is Canberra.",
        correction_type="factual_error"
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

def _get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class CorrectionEmitterConfig:
    """Configuration for correction emission."""
    # Reactor-Core connection
    reactor_host: str = field(default_factory=lambda: os.getenv("REACTOR_CORE_HOST", "localhost"))
    reactor_port: int = field(default_factory=lambda: _get_env_int("REACTOR_CORE_PORT", 8003))

    # Batching
    batch_size: int = field(default_factory=lambda: _get_env_int("CORRECTION_BATCH_SIZE", 5))
    batch_timeout: float = field(default_factory=lambda: _get_env_float("CORRECTION_BATCH_TIMEOUT", 60.0))

    # Retry
    max_retries: int = field(default_factory=lambda: _get_env_int("CORRECTION_MAX_RETRIES", 3))
    retry_base_delay: float = field(default_factory=lambda: _get_env_float("CORRECTION_RETRY_DELAY", 1.0))

    # Circuit breaker
    circuit_failure_threshold: int = field(default_factory=lambda: _get_env_int("CORRECTION_CIRCUIT_THRESHOLD", 5))
    circuit_reset_timeout: float = field(default_factory=lambda: _get_env_float("CORRECTION_CIRCUIT_RESET", 60.0))

    # Disk queue
    queue_dir: str = field(default_factory=lambda: os.getenv(
        "CORRECTION_QUEUE_DIR",
        os.path.join(os.path.expanduser("~"), ".jarvis", "correction_queue")
    ))

    @property
    def corrections_url(self) -> str:
        return f"http://{self.reactor_host}:{self.reactor_port}/api/v1/corrections/stream"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CorrectionType(Enum):
    """Types of corrections."""
    FACTUAL_ERROR = "factual_error"       # Wrong facts
    TONE_ERROR = "tone_error"             # Wrong tone/style
    INCOMPLETE = "incomplete"             # Missing information
    INCORRECT_FORMAT = "incorrect_format" # Wrong format
    MISUNDERSTOOD = "misunderstood"       # Misunderstood intent
    OTHER = "other"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class Correction:
    """A single correction record."""
    correction_id: str
    original_prompt: str
    original_response: str
    corrected_response: str
    correction_type: CorrectionType
    timestamp: float
    context: Optional[List[Dict[str, str]]] = None
    user_feedback: Optional[str] = None
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correction_id": self.correction_id,
            "original_prompt": self.original_prompt,
            "original_response": self.original_response,
            "corrected_response": self.corrected_response,
            "correction_type": self.correction_type.value,
            "timestamp": self.timestamp,
            "context": self.context,
            "user_feedback": self.user_feedback,
            "quality_score": self.quality_score,
            "metadata": self.metadata,
        }


# =============================================================================
# CORRECTION DETECTION
# =============================================================================

class CorrectionDetector:
    """
    Detects when a user is correcting Prime's response.

    Patterns:
    - "No, that's wrong..."
    - "Actually, it's..."
    - "You're incorrect..."
    - "The correct answer is..."
    """

    # Correction patterns (regex)
    CORRECTION_PATTERNS = [
        r"^no[,.]?\s*(that'?s|it'?s|you'?re)\s+(wrong|incorrect|not right)",
        r"^actually[,.]?\s+(it'?s|the|that)",
        r"^(you'?re|that'?s)\s+(wrong|incorrect|mistaken)",
        r"^the\s+(correct|right|actual)\s+(answer|response|way)",
        r"^that'?s\s+not\s+(right|correct|true|accurate)",
        r"^i\s+(meant|mean|was asking|wanted)",
        r"^not\s+quite[,.]",
        r"^close[,.]?\s+but",
        r"^almost[,.]?\s+(but|however)",
    ]

    def __init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.CORRECTION_PATTERNS]

    def is_correction(self, user_message: str) -> bool:
        """Check if user message is a correction."""
        message_clean = user_message.strip()
        for pattern in self._patterns:
            if pattern.search(message_clean):
                return True
        return False

    def extract_correction_type(self, user_message: str) -> CorrectionType:
        """Attempt to classify the correction type."""
        message_lower = user_message.lower()

        if any(word in message_lower for word in ["fact", "wrong", "incorrect", "false", "true"]):
            return CorrectionType.FACTUAL_ERROR
        if any(word in message_lower for word in ["tone", "rude", "polite", "friendly", "formal"]):
            return CorrectionType.TONE_ERROR
        if any(word in message_lower for word in ["more", "also", "missing", "forgot", "include"]):
            return CorrectionType.INCOMPLETE
        if any(word in message_lower for word in ["format", "list", "bullet", "table", "code"]):
            return CorrectionType.INCORRECT_FORMAT
        if any(word in message_lower for word in ["meant", "asking", "wanted", "trying"]):
            return CorrectionType.MISUNDERSTOOD

        return CorrectionType.OTHER


# =============================================================================
# CORRECTION EMITTER
# =============================================================================

class CorrectionEmitter:
    """
    Emits corrections to Reactor-Core for training.
    """

    def __init__(self, config: Optional[CorrectionEmitterConfig] = None):
        self._config = config or CorrectionEmitterConfig()
        self._detector = CorrectionDetector()
        self._queue: List[Correction] = []
        self._queue_lock = asyncio.Lock()
        self._circuit_state = CircuitState.CLOSED
        self._circuit_failures = 0
        self._last_circuit_failure = 0.0
        self._flush_task: Optional[asyncio.Task] = None
        self._initialized = False
        self._shutdown = False
        self._http_client = None

        # Stats
        self._corrections_recorded = 0
        self._corrections_emitted = 0
        self._emissions_failed = 0

    async def initialize(self) -> None:
        """Initialize the emitter."""
        if self._initialized:
            return

        # Ensure queue directory exists
        Path(self._config.queue_dir).mkdir(parents=True, exist_ok=True)

        # Load persisted corrections
        await self._load_persisted_corrections()

        # Initialize HTTP client
        try:
            import aiohttp
            self._http_client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"Content-Type": "application/json"},
            )
        except ImportError:
            logger.warning("[CorrectionEmitter] aiohttp not available")

        # Start flush task
        self._flush_task = asyncio.create_task(self._flush_loop())

        self._initialized = True
        logger.info("[CorrectionEmitter] Initialized")

    async def close(self) -> None:
        """Shutdown the emitter."""
        self._shutdown = True

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining
        await self._flush_queue()

        # Persist any remaining
        await self._persist_remaining_corrections()

        if self._http_client:
            await self._http_client.close()

        logger.info(
            f"[CorrectionEmitter] Closed. "
            f"Recorded: {self._corrections_recorded}, "
            f"Emitted: {self._corrections_emitted}, "
            f"Failed: {self._emissions_failed}"
        )

    async def record_correction(
        self,
        original_prompt: str,
        original_response: str,
        corrected_response: str,
        correction_type: Optional[CorrectionType] = None,
        context: Optional[List[Dict[str, str]]] = None,
        user_feedback: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a correction for training.

        Args:
            original_prompt: The original user prompt
            original_response: Prime's original response
            corrected_response: The corrected/desired response
            correction_type: Type of correction (auto-detected if not provided)
            context: Conversation history
            user_feedback: User's explanation of the correction
            metadata: Additional metadata

        Returns:
            Correction ID
        """
        if not self._initialized:
            await self.initialize()

        # Auto-detect correction type if not provided
        if correction_type is None:
            correction_type = self._detector.extract_correction_type(corrected_response)

        # Calculate quality score based on response lengths
        quality_score = self._calculate_quality_score(
            original_response, corrected_response, user_feedback
        )

        correction = Correction(
            correction_id=str(uuid.uuid4()),
            original_prompt=original_prompt,
            original_response=original_response,
            corrected_response=corrected_response,
            correction_type=correction_type,
            timestamp=time.time(),
            context=context,
            user_feedback=user_feedback,
            quality_score=quality_score,
            metadata=metadata or {},
        )

        async with self._queue_lock:
            self._queue.append(correction)
            self._corrections_recorded += 1

        logger.info(
            f"[CorrectionEmitter] Recorded correction {correction.correction_id[:8]} "
            f"(type: {correction_type.value}, quality: {quality_score:.2f})"
        )

        # Flush if batch is full
        if len(self._queue) >= self._config.batch_size:
            asyncio.create_task(self._flush_queue())

        return correction.correction_id

    def detect_and_record(
        self,
        user_message: str,
        previous_prompt: str,
        previous_response: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        """
        Detect if user message is a correction and record it.

        This is a convenience method for automatic correction detection.

        Returns:
            Correction ID if correction was detected and recorded, None otherwise
        """
        if not self._detector.is_correction(user_message):
            return None

        # User message is likely a correction
        asyncio.create_task(
            self.record_correction(
                original_prompt=previous_prompt,
                original_response=previous_response,
                corrected_response=user_message,
                context=context,
                user_feedback=user_message,
            )
        )

        return str(uuid.uuid4())[:8]  # Return partial ID

    def _calculate_quality_score(
        self,
        original: str,
        corrected: str,
        feedback: Optional[str],
    ) -> float:
        """Calculate quality score for a correction."""
        score = 1.0

        # Longer corrections are generally more informative
        if len(corrected) > len(original) * 2:
            score += 0.1
        elif len(corrected) < len(original) * 0.5:
            score -= 0.1

        # Having user feedback is valuable
        if feedback:
            score += 0.2

        # Cap score
        return max(0.1, min(1.5, score))

    async def _flush_loop(self) -> None:
        """Background loop to flush corrections."""
        while not self._shutdown:
            try:
                await asyncio.sleep(self._config.batch_timeout)
                await self._flush_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CorrectionEmitter] Flush loop error: {e}")

    async def _flush_queue(self) -> None:
        """Flush corrections to Reactor-Core."""
        async with self._queue_lock:
            if not self._queue:
                return

            corrections = self._queue[:self._config.batch_size]
            self._queue = self._queue[self._config.batch_size:]

        # Check circuit breaker
        if self._circuit_state == CircuitState.OPEN:
            if time.time() - self._last_circuit_failure >= self._config.circuit_reset_timeout:
                self._circuit_state = CircuitState.HALF_OPEN
            else:
                # Persist for later
                for c in corrections:
                    await self._persist_correction(c)
                return

        # Emit to Reactor-Core
        success = await self._emit_batch(corrections)

        if success:
            self._corrections_emitted += len(corrections)
            if self._circuit_state == CircuitState.HALF_OPEN:
                self._circuit_state = CircuitState.CLOSED
                self._circuit_failures = 0
        else:
            self._emissions_failed += len(corrections)
            self._circuit_failures += 1
            self._last_circuit_failure = time.time()

            if self._circuit_failures >= self._config.circuit_failure_threshold:
                self._circuit_state = CircuitState.OPEN
                logger.warning("[CorrectionEmitter] Circuit breaker OPEN")

            # Persist for retry
            for c in corrections:
                await self._persist_correction(c)

    async def _emit_batch(self, corrections: List[Correction]) -> bool:
        """Emit a batch of corrections to Reactor-Core."""
        if not self._http_client:
            return False

        payload = {
            "corrections": [c.to_dict() for c in corrections],
            "source": "jarvis_prime",
            "timestamp": time.time(),
        }

        for attempt in range(self._config.max_retries):
            try:
                async with self._http_client.post(
                    self._config.corrections_url,
                    json=payload,
                ) as resp:
                    if resp.status == 200:
                        logger.info(f"[CorrectionEmitter] Emitted {len(corrections)} corrections")
                        return True
                    else:
                        logger.warning(f"[CorrectionEmitter] Reactor-Core returned {resp.status}")

            except Exception as e:
                logger.warning(f"[CorrectionEmitter] Attempt {attempt + 1} failed: {e}")

            if attempt < self._config.max_retries - 1:
                await asyncio.sleep(self._config.retry_base_delay * (2 ** attempt))

        return False

    async def _persist_correction(self, correction: Correction) -> None:
        """Persist correction to disk for later retry."""
        try:
            path = Path(self._config.queue_dir) / f"{correction.correction_id}.json"
            path.write_text(json.dumps(correction.to_dict()))
        except Exception as e:
            logger.warning(f"[CorrectionEmitter] Failed to persist correction: {e}")

    async def _load_persisted_corrections(self) -> None:
        """Load persisted corrections from disk."""
        try:
            queue_dir = Path(self._config.queue_dir)
            if not queue_dir.exists():
                return

            files = list(queue_dir.glob("*.json"))
            for file_path in files[:100]:  # Limit to 100
                try:
                    data = json.loads(file_path.read_text())
                    correction = Correction(
                        correction_id=data["correction_id"],
                        original_prompt=data["original_prompt"],
                        original_response=data["original_response"],
                        corrected_response=data["corrected_response"],
                        correction_type=CorrectionType(data["correction_type"]),
                        timestamp=data["timestamp"],
                        context=data.get("context"),
                        user_feedback=data.get("user_feedback"),
                        quality_score=data.get("quality_score", 1.0),
                        metadata=data.get("metadata", {}),
                    )
                    self._queue.append(correction)
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"[CorrectionEmitter] Failed to load {file_path}: {e}")

            if self._queue:
                logger.info(f"[CorrectionEmitter] Loaded {len(self._queue)} persisted corrections")

        except Exception as e:
            logger.warning(f"[CorrectionEmitter] Failed to load persisted corrections: {e}")

    async def _persist_remaining_corrections(self) -> None:
        """Persist remaining corrections on shutdown."""
        for correction in self._queue:
            await self._persist_correction(correction)

    def get_stats(self) -> Dict[str, Any]:
        """Get emitter statistics."""
        return {
            "corrections_recorded": self._corrections_recorded,
            "corrections_emitted": self._corrections_emitted,
            "emissions_failed": self._emissions_failed,
            "queue_size": len(self._queue),
            "circuit_state": self._circuit_state.value,
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_correction_emitter: Optional[CorrectionEmitter] = None
_emitter_lock = asyncio.Lock()


async def get_correction_emitter(config: Optional[CorrectionEmitterConfig] = None) -> CorrectionEmitter:
    """Get the singleton CorrectionEmitter instance."""
    global _correction_emitter

    if _correction_emitter is not None and _correction_emitter._initialized:
        return _correction_emitter

    async with _emitter_lock:
        if _correction_emitter is not None and _correction_emitter._initialized:
            return _correction_emitter

        _correction_emitter = CorrectionEmitter(config)
        await _correction_emitter.initialize()
        return _correction_emitter


async def close_correction_emitter() -> None:
    """Close the singleton emitter."""
    global _correction_emitter

    if _correction_emitter:
        await _correction_emitter.close()
        _correction_emitter = None
