"""
Telemetry Hook - Logging with PII Anonymization
================================================

Logs every prompt and completion for reactor-core training data collection,
with automatic PII detection and anonymization.

Features:
- Async, non-blocking logging
- Automatic PII detection and redaction
- Structured JSONL output for training pipelines
- Configurable anonymization patterns
- Reactor-core compatible format
- Batch writing for performance
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII that can be detected and anonymized"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    API_KEY = "api_key"
    PASSWORD = "password"
    URL_WITH_PARAMS = "url_with_params"
    DATE_OF_BIRTH = "dob"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """A detected PII match"""
    pii_type: PIIType
    original: str
    anonymized: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class AnonymizationResult:
    """Result of anonymization process"""
    original_text: str
    anonymized_text: str
    matches: List[PIIMatch] = field(default_factory=list)
    pii_detected: bool = False

    @property
    def pii_types_found(self) -> Set[PIIType]:
        return {m.pii_type for m in self.matches}


class PIIAnonymizer:
    """
    Detects and anonymizes PII in text.

    Uses regex patterns for detection and consistent hashing
    for anonymization (same input → same output for consistency).

    Usage:
        anonymizer = PIIAnonymizer()
        result = anonymizer.anonymize("Contact me at john@example.com")
        # result.anonymized_text = "Contact me at [EMAIL:a1b2c3]"
    """

    # Default PII patterns
    DEFAULT_PATTERNS: Dict[PIIType, List[Pattern]] = {}

    def __init__(
        self,
        custom_patterns: Optional[Dict[PIIType, List[str]]] = None,
        salt: Optional[str] = None,
        redaction_format: str = "[{type}:{hash}]",
    ):
        self.salt = salt or os.environ.get("PII_SALT", "jarvis-prime-default-salt")
        self.redaction_format = redaction_format

        # Compile default patterns
        self._patterns: Dict[PIIType, List[Pattern]] = {}
        self._compile_default_patterns()

        # Add custom patterns
        if custom_patterns:
            for pii_type, patterns in custom_patterns.items():
                if pii_type not in self._patterns:
                    self._patterns[pii_type] = []
                for pattern in patterns:
                    self._patterns[pii_type].append(re.compile(pattern, re.IGNORECASE))

    def _compile_default_patterns(self) -> None:
        """Compile default PII detection patterns"""
        patterns = {
            PIIType.EMAIL: [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            ],
            PIIType.PHONE: [
                r'\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b',
                r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',
            ],
            PIIType.SSN: [
                r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
            ],
            PIIType.CREDIT_CARD: [
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            ],
            PIIType.IP_ADDRESS: [
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
            ],
            PIIType.API_KEY: [
                r'\b(?:sk-|api[_-]?key[_-]?|token[_-]?|secret[_-]?)[a-zA-Z0-9]{20,}\b',
                r'\bsk-[a-zA-Z0-9]{48}\b',  # OpenAI key format
                r'\bAIza[a-zA-Z0-9_-]{35}\b',  # Google API key
                r'\bghp_[a-zA-Z0-9]{36}\b',  # GitHub PAT
            ],
            PIIType.PASSWORD: [
                r'(?:password|passwd|pwd)[\s:=]+[^\s]{4,}',
            ],
            PIIType.URL_WITH_PARAMS: [
                r'https?://[^\s]+\?[^\s]*(?:key|token|password|secret|auth)[^\s]*',
            ],
            PIIType.DATE_OF_BIRTH: [
                r'\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b',
                r'\b(?:19|20)\d{2}[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])\b',
            ],
        }

        for pii_type, pattern_strs in patterns.items():
            self._patterns[pii_type] = [
                re.compile(p, re.IGNORECASE) for p in pattern_strs
            ]

    def _generate_hash(self, text: str, pii_type: PIIType) -> str:
        """Generate consistent hash for anonymization"""
        salted = f"{self.salt}:{pii_type.value}:{text}"
        return hashlib.sha256(salted.encode()).hexdigest()[:8]

    def anonymize(self, text: str) -> AnonymizationResult:
        """
        Anonymize PII in text.

        Args:
            text: Input text to anonymize

        Returns:
            AnonymizationResult with anonymized text and match details
        """
        if not text:
            return AnonymizationResult(original_text=text, anonymized_text=text)

        matches: List[PIIMatch] = []

        # Find all matches
        for pii_type, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    original = match.group()
                    hash_value = self._generate_hash(original, pii_type)
                    anonymized = self.redaction_format.format(
                        type=pii_type.value.upper(),
                        hash=hash_value,
                    )

                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        original=original,
                        anonymized=anonymized,
                        start=match.start(),
                        end=match.end(),
                    ))

        if not matches:
            return AnonymizationResult(
                original_text=text,
                anonymized_text=text,
                pii_detected=False,
            )

        # Sort by position (reverse) for replacement
        matches.sort(key=lambda m: m.start, reverse=True)

        # Remove overlapping matches (keep longest)
        filtered_matches = []
        for match in matches:
            overlaps = False
            for existing in filtered_matches:
                if (match.start < existing.end and match.end > existing.start):
                    overlaps = True
                    break
            if not overlaps:
                filtered_matches.append(match)

        # Apply replacements
        anonymized = text
        for match in filtered_matches:
            anonymized = anonymized[:match.start] + match.anonymized + anonymized[match.end:]

        return AnonymizationResult(
            original_text=text,
            anonymized_text=anonymized,
            matches=filtered_matches,
            pii_detected=True,
        )


@dataclass
class TelemetryRecord:
    """A single telemetry record for logging"""
    # Identification
    record_id: str
    timestamp: datetime
    session_id: Optional[str] = None

    # Request
    prompt: str = ""
    prompt_anonymized: str = ""
    prompt_tokens: int = 0

    # Response
    completion: str = ""
    completion_anonymized: str = ""
    completion_tokens: int = 0

    # Routing
    tier: str = "tier_0"
    model_version: str = ""
    task_type: str = ""
    complexity_score: float = 0.0

    # Performance
    latency_ms: float = 0.0
    queue_time_ms: float = 0.0
    generation_time_ms: float = 0.0

    # Outcome
    success: bool = True
    error_message: Optional[str] = None
    user_feedback: Optional[str] = None

    # PII tracking
    pii_detected_prompt: bool = False
    pii_detected_completion: bool = False
    pii_types_found: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "prompt": self.prompt_anonymized,  # Always use anonymized
            "prompt_tokens": self.prompt_tokens,
            "completion": self.completion_anonymized,  # Always use anonymized
            "completion_tokens": self.completion_tokens,
            "tier": self.tier,
            "model_version": self.model_version,
            "task_type": self.task_type,
            "complexity_score": round(self.complexity_score, 3),
            "latency_ms": round(self.latency_ms, 1),
            "queue_time_ms": round(self.queue_time_ms, 1),
            "generation_time_ms": round(self.generation_time_ms, 1),
            "success": self.success,
            "error_message": self.error_message,
            "user_feedback": self.user_feedback,
            "pii_detected": self.pii_detected_prompt or self.pii_detected_completion,
            "pii_types": self.pii_types_found,
            "metadata": self.metadata,
        }

    def to_training_format(self) -> Dict[str, Any]:
        """Convert to format suitable for reactor-core training"""
        return {
            "id": self.record_id,
            "messages": [
                {"role": "user", "content": self.prompt_anonymized},
                {"role": "assistant", "content": self.completion_anonymized},
            ],
            "metadata": {
                "tier": self.tier,
                "model_version": self.model_version,
                "task_type": self.task_type,
                "complexity": self.complexity_score,
                "success": self.success,
                "feedback": self.user_feedback,
            },
        }


class TelemetryHook:
    """
    Async telemetry logging with PII anonymization.

    Logs all prompts and completions to JSONL files that reactor-core
    can read for training data collection.

    Usage:
        hook = TelemetryHook(output_dir="./telemetry")
        await hook.start()

        # Log a request/response
        record = await hook.log(
            prompt="Hello, my email is john@example.com",
            completion="Hi John! How can I help?",
            model_version="v1.0",
            latency_ms=150.0,
        )

        await hook.stop()
    """

    def __init__(
        self,
        output_dir: str | Path = "./telemetry",
        anonymizer: Optional[PIIAnonymizer] = None,
        batch_size: int = 100,
        flush_interval: float = 30.0,  # seconds
        enable_training_format: bool = True,
        max_file_size_mb: float = 100.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.anonymizer = anonymizer or PIIAnonymizer()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.enable_training_format = enable_training_format
        self.max_file_size_mb = max_file_size_mb

        # State
        self._buffer: List[TelemetryRecord] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False

        # Files
        self._current_file: Optional[Path] = None
        self._training_file: Optional[Path] = None
        self._file_record_count = 0

        # Statistics
        self._total_records = 0
        self._total_pii_detected = 0
        self._total_tier_0 = 0
        self._total_tier_1 = 0

        logger.info(f"TelemetryHook initialized: {self.output_dir}")

    async def start(self) -> None:
        """Start the telemetry hook"""
        self._running = True
        self._rotate_files()
        self._flush_task = asyncio.create_task(self._periodic_flush())
        logger.info("TelemetryHook started")

    async def stop(self) -> None:
        """Stop the telemetry hook and flush remaining records"""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self._flush()
        logger.info(f"TelemetryHook stopped. Total records: {self._total_records}")

    def _rotate_files(self) -> None:
        """Create new output files with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_file = self.output_dir / f"telemetry_{timestamp}.jsonl"
        self._training_file = self.output_dir / f"training_{timestamp}.jsonl"
        self._file_record_count = 0

    def _check_file_rotation(self) -> None:
        """Rotate files if size limit exceeded"""
        if self._current_file and self._current_file.exists():
            size_mb = self._current_file.stat().st_size / 1024 / 1024
            if size_mb >= self.max_file_size_mb:
                self._rotate_files()

    async def log(
        self,
        prompt: str,
        completion: str,
        model_version: str = "",
        tier: str = "tier_0",
        task_type: str = "",
        complexity_score: float = 0.0,
        latency_ms: float = 0.0,
        queue_time_ms: float = 0.0,
        generation_time_ms: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TelemetryRecord:
        """
        Log a prompt/completion pair.

        Automatically anonymizes PII and queues for batch writing.

        Args:
            prompt: User prompt
            completion: Model completion
            model_version: Model version ID
            tier: Routing tier used
            task_type: Detected task type
            complexity_score: Complexity score (0-1)
            latency_ms: Total latency
            queue_time_ms: Time spent in queue
            generation_time_ms: Time for generation
            success: Whether request succeeded
            error_message: Error message if failed
            session_id: Optional session identifier
            metadata: Additional metadata

        Returns:
            The created TelemetryRecord
        """
        # Generate record ID
        record_id = hashlib.md5(
            f"{time.time()}:{prompt[:50]}".encode()
        ).hexdigest()[:16]

        # Anonymize prompt and completion
        prompt_result = self.anonymizer.anonymize(prompt)
        completion_result = self.anonymizer.anonymize(completion)

        # Collect PII types found
        pii_types = list({
            t.value for t in
            prompt_result.pii_types_found | completion_result.pii_types_found
        })

        # Estimate token counts (rough approximation)
        prompt_tokens = len(prompt.split()) + len(prompt) // 4
        completion_tokens = len(completion.split()) + len(completion) // 4

        # Create record
        record = TelemetryRecord(
            record_id=record_id,
            timestamp=datetime.now(),
            session_id=session_id,
            prompt=prompt,
            prompt_anonymized=prompt_result.anonymized_text,
            prompt_tokens=prompt_tokens,
            completion=completion,
            completion_anonymized=completion_result.anonymized_text,
            completion_tokens=completion_tokens,
            tier=tier,
            model_version=model_version,
            task_type=task_type,
            complexity_score=complexity_score,
            latency_ms=latency_ms,
            queue_time_ms=queue_time_ms,
            generation_time_ms=generation_time_ms,
            success=success,
            error_message=error_message,
            pii_detected_prompt=prompt_result.pii_detected,
            pii_detected_completion=completion_result.pii_detected,
            pii_types_found=pii_types,
            metadata=metadata or {},
        )

        # Update statistics
        self._total_records += 1
        if record.pii_detected_prompt or record.pii_detected_completion:
            self._total_pii_detected += 1
        if tier == "tier_0":
            self._total_tier_0 += 1
        else:
            self._total_tier_1 += 1

        # Add to buffer
        async with self._buffer_lock:
            self._buffer.append(record)

            if len(self._buffer) >= self.batch_size:
                await self._flush_internal()

        return record

    async def record_feedback(
        self,
        record_id: str,
        feedback: str,
    ) -> bool:
        """
        Record user feedback for a previous record.

        Feedback is written to a separate file for reactor-core RLHF.
        """
        feedback_file = self.output_dir / "feedback.jsonl"

        try:
            feedback_record = {
                "record_id": record_id,
                "feedback": feedback,
                "timestamp": datetime.now().isoformat(),
            }

            with open(feedback_file, "a") as f:
                f.write(json.dumps(feedback_record) + "\n")

            return True

        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False

    async def _periodic_flush(self) -> None:
        """Periodically flush buffer to disk"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic flush error: {e}")

    async def _flush(self) -> None:
        """Flush buffer to disk"""
        async with self._buffer_lock:
            await self._flush_internal()

    async def _flush_internal(self) -> None:
        """Internal flush (must hold lock)"""
        if not self._buffer:
            return

        self._check_file_rotation()

        records_to_write = self._buffer
        self._buffer = []

        try:
            # Write telemetry records
            with open(self._current_file, "a") as f:
                for record in records_to_write:
                    f.write(json.dumps(record.to_dict()) + "\n")

            # Write training format if enabled
            if self.enable_training_format and self._training_file:
                with open(self._training_file, "a") as f:
                    for record in records_to_write:
                        if record.success:  # Only include successful completions
                            f.write(json.dumps(record.to_training_format()) + "\n")

            self._file_record_count += len(records_to_write)
            logger.debug(f"Flushed {len(records_to_write)} records to {self._current_file}")

        except Exception as e:
            logger.error(f"Failed to flush records: {e}")
            # Put records back in buffer
            self._buffer = records_to_write + self._buffer

    def get_statistics(self) -> Dict[str, Any]:
        """Get telemetry statistics"""
        return {
            "total_records": self._total_records,
            "total_pii_detected": self._total_pii_detected,
            "pii_detection_rate": self._total_pii_detected / max(self._total_records, 1),
            "tier_0_count": self._total_tier_0,
            "tier_1_count": self._total_tier_1,
            "tier_0_ratio": self._total_tier_0 / max(self._total_records, 1),
            "buffer_size": len(self._buffer),
            "current_file": str(self._current_file) if self._current_file else None,
            "file_record_count": self._file_record_count,
        }

    def get_reactor_core_path(self) -> Path:
        """
        Get the path to the training data directory.

        This is the standardized path that reactor-core reads from.
        """
        return self.output_dir
