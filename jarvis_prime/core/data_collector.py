"""
JARVIS Data Collector - Continuous Learning Pipeline
======================================================

v78.0 - Automated data collection from JARVIS interactions

Collects, processes, and stores training data from:
- User commands and responses
- Action execution results
- Reasoning traces
- Feedback signals

PIPELINE:
    User Interaction → Collection → Processing → Storage → Training

FEATURES:
    - Real-time collection from JARVIS
    - Automatic labeling and categorization
    - Privacy-preserving anonymization
    - Quality filtering
    - Deduplication
    - Export for fine-tuning
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
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
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA TYPES
# =============================================================================


class DataCategory(Enum):
    """Categories of collected data."""

    COMMAND = auto()          # User commands
    RESPONSE = auto()         # Model responses
    ACTION = auto()           # Executed actions
    REASONING = auto()        # Reasoning traces
    FEEDBACK = auto()         # User feedback
    SCREEN = auto()           # Screen understanding
    ERROR = auto()            # Error cases
    CORRECTION = auto()       # User corrections


class DataQuality(Enum):
    """Quality levels for collected data."""

    HIGH = "high"             # Confirmed correct
    MEDIUM = "medium"         # Likely correct
    LOW = "low"               # Uncertain
    REJECTED = "rejected"     # Known incorrect


@dataclass
class DataSample:
    """Single data sample for training."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: DataCategory = DataCategory.COMMAND
    quality: DataQuality = DataQuality.MEDIUM

    # Content
    input_text: str = ""
    output_text: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # Anonymized

    # Labels
    labels: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    # Signals
    feedback_score: Optional[float] = None  # -1 to 1
    was_corrected: bool = False
    correction_text: Optional[str] = None

    # Processing
    processed: bool = False
    exported: bool = False
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for deduplication."""
        content = f"{self.input_text}|{self.output_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.name,
            "quality": self.quality.value,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "labels": self.labels,
            "tags": self.tags,
            "feedback_score": self.feedback_score,
            "was_corrected": self.was_corrected,
            "correction_text": self.correction_text,
            "processed": self.processed,
            "exported": self.exported,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSample":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            category=DataCategory[data.get("category", "COMMAND")],
            quality=DataQuality(data.get("quality", "medium")),
            input_text=data.get("input_text", ""),
            output_text=data.get("output_text", ""),
            context=data.get("context", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            labels=data.get("labels", []),
            tags=data.get("tags", []),
            feedback_score=data.get("feedback_score"),
            was_corrected=data.get("was_corrected", False),
            correction_text=data.get("correction_text"),
            processed=data.get("processed", False),
            exported=data.get("exported", False),
            content_hash=data.get("content_hash", ""),
        )

    def to_training_format(self) -> Dict[str, str]:
        """Convert to training format (prompt/completion)."""
        # Use correction if available
        output = self.correction_text if self.was_corrected else self.output_text

        return {
            "prompt": self.input_text,
            "completion": output,
        }


@dataclass
class CollectorConfig:
    """Configuration for data collector."""

    # Storage
    storage_dir: Path = field(
        default_factory=lambda: Path.home() / ".jarvis" / "prime" / "data"
    )
    max_samples: int = 100000
    auto_cleanup: bool = True

    # Collection
    collect_commands: bool = True
    collect_responses: bool = True
    collect_actions: bool = True
    collect_reasoning: bool = True
    collect_feedback: bool = True

    # Filtering
    min_input_length: int = 5
    max_input_length: int = 10000
    min_output_length: int = 1
    max_output_length: int = 50000

    # Privacy
    anonymize_user_ids: bool = True
    redact_pii: bool = True
    pii_patterns: List[str] = field(default_factory=lambda: [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
        r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",  # SSN
    ])

    # Quality
    auto_label: bool = True
    require_feedback_for_high_quality: bool = True


# =============================================================================
# PII REDACTOR
# =============================================================================


class PIIRedactor:
    """Redacts personally identifiable information."""

    def __init__(self, patterns: List[str]) -> None:
        self._patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def redact(self, text: str) -> str:
        """Redact PII from text."""
        result = text
        for pattern in self._patterns:
            result = pattern.sub("[REDACTED]", result)
        return result


# =============================================================================
# AUTO LABELER
# =============================================================================


class AutoLabeler:
    """Automatically labels data samples."""

    # Category keywords
    CATEGORY_KEYWORDS = {
        "code": ["code", "function", "class", "def", "import", "```"],
        "question": ["what", "why", "how", "when", "where", "?"],
        "action": ["click", "type", "open", "close", "move", "execute"],
        "creative": ["write", "create", "compose", "generate", "story"],
        "analysis": ["analyze", "compare", "evaluate", "explain"],
    }

    def label(self, sample: DataSample) -> List[str]:
        """Generate labels for a sample."""
        labels = []
        text = f"{sample.input_text} {sample.output_text}".lower()

        for label, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                labels.append(label)

        # Add length-based labels
        if len(sample.input_text) < 50:
            labels.append("short_input")
        elif len(sample.input_text) > 500:
            labels.append("long_input")

        # Add quality-based labels
        if sample.feedback_score is not None:
            if sample.feedback_score > 0.5:
                labels.append("positive_feedback")
            elif sample.feedback_score < -0.5:
                labels.append("negative_feedback")

        return labels


# =============================================================================
# DATA COLLECTOR
# =============================================================================


class JARVISDataCollector:
    """
    Collects training data from JARVIS interactions.

    Integrates with JARVIS Prime Bridge and Learning Engine
    to capture high-quality training data.

    Usage:
        collector = JARVISDataCollector()
        await collector.initialize()

        # Collect command-response pair
        await collector.collect(
            input_text="Open Safari",
            output_text="Opening Safari browser...",
            category=DataCategory.COMMAND,
        )

        # Add feedback
        await collector.add_feedback(sample_id, score=0.8)

        # Export for training
        samples = await collector.export_for_training(min_quality=DataQuality.MEDIUM)
    """

    def __init__(self, config: Optional[CollectorConfig] = None) -> None:
        self._config = config or CollectorConfig()
        self._samples: Dict[str, DataSample] = {}
        self._hash_index: Set[str] = set()  # For deduplication

        # Components
        self._redactor = PIIRedactor(self._config.pii_patterns)
        self._labeler = AutoLabeler()

        # State
        self._initialized = False
        self._lock = asyncio.Lock()

        # Metrics
        self._collected_count = 0
        self._rejected_count = 0
        self._exported_count = 0

    async def initialize(self) -> bool:
        """Initialize collector."""
        try:
            # Create storage directory
            self._config.storage_dir.mkdir(parents=True, exist_ok=True)

            # Load existing samples
            await self._load_samples()

            self._initialized = True
            logger.info(f"Data collector initialized with {len(self._samples)} samples")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize collector: {e}")
            return False

    async def collect(
        self,
        input_text: str,
        output_text: str,
        category: DataCategory = DataCategory.COMMAND,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Collect a data sample.

        Returns sample ID or None if rejected.
        """
        async with self._lock:
            # Apply filters
            if not self._passes_filters(input_text, output_text):
                self._rejected_count += 1
                return None

            # Apply privacy
            if self._config.redact_pii:
                input_text = self._redactor.redact(input_text)
                output_text = self._redactor.redact(output_text)

            if self._config.anonymize_user_ids and user_id:
                user_id = hashlib.sha256(user_id.encode()).hexdigest()[:8]

            # Create sample
            sample = DataSample(
                category=category,
                input_text=input_text,
                output_text=output_text,
                context=context or {},
                session_id=session_id,
                user_id=user_id,
                tags=tags or [],
            )

            # Check for duplicates
            if sample.content_hash in self._hash_index:
                logger.debug(f"Duplicate sample rejected: {sample.content_hash}")
                self._rejected_count += 1
                return None

            # Auto-label
            if self._config.auto_label:
                sample.labels = self._labeler.label(sample)

            # Store
            self._samples[sample.id] = sample
            self._hash_index.add(sample.content_hash)
            self._collected_count += 1

            # Auto-cleanup if needed
            if self._config.auto_cleanup and len(self._samples) > self._config.max_samples:
                await self._cleanup_old_samples()

            return sample.id

    def _passes_filters(self, input_text: str, output_text: str) -> bool:
        """Check if sample passes quality filters."""
        if len(input_text) < self._config.min_input_length:
            return False
        if len(input_text) > self._config.max_input_length:
            return False
        if len(output_text) < self._config.min_output_length:
            return False
        if len(output_text) > self._config.max_output_length:
            return False
        return True

    async def add_feedback(
        self,
        sample_id: str,
        score: float,
        correction: Optional[str] = None,
    ) -> bool:
        """Add feedback to a sample."""
        async with self._lock:
            if sample_id not in self._samples:
                return False

            sample = self._samples[sample_id]
            sample.feedback_score = max(-1, min(1, score))

            if correction:
                sample.was_corrected = True
                sample.correction_text = correction

            # Update quality based on feedback
            if score > 0.7:
                sample.quality = DataQuality.HIGH
            elif score > 0.3:
                sample.quality = DataQuality.MEDIUM
            elif score > -0.3:
                sample.quality = DataQuality.LOW
            else:
                sample.quality = DataQuality.REJECTED

            return True

    async def get_sample(self, sample_id: str) -> Optional[DataSample]:
        """Get a sample by ID."""
        return self._samples.get(sample_id)

    async def query(
        self,
        category: Optional[DataCategory] = None,
        quality: Optional[DataQuality] = None,
        labels: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[DataSample]:
        """Query samples by criteria."""
        results = []

        for sample in self._samples.values():
            if category and sample.category != category:
                continue
            if quality and sample.quality != quality:
                continue
            if labels and not all(l in sample.labels for l in labels):
                continue

            results.append(sample)
            if len(results) >= limit:
                break

        return results

    async def export_for_training(
        self,
        min_quality: DataQuality = DataQuality.MEDIUM,
        categories: Optional[List[DataCategory]] = None,
        output_path: Optional[Path] = None,
        format: str = "jsonl",  # jsonl, json, csv
    ) -> List[Dict[str, str]]:
        """Export samples for training."""
        samples = []
        quality_order = [DataQuality.HIGH, DataQuality.MEDIUM, DataQuality.LOW]

        for sample in self._samples.values():
            # Quality filter
            if quality_order.index(sample.quality) > quality_order.index(min_quality):
                continue

            # Category filter
            if categories and sample.category not in categories:
                continue

            # Skip rejected
            if sample.quality == DataQuality.REJECTED:
                continue

            samples.append(sample.to_training_format())
            sample.exported = True
            self._exported_count += 1

        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "jsonl":
                with open(output_path, "w") as f:
                    for sample in samples:
                        f.write(json.dumps(sample) + "\n")
            elif format == "json":
                with open(output_path, "w") as f:
                    json.dump(samples, f, indent=2)
            elif format == "csv":
                import csv
                with open(output_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["prompt", "completion"])
                    writer.writeheader()
                    writer.writerows(samples)

        return samples

    async def _cleanup_old_samples(self) -> None:
        """Remove oldest samples to stay under limit."""
        if len(self._samples) <= self._config.max_samples:
            return

        # Sort by timestamp
        sorted_samples = sorted(
            self._samples.values(),
            key=lambda s: s.timestamp,
        )

        # Remove oldest 10%
        remove_count = len(sorted_samples) // 10
        for sample in sorted_samples[:remove_count]:
            self._samples.pop(sample.id, None)
            self._hash_index.discard(sample.content_hash)

        logger.info(f"Cleaned up {remove_count} old samples")

    async def _load_samples(self) -> None:
        """Load samples from storage."""
        samples_file = self._config.storage_dir / "samples.jsonl"

        if not samples_file.exists():
            return

        try:
            with open(samples_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        sample = DataSample.from_dict(data)
                        self._samples[sample.id] = sample
                        self._hash_index.add(sample.content_hash)

        except Exception as e:
            logger.warning(f"Failed to load samples: {e}")

    async def save(self) -> bool:
        """Save samples to storage."""
        try:
            samples_file = self._config.storage_dir / "samples.jsonl"

            with open(samples_file, "w") as f:
                for sample in self._samples.values():
                    f.write(json.dumps(sample.to_dict()) + "\n")

            return True

        except Exception as e:
            logger.error(f"Failed to save samples: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        category_counts = {}
        quality_counts = {}

        for sample in self._samples.values():
            cat = sample.category.name
            qual = sample.quality.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
            quality_counts[qual] = quality_counts.get(qual, 0) + 1

        return {
            "total_samples": len(self._samples),
            "collected_count": self._collected_count,
            "rejected_count": self._rejected_count,
            "exported_count": self._exported_count,
            "category_counts": category_counts,
            "quality_counts": quality_counts,
            "unique_hashes": len(self._hash_index),
        }


# =============================================================================
# TRINITY INTEGRATION
# =============================================================================


class TrinityDataCollector:
    """
    Collects data from Trinity Protocol messages.

    Listens to JARVIS interactions via Trinity and automatically
    collects training data.
    """

    def __init__(
        self,
        collector: JARVISDataCollector,
    ) -> None:
        self._collector = collector
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start collecting from Trinity."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info("Trinity data collector started")

    async def stop(self) -> None:
        """Stop collecting."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _collect_loop(self) -> None:
        """Main collection loop."""
        try:
            from jarvis_prime.core.trinity_protocol import get_trinity_client, TrinityNode

            client = await get_trinity_client()

            # Register handlers
            @client.on("command_executed")
            async def handle_command(message):
                await self._collector.collect(
                    input_text=message.payload.get("command", ""),
                    output_text=message.payload.get("result", ""),
                    category=DataCategory.COMMAND,
                    session_id=message.metadata.get("session_id"),
                )

            @client.on("action_completed")
            async def handle_action(message):
                await self._collector.collect(
                    input_text=json.dumps(message.payload.get("action", {})),
                    output_text=message.payload.get("result", ""),
                    category=DataCategory.ACTION,
                )

            @client.on("user_feedback")
            async def handle_feedback(message):
                sample_id = message.payload.get("sample_id")
                score = message.payload.get("score", 0)
                correction = message.payload.get("correction")

                if sample_id:
                    await self._collector.add_feedback(sample_id, score, correction)

            # Keep running
            while self._running:
                await asyncio.sleep(1)

        except ImportError:
            logger.warning("Trinity protocol not available")
        except Exception as e:
            logger.error(f"Trinity collection error: {e}")


# =============================================================================
# SINGLETON
# =============================================================================


_data_collector: Optional[JARVISDataCollector] = None
_collector_lock = asyncio.Lock()


async def get_data_collector(
    config: Optional[CollectorConfig] = None,
) -> JARVISDataCollector:
    """Get or create global data collector."""
    global _data_collector

    async with _collector_lock:
        if _data_collector is None:
            _data_collector = JARVISDataCollector(config)
            await _data_collector.initialize()

        return _data_collector
