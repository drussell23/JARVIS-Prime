"""
Training Data Pipeline - Conversation Collection & Preparation
================================================================

v92.0 - Comprehensive Training Data Management

This module provides production-grade training data collection with:
- Real-time conversation capture from JARVIS interactions
- Automatic data annotation and quality filtering
- Instruction tuning data generation
- DPO/RLHF preference pair creation
- Data augmentation and diversity enhancement
- Cross-repo data synchronization with Reactor Core

ARCHITECTURE:
    TrainingDataPipeline
        |
        +-- ConversationCollector (real-time capture)
        +-- DataAnnotator (quality & category labeling)
        +-- InstructionGenerator (convert to instruction format)
        +-- PreferencePairGenerator (for DPO/RLHF)
        +-- DataAugmentor (diversity enhancement)
        +-- DataExporter (format conversion)
        +-- ReactorCoreSyncer (cross-repo sync)

DATA FLOW:
    User Interaction → ConversationCollector → DataAnnotator
        → InstructionGenerator → PreferencePairGenerator
        → DataAugmentor → DataExporter → Reactor Core

USAGE:
    pipeline = await get_training_data_pipeline()

    # Capture conversation
    await pipeline.capture_conversation(
        user_input="What's the weather?",
        assistant_response="I'll check the weather for you...",
        context={"intent": "weather_query"}
    )

    # Export for training
    await pipeline.export_for_training(
        format="jsonl",
        output_path="training_data.jsonl"
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
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
class TrainingDataConfig:
    """Configuration for training data pipeline."""

    # Storage
    data_dir: Path = field(default_factory=lambda: Path(
        os.getenv("TRAINING_DATA_DIR", str(Path.home() / ".jarvis" / "training_data"))
    ))
    database_path: Optional[Path] = None  # Will use data_dir/conversations.db

    # Collection settings
    min_conversation_quality: float = field(default_factory=lambda: float(
        os.getenv("TRAINING_MIN_QUALITY", "0.6")
    ))
    max_conversations_per_day: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_MAX_PER_DAY", "10000")
    ))

    # Filtering
    min_response_length: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_MIN_RESPONSE_LEN", "10")
    ))
    max_response_length: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_MAX_RESPONSE_LEN", "4096")
    ))

    # Augmentation
    enable_augmentation: bool = field(default_factory=lambda:
        os.getenv("TRAINING_ENABLE_AUGMENT", "true").lower() == "true"
    )
    augmentation_factor: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_AUGMENT_FACTOR", "2")
    ))

    # Sync
    enable_reactor_sync: bool = field(default_factory=lambda:
        os.getenv("TRAINING_REACTOR_SYNC", "true").lower() == "true"
    )
    sync_interval_minutes: int = field(default_factory=lambda: int(
        os.getenv("TRAINING_SYNC_INTERVAL", "60")
    ))

    # Privacy
    enable_pii_filtering: bool = field(default_factory=lambda:
        os.getenv("TRAINING_PII_FILTER", "true").lower() == "true"
    )

    def __post_init__(self):
        if self.database_path is None:
            self.database_path = self.data_dir / "conversations.db"
        self.data_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


class ConversationCategory(Enum):
    """Categories of conversations."""
    GENERAL_CHAT = "general_chat"
    QUESTION_ANSWER = "question_answer"
    CODE_ASSISTANCE = "code_assistance"
    SYSTEM_CONTROL = "system_control"
    FILE_OPERATIONS = "file_operations"
    WEB_BROWSING = "web_browsing"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    TROUBLESHOOTING = "troubleshooting"
    PLANNING = "planning"


class DataQuality(Enum):
    """Quality levels for training data."""
    EXCELLENT = "excellent"  # 0.9+
    GOOD = "good"  # 0.7-0.9
    ACCEPTABLE = "acceptable"  # 0.5-0.7
    POOR = "poor"  # 0.3-0.5
    UNUSABLE = "unusable"  # <0.3


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """A complete conversation for training."""
    id: str
    turns: List[ConversationTurn]
    category: ConversationCategory = ConversationCategory.GENERAL_CHAT
    quality_score: float = 0.0
    quality_level: DataQuality = DataQuality.ACCEPTABLE

    # Metadata
    source: str = "jarvis"  # "jarvis", "synthetic", "imported"
    created_at: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None  # Anonymized
    session_id: Optional[str] = None

    # Annotations
    annotations: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Processing status
    processed: bool = False
    exported: bool = False

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to OpenAI messages format."""
        return [
            {"role": turn.role, "content": turn.content}
            for turn in self.turns
        ]

    def to_instruction_format(self) -> Dict[str, str]:
        """Convert to instruction tuning format."""
        # Extract user instruction and assistant response
        instruction = ""
        response = ""

        for turn in self.turns:
            if turn.role == "user":
                instruction = turn.content
            elif turn.role == "assistant":
                response = turn.content

        return {
            "instruction": instruction,
            "input": "",
            "output": response,
        }

    def to_chat_ml(self) -> str:
        """Convert to ChatML format."""
        lines = []
        for turn in self.turns:
            lines.append(f"<|im_start|>{turn.role}")
            lines.append(turn.content)
            lines.append("<|im_end|>")
        return "\n".join(lines)


@dataclass
class PreferencePair:
    """A preference pair for RLHF/DPO training."""
    id: str
    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Non-preferred response
    category: ConversationCategory = ConversationCategory.GENERAL_CHAT

    # Metadata
    preference_source: str = "user"  # "user", "model", "heuristic"
    preference_strength: float = 1.0  # How strong the preference is
    created_at: datetime = field(default_factory=datetime.now)

    def to_dpo_format(self) -> Dict[str, str]:
        """Convert to DPO training format."""
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


@dataclass
class InstructionExample:
    """An instruction tuning example."""
    id: str
    instruction: str
    input_text: str  # Optional additional input
    output: str
    category: ConversationCategory = ConversationCategory.GENERAL_CHAT

    # Metadata
    complexity: float = 0.5
    is_multi_turn: bool = False
    requires_tools: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    def to_alpaca_format(self) -> Dict[str, str]:
        """Convert to Alpaca instruction format."""
        return {
            "instruction": self.instruction,
            "input": self.input_text,
            "output": self.output,
        }

    def to_sharegpt_format(self) -> Dict[str, Any]:
        """Convert to ShareGPT format."""
        conversations = []
        if self.input_text:
            conversations.append({
                "from": "human",
                "value": f"{self.instruction}\n\n{self.input_text}"
            })
        else:
            conversations.append({
                "from": "human",
                "value": self.instruction
            })
        conversations.append({
            "from": "gpt",
            "value": self.output
        })
        return {"conversations": conversations}


# =============================================================================
# DATA ANNOTATOR
# =============================================================================


class DataAnnotator:
    """Annotates and scores conversation data."""

    # Quality signals
    QUALITY_SIGNALS = {
        "response_length": 0.2,
        "coherence": 0.25,
        "relevance": 0.25,
        "completeness": 0.15,
        "grammar": 0.15,
    }

    # Category keywords
    CATEGORY_KEYWORDS = {
        ConversationCategory.CODE_ASSISTANCE: [
            "code", "function", "class", "python", "javascript", "debug",
            "error", "programming", "script", "variable", "loop", "array"
        ],
        ConversationCategory.SYSTEM_CONTROL: [
            "open", "close", "click", "type", "scroll", "window", "app",
            "application", "screen", "desktop", "browser"
        ],
        ConversationCategory.FILE_OPERATIONS: [
            "file", "folder", "directory", "create", "delete", "move",
            "copy", "rename", "save", "download", "upload"
        ],
        ConversationCategory.WEB_BROWSING: [
            "search", "google", "website", "url", "link", "browse",
            "navigate", "page", "tab"
        ],
        ConversationCategory.CREATIVE_WRITING: [
            "write", "story", "poem", "essay", "creative", "imagine",
            "fiction", "narrative", "character"
        ],
        ConversationCategory.ANALYSIS: [
            "analyze", "explain", "understand", "interpret", "breakdown",
            "summarize", "evaluate", "assess", "review"
        ],
        ConversationCategory.TROUBLESHOOTING: [
            "fix", "problem", "issue", "error", "not working", "help",
            "broken", "crash", "fail", "stuck"
        ],
        ConversationCategory.PLANNING: [
            "plan", "schedule", "organize", "remind", "calendar", "todo",
            "task", "deadline", "meeting"
        ],
    }

    def __init__(self, config: TrainingDataConfig):
        self._config = config

    async def annotate(self, conversation: Conversation) -> Conversation:
        """Annotate a conversation with quality scores and categories."""
        # Calculate quality score
        quality_score = await self._calculate_quality(conversation)
        conversation.quality_score = quality_score
        conversation.quality_level = self._score_to_level(quality_score)

        # Categorize
        category = self._categorize(conversation)
        conversation.category = category

        # Extract additional annotations
        conversation.annotations = await self._extract_annotations(conversation)

        # Generate tags
        conversation.tags = self._generate_tags(conversation)

        return conversation

    async def _calculate_quality(self, conversation: Conversation) -> float:
        """Calculate overall quality score."""
        scores: Dict[str, float] = {}

        # Response length score
        total_response_len = sum(
            len(t.content) for t in conversation.turns if t.role == "assistant"
        )
        if total_response_len < self._config.min_response_length:
            scores["response_length"] = 0.0
        elif total_response_len > self._config.max_response_length:
            scores["response_length"] = 0.5
        else:
            # Prefer medium-length responses
            optimal_len = (self._config.min_response_length + self._config.max_response_length) / 2
            scores["response_length"] = 1.0 - abs(total_response_len - optimal_len) / optimal_len

        # Coherence score (simple heuristics)
        scores["coherence"] = self._calculate_coherence(conversation)

        # Relevance score
        scores["relevance"] = self._calculate_relevance(conversation)

        # Completeness score
        scores["completeness"] = self._calculate_completeness(conversation)

        # Grammar score (simple check)
        scores["grammar"] = self._calculate_grammar(conversation)

        # Weighted average
        total = 0.0
        for signal, weight in self.QUALITY_SIGNALS.items():
            total += scores.get(signal, 0.5) * weight

        return min(1.0, max(0.0, total))

    def _calculate_coherence(self, conversation: Conversation) -> float:
        """Calculate coherence score."""
        score = 1.0

        for turn in conversation.turns:
            if turn.role == "assistant":
                content = turn.content.lower()

                # Penalize repetition
                words = content.split()
                if words:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.5:
                        score *= 0.5

                # Penalize gibberish patterns
                if re.search(r'(.)\1{5,}', content):
                    score *= 0.3

                # Penalize very short non-acknowledgment responses
                if len(content) < 20 and not any(
                    ack in content for ack in ["ok", "done", "yes", "no", "understood"]
                ):
                    score *= 0.7

        return score

    def _calculate_relevance(self, conversation: Conversation) -> float:
        """Calculate relevance score based on question-answer alignment."""
        if len(conversation.turns) < 2:
            return 0.5

        score = 1.0

        for i in range(len(conversation.turns) - 1):
            if conversation.turns[i].role == "user":
                user_content = conversation.turns[i].content.lower()
                # Find next assistant response
                for j in range(i + 1, len(conversation.turns)):
                    if conversation.turns[j].role == "assistant":
                        assistant_content = conversation.turns[j].content.lower()

                        # Check for keyword overlap
                        user_words = set(user_content.split())
                        assistant_words = set(assistant_content.split())
                        common = user_words & assistant_words

                        # Remove common words
                        common -= {"the", "a", "an", "is", "are", "was", "were", "i", "you", "me"}

                        if len(user_words) > 0:
                            overlap_ratio = len(common) / len(user_words)
                            if overlap_ratio < 0.1:  # Very low overlap
                                score *= 0.8

                        break

        return score

    def _calculate_completeness(self, conversation: Conversation) -> float:
        """Calculate completeness score."""
        # Check for complete sentences
        score = 1.0

        for turn in conversation.turns:
            if turn.role == "assistant":
                content = turn.content.strip()

                # Check if response ends properly
                if content and not content[-1] in ".!?\"')`:":
                    score *= 0.9

                # Check for incomplete thoughts
                if content.endswith("...") or content.endswith(".."):
                    score *= 0.8

        return score

    def _calculate_grammar(self, conversation: Conversation) -> float:
        """Calculate basic grammar score."""
        score = 1.0

        for turn in conversation.turns:
            if turn.role == "assistant":
                content = turn.content

                # Check for basic capitalization
                sentences = re.split(r'[.!?]\s+', content)
                for sentence in sentences:
                    if sentence and sentence[0].islower() and not sentence.startswith("e.g."):
                        score *= 0.95

                # Check for common errors
                common_errors = [
                    (r'\bi\b(?![\'m])', 0.95),  # Lowercase "i"
                    (r'\s{2,}', 0.98),  # Multiple spaces
                ]
                for pattern, penalty in common_errors:
                    if re.search(pattern, content):
                        score *= penalty

        return score

    def _score_to_level(self, score: float) -> DataQuality:
        """Convert numeric score to quality level."""
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.ACCEPTABLE
        elif score >= 0.3:
            return DataQuality.POOR
        else:
            return DataQuality.UNUSABLE

    def _categorize(self, conversation: Conversation) -> ConversationCategory:
        """Categorize the conversation based on content."""
        all_content = " ".join(t.content.lower() for t in conversation.turns)

        # Count keyword matches for each category
        scores: Dict[ConversationCategory, int] = defaultdict(int)

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in all_content:
                    scores[category] += 1

        # Return category with highest score, or default
        if scores:
            return max(scores, key=scores.get)
        return ConversationCategory.GENERAL_CHAT

    async def _extract_annotations(self, conversation: Conversation) -> Dict[str, Any]:
        """Extract additional annotations."""
        annotations: Dict[str, Any] = {}

        # Turn count
        annotations["turn_count"] = len(conversation.turns)

        # Average turn length
        total_len = sum(len(t.content) for t in conversation.turns)
        annotations["avg_turn_length"] = total_len / len(conversation.turns) if conversation.turns else 0

        # Has code
        all_content = " ".join(t.content for t in conversation.turns)
        annotations["has_code"] = "```" in all_content or "def " in all_content

        # Has questions
        annotations["has_questions"] = "?" in all_content

        # Sentiment (simple)
        positive_words = ["thank", "great", "perfect", "excellent", "helpful"]
        negative_words = ["error", "wrong", "bad", "terrible", "fail"]

        positive_count = sum(1 for w in positive_words if w in all_content.lower())
        negative_count = sum(1 for w in negative_words if w in all_content.lower())

        if positive_count > negative_count:
            annotations["sentiment"] = "positive"
        elif negative_count > positive_count:
            annotations["sentiment"] = "negative"
        else:
            annotations["sentiment"] = "neutral"

        return annotations

    def _generate_tags(self, conversation: Conversation) -> List[str]:
        """Generate tags for the conversation."""
        tags = [conversation.category.value]

        # Add quality tag
        tags.append(f"quality:{conversation.quality_level.value}")

        # Add source tag
        tags.append(f"source:{conversation.source}")

        # Add annotation-based tags
        if conversation.annotations.get("has_code"):
            tags.append("has_code")
        if conversation.annotations.get("has_questions"):
            tags.append("has_questions")

        return tags


# =============================================================================
# CONVERSATION COLLECTOR
# =============================================================================


class ConversationCollector:
    """Collects and stores conversations."""

    def __init__(self, config: TrainingDataConfig):
        self._config = config
        self._db_lock = asyncio.Lock()
        self._daily_count = 0
        self._last_reset = datetime.now().date()

        # Initialize database
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database."""
        self._config.data_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self._config.database_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                category TEXT,
                quality_score REAL,
                quality_level TEXT,
                source TEXT,
                created_at TEXT,
                processed INTEGER DEFAULT 0,
                exported INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON conversations(category)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality ON conversations(quality_score)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_processed ON conversations(processed)
        """)

        conn.commit()
        conn.close()

    async def capture(
        self,
        user_input: str,
        assistant_response: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Conversation]:
        """Capture a conversation exchange."""
        # Check daily limit
        today = datetime.now().date()
        if today != self._last_reset:
            self._daily_count = 0
            self._last_reset = today

        if self._daily_count >= self._config.max_conversations_per_day:
            logger.debug("Daily conversation limit reached")
            return None

        # Create conversation
        turns = []

        if system_prompt:
            turns.append(ConversationTurn(role="system", content=system_prompt))

        turns.append(ConversationTurn(role="user", content=user_input))
        turns.append(ConversationTurn(role="assistant", content=assistant_response))

        conversation = Conversation(
            id=str(uuid.uuid4()),
            turns=turns,
            source="jarvis",
        )

        # Apply context
        if context:
            conversation.annotations.update(context)
            if "category" in context:
                try:
                    conversation.category = ConversationCategory(context["category"])
                except ValueError:
                    pass

        # Store
        await self._store(conversation)
        self._daily_count += 1

        return conversation

    async def capture_multi_turn(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Conversation]:
        """Capture a multi-turn conversation."""
        turns = [
            ConversationTurn(role=m["role"], content=m["content"])
            for m in messages
        ]

        conversation = Conversation(
            id=str(uuid.uuid4()),
            turns=turns,
            source="jarvis",
        )

        if context:
            conversation.annotations.update(context)

        await self._store(conversation)
        return conversation

    async def _store(self, conversation: Conversation) -> None:
        """Store conversation in database."""
        async with self._db_lock:
            conn = sqlite3.connect(str(self._config.database_path))
            cursor = conn.cursor()

            data = json.dumps({
                "turns": [
                    {
                        "role": t.role,
                        "content": t.content,
                        "timestamp": t.timestamp.isoformat(),
                        "metadata": t.metadata,
                    }
                    for t in conversation.turns
                ],
                "annotations": conversation.annotations,
                "tags": conversation.tags,
            })

            cursor.execute("""
                INSERT OR REPLACE INTO conversations
                (id, data, category, quality_score, quality_level, source, created_at, processed, exported)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation.id,
                data,
                conversation.category.value,
                conversation.quality_score,
                conversation.quality_level.value,
                conversation.source,
                conversation.created_at.isoformat(),
                1 if conversation.processed else 0,
                1 if conversation.exported else 0,
            ))

            conn.commit()
            conn.close()

    async def get_conversations(
        self,
        category: Optional[ConversationCategory] = None,
        min_quality: Optional[float] = None,
        processed: Optional[bool] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[Conversation]:
        """Retrieve conversations with filters."""
        async with self._db_lock:
            conn = sqlite3.connect(str(self._config.database_path))
            cursor = conn.cursor()

            query = "SELECT * FROM conversations WHERE 1=1"
            params: List[Any] = []

            if category:
                query += " AND category = ?"
                params.append(category.value)

            if min_quality is not None:
                query += " AND quality_score >= ?"
                params.append(min_quality)

            if processed is not None:
                query += " AND processed = ?"
                params.append(1 if processed else 0)

            query += f" ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            conversations = []
            for row in rows:
                conv = self._row_to_conversation(row)
                if conv:
                    conversations.append(conv)

            return conversations

    def _row_to_conversation(self, row: Tuple) -> Optional[Conversation]:
        """Convert database row to Conversation object."""
        try:
            data = json.loads(row[1])

            turns = [
                ConversationTurn(
                    role=t["role"],
                    content=t["content"],
                    timestamp=datetime.fromisoformat(t.get("timestamp", datetime.now().isoformat())),
                    metadata=t.get("metadata", {}),
                )
                for t in data["turns"]
            ]

            return Conversation(
                id=row[0],
                turns=turns,
                category=ConversationCategory(row[2]) if row[2] else ConversationCategory.GENERAL_CHAT,
                quality_score=row[3] or 0.0,
                quality_level=DataQuality(row[4]) if row[4] else DataQuality.ACCEPTABLE,
                source=row[5] or "unknown",
                created_at=datetime.fromisoformat(row[6]) if row[6] else datetime.now(),
                processed=bool(row[7]),
                exported=bool(row[8]),
                annotations=data.get("annotations", {}),
                tags=data.get("tags", []),
            )
        except Exception as e:
            logger.warning(f"Failed to parse conversation: {e}")
            return None

    async def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        async with self._db_lock:
            conn = sqlite3.connect(str(self._config.database_path))
            cursor = conn.cursor()

            stats: Dict[str, Any] = {}

            # Total count
            cursor.execute("SELECT COUNT(*) FROM conversations")
            stats["total_conversations"] = cursor.fetchone()[0]

            # By category
            cursor.execute("""
                SELECT category, COUNT(*) FROM conversations
                GROUP BY category
            """)
            stats["by_category"] = dict(cursor.fetchall())

            # By quality level
            cursor.execute("""
                SELECT quality_level, COUNT(*) FROM conversations
                GROUP BY quality_level
            """)
            stats["by_quality"] = dict(cursor.fetchall())

            # Average quality score
            cursor.execute("SELECT AVG(quality_score) FROM conversations")
            avg = cursor.fetchone()[0]
            stats["avg_quality_score"] = avg if avg else 0.0

            # Processed/exported counts
            cursor.execute("SELECT COUNT(*) FROM conversations WHERE processed = 1")
            stats["processed_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM conversations WHERE exported = 1")
            stats["exported_count"] = cursor.fetchone()[0]

            conn.close()

            return stats


# =============================================================================
# INSTRUCTION GENERATOR
# =============================================================================


class InstructionGenerator:
    """Generates instruction tuning examples from conversations."""

    # JARVIS-specific instruction templates
    INSTRUCTION_TEMPLATES = {
        ConversationCategory.SYSTEM_CONTROL: [
            "Help me {action} on my computer.",
            "I need to {action}.",
            "Can you {action} for me?",
            "Please {action}.",
        ],
        ConversationCategory.CODE_ASSISTANCE: [
            "Write a {language} function that {task}.",
            "Help me debug this {language} code: {code}",
            "Explain what this code does: {code}",
            "How do I {task} in {language}?",
        ],
        ConversationCategory.FILE_OPERATIONS: [
            "Create a file called {filename}.",
            "Move {source} to {destination}.",
            "Delete {filename}.",
            "List files in {directory}.",
        ],
    }

    def __init__(self, config: TrainingDataConfig):
        self._config = config

    async def generate_from_conversation(
        self,
        conversation: Conversation,
    ) -> List[InstructionExample]:
        """Generate instruction examples from a conversation."""
        examples: List[InstructionExample] = []

        # Extract instruction-response pairs from turns
        for i in range(len(conversation.turns) - 1):
            if conversation.turns[i].role == "user":
                for j in range(i + 1, len(conversation.turns)):
                    if conversation.turns[j].role == "assistant":
                        example = InstructionExample(
                            id=f"{conversation.id}_{i}_{j}",
                            instruction=conversation.turns[i].content,
                            input_text="",
                            output=conversation.turns[j].content,
                            category=conversation.category,
                            complexity=conversation.annotations.get("complexity", 0.5),
                            is_multi_turn=len(conversation.turns) > 2,
                            requires_tools="tool" in conversation.turns[j].content.lower(),
                        )
                        examples.append(example)
                        break

        return examples

    async def generate_synthetic(
        self,
        category: ConversationCategory,
        count: int = 10,
    ) -> List[InstructionExample]:
        """Generate synthetic instruction examples."""
        examples: List[InstructionExample] = []

        templates = self.INSTRUCTION_TEMPLATES.get(category, [])
        if not templates:
            return examples

        # This would ideally use an LLM to generate diverse examples
        # For now, we'll create placeholder examples
        for i in range(count):
            template = random.choice(templates)
            example = InstructionExample(
                id=f"synthetic_{category.value}_{i}",
                instruction=template.format(
                    action="perform the action",
                    language="Python",
                    task="complete the task",
                    code="# code here",
                    filename="example.txt",
                    source="source",
                    destination="destination",
                    directory="/home/user",
                ),
                input_text="",
                output="[Assistant would provide appropriate response]",
                category=category,
                complexity=0.5,
            )
            examples.append(example)

        return examples


# =============================================================================
# PREFERENCE PAIR GENERATOR
# =============================================================================


class PreferencePairGenerator:
    """Generates preference pairs for RLHF/DPO training."""

    def __init__(self, config: TrainingDataConfig):
        self._config = config

    async def generate_from_conversations(
        self,
        chosen_conversation: Conversation,
        rejected_conversation: Conversation,
    ) -> Optional[PreferencePair]:
        """Generate preference pair from two conversations with same prompt."""
        # Find matching prompts
        chosen_prompt = None
        chosen_response = None
        rejected_response = None

        for turn in chosen_conversation.turns:
            if turn.role == "user":
                chosen_prompt = turn.content
            elif turn.role == "assistant":
                chosen_response = turn.content

        for turn in rejected_conversation.turns:
            if turn.role == "assistant":
                rejected_response = turn.content

        if not all([chosen_prompt, chosen_response, rejected_response]):
            return None

        # Verify the conversations have the same prompt
        for turn in rejected_conversation.turns:
            if turn.role == "user" and turn.content != chosen_prompt:
                return None

        return PreferencePair(
            id=f"pref_{chosen_conversation.id}_{rejected_conversation.id}",
            prompt=chosen_prompt,
            chosen=chosen_response,
            rejected=rejected_response,
            category=chosen_conversation.category,
            preference_source="quality",
            preference_strength=chosen_conversation.quality_score - rejected_conversation.quality_score,
        )

    async def generate_from_feedback(
        self,
        prompt: str,
        responses: List[str],
        rankings: List[int],  # Lower is better
    ) -> List[PreferencePair]:
        """Generate preference pairs from explicit feedback."""
        pairs: List[PreferencePair] = []

        # Create pairs from rankings
        sorted_responses = sorted(zip(rankings, responses))

        for i in range(len(sorted_responses)):
            for j in range(i + 1, len(sorted_responses)):
                if sorted_responses[i][0] < sorted_responses[j][0]:
                    pair = PreferencePair(
                        id=f"feedback_{uuid.uuid4().hex[:8]}",
                        prompt=prompt,
                        chosen=sorted_responses[i][1],
                        rejected=sorted_responses[j][1],
                        preference_source="user",
                        preference_strength=float(sorted_responses[j][0] - sorted_responses[i][0]),
                    )
                    pairs.append(pair)

        return pairs


# =============================================================================
# DATA AUGMENTOR
# =============================================================================


class DataAugmentor:
    """Augments training data for diversity."""

    def __init__(self, config: TrainingDataConfig):
        self._config = config

    async def augment_instruction(
        self,
        example: InstructionExample,
        num_variants: int = 2,
    ) -> List[InstructionExample]:
        """Generate augmented variants of an instruction example."""
        variants = [example]  # Include original

        if not self._config.enable_augmentation:
            return variants

        for i in range(num_variants):
            # Create variant with minor modifications
            variant = InstructionExample(
                id=f"{example.id}_aug_{i}",
                instruction=self._rephrase_instruction(example.instruction, i),
                input_text=example.input_text,
                output=example.output,
                category=example.category,
                complexity=example.complexity,
                is_multi_turn=example.is_multi_turn,
                requires_tools=example.requires_tools,
            )
            variants.append(variant)

        return variants

    def _rephrase_instruction(self, instruction: str, variant_idx: int) -> str:
        """Simple instruction rephrasing."""
        # Simple transformations
        transformations = [
            # Add politeness
            lambda s: f"Please {s[0].lower() + s[1:]}" if not s.startswith("Please") else s,
            # Add formality
            lambda s: f"Could you {s[0].lower() + s[1:]}" if not s.startswith("Could") else s,
            # Make direct
            lambda s: s.replace("Could you ", "").replace("Please ", "").capitalize(),
        ]

        if variant_idx < len(transformations):
            return transformations[variant_idx](instruction)
        return instruction

    async def augment_conversation(
        self,
        conversation: Conversation,
    ) -> List[Conversation]:
        """Generate augmented variants of a conversation."""
        if not self._config.enable_augmentation:
            return [conversation]

        # For now, return original (could add noise injection, etc.)
        return [conversation]


# =============================================================================
# DATA EXPORTER
# =============================================================================


class DataExporter:
    """Exports training data in various formats."""

    def __init__(self, config: TrainingDataConfig):
        self._config = config

    async def export_instructions(
        self,
        examples: List[InstructionExample],
        output_path: Path,
        format: str = "jsonl",  # "jsonl", "alpaca", "sharegpt"
    ) -> None:
        """Export instruction examples to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for example in examples:
                if format == "jsonl" or format == "alpaca":
                    data = example.to_alpaca_format()
                elif format == "sharegpt":
                    data = example.to_sharegpt_format()
                else:
                    data = example.to_alpaca_format()

                f.write(json.dumps(data) + "\n")

        logger.info(f"Exported {len(examples)} instruction examples to {output_path}")

    async def export_preference_pairs(
        self,
        pairs: List[PreferencePair],
        output_path: Path,
        format: str = "dpo",
    ) -> None:
        """Export preference pairs for DPO/RLHF training."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for pair in pairs:
                data = pair.to_dpo_format()
                f.write(json.dumps(data) + "\n")

        logger.info(f"Exported {len(pairs)} preference pairs to {output_path}")

    async def export_conversations(
        self,
        conversations: List[Conversation],
        output_path: Path,
        format: str = "messages",  # "messages", "chatml"
    ) -> None:
        """Export conversations for training."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for conv in conversations:
                if format == "messages":
                    data = {"messages": conv.to_messages()}
                elif format == "chatml":
                    data = {"text": conv.to_chat_ml()}
                else:
                    data = {"messages": conv.to_messages()}

                f.write(json.dumps(data) + "\n")

        logger.info(f"Exported {len(conversations)} conversations to {output_path}")


# =============================================================================
# PRIVACY FILTER
# =============================================================================


class PrivacyFilter:
    """Filters PII from training data."""

    # Patterns for common PII
    PII_PATTERNS = [
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),
        (r'\b\d{3}[-]?\d{2}[-]?\d{4}\b', '[SSN]'),
        (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[CARD]'),
        (r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', '[IP]'),
    ]

    def __init__(self, config: TrainingDataConfig):
        self._config = config

    async def filter_conversation(self, conversation: Conversation) -> Conversation:
        """Filter PII from conversation."""
        if not self._config.enable_pii_filtering:
            return conversation

        for turn in conversation.turns:
            turn.content = self._filter_text(turn.content)

        return conversation

    def _filter_text(self, text: str) -> str:
        """Filter PII from text."""
        filtered = text

        for pattern, replacement in self.PII_PATTERNS:
            filtered = re.sub(pattern, replacement, filtered)

        return filtered


# =============================================================================
# TRAINING DATA PIPELINE
# =============================================================================


class TrainingDataPipeline:
    """Main orchestrator for training data collection and preparation."""

    def __init__(self, config: Optional[TrainingDataConfig] = None):
        self._config = config or TrainingDataConfig()

        self._collector = ConversationCollector(self._config)
        self._annotator = DataAnnotator(self._config)
        self._instruction_gen = InstructionGenerator(self._config)
        self._preference_gen = PreferencePairGenerator(self._config)
        self._augmentor = DataAugmentor(self._config)
        self._exporter = DataExporter(self._config)
        self._privacy_filter = PrivacyFilter(self._config)

        self._sync_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the pipeline background tasks."""
        if self._config.enable_reactor_sync:
            self._sync_task = asyncio.create_task(self._sync_loop())
            logger.info("Started Reactor Core sync loop")

    async def stop(self) -> None:
        """Stop background tasks."""
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

    async def capture_conversation(
        self,
        user_input: str,
        assistant_response: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Conversation]:
        """Capture and process a conversation."""
        # Capture
        conversation = await self._collector.capture(
            user_input=user_input,
            assistant_response=assistant_response,
            system_prompt=system_prompt,
            context=context,
        )

        if not conversation:
            return None

        # Apply privacy filter
        conversation = await self._privacy_filter.filter_conversation(conversation)

        # Annotate
        conversation = await self._annotator.annotate(conversation)

        # Check quality threshold
        if conversation.quality_score < self._config.min_conversation_quality:
            logger.debug(f"Conversation quality {conversation.quality_score:.2f} below threshold")
            return None

        # Store with annotations
        await self._collector._store(conversation)

        return conversation

    async def export_for_training(
        self,
        output_dir: Path,
        min_quality: float = 0.6,
        include_augmented: bool = True,
    ) -> Dict[str, Path]:
        """Export all training data to specified directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get high-quality conversations
        conversations = await self._collector.get_conversations(
            min_quality=min_quality,
            processed=False,
            limit=10000,
        )

        logger.info(f"Exporting {len(conversations)} conversations")

        # Generate instruction examples
        all_instructions: List[InstructionExample] = []
        for conv in conversations:
            examples = await self._instruction_gen.generate_from_conversation(conv)
            if include_augmented:
                for ex in examples:
                    augmented = await self._augmentor.augment_instruction(ex)
                    all_instructions.extend(augmented)
            else:
                all_instructions.extend(examples)

        # Export files
        output_files: Dict[str, Path] = {}

        # Instructions (Alpaca format)
        instructions_path = output_dir / "instructions_alpaca.jsonl"
        await self._exporter.export_instructions(all_instructions, instructions_path, "alpaca")
        output_files["instructions_alpaca"] = instructions_path

        # Instructions (ShareGPT format)
        sharegpt_path = output_dir / "instructions_sharegpt.jsonl"
        await self._exporter.export_instructions(all_instructions, sharegpt_path, "sharegpt")
        output_files["instructions_sharegpt"] = sharegpt_path

        # Conversations (messages format)
        conversations_path = output_dir / "conversations.jsonl"
        await self._exporter.export_conversations(conversations, conversations_path)
        output_files["conversations"] = conversations_path

        # ChatML format
        chatml_path = output_dir / "conversations_chatml.jsonl"
        await self._exporter.export_conversations(conversations, chatml_path, "chatml")
        output_files["chatml"] = chatml_path

        logger.info(f"Exported training data to {output_dir}")

        return output_files

    async def _sync_loop(self) -> None:
        """Background loop to sync with Reactor Core."""
        while True:
            try:
                await asyncio.sleep(self._config.sync_interval_minutes * 60)
                await self._sync_to_reactor()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

    async def _sync_to_reactor(self) -> None:
        """Sync training data to Reactor Core."""
        try:
            from jarvis_prime.core.reactor_core_bridge import get_reactor_core_bridge

            bridge = await get_reactor_core_bridge()

            # Get unexported conversations
            conversations = await self._collector.get_conversations(
                exported=False,
                min_quality=self._config.min_conversation_quality,
                limit=1000,
            )

            if not conversations:
                return

            # Prepare data
            instructions = []
            for conv in conversations:
                examples = await self._instruction_gen.generate_from_conversation(conv)
                instructions.extend([ex.to_alpaca_format() for ex in examples])

            if instructions:
                # Send to Reactor Core
                await bridge.upload_training_data(
                    data=instructions,
                    data_type="instructions",
                )

                logger.info(f"Synced {len(instructions)} examples to Reactor Core")

        except ImportError:
            logger.debug("Reactor Core bridge not available")
        except Exception as e:
            logger.error(f"Failed to sync to Reactor Core: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "config": {
                "data_dir": str(self._config.data_dir),
                "min_quality": self._config.min_conversation_quality,
                "augmentation_enabled": self._config.enable_augmentation,
                "reactor_sync_enabled": self._config.enable_reactor_sync,
                "pii_filtering_enabled": self._config.enable_pii_filtering,
            },
            "sync_task_running": self._sync_task is not None and not self._sync_task.done(),
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


_pipeline: Optional[TrainingDataPipeline] = None
_pipeline_lock = asyncio.Lock()


async def get_training_data_pipeline(
    config: Optional[TrainingDataConfig] = None,
) -> TrainingDataPipeline:
    """Get or create the global training data pipeline."""
    global _pipeline

    async with _pipeline_lock:
        if _pipeline is None:
            _pipeline = TrainingDataPipeline(config)
            await _pipeline.start()

        return _pipeline


async def shutdown_training_data_pipeline() -> None:
    """Shutdown the global training data pipeline."""
    global _pipeline

    async with _pipeline_lock:
        if _pipeline is not None:
            await _pipeline.stop()
            _pipeline = None


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Config
    "TrainingDataConfig",
    # Enums
    "ConversationCategory",
    "DataQuality",
    # Data structures
    "ConversationTurn",
    "Conversation",
    "PreferencePair",
    "InstructionExample",
    # Components
    "DataAnnotator",
    "ConversationCollector",
    "InstructionGenerator",
    "PreferencePairGenerator",
    "DataAugmentor",
    "DataExporter",
    "PrivacyFilter",
    # Pipeline
    "TrainingDataPipeline",
    # Factory
    "get_training_data_pipeline",
    "shutdown_training_data_pipeline",
]
