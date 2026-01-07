"""
Multi-Modal Fusion Engine - Cross-Modal Understanding and Integration
=======================================================================

v76.0 - Advanced Multi-Modal Processing

This module provides multi-modal capabilities for JARVIS Prime:
- Screen understanding (beyond basic vision)
- Audio-visual fusion
- Temporal reasoning across frames
- Spatial reasoning for macOS UI
- Gesture and interaction understanding

ARCHITECTURE:
    Multi-Modal Input -> Modality Encoders -> Fusion Layer -> Unified Representation

FEATURES:
    - Late fusion for flexible modal combination
    - Temporal attention for video/screen sequences
    - Spatial reasoning for UI element relationships
    - Cross-modal attention mechanisms
    - Dynamic modality weighting
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import struct
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class Modality(Enum):
    """Supported input modalities."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SCREEN = "screen"           # macOS screen capture
    UI_ELEMENTS = "ui_elements"  # Accessibility tree
    GESTURE = "gesture"         # Mouse/trackpad gestures
    KEYBOARD = "keyboard"       # Keyboard input patterns


class FusionStrategy(Enum):
    """Strategies for multi-modal fusion."""
    EARLY = "early"             # Concatenate raw features
    LATE = "late"               # Fuse after modal-specific processing
    ATTENTION = "attention"     # Cross-modal attention
    HIERARCHICAL = "hierarchical"  # Multiple levels of fusion
    DYNAMIC = "dynamic"         # Adaptive based on input


class TemporalMode(Enum):
    """Temporal processing modes."""
    SINGLE = "single"           # Single frame/instant
    SEQUENCE = "sequence"       # Fixed-length sequence
    STREAM = "stream"           # Continuous stream
    EVENT = "event"             # Event-triggered


class SpatialRelation(Enum):
    """Spatial relationships for UI reasoning."""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"
    CONTAINS = "contains"
    OVERLAPS = "overlaps"
    NEAR = "near"
    FAR = "far"
    ALIGNED = "aligned"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ModalityInput:
    """Input from a single modality."""
    modality: Modality
    data: Any
    timestamp: float = field(default_factory=time.time)

    # Metadata
    source: str = ""
    format: str = ""
    dimensions: Optional[Tuple[int, ...]] = None

    # Quality indicators
    confidence: float = 1.0
    noise_level: float = 0.0

    # For sequences
    frame_index: Optional[int] = None
    sequence_id: Optional[str] = None


@dataclass
class UIElement:
    """Represents a UI element from the accessibility tree."""
    id: str = ""
    role: str = ""              # button, text, image, etc.
    label: str = ""
    value: str = ""

    # Position
    bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, width, height
    center: Tuple[int, int] = (0, 0)

    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0

    # State
    is_focused: bool = False
    is_enabled: bool = True
    is_visible: bool = True

    # Relationships (computed)
    spatial_relations: Dict[str, List[str]] = field(default_factory=dict)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside this element."""
        bx, by, bw, bh = self.bounds
        return bx <= x <= bx + bw and by <= y <= by + bh

    def distance_to(self, other: "UIElement") -> float:
        """Calculate distance to another element."""
        dx = self.center[0] - other.center[0]
        dy = self.center[1] - other.center[1]
        return math.sqrt(dx * dx + dy * dy)


@dataclass
class ScreenFrame:
    """A single screen capture frame."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # Image data
    image_data: Optional[bytes] = None
    image_format: str = "png"
    width: int = 0
    height: int = 0

    # UI elements (from accessibility)
    ui_elements: List[UIElement] = field(default_factory=list)
    ui_tree_root: Optional[str] = None

    # Cursor info
    cursor_position: Tuple[int, int] = (0, 0)
    cursor_type: str = "arrow"

    # Active window
    active_window: Optional[str] = None
    active_app: Optional[str] = None

    def get_element_at(self, x: int, y: int) -> Optional[UIElement]:
        """Get UI element at position."""
        for elem in reversed(self.ui_elements):  # Reverse for z-order
            if elem.contains_point(x, y) and elem.is_visible:
                return elem
        return None


@dataclass
class GestureEvent:
    """A gesture or input event."""
    type: str = ""  # click, scroll, swipe, pinch, etc.
    timestamp: float = field(default_factory=time.time)

    # Position
    start_position: Tuple[int, int] = (0, 0)
    end_position: Optional[Tuple[int, int]] = None

    # Properties
    button: str = "left"  # left, right, middle
    modifiers: List[str] = field(default_factory=list)  # shift, ctrl, cmd, opt
    magnitude: float = 0.0  # For scroll/swipe
    direction: Optional[str] = None  # up, down, left, right

    # Duration
    duration_ms: float = 0.0


@dataclass
class AudioSegment:
    """An audio segment for processing."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # Audio data
    samples: Optional[bytes] = None
    sample_rate: int = 16000
    channels: int = 1
    duration_seconds: float = 0.0

    # Features (extracted)
    features: Optional[Dict[str, Any]] = None

    # Transcription
    transcription: Optional[str] = None
    transcription_confidence: float = 0.0

    # Classification
    audio_type: str = "speech"  # speech, music, noise, silence


@dataclass
class TemporalContext:
    """Context for temporal reasoning across frames."""
    sequence_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    frames: List[ScreenFrame] = field(default_factory=list)
    gestures: List[GestureEvent] = field(default_factory=list)
    audio_segments: List[AudioSegment] = field(default_factory=list)

    # Time window
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_seconds: float = 0.0

    # Detected patterns
    detected_actions: List[Dict[str, Any]] = field(default_factory=list)
    state_changes: List[Dict[str, Any]] = field(default_factory=list)

    def add_frame(self, frame: ScreenFrame) -> None:
        """Add a frame to the sequence."""
        self.frames.append(frame)
        self.end_time = frame.timestamp
        self.duration_seconds = (self.end_time - self.start_time) if self.start_time else 0.0

    def get_frame_at(self, timestamp: float, tolerance: float = 0.1) -> Optional[ScreenFrame]:
        """Get frame closest to timestamp."""
        for frame in self.frames:
            if abs(frame.timestamp - timestamp) <= tolerance:
                return frame
        return None


@dataclass
class FusedRepresentation:
    """Unified representation from multi-modal fusion."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    # Source modalities
    modalities_used: List[Modality] = field(default_factory=list)
    modal_weights: Dict[str, float] = field(default_factory=dict)

    # Fused features
    features: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    # Semantic content
    description: str = ""
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    detected_text: List[str] = field(default_factory=list)
    detected_actions: List[str] = field(default_factory=list)

    # Spatial understanding
    spatial_layout: Dict[str, Any] = field(default_factory=dict)
    element_relations: List[Tuple[str, SpatialRelation, str]] = field(default_factory=list)

    # Confidence
    overall_confidence: float = 0.5
    modality_confidences: Dict[str, float] = field(default_factory=dict)


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal processing."""
    # Enabled modalities
    enabled_modalities: List[Modality] = field(default_factory=lambda: [
        Modality.TEXT, Modality.IMAGE, Modality.SCREEN
    ])

    # Fusion
    fusion_strategy: FusionStrategy = FusionStrategy.ATTENTION
    modality_weights: Dict[str, float] = field(default_factory=lambda: {
        "text": 0.4, "image": 0.3, "screen": 0.2, "audio": 0.1
    })

    # Temporal
    temporal_mode: TemporalMode = TemporalMode.SEQUENCE
    max_sequence_length: int = 30  # frames
    temporal_window_seconds: float = 10.0

    # Spatial
    enable_spatial_reasoning: bool = True
    spatial_relation_threshold: float = 0.3  # Min confidence for relations

    # Screen processing
    screen_sample_rate: float = 2.0  # FPS
    extract_ui_elements: bool = True
    track_cursor: bool = True

    # Audio processing
    audio_sample_rate: int = 16000
    enable_transcription: bool = True

    # Performance
    max_image_resolution: int = 1024
    enable_caching: bool = True
    cache_size: int = 100


# =============================================================================
# MODALITY ENCODERS
# =============================================================================

class ModalityEncoder(ABC):
    """Base class for modality-specific encoders."""

    modality: Modality

    @abstractmethod
    async def encode(
        self,
        input_data: ModalityInput,
    ) -> Dict[str, Any]:
        """Encode input to feature representation."""
        ...

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get output embedding dimension."""
        ...


class TextEncoder(ModalityEncoder):
    """Encoder for text modality."""

    modality = Modality.TEXT

    def __init__(self, embedding_dim: int = 768):
        self._embedding_dim = embedding_dim

    async def encode(self, input_data: ModalityInput) -> Dict[str, Any]:
        """Encode text to features."""
        text = str(input_data.data)

        # Simple feature extraction (would use transformer in production)
        features = {
            "tokens": text.split(),
            "length": len(text),
            "word_count": len(text.split()),
            "has_question": "?" in text,
            "has_command": any(w in text.lower() for w in ["click", "type", "open", "close"]),
        }

        # Placeholder embedding
        embedding = [0.0] * self._embedding_dim
        for i, char in enumerate(text[:self._embedding_dim]):
            embedding[i] = ord(char) / 256.0

        return {
            "features": features,
            "embedding": embedding,
        }

    def get_embedding_dim(self) -> int:
        return self._embedding_dim


class ImageEncoder(ModalityEncoder):
    """Encoder for image modality."""

    modality = Modality.IMAGE

    def __init__(self, embedding_dim: int = 512):
        self._embedding_dim = embedding_dim

    async def encode(self, input_data: ModalityInput) -> Dict[str, Any]:
        """Encode image to features."""
        # Image data could be bytes, path, or array
        data = input_data.data

        features = {
            "dimensions": input_data.dimensions,
            "format": input_data.format,
        }

        # Would use vision model for actual encoding
        embedding = [0.0] * self._embedding_dim

        return {
            "features": features,
            "embedding": embedding,
        }

    def get_embedding_dim(self) -> int:
        return self._embedding_dim


class ScreenEncoder(ModalityEncoder):
    """Encoder for screen capture modality."""

    modality = Modality.SCREEN

    def __init__(self, embedding_dim: int = 1024):
        self._embedding_dim = embedding_dim

    async def encode(self, input_data: ModalityInput) -> Dict[str, Any]:
        """Encode screen frame to features."""
        if isinstance(input_data.data, ScreenFrame):
            frame = input_data.data
        else:
            frame = ScreenFrame()

        features = {
            "width": frame.width,
            "height": frame.height,
            "ui_element_count": len(frame.ui_elements),
            "active_app": frame.active_app,
            "cursor_position": frame.cursor_position,
        }

        # Extract UI hierarchy features
        if frame.ui_elements:
            features["element_types"] = defaultdict(int)
            for elem in frame.ui_elements:
                features["element_types"][elem.role] += 1

        embedding = [0.0] * self._embedding_dim

        return {
            "features": features,
            "embedding": embedding,
            "ui_elements": [self._encode_ui_element(e) for e in frame.ui_elements],
        }

    def _encode_ui_element(self, element: UIElement) -> Dict[str, Any]:
        """Encode a single UI element."""
        return {
            "id": element.id,
            "role": element.role,
            "label": element.label[:50] if element.label else "",
            "bounds": element.bounds,
            "is_focused": element.is_focused,
            "is_visible": element.is_visible,
        }

    def get_embedding_dim(self) -> int:
        return self._embedding_dim


class AudioEncoder(ModalityEncoder):
    """Encoder for audio modality."""

    modality = Modality.AUDIO

    def __init__(self, embedding_dim: int = 256):
        self._embedding_dim = embedding_dim

    async def encode(self, input_data: ModalityInput) -> Dict[str, Any]:
        """Encode audio segment to features."""
        if isinstance(input_data.data, AudioSegment):
            segment = input_data.data
        else:
            segment = AudioSegment()

        features = {
            "duration_seconds": segment.duration_seconds,
            "sample_rate": segment.sample_rate,
            "audio_type": segment.audio_type,
            "transcription": segment.transcription,
        }

        embedding = [0.0] * self._embedding_dim

        return {
            "features": features,
            "embedding": embedding,
        }

    def get_embedding_dim(self) -> int:
        return self._embedding_dim


class GestureEncoder(ModalityEncoder):
    """Encoder for gesture/input events."""

    modality = Modality.GESTURE

    def __init__(self, embedding_dim: int = 64):
        self._embedding_dim = embedding_dim

    async def encode(self, input_data: ModalityInput) -> Dict[str, Any]:
        """Encode gesture events to features."""
        if isinstance(input_data.data, list):
            gestures = input_data.data
        elif isinstance(input_data.data, GestureEvent):
            gestures = [input_data.data]
        else:
            gestures = []

        features = {
            "gesture_count": len(gestures),
            "gesture_types": defaultdict(int),
        }

        for gesture in gestures:
            features["gesture_types"][gesture.type] += 1

        embedding = [0.0] * self._embedding_dim

        return {
            "features": features,
            "embedding": embedding,
            "gestures": [self._encode_gesture(g) for g in gestures],
        }

    def _encode_gesture(self, gesture: GestureEvent) -> Dict[str, Any]:
        """Encode a single gesture."""
        return {
            "type": gesture.type,
            "start": gesture.start_position,
            "end": gesture.end_position,
            "modifiers": gesture.modifiers,
            "duration_ms": gesture.duration_ms,
        }

    def get_embedding_dim(self) -> int:
        return self._embedding_dim


# =============================================================================
# SPATIAL REASONING
# =============================================================================

class SpatialReasoner:
    """
    Spatial reasoning for UI elements.

    Computes and understands spatial relationships between UI elements
    for better screen understanding.
    """

    def __init__(self, config: MultiModalConfig):
        self.config = config

    def compute_relations(
        self,
        elements: List[UIElement],
    ) -> List[Tuple[str, SpatialRelation, str]]:
        """Compute spatial relations between all element pairs."""
        relations = []

        for i, elem1 in enumerate(elements):
            for elem2 in elements[i + 1:]:
                relation = self._compute_relation(elem1, elem2)
                if relation:
                    relations.append((elem1.id, relation, elem2.id))

        return relations

    def _compute_relation(
        self,
        elem1: UIElement,
        elem2: UIElement,
    ) -> Optional[SpatialRelation]:
        """Compute spatial relation between two elements."""
        x1, y1, w1, h1 = elem1.bounds
        x2, y2, w2, h2 = elem2.bounds

        # Check containment
        if self._contains(elem1.bounds, elem2.bounds):
            return SpatialRelation.CONTAINS
        if self._contains(elem2.bounds, elem1.bounds):
            return SpatialRelation.INSIDE

        # Check overlap
        if self._overlaps(elem1.bounds, elem2.bounds):
            return SpatialRelation.OVERLAPS

        # Check directional relations
        center1 = elem1.center
        center2 = elem2.center

        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]

        # Determine primary direction
        if abs(dx) > abs(dy):
            if dx > 0:
                return SpatialRelation.LEFT_OF  # elem1 is left of elem2
            else:
                return SpatialRelation.RIGHT_OF
        else:
            if dy > 0:
                return SpatialRelation.ABOVE  # elem1 is above elem2
            else:
                return SpatialRelation.BELOW

    def _contains(
        self,
        outer: Tuple[int, int, int, int],
        inner: Tuple[int, int, int, int],
    ) -> bool:
        """Check if outer contains inner."""
        ox, oy, ow, oh = outer
        ix, iy, iw, ih = inner

        return (ox <= ix and oy <= iy and
                ox + ow >= ix + iw and oy + oh >= iy + ih)

    def _overlaps(
        self,
        bounds1: Tuple[int, int, int, int],
        bounds2: Tuple[int, int, int, int],
    ) -> bool:
        """Check if bounds overlap."""
        x1, y1, w1, h1 = bounds1
        x2, y2, w2, h2 = bounds2

        return not (x1 + w1 < x2 or x2 + w2 < x1 or
                   y1 + h1 < y2 or y2 + h2 < y1)

    def find_element_by_description(
        self,
        elements: List[UIElement],
        description: str,
    ) -> List[UIElement]:
        """Find elements matching a natural language description."""
        matches = []
        desc_lower = description.lower()

        for elem in elements:
            score = 0

            # Match by role
            if elem.role.lower() in desc_lower:
                score += 0.5

            # Match by label
            if elem.label and elem.label.lower() in desc_lower:
                score += 0.8

            # Match by role keywords
            role_keywords = {
                "button": ["click", "press", "button"],
                "text": ["text", "label", "title"],
                "image": ["image", "picture", "icon"],
                "input": ["type", "enter", "field", "input"],
            }

            for role, keywords in role_keywords.items():
                if elem.role.lower() == role:
                    if any(kw in desc_lower for kw in keywords):
                        score += 0.3

            if score > self.config.spatial_relation_threshold:
                matches.append((elem, score))

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)
        return [m[0] for m in matches]


# =============================================================================
# TEMPORAL REASONING
# =============================================================================

class TemporalReasoner:
    """
    Temporal reasoning across frames and events.

    Detects patterns, state changes, and user actions over time.
    """

    def __init__(self, config: MultiModalConfig):
        self.config = config
        self._action_patterns: Dict[str, List[str]] = {
            "click": ["mouse_down", "mouse_up"],
            "double_click": ["mouse_down", "mouse_up", "mouse_down", "mouse_up"],
            "drag": ["mouse_down", "mouse_move+", "mouse_up"],
            "scroll": ["scroll"],
            "type": ["key_down+"],
        }

    def analyze_sequence(
        self,
        context: TemporalContext,
    ) -> Dict[str, Any]:
        """Analyze a temporal sequence."""
        analysis = {
            "duration_seconds": context.duration_seconds,
            "frame_count": len(context.frames),
            "gesture_count": len(context.gestures),
            "detected_actions": [],
            "state_changes": [],
            "activity_summary": "",
        }

        # Detect actions from gestures
        analysis["detected_actions"] = self._detect_actions(context.gestures)

        # Detect state changes from frames
        if len(context.frames) >= 2:
            analysis["state_changes"] = self._detect_state_changes(context.frames)

        # Generate summary
        analysis["activity_summary"] = self._generate_summary(analysis)

        return analysis

    def _detect_actions(
        self,
        gestures: List[GestureEvent],
    ) -> List[Dict[str, Any]]:
        """Detect high-level actions from gesture events."""
        actions = []

        for gesture in gestures:
            action = {
                "type": gesture.type,
                "timestamp": gesture.timestamp,
                "position": gesture.start_position,
                "confidence": 0.8,
            }

            # Classify action type
            if gesture.type == "click":
                if gesture.button == "right":
                    action["semantic"] = "context_menu"
                else:
                    action["semantic"] = "select"

            elif gesture.type == "double_click":
                action["semantic"] = "open"

            elif gesture.type == "scroll":
                action["semantic"] = "navigate"
                action["direction"] = gesture.direction

            elif gesture.type == "type":
                action["semantic"] = "input"

            actions.append(action)

        return actions

    def _detect_state_changes(
        self,
        frames: List[ScreenFrame],
    ) -> List[Dict[str, Any]]:
        """Detect state changes between frames."""
        changes = []

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            change = {
                "timestamp": curr_frame.timestamp,
                "changes": [],
            }

            # Check for window changes
            if prev_frame.active_window != curr_frame.active_window:
                change["changes"].append({
                    "type": "window_change",
                    "from": prev_frame.active_window,
                    "to": curr_frame.active_window,
                })

            # Check for app changes
            if prev_frame.active_app != curr_frame.active_app:
                change["changes"].append({
                    "type": "app_change",
                    "from": prev_frame.active_app,
                    "to": curr_frame.active_app,
                })

            # Check for UI element changes
            prev_ids = {e.id for e in prev_frame.ui_elements}
            curr_ids = {e.id for e in curr_frame.ui_elements}

            new_elements = curr_ids - prev_ids
            removed_elements = prev_ids - curr_ids

            if new_elements:
                change["changes"].append({
                    "type": "elements_added",
                    "count": len(new_elements),
                })

            if removed_elements:
                change["changes"].append({
                    "type": "elements_removed",
                    "count": len(removed_elements),
                })

            if change["changes"]:
                changes.append(change)

        return changes

    def _generate_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate natural language summary of activity."""
        parts = []

        if analysis["frame_count"] > 0:
            parts.append(f"Analyzed {analysis['frame_count']} frames over {analysis['duration_seconds']:.1f}s")

        if analysis["detected_actions"]:
            action_types = defaultdict(int)
            for action in analysis["detected_actions"]:
                action_types[action.get("semantic", action["type"])] += 1

            action_str = ", ".join(f"{count} {action}" for action, count in action_types.items())
            parts.append(f"Detected actions: {action_str}")

        if analysis["state_changes"]:
            parts.append(f"Observed {len(analysis['state_changes'])} state changes")

        return ". ".join(parts) if parts else "No significant activity detected"


# =============================================================================
# MULTI-MODAL FUSION ENGINE
# =============================================================================

class MultiModalFusionEngine:
    """
    Main engine for multi-modal fusion.

    Combines inputs from multiple modalities into unified representations
    for downstream reasoning.
    """

    # Encoder registry
    ENCODERS: Dict[Modality, Type[ModalityEncoder]] = {
        Modality.TEXT: TextEncoder,
        Modality.IMAGE: ImageEncoder,
        Modality.SCREEN: ScreenEncoder,
        Modality.AUDIO: AudioEncoder,
        Modality.GESTURE: GestureEncoder,
    }

    def __init__(
        self,
        config: Optional[MultiModalConfig] = None,
        executor: Optional[Any] = None,
    ):
        self.config = config or MultiModalConfig()
        self.executor = executor

        # Initialize encoders
        self._encoders: Dict[Modality, ModalityEncoder] = {}
        for modality in self.config.enabled_modalities:
            if modality in self.ENCODERS:
                self._encoders[modality] = self.ENCODERS[modality]()

        # Initialize reasoners
        self._spatial_reasoner = SpatialReasoner(self.config)
        self._temporal_reasoner = TemporalReasoner(self.config)

        # Caching
        self._cache: Dict[str, FusedRepresentation] = {}

        # Statistics
        self._total_fusions = 0
        self._modality_usage: Dict[str, int] = defaultdict(int)

        logger.info(f"MultiModalFusionEngine initialized with {len(self._encoders)} encoders")

    async def fuse(
        self,
        inputs: List[ModalityInput],
        temporal_context: Optional[TemporalContext] = None,
    ) -> FusedRepresentation:
        """
        Fuse multiple modal inputs into unified representation.

        Args:
            inputs: List of modal inputs
            temporal_context: Optional temporal context for sequence reasoning

        Returns:
            FusedRepresentation with unified features
        """
        self._total_fusions += 1

        # Check cache
        cache_key = self._make_cache_key(inputs)
        if self.config.enable_caching and cache_key in self._cache:
            return self._cache[cache_key]

        # Encode each modality
        encodings: Dict[Modality, Dict[str, Any]] = {}

        for modal_input in inputs:
            if modal_input.modality in self._encoders:
                encoder = self._encoders[modal_input.modality]
                encoding = await encoder.encode(modal_input)
                encodings[modal_input.modality] = encoding
                self._modality_usage[modal_input.modality.value] += 1

        # Apply fusion strategy
        if self.config.fusion_strategy == FusionStrategy.ATTENTION:
            fused = self._attention_fusion(encodings)
        elif self.config.fusion_strategy == FusionStrategy.LATE:
            fused = self._late_fusion(encodings)
        elif self.config.fusion_strategy == FusionStrategy.HIERARCHICAL:
            fused = self._hierarchical_fusion(encodings)
        else:
            fused = self._early_fusion(encodings)

        # Add spatial reasoning
        screen_encoding = encodings.get(Modality.SCREEN, {})
        ui_elements = screen_encoding.get("ui_elements", [])

        if ui_elements and self.config.enable_spatial_reasoning:
            # Convert back to UIElement objects for spatial reasoning
            elements = []
            for elem_dict in ui_elements:
                elem = UIElement(
                    id=elem_dict.get("id", ""),
                    role=elem_dict.get("role", ""),
                    label=elem_dict.get("label", ""),
                    bounds=tuple(elem_dict.get("bounds", (0, 0, 0, 0))),
                )
                elements.append(elem)

            relations = self._spatial_reasoner.compute_relations(elements)
            fused.element_relations = relations
            fused.spatial_layout = {"element_count": len(elements)}

        # Add temporal reasoning
        if temporal_context:
            temporal_analysis = self._temporal_reasoner.analyze_sequence(temporal_context)
            fused.detected_actions = temporal_analysis.get("detected_actions", [])
            fused.features["temporal"] = temporal_analysis

        # Set metadata
        fused.modalities_used = [m for m in encodings.keys()]
        fused.modal_weights = {m.value: self.config.modality_weights.get(m.value, 0.5)
                               for m in encodings.keys()}

        # Calculate overall confidence
        fused.overall_confidence = self._calculate_confidence(encodings)

        # Cache result
        if self.config.enable_caching:
            self._cache[cache_key] = fused
            if len(self._cache) > self.config.cache_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]

        return fused

    def _early_fusion(
        self,
        encodings: Dict[Modality, Dict[str, Any]],
    ) -> FusedRepresentation:
        """Early fusion - concatenate raw features."""
        fused = FusedRepresentation()

        # Concatenate embeddings
        all_embeddings = []
        for modality, encoding in encodings.items():
            if "embedding" in encoding:
                all_embeddings.extend(encoding["embedding"])

        fused.embedding = all_embeddings

        # Merge features
        for modality, encoding in encodings.items():
            if "features" in encoding:
                fused.features[modality.value] = encoding["features"]

        return fused

    def _late_fusion(
        self,
        encodings: Dict[Modality, Dict[str, Any]],
    ) -> FusedRepresentation:
        """Late fusion - weighted combination of modal outputs."""
        fused = FusedRepresentation()

        # Weighted average of embeddings
        weighted_embedding = None
        total_weight = 0.0

        for modality, encoding in encodings.items():
            weight = self.config.modality_weights.get(modality.value, 0.5)
            if "embedding" in encoding:
                emb = encoding["embedding"]
                if weighted_embedding is None:
                    weighted_embedding = [v * weight for v in emb]
                else:
                    for i, v in enumerate(emb):
                        if i < len(weighted_embedding):
                            weighted_embedding[i] += v * weight
                total_weight += weight

        if weighted_embedding and total_weight > 0:
            fused.embedding = [v / total_weight for v in weighted_embedding]

        # Merge features
        for modality, encoding in encodings.items():
            if "features" in encoding:
                fused.features[modality.value] = encoding["features"]

        return fused

    def _attention_fusion(
        self,
        encodings: Dict[Modality, Dict[str, Any]],
    ) -> FusedRepresentation:
        """Attention-based fusion - cross-modal attention."""
        fused = FusedRepresentation()

        # Compute attention weights based on feature importance
        attention_weights = {}
        for modality, encoding in encodings.items():
            # Simple importance score based on feature density
            features = encoding.get("features", {})
            score = len(features) * 0.1 + self.config.modality_weights.get(modality.value, 0.5)
            attention_weights[modality] = score

        # Normalize
        total = sum(attention_weights.values())
        if total > 0:
            attention_weights = {k: v / total for k, v in attention_weights.items()}

        # Apply attention-weighted fusion
        fused.modal_weights = {m.value: w for m, w in attention_weights.items()}

        # Weighted embedding combination
        weighted_embedding = None

        for modality, encoding in encodings.items():
            weight = attention_weights.get(modality, 0.5)
            if "embedding" in encoding:
                emb = encoding["embedding"]
                if weighted_embedding is None:
                    weighted_embedding = [v * weight for v in emb]
                else:
                    for i, v in enumerate(emb):
                        if i < len(weighted_embedding):
                            weighted_embedding[i] += v * weight

        fused.embedding = weighted_embedding

        # Merge features
        for modality, encoding in encodings.items():
            if "features" in encoding:
                fused.features[modality.value] = encoding["features"]

        return fused

    def _hierarchical_fusion(
        self,
        encodings: Dict[Modality, Dict[str, Any]],
    ) -> FusedRepresentation:
        """Hierarchical fusion - multiple levels."""
        # Level 1: Group similar modalities
        visual_modalities = [Modality.IMAGE, Modality.SCREEN, Modality.VIDEO]
        audio_modalities = [Modality.AUDIO]
        text_modalities = [Modality.TEXT]
        interaction_modalities = [Modality.GESTURE, Modality.KEYBOARD]

        groups = {
            "visual": {m: encodings[m] for m in visual_modalities if m in encodings},
            "audio": {m: encodings[m] for m in audio_modalities if m in encodings},
            "text": {m: encodings[m] for m in text_modalities if m in encodings},
            "interaction": {m: encodings[m] for m in interaction_modalities if m in encodings},
        }

        # Level 2: Fuse within groups
        group_fusions = {}
        for group_name, group_encodings in groups.items():
            if group_encodings:
                group_fusions[group_name] = self._late_fusion(group_encodings)

        # Level 3: Fuse across groups
        fused = FusedRepresentation()

        all_embeddings = []
        for group_name, group_fused in group_fusions.items():
            if group_fused.embedding:
                all_embeddings.extend(group_fused.embedding)
            fused.features[group_name] = group_fused.features

        fused.embedding = all_embeddings

        # Copy modality info
        for modality in encodings.keys():
            if "features" in encodings[modality]:
                fused.modality_confidences[modality.value] = 0.8

        return fused

    def _calculate_confidence(
        self,
        encodings: Dict[Modality, Dict[str, Any]],
    ) -> float:
        """Calculate overall confidence from encodings."""
        if not encodings:
            return 0.0

        confidences = []
        for modality, encoding in encodings.items():
            # Base confidence from weight
            conf = self.config.modality_weights.get(modality.value, 0.5)

            # Boost for feature richness
            features = encoding.get("features", {})
            if features:
                conf += min(len(features) * 0.05, 0.2)

            confidences.append(conf)

        return min(sum(confidences) / len(confidences), 1.0)

    def _make_cache_key(self, inputs: List[ModalityInput]) -> str:
        """Create cache key from inputs."""
        key_parts = []
        for inp in inputs:
            part = f"{inp.modality.value}:{inp.timestamp}"
            if isinstance(inp.data, str):
                part += f":{hashlib.md5(inp.data.encode()).hexdigest()[:8]}"
            key_parts.append(part)

        return hashlib.md5(":".join(key_parts).encode()).hexdigest()

    async def process_screen_sequence(
        self,
        frames: List[ScreenFrame],
        gestures: Optional[List[GestureEvent]] = None,
    ) -> FusedRepresentation:
        """
        Process a sequence of screen frames.

        Combines spatial and temporal reasoning for comprehensive understanding.
        """
        # Create temporal context
        context = TemporalContext()
        for frame in frames:
            context.add_frame(frame)

        if gestures:
            context.gestures = gestures

        # Create inputs
        inputs = []
        for frame in frames:
            inputs.append(ModalityInput(
                modality=Modality.SCREEN,
                data=frame,
                timestamp=frame.timestamp,
            ))

        if gestures:
            inputs.append(ModalityInput(
                modality=Modality.GESTURE,
                data=gestures,
            ))

        return await self.fuse(inputs, temporal_context=context)

    async def understand_screen(
        self,
        frame: ScreenFrame,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-level screen understanding.

        Args:
            frame: Screen frame to analyze
            query: Optional natural language query about the screen

        Returns:
            Understanding results
        """
        inputs = [
            ModalityInput(modality=Modality.SCREEN, data=frame),
        ]

        if query:
            inputs.append(ModalityInput(modality=Modality.TEXT, data=query))

        fused = await self.fuse(inputs)

        result = {
            "description": self._generate_screen_description(frame, fused),
            "elements": [
                {"role": e.role, "label": e.label, "bounds": e.bounds}
                for e in frame.ui_elements[:10]
            ],
            "active_app": frame.active_app,
            "cursor_position": frame.cursor_position,
            "spatial_relations": fused.element_relations[:10],
            "confidence": fused.overall_confidence,
        }

        # If query provided, find relevant elements
        if query and frame.ui_elements:
            elements = [UIElement(
                id=e.get("id", ""),
                role=e.get("role", ""),
                label=e.get("label", ""),
                bounds=tuple(e.get("bounds", (0, 0, 0, 0))),
            ) for e in result["elements"]]

            matches = self._spatial_reasoner.find_element_by_description(elements, query)
            result["query_matches"] = [
                {"role": m.role, "label": m.label, "bounds": m.bounds}
                for m in matches[:5]
            ]

        return result

    def _generate_screen_description(
        self,
        frame: ScreenFrame,
        fused: FusedRepresentation,
    ) -> str:
        """Generate natural language description of screen."""
        parts = []

        if frame.active_app:
            parts.append(f"Viewing {frame.active_app}")

        if frame.active_window:
            parts.append(f"in window '{frame.active_window}'")

        if frame.ui_elements:
            element_types = defaultdict(int)
            for elem in frame.ui_elements:
                element_types[elem.role] += 1

            type_str = ", ".join(f"{c} {t}s" for t, c in element_types.items() if c > 1)
            if type_str:
                parts.append(f"with {type_str}")

        return " ".join(parts) if parts else "Screen capture"

    def get_statistics(self) -> Dict[str, Any]:
        """Get fusion engine statistics."""
        return {
            "total_fusions": self._total_fusions,
            "modality_usage": dict(self._modality_usage),
            "encoders_loaded": list(self._encoders.keys()),
            "cache_size": len(self._cache),
            "config": {
                "fusion_strategy": self.config.fusion_strategy.value,
                "enabled_modalities": [m.value for m in self.config.enabled_modalities],
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_multimodal_engine(
    config: Optional[MultiModalConfig] = None,
    executor: Optional[Any] = None,
) -> MultiModalFusionEngine:
    """Factory function to create multi-modal fusion engine."""
    return MultiModalFusionEngine(config=config, executor=executor)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "Modality",
    "FusionStrategy",
    "TemporalMode",
    "SpatialRelation",
    # Data classes
    "ModalityInput",
    "UIElement",
    "ScreenFrame",
    "GestureEvent",
    "AudioSegment",
    "TemporalContext",
    "FusedRepresentation",
    "MultiModalConfig",
    # Encoders
    "ModalityEncoder",
    "TextEncoder",
    "ImageEncoder",
    "ScreenEncoder",
    "AudioEncoder",
    "GestureEncoder",
    # Reasoners
    "SpatialReasoner",
    "TemporalReasoner",
    # Engine
    "MultiModalFusionEngine",
    "create_multimodal_engine",
]
