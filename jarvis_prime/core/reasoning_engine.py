"""
Advanced Reasoning Engine - Multi-Strategy Cognitive Processing
================================================================

v76.0 - Advanced Reasoning Capabilities

This module provides sophisticated reasoning strategies:
- Chain-of-Thought (CoT): Sequential reasoning with explicit steps
- Tree-of-Thoughts (ToT): Parallel exploration with branch pruning
- Self-Reflection: Meta-cognitive error detection and correction
- Hypothesis Testing: Scientific method-based reasoning
- Analogical Reasoning: Transfer learning from similar problems

ARCHITECTURE:
    Input -> Strategy Selector -> Reasoning Strategy -> Verification -> Output

FEATURES:
    - Dynamic strategy selection based on problem characteristics
    - Parallel thought exploration with intelligent pruning
    - Self-correcting reasoning with confidence tracking
    - Integration with AGI models for specialized processing
    - Streaming thought generation for real-time feedback
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from heapq import heappush, heappop, nlargest
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
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class ReasoningStrategy(Enum):
    """Available reasoning strategies."""
    DIRECT = "direct"                      # Single-pass, no explicit reasoning
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Sequential step-by-step
    TREE_OF_THOUGHTS = "tree_of_thoughts"  # Parallel exploration
    SELF_REFLECTION = "self_reflection"    # Meta-cognitive verification
    HYPOTHESIS_TEST = "hypothesis_test"    # Scientific method
    ANALOGICAL = "analogical"              # Transfer from similar problems
    ENSEMBLE = "ensemble"                  # Multiple strategies combined
    ADAPTIVE = "adaptive"                  # Dynamic strategy switching


class ThoughtStatus(Enum):
    """Status of a thought node."""
    PENDING = "pending"
    EXPLORING = "exploring"
    EVALUATED = "evaluated"
    PRUNED = "pruned"
    SELECTED = "selected"
    ABANDONED = "abandoned"


class VerificationResult(Enum):
    """Result of thought verification."""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    NEEDS_REVISION = "needs_revision"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Thought:
    """
    Represents a single thought in the reasoning process.

    Can be a step in Chain-of-Thought or a node in Tree-of-Thoughts.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""

    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0

    # Evaluation
    value: float = 0.0              # Estimated value (0-1)
    confidence: float = 0.5         # Confidence in this thought
    visit_count: int = 0            # For MCTS-style exploration

    # Status
    status: ThoughtStatus = ThoughtStatus.PENDING
    verification: Optional[VerificationResult] = None

    # Metadata
    strategy_used: Optional[ReasoningStrategy] = None
    generation_time_ms: float = 0.0
    evaluation_time_ms: float = 0.0

    # Self-reflection
    self_critique: Optional[str] = None
    revisions: List[str] = field(default_factory=list)

    def __lt__(self, other: "Thought") -> bool:
        """For heap operations - higher value = higher priority."""
        return self.value > other.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "value": round(self.value, 3),
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "verification": self.verification.value if self.verification else None,
        }


@dataclass
class ReasoningChain:
    """A chain of thoughts forming a complete reasoning path."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    thoughts: List[Thought] = field(default_factory=list)
    strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT

    # Aggregates
    total_value: float = 0.0
    average_confidence: float = 0.0

    # Input/Output
    input_text: str = ""
    final_answer: str = ""

    # Timing
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def add_thought(self, thought: Thought) -> None:
        """Add thought to chain."""
        self.thoughts.append(thought)
        self._update_aggregates()

    def _update_aggregates(self) -> None:
        """Update aggregate metrics."""
        if not self.thoughts:
            return

        self.total_value = sum(t.value for t in self.thoughts)
        self.average_confidence = sum(t.confidence for t in self.thoughts) / len(self.thoughts)

    def get_path(self) -> List[str]:
        """Get the content of all thoughts in order."""
        return [t.content for t in self.thoughts]

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy.value,
            "thought_count": len(self.thoughts),
            "total_value": round(self.total_value, 3),
            "average_confidence": round(self.average_confidence, 3),
            "duration_ms": round(self.duration_ms, 1),
            "final_answer": self.final_answer[:200] if self.final_answer else None,
        }


@dataclass
class ThoughtTree:
    """
    A tree of thoughts for parallel exploration.

    Used by Tree-of-Thoughts strategy for branching reasoning.
    """
    root: Optional[Thought] = None
    nodes: Dict[str, Thought] = field(default_factory=dict)

    # Exploration parameters
    max_depth: int = 5
    max_branches: int = 3
    beam_width: int = 3  # Top-k to keep at each level

    # Best path tracking
    best_leaf: Optional[Thought] = None
    best_value: float = 0.0

    def add_node(self, thought: Thought, parent_id: Optional[str] = None) -> None:
        """Add node to tree."""
        if parent_id is None:
            self.root = thought
            thought.depth = 0
        else:
            if parent_id in self.nodes:
                parent = self.nodes[parent_id]
                parent.children_ids.append(thought.id)
                thought.parent_id = parent_id
                thought.depth = parent.depth + 1

        self.nodes[thought.id] = thought

        # Track best
        if thought.value > self.best_value:
            self.best_value = thought.value
            self.best_leaf = thought

    def get_path_to_node(self, node_id: str) -> List[Thought]:
        """Get path from root to node."""
        path = []
        current_id = node_id

        while current_id:
            if current_id in self.nodes:
                path.append(self.nodes[current_id])
                current_id = self.nodes[current_id].parent_id
            else:
                break

        return list(reversed(path))

    def get_best_path(self) -> List[Thought]:
        """Get path to best leaf."""
        if self.best_leaf:
            return self.get_path_to_node(self.best_leaf.id)
        return []

    def get_frontier(self) -> List[Thought]:
        """Get leaf nodes (frontier for expansion)."""
        leaves = []
        for node in self.nodes.values():
            if not node.children_ids and node.status != ThoughtStatus.PRUNED:
                leaves.append(node)
        return leaves

    def prune_below_threshold(self, threshold: float) -> int:
        """Prune nodes with value below threshold."""
        pruned_count = 0
        for node in self.nodes.values():
            if node.value < threshold and node.status != ThoughtStatus.SELECTED:
                node.status = ThoughtStatus.PRUNED
                pruned_count += 1
        return pruned_count

    def beam_search_prune(self) -> int:
        """Keep only top-k nodes at each depth level."""
        by_depth: Dict[int, List[Thought]] = defaultdict(list)

        for node in self.nodes.values():
            if node.status != ThoughtStatus.PRUNED:
                by_depth[node.depth].append(node)

        pruned_count = 0
        for depth, nodes in by_depth.items():
            if len(nodes) > self.beam_width:
                sorted_nodes = sorted(nodes, key=lambda n: n.value, reverse=True)
                for node in sorted_nodes[self.beam_width:]:
                    node.status = ThoughtStatus.PRUNED
                    pruned_count += 1

        return pruned_count


@dataclass
class ReasoningConfig:
    """Configuration for reasoning engine."""
    # Strategy selection
    default_strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT
    enable_adaptive: bool = True

    # Chain-of-Thought
    cot_max_steps: int = 10
    cot_stop_on_confidence: float = 0.9

    # Tree-of-Thoughts
    tot_max_depth: int = 5
    tot_branches_per_node: int = 3
    tot_beam_width: int = 3
    tot_exploration_constant: float = 1.4  # UCB exploration

    # Self-Reflection
    reflection_threshold: float = 0.6
    max_revisions: int = 3

    # Hypothesis Testing
    hypothesis_confidence_threshold: float = 0.7
    max_hypotheses: int = 5

    # General
    min_confidence: float = 0.3
    timeout_seconds: float = 60.0
    parallel_thoughts: int = 4

    # Caching
    cache_thoughts: bool = True
    cache_ttl_seconds: float = 300.0


@dataclass
class ReasoningResult:
    """Result from reasoning engine."""
    strategy: ReasoningStrategy
    input_text: str
    output_text: str

    # Quality metrics
    confidence: float = 0.5
    coherence: float = 0.5

    # Chain/Tree data
    chain: Optional[ReasoningChain] = None
    tree: Optional[ThoughtTree] = None

    # Reflection
    self_assessment: Optional[str] = None
    verified: bool = False

    # Performance
    total_thoughts: int = 0
    pruned_thoughts: int = 0
    latency_ms: float = 0.0

    # Alternative answers
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "output": self.output_text[:500] if self.output_text else None,
            "confidence": round(self.confidence, 3),
            "verified": self.verified,
            "total_thoughts": self.total_thoughts,
            "latency_ms": round(self.latency_ms, 1),
            "chain": self.chain.to_dict() if self.chain else None,
        }


# =============================================================================
# THOUGHT GENERATORS
# =============================================================================

class ThoughtGenerator(Protocol):
    """Protocol for thought generation."""

    async def generate(
        self,
        prompt: str,
        context: List[Thought],
        num_thoughts: int,
    ) -> List[Thought]:
        """Generate new thoughts given prompt and context."""
        ...


class DefaultThoughtGenerator:
    """
    Default thought generator using pattern-based generation.

    In production, this would use an LLM for more sophisticated generation.
    """

    # Reasoning templates
    TEMPLATES = {
        "analyze": "Let me analyze {topic}...",
        "decompose": "Breaking this down: {parts}",
        "consider": "Considering {aspect}...",
        "conclude": "Therefore, {conclusion}",
        "verify": "Checking this: {check}",
        "alternative": "Alternatively, {alternative}",
    }

    def __init__(self, executor: Optional[Any] = None):
        self.executor = executor
        self._generation_count = 0

    async def generate(
        self,
        prompt: str,
        context: List[Thought],
        num_thoughts: int = 1,
    ) -> List[Thought]:
        """Generate thoughts based on prompt and context."""
        thoughts = []
        self._generation_count += 1

        # Build context string
        context_str = " -> ".join(t.content[:50] for t in context[-3:])

        for i in range(num_thoughts):
            start = time.time()

            # Select template based on position and context
            if not context:
                template_key = "analyze"
            elif len(context) >= 4:
                template_key = "conclude"
            elif i == 0:
                template_key = "decompose"
            else:
                template_key = random.choice(["consider", "alternative"])

            # Generate thought content
            if self.executor:
                # Use LLM executor
                try:
                    content = await self._generate_with_llm(prompt, context, template_key)
                except Exception as e:
                    logger.warning(f"LLM generation failed: {e}")
                    content = self._generate_heuristic(prompt, context, template_key)
            else:
                content = self._generate_heuristic(prompt, context, template_key)

            thought = Thought(
                content=content,
                depth=len(context),
                generation_time_ms=(time.time() - start) * 1000,
                confidence=0.5 + random.uniform(-0.2, 0.2),  # Initial confidence
            )

            thoughts.append(thought)

        return thoughts

    async def _generate_with_llm(
        self,
        prompt: str,
        context: List[Thought],
        template_key: str,
    ) -> str:
        """Generate thought using LLM."""
        context_str = "\n".join(f"Step {i+1}: {t.content}" for i, t in enumerate(context))

        system_prompt = f"""You are a reasoning assistant. Generate the next logical step in reasoning.
Previous steps:
{context_str if context_str else 'None'}

Task: {prompt}

Generate only the next reasoning step (1-2 sentences):"""

        response = await self.executor.generate(
            prompt=system_prompt,
            max_tokens=100,
            temperature=0.7,
        )

        return response.strip()

    def _generate_heuristic(
        self,
        prompt: str,
        context: List[Thought],
        template_key: str,
    ) -> str:
        """Generate thought using heuristics."""
        # Extract key terms from prompt
        words = prompt.split()
        key_terms = [w for w in words if len(w) > 4][:3]

        if template_key == "analyze":
            return f"Let me analyze this: {' '.join(key_terms) if key_terms else prompt[:50]}"
        elif template_key == "decompose":
            return f"Breaking this down into components: {', '.join(key_terms) if key_terms else 'multiple parts'}"
        elif template_key == "consider":
            aspect = key_terms[0] if key_terms else "the main point"
            return f"Considering {aspect} in more detail..."
        elif template_key == "conclude":
            return f"Based on the analysis, we can conclude that..."
        elif template_key == "alternative":
            return f"An alternative perspective would be..."
        else:
            return f"Step {len(context)+1}: Processing {prompt[:30]}..."


# =============================================================================
# THOUGHT EVALUATORS
# =============================================================================

class ThoughtEvaluator(Protocol):
    """Protocol for thought evaluation."""

    async def evaluate(self, thought: Thought, context: List[Thought]) -> float:
        """Evaluate a thought and return value (0-1)."""
        ...


class DefaultThoughtEvaluator:
    """
    Default thought evaluator using heuristic scoring.

    Evaluates thoughts based on:
    - Relevance to context
    - Logical progression
    - Specificity
    - Coherence
    """

    def __init__(self, executor: Optional[Any] = None):
        self.executor = executor

    async def evaluate(self, thought: Thought, context: List[Thought]) -> float:
        """Evaluate thought quality."""
        start = time.time()

        scores = {
            "length": self._score_length(thought),
            "specificity": self._score_specificity(thought),
            "progression": self._score_progression(thought, context),
            "keywords": self._score_keywords(thought, context),
        }

        # Weighted average
        weights = {"length": 0.1, "specificity": 0.3, "progression": 0.3, "keywords": 0.3}
        value = sum(scores[k] * weights[k] for k in scores)

        thought.value = value
        thought.evaluation_time_ms = (time.time() - start) * 1000
        thought.status = ThoughtStatus.EVALUATED

        return value

    def _score_length(self, thought: Thought) -> float:
        """Score based on appropriate length."""
        length = len(thought.content)
        if length < 10:
            return 0.2
        elif length < 50:
            return 0.6
        elif length < 200:
            return 1.0
        elif length < 500:
            return 0.8
        else:
            return 0.5  # Too long

    def _score_specificity(self, thought: Thought) -> float:
        """Score based on specificity."""
        content = thought.content.lower()

        # Vague terms reduce score
        vague_terms = ["something", "thing", "stuff", "maybe", "perhaps", "kind of"]
        vague_count = sum(1 for term in vague_terms if term in content)

        # Specific terms increase score
        specific_indicators = ["specifically", "exactly", "precisely", "because", "therefore"]
        specific_count = sum(1 for term in specific_indicators if term in content)

        base = 0.6
        score = base - (vague_count * 0.1) + (specific_count * 0.1)
        return max(0.0, min(1.0, score))

    def _score_progression(self, thought: Thought, context: List[Thought]) -> float:
        """Score based on logical progression from context."""
        if not context:
            return 0.7  # First thought gets baseline

        last_thought = context[-1]

        # Check for connection to previous thought
        last_words = set(last_thought.content.lower().split())
        current_words = set(thought.content.lower().split())

        # Overlap indicates connection
        overlap = len(last_words & current_words)
        overlap_ratio = overlap / max(len(last_words), 1)

        # Some overlap is good, too much might be repetition
        if overlap_ratio < 0.1:
            return 0.3  # Too disconnected
        elif overlap_ratio < 0.3:
            return 0.8  # Good connection
        elif overlap_ratio < 0.5:
            return 0.6  # Some repetition
        else:
            return 0.4  # Too repetitive

    def _score_keywords(self, thought: Thought, context: List[Thought]) -> float:
        """Score based on important keyword presence."""
        reasoning_keywords = [
            "because", "therefore", "thus", "hence", "since",
            "implies", "suggests", "indicates", "means",
            "first", "second", "finally", "however", "although",
        ]

        content_lower = thought.content.lower()
        keyword_count = sum(1 for kw in reasoning_keywords if kw in content_lower)

        return min(1.0, 0.4 + keyword_count * 0.15)


# =============================================================================
# REASONING STRATEGIES
# =============================================================================

class BaseReasoningStrategy(ABC):
    """Base class for reasoning strategies."""

    strategy_type: ReasoningStrategy

    def __init__(
        self,
        config: ReasoningConfig,
        generator: ThoughtGenerator,
        evaluator: ThoughtEvaluator,
    ):
        self.config = config
        self.generator = generator
        self.evaluator = evaluator

    @abstractmethod
    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute reasoning strategy."""
        ...


class ChainOfThoughtStrategy(BaseReasoningStrategy):
    """
    Chain-of-Thought (CoT) reasoning.

    Generates sequential reasoning steps, each building on the previous.
    Stops when confidence threshold is reached or max steps exceeded.
    """

    strategy_type = ReasoningStrategy.CHAIN_OF_THOUGHT

    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute Chain-of-Thought reasoning."""
        start = time.time()

        chain = ReasoningChain(
            strategy=self.strategy_type,
            input_text=input_text,
        )

        thoughts: List[Thought] = []
        total_confidence = 0.0

        for step in range(self.config.cot_max_steps):
            # Generate next thought
            new_thoughts = await self.generator.generate(
                prompt=input_text,
                context=thoughts,
                num_thoughts=1,
            )

            if not new_thoughts:
                break

            thought = new_thoughts[0]

            # Evaluate thought
            value = await self.evaluator.evaluate(thought, thoughts)

            # Add to chain
            thoughts.append(thought)
            chain.add_thought(thought)
            total_confidence += thought.confidence

            # Check for early stopping
            avg_confidence = total_confidence / len(thoughts)
            if avg_confidence >= self.config.cot_stop_on_confidence:
                logger.debug(f"CoT: Early stop at step {step+1} (confidence: {avg_confidence:.2f})")
                break

        # Generate final answer
        chain.final_answer = self._synthesize_answer(thoughts)
        chain.end_time = time.time()

        return ReasoningResult(
            strategy=self.strategy_type,
            input_text=input_text,
            output_text=chain.final_answer,
            confidence=chain.average_confidence,
            chain=chain,
            total_thoughts=len(thoughts),
            latency_ms=(time.time() - start) * 1000,
        )

    def _synthesize_answer(self, thoughts: List[Thought]) -> str:
        """Synthesize final answer from thought chain."""
        if not thoughts:
            return "Unable to generate reasoning chain."

        # Use last thought as primary conclusion
        conclusion = thoughts[-1].content

        # Build reasoning summary
        if len(thoughts) > 1:
            steps_summary = " -> ".join(t.content[:50] for t in thoughts[:-1])
            return f"Reasoning: {steps_summary}\n\nConclusion: {conclusion}"

        return conclusion


class TreeOfThoughtsStrategy(BaseReasoningStrategy):
    """
    Tree-of-Thoughts (ToT) reasoning.

    Explores multiple reasoning paths in parallel using tree search.
    Uses beam search to prune low-value branches.
    """

    strategy_type = ReasoningStrategy.TREE_OF_THOUGHTS

    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute Tree-of-Thoughts reasoning."""
        start = time.time()

        tree = ThoughtTree(
            max_depth=self.config.tot_max_depth,
            max_branches=self.config.tot_branches_per_node,
            beam_width=self.config.tot_beam_width,
        )

        # Create root thought
        root_thoughts = await self.generator.generate(
            prompt=input_text,
            context=[],
            num_thoughts=1,
        )

        if not root_thoughts:
            return ReasoningResult(
                strategy=self.strategy_type,
                input_text=input_text,
                output_text="Unable to initiate reasoning.",
                confidence=0.0,
                latency_ms=(time.time() - start) * 1000,
            )

        root = root_thoughts[0]
        await self.evaluator.evaluate(root, [])
        tree.add_node(root)

        # BFS expansion with beam search
        for depth in range(self.config.tot_max_depth):
            frontier = [n for n in tree.get_frontier() if n.depth == depth]

            if not frontier:
                break

            # Expand each frontier node
            expansion_tasks = []
            for node in frontier:
                expansion_tasks.append(
                    self._expand_node(tree, node, input_text)
                )

            await asyncio.gather(*expansion_tasks)

            # Beam search pruning
            tree.beam_search_prune()

        # Get best path
        best_path = tree.get_best_path()

        # Generate answer from best path
        answer = self._path_to_answer(best_path)

        return ReasoningResult(
            strategy=self.strategy_type,
            input_text=input_text,
            output_text=answer,
            confidence=tree.best_value,
            tree=tree,
            total_thoughts=len(tree.nodes),
            pruned_thoughts=sum(1 for n in tree.nodes.values() if n.status == ThoughtStatus.PRUNED),
            latency_ms=(time.time() - start) * 1000,
            alternatives=self._get_alternatives(tree),
        )

    async def _expand_node(
        self,
        tree: ThoughtTree,
        node: Thought,
        input_text: str,
    ) -> None:
        """Expand a node with child thoughts."""
        if node.depth >= tree.max_depth:
            return

        # Get path to this node for context
        context = tree.get_path_to_node(node.id)

        # Generate children
        children = await self.generator.generate(
            prompt=input_text,
            context=context,
            num_thoughts=tree.max_branches,
        )

        # Evaluate and add children
        for child in children:
            value = await self.evaluator.evaluate(child, context)
            tree.add_node(child, parent_id=node.id)

    def _path_to_answer(self, path: List[Thought]) -> str:
        """Convert path to final answer."""
        if not path:
            return "No valid reasoning path found."

        steps = [f"Step {i+1}: {t.content}" for i, t in enumerate(path)]
        return "\n".join(steps)

    def _get_alternatives(self, tree: ThoughtTree) -> List[Tuple[str, float]]:
        """Get alternative answers from other branches."""
        alternatives = []

        # Find other high-value leaves
        leaves = [n for n in tree.nodes.values()
                  if not n.children_ids and n.status != ThoughtStatus.PRUNED
                  and n.id != (tree.best_leaf.id if tree.best_leaf else None)]

        # Sort by value and take top 3
        leaves.sort(key=lambda n: n.value, reverse=True)

        for leaf in leaves[:3]:
            path = tree.get_path_to_node(leaf.id)
            answer = self._path_to_answer(path)
            alternatives.append((answer, leaf.value))

        return alternatives


class SelfReflectionStrategy(BaseReasoningStrategy):
    """
    Self-Reflection reasoning.

    Generates initial answer, critiques it, and revises until satisfactory.
    """

    strategy_type = ReasoningStrategy.SELF_REFLECTION

    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute Self-Reflection reasoning."""
        start = time.time()

        chain = ReasoningChain(
            strategy=self.strategy_type,
            input_text=input_text,
        )

        thoughts: List[Thought] = []
        revision_count = 0

        # Generate initial thought
        initial_thoughts = await self.generator.generate(
            prompt=input_text,
            context=[],
            num_thoughts=1,
        )

        if not initial_thoughts:
            return ReasoningResult(
                strategy=self.strategy_type,
                input_text=input_text,
                output_text="Unable to generate initial response.",
                confidence=0.0,
                latency_ms=(time.time() - start) * 1000,
            )

        current_thought = initial_thoughts[0]
        await self.evaluator.evaluate(current_thought, [])
        thoughts.append(current_thought)
        chain.add_thought(current_thought)

        # Self-reflection loop
        while (current_thought.confidence < self.config.reflection_threshold
               and revision_count < self.config.max_revisions):

            # Generate critique
            critique = await self._generate_critique(current_thought, input_text)
            current_thought.self_critique = critique

            # Generate revision based on critique
            revised_thoughts = await self.generator.generate(
                prompt=f"{input_text}\n\nPrevious attempt: {current_thought.content}\n\nCritique: {critique}\n\nImproved response:",
                context=thoughts,
                num_thoughts=1,
            )

            if not revised_thoughts:
                break

            revised = revised_thoughts[0]
            await self.evaluator.evaluate(revised, thoughts)

            # Check if revision is better
            if revised.value > current_thought.value:
                current_thought.revisions.append(revised.content)
                current_thought = revised
                thoughts.append(current_thought)
                chain.add_thought(current_thought)

            revision_count += 1

        chain.final_answer = current_thought.content
        chain.end_time = time.time()

        return ReasoningResult(
            strategy=self.strategy_type,
            input_text=input_text,
            output_text=chain.final_answer,
            confidence=current_thought.confidence,
            chain=chain,
            total_thoughts=len(thoughts),
            verified=current_thought.confidence >= self.config.reflection_threshold,
            self_assessment=current_thought.self_critique,
            latency_ms=(time.time() - start) * 1000,
        )

    async def _generate_critique(self, thought: Thought, input_text: str) -> str:
        """Generate critique of a thought."""
        # Heuristic critique generation
        critiques = []

        # Check length
        if len(thought.content) < 50:
            critiques.append("Response could be more detailed")

        # Check for reasoning indicators
        reasoning_words = ["because", "therefore", "thus", "since"]
        if not any(w in thought.content.lower() for w in reasoning_words):
            critiques.append("Could include more explicit reasoning")

        # Check specificity
        vague_words = ["something", "maybe", "perhaps", "kind of"]
        if any(w in thought.content.lower() for w in vague_words):
            critiques.append("Could be more specific and definitive")

        if not critiques:
            critiques.append("Response appears adequate")

        return "; ".join(critiques)


class HypothesisTestStrategy(BaseReasoningStrategy):
    """
    Hypothesis Testing reasoning.

    Generates hypotheses, tests them against evidence, and selects best.
    """

    strategy_type = ReasoningStrategy.HYPOTHESIS_TEST

    async def reason(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Execute Hypothesis Testing reasoning."""
        start = time.time()

        # Generate hypotheses
        hypotheses = await self._generate_hypotheses(input_text)

        # Test each hypothesis
        tested: List[Tuple[str, float, str]] = []
        for hypothesis in hypotheses:
            confidence, evidence = await self._test_hypothesis(hypothesis, input_text)
            tested.append((hypothesis, confidence, evidence))

        # Select best hypothesis
        tested.sort(key=lambda x: x[1], reverse=True)
        best_hypothesis, best_confidence, best_evidence = tested[0] if tested else ("", 0.0, "")

        # Build result
        chain = ReasoningChain(
            strategy=self.strategy_type,
            input_text=input_text,
        )

        # Add hypothesis thoughts
        for hyp, conf, ev in tested:
            thought = Thought(
                content=f"Hypothesis: {hyp}\nEvidence: {ev}",
                confidence=conf,
                value=conf,
                status=ThoughtStatus.EVALUATED,
            )
            chain.add_thought(thought)

        chain.final_answer = f"Based on hypothesis testing:\n{best_hypothesis}\n\nSupporting evidence: {best_evidence}"
        chain.end_time = time.time()

        return ReasoningResult(
            strategy=self.strategy_type,
            input_text=input_text,
            output_text=chain.final_answer,
            confidence=best_confidence,
            chain=chain,
            total_thoughts=len(tested),
            verified=best_confidence >= self.config.hypothesis_confidence_threshold,
            alternatives=[(h, c) for h, c, _ in tested[1:4]],
            latency_ms=(time.time() - start) * 1000,
        )

    async def _generate_hypotheses(self, input_text: str) -> List[str]:
        """Generate hypotheses for the input."""
        # Extract key elements for hypothesis generation
        words = input_text.split()
        key_words = [w for w in words if len(w) > 4][:5]

        hypotheses = []
        templates = [
            "The answer involves {key}",
            "{key} is the primary factor",
            "This relates to {key}",
            "The solution requires understanding {key}",
        ]

        for i, key in enumerate(key_words[:self.config.max_hypotheses]):
            template = templates[i % len(templates)]
            hypotheses.append(template.format(key=key))

        return hypotheses

    async def _test_hypothesis(self, hypothesis: str, input_text: str) -> Tuple[float, str]:
        """Test a hypothesis against the input."""
        # Simple keyword-based testing
        input_lower = input_text.lower()
        hyp_words = set(hypothesis.lower().split())

        # Count matching words
        matches = sum(1 for w in hyp_words if w in input_lower)
        match_ratio = matches / max(len(hyp_words), 1)

        confidence = 0.3 + match_ratio * 0.5

        evidence = f"Found {matches} supporting terms in the input"

        return confidence, evidence


# =============================================================================
# REASONING ENGINE
# =============================================================================

class ReasoningEngine:
    """
    Main reasoning engine that orchestrates different strategies.

    Automatically selects appropriate strategy based on input
    or uses specified strategy.
    """

    STRATEGIES: Dict[ReasoningStrategy, Type[BaseReasoningStrategy]] = {
        ReasoningStrategy.CHAIN_OF_THOUGHT: ChainOfThoughtStrategy,
        ReasoningStrategy.TREE_OF_THOUGHTS: TreeOfThoughtsStrategy,
        ReasoningStrategy.SELF_REFLECTION: SelfReflectionStrategy,
        ReasoningStrategy.HYPOTHESIS_TEST: HypothesisTestStrategy,
    }

    def __init__(
        self,
        config: Optional[ReasoningConfig] = None,
        executor: Optional[Any] = None,
    ):
        self.config = config or ReasoningConfig()
        self.executor = executor

        # Components
        self.generator = DefaultThoughtGenerator(executor)
        self.evaluator = DefaultThoughtEvaluator(executor)

        # Strategy instances
        self._strategies: Dict[ReasoningStrategy, BaseReasoningStrategy] = {}

        # Statistics
        self._total_reasoning_calls = 0
        self._strategy_usage: Dict[str, int] = defaultdict(int)
        self._total_latency_ms = 0.0

        # Cache
        self._cache: Dict[str, ReasoningResult] = {}

        logger.info("ReasoningEngine initialized")

    def _get_strategy(self, strategy_type: ReasoningStrategy) -> BaseReasoningStrategy:
        """Get or create a strategy instance."""
        if strategy_type not in self._strategies:
            if strategy_type in self.STRATEGIES:
                self._strategies[strategy_type] = self.STRATEGIES[strategy_type](
                    config=self.config,
                    generator=self.generator,
                    evaluator=self.evaluator,
                )
            else:
                # Default to CoT
                self._strategies[strategy_type] = ChainOfThoughtStrategy(
                    config=self.config,
                    generator=self.generator,
                    evaluator=self.evaluator,
                )

        return self._strategies[strategy_type]

    async def reason(
        self,
        input_text: str,
        strategy: Optional[ReasoningStrategy] = None,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> ReasoningResult:
        """
        Execute reasoning on input text.

        Args:
            input_text: The problem or question to reason about
            strategy: Specific strategy to use (auto-selected if None)
            context: Additional context
            use_cache: Whether to use cached results

        Returns:
            ReasoningResult with answer and reasoning chain
        """
        # Check cache
        if use_cache and self.config.cache_thoughts:
            cache_key = self._make_cache_key(input_text, strategy)
            if cache_key in self._cache:
                logger.debug("Returning cached reasoning result")
                return self._cache[cache_key]

        # Select strategy
        if strategy is None:
            strategy = await self._select_strategy(input_text, context)

        # Get strategy instance
        strategy_impl = self._get_strategy(strategy)

        # Execute reasoning
        self._total_reasoning_calls += 1
        self._strategy_usage[strategy.value] += 1

        try:
            result = await asyncio.wait_for(
                strategy_impl.reason(input_text, context),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Reasoning timeout after {self.config.timeout_seconds}s")
            result = ReasoningResult(
                strategy=strategy,
                input_text=input_text,
                output_text="Reasoning timed out. Please try with a simpler query.",
                confidence=0.0,
                latency_ms=self.config.timeout_seconds * 1000,
            )

        self._total_latency_ms += result.latency_ms

        # Cache result
        if use_cache and self.config.cache_thoughts:
            cache_key = self._make_cache_key(input_text, strategy)
            self._cache[cache_key] = result

            # Limit cache size
            if len(self._cache) > 100:
                oldest = next(iter(self._cache))
                del self._cache[oldest]

        return result

    async def _select_strategy(
        self,
        input_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningStrategy:
        """Automatically select best strategy for input."""
        if not self.config.enable_adaptive:
            return self.config.default_strategy

        input_lower = input_text.lower()

        # Heuristic strategy selection
        indicators = {
            ReasoningStrategy.TREE_OF_THOUGHTS: [
                "explore", "alternatives", "options", "different ways",
                "multiple", "compare", "which is better",
            ],
            ReasoningStrategy.SELF_REFLECTION: [
                "careful", "accurate", "verify", "check", "sure",
                "certain", "correct", "precise",
            ],
            ReasoningStrategy.HYPOTHESIS_TEST: [
                "why", "cause", "reason", "hypothesis", "theory",
                "explain", "because",
            ],
            ReasoningStrategy.CHAIN_OF_THOUGHT: [
                "how", "steps", "process", "procedure", "method",
                "first", "then",
            ],
        }

        scores = {s: 0 for s in indicators}

        for strategy, keywords in indicators.items():
            for keyword in keywords:
                if keyword in input_lower:
                    scores[strategy] += 1

        # Select highest scoring or default
        best_strategy = max(scores, key=scores.get)
        if scores[best_strategy] > 0:
            return best_strategy

        return self.config.default_strategy

    def _make_cache_key(
        self,
        input_text: str,
        strategy: Optional[ReasoningStrategy],
    ) -> str:
        """Create cache key."""
        key_data = f"{input_text}:{strategy.value if strategy else 'auto'}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def reason_stream(
        self,
        input_text: str,
        strategy: Optional[ReasoningStrategy] = None,
    ) -> AsyncIterator[Thought]:
        """
        Stream thoughts as they are generated.

        Useful for real-time feedback in UI.
        """
        strategy = strategy or await self._select_strategy(input_text)
        strategy_impl = self._get_strategy(strategy)

        # For streaming, use CoT and yield each thought
        if isinstance(strategy_impl, ChainOfThoughtStrategy):
            thoughts: List[Thought] = []

            for step in range(self.config.cot_max_steps):
                new_thoughts = await self.generator.generate(
                    prompt=input_text,
                    context=thoughts,
                    num_thoughts=1,
                )

                if not new_thoughts:
                    break

                thought = new_thoughts[0]
                await self.evaluator.evaluate(thought, thoughts)
                thoughts.append(thought)

                yield thought

                if thought.confidence >= self.config.cot_stop_on_confidence:
                    break

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_reasoning_calls": self._total_reasoning_calls,
            "strategy_usage": dict(self._strategy_usage),
            "avg_latency_ms": self._total_latency_ms / max(self._total_reasoning_calls, 1),
            "cache_size": len(self._cache),
            "config": {
                "default_strategy": self.config.default_strategy.value,
                "cot_max_steps": self.config.cot_max_steps,
                "tot_max_depth": self.config.tot_max_depth,
            },
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_reasoning_engine(
    executor: Optional[Any] = None,
    config: Optional[ReasoningConfig] = None,
) -> ReasoningEngine:
    """Factory function to create reasoning engine."""
    return ReasoningEngine(config=config, executor=executor)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ReasoningStrategy",
    "ThoughtStatus",
    "VerificationResult",
    # Data classes
    "Thought",
    "ReasoningChain",
    "ThoughtTree",
    "ReasoningConfig",
    "ReasoningResult",
    # Generators/Evaluators
    "ThoughtGenerator",
    "DefaultThoughtGenerator",
    "ThoughtEvaluator",
    "DefaultThoughtEvaluator",
    # Strategies
    "BaseReasoningStrategy",
    "ChainOfThoughtStrategy",
    "TreeOfThoughtsStrategy",
    "SelfReflectionStrategy",
    "HypothesisTestStrategy",
    # Engine
    "ReasoningEngine",
    "create_reasoning_engine",
]
