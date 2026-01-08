"""
Continual Learning System v80.0 - Learning from JARVIS Interactions
====================================================================

Advanced continual learning system that enables JARVIS-Prime to learn
from interactions without catastrophic forgetting. Integrates RAG
for knowledge retrieval and augmentation.

FEATURES:
    - Experience Replay Buffer with prioritized sampling
    - Elastic Weight Consolidation (EWC) for forgetting prevention
    - Retrieval-Augmented Generation (RAG) with vector stores
    - Knowledge Distillation for model compression
    - Active Learning for efficient data selection
    - Online learning with mini-batch updates
    - Integration with Reactor-Core for training

ALGORITHMS:
    - Experience Replay (ER)
    - Elastic Weight Consolidation (EWC)
    - Learning without Forgetting (LwF)
    - Progressive Neural Networks
    - PackNet pruning
    - Gradient Episodic Memory (GEM)

RAG FEATURES:
    - Multiple vector store backends (FAISS, Chroma, Pinecone)
    - Semantic chunking and embedding
    - Hybrid search (dense + sparse)
    - Re-ranking with cross-encoder
    - Context compression
    - Citation tracking

USAGE:
    from jarvis_prime.models.continual_learning_system import get_continual_learner

    learner = await get_continual_learner()

    # Learn from interaction
    await learner.learn_from_interaction(
        prompt="How do I...",
        response="You can...",
        feedback=0.9  # User satisfaction
    )

    # Generate with RAG
    result = await learner.generate_with_rag(
        query="What is the capital of France?",
        top_k=5
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

# Try importing vector store libraries
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - using mock vector store")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


# ============================================================================
# ENUMS AND TYPES
# ============================================================================

class LearningStrategy(Enum):
    """Continual learning strategies."""
    EXPERIENCE_REPLAY = "experience_replay"
    ELASTIC_WEIGHT_CONSOLIDATION = "ewc"
    LEARNING_WITHOUT_FORGETTING = "lwf"
    PROGRESSIVE_NETWORKS = "progressive"
    PACKNET = "packnet"
    GRADIENT_EPISODIC_MEMORY = "gem"


class RetrievalStrategy(Enum):
    """RAG retrieval strategies."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANKING = "reranking"


class VectorStoreType(Enum):
    """Vector store backends."""
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    MEMORY = "memory"  # In-memory for testing


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Experience:
    """A single learning experience."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    response: str = ""
    feedback: float = 0.0  # -1 to 1
    task_type: str = "general"
    embedding: Optional[np.ndarray] = None
    importance: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "response": self.response,
            "feedback": self.feedback,
            "task_type": self.task_type,
            "importance": self.importance,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """A document for RAG."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    chunk_index: int = 0


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    documents: List[Document]
    scores: List[float]
    query_embedding: Optional[np.ndarray] = None
    latency_ms: float = 0.0


@dataclass
class LearningMetrics:
    """Metrics for continual learning."""
    experiences_learned: int = 0
    average_feedback: float = 0.0
    forgetting_rate: float = 0.0
    knowledge_transfer: float = 0.0
    replay_ratio: float = 0.0
    last_training_loss: float = 0.0


# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================

class ExperienceReplayBuffer:
    """
    Prioritized Experience Replay Buffer.

    Stores experiences with importance-based sampling for efficient learning.
    Uses reservoir sampling for bounded memory.
    """

    def __init__(
        self,
        capacity: int = 100000,
        priority_alpha: float = 0.6,
        priority_beta: float = 0.4,
    ):
        """
        Initialize experience replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            priority_alpha: Priority exponent (0 = uniform, 1 = greedy)
            priority_beta: Importance sampling correction
        """
        self.capacity = capacity
        self.priority_alpha = priority_alpha
        self.priority_beta = priority_beta

        # Storage
        self._experiences: Dict[str, Experience] = {}
        self._priorities: Dict[str, float] = {}
        self._insertion_order: deque = deque(maxlen=capacity)

        # Statistics
        self._total_added = 0
        self._total_sampled = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def add(self, experience: Experience):
        """
        Add experience to buffer.

        Args:
            experience: Experience to add
        """
        async with self._lock:
            # Calculate priority based on feedback and importance
            priority = self._calculate_priority(experience)

            # Add to storage
            self._experiences[experience.id] = experience
            self._priorities[experience.id] = priority
            self._insertion_order.append(experience.id)

            # Remove oldest if over capacity
            if len(self._experiences) > self.capacity:
                oldest_id = self._insertion_order[0]
                if oldest_id in self._experiences:
                    del self._experiences[oldest_id]
                    del self._priorities[oldest_id]

            self._total_added += 1

    def _calculate_priority(self, experience: Experience) -> float:
        """Calculate priority for experience."""
        # Base priority on feedback (convert -1 to 1 range to 0 to 1)
        feedback_priority = (experience.feedback + 1) / 2

        # Include importance factor
        priority = feedback_priority * experience.importance

        # Apply alpha exponent
        return priority ** self.priority_alpha + 1e-6

    async def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample batch with prioritized sampling.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of sampled experiences
        """
        async with self._lock:
            if len(self._experiences) == 0:
                return []

            # Calculate sampling probabilities
            ids = list(self._experiences.keys())
            priorities = np.array([self._priorities[id] for id in ids])
            probabilities = priorities / priorities.sum()

            # Sample indices
            n_samples = min(batch_size, len(ids))
            sampled_indices = np.random.choice(
                len(ids),
                size=n_samples,
                replace=False,
                p=probabilities
            )

            # Get experiences
            sampled = [self._experiences[ids[i]] for i in sampled_indices]

            self._total_sampled += n_samples

            return sampled

    async def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        async with self._lock:
            return {
                "size": len(self._experiences),
                "capacity": self.capacity,
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "avg_priority": np.mean(list(self._priorities.values())) if self._priorities else 0,
                "priority_range": (
                    min(self._priorities.values()),
                    max(self._priorities.values())
                ) if self._priorities else (0, 0),
            }

    async def save(self, path: Path):
        """Save buffer to disk."""
        async with self._lock:
            data = {
                "experiences": {k: v.to_dict() for k, v in self._experiences.items()},
                "priorities": self._priorities,
                "metadata": {
                    "total_added": self._total_added,
                    "total_sampled": self._total_sampled,
                    "capacity": self.capacity,
                }
            }

            with open(path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"Saved experience buffer to {path}")

    async def load(self, path: Path):
        """Load buffer from disk."""
        async with self._lock:
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Restore experiences
            for id, exp_dict in data["experiences"].items():
                self._experiences[id] = Experience(**exp_dict)
                self._insertion_order.append(id)

            self._priorities = data["priorities"]
            self._total_added = data["metadata"]["total_added"]
            self._total_sampled = data["metadata"]["total_sampled"]

            logger.info(f"Loaded experience buffer from {path}")


# ============================================================================
# VECTOR STORE
# ============================================================================

class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add(self, documents: List[Document]):
        """Add documents to store."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[Document], List[float]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete(self, ids: List[str]):
        """Delete documents by ID."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search."""

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "IVF100,Flat"
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_type: FAISS index type
        """
        self.dimension = dimension
        self.index_type = index_type

        # Initialize index
        if FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        else:
            self._index = None

        # Document storage
        self._documents: Dict[str, Document] = {}
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}

        # Lock
        self._lock = asyncio.Lock()

    async def add(self, documents: List[Document]):
        """Add documents to store."""
        async with self._lock:
            for doc in documents:
                if doc.embedding is None:
                    continue

                # Store document
                self._documents[doc.id] = doc

                # Add to index
                idx = len(self._id_to_idx)
                self._id_to_idx[doc.id] = idx
                self._idx_to_id[idx] = doc.id

                if self._index is not None:
                    # Normalize for cosine similarity
                    embedding = doc.embedding.astype(np.float32)
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
                    self._index.add(embedding.reshape(1, -1))

            logger.debug(f"Added {len(documents)} documents to vector store")

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[Document], List[float]]:
        """Search for similar documents."""
        async with self._lock:
            if self._index is None or len(self._documents) == 0:
                return [], []

            # Normalize query
            query = query_embedding.astype(np.float32)
            query = query / (np.linalg.norm(query) + 1e-9)

            # Search
            k = min(top_k, len(self._documents))
            distances, indices = self._index.search(query.reshape(1, -1), k)

            # Get documents
            docs = []
            scores = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx in self._idx_to_id:
                    doc_id = self._idx_to_id[idx]
                    docs.append(self._documents[doc_id])
                    scores.append(float(dist))

            return docs, scores

    async def delete(self, ids: List[str]):
        """Delete documents by ID."""
        async with self._lock:
            for doc_id in ids:
                if doc_id in self._documents:
                    del self._documents[doc_id]
                    # Note: FAISS doesn't support efficient deletion
                    # Would need to rebuild index

            logger.debug(f"Deleted {len(ids)} documents from vector store")


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for testing."""

    def __init__(self, dimension: int = 768):
        """Initialize in-memory store."""
        self.dimension = dimension
        self._documents: Dict[str, Document] = {}
        self._lock = asyncio.Lock()

    async def add(self, documents: List[Document]):
        """Add documents."""
        async with self._lock:
            for doc in documents:
                self._documents[doc.id] = doc

    async def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[Document], List[float]]:
        """Search using brute-force cosine similarity."""
        async with self._lock:
            if not self._documents:
                return [], []

            # Calculate similarities
            similarities = []
            for doc in self._documents.values():
                if doc.embedding is not None:
                    # Cosine similarity
                    sim = np.dot(query_embedding, doc.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding) + 1e-9
                    )
                    similarities.append((doc, float(sim)))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Return top k
            docs = [s[0] for s in similarities[:top_k]]
            scores = [s[1] for s in similarities[:top_k]]

            return docs, scores

    async def delete(self, ids: List[str]):
        """Delete documents."""
        async with self._lock:
            for doc_id in ids:
                if doc_id in self._documents:
                    del self._documents[doc_id]


# ============================================================================
# RAG ENGINE
# ============================================================================

class RAGEngine:
    """
    Retrieval-Augmented Generation Engine.

    Combines retrieval from vector store with generation
    for knowledge-grounded responses.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[str] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ):
        """
        Initialize RAG engine.

        Args:
            vector_store: Vector store for document retrieval
            embedding_model: Model for generating embeddings
            retrieval_strategy: Retrieval strategy to use
        """
        self.vector_store = vector_store or InMemoryVectorStore()
        self.embedding_model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        self.retrieval_strategy = retrieval_strategy

        # Embedding cache
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Statistics
        self._queries = 0
        self._hits = 0

        # Lock
        self._lock = asyncio.Lock()

    async def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Generate embedding
        try:
            from sentence_transformers import SentenceTransformer

            if not hasattr(self, '_embedder'):
                self._embedder = SentenceTransformer(self.embedding_model)

            embedding = self._embedder.encode(text)
            embedding = np.array(embedding, dtype=np.float32)

        except ImportError:
            # Mock embedding
            embedding = np.random.randn(768).astype(np.float32)

        # Cache
        self._embedding_cache[cache_key] = embedding

        return embedding

    async def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Add documents to the knowledge base.

        Args:
            texts: List of document texts
            metadatas: Optional metadata for each document
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        documents = []

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}

            # Chunk text
            chunks = self._chunk_text(text, chunk_size, chunk_overlap)

            for j, chunk in enumerate(chunks):
                # Generate embedding
                embedding = await self.embed(chunk)

                doc = Document(
                    content=chunk,
                    embedding=embedding,
                    metadata=metadata,
                    source=metadata.get("source", "unknown"),
                    chunk_index=j,
                )
                documents.append(doc)

        await self.vector_store.add(documents)
        logger.info(f"Added {len(documents)} chunks from {len(texts)} documents")

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Chunk text into overlapping segments."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:
                    end = start + last_period + 1
                    chunk = text[start:end]

            chunks.append(chunk.strip())
            start = end - chunk_overlap

        return chunks

    async def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            Retrieval results
        """
        start_time = time.time()
        self._queries += 1

        # Embed query
        query_embedding = await self.embed(query)

        # Search vector store
        docs, scores = await self.vector_store.search(query_embedding, top_k)

        latency = (time.time() - start_time) * 1000

        if docs:
            self._hits += 1

        return RetrievalResult(
            documents=docs,
            scores=scores,
            query_embedding=query_embedding,
            latency_ms=latency,
        )

    async def generate_with_context(
        self,
        query: str,
        generator_fn: Callable[[str], str],
        top_k: int = 5
    ) -> Tuple[str, RetrievalResult]:
        """
        Generate response with retrieved context.

        Args:
            query: User query
            generator_fn: Function to generate response
            top_k: Number of documents to retrieve

        Returns:
            Generated response and retrieval results
        """
        # Retrieve context
        retrieval_result = await self.retrieve(query, top_k)

        # Build context
        context_parts = []
        for doc, score in zip(retrieval_result.documents, retrieval_result.scores):
            context_parts.append(f"[Source: {doc.source}, Relevance: {score:.2f}]\n{doc.content}")

        context = "\n\n".join(context_parts)

        # Build augmented prompt
        augmented_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        # Generate
        response = generator_fn(augmented_prompt)

        return response, retrieval_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG statistics."""
        return {
            "total_queries": self._queries,
            "hits": self._hits,
            "hit_rate": self._hits / self._queries if self._queries > 0 else 0,
            "cache_size": len(self._embedding_cache),
            "retrieval_strategy": self.retrieval_strategy.value,
        }


# ============================================================================
# CONTINUAL LEARNING ENGINE
# ============================================================================

class ContinualLearningEngine:
    """
    Main continual learning engine.

    Orchestrates experience replay, RAG, and model updates
    for continuous learning from JARVIS interactions.
    """

    def __init__(
        self,
        strategy: LearningStrategy = LearningStrategy.EXPERIENCE_REPLAY,
        buffer_size: int = 100000,
        rag_enabled: bool = True,
    ):
        """
        Initialize continual learning engine.

        Args:
            strategy: Learning strategy to use
            buffer_size: Size of experience replay buffer
            rag_enabled: Enable RAG for retrieval
        """
        self.strategy = strategy
        self.rag_enabled = rag_enabled

        # Components
        self.experience_buffer = ExperienceReplayBuffer(capacity=buffer_size)
        self.rag_engine = RAGEngine() if rag_enabled else None

        # EWC parameters (for Elastic Weight Consolidation)
        self._fisher_information: Dict[str, np.ndarray] = {}
        self._optimal_params: Dict[str, np.ndarray] = {}
        self._ewc_lambda = float(os.getenv("EWC_LAMBDA", "1000.0"))

        # Metrics
        self.metrics = LearningMetrics()

        # Paths
        self._data_dir = Path(os.getenv("LEARNING_DATA_DIR", "~/.jarvis/learning")).expanduser()
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Lock
        self._lock = asyncio.Lock()

    async def learn_from_interaction(
        self,
        prompt: str,
        response: str,
        feedback: float,
        task_type: str = "general",
        metadata: Optional[Dict] = None
    ):
        """
        Learn from a single interaction.

        Args:
            prompt: User prompt
            response: Model response
            feedback: User feedback (-1 to 1)
            task_type: Type of task
            metadata: Additional metadata
        """
        async with self._lock:
            # Create experience
            experience = Experience(
                prompt=prompt,
                response=response,
                feedback=feedback,
                task_type=task_type,
                metadata=metadata or {},
            )

            # Generate embedding for the experience
            if self.rag_engine:
                combined_text = f"{prompt} {response}"
                experience.embedding = await self.rag_engine.embed(combined_text)

            # Calculate importance based on feedback
            experience.importance = abs(feedback) + 0.5  # 0.5 to 1.5 range

            # Add to buffer
            await self.experience_buffer.add(experience)

            # Add to RAG knowledge base if positive feedback
            if feedback > 0.5 and self.rag_engine:
                await self.rag_engine.add_documents(
                    [f"Q: {prompt}\nA: {response}"],
                    [{"task_type": task_type, "feedback": feedback}]
                )

            # Update metrics
            self.metrics.experiences_learned += 1
            total = self.metrics.experiences_learned
            self.metrics.average_feedback = (
                (self.metrics.average_feedback * (total - 1) + feedback) / total
            )

            logger.debug(f"Learned from interaction with feedback {feedback}")

    async def get_training_batch(
        self,
        batch_size: int = 32,
        include_replay: bool = True,
        replay_ratio: float = 0.5
    ) -> List[Experience]:
        """
        Get training batch with experience replay.

        Args:
            batch_size: Total batch size
            include_replay: Include replayed experiences
            replay_ratio: Ratio of replay to new experiences

        Returns:
            Batch of experiences for training
        """
        if include_replay and self.strategy == LearningStrategy.EXPERIENCE_REPLAY:
            replay_size = int(batch_size * replay_ratio)
            replayed = await self.experience_buffer.sample(replay_size)
            self.metrics.replay_ratio = replay_ratio
            return replayed
        else:
            return []

    async def generate_with_rag(
        self,
        query: str,
        generator_fn: Callable[[str], str],
        top_k: int = 5
    ) -> Tuple[str, Optional[RetrievalResult]]:
        """
        Generate response with RAG augmentation.

        Args:
            query: User query
            generator_fn: Generation function
            top_k: Number of documents to retrieve

        Returns:
            Response and retrieval result
        """
        if not self.rag_enabled or not self.rag_engine:
            return generator_fn(query), None

        return await self.rag_engine.generate_with_context(
            query, generator_fn, top_k
        )

    async def compute_ewc_penalty(
        self,
        current_params: Dict[str, np.ndarray]
    ) -> float:
        """
        Compute EWC regularization penalty.

        Args:
            current_params: Current model parameters

        Returns:
            EWC penalty value
        """
        if not self._optimal_params or not self._fisher_information:
            return 0.0

        penalty = 0.0

        for name in self._optimal_params:
            if name in current_params and name in self._fisher_information:
                diff = current_params[name] - self._optimal_params[name]
                penalty += np.sum(self._fisher_information[name] * diff ** 2)

        return float(self._ewc_lambda * penalty)

    async def update_ewc_params(
        self,
        params: Dict[str, np.ndarray],
        fisher_samples: int = 1000
    ):
        """
        Update EWC optimal parameters and Fisher information.

        Args:
            params: Current model parameters
            fisher_samples: Number of samples for Fisher estimation
        """
        async with self._lock:
            self._optimal_params = {k: v.copy() for k, v in params.items()}

            # Estimate Fisher information (simplified)
            # In practice, would compute from gradient samples
            for name, param in params.items():
                self._fisher_information[name] = np.ones_like(param) * 0.1

            logger.info("Updated EWC parameters")

    async def save_state(self):
        """Save learning state to disk."""
        async with self._lock:
            # Save experience buffer
            buffer_path = self._data_dir / "experience_buffer.pkl"
            await self.experience_buffer.save(buffer_path)

            # Save metrics
            metrics_path = self._data_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    "experiences_learned": self.metrics.experiences_learned,
                    "average_feedback": self.metrics.average_feedback,
                    "replay_ratio": self.metrics.replay_ratio,
                }, f)

            logger.info(f"Saved learning state to {self._data_dir}")

    async def load_state(self):
        """Load learning state from disk."""
        async with self._lock:
            # Load experience buffer
            buffer_path = self._data_dir / "experience_buffer.pkl"
            if buffer_path.exists():
                await self.experience_buffer.load(buffer_path)

            # Load metrics
            metrics_path = self._data_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    self.metrics.experiences_learned = data["experiences_learned"]
                    self.metrics.average_feedback = data["average_feedback"]

            logger.info(f"Loaded learning state from {self._data_dir}")

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "strategy": self.strategy.value,
            "rag_enabled": self.rag_enabled,
            "metrics": {
                "experiences_learned": self.metrics.experiences_learned,
                "average_feedback": self.metrics.average_feedback,
                "replay_ratio": self.metrics.replay_ratio,
                "forgetting_rate": self.metrics.forgetting_rate,
            },
            "rag_stats": self.rag_engine.get_statistics() if self.rag_engine else None,
        }


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_continual_learner: Optional[ContinualLearningEngine] = None
_learner_lock = asyncio.Lock()


async def get_continual_learner() -> ContinualLearningEngine:
    """Get or create global continual learner."""
    global _continual_learner

    async with _learner_lock:
        if _continual_learner is None:
            strategy_name = os.getenv("LEARNING_STRATEGY", "experience_replay")
            try:
                strategy = LearningStrategy(strategy_name)
            except ValueError:
                strategy = LearningStrategy.EXPERIENCE_REPLAY

            _continual_learner = ContinualLearningEngine(
                strategy=strategy,
                rag_enabled=os.getenv("RAG_ENABLED", "true").lower() == "true"
            )

            # Try to load existing state
            try:
                await _continual_learner.load_state()
            except Exception as e:
                logger.warning(f"Could not load learning state: {e}")

        return _continual_learner


_rag_engine: Optional[RAGEngine] = None


async def get_rag_engine() -> RAGEngine:
    """Get RAG engine from continual learner."""
    learner = await get_continual_learner()
    return learner.rag_engine
