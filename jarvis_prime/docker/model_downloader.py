"""
Model Downloader - Intelligent GGUF Model Management
=====================================================

Downloads and manages GGUF models from HuggingFace Hub with:
- Intelligent model selection (auto-picks best quantization)
- Progress tracking with resume support
- Version management and caching
- Reactor-core integration for receiving trained models

Recommended Models for M1 Mac (16GB RAM):
- TinyLlama-1.1B-Chat-v1.0 (Q4_K_M) - ~700MB, fast, good for simple tasks
- Phi-2 (Q4_K_M) - ~1.6GB, excellent reasoning
- Mistral-7B-Instruct (Q4_K_M) - ~4GB, production quality
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from huggingface_hub import hf_hub_download, HfApi, list_repo_files
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)


# ============================================================================
# Pre-configured Model Catalog
# ============================================================================

@dataclass
class ModelSpec:
    """Specification for a pre-configured model."""
    name: str
    repo_id: str
    filename: str
    size_mb: int
    description: str
    recommended_for: List[str]
    quantization: str = "Q4_K_M"
    context_length: int = 4096


# =============================================================================
# v84.0: Enhanced Model Catalog with Capability Routing
# =============================================================================

@dataclass
class ModelCapability:
    """Model capability scores for intelligent routing (0.0-1.0)."""
    coding: float = 0.5
    reasoning: float = 0.5
    math: float = 0.5
    creative: float = 0.5
    instruction_following: float = 0.5
    multilingual: float = 0.3
    context_efficiency: float = 0.5  # How well it uses long context
    speed: float = 0.5  # Inference speed (higher = faster)


@dataclass
class ModelSpec:
    """Specification for a pre-configured model."""
    name: str
    repo_id: str
    filename: str
    size_mb: int
    description: str
    recommended_for: List[str]
    quantization: str = "Q4_K_M"
    context_length: int = 4096
    # v84.0: Enhanced capabilities
    capabilities: ModelCapability = field(default_factory=ModelCapability)
    chat_template: str = "chatml"  # chatml, llama, mistral, alpaca
    stop_tokens: List[str] = field(default_factory=list)


# v84.0: Comprehensive model catalog with intelligent routing support
MODEL_CATALOG: Dict[str, ModelSpec] = {
    # ==========================================================================
    # TIER 0: Lightweight Models (< 2GB) - Fast, Low Memory
    # ==========================================================================
    "tinyllama-chat": ModelSpec(
        name="TinyLlama 1.1B Chat",
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        size_mb=670,
        description="Fast, lightweight model for simple conversations",
        recommended_for=["testing", "simple-chat", "low-memory", "fast"],
        context_length=2048,
        capabilities=ModelCapability(
            coding=0.3, reasoning=0.3, math=0.2, creative=0.4,
            instruction_following=0.5, speed=0.95,
        ),
        chat_template="chatml",
        stop_tokens=["<|im_end|>"],
    ),
    "phi-3.5-mini": ModelSpec(
        name="Phi-3.5 Mini 3.8B",
        repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        size_mb=2300,
        description="Microsoft's efficient reasoning model, excellent for its size",
        recommended_for=["reasoning", "coding", "fast", "balanced"],
        context_length=128000,  # 128K context!
        capabilities=ModelCapability(
            coding=0.75, reasoning=0.8, math=0.75, creative=0.5,
            instruction_following=0.8, context_efficiency=0.9, speed=0.8,
        ),
        chat_template="phi",
        stop_tokens=["<|end|>", "<|endoftext|>"],
    ),

    # ==========================================================================
    # TIER 1: Medium Models (2-5GB) - Balanced Performance
    # ==========================================================================
    "phi-2": ModelSpec(
        name="Phi-2 2.7B",
        repo_id="TheBloke/phi-2-GGUF",
        filename="phi-2.Q4_K_M.gguf",
        size_mb=1600,
        description="Excellent reasoning for its size, good for coding",
        recommended_for=["coding", "reasoning", "balanced"],
        context_length=2048,
        capabilities=ModelCapability(
            coding=0.65, reasoning=0.7, math=0.6, creative=0.4,
            instruction_following=0.6, speed=0.85,
        ),
        chat_template="chatml",
    ),
    "deepseek-coder-6.7b": ModelSpec(
        name="DeepSeek Coder 6.7B Instruct",
        repo_id="TheBloke/deepseek-coder-6.7b-instruct-GGUF",
        filename="deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
        size_mb=3800,
        description="Specialized coding model, excellent for Python/JS/TS",
        recommended_for=["coding", "code-generation", "refactoring", "debugging"],
        context_length=16384,
        capabilities=ModelCapability(
            coding=0.92, reasoning=0.7, math=0.8, creative=0.3,
            instruction_following=0.75, context_efficiency=0.8, speed=0.7,
        ),
        chat_template="deepseek",
        stop_tokens=["<|EOT|>"],
    ),
    "codellama-7b-instruct": ModelSpec(
        name="CodeLlama 7B Instruct",
        repo_id="TheBloke/CodeLlama-7B-Instruct-GGUF",
        filename="codellama-7b-instruct.Q4_K_M.gguf",
        size_mb=4200,
        description="Meta's CodeLlama - excellent for Python, JavaScript, TypeScript",
        recommended_for=["coding", "code-generation", "refactoring", "debugging", "code-review"],
        context_length=16384,
        capabilities=ModelCapability(
            coding=0.9, reasoning=0.65, math=0.7, creative=0.3,
            instruction_following=0.75, context_efficiency=0.75, speed=0.7,
        ),
        chat_template="llama",
        stop_tokens=["</s>", "[INST]"],
    ),
    "mistral-7b-instruct": ModelSpec(
        name="Mistral 7B Instruct v0.3",
        repo_id="TheBloke/Mistral-7B-Instruct-v0.3-GGUF",
        filename="mistral-7b-instruct-v0.3.Q4_K_M.gguf",
        size_mb=4370,
        description="Production-quality instruction following with sliding window attention",
        recommended_for=["production", "instructions", "complex-tasks", "general"],
        context_length=32768,
        capabilities=ModelCapability(
            coding=0.75, reasoning=0.8, math=0.7, creative=0.7,
            instruction_following=0.85, context_efficiency=0.85, speed=0.7,
        ),
        chat_template="mistral",
        stop_tokens=["</s>", "[INST]"],
    ),
    "qwen2.5-7b-instruct": ModelSpec(
        name="Qwen2.5 7B Instruct",
        repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        size_mb=4650,
        description="Strong math, logic, reasoning, and multilingual capabilities",
        recommended_for=["reasoning", "math", "logic", "analysis", "multilingual", "coding"],
        context_length=32768,
        capabilities=ModelCapability(
            coding=0.85, reasoning=0.9, math=0.92, creative=0.6,
            instruction_following=0.88, multilingual=0.9, context_efficiency=0.85, speed=0.65,
        ),
        chat_template="chatml",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
    ),
    "llama-3-8b-instruct": ModelSpec(
        name="Llama 3 8B Instruct",
        repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        size_mb=4920,
        description="Latest Llama 3 with excellent instruction following",
        recommended_for=["production", "complex-reasoning", "latest", "general"],
        context_length=8192,
        capabilities=ModelCapability(
            coding=0.8, reasoning=0.85, math=0.75, creative=0.75,
            instruction_following=0.9, speed=0.65,
        ),
        chat_template="llama3",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
    ),
    "llama-3.1-8b-instruct": ModelSpec(
        name="Llama 3.1 8B Instruct",
        repo_id="QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
        size_mb=4920,
        description="Latest Llama 3.1 with improved reasoning and 128K context",
        recommended_for=["production", "complex-reasoning", "latest", "long-context"],
        context_length=131072,  # 128K context!
        capabilities=ModelCapability(
            coding=0.82, reasoning=0.88, math=0.78, creative=0.78,
            instruction_following=0.92, context_efficiency=0.9, speed=0.6,
        ),
        chat_template="llama3",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
    ),
    "qwen2-7b-instruct": ModelSpec(
        name="Qwen2 7B Instruct",
        repo_id="Qwen/Qwen2-7B-Instruct-GGUF",
        filename="qwen2-7b-instruct-q4_k_m.gguf",
        size_mb=4650,
        description="Strong multilingual and coding capabilities",
        recommended_for=["multilingual", "coding", "production"],
        context_length=32768,
        capabilities=ModelCapability(
            coding=0.82, reasoning=0.85, math=0.88, creative=0.55,
            instruction_following=0.85, multilingual=0.88, speed=0.65,
        ),
        chat_template="chatml",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
    ),

    # ==========================================================================
    # TIER 2: Large Models (5-10GB) - High Performance
    # ==========================================================================
    "codellama-13b-instruct": ModelSpec(
        name="CodeLlama 13B Instruct",
        repo_id="TheBloke/CodeLlama-13B-Instruct-GGUF",
        filename="codellama-13b-instruct.Q4_K_M.gguf",
        size_mb=7300,
        description="Larger CodeLlama for complex code generation and refactoring",
        recommended_for=["coding", "complex-code", "refactoring", "architecture"],
        context_length=16384,
        capabilities=ModelCapability(
            coding=0.95, reasoning=0.75, math=0.8, creative=0.35,
            instruction_following=0.8, context_efficiency=0.8, speed=0.5,
        ),
        chat_template="llama",
        stop_tokens=["</s>", "[INST]"],
    ),
    "qwen2.5-14b-instruct": ModelSpec(
        name="Qwen2.5 14B Instruct",
        repo_id="Qwen/Qwen2.5-14B-Instruct-GGUF",
        filename="qwen2.5-14b-instruct-q4_k_m.gguf",
        size_mb=8200,
        description="Advanced reasoning and math capabilities",
        recommended_for=["reasoning", "math", "analysis", "complex-tasks"],
        context_length=32768,
        capabilities=ModelCapability(
            coding=0.88, reasoning=0.95, math=0.96, creative=0.65,
            instruction_following=0.92, multilingual=0.92, context_efficiency=0.88, speed=0.45,
        ),
        chat_template="chatml",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
    ),
    "wizardcoder-15b": ModelSpec(
        name="WizardCoder 15B v1.0",
        repo_id="TheBloke/WizardCoder-15B-1.0-GGUF",
        filename="wizardcoder-15b-1.0.Q4_K_M.gguf",
        size_mb=8500,
        description="WizardLM's code-focused model with evol-instruct training",
        recommended_for=["coding", "code-generation", "debugging", "complex-code"],
        context_length=8192,
        capabilities=ModelCapability(
            coding=0.93, reasoning=0.7, math=0.75, creative=0.3,
            instruction_following=0.8, speed=0.4,
        ),
        chat_template="alpaca",
        stop_tokens=["### Response:", "### Instruction:"],
    ),

    # ==========================================================================
    # TIER 3: Cloud-Scale Models (10-30GB) - For GCP/Cloud Deployment
    # ==========================================================================
    "llama-3.1-70b-instruct": ModelSpec(
        name="Llama 3.1 70B Instruct",
        repo_id="QuantFactory/Meta-Llama-3.1-70B-Instruct-GGUF",
        filename="Meta-Llama-3.1-70B-Instruct.Q4_K_M.gguf",
        size_mb=20000,
        description="State-of-the-art open model for advanced reasoning",
        recommended_for=["advanced-reasoning", "cloud", "production", "complex-tasks"],
        context_length=131072,
        capabilities=ModelCapability(
            coding=0.92, reasoning=0.98, math=0.95, creative=0.9,
            instruction_following=0.98, context_efficiency=0.95, speed=0.15,
        ),
        chat_template="llama3",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
    ),
    "qwen2.5-32b-instruct": ModelSpec(
        name="Qwen2.5 32B Instruct",
        repo_id="Qwen/Qwen2.5-32B-Instruct-GGUF",
        filename="qwen2.5-32b-instruct-q4_k_m.gguf",
        size_mb=18000,
        description="Advanced math and science capabilities",
        recommended_for=["math", "science", "cloud", "advanced-reasoning"],
        context_length=32768,
        capabilities=ModelCapability(
            coding=0.92, reasoning=0.97, math=0.98, creative=0.75,
            instruction_following=0.95, multilingual=0.95, context_efficiency=0.9, speed=0.2,
        ),
        chat_template="chatml",
        stop_tokens=["<|im_end|>", "<|endoftext|>"],
    ),
}


# =============================================================================
# v84.0: Model Selection by Capability
# =============================================================================

def get_best_model_for_task(
    task_type: str,
    max_memory_gb: float = 10.0,
    min_capability_score: float = 0.7,
) -> Optional[str]:
    """
    v84.0: Intelligently select the best model for a given task type.

    Args:
        task_type: One of "coding", "reasoning", "math", "creative", "general"
        max_memory_gb: Maximum memory available
        min_capability_score: Minimum required capability score

    Returns:
        Model key or None if no suitable model found
    """
    max_size_mb = max_memory_gb * 1024

    # Map task type to capability attribute
    capability_map = {
        "coding": "coding",
        "code-generation": "coding",
        "refactoring": "coding",
        "debugging": "coding",
        "reasoning": "reasoning",
        "logic": "reasoning",
        "analysis": "reasoning",
        "math": "math",
        "science": "math",
        "creative": "creative",
        "writing": "creative",
        "general": "instruction_following",
        "instructions": "instruction_following",
    }

    capability_attr = capability_map.get(task_type, "instruction_following")

    # Filter and score models
    candidates = []
    for key, spec in MODEL_CATALOG.items():
        if spec.size_mb > max_size_mb:
            continue

        score = getattr(spec.capabilities, capability_attr, 0.5)
        if score >= min_capability_score:
            # Composite score: capability * speed (balance performance and speed)
            composite = score * 0.7 + spec.capabilities.speed * 0.3
            candidates.append((key, spec, composite))

    if not candidates:
        return None

    # Sort by composite score (highest first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0]


def get_coding_models(max_memory_gb: float = 10.0) -> List[str]:
    """Get all models suitable for coding tasks."""
    max_size_mb = max_memory_gb * 1024
    return [
        key for key, spec in MODEL_CATALOG.items()
        if spec.size_mb <= max_size_mb and spec.capabilities.coding >= 0.8
    ]


def get_reasoning_models(max_memory_gb: float = 10.0) -> List[str]:
    """Get all models suitable for reasoning tasks."""
    max_size_mb = max_memory_gb * 1024
    return [
        key for key, spec in MODEL_CATALOG.items()
        if spec.size_mb <= max_size_mb and spec.capabilities.reasoning >= 0.8
    ]


# ============================================================================
# Download Manager
# ============================================================================

@dataclass
class DownloadProgress:
    """Track download progress."""
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: float = 0.0
    status: str = "pending"
    error: Optional[str] = None

    @property
    def percent(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100


@dataclass
class ModelMetadata:
    """Metadata for a downloaded model."""
    name: str
    repo_id: str
    filename: str
    local_path: Path
    size_bytes: int
    sha256: str
    downloaded_at: datetime
    quantization: str
    context_length: int
    version: str = "v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "local_path": str(self.local_path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "downloaded_at": self.downloaded_at.isoformat(),
            "quantization": self.quantization,
            "context_length": self.context_length,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            name=data["name"],
            repo_id=data["repo_id"],
            filename=data["filename"],
            local_path=Path(data["local_path"]),
            size_bytes=data["size_bytes"],
            sha256=data["sha256"],
            downloaded_at=datetime.fromisoformat(data["downloaded_at"]),
            quantization=data["quantization"],
            context_length=data["context_length"],
            version=data.get("version", "v1.0"),
        )


class ModelDownloader:
    """
    Intelligent model downloader and manager.

    Features:
    - Download from HuggingFace Hub with progress
    - Auto-select optimal quantization for hardware
    - Version management and rollback
    - Integrity verification (SHA256)
    - Resume interrupted downloads

    Usage:
        downloader = ModelDownloader(models_dir="./models")

        # Download from catalog
        path = await downloader.download_catalog_model("mistral-7b-instruct")

        # Download custom model
        path = await downloader.download(
            repo_id="TheBloke/some-model-GGUF",
            filename="model.Q4_K_M.gguf",
        )

        # List available models
        models = downloader.list_local_models()

        # Set active model
        await downloader.set_active_model("mistral-7b-instruct")
    """

    def __init__(
        self,
        models_dir: str | Path = "./models",
        hf_token: Optional[str] = None,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._metadata_file = self.models_dir / "models_metadata.json"
        self._metadata: Dict[str, ModelMetadata] = {}
        self._current_model: Optional[str] = None
        self._progress_callback: Optional[Callable[[DownloadProgress], None]] = None

        # Load existing metadata
        self._load_metadata()

        # Initialize HF API
        self._hf_api = HfApi(token=self.hf_token)

    def _load_metadata(self) -> None:
        """Load model metadata from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    data = json.load(f)
                    self._metadata = {
                        k: ModelMetadata.from_dict(v)
                        for k, v in data.get("models", {}).items()
                    }
                    self._current_model = data.get("current_model")
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self._metadata = {}

    def _save_metadata(self) -> None:
        """Save model metadata to disk."""
        data = {
            "models": {k: v.to_dict() for k, v in self._metadata.items()},
            "current_model": self._current_model,
            "updated_at": datetime.now().isoformat(),
        }
        with open(self._metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def set_progress_callback(self, callback: Callable[[DownloadProgress], None]) -> None:
        """Set callback for download progress updates."""
        self._progress_callback = callback

    async def download_catalog_model(
        self,
        model_key: str,
        force: bool = False,
    ) -> Path:
        """
        Download a model from the pre-configured catalog.

        Args:
            model_key: Key from MODEL_CATALOG (e.g., "mistral-7b-instruct")
            force: Re-download even if exists

        Returns:
            Path to downloaded model
        """
        if model_key not in MODEL_CATALOG:
            available = ", ".join(MODEL_CATALOG.keys())
            raise ValueError(f"Unknown model: {model_key}. Available: {available}")

        spec = MODEL_CATALOG[model_key]
        return await self.download(
            repo_id=spec.repo_id,
            filename=spec.filename,
            name=model_key,
            force=force,
            context_length=spec.context_length,
            quantization=spec.quantization,
        )

    async def download(
        self,
        repo_id: str,
        filename: str,
        name: Optional[str] = None,
        force: bool = False,
        context_length: int = 4096,
        quantization: str = "unknown",
    ) -> Path:
        """
        Download a GGUF model from HuggingFace.

        Args:
            repo_id: HuggingFace repo (e.g., "TheBloke/Mistral-7B-GGUF")
            filename: Model filename (e.g., "mistral-7b.Q4_K_M.gguf")
            name: Local name for the model
            force: Re-download even if exists
            context_length: Model context length
            quantization: Quantization method

        Returns:
            Path to downloaded model
        """
        name = name or filename.replace(".gguf", "")
        local_path = self.models_dir / filename

        # Check if already downloaded
        if not force and name in self._metadata:
            meta = self._metadata[name]
            if meta.local_path.exists():
                logger.info(f"Model already downloaded: {name}")
                return meta.local_path

        logger.info(f"Downloading {filename} from {repo_id}...")

        # Create progress tracker
        progress = DownloadProgress(status="downloading")

        try:
            # Download with progress
            loop = asyncio.get_event_loop()
            downloaded_path = await loop.run_in_executor(
                None,
                lambda: hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=self.models_dir,
                    token=self.hf_token,
                ),
            )

            downloaded_path = Path(downloaded_path)

            # Move to final location if needed
            if downloaded_path != local_path:
                shutil.move(str(downloaded_path), str(local_path))

            # Calculate checksum
            sha256 = await self._calculate_sha256(local_path)

            # Create metadata
            metadata = ModelMetadata(
                name=name,
                repo_id=repo_id,
                filename=filename,
                local_path=local_path,
                size_bytes=local_path.stat().st_size,
                sha256=sha256,
                downloaded_at=datetime.now(),
                quantization=quantization,
                context_length=context_length,
            )

            # Save metadata
            self._metadata[name] = metadata
            self._save_metadata()

            progress.status = "complete"
            progress.downloaded_bytes = metadata.size_bytes
            progress.total_bytes = metadata.size_bytes

            if self._progress_callback:
                self._progress_callback(progress)

            logger.info(f"Downloaded {name}: {local_path}")
            return local_path

        except Exception as e:
            progress.status = "error"
            progress.error = str(e)
            if self._progress_callback:
                self._progress_callback(progress)
            raise

    async def _calculate_sha256(self, path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        loop = asyncio.get_event_loop()

        def _hash_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()

        return await loop.run_in_executor(None, _hash_file)

    async def set_active_model(self, name: str) -> Path:
        """
        Set the active model (creates/updates 'current.gguf' symlink).

        Args:
            name: Model name from metadata

        Returns:
            Path to active model
        """
        if name not in self._metadata:
            raise ValueError(f"Model not found: {name}")

        meta = self._metadata[name]
        if not meta.local_path.exists():
            raise FileNotFoundError(f"Model file not found: {meta.local_path}")

        # Create/update symlink
        current_link = self.models_dir / "current.gguf"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()

        current_link.symlink_to(meta.local_path.name)
        self._current_model = name
        self._save_metadata()

        logger.info(f"Active model set to: {name}")
        return current_link

    def get_active_model(self) -> Optional[ModelMetadata]:
        """Get the currently active model metadata."""
        if self._current_model and self._current_model in self._metadata:
            return self._metadata[self._current_model]
        return None

    def list_local_models(self) -> List[ModelMetadata]:
        """List all locally downloaded models."""
        # Verify models still exist
        valid = {}
        for name, meta in self._metadata.items():
            if meta.local_path.exists():
                valid[name] = meta
        self._metadata = valid
        self._save_metadata()
        return list(self._metadata.values())

    def list_catalog_models(self) -> List[ModelSpec]:
        """List all models in the catalog."""
        return list(MODEL_CATALOG.values())

    async def delete_model(self, name: str) -> bool:
        """Delete a downloaded model."""
        if name not in self._metadata:
            return False

        meta = self._metadata[name]

        # Remove file
        if meta.local_path.exists():
            meta.local_path.unlink()

        # Update metadata
        del self._metadata[name]
        if self._current_model == name:
            self._current_model = None
        self._save_metadata()

        logger.info(f"Deleted model: {name}")
        return True

    async def verify_model(self, name: str) -> bool:
        """Verify model integrity via SHA256."""
        if name not in self._metadata:
            return False

        meta = self._metadata[name]
        if not meta.local_path.exists():
            return False

        actual_sha256 = await self._calculate_sha256(meta.local_path)
        return actual_sha256 == meta.sha256

    async def auto_select_model(self, max_memory_gb: float = 10.0) -> Optional[str]:
        """
        Auto-select the best model based on available memory.

        Args:
            max_memory_gb: Maximum memory to use for model

        Returns:
            Model key that fits, or None
        """
        max_size_mb = max_memory_gb * 1024

        # Filter catalog by size
        suitable = [
            (key, spec)
            for key, spec in MODEL_CATALOG.items()
            if spec.size_mb < max_size_mb
        ]

        if not suitable:
            return None

        # Sort by size (largest that fits)
        suitable.sort(key=lambda x: x[1].size_mb, reverse=True)
        return suitable[0][0]

    def get_status(self) -> Dict[str, Any]:
        """Get downloader status."""
        return {
            "models_dir": str(self.models_dir),
            "local_models": len(self._metadata),
            "current_model": self._current_model,
            "catalog_models": len(MODEL_CATALOG),
        }


# ============================================================================
# Convenience Functions
# ============================================================================

async def download_model(
    model_key: str = "tinyllama-chat",
    models_dir: str = "./models",
    set_active: bool = True,
) -> Path:
    """
    Convenience function to download and set up a model.

    Args:
        model_key: Key from MODEL_CATALOG
        models_dir: Directory to store models
        set_active: Whether to set as active model

    Returns:
        Path to model file
    """
    downloader = ModelDownloader(models_dir=models_dir)
    path = await downloader.download_catalog_model(model_key)

    if set_active:
        await downloader.set_active_model(model_key)

    return path


def list_available_models() -> Dict[str, ModelSpec]:
    """List all available models in the catalog."""
    return MODEL_CATALOG.copy()


def recommend_model(
    use_case: str = "balanced",
    max_memory_gb: float = 10.0,
) -> Optional[str]:
    """
    Recommend a model based on use case and memory constraints.

    Args:
        use_case: One of "testing", "coding", "production", "balanced"
        max_memory_gb: Maximum memory to use

    Returns:
        Recommended model key
    """
    max_size_mb = max_memory_gb * 1024

    # Filter by memory
    candidates = [
        (key, spec)
        for key, spec in MODEL_CATALOG.items()
        if spec.size_mb < max_size_mb
    ]

    if not candidates:
        return None

    # Filter by use case
    matching = [
        (key, spec)
        for key, spec in candidates
        if use_case in spec.recommended_for
    ]

    if matching:
        # Return largest matching
        matching.sort(key=lambda x: x[1].size_mb, reverse=True)
        return matching[0][0]

    # Fallback to largest that fits
    candidates.sort(key=lambda x: x[1].size_mb, reverse=True)
    return candidates[0][0]
