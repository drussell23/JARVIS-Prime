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


# Recommended models for JARVIS-Prime
MODEL_CATALOG: Dict[str, ModelSpec] = {
    "tinyllama-chat": ModelSpec(
        name="TinyLlama 1.1B Chat",
        repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        size_mb=670,
        description="Fast, lightweight model for simple conversations",
        recommended_for=["testing", "simple-chat", "low-memory"],
        context_length=2048,
    ),
    "phi-2": ModelSpec(
        name="Phi-2 2.7B",
        repo_id="TheBloke/phi-2-GGUF",
        filename="phi-2.Q4_K_M.gguf",
        size_mb=1600,
        description="Excellent reasoning for its size, good for coding",
        recommended_for=["coding", "reasoning", "balanced"],
        context_length=2048,
    ),
    "mistral-7b-instruct": ModelSpec(
        name="Mistral 7B Instruct v0.2",
        repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        size_mb=4370,
        description="Production-quality instruction following",
        recommended_for=["production", "instructions", "complex-tasks"],
        context_length=32768,
    ),
    "llama-3-8b-instruct": ModelSpec(
        name="Llama 3 8B Instruct",
        repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        size_mb=4920,
        description="Latest Llama 3 with excellent instruction following",
        recommended_for=["production", "complex-reasoning", "latest"],
        context_length=8192,
    ),
    "qwen2-7b-instruct": ModelSpec(
        name="Qwen2 7B Instruct",
        repo_id="Qwen/Qwen2-7B-Instruct-GGUF",
        filename="qwen2-7b-instruct-q4_k_m.gguf",
        size_mb=4650,
        description="Strong multilingual and coding capabilities",
        recommended_for=["multilingual", "coding", "production"],
        context_length=32768,
    ),
}


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
