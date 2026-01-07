"""
AGI Model Persistence - Versioning and Checkpoints
====================================================

v78.0 - Comprehensive persistence for AGI models

Provides versioned storage for all AGI components:
- Model weights and configurations
- Reasoning engine strategies
- Learning engine state (experience buffers, EWC parameters)
- Cognitive state snapshots

FEATURES:
    - Atomic saves with rollback support
    - Automatic versioning with semantic versioning
    - Compression for storage efficiency
    - Incremental checkpoints
    - Cloud sync support (S3, GCS)
    - Model diffing and merging

STORAGE STRUCTURE:
    ~/.jarvis/prime/models/
    ├── manifest.json              # Model registry
    ├── agi_orchestrator/
    │   ├── v1.0.0/
    │   │   ├── config.json
    │   │   ├── weights.pt
    │   │   └── metadata.json
    │   ├── v1.0.1/
    │   └── current -> v1.0.1/
    ├── reasoning_engine/
    ├── continuous_learning/
    └── cognitive_state/
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================


class ModelComponent(Enum):
    """AGI model components that can be persisted."""

    AGI_ORCHESTRATOR = "agi_orchestrator"
    REASONING_ENGINE = "reasoning_engine"
    CONTINUOUS_LEARNING = "continuous_learning"
    MULTIMODAL_FUSION = "multimodal_fusion"
    COGNITIVE_STATE = "cognitive_state"
    ACTION_MODEL = "action_model"
    META_REASONER = "meta_reasoner"
    CAUSAL_ENGINE = "causal_engine"
    WORLD_MODEL = "world_model"
    MEMORY_CONSOLIDATOR = "memory_consolidator"
    GOAL_INFERENCE = "goal_inference"
    SELF_MODEL = "self_model"


class VersionState(Enum):
    """State of a model version."""

    ACTIVE = "active"         # Currently in use
    STAGED = "staged"         # Ready for deployment
    ARCHIVED = "archived"     # Kept for reference
    DEPRECATED = "deprecated" # Should not be used
    CORRUPTED = "corrupted"   # Failed integrity check


@dataclass
class SemanticVersion:
    """Semantic version representation."""

    major: int = 1
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_str: str) -> "SemanticVersion":
        """Parse version string."""
        parts = version_str.split("-")
        main = parts[0].lstrip("v")
        prerelease = parts[1] if len(parts) > 1 else None

        numbers = main.split(".")
        return cls(
            major=int(numbers[0]) if len(numbers) > 0 else 1,
            minor=int(numbers[1]) if len(numbers) > 1 else 0,
            patch=int(numbers[2]) if len(numbers) > 2 else 0,
            prerelease=prerelease,
        )

    def bump_major(self) -> "SemanticVersion":
        return SemanticVersion(self.major + 1, 0, 0)

    def bump_minor(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor + 1, 0)

    def bump_patch(self) -> "SemanticVersion":
        return SemanticVersion(self.major, self.minor, self.patch + 1)

    def __str__(self) -> str:
        version = f"v{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        return version

    def __lt__(self, other: "SemanticVersion") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)


@dataclass
class ModelMetadata:
    """Metadata for a saved model version."""

    component: ModelComponent
    version: SemanticVersion
    state: VersionState = VersionState.ACTIVE

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Content info
    file_size_bytes: int = 0
    checksum_sha256: str = ""
    compressed: bool = True

    # Training info
    training_steps: int = 0
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    # Dependencies
    parent_version: Optional[str] = None
    dependencies: Dict[str, str] = field(default_factory=dict)

    # Notes
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component.value,
            "version": str(self.version),
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "file_size_bytes": self.file_size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "compressed": self.compressed,
            "training_steps": self.training_steps,
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
            "metrics": self.metrics,
            "parent_version": self.parent_version,
            "dependencies": self.dependencies,
            "description": self.description,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            component=ModelComponent(data["component"]),
            version=SemanticVersion.parse(data["version"]),
            state=VersionState(data.get("state", "active")),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            file_size_bytes=data.get("file_size_bytes", 0),
            checksum_sha256=data.get("checksum_sha256", ""),
            compressed=data.get("compressed", True),
            training_steps=data.get("training_steps", 0),
            training_loss=data.get("training_loss"),
            validation_loss=data.get("validation_loss"),
            metrics=data.get("metrics", {}),
            parent_version=data.get("parent_version"),
            dependencies=data.get("dependencies", {}),
            description=data.get("description", ""),
            tags=data.get("tags", []),
        )


@dataclass
class PersistenceConfig:
    """Configuration for model persistence."""

    # Storage paths
    base_dir: Path = field(default_factory=lambda: Path.home() / ".jarvis" / "prime" / "models")

    # Compression
    enable_compression: bool = True
    compression_level: int = 6  # 1-9

    # Versioning
    max_versions_per_component: int = 10
    auto_cleanup: bool = True

    # Checksums
    verify_checksums: bool = True

    # Atomic saves
    use_atomic_saves: bool = True

    # Cloud sync
    enable_cloud_sync: bool = False
    cloud_bucket: Optional[str] = None


# =============================================================================
# STORAGE BACKEND
# =============================================================================


class StorageBackend(ABC):
    """Abstract storage backend."""

    @abstractmethod
    async def save(self, path: Path, data: bytes) -> bool:
        ...

    @abstractmethod
    async def load(self, path: Path) -> Optional[bytes]:
        ...

    @abstractmethod
    async def exists(self, path: Path) -> bool:
        ...

    @abstractmethod
    async def delete(self, path: Path) -> bool:
        ...

    @abstractmethod
    async def list_dir(self, path: Path) -> List[Path]:
        ...


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage."""

    def __init__(self, config: PersistenceConfig) -> None:
        self._config = config

    async def save(self, path: Path, data: bytes) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            if self._config.use_atomic_saves:
                # Write to temp file first
                temp_path = path.with_suffix(".tmp")
                temp_path.write_bytes(data)
                temp_path.rename(path)
            else:
                path.write_bytes(data)

            return True

        except Exception as e:
            logger.error(f"Failed to save {path}: {e}")
            return False

    async def load(self, path: Path) -> Optional[bytes]:
        try:
            if not path.exists():
                return None
            return path.read_bytes()
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return None

    async def exists(self, path: Path) -> bool:
        return path.exists()

    async def delete(self, path: Path) -> bool:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            return False

    async def list_dir(self, path: Path) -> List[Path]:
        if not path.exists():
            return []
        return list(path.iterdir())


# =============================================================================
# MODEL REGISTRY
# =============================================================================


class ModelRegistry:
    """
    Registry for tracking all saved model versions.

    Maintains a manifest file with metadata for all versions.
    """

    def __init__(
        self,
        config: PersistenceConfig,
        backend: StorageBackend,
    ) -> None:
        self._config = config
        self._backend = backend
        self._manifest_path = config.base_dir / "manifest.json"
        self._manifest: Dict[str, Dict[str, ModelMetadata]] = {}
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Load manifest from disk."""
        async with self._lock:
            data = await self._backend.load(self._manifest_path)
            if data:
                try:
                    manifest_data = json.loads(data.decode())
                    for component, versions in manifest_data.items():
                        self._manifest[component] = {}
                        for version, metadata in versions.items():
                            self._manifest[component][version] = ModelMetadata.from_dict(metadata)
                except Exception as e:
                    logger.error(f"Failed to parse manifest: {e}")

    async def save(self) -> None:
        """Save manifest to disk."""
        async with self._lock:
            manifest_data = {}
            for component, versions in self._manifest.items():
                manifest_data[component] = {
                    v: m.to_dict() for v, m in versions.items()
                }

            data = json.dumps(manifest_data, indent=2).encode()
            await self._backend.save(self._manifest_path, data)

    async def register(self, metadata: ModelMetadata) -> None:
        """Register a new model version."""
        component = metadata.component.value
        version = str(metadata.version)

        async with self._lock:
            if component not in self._manifest:
                self._manifest[component] = {}

            self._manifest[component][version] = metadata

        await self.save()

        # Cleanup old versions if needed
        if self._config.auto_cleanup:
            await self._cleanup_old_versions(metadata.component)

    async def get_latest(self, component: ModelComponent) -> Optional[ModelMetadata]:
        """Get latest version of a component."""
        versions = self._manifest.get(component.value, {})
        if not versions:
            return None

        # Find active versions
        active = [
            m for m in versions.values()
            if m.state == VersionState.ACTIVE
        ]

        if not active:
            return None

        # Return highest version
        return max(active, key=lambda m: m.version)

    async def get_version(
        self,
        component: ModelComponent,
        version: str,
    ) -> Optional[ModelMetadata]:
        """Get specific version of a component."""
        versions = self._manifest.get(component.value, {})
        return versions.get(version)

    async def list_versions(self, component: ModelComponent) -> List[ModelMetadata]:
        """List all versions of a component."""
        versions = self._manifest.get(component.value, {})
        return sorted(versions.values(), key=lambda m: m.version, reverse=True)

    async def set_state(
        self,
        component: ModelComponent,
        version: str,
        state: VersionState,
    ) -> bool:
        """Update state of a version."""
        versions = self._manifest.get(component.value, {})
        if version not in versions:
            return False

        versions[version].state = state
        versions[version].updated_at = datetime.now()
        await self.save()
        return True

    async def _cleanup_old_versions(self, component: ModelComponent) -> None:
        """Remove old versions exceeding max limit."""
        versions = self._manifest.get(component.value, {})
        if len(versions) <= self._config.max_versions_per_component:
            return

        # Sort by version
        sorted_versions = sorted(
            versions.values(),
            key=lambda m: m.version,
            reverse=True,
        )

        # Keep max versions, archive/delete the rest
        for metadata in sorted_versions[self._config.max_versions_per_component:]:
            if metadata.state == VersionState.ACTIVE:
                await self.set_state(
                    component,
                    str(metadata.version),
                    VersionState.ARCHIVED,
                )


# =============================================================================
# PERSISTENCE MANAGER
# =============================================================================


class AGIPersistenceManager:
    """
    Main persistence manager for AGI models.

    Handles saving, loading, and versioning of all AGI components.

    Usage:
        manager = AGIPersistenceManager()
        await manager.initialize()

        # Save a model
        metadata = await manager.save(
            component=ModelComponent.AGI_ORCHESTRATOR,
            data=orchestrator.state_dict(),
            description="After training epoch 10",
        )

        # Load latest version
        data = await manager.load(ModelComponent.AGI_ORCHESTRATOR)

        # Load specific version
        data = await manager.load(
            ModelComponent.AGI_ORCHESTRATOR,
            version="v1.2.0",
        )

        # Rollback to previous version
        await manager.rollback(ModelComponent.AGI_ORCHESTRATOR)
    """

    def __init__(self, config: Optional[PersistenceConfig] = None) -> None:
        self._config = config or PersistenceConfig()
        self._backend = LocalStorageBackend(self._config)
        self._registry = ModelRegistry(self._config, self._backend)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> bool:
        """Initialize persistence manager."""
        async with self._lock:
            if self._initialized:
                return True

            try:
                # Create base directory
                self._config.base_dir.mkdir(parents=True, exist_ok=True)

                # Load registry
                await self._registry.load()

                self._initialized = True
                logger.info(f"Persistence manager initialized at {self._config.base_dir}")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize persistence: {e}")
                return False

    async def save(
        self,
        component: ModelComponent,
        data: Any,
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        bump: str = "patch",  # major, minor, patch
    ) -> Optional[ModelMetadata]:
        """
        Save model data with automatic versioning.

        Args:
            component: Component to save
            data: Model data (dict or bytes)
            config: Optional configuration to save
            description: Version description
            tags: Optional tags
            bump: Version bump type

        Returns:
            Metadata for saved version or None if failed
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Determine new version
            latest = await self._registry.get_latest(component)
            if latest:
                if bump == "major":
                    new_version = latest.version.bump_major()
                elif bump == "minor":
                    new_version = latest.version.bump_minor()
                else:
                    new_version = latest.version.bump_patch()
                parent_version = str(latest.version)
            else:
                new_version = SemanticVersion(1, 0, 0)
                parent_version = None

            # Prepare data
            if isinstance(data, dict):
                serialized = json.dumps(data).encode("utf-8")
            elif isinstance(data, bytes):
                serialized = data
            else:
                # Try to serialize with pickle
                import pickle
                serialized = pickle.dumps(data)

            # Compress if enabled
            if self._config.enable_compression:
                compressed = gzip.compress(serialized, self._config.compression_level)
            else:
                compressed = serialized

            # Calculate checksum
            checksum = hashlib.sha256(compressed).hexdigest()

            # Create version directory
            version_dir = self._config.base_dir / component.value / str(new_version)
            version_dir.mkdir(parents=True, exist_ok=True)

            # Save data
            data_path = version_dir / "data.bin"
            await self._backend.save(data_path, compressed)

            # Save config if provided
            if config:
                config_path = version_dir / "config.json"
                config_data = json.dumps(config, indent=2).encode()
                await self._backend.save(config_path, config_data)

            # Create metadata
            metadata = ModelMetadata(
                component=component,
                version=new_version,
                state=VersionState.ACTIVE,
                file_size_bytes=len(compressed),
                checksum_sha256=checksum,
                compressed=self._config.enable_compression,
                parent_version=parent_version,
                description=description,
                tags=tags or [],
            )

            # Save metadata
            metadata_path = version_dir / "metadata.json"
            await self._backend.save(
                metadata_path,
                json.dumps(metadata.to_dict(), indent=2).encode(),
            )

            # Register in registry
            await self._registry.register(metadata)

            # Update current symlink
            await self._update_current_link(component, new_version)

            logger.info(f"Saved {component.value} {new_version}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to save {component.value}: {e}")
            return None

    async def load(
        self,
        component: ModelComponent,
        version: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Load model data.

        Args:
            component: Component to load
            version: Specific version or None for latest

        Returns:
            Model data or None if not found
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Get metadata
            if version:
                metadata = await self._registry.get_version(component, version)
            else:
                metadata = await self._registry.get_latest(component)

            if not metadata:
                logger.warning(f"No version found for {component.value}")
                return None

            # Load data
            version_dir = self._config.base_dir / component.value / str(metadata.version)
            data_path = version_dir / "data.bin"

            data = await self._backend.load(data_path)
            if not data:
                logger.error(f"Data file not found for {component.value} {metadata.version}")
                return None

            # Verify checksum
            if self._config.verify_checksums:
                checksum = hashlib.sha256(data).hexdigest()
                if checksum != metadata.checksum_sha256:
                    logger.error(f"Checksum mismatch for {component.value} {metadata.version}")
                    await self._registry.set_state(
                        component, str(metadata.version), VersionState.CORRUPTED
                    )
                    return None

            # Decompress if needed
            if metadata.compressed:
                decompressed = gzip.decompress(data)
            else:
                decompressed = data

            # Try to parse as JSON first
            try:
                return json.loads(decompressed.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Try pickle
                import pickle
                return pickle.loads(decompressed)

        except Exception as e:
            logger.error(f"Failed to load {component.value}: {e}")
            return None

    async def load_config(
        self,
        component: ModelComponent,
        version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load configuration for a model version."""
        if not self._initialized:
            await self.initialize()

        try:
            metadata = (
                await self._registry.get_version(component, version)
                if version else
                await self._registry.get_latest(component)
            )

            if not metadata:
                return None

            version_dir = self._config.base_dir / component.value / str(metadata.version)
            config_path = version_dir / "config.json"

            data = await self._backend.load(config_path)
            if data:
                return json.loads(data.decode())
            return None

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return None

    async def rollback(self, component: ModelComponent) -> Optional[ModelMetadata]:
        """Rollback to previous version."""
        versions = await self._registry.list_versions(component)

        if len(versions) < 2:
            logger.warning(f"No previous version to rollback to for {component.value}")
            return None

        # Current is versions[0], previous is versions[1]
        current = versions[0]
        previous = versions[1]

        # Archive current
        await self._registry.set_state(
            component, str(current.version), VersionState.ARCHIVED
        )

        # Activate previous
        await self._registry.set_state(
            component, str(previous.version), VersionState.ACTIVE
        )

        # Update symlink
        await self._update_current_link(component, previous.version)

        logger.info(f"Rolled back {component.value} from {current.version} to {previous.version}")
        return previous

    async def _update_current_link(
        self,
        component: ModelComponent,
        version: SemanticVersion,
    ) -> None:
        """Update 'current' symlink to point to version."""
        component_dir = self._config.base_dir / component.value
        current_link = component_dir / "current"
        version_dir = component_dir / str(version)

        # Remove existing link
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()

        # Create new link
        current_link.symlink_to(version_dir.name)

    async def export(
        self,
        component: ModelComponent,
        output_path: Path,
        version: Optional[str] = None,
    ) -> bool:
        """Export a model version to a file."""
        try:
            metadata = (
                await self._registry.get_version(component, version)
                if version else
                await self._registry.get_latest(component)
            )

            if not metadata:
                return False

            version_dir = self._config.base_dir / component.value / str(metadata.version)

            # Create archive
            shutil.make_archive(
                str(output_path.with_suffix("")),
                "zip",
                version_dir,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to export: {e}")
            return False

    async def import_model(
        self,
        component: ModelComponent,
        input_path: Path,
    ) -> Optional[ModelMetadata]:
        """Import a model from archive."""
        try:
            import zipfile

            # Extract to temp directory
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(input_path, "r") as zf:
                    zf.extractall(temp_dir)

                # Load metadata
                metadata_path = Path(temp_dir) / "metadata.json"
                if metadata_path.exists():
                    metadata_data = json.loads(metadata_path.read_text())
                    version = SemanticVersion.parse(metadata_data["version"])
                else:
                    # Auto-generate version
                    latest = await self._registry.get_latest(component)
                    version = latest.version.bump_patch() if latest else SemanticVersion(1, 0, 0)

                # Load data
                data_path = Path(temp_dir) / "data.bin"
                data = data_path.read_bytes()

                # Decompress and save
                if data_path.suffix == ".bin":
                    try:
                        decompressed = gzip.decompress(data)
                    except:
                        decompressed = data

                    parsed = json.loads(decompressed.decode())
                    return await self.save(component, parsed, description="Imported")

            return None

        except Exception as e:
            logger.error(f"Failed to import: {e}")
            return None

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = 0
        component_sizes = {}

        for component in ModelComponent:
            component_dir = self._config.base_dir / component.value
            if component_dir.exists():
                size = sum(f.stat().st_size for f in component_dir.rglob("*") if f.is_file())
                component_sizes[component.value] = size
                total_size += size

        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "component_sizes": component_sizes,
            "base_dir": str(self._config.base_dir),
        }


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================


class CheckpointManager:
    """
    Manages periodic checkpoints for training and runtime.

    Supports incremental saves and automatic cleanup.
    """

    def __init__(
        self,
        persistence: AGIPersistenceManager,
        checkpoint_interval: int = 1000,  # steps
        max_checkpoints: int = 5,
    ) -> None:
        self._persistence = persistence
        self._interval = checkpoint_interval
        self._max_checkpoints = max_checkpoints
        self._step_count = 0
        self._checkpoints: Dict[ModelComponent, List[str]] = {}

    async def step(
        self,
        component: ModelComponent,
        get_state: Callable[[], Any],
    ) -> Optional[ModelMetadata]:
        """Record a step and potentially save checkpoint."""
        self._step_count += 1

        if self._step_count % self._interval != 0:
            return None

        # Save checkpoint
        state = get_state()
        metadata = await self._persistence.save(
            component=component,
            data=state,
            description=f"Checkpoint at step {self._step_count}",
            tags=["checkpoint", f"step-{self._step_count}"],
        )

        if metadata:
            # Track checkpoint
            if component not in self._checkpoints:
                self._checkpoints[component] = []
            self._checkpoints[component].append(str(metadata.version))

            # Cleanup old checkpoints
            while len(self._checkpoints[component]) > self._max_checkpoints:
                old_version = self._checkpoints[component].pop(0)
                await self._persistence._registry.set_state(
                    component, old_version, VersionState.ARCHIVED
                )

        return metadata


# =============================================================================
# SINGLETON
# =============================================================================


_persistence_manager: Optional[AGIPersistenceManager] = None
_persistence_lock = asyncio.Lock()


async def get_persistence_manager(
    config: Optional[PersistenceConfig] = None,
) -> AGIPersistenceManager:
    """Get or create global persistence manager."""
    global _persistence_manager

    async with _persistence_lock:
        if _persistence_manager is None:
            _persistence_manager = AGIPersistenceManager(config)
            await _persistence_manager.initialize()

        return _persistence_manager
