"""
Model Registry & Versioning System
===================================

Tracks model lineage, versions, and handles rollbacks.
Zero hardcoding - all configuration driven dynamically.

Features:
- Semantic versioning (v1.0-base, v1.1-weekly-2025-05-12)
- Active version tracking with rollback capability
- Reactor-core integration for update notifications
- Async file watching for new model arrivals
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
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

logger = logging.getLogger(__name__)


class VersionState(Enum):
    """State of a model version in the registry"""
    PENDING = "pending"          # Detected but not validated
    VALIDATING = "validating"    # Running validation checks
    VALIDATED = "validated"      # Passed validation, ready to use
    ACTIVE = "active"            # Currently serving requests
    STANDBY = "standby"          # Loaded in background, ready for swap
    DEPRECATED = "deprecated"    # Marked for removal
    FAILED = "failed"            # Failed validation or loading
    ARCHIVED = "archived"        # Moved to cold storage


@dataclass
class ModelMetadata:
    """Rich metadata for model versions"""
    created_at: datetime
    source: str                  # "reactor-core", "manual", "huggingface"
    training_config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    compatibility: Dict[str, str] = field(default_factory=dict)
    checksum: Optional[str] = None
    size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "training_config": self.training_config,
            "performance_metrics": self.performance_metrics,
            "capabilities": self.capabilities,
            "compatibility": self.compatibility,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        return cls(
            created_at=datetime.fromisoformat(data["created_at"]),
            source=data.get("source", "unknown"),
            training_config=data.get("training_config", {}),
            performance_metrics=data.get("performance_metrics", {}),
            capabilities=data.get("capabilities", []),
            compatibility=data.get("compatibility", {}),
            checksum=data.get("checksum"),
            size_bytes=data.get("size_bytes", 0),
        )


@dataclass
class ModelVersion:
    """
    Represents a specific version of a model.

    Version format: v{major}.{minor}[-{variant}][-{date}]
    Examples:
        - v1.0-base
        - v1.1-weekly-2025-05-12
        - v2.0-dpo-trained
    """
    version_id: str              # Unique identifier (e.g., "v1.1-weekly-2025-05-12")
    model_path: Path             # Path to model files
    state: VersionState = VersionState.PENDING
    metadata: ModelMetadata = field(default_factory=lambda: ModelMetadata(
        created_at=datetime.now(),
        source="unknown"
    ))

    # Runtime tracking
    load_count: int = 0
    total_requests: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    last_used: Optional[datetime] = None

    # Version parsing
    _parsed: bool = field(default=False, repr=False)
    _major: int = field(default=0, repr=False)
    _minor: int = field(default=0, repr=False)
    _variant: str = field(default="", repr=False)
    _date: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Parse version string into components"""
        if not self._parsed:
            self._parse_version()

    def _parse_version(self):
        """Parse version ID into components"""
        # Pattern: v{major}.{minor}[-{variant}][-{YYYY-MM-DD}]
        pattern = r"v(\d+)\.(\d+)(?:-([a-zA-Z]+(?:-[a-zA-Z]+)*))?(?:-(\d{4}-\d{2}-\d{2}))?$"
        match = re.match(pattern, self.version_id)

        if match:
            self._major = int(match.group(1))
            self._minor = int(match.group(2))
            self._variant = match.group(3) or "base"
            self._date = match.group(4)
        else:
            # Fallback: treat entire string as variant
            self._major = 1
            self._minor = 0
            self._variant = self.version_id

        self._parsed = True

    @property
    def major(self) -> int:
        return self._major

    @property
    def minor(self) -> int:
        return self._minor

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def date(self) -> Optional[str]:
        return self._date

    def is_newer_than(self, other: "ModelVersion") -> bool:
        """Check if this version is newer than another"""
        if self._major != other._major:
            return self._major > other._major
        if self._minor != other._minor:
            return self._minor > other._minor
        if self._date and other._date:
            return self._date > other._date
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_path": str(self.model_path),
            "state": self.state.value,
            "metadata": self.metadata.to_dict(),
            "load_count": self.load_count,
            "total_requests": self.total_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "error_count": self.error_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        return cls(
            version_id=data["version_id"],
            model_path=Path(data["model_path"]),
            state=VersionState(data.get("state", "pending")),
            metadata=ModelMetadata.from_dict(data.get("metadata", {
                "created_at": datetime.now().isoformat(),
                "source": "unknown"
            })),
            load_count=data.get("load_count", 0),
            total_requests=data.get("total_requests", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            error_count=data.get("error_count", 0),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
        )


@dataclass
class ModelLineage:
    """
    Tracks the complete lineage of a model family.

    A lineage represents all versions of a specific model type
    (e.g., all versions of "jarvis-prime-7b").
    """
    name: str                                    # Model family name
    description: str = ""
    versions: Dict[str, ModelVersion] = field(default_factory=dict)
    active_version_id: Optional[str] = None      # Currently serving version
    rollback_version_id: Optional[str] = None    # Previous version for rollback

    # Configuration
    auto_activate_new: bool = True               # Auto-activate validated versions
    max_versions: int = 10                       # Max versions to keep
    archive_old: bool = True                     # Archive instead of delete

    def get_active(self) -> Optional[ModelVersion]:
        """Get the currently active version"""
        if self.active_version_id:
            return self.versions.get(self.active_version_id)
        return None

    def get_rollback(self) -> Optional[ModelVersion]:
        """Get the rollback version"""
        if self.rollback_version_id:
            return self.versions.get(self.rollback_version_id)
        return None

    def get_latest(self) -> Optional[ModelVersion]:
        """Get the latest validated version"""
        validated = [
            v for v in self.versions.values()
            if v.state in (VersionState.VALIDATED, VersionState.ACTIVE, VersionState.STANDBY)
        ]
        if not validated:
            return None
        return max(validated, key=lambda v: (v.major, v.minor, v.date or ""))

    def add_version(self, version: ModelVersion) -> None:
        """Add a new version to the lineage"""
        self.versions[version.version_id] = version
        logger.info(f"Added version {version.version_id} to lineage {self.name}")

        # Cleanup old versions if needed
        self._cleanup_old_versions()

    def _cleanup_old_versions(self) -> None:
        """Remove or archive old versions beyond max_versions"""
        if len(self.versions) <= self.max_versions:
            return

        # Sort by version, keep newest
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: (v.major, v.minor, v.date or ""),
            reverse=True
        )

        for version in sorted_versions[self.max_versions:]:
            if version.version_id in (self.active_version_id, self.rollback_version_id):
                continue  # Never remove active or rollback

            if self.archive_old:
                version.state = VersionState.ARCHIVED
            else:
                del self.versions[version.version_id]

            logger.info(f"Cleaned up old version: {version.version_id}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "active_version_id": self.active_version_id,
            "rollback_version_id": self.rollback_version_id,
            "auto_activate_new": self.auto_activate_new,
            "max_versions": self.max_versions,
            "archive_old": self.archive_old,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelLineage":
        lineage = cls(
            name=data["name"],
            description=data.get("description", ""),
            active_version_id=data.get("active_version_id"),
            rollback_version_id=data.get("rollback_version_id"),
            auto_activate_new=data.get("auto_activate_new", True),
            max_versions=data.get("max_versions", 10),
            archive_old=data.get("archive_old", True),
        )

        for version_id, version_data in data.get("versions", {}).items():
            lineage.versions[version_id] = ModelVersion.from_dict(version_data)

        return lineage


class ModelValidator(Protocol):
    """Protocol for model validation strategies"""

    async def validate(self, version: ModelVersion) -> bool:
        """Validate a model version before activation"""
        ...


class DefaultModelValidator:
    """Default model validator - checks file existence and integrity"""

    async def validate(self, version: ModelVersion) -> bool:
        """Validate model by checking path and checksum"""
        try:
            # Check path exists
            if not version.model_path.exists():
                logger.error(f"Model path does not exist: {version.model_path}")
                return False

            # Check file integrity if checksum provided
            if version.metadata.checksum:
                actual_checksum = await self._compute_checksum(version.model_path)
                if actual_checksum != version.metadata.checksum:
                    logger.error(f"Checksum mismatch for {version.version_id}")
                    return False

            # Basic loading test could go here
            logger.info(f"Validated model version: {version.version_id}")
            return True

        except Exception as e:
            logger.error(f"Validation failed for {version.version_id}: {e}")
            return False

    async def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of model files"""
        sha256 = hashlib.sha256()

        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
        else:
            # Directory: hash all files recursively
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            sha256.update(chunk)

        return sha256.hexdigest()


class ModelRegistry:
    """
    Central registry for all model versions and lineages.

    Provides:
    - Version tracking and lineage management
    - Automatic model discovery from reactor-core
    - Validation pipeline before activation
    - Rollback capability for failed deployments
    - Async file watching for new model arrivals

    Usage:
        registry = ModelRegistry(models_dir="./models")
        await registry.start()  # Start file watcher

        # Register a new version
        version = await registry.register_version(
            lineage_name="jarvis-prime-7b",
            version_id="v1.1-weekly-2025-05-12",
            model_path=Path("./models/jarvis-7b-v1.1"),
            metadata={...}
        )

        # Get active version for inference
        active = registry.get_active_version("jarvis-prime-7b")
    """

    def __init__(
        self,
        models_dir: str | Path = "./models",
        registry_file: str | Path | None = None,
        validator: Optional[ModelValidator] = None,
        reactor_core_watch_dir: str | Path | None = None,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = Path(registry_file) if registry_file else self.models_dir / "registry.json"
        self.validator = validator or DefaultModelValidator()
        self.reactor_core_watch_dir = Path(reactor_core_watch_dir) if reactor_core_watch_dir else None

        # State
        self.lineages: Dict[str, ModelLineage] = {}
        self._watch_task: Optional[asyncio.Task] = None
        self._running = False

        # Callbacks
        self._on_new_version_callbacks: List[Callable[[ModelVersion], None]] = []
        self._on_activation_callbacks: List[Callable[[ModelVersion], None]] = []

        # Load existing registry
        self._load_registry()

        logger.info(f"ModelRegistry initialized with {len(self.lineages)} lineages")

    def _load_registry(self) -> None:
        """Load registry state from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    data = json.load(f)

                for lineage_data in data.get("lineages", []):
                    lineage = ModelLineage.from_dict(lineage_data)
                    self.lineages[lineage.name] = lineage

                logger.info(f"Loaded {len(self.lineages)} lineages from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Persist registry state to disk"""
        try:
            data = {
                "lineages": [l.to_dict() for l in self.lineages.values()],
                "updated_at": datetime.now().isoformat(),
            }

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Registry saved to disk")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    async def start(self) -> None:
        """Start the registry, including file watcher"""
        self._running = True

        if self.reactor_core_watch_dir:
            self._watch_task = asyncio.create_task(self._watch_for_new_models())
            logger.info(f"Started watching {self.reactor_core_watch_dir} for new models")

    async def stop(self) -> None:
        """Stop the registry and file watcher"""
        self._running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        self._save_registry()
        logger.info("Registry stopped")

    async def _watch_for_new_models(self) -> None:
        """Watch for new model files from reactor-core"""
        if not self.reactor_core_watch_dir:
            return

        seen_files: set = set()

        while self._running:
            try:
                watch_dir = Path(self.reactor_core_watch_dir)
                if watch_dir.exists():
                    for model_dir in watch_dir.iterdir():
                        if model_dir.is_dir() and model_dir not in seen_files:
                            seen_files.add(model_dir)

                            # Check for manifest file
                            manifest = model_dir / "manifest.json"
                            if manifest.exists():
                                await self._process_new_model(model_dir, manifest)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in model watcher: {e}")
                await asyncio.sleep(10)

    async def _process_new_model(self, model_dir: Path, manifest_path: Path) -> None:
        """Process a newly discovered model from reactor-core"""
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)

            lineage_name = manifest.get("lineage", "jarvis-prime")
            version_id = manifest.get("version", f"v1.0-{datetime.now().strftime('%Y-%m-%d')}")

            await self.register_version(
                lineage_name=lineage_name,
                version_id=version_id,
                model_path=model_dir,
                metadata=ModelMetadata(
                    created_at=datetime.now(),
                    source="reactor-core",
                    training_config=manifest.get("training_config", {}),
                    performance_metrics=manifest.get("metrics", {}),
                    capabilities=manifest.get("capabilities", []),
                    checksum=manifest.get("checksum"),
                ),
            )

            logger.info(f"Processed new model from reactor-core: {version_id}")

        except Exception as e:
            logger.error(f"Failed to process new model from {model_dir}: {e}")

    async def register_version(
        self,
        lineage_name: str,
        version_id: str,
        model_path: Path,
        metadata: Optional[ModelMetadata] = None,
        auto_validate: bool = True,
        auto_activate: bool = True,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            lineage_name: Name of the model lineage
            version_id: Version identifier (e.g., "v1.1-weekly-2025-05-12")
            model_path: Path to model files
            metadata: Optional model metadata
            auto_validate: Run validation automatically
            auto_activate: Activate after validation if lineage allows

        Returns:
            The registered ModelVersion
        """
        # Get or create lineage
        if lineage_name not in self.lineages:
            self.lineages[lineage_name] = ModelLineage(name=lineage_name)

        lineage = self.lineages[lineage_name]

        # Create version
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            state=VersionState.PENDING,
            metadata=metadata or ModelMetadata(created_at=datetime.now(), source="manual"),
        )

        lineage.add_version(version)

        # Notify callbacks
        for callback in self._on_new_version_callbacks:
            try:
                callback(version)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        # Auto-validate
        if auto_validate:
            await self.validate_version(lineage_name, version_id)

            # Auto-activate if validation passed and lineage allows
            if auto_activate and lineage.auto_activate_new and version.state == VersionState.VALIDATED:
                await self.activate_version(lineage_name, version_id)

        self._save_registry()
        return version

    async def validate_version(self, lineage_name: str, version_id: str) -> bool:
        """Validate a model version"""
        lineage = self.lineages.get(lineage_name)
        if not lineage:
            raise ValueError(f"Unknown lineage: {lineage_name}")

        version = lineage.versions.get(version_id)
        if not version:
            raise ValueError(f"Unknown version: {version_id}")

        version.state = VersionState.VALIDATING

        try:
            valid = await self.validator.validate(version)
            version.state = VersionState.VALIDATED if valid else VersionState.FAILED
            self._save_registry()
            return valid
        except Exception as e:
            version.state = VersionState.FAILED
            logger.error(f"Validation error: {e}")
            self._save_registry()
            return False

    async def activate_version(self, lineage_name: str, version_id: str) -> bool:
        """
        Activate a model version for serving.

        The previous active version becomes the rollback version.
        """
        lineage = self.lineages.get(lineage_name)
        if not lineage:
            raise ValueError(f"Unknown lineage: {lineage_name}")

        version = lineage.versions.get(version_id)
        if not version:
            raise ValueError(f"Unknown version: {version_id}")

        if version.state not in (VersionState.VALIDATED, VersionState.STANDBY):
            raise ValueError(f"Cannot activate version in state: {version.state}")

        # Store current active as rollback
        if lineage.active_version_id:
            old_active = lineage.versions.get(lineage.active_version_id)
            if old_active:
                old_active.state = VersionState.STANDBY
                lineage.rollback_version_id = lineage.active_version_id

        # Activate new version
        lineage.active_version_id = version_id
        version.state = VersionState.ACTIVE

        # Notify callbacks
        for callback in self._on_activation_callbacks:
            try:
                callback(version)
            except Exception as e:
                logger.error(f"Activation callback error: {e}")

        self._save_registry()
        logger.info(f"Activated version {version_id} for {lineage_name}")
        return True

    async def rollback(self, lineage_name: str) -> bool:
        """
        Rollback to the previous version.

        Swaps active and rollback versions.
        """
        lineage = self.lineages.get(lineage_name)
        if not lineage:
            raise ValueError(f"Unknown lineage: {lineage_name}")

        if not lineage.rollback_version_id:
            raise ValueError("No rollback version available")

        rollback_version = lineage.versions.get(lineage.rollback_version_id)
        if not rollback_version:
            raise ValueError("Rollback version not found")

        # Swap versions
        old_active_id = lineage.active_version_id
        lineage.active_version_id = lineage.rollback_version_id
        lineage.rollback_version_id = old_active_id

        # Update states
        rollback_version.state = VersionState.ACTIVE
        if old_active_id:
            old_active = lineage.versions.get(old_active_id)
            if old_active:
                old_active.state = VersionState.STANDBY

        self._save_registry()
        logger.info(f"Rolled back {lineage_name} to {lineage.active_version_id}")
        return True

    def get_active_version(self, lineage_name: str) -> Optional[ModelVersion]:
        """Get the currently active version for a lineage"""
        lineage = self.lineages.get(lineage_name)
        if lineage:
            return lineage.get_active()
        return None

    def get_all_lineages(self) -> List[ModelLineage]:
        """Get all registered lineages"""
        return list(self.lineages.values())

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive registry status"""
        return {
            "total_lineages": len(self.lineages),
            "total_versions": sum(len(l.versions) for l in self.lineages.values()),
            "active_versions": [
                {
                    "lineage": l.name,
                    "version": l.active_version_id,
                    "state": l.versions[l.active_version_id].state.value if l.active_version_id else None,
                }
                for l in self.lineages.values()
                if l.active_version_id
            ],
            "watching": str(self.reactor_core_watch_dir) if self.reactor_core_watch_dir else None,
            "running": self._running,
        }

    def on_new_version(self, callback: Callable[[ModelVersion], None]) -> None:
        """Register callback for new version events"""
        self._on_new_version_callbacks.append(callback)

    def on_activation(self, callback: Callable[[ModelVersion], None]) -> None:
        """Register callback for activation events"""
        self._on_activation_callbacks.append(callback)
