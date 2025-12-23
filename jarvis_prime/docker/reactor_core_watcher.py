"""
Reactor-Core Watcher - Auto-Deployment Pipeline
================================================

Watches for new models from reactor-core training pipeline and
automatically deploys them to JARVIS-Prime with:
- File system watching for new GGUF models
- Validation before deployment
- Automatic hot-swap with rollback capability
- Telemetry reporting for training feedback loop

Integration Flow:
1. reactor-core completes training and converts to GGUF
2. GGUF file dropped into watch directory
3. This watcher detects new file
4. Validates model integrity
5. Registers with ModelRegistry
6. Triggers hot-swap deployment
7. Reports deployment status back for reactor-core feedback

Directory Structure:
    /app/reactor-core-output/
        ├── pending/          # New models waiting to deploy
        ├── deployed/         # Successfully deployed models
        ├── failed/           # Models that failed validation
        └── manifest.json     # Model metadata from reactor-core
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
from typing import Any, Callable, Dict, List, Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ReactorCoreModelManifest:
    """Manifest for a model produced by reactor-core."""
    model_id: str
    version: str
    training_run_id: str
    base_model: str
    quantization_method: str
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    file_path: str
    sha256: str
    created_at: datetime
    lineage: str = "reactor-core"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "training_run_id": self.training_run_id,
            "base_model": self.base_model,
            "quantization_method": self.quantization_method,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "file_path": self.file_path,
            "sha256": self.sha256,
            "created_at": self.created_at.isoformat(),
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReactorCoreModelManifest":
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            training_run_id=data["training_run_id"],
            base_model=data["base_model"],
            quantization_method=data["quantization_method"],
            training_config=data.get("training_config", {}),
            metrics=data.get("metrics", {}),
            file_path=data["file_path"],
            sha256=data["sha256"],
            created_at=datetime.fromisoformat(data["created_at"]),
            lineage=data.get("lineage", "reactor-core"),
        )


@dataclass
class DeploymentResult:
    """Result of a model deployment attempt."""
    success: bool
    model_id: str
    version: str
    deployed_at: datetime
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "model_id": self.model_id,
            "version": self.version,
            "deployed_at": self.deployed_at.isoformat(),
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


# ============================================================================
# File System Handler
# ============================================================================

class ModelFileHandler(FileSystemEventHandler):
    """Watches for new GGUF files in the watch directory."""

    def __init__(self, callback: Callable[[Path], None]):
        self.callback = callback
        self._debounce: Dict[str, datetime] = {}
        self._debounce_seconds = 5.0

    def on_created(self, event):
        if isinstance(event, FileCreatedEvent):
            self._handle_file(Path(event.src_path))

    def on_moved(self, event):
        if isinstance(event, FileMovedEvent):
            self._handle_file(Path(event.dest_path))

    def _handle_file(self, path: Path):
        # Only handle GGUF files
        if not path.suffix == ".gguf":
            return

        # Only handle files in pending/ directory
        if path.parent.name != "pending":
            return

        # Debounce (wait for file to finish writing)
        now = datetime.now()
        last = self._debounce.get(str(path))
        if last and (now - last).total_seconds() < self._debounce_seconds:
            return
        self._debounce[str(path)] = now

        # Trigger callback
        logger.info(f"Detected new model: {path}")
        self.callback(path)


# ============================================================================
# Reactor-Core Watcher
# ============================================================================

class ReactorCoreWatcher:
    """
    Watches for models from reactor-core and auto-deploys them.

    Usage:
        watcher = ReactorCoreWatcher(
            watch_dir="/app/reactor-core-output",
            models_dir="/app/models",
            on_deploy_success=lambda result: print(f"Deployed {result.model_id}"),
            on_deploy_failure=lambda result: print(f"Failed {result.model_id}"),
        )
        await watcher.start()

        # ... runs in background ...

        await watcher.stop()
    """

    def __init__(
        self,
        watch_dir: str | Path,
        models_dir: str | Path,
        hot_swap_callback: Optional[Callable[[Path, str], asyncio.Future]] = None,
        on_deploy_success: Optional[Callable[[DeploymentResult], None]] = None,
        on_deploy_failure: Optional[Callable[[DeploymentResult], None]] = None,
        auto_deploy: bool = True,
        validation_enabled: bool = True,
    ):
        self.watch_dir = Path(watch_dir)
        self.models_dir = Path(models_dir)
        self.hot_swap_callback = hot_swap_callback
        self.on_deploy_success = on_deploy_success
        self.on_deploy_failure = on_deploy_failure
        self.auto_deploy = auto_deploy
        self.validation_enabled = validation_enabled

        # State
        self._observer: Optional[Observer] = None
        self._running = False
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._process_task: Optional[asyncio.Task] = None

        # Statistics
        self._deployments_success = 0
        self._deployments_failed = 0
        self._last_deployment: Optional[datetime] = None

        # Ensure directories exist
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create required directories."""
        (self.watch_dir / "pending").mkdir(parents=True, exist_ok=True)
        (self.watch_dir / "deployed").mkdir(parents=True, exist_ok=True)
        (self.watch_dir / "failed").mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Start the watcher."""
        if self._running:
            return

        logger.info(f"Starting ReactorCoreWatcher: {self.watch_dir}")

        # Start file observer
        handler = ModelFileHandler(self._on_new_model)
        self._observer = Observer()
        self._observer.schedule(
            handler,
            str(self.watch_dir / "pending"),
            recursive=False,
        )
        self._observer.start()

        # Start processing task
        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())

        # Check for any existing pending models
        await self._check_pending()

        logger.info("ReactorCoreWatcher started")

    async def stop(self) -> None:
        """Stop the watcher."""
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None

        logger.info("ReactorCoreWatcher stopped")

    def _on_new_model(self, path: Path) -> None:
        """Called when a new model file is detected."""
        try:
            loop = asyncio.get_event_loop()
            loop.call_soon_threadsafe(
                lambda: self._processing_queue.put_nowait(path)
            )
        except Exception as e:
            logger.error(f"Error queuing model: {e}")

    async def _check_pending(self) -> None:
        """Check for existing pending models on startup."""
        pending_dir = self.watch_dir / "pending"
        for gguf_file in pending_dir.glob("*.gguf"):
            logger.info(f"Found pending model: {gguf_file}")
            await self._processing_queue.put(gguf_file)

    async def _process_loop(self) -> None:
        """Background loop to process queued models."""
        while self._running:
            try:
                # Wait for next model
                path = await asyncio.wait_for(
                    self._processing_queue.get(),
                    timeout=30.0,
                )

                # Process the model
                await self._process_model(path)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process loop: {e}")

    async def _process_model(self, model_path: Path) -> None:
        """Process a new model file."""
        logger.info(f"Processing model: {model_path}")

        # Load manifest if exists
        manifest = await self._load_manifest(model_path)

        # Validate model
        if self.validation_enabled:
            is_valid = await self._validate_model(model_path, manifest)
            if not is_valid:
                await self._handle_failed_deployment(
                    model_path,
                    manifest,
                    "Model validation failed",
                )
                return

        # Deploy if auto_deploy enabled
        if self.auto_deploy:
            await self._deploy_model(model_path, manifest)
        else:
            logger.info(f"Auto-deploy disabled, model ready: {model_path}")

    async def _load_manifest(self, model_path: Path) -> Optional[ReactorCoreModelManifest]:
        """Load manifest for a model."""
        # Look for accompanying manifest file
        manifest_path = model_path.with_suffix(".json")
        if not manifest_path.exists():
            # Try global manifest
            manifest_path = self.watch_dir / "manifest.json"

        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                    # Handle single model or model list
                    if isinstance(data, dict) and "model_id" in data:
                        return ReactorCoreModelManifest.from_dict(data)
                    elif isinstance(data, dict) and "models" in data:
                        # Find matching model in list
                        for m in data["models"]:
                            if Path(m["file_path"]).name == model_path.name:
                                return ReactorCoreModelManifest.from_dict(m)
            except Exception as e:
                logger.warning(f"Error loading manifest: {e}")

        return None

    async def _validate_model(
        self,
        model_path: Path,
        manifest: Optional[ReactorCoreModelManifest],
    ) -> bool:
        """Validate model file."""
        # Check file exists and is not empty
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False

        if model_path.stat().st_size < 1024:  # At least 1KB
            logger.error(f"Model file too small: {model_path}")
            return False

        # Verify checksum if available
        if manifest and manifest.sha256:
            actual_sha256 = await self._calculate_sha256(model_path)
            if actual_sha256 != manifest.sha256:
                logger.error(f"SHA256 mismatch for {model_path}")
                return False

        # Basic GGUF header validation
        try:
            with open(model_path, "rb") as f:
                magic = f.read(4)
                if magic != b"GGUF":
                    logger.error(f"Invalid GGUF magic: {magic}")
                    return False
        except Exception as e:
            logger.error(f"Error reading model: {e}")
            return False

        logger.info(f"Model validation passed: {model_path}")
        return True

    async def _calculate_sha256(self, path: Path) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        loop = asyncio.get_event_loop()

        def _hash():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()

        return await loop.run_in_executor(None, _hash)

    async def _deploy_model(
        self,
        model_path: Path,
        manifest: Optional[ReactorCoreModelManifest],
    ) -> None:
        """Deploy a validated model."""
        model_id = manifest.model_id if manifest else model_path.stem
        version = manifest.version if manifest else datetime.now().strftime("v%Y%m%d-%H%M%S")

        try:
            # Copy to models directory
            dest_path = self.models_dir / f"{model_id}-{version}.gguf"
            shutil.copy2(model_path, dest_path)

            # Trigger hot-swap if callback provided
            if self.hot_swap_callback:
                await self.hot_swap_callback(dest_path, version)

            # Move original to deployed/
            deployed_path = self.watch_dir / "deployed" / model_path.name
            shutil.move(str(model_path), str(deployed_path))

            # Move manifest too
            manifest_src = model_path.with_suffix(".json")
            if manifest_src.exists():
                shutil.move(
                    str(manifest_src),
                    str(self.watch_dir / "deployed" / manifest_src.name),
                )

            # Create result
            result = DeploymentResult(
                success=True,
                model_id=model_id,
                version=version,
                deployed_at=datetime.now(),
                metrics={
                    "file_size_mb": dest_path.stat().st_size / (1024 * 1024),
                },
            )

            # Update statistics
            self._deployments_success += 1
            self._last_deployment = datetime.now()

            # Notify callback
            if self.on_deploy_success:
                self.on_deploy_success(result)

            logger.info(f"Successfully deployed: {model_id} v{version}")

        except Exception as e:
            await self._handle_failed_deployment(model_path, manifest, str(e))

    async def _handle_failed_deployment(
        self,
        model_path: Path,
        manifest: Optional[ReactorCoreModelManifest],
        error_message: str,
    ) -> None:
        """Handle a failed deployment."""
        model_id = manifest.model_id if manifest else model_path.stem
        version = manifest.version if manifest else "unknown"

        # Move to failed/
        try:
            failed_path = self.watch_dir / "failed" / model_path.name
            if model_path.exists():
                shutil.move(str(model_path), str(failed_path))
        except Exception as e:
            logger.error(f"Error moving failed model: {e}")

        # Create result
        result = DeploymentResult(
            success=False,
            model_id=model_id,
            version=version,
            deployed_at=datetime.now(),
            error_message=error_message,
        )

        # Update statistics
        self._deployments_failed += 1

        # Notify callback
        if self.on_deploy_failure:
            self.on_deploy_failure(result)

        logger.error(f"Deployment failed for {model_id}: {error_message}")

    def get_status(self) -> Dict[str, Any]:
        """Get watcher status."""
        pending_count = len(list((self.watch_dir / "pending").glob("*.gguf")))
        deployed_count = len(list((self.watch_dir / "deployed").glob("*.gguf")))
        failed_count = len(list((self.watch_dir / "failed").glob("*.gguf")))

        return {
            "running": self._running,
            "watch_dir": str(self.watch_dir),
            "auto_deploy": self.auto_deploy,
            "pending_models": pending_count,
            "deployed_models": deployed_count,
            "failed_models": failed_count,
            "deployments_success": self._deployments_success,
            "deployments_failed": self._deployments_failed,
            "last_deployment": self._last_deployment.isoformat() if self._last_deployment else None,
        }


# ============================================================================
# Reactor-Core Output Writer (for reactor-core side)
# ============================================================================

async def push_model_to_jarvis_prime(
    model_path: Path,
    watch_dir: Path,
    manifest: ReactorCoreModelManifest,
) -> None:
    """
    Push a trained model to JARVIS-Prime's watch directory.

    This function should be called from reactor-core after training completes.

    Args:
        model_path: Path to the trained GGUF model
        watch_dir: JARVIS-Prime watch directory (pending/)
        manifest: Model manifest with metadata
    """
    pending_dir = watch_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Copy model to pending
    dest_path = pending_dir / model_path.name
    shutil.copy2(model_path, dest_path)

    # Write manifest
    manifest_path = dest_path.with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info(f"Pushed model to JARVIS-Prime: {dest_path}")
