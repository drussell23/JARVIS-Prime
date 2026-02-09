"""
v241.0: GCP Model Swap Coordinator — Pre-hook for multi-model serving.

Called BEFORE generation to ensure the correct model is loaded for the
incoming task type. Does NOT replace the generation pipeline — just swaps
the model in the existing LlamaCppExecutor, then returns control to
chat_completions() which uses _executor.format_messages / generate /
generate_stream unchanged.

Constraints:
- Only one model loaded at a time (~5.5 GB for 7B on 16 GB RAM)
- Swap = unload + load + validate (~20-30s for 7B GGUF from SSD)
- Sticky routing + per-model-size cooldown prevents thrashing
- manifest.json is primary inventory (built at image creation time)
- Filename scanning is fallback if manifest is missing

Architecture note: JARVIS Body's _infer_task_type() in query_handler.py
is the classifier that produces task_type hints. J-Prime's own TaskClassifier
in dynamic_model_registry.py is NOT consulted here because the task type must
be known BEFORE the request reaches J-Prime so the coordinator can swap models
ahead of inference. See query_handler.py for the sister classifier.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("jarvis_prime.gcp_swap")


# ---------------------------------------------------------------------------
# Env-var helpers (match existing pattern in unified_supervisor.py)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    """Read env var as float with fallback."""
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_int(name: str, default: int) -> int:
    """Read env var as int with fallback."""
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelEntry:
    """Inventory entry for a single model on disk."""
    path: Path
    size_gb: float
    routable: bool              # False for LLaVA (needs CLIP), BGE (embedding), TinyLlama (draft)
    config_overrides: Dict[str, Any]  # Per-model LlamaCppConfig kwargs


# ---------------------------------------------------------------------------
# Filename patterns for fallback inventory scanning (Issue #7)
# ---------------------------------------------------------------------------

_FILENAME_PATTERNS = {
    # Order matters: more specific patterns MUST come before generic ones.
    # _scan_filenames() iterates and breaks on first match.
    re.compile(r"mistral.*7b.*instruct", re.IGNORECASE): "mistral-7b",
    re.compile(r"qwen.*2\.?5.*coder.*7b", re.IGNORECASE): "qwen-2.5-coder-7b",
    re.compile(r"qwen.*2\.?5.*math.*7b", re.IGNORECASE): "qwen-2.5-math-7b",
    re.compile(r"qwen.*2\.?5.*7b.*instruct", re.IGNORECASE): "qwen-2.5-7b",  # Generic Qwen — after coder/math
    re.compile(r"deepseek.*r1.*(?:distill|qwen).*7b", re.IGNORECASE): "deepseek-r1-qwen-7b",
    re.compile(r"gemma.*2.*9b", re.IGNORECASE): "gemma-2-9b",
    re.compile(r"llava.*1\.?6.*mistral", re.IGNORECASE): "llava-v1.6-mistral-7b",
    re.compile(r"phi.*3\.?5.*mini", re.IGNORECASE): "phi-3.5-mini",
    re.compile(r"llama.*3\.?1.*8b", re.IGNORECASE): "llama-3.1-8b",
    re.compile(r"tinyllama.*1\.?1b", re.IGNORECASE): "tinyllama-1.1b",
    re.compile(r"bge.*large", re.IGNORECASE): "bge-large-en",
}

# Models that are NOT routable when discovered via filename scan
_NOT_ROUTABLE_IDS = frozenset({
    "llava-v1.6-mistral-7b",   # Needs CLIP encoder + multimodal path (v242)
    "bge-large-en",             # Embedding model — no generate() path
    "tinyllama-1.1b",           # Speculative decoding draft — not independently routable
})


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class GCPModelSwapCoordinator:
    """
    Pre-hook coordinator for multi-model serving on RAM-constrained GCP VM.

    Usage in run_server.py::

        # In chat_completions(), BEFORE any _executor usage:
        active_model_id = await _model_coordinator.ensure_model(task_type)
        # ...existing _executor code runs unchanged...
    """

    def __init__(
        self,
        executor,  # LlamaCppExecutor — not typed to avoid circular import
        models_dir: Path,
        manifest_path: Path,
    ):
        self._executor = executor
        self._models_dir = models_dir
        self._manifest_path = manifest_path

        # State
        self._current_model_id: Optional[str] = None
        self._previous_model_id: Optional[str] = None  # For rollback
        self._inventory: Dict[str, ModelEntry] = {}
        self._swap_lock = asyncio.Lock()
        self._last_swap_time: float = 0.0
        self._swap_count: int = 0

        # Bounded queue counter (R2-1: atomic in single event loop, no TOCTOU)
        self._waiting_count: int = 0

        # Env-var configurable
        self._default_model = os.getenv("GCP_DEFAULT_MODEL", "mistral-7b")
        self._queue_max = _env_int("GCP_SWAP_QUEUE_SIZE", 50)
        self._validation_timeout = _env_float("GCP_SWAP_VALIDATION_TIMEOUT", 30.0)

        # Per-model-size cooldown (Issue #10)
        self._cooldown_small = _env_float("GCP_COOLDOWN_SMALL", 30.0)    # < 3 GB
        self._cooldown_medium = _env_float("GCP_COOLDOWN_MEDIUM", 60.0)  # 3-5 GB
        self._cooldown_large = _env_float("GCP_COOLDOWN_LARGE", 90.0)    # > 5 GB

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load inventory from manifest.json (primary) or fallback to filename scanning."""
        if self._manifest_path.exists():
            self._load_manifest()
        else:
            logger.warning("[v241] No manifest.json — falling back to filename scanning")
            self._scan_filenames()

        # Record whatever model is currently loaded as current
        if self._executor.is_loaded():
            self._current_model_id = self._detect_current_model()
            logger.info(f"[v241] Detected currently loaded model: {self._current_model_id}")

        routable_count = sum(1 for e in self._inventory.values() if e.routable)
        logger.info(
            f"[v241] Inventory: {len(self._inventory)} models total, "
            f"{routable_count} routable"
        )

    def _load_manifest(self) -> None:
        """Parse manifest.json — primary inventory source (Issue #7)."""
        try:
            with open(self._manifest_path) as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"[v241] Failed to read manifest: {e}")
            self._scan_filenames()
            return

        for model_id, entry in manifest.get("models", {}).items():
            path = Path(entry.get("path", ""))
            if not path.exists():
                logger.warning(
                    f"[v241] Manifest entry {model_id} → {path} missing on disk, skipping"
                )
                continue

            routable = entry.get("routable", True)
            self._inventory[model_id] = ModelEntry(
                path=path,
                size_gb=entry.get("size_gb", 0.0),
                routable=routable,
                config_overrides=entry.get("config_overrides", {}),
            )
            if not routable:
                logger.info(
                    f"[v241] Model {model_id} pre-staged but NOT routable "
                    f"(requires special inference path)"
                )

    def _scan_filenames(self) -> None:
        """Fallback: scan GGUF files and fuzzy-match to model IDs."""
        for gguf_path in sorted(self._models_dir.glob("*.gguf")):
            name_lower = gguf_path.name.lower()
            for pattern, model_id in _FILENAME_PATTERNS.items():
                if pattern.search(name_lower):
                    self._inventory[model_id] = ModelEntry(
                        path=gguf_path,
                        size_gb=gguf_path.stat().st_size / (1024 ** 3),
                        routable=(model_id not in _NOT_ROUTABLE_IDS),
                        config_overrides={},
                    )
                    logger.info(f"[v241] Scanned: {gguf_path.name} → {model_id}")
                    break

    def _detect_current_model(self) -> Optional[str]:
        """Identify which inventory model matches the currently loaded executor model."""
        executor_path = getattr(self._executor, "_model_path", None)
        if executor_path is None:
            return self._default_model

        executor_path = Path(executor_path)
        for model_id, entry in self._inventory.items():
            if entry.path.resolve() == executor_path.resolve():
                return model_id

        # Fallback: try filename matching
        executor_name = executor_path.name.lower()
        for pattern, model_id in _FILENAME_PATTERNS.items():
            if pattern.search(executor_name):
                return model_id

        return self._default_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def inventory(self) -> Dict[str, ModelEntry]:
        """Read-only inventory access."""
        return self._inventory

    @property
    def current_model_id(self) -> Optional[str]:
        """Currently loaded model ID."""
        return self._current_model_id

    @property
    def swap_count(self) -> int:
        """Total number of model swaps performed."""
        return self._swap_count

    async def ensure_model(self, task_type: Optional[str]) -> Optional[str]:
        """
        Pre-hook: ensure the correct model is loaded for *task_type*.

        Returns the active model_id for telemetry (Issue #9).
        Raises ``HTTPException(503)`` if too many requests are queued
        during an ongoing swap (R2-1).
        """
        target = self._resolve_model(task_type)

        if target == self._current_model_id:
            return self._current_model_id  # Already loaded — no-op

        if not self._should_swap(target):
            logger.debug(
                f"[v241] Cooldown active, staying on {self._current_model_id} "
                f"(wanted {target})"
            )
            return self._current_model_id

        # R2-1: Bounded queue — reject excess requests during swap.
        # Safe in single asyncio event loop: no await between check and increment.
        if self._waiting_count >= self._queue_max:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="Model swap in progress, server at capacity",
                headers={"Retry-After": "30"},
            )

        self._waiting_count += 1
        try:
            await self._swap_model(target)
        finally:
            self._waiting_count -= 1

        return self._current_model_id

    # ------------------------------------------------------------------
    # Resolution & cooldown
    # ------------------------------------------------------------------

    def _resolve_model(self, task_type: Optional[str]) -> str:
        """Map task_type → model_id.  Only returns routable models."""
        if not task_type:
            return self._current_model_id or self._default_model

        # Lazy import to avoid circular dependency at module level
        from .dynamic_model_registry import GCP_TASK_MODEL_MAPPING

        model_id = GCP_TASK_MODEL_MAPPING.get(task_type, self._default_model)

        # Verify model is in inventory AND routable
        entry = self._inventory.get(model_id)
        if not entry or not entry.routable:
            return self._current_model_id or self._default_model

        return model_id

    def _should_swap(self, target: str) -> bool:
        """Per-model-size cooldown check (Issue #10)."""
        if self._last_swap_time == 0.0:
            return True  # First swap always allowed

        now = time.time()
        entry = self._inventory.get(target)
        size_gb = entry.size_gb if entry else 5.0

        if size_gb < 3.0:
            cooldown = self._cooldown_small
        elif size_gb <= 5.0:
            cooldown = self._cooldown_medium
        else:
            cooldown = self._cooldown_large

        return (now - self._last_swap_time) >= cooldown

    # ------------------------------------------------------------------
    # Swap engine
    # ------------------------------------------------------------------

    async def _swap_model(self, model_id: str) -> None:
        """
        Unload current → load target with per-model config → validate.
        On failure, roll back to previous model (Issue #8).
        """
        async with self._swap_lock:
            # Race-condition guard: another request may have swapped already
            if self._current_model_id == model_id:
                return

            entry = self._inventory.get(model_id)
            if not entry:
                logger.error(f"[v241] Model {model_id} not in inventory, skipping swap")
                return

            previous = self._current_model_id
            logger.info(
                f"[v241] Swapping: {previous} → {model_id} "
                f"({entry.size_gb:.1f} GB)"
            )
            swap_start = time.time()

            try:
                # 1. Unload current model (NOT close — that shuts down ThreadPoolExecutor)
                if self._executor.is_loaded():
                    await self._executor.unload()

                # 2. Load target with per-model config overrides (Issue #1)
                config_overrides = self._get_config_overrides(model_id)
                load_timeout = self._validation_timeout * 3  # ~90s for large models
                await asyncio.wait_for(
                    self._executor.load(entry.path, **config_overrides),
                    timeout=load_timeout,
                )

                # 3. Post-swap validation (Issue #8, R2-4)
                valid = await self._validate_after_swap()
                if not valid:
                    raise RuntimeError(
                        f"Model {model_id} failed post-swap validation"
                    )

                self._previous_model_id = previous
                self._current_model_id = model_id
                self._last_swap_time = time.time()
                self._swap_count += 1

                elapsed = time.time() - swap_start
                logger.info(
                    f"[v241] Swap complete: now serving {model_id} "
                    f"(swap #{self._swap_count}, {elapsed:.1f}s)"
                )

            except Exception as e:
                elapsed = time.time() - swap_start
                logger.error(
                    f"[v241] Swap to {model_id} FAILED after {elapsed:.1f}s: {e}. "
                    f"Rolling back to {previous}"
                )
                await self._rollback(previous)

    async def _rollback(self, previous_model_id: Optional[str]) -> None:
        """Attempt to reload the previous model after a failed swap."""
        if not previous_model_id or previous_model_id not in self._inventory:
            logger.critical(
                "[v241] Cannot rollback — no previous model available. "
                "Executor has no model loaded!"
            )
            self._current_model_id = None
            return

        try:
            prev_entry = self._inventory[previous_model_id]
            prev_overrides = self._get_config_overrides(previous_model_id)
            await self._executor.load(prev_entry.path, **prev_overrides)
            self._current_model_id = previous_model_id
            logger.info(f"[v241] Rollback to {previous_model_id} successful")
        except Exception as rollback_err:
            logger.critical(
                f"[v241] Rollback to {previous_model_id} ALSO failed: "
                f"{rollback_err}. Executor has no model loaded!"
            )
            self._current_model_id = None

    # ------------------------------------------------------------------
    # Config & validation helpers
    # ------------------------------------------------------------------

    def _get_config_overrides(self, model_id: str) -> Dict[str, Any]:
        """
        Get per-model LlamaCppConfig overrides (Issue #1).

        Priority: registry defaults → manifest overrides (manifest wins).
        """
        # Start with registry-level defaults
        try:
            from .dynamic_model_registry import GCP_MODEL_CONFIGS
            overrides = dict(GCP_MODEL_CONFIGS.get(model_id, {}))
        except ImportError:
            overrides = {}

        # Manifest entry overrides win
        entry = self._inventory.get(model_id)
        if entry and entry.config_overrides:
            overrides.update(entry.config_overrides)

        return overrides

    async def _validate_after_swap(self) -> bool:
        """
        Post-swap warmup: verify model responds (Issue #8).

        Uses the executor's built-in validate() if available, otherwise
        falls back to a 5-token generation test (R2-4).
        """
        try:
            # Prefer the executor's own validate() method
            if hasattr(self._executor, "validate"):
                return await asyncio.wait_for(
                    self._executor.validate(),
                    timeout=self._validation_timeout,
                )

            # Fallback: manual 5-token test
            result = await asyncio.wait_for(
                self._executor.generate(
                    prompt="Hello",
                    max_tokens=5,
                    temperature=0.0,
                ),
                timeout=self._validation_timeout,
            )
            return result is not None and len(str(result).strip()) > 0

        except Exception as e:
            logger.warning(f"[v241] Post-swap validation failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Status / telemetry
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return coordinator status for health endpoints."""
        return {
            "current_model": self._current_model_id,
            "previous_model": self._previous_model_id,
            "swap_count": self._swap_count,
            "last_swap_time": self._last_swap_time,
            "waiting_count": self._waiting_count,
            "inventory_count": len(self._inventory),
            "routable_count": sum(1 for e in self._inventory.values() if e.routable),
            "models": {
                mid: {
                    "size_gb": e.size_gb,
                    "routable": e.routable,
                    "loaded": (mid == self._current_model_id),
                }
                for mid, e in self._inventory.items()
            },
        }
