"""
Model Dispatcher — Dynamic GGUF dispatch from task_profile (Task 6)
====================================================================

Routes incoming inference requests to the appropriate LlamaCppExecutor
based on the ``model`` field in PrimeRequest / task_profile forwarded
from the JARVIS governance pipeline (Prong 1, Task 3).

Architecture
------------
- Reads ``jarvis_prime/config/model_catalogue.yaml`` for model → GGUF mapping.
- Maintains a bounded LRU cache of loaded LlamaCppExecutor instances.
- Falls back to the caller-supplied default executor when the requested
  model is unknown or fails to load.
- Thread-safe: all mutation is guarded by an asyncio.Lock.

Usage
-----
    dispatcher = ModelDispatcher(models_dir=Path("/opt/jarvis-prime/models"))
    await dispatcher.initialize()

    executor = await dispatcher.get_executor(
        model_name="qwen-2.5-coder-7b",
        default_executor=hot_swap_executor,
    )
    result = await executor.generate(prompt, max_tokens=2048)
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("jprime.model_dispatcher")

_CATALOGUE_PATH = (
    Path(__file__).parent.parent / "config" / "model_catalogue.yaml"
)


def _load_catalogue() -> Dict[str, Any]:
    """Load model_catalogue.yaml; returns empty dict on failure."""
    try:
        import yaml  # type: ignore[import]
        load_fn = yaml.safe_load
    except ImportError:
        logger.debug("[ModelDispatcher] PyYAML not available — catalogue disabled")
        return {}

    try:
        with _CATALOGUE_PATH.open("r") as fh:
            data = load_fn(fh)
        return data if isinstance(data, dict) else {}
    except OSError as exc:
        logger.warning("[ModelDispatcher] Cannot read catalogue: %s", exc)
        return {}


class ModelDispatcher:
    """Lazy-loading multi-model router.

    Maintains a bounded LRU cache (``max_loaded``) of
    LlamaCppExecutor instances.  When the requested model is not in
    the cache and ``max_loaded`` is reached, the least-recently-used
    entry is unloaded first.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        max_loaded: int = 2,
    ) -> None:
        self._models_dir = models_dir or self._resolve_models_dir()
        self._max_loaded = max_loaded
        self._catalogue: Dict[str, Any] = {}
        self._executors: Dict[str, Any] = {}   # model_name → LlamaCppExecutor
        self._lru: list[str] = []              # front = most-recently-used
        self._lock = asyncio.Lock()
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Load catalogue. Call once at server startup."""
        self._catalogue = _load_catalogue()
        models = self._catalogue.get("models", {})
        logger.info(
            "[ModelDispatcher] Initialized with %d catalogue entries (models_dir=%s)",
            len(models),
            self._models_dir,
        )
        self._initialized = True

    async def get_executor(
        self,
        model_name: Optional[str],
        default_executor: Any,
    ) -> Any:
        """Return the executor for *model_name*.

        Falls back to *default_executor* when:
        - model_name is None or empty
        - model_name is "jarvis-prime" (the default alias)
        - The GGUF is not found on disk
        - The model fails to load

        Parameters
        ----------
        model_name:
            Model alias (e.g. "qwen-2.5-coder-7b") or None.
        default_executor:
            The already-loaded executor to use as fallback.
        """
        # Normalise
        if not model_name or model_name in ("jarvis-prime",):
            return default_executor

        async with self._lock:
            # Cache hit
            if model_name in self._executors:
                self._touch_lru(model_name)
                logger.debug("[ModelDispatcher] Cache hit: %s", model_name)
                return self._executors[model_name]

            # Attempt to load
            executor = await self._load_executor(model_name)
            if executor is None:
                logger.warning(
                    "[ModelDispatcher] Falling back to default executor for %s",
                    model_name,
                )
                return default_executor

            # Evict LRU if at capacity
            if len(self._executors) >= self._max_loaded:
                await self._evict_lru()

            self._executors[model_name] = executor
            self._lru.insert(0, model_name)
            logger.info("[ModelDispatcher] Loaded: %s (cached: %d)", model_name, len(self._executors))
            return executor

    def list_loaded(self) -> list[str]:
        """Return model names currently loaded in cache."""
        return list(self._lru)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    async def _load_executor(self, model_name: str) -> Optional[Any]:
        """Resolve GGUF path and instantiate a new LlamaCppExecutor."""
        gguf_path = self._resolve_gguf(model_name)
        if gguf_path is None:
            logger.warning("[ModelDispatcher] No GGUF found for model: %s", model_name)
            return None

        try:
            from jarvis_prime.core.llama_cpp_executor import LlamaCppExecutor

            cfg = self._catalogue.get("models", {}).get(model_name, {})
            executor = LlamaCppExecutor(
                n_ctx=cfg.get("max_context", 8192),
                temperature=cfg.get("temperature", 0.2),
            )
            await executor.load(gguf_path)
            return executor
        except Exception as exc:
            logger.error("[ModelDispatcher] Failed to load %s: %s", model_name, exc)
            return None

    def _resolve_gguf(self, model_name: str) -> Optional[Path]:
        """Find the GGUF file for *model_name* on disk.

        Resolution order:
          1. Catalogue entry's gguf_file in models_dir
          2. Any *.gguf file whose stem matches model_name in models_dir
        """
        models = self._catalogue.get("models", {})
        entry = models.get(model_name, {})

        # Priority 1: catalogue entry
        if entry:
            gguf_file: str = entry.get("gguf_file", "")
            if gguf_file:
                candidate = self._models_dir / gguf_file
                if candidate.exists():
                    return candidate

        # Priority 2: fuzzy filename match (tolerates case / separator diffs)
        if self._models_dir.exists():
            for path in self._models_dir.glob("*.gguf"):
                if model_name.lower().replace("-", "") in path.stem.lower().replace("-", "").replace("_", ""):
                    logger.debug("[ModelDispatcher] Fuzzy match: %s → %s", model_name, path.name)
                    return path

        return None

    def _touch_lru(self, model_name: str) -> None:
        """Move model_name to the front of the LRU list."""
        if model_name in self._lru:
            self._lru.remove(model_name)
        self._lru.insert(0, model_name)

    async def _evict_lru(self) -> None:
        """Unload the least-recently-used executor."""
        if not self._lru:
            return
        victim = self._lru.pop()   # LRU = last element
        executor = self._executors.pop(victim, None)
        if executor is not None:
            try:
                if hasattr(executor, "unload"):
                    await executor.unload()
            except Exception as exc:
                logger.debug("[ModelDispatcher] Eviction cleanup error for %s: %s", victim, exc)
        logger.info("[ModelDispatcher] Evicted LRU model: %s", victim)

    @staticmethod
    def _resolve_models_dir() -> Path:
        """Determine models directory from env vars or catalogue defaults."""
        env_override = os.environ.get("JPRIME_MODELS_DIR")
        if env_override:
            return Path(env_override)

        golden_image_dir = Path("/opt/jarvis-prime/models")
        if golden_image_dir.exists():
            return golden_image_dir

        return Path.home() / ".jarvis" / "prime" / "models"
