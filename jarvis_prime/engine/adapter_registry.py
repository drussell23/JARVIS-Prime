"""Adapter registry -- J-Prime owns which LoRA its served models carry.

A fine-tune is published TO J-Prime (Reactor-Core uploads the GGUF adapter),
versioned here, and made live by rewriting the model's entry in the declared
models file the model store already reads first. Nothing else changes for a
caller: ``qwen3-coder-ov:30b`` keeps its name and gains new weights.

Contract:

* **Validated from the file's own header, never trusted from the caller:**
  GGUF magic, ``general.type == adapter``, ``adapter.type == lora`` and the
  SAME ``general.architecture`` as the base it is attached to. A LoRA trained
  against another base loads and answers wrongly -- the worst failure shape --
  so a mismatch is refused at publish time.
* **Integrity:** the caller's sha256 must match the bytes received.
* **Versions are kept.** Publishing activates the new version and records the
  previous one; :meth:`rollback` reactivates it. Files are never overwritten.
* **Atomic:** the adapter file and the models file are written to a temp path
  and ``os.replace``d, so a crash mid-publish leaves the previous state.

The base an adapter attaches to is the CURRENT model's base, resolved by the
store -- for the Ollama-store model that is its base blob -- so the registry
never needs a second copy of 17 GB of weights.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import gguf_meta
from .model_store import DeclaredModels, ModelStore, canonical_name

logger = logging.getLogger(__name__)


class AdapterRejected(ValueError):
    pass


def _engine_root() -> Path:
    from .engine_pool import default_engine_home
    return default_engine_home().parent


def adapters_dir() -> Path:
    raw = os.environ.get("JPRIME_ENGINE_ADAPTER_DIR", "").strip()
    return Path(raw) if raw else _engine_root() / "adapters"


def declared_models_path() -> Path:
    """The declared-models file: the env the store reads, else J-Prime's own."""
    raw = os.environ.get("JPRIME_ENGINE_MODELS", "").strip()
    return Path(raw) if raw else _engine_root() / "models.yaml"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", canonical_name(name))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class AdapterRegistry:
    def __init__(self, store: ModelStore) -> None:
        self.store = store

    # ---------------------------------------------------------------- state
    def _versions_path(self, name: str) -> Path:
        return adapters_dir() / _safe(name) / "versions.json"

    def versions(self, name: str) -> Dict[str, Any]:
        try:
            return json.loads(self._versions_path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"model": canonical_name(name), "active": None, "versions": []}

    def _save_versions(self, name: str, data: Dict[str, Any]) -> None:
        _atomic_write(self._versions_path(name), json.dumps(data, indent=2).encode())

    def _declared(self) -> Dict[str, Any]:
        import yaml
        try:
            return yaml.safe_load(declared_models_path().read_text(encoding="utf-8")) or {}
        except OSError:
            return {}

    def _write_declared(self, entry: Dict[str, Any]) -> None:
        import yaml
        data = self._declared()
        models = [m for m in (data.get("models") or [])
                  if canonical_name(str(m.get("name", ""))) != canonical_name(entry["name"])]
        models.append(entry)
        data["models"] = models
        _atomic_write(declared_models_path(), yaml.safe_dump(data, sort_keys=False).encode())
        # The store reads the declared file from the env path; make sure that
        # is THIS file for the life of the process.
        os.environ.setdefault("JPRIME_ENGINE_MODELS", str(declared_models_path()))
        self.store.declared = DeclaredModels(declared_models_path())

    # ---------------------------------------------------------------- publish
    def _validate(self, base_model: Path, adapter: Path) -> Dict[str, Any]:
        meta = gguf_meta.read_metadata(adapter)
        if not meta:
            raise AdapterRejected("not a readable GGUF")
        if meta.get("general.type") != "adapter" or meta.get("adapter.type") != "lora":
            raise AdapterRejected(f"not a LoRA adapter (general.type={meta.get('general.type')!r}, "
                                  f"adapter.type={meta.get('adapter.type')!r})")
        base_arch = gguf_meta.read_metadata(base_model).get("general.architecture")
        if meta.get("general.architecture") != base_arch:
            raise AdapterRejected(f"architecture mismatch: adapter {meta.get('general.architecture')!r} "
                                  f"vs base {base_arch!r}")
        return meta

    def publish(self, name: str, data: bytes, *, sha256: str, source: Optional[Dict[str, Any]] = None,
                activate: bool = True) -> Dict[str, Any]:
        """Store, validate and (by default) activate a new adapter for ``name``."""
        name = canonical_name(name)
        digest = hashlib.sha256(data).hexdigest()
        if not sha256 or digest != sha256.lower():
            raise AdapterRejected(f"sha256 mismatch: received {digest}, declared {sha256!r}")
        current = self.store.resolve(name)
        if current is None:
            raise AdapterRejected(f"unknown model {name!r}: publish an adapter onto a served model")
        state = self.versions(name)
        base = Path(state.get("base_model") or current.model_path)
        version = f"{time.strftime('%Y%m%d-%H%M%S')}-{digest[:12]}"
        path = adapters_dir() / _safe(name) / f"{version}.gguf"
        _atomic_write(path, data)
        try:
            meta = self._validate(base, path)
        except AdapterRejected:
            path.unlink(missing_ok=True)
            raise
        if not state["versions"]:
            # Remember what the model carried BEFORE J-Prime first managed it,
            # so the very first fine-tune is also reversible.
            state["base_model"] = str(base)
            state["defaults"] = dict(current.defaults)
            state["versions"].append({"version": "origin", "adapters": [str(p) for p in current.adapter_paths],
                                      "sha256": "", "published_at": None, "source": {"from": current.source}})
            state["active"] = "origin"
        state["versions"].append({"version": version, "adapters": [str(path)], "sha256": digest,
                                  "published_at": time.time(), "source": source or {},
                                  "base_model_name": meta.get("general.base_model.0.name", "")})
        self._save_versions(name, state)
        logger.warning("[Adapters] %s: published %s (%d bytes)", name, version, len(data))
        return self.activate(name, version) if activate else {"model": name, "version": version, "active": state["active"]}

    def activate(self, name: str, version: str) -> Dict[str, Any]:
        name = canonical_name(name)
        state = self.versions(name)
        entry = next((v for v in state["versions"] if v["version"] == version), None)
        if entry is None:
            raise AdapterRejected(f"{name}: no version {version!r}")
        if entry.get("status") == "rejected":
            raise AdapterRejected(f"{name}: version {version} was rejected ({entry.get('reason', '')[:120]})")
        previous = state.get("active")
        self._write_declared({"name": name, "model": state["base_model"], "adapters": entry["adapters"],
                              "projector": None, "defaults": state.get("defaults") or {}})
        if previous != version:
            state["previous"] = previous
        state["active"] = version
        self._save_versions(name, state)
        logger.warning("[Adapters] %s: ACTIVE %s (previous %s)", name, version, previous)
        return {"model": name, "active": version, "previous": state.get("previous")}

    def rollback(self, name: str) -> Dict[str, Any]:
        state = self.versions(canonical_name(name))
        prev = state.get("previous")
        if not prev:
            raise AdapterRejected(f"{name}: nothing to roll back to")
        return self.activate(name, prev)

    def reject(self, name: str, version: str, reason: str) -> Dict[str, Any]:
        """A version proved bad: DELETE its weights, record why, and if it was
        active put the newest remaining good version back (never another
        rejected one). The record stays -- an audit of what was refused --
        but the file cannot be activated again by anyone."""
        name = canonical_name(name)
        state = self.versions(name)
        entry = next((v for v in state["versions"] if v["version"] == version), None)
        if entry is None:
            raise AdapterRejected(f"{name}: no version {version!r}")
        if version == "origin":
            raise AdapterRejected("the origin version is the fallback of last resort; it is never deleted")
        for p in entry.get("adapters") or []:
            # Only files this registry owns; origin's blobs belong to their store.
            if Path(p).resolve().is_relative_to(adapters_dir().resolve()):
                Path(p).unlink(missing_ok=True)
        entry.update({"status": "rejected", "rejected_at": time.time(), "reason": reason[:500],
                      "adapters": []})
        restored = None
        if state.get("active") == version:
            good = [v for v in state["versions"] if v.get("status") != "rejected" and v["version"] != version]
            prev = state.get("previous")
            target = prev if any(v["version"] == prev for v in good) else (good[-1]["version"] if good else None)
            if target is None:
                raise AdapterRejected(f"{name}: no good version left to serve")
            self._save_versions(name, state)
            restored = self.activate(name, target)
            state = self.versions(name)
        state["previous"] = None if state.get("previous") == version else state.get("previous")
        self._save_versions(name, state)
        logger.error("[Adapters] %s: REJECTED %s (%s); serving %s", name, version, reason[:200], state["active"])
        return {"model": name, "rejected": version, "active": state["active"], "restored": restored}
