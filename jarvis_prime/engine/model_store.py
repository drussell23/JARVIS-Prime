"""Model store -- resolve a served model name to the files that make it.

A served model is more than one GGUF. ``qwen3-coder-ov:30b`` is the
qwen3-coder 30B base weights PLUS a Reactor-trained LoRA adapter PLUS the
sampling defaults it was published with. The store answers "what do I load
for this name" once, for every consumer (the engine pool, ``/api/tags``,
``/api/show``).

Two sources, consulted in order:

1. **Declared models** -- ``JPRIME_ENGINE_MODELS`` points at a YAML file of
   explicit entries (base path, adapters, projector, defaults). This is how a
   node with no Ollama store (a GCP L4) declares what it serves, and how
   Reactor-Core publishes a new adapter without any other daemon.
2. **The Ollama manifest store** -- read in place (``OLLAMA_MODELS`` or
   ``~/.ollama/models``). Blobs are content-addressed GGUF files, so J-Prime
   can serve every model already pulled on this machine with zero copies.
   Ollama does not have to be running; only its files are read.

Never raises for a missing or malformed source: an unreadable manifest is
skipped and logged, because one bad entry must not hide every other model.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_HOST = "registry.ollama.ai"
_DEFAULT_NAMESPACE = "library"
_DEFAULT_TAG = "latest"

_MT_MODEL = "application/vnd.ollama.image.model"
_MT_ADAPTER = "application/vnd.ollama.image.adapter"
_MT_PROJECTOR = "application/vnd.ollama.image.projector"
_MT_PARAMS = "application/vnd.ollama.image.params"


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to load and describe one served model."""

    name: str
    model_path: Path
    adapter_paths: Tuple[Path, ...] = ()
    projector_path: Optional[Path] = None
    defaults: Dict[str, Any] = field(default_factory=dict)
    digest: str = ""
    modified_at: str = ""
    source: str = "declared"

    @property
    def size_bytes(self) -> int:
        """Bytes on disk of every file the engine maps for this model."""
        total = 0
        for p in (self.model_path, *self.adapter_paths, self.projector_path):
            if p is None:
                continue
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    @property
    def has_vision(self) -> bool:
        return self.projector_path is not None


def canonical_name(name: str) -> str:
    """``qwen3-coder`` -> ``qwen3-coder:latest``; names are compared canonically."""
    name = (name or "").strip()
    if not name:
        return name
    last = name.rsplit("/", 1)[-1]
    return name if ":" in last else f"{name}:{_DEFAULT_TAG}"


def _manifest_path(root: Path, name: str) -> Path:
    """Where Ollama keeps the manifest for ``name`` (host/ns/model:tag)."""
    canon = canonical_name(name)
    repo, tag = canon.rsplit(":", 1)
    parts = repo.split("/")
    if len(parts) == 1:
        host, ns, model = _DEFAULT_REGISTRY_HOST, _DEFAULT_NAMESPACE, parts[0]
    elif len(parts) == 2:
        host, ns, model = _DEFAULT_REGISTRY_HOST, parts[0], parts[1]
    else:
        host, ns, model = parts[0], "/".join(parts[1:-1]), parts[-1]
    return root / "manifests" / host / ns / model / tag


def _name_from_manifest(root: Path, path: Path) -> str:
    rel = path.relative_to(root / "manifests").parts
    host, *middle, model, tag = rel
    ns = "/".join(middle)
    if host == _DEFAULT_REGISTRY_HOST and ns == _DEFAULT_NAMESPACE:
        return f"{model}:{tag}"
    if host == _DEFAULT_REGISTRY_HOST:
        return f"{ns}/{model}:{tag}"
    return f"{host}/{ns}/{model}:{tag}"


def _blob(root: Path, digest: str) -> Path:
    return root / "blobs" / digest.replace(":", "-")


class OllamaManifestStore:
    """Read-only view of an Ollama model directory."""

    def __init__(self, root: Optional[Path] = None) -> None:
        env = os.environ.get("OLLAMA_MODELS", "").strip()
        self.root = Path(root) if root else (Path(env) if env else Path.home() / ".ollama" / "models")

    def available(self) -> bool:
        return (self.root / "manifests").is_dir()

    def names(self) -> List[str]:
        base = self.root / "manifests"
        if not base.is_dir():
            return []
        out = []
        for p in base.rglob("*"):
            if p.is_file():
                try:
                    out.append(_name_from_manifest(self.root, p))
                except ValueError:
                    continue
        return sorted(out)

    def resolve(self, name: str) -> Optional[ModelSpec]:
        path = _manifest_path(self.root, name)
        if not path.is_file():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("[ModelStore] unreadable manifest %s: %s", path, exc)
            return None
        model_path: Optional[Path] = None
        adapters: List[Path] = []
        projector: Optional[Path] = None
        defaults: Dict[str, Any] = {}
        for layer in manifest.get("layers") or []:
            mt, digest = layer.get("mediaType", ""), layer.get("digest", "")
            if not digest:
                continue
            if mt == _MT_MODEL:
                model_path = _blob(self.root, digest)
            elif mt == _MT_ADAPTER:
                adapters.append(_blob(self.root, digest))
            elif mt == _MT_PROJECTOR:
                projector = _blob(self.root, digest)
            elif mt == _MT_PARAMS:
                try:
                    defaults = json.loads(_blob(self.root, digest).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    defaults = {}
        if model_path is None or not model_path.is_file():
            logger.warning("[ModelStore] %s has no readable model layer", name)
            return None
        cfg = (manifest.get("config") or {}).get("digest", "")
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        return ModelSpec(
            name=canonical_name(_name_from_manifest(self.root, path)),
            model_path=model_path,
            adapter_paths=tuple(adapters),
            projector_path=projector,
            defaults=defaults,
            digest=cfg.split(":", 1)[-1],
            modified_at=mtime,
            source="ollama-store",
        )


class DeclaredModels:
    """Models declared in a YAML file (``JPRIME_ENGINE_MODELS``).

    ::

        models:
          - name: qwen3-coder-ov:30b
            model: /opt/jarvis-prime/models/qwen3-coder-30b-q4_k_m.gguf
            adapters: [/opt/jarvis-prime/adapters/ov-2026-10-07.gguf]
            projector: null
            defaults: {temperature: 0.7, top_k: 20, top_p: 0.8}
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        env = os.environ.get("JPRIME_ENGINE_MODELS", "").strip()
        self.path = Path(path) if path else (Path(env) if env else None)

    def _entries(self) -> Iterable[Dict[str, Any]]:
        if self.path is None or not self.path.is_file():
            return []
        try:
            import yaml  # local: only needed when a file is declared
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001 -- a bad file must not take the store down
            logger.warning("[ModelStore] unreadable declared models %s: %s", self.path, exc)
            return []
        return [e for e in (data.get("models") or []) if isinstance(e, dict) and e.get("name")]

    def names(self) -> List[str]:
        return sorted(canonical_name(e["name"]) for e in self._entries())

    def resolve(self, name: str) -> Optional[ModelSpec]:
        want = canonical_name(name)
        for e in self._entries():
            if canonical_name(e["name"]) != want:
                continue
            model = Path(str(e.get("model", "")))
            if not model.is_file():
                logger.warning("[ModelStore] declared %s: model file missing %s", want, model)
                return None
            proj = e.get("projector")
            return ModelSpec(
                name=want,
                model_path=model,
                adapter_paths=tuple(Path(str(a)) for a in (e.get("adapters") or [])),
                projector_path=Path(str(proj)) if proj else None,
                defaults=dict(e.get("defaults") or {}),
                modified_at=datetime.fromtimestamp(model.stat().st_mtime, tz=timezone.utc).isoformat(),
                source="declared",
            )
        return None


class ModelStore:
    """Declared models first, then the Ollama store. One answer per name."""

    def __init__(self, declared: Optional[DeclaredModels] = None,
                 ollama: Optional[OllamaManifestStore] = None) -> None:
        self.declared = declared or DeclaredModels()
        self.ollama = ollama or OllamaManifestStore()

    def names(self) -> List[str]:
        seen = dict.fromkeys(self.declared.names())
        for n in self.ollama.names():
            seen.setdefault(canonical_name(n), None)
        return list(seen)

    def resolve(self, name: str) -> Optional[ModelSpec]:
        return self.declared.resolve(name) or self.ollama.resolve(name)

    def all_specs(self) -> List[ModelSpec]:
        specs = []
        for n in self.names():
            s = self.resolve(n)
            if s is not None:
                specs.append(s)
        return specs
