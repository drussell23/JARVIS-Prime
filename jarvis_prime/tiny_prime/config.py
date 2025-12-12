"""Tiny Prime dynamic configuration (zero hardcoding).

This module intentionally keeps the config system flexible:
- YAML/JSON loading
- Environment variable overrides using nested `__` paths
- Minimal schema assumptions (stores unknown keys)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge `incoming` into `base` (returns a new dict)."""
    out: Dict[str, Any] = dict(base)
    for k, v in incoming.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _split_path(path: str) -> Tuple[str, ...]:
    return tuple(p for p in path.split(".") if p)


def _get_in(d: Dict[str, Any], path: Tuple[str, ...], default: Any = None) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _set_in(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    cur: Dict[str, Any] = d
    for p in path[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[path[-1]] = value


def _parse_env_value(raw: str) -> Any:
    """Best-effort parse for env var overrides."""
    s = raw.strip()

    # JSON literals/objects/lists/numbers
    if s and s[0] in "[{\"" or s in {"true", "false", "null"}:
        try:
            return json.loads(s)
        except Exception:
            pass

    # bools
    low = s.lower()
    if low in {"true", "yes", "y", "1"}:
        return True
    if low in {"false", "no", "n", "0"}:
        return False

    # ints/floats
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


@dataclass(frozen=True)
class TinyPrimeConfig:
    """Dynamic config wrapper."""

    raw: Dict[str, Any]
    source_path: Optional[Path] = None

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> "TinyPrimeConfig":
        p = Path(path)
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("PyYAML is required to load YAML configs. Install with: pip install pyyaml") from e

        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Config YAML must contain a mapping at the root")
        return cls(raw=data, source_path=p)

    @classmethod
    def from_json(cls, path: str | os.PathLike[str]) -> "TinyPrimeConfig":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Config JSON must contain a mapping at the root")
        return cls(raw=data, source_path=p)

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str = "TINY_PRIME__",
        base: Optional["TinyPrimeConfig"] = None,
    ) -> "TinyPrimeConfig":
        """Load overrides from env vars like `TINY_PRIME__TRAINING__BATCH_SIZE=16`.

        - `__` means nesting
        - values are parsed (json/bool/number/string)
        """
        overrides: Dict[str, Any] = {}

        for k, v in os.environ.items():
            if not k.startswith(prefix):
                continue
            suffix = k[len(prefix) :]
            if not suffix:
                continue
            parts = tuple(p.lower() for p in suffix.split("__") if p)
            if not parts:
                continue
            _set_in(overrides, parts, _parse_env_value(v))

        if base is None:
            return cls(raw=overrides, source_path=None)
        merged = _deep_merge(base.raw, overrides)
        return cls(raw=merged, source_path=base.source_path)

    def merge(self, overrides: Dict[str, Any]) -> "TinyPrimeConfig":
        return TinyPrimeConfig(raw=_deep_merge(self.raw, overrides), source_path=self.source_path)

    def get(self, path: str, default: Any = None) -> Any:
        return _get_in(self.raw, _split_path(path), default)

    def require(self, path: str) -> Any:
        val = self.get(path, None)
        if val is None:
            raise KeyError(f"Missing required config key: {path}")
        return val

    def base_dir(self) -> Path:
        return self.source_path.parent if self.source_path else Path.cwd()

    def project_root(self) -> Path:
        """Best-effort project root discovery (prefers repo/pyproject root)."""
        start = self.base_dir()
        for p in [start] + list(start.parents):
            if (p / "pyproject.toml").exists() or (p / ".git").exists():
                return p
        return start

    def resolve_path(self, maybe_path: str) -> str:
        p = Path(maybe_path)
        if p.is_absolute():
            return str(p)
        return str((self.project_root() / p).resolve())

    def resolve_path_for(self, key: str) -> str:
        val = self.require(key)
        if not isinstance(val, str):
            raise TypeError(f"Config key {key} must be a string path")
        return self.resolve_path(val)

    def intent_labels(self) -> list[str]:
        labels = self.require("tiny_prime.intents.labels")
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise TypeError("tiny_prime.intents.labels must be a list[str]")
        return labels

    def special_tokens(self) -> list[str]:
        toks = self.require("tokenizer.special_tokens")
        if not isinstance(toks, list) or not all(isinstance(x, str) for x in toks):
            raise TypeError("tokenizer.special_tokens must be a list[str]")
        return toks

    def format_tokens(self) -> Dict[str, str]:
        fmt = self.require("tiny_prime.format")
        if not isinstance(fmt, dict):
            raise TypeError("tiny_prime.format must be a mapping")
        return {
            "user_open": str(fmt.get("user_open", "<user>")),
            "user_close": str(fmt.get("user_close", "</user>")),
            "intent_open": str(fmt.get("intent_open", "<intent>")),
            "intent_close": str(fmt.get("intent_close", "</intent>")),
        }

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.raw)

    def save_yaml(self, path: str | os.PathLike[str]) -> None:
        p = Path(path)
        try:
            import yaml  # type: ignore
        except ImportError as e:
            raise ImportError("PyYAML is required to save YAML configs. Install with: pip install pyyaml") from e

        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False)

    def save_json(self, path: str | os.PathLike[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.raw, f, indent=2)
